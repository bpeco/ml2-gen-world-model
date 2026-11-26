"""Destilación por secuencias (opción B): el maestro GPT2M genera continuaciones desde contextos reales de entrenamiento.

Cada ejemplo: 19 frames reales de contexto (segmentos de train de data/train_tokens.npy, sin el holdout de Windows;
data-0001 nunca se toca) + G frames generados por GPT2M a temperatura T. Se guarda (N, 19 + G, 128) int16 en
data/maestro_seq.npy, con guardado parcial para reanudar.

En train.py (--seq-maestro) una fracción del batch se reemplaza por ventanas de 20 frames tomadas de estas secuencias
en un offset al azar (entre 1 y 20 frames generados por ventana). La pérdida no cambia: con --kd, alpha·CE sobre los
tokens del maestro + (1-alpha)·KL contra los logits del maestro sobre esa misma secuencia (que son exactamente las
distribuciones con las que la generó, porque ve el mismo contexto de 19 frames).

  python -m src.maestro_seq --n 1250 --gen 20 --bs 25                  genera (reanuda si hay parcial)
  python -m src.maestro_seq --n 50 --gen 20 --bs 25 --out /tmp/x.npy   prueba de velocidad
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

CTX = 19


def ventanas(n, n_train, T, seed):
  rng = np.random.default_rng(seed)
  return rng.integers(0, n_train, n), rng.integers(0, T - CTX + 1, n)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--n", type=int, required=True, help="cantidad de secuencias")
  ap.add_argument("--gen", type=int, default=20, help="frames generados por secuencia")
  ap.add_argument("--bs", type=int, default=25)
  ap.add_argument("--temperature", type=float, default=1.0)
  ap.add_argument("--seed", type=int, default=777)
  ap.add_argument("--data", default="data/train_tokens.npy")
  ap.add_argument("--holdout", type=int, default=50)
  ap.add_argument("--out", default="data/maestro_seq.npy")
  a = ap.parse_args()
  from src.common import device
  from src.models import load_model
  dev = device()
  tok = np.load(a.data, mmap_mode="r")
  segs, starts = ventanas(a.n, len(tok) - a.holdout, tok.shape[1], a.seed)
  out_p, meta_p = Path(a.out), Path(a.out).with_suffix(".json")
  firma = {k: getattr(a, k) for k in ("n", "gen", "temperature", "seed", "data", "holdout")}
  out = np.zeros((a.n, CTX + a.gen, 128), np.int16)
  hecho = 0
  if meta_p.exists() and out_p.exists():
    m = json.loads(meta_p.read_text())
    if m["firma"] == firma:
      hecho = m["hecho"]
      out[:hecho] = np.load(out_p)[:hecho]
      print(f"reanudando desde {hecho}/{a.n}", flush=True)
  model = load_model("gpt2m", dev).eval().to(torch.float16)
  model.use_cuda_graphs(True)
  t0 = time.time()
  for i in range(hecho, a.n, a.bs):
    j = min(i + a.bs, a.n)
    ctx = np.stack([tok[s, t:t + CTX] for s, t in zip(segs[i:j], starts[i:j])]).astype(np.int64)
    g = torch.Generator(device=dev).manual_seed(a.seed * 100003 + i)
    r = model.rollout(torch.as_tensor(ctx, device=dev), a.gen, temperature=a.temperature, generator=g)
    out[i:j, :CTX], out[i:j, CTX:] = ctx, r.cpu().numpy()
    if (i // a.bs) % 4 == 3 or j == a.n:
      np.save(out_p, out[:j])
      meta_p.write_text(json.dumps({"firma": firma, "hecho": j, "segs": segs.tolist(), "starts": starts.tolist()}))
    dt = time.time() - t0
    print(f"{j}/{a.n}  {(j - hecho) * a.gen / dt:.1f} frames gen/s", flush=True)
  np.save(out_p, out)


if __name__ == "__main__":
  main()
