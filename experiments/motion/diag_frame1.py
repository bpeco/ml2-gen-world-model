"""Diagnóstico 0: ¿a qué frame del contexto se parece el primer frame generado?

  python experiments/motion/diag_frame1.py     escribe results/motion/diag_frame1.json

Para cada modelo (rollouts de la fase 1 en clips_motion) compara el frame 1 generado, decodificado, con:
  - los frames de contexto t-1 ... t-K (t-1 = el último), decodificados
  - el promedio en píxeles de t-1 y t-2, y el promedio de los 20 frames de contexto
con L1 en píxeles, LPIPS (alex) y coincidencia de tokens. La referencia es el frame 1 real: un modelo que "retrocede"
tiene, respecto del real, la distancia a t-2 más chica que la distancia a t-1 (o más chica de lo que la tiene el real).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from src.viz import decode, load_decoder  # noqa: E402

MODELOS = ["real", "gpt2m", "gpt_s_kd", "gpt_s_d050", "mg_p1", "mg_p4", "mg_p8", "genie_s_p4"]
K = 6
OUT = Path("results/motion/diag_frame1.json")


def main():
  import lpips
  dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  d = json.loads(Path("results/clips_motion.json").read_text())
  C = d["clips"]
  T = np.load(d["tokens"], mmap_mode="r")
  ctx = np.stack([T[c["segment"], c["start"]:c["start"] + 20] for c in C]).astype(np.int64)
  cur = np.array([c["tipo"] == "curva" for c in C])
  dec = load_decoder(dev)
  lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
  to_t = lambda x: torch.as_tensor(x, device=dev).permute(0, 3, 1, 2).float() / 127.5 - 1  # noqa: E731

  # contexto decodificado: los últimos K frames y el promedio de los 20 (por clip, en píxeles)
  ctx_px = np.stack([decode(dec, ctx[i, -K:], bs=32) for i in range(len(C))]).astype(np.float32)    # (N, K, H, W, 3)
  ctx_mean = np.stack([decode(dec, ctx[i], bs=32).astype(np.float32).mean(0) for i in range(len(C))])
  refs = {f"t-{k}": ctx_px[:, K - k] for k in range(1, K + 1)}
  refs["media(t-1,t-2)"] = (ctx_px[:, -1] + ctx_px[:, -2]) / 2
  refs["media del contexto"] = ctx_mean

  res = {}
  for m in MODELOS:
    f1 = (T[[c["segment"] for c in C], [c["start"] + 20 for c in C]] if m == "real"
          else np.load(f"results/rollouts/{m}_motion.npy")[:, 0]).astype(np.int64)
    px = decode(dec, f1, bs=32).astype(np.float32)
    r = {}
    for nombre, ref in refs.items():
      l1 = np.abs(px - ref).mean((1, 2, 3))
      with torch.no_grad():
        lps = np.concatenate([lp(to_t(px[i:i + 50]), to_t(ref[i:i + 50])).flatten().cpu().numpy()
                              for i in range(0, len(px), 50)])
      tok = None
      if nombre.startswith("t-"):
        tok = (f1 == ctx[:, 20 - int(nombre[2:])]).mean(1)
      r[nombre] = {g: {"l1": float(l1[s].mean()), "lpips": float(lps[s].mean()),
                       "tokens_iguales": None if tok is None else float(tok[s].mean())}
                   for g, s in (("todos", slice(None)), ("curva", cur), ("recta", ~cur))}
    res[m] = r
    c = r
    print(f"{m:12s} curvas LPIPS " + " ".join(f"{k}:{c[k]['curva']['lpips']:.3f}" for k in refs), flush=True)
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps({"clips": "motion", "K": K, "modelos": res}, indent=1))
  print(OUT)


if __name__ == "__main__":
  main()
