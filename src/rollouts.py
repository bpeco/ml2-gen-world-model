"""Set fijo de clips de evaluación y rollouts de los modelos (plan 2.4).

Clips: 100 segmentos de val x 4 arranques = 400 clips; 20 frames de contexto (1 s) + 60 de rollout (3 s).
  python -m src.rollouts --make-clips        escribe results/clips_eval.json

Rollouts:
  python -m src.rollouts --model SPEC [--name NAME] [--bs 50] [--steps S] [--temperature 1.0]
                         [--dtype fp32|fp16|bf16] [--ctx-frames K] [--limit N] [--seed 0]
  SPEC es lo que entiende src.models.load_model, o "real" (los tokens reales como "rollout").
  Escribe results/rollouts/<name>.npy (N, 60, 128) int16 y <name>.json con los metadatos.

Nombre por defecto: model.name + sufijos solo para lo que no es el valor por defecto:
  _s<steps> (--steps), _<dtype> (fp16/bf16), _ctx<K> (--ctx-frames), _t<temp> (temperatura != 1).
  Ejemplos: maskgit_s_s4, gpt_s_fp16, gpt_s_ctx8, gpt_s_t0.8. "real" se llama "real".
Cada lote usa un Generator con semilla seed * 100003 + índice del primer clip del lote.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.common import CTX_FRAMES, FPS, device, device_info, log_device

CLIPS = Path("results/clips_eval.json")
SPLIT = Path("results/val_split.json")
VAL = "data/val_tokens.npy"
OUT = Path("results/rollouts")
HORIZON = 60
STARTS = [100, 370, 640, 910]
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def make_clips(path=CLIPS):
  keys = json.loads(SPLIT.read_text())["segments"]
  clips = [{"clip": i * len(STARTS) + j, "segment": i, "key": k, "start": s}
           for i, k in enumerate(keys) for j, s in enumerate(STARTS)]
  d = {"ctx": CTX_FRAMES, "horizon": HORIZON, "fps": FPS, "starts": STARTS, "clips": clips}
  path.write_text(json.dumps(d, indent=1))
  return d


def load_clips(path=CLIPS):
  """Crea el archivo si falta."""
  return json.loads(path.read_text()) if path.exists() else make_clips(path)


def clips_hash(path=CLIPS):
  load_clips(path)
  return hashlib.sha1(path.read_bytes()).hexdigest()[:10]


def clip_tokens(val, clips):
  """val (S, 1200, 128), clips (dict de load_clips o lista) -> ctx (N, 20, 128), real (N, 60, 128) int16."""
  c = clips["clips"] if isinstance(clips, dict) else clips
  ctx_n = clips["ctx"] if isinstance(clips, dict) else CTX_FRAMES
  hor = clips["horizon"] if isinstance(clips, dict) else HORIZON
  ctx = np.stack([val[x["segment"], x["start"]:x["start"] + ctx_n] for x in c])
  real = np.stack([val[x["segment"], x["start"] + ctx_n:x["start"] + ctx_n + hor] for x in c])
  return ctx, real


def default_name(base, steps=None, dtype="fp32", ctx_frames=CTX_FRAMES, temperature=1.0):
  name = base
  if steps is not None:
    name += f"_s{steps}"
  if dtype != "fp32":
    name += f"_{dtype}"
  if ctx_frames != CTX_FRAMES:
    name += f"_ctx{ctx_frames}"
  if temperature != 1.0:
    name += f"_t{temperature:g}"
  return name


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--make-clips", action="store_true")
  ap.add_argument("--model")
  ap.add_argument("--name")
  ap.add_argument("--bs", type=int, default=50)
  ap.add_argument("--steps", type=int)
  ap.add_argument("--temperature", type=float, default=1.0)
  ap.add_argument("--dtype", default="fp32", choices=list(DTYPES))
  ap.add_argument("--ctx-frames", type=int, default=CTX_FRAMES)
  ap.add_argument("--limit", type=int)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--graphs", action="store_true", help="CUDA graphs en el decode (mismos tokens, más rápido)")
  ap.add_argument("--opt", action="append", help="campo=valor de la Config del modelo (p. ej. rollout=latent); repetible")
  a = ap.parse_args(argv)

  if a.make_clips:
    d = make_clips()
    print(f"{CLIPS}: {len(d['clips'])} clips, hash {clips_hash()}")
    if not a.model:
      return CLIPS
  assert a.model, "falta --model SPEC (o --make-clips)"

  clips = load_clips()
  n = min(a.limit or len(clips["clips"]), len(clips["clips"]))
  clips = {**clips, "clips": clips["clips"][:n]}
  val = np.load(VAL, mmap_mode="r")
  ctx_all, real_all = clip_tokens(val, clips)
  ctx_all = ctx_all[:, -a.ctx_frames:]
  dev = device()
  info = log_device(dev)

  real = a.model == "real"
  model = None
  if not real:
    from src.models import apply_opts, load_model
    model = apply_opts(load_model(a.model, dev), a.opt).eval().to(DTYPES[a.dtype])
    if a.graphs:
      model.use_cuda_graphs(True)
  base = "real" if real else model.name
  name = a.name or (base if real else default_name(base, a.steps, a.dtype, a.ctx_frames, a.temperature))

  out = np.zeros((n, HORIZON, 128), np.int16)
  t0 = time.time()
  n_batches = 0
  for i in range(0, n, a.bs):
    if real:
      out[i:i + a.bs] = real_all[i:i + a.bs]
      continue
    ctx = torch.as_tensor(ctx_all[i:i + a.bs].astype(np.int64), device=dev)
    g = torch.Generator(device=dev).manual_seed(a.seed * 100003 + i)
    r = model.rollout(ctx, HORIZON, temperature=a.temperature, steps=a.steps, generator=g)
    out[i:i + a.bs] = r.cpu().numpy().astype(np.int16)
    n_batches += 1
    print(f"  clips {i + len(ctx)}/{n}  {time.time() - t0:.1f} s", flush=True)
  if dev.type == "cuda":
    torch.cuda.synchronize()
  sec = time.time() - t0

  OUT.mkdir(parents=True, exist_ok=True)
  np.save(OUT / f"{name}.npy", out)
  meta = {"model": a.model, "name": name, "opt": a.opt, "steps": a.steps, "temperature": a.temperature, "dtype": a.dtype,
          "bs": a.bs, "seed": a.seed, "graphs": a.graphs, "ctx_frames": a.ctx_frames, "n_clips": n, "seconds": sec,
          "ms_per_frame_batch": 1000 * sec / (max(n_batches, 1) * HORIZON) if not real else None,
          "device_info": info, "clips_hash": clips_hash()}
  (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1))
  print(f"{OUT / name}.npy {out.shape}  {sec:.1f} s")
  return OUT / f"{name}.npy"


if __name__ == "__main__":
  main()
