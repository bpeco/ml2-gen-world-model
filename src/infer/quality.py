"""Calidad de la inferencia por stream (Docs/uso.md, "Dispositivos e inferencia").

  python -m src.infer.quality online --model maskgit_s [--k 1] [--segments 50] [--frames 400]
  python -m src.infer.quality online --model maskgit_s --frames 1200 --segments 20                  # deriva
  python -m src.infer.quality imagine --model maskgit_s --steps 4 --name maskgit_s_s4

online: el modelo ve los frames reales de val uno por uno (tokens del dataset, para aislar al world model del
  encoder) y en cada frame t predice el t+k. Se mide desde que el contexto está lleno (t >= ctx_frames - 1):
  - acierto de tokens contra el frame real t+k, por frame y por tramo del segmento (deriva en streams largos);
  - LPIPS (alex) del frame predicho decodificado contra el real decodificado, cada --every frames, con IC 95% por
    bootstrap sobre segmentos;
  - la misma métrica para "repetir el último frame" (frame congelado), que a 20 FPS es una referencia fuerte.
  Escribe results/infer/online_<nombre>[_k<k>][_f<frames>].json
imagine: rollouts de los 400 clips de evaluación con el camino nuevo (Stream), mismos lotes y semillas que
  src.rollouts. Compara los tokens con results/rollouts/<name>.npy y guarda <name>_stream.npy para medir
  LPIPS/FVD con src.metrics_video (--name <name>_stream). Con temperatura 1, una diferencia numérica mínima en
  un token cambia el resto del rollout, así que la comparación justa es la de las métricas de video.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.common import device_info
from src.infer.codec import Codec
from src.infer.stream import make_stream

OUT = Path("results/infer")
VAL = "data/val_tokens.npy"
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
WINDOWS = [(19, 100), (100, 400), (400, 1200)]


def load(spec, dev, dtype, opts=None):
  from src.models import apply_opts, load_model
  return apply_opts(load_model(spec, "cpu"), opts).eval().to(dev, DTYPES[dtype])


def boot_ci(per_seg, reps=1000, seed=0):
  rng = np.random.default_rng(seed)
  n = len(per_seg)
  means = [per_seg[rng.integers(0, n, n)].mean() for _ in range(reps)]
  return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


@torch.no_grad()
def online(a, dev):
  import lpips
  model = load(a.model, dev, a.dtype, a.opt)
  kw = dict(temperature=a.temperature, steps=a.steps, seed=a.seed, graphs=dev.type == "cuda")
  s = make_stream(model, **kw)
  val = np.load(VAL, mmap_mode="r")
  tok = torch.as_tensor(np.array(val[:a.segments, :a.frames]).astype(np.int64), device=dev)
  S, T = tok.shape[:2]
  codec = Codec(dev, "fp16", graphs=dev.type == "cuda", encoder=False)
  lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
  to_t = lambda u: u.permute(0, 3, 1, 2).float().div(127.5).sub(1)

  def lpips_pair(x, y):
    return lp(to_t(codec.decode(x).clone()), to_t(codec.decode(y).clone())).flatten()

  t0 = s.ctx_frames - 1
  acc = np.full((T, S), np.nan)
  acc_frozen = np.full((T, S), np.nan)
  lp_m, lp_f = [], []
  t_start = time.time()
  for t in range(T - a.k):
    s.observe(tok[:, t])
    if t < t0:
      continue
    pred = s.predict(a.k)
    real = tok[:, t + a.k]
    acc[t] = (pred == real).float().mean(1).cpu().numpy()
    acc_frozen[t] = (tok[:, t] == real).float().mean(1).cpu().numpy()
    if (t - t0) % a.every == 0:
      lp_m.append(lpips_pair(pred, real).cpu().numpy())
      lp_f.append(lpips_pair(tok[:, t], real).cpu().numpy())
  sec = time.time() - t_start
  lp_m, lp_f = np.stack(lp_m), np.stack(lp_f)        # (n_eval, S)

  def summary(acc_tS, lp_nS):
    seg_acc = np.nanmean(acc_tS, 0)
    seg_lp = lp_nS.mean(0)
    win = {f"{lo}-{hi}": float(np.nanmean(acc_tS[lo:min(hi, T)])) for lo, hi in WINDOWS if lo < T - a.k}
    return {"token_acc": float(np.nanmean(acc_tS)), "token_acc_ci": boot_ci(seg_acc),
            "token_acc_by_window": win, "lpips": float(lp_nS.mean()), "lpips_ci": boot_ci(seg_lp)}

  name = a.name or model.name + (f"_s{a.steps}" if a.steps else "")
  res = {"name": name, "model": a.model, "steps": a.steps, "k": a.k, "temperature": a.temperature, "dtype": a.dtype,
         "segments": S, "frames": T, "every": a.every, "ctx_frames": s.ctx_frames,
         "model_metrics": summary(acc, lp_m), "frozen_metrics": summary(acc_frozen, lp_f),
         "acc_curve": acc[t0:T - a.k].mean(1).tolist(), "curve_start": t0, "seconds": sec,
         "device": device_info(dev), "timestamp": datetime.now().isoformat(timespec="seconds")}
  suffix = (f"_k{a.k}" if a.k > 1 else "") \
    + (f"_f{a.frames}" if a.frames != 400 else "") + (f"_t{a.temperature:g}" if a.temperature != 1.0 else "")
  OUT.mkdir(parents=True, exist_ok=True)
  path = OUT / f"online_{name}{suffix}.json"
  path.write_text(json.dumps(res, indent=1))
  m, f = res["model_metrics"], res["frozen_metrics"]
  print(f"{name} k={a.k}: acierto {m['token_acc']:.4f} (congelado {f['token_acc']:.4f})  "
        f"LPIPS {m['lpips']:.4f} (congelado {f['lpips']:.4f})  por tramo {m['token_acc_by_window']}  "
        f"{sec:.0f} s -> {path}", flush=True)
  return res


@torch.no_grad()
def imagine(a, dev):
  from src.rollouts import HORIZON, OUT as ROLL, clip_tokens, clips_hash, load_clips
  model = load(a.model, dev, a.dtype, a.opt)
  clips = load_clips()
  val = np.load(VAL, mmap_mode="r")
  ctx_all, _ = clip_tokens(val, clips)
  n = len(ctx_all)
  out = np.zeros((n, HORIZON, 128), np.int16)
  t_start = time.time()
  for i in range(0, n, a.bs):
    ctx = torch.as_tensor(ctx_all[i:i + a.bs].astype(np.int64), device=dev)
    s = make_stream(model, temperature=a.temperature, steps=a.steps, seed=a.seed * 100003 + i,
                    graphs=dev.type == "cuda")
    ctx = ctx[:, -s.ctx_frames:]  # lo mismo que ve generate
    for f in range(ctx.shape[1]):
      s.observe(ctx[:, f])
    out[i:i + a.bs] = torch.stack(list(s.imagine(HORIZON)), 1).cpu().numpy().astype(np.int16)
  sec = time.time() - t_start
  name = a.name or model.name
  ROLL.mkdir(parents=True, exist_ok=True)
  np.save(ROLL / f"{name}_stream.npy", out)
  meta = {"model": a.model, "name": f"{name}_stream", "via": "src.infer.stream", "steps": a.steps,
          "temperature": a.temperature, "dtype": a.dtype, "bs": a.bs, "seed": a.seed, "opt": a.opt, "n_clips": n,
          "seconds": sec, "device_info": device_info(dev), "clips_hash": clips_hash()}
  (ROLL / f"{name}_stream.json").write_text(json.dumps(meta, indent=1))
  ref_path = ROLL / f"{name}.npy"
  res = {"name": name, "seconds": sec}
  if ref_path.exists():
    ref = np.load(ref_path)
    same = out == ref
    res.update({"tokens_equal": float(same.mean()), "first_frame_equal": float(same[:, 0].mean()),
                "clips_identical": float(same.reshape(n, -1).all(1).mean())})
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / f"imagine_check_{name}.json").write_text(json.dumps(res, indent=1))
  print(res, f"-> {ROLL / name}_stream.npy (medir con: python -m src.metrics_video --name {name}_stream)", flush=True)
  return res


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("what", choices=["online", "imagine"])
  ap.add_argument("--model", required=True)
  ap.add_argument("--name")
  ap.add_argument("--steps", type=int)
  ap.add_argument("--k", type=int, default=1)
  ap.add_argument("--temperature", type=float, default=1.0)
  ap.add_argument("--dtype", default="fp16", choices=DTYPES)
  ap.add_argument("--segments", type=int, default=50)
  ap.add_argument("--frames", type=int, default=400)
  ap.add_argument("--every", type=int, default=4, help="online: LPIPS cada cuántos frames")
  ap.add_argument("--bs", type=int, default=50)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--opt", action="append")
  a = ap.parse_args(argv)
  dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return online(a, dev) if a.what == "online" else imagine(a, dev)


if __name__ == "__main__":
  main()
