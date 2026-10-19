"""Caso de prueba visual: real vs modelos, lado a lado, como GIF y MP4.

Toma un segmento de val, da 19 frames reales de contexto (~1 s) y cada modelo
imagina los N frames siguientes. Las columnas muestran el video real y lo generado.

Uso: python -m src.demo [--segment 0] [--start 600] [--seconds 3] [--models gpt_mini gpt2m]
"""
import argparse
import time

import cv2
import numpy as np
import torch
from PIL import Image

from src.common import CTX_FRAMES, device
from src.evaluate import load_model
from src.generate import FPS, rollout_gpt2m, rollout_mini
from src.viz import decode, load_decoder

MODELS = {"gpt_mini": "checkpoints/gpt_mini.pt", "gpt2m": "gpt2m"}
TITLES = {"real": "real", "gpt_mini": "nuestro GPT (11M)", "gpt2m": "GPT de comma (300M)"}


def label(img, text, color=(255, 255, 255)):
  img = cv2.resize(img, (512, 256), interpolation=cv2.INTER_NEAREST)
  cv2.rectangle(img, (0, 0), (512, 26), (0, 0, 0), -1)
  cv2.putText(img, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
  return img


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--segment", type=int, default=0)
  ap.add_argument("--start", type=int, default=600)
  ap.add_argument("--seconds", type=float, default=3)
  ap.add_argument("--models", nargs="+", default=["gpt_mini", "gpt2m"])
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  dev, n_new = device(), int(args.seconds * FPS)
  val = np.load("data/val_tokens.npy").astype(np.int64)
  seg, s0 = val[args.segment], args.start
  ctx = seg[s0 - (CTX_FRAMES - 1):s0]
  cols = {"real": seg[s0 - (CTX_FRAMES - 1):s0 + n_new]}
  for name in args.models:
    torch.manual_seed(args.seed)
    model = load_model(MODELS[name], dev)
    t0 = time.time()
    roll = rollout_gpt2m if name == "gpt2m" else rollout_mini
    gen = roll(model, torch.from_numpy(ctx[None]).to(dev), n_new)[0].cpu().numpy()
    print(f"{name}: {n_new} frames en {time.time() - t0:.0f}s", flush=True)
    cols[name] = np.concatenate([ctx, gen])
    del model

  dec = load_decoder()
  imgs = {k: decode(dec, v) for k, v in cols.items()}
  frames = []
  for i in range(len(cols["real"])):
    t = (i - (CTX_FRAMES - 1) + 1) / FPS
    row = []
    for k, v in imgs.items():
      if t <= 0:
        row.append(label(v[i], f"{TITLES[k]} | contexto real", (180, 180, 180)))
      else:
        row.append(label(v[i], f"{TITLES[k]} | +{t:.2f} s", (255, 255, 255) if k == "real" else (80, 170, 255)))
    frames.append(np.concatenate(row, 1))

  out = f"results/demo_seg{args.segment}"
  Image.fromarray(frames[0]).save(
    out + ".gif", save_all=True, append_images=[Image.fromarray(f) for f in frames[1:]], duration=1000 // FPS, loop=0)
  h, w = frames[0].shape[:2]
  vw = cv2.VideoWriter(out + ".mp4", cv2.VideoWriter_fourcc(*"avc1"), FPS, (w, h))
  for f in frames:
    vw.write(f[..., ::-1])
  vw.release()
  print("guardado:", out + ".gif", out + ".mp4")
