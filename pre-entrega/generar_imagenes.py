"""Genera las imágenes y los datos de las animaciones de pre-entrega/index.html.

Usa el segmento 0 de val, a partir del frame 600:
- anim_real.jpg: los 19 frames reales de contexto (581-599).
- anim_gen.jpg: 20 frames generados por gpt2m desde ese contexto, con KV cache.
- race_real.jpg: 50 frames reales (600-649) para la carrera del último slide.
- TOK_IN, TOK_OUT y TOP5: se imprimen para pegar en el <script> del HTML.
  TOK_IN es el último frame de contexto, TOK_OUT el primer frame generado y TOP5 las
  5 probabilidades más altas de gpt2m en cada celda de TOK_OUT, más la del token elegido.

Necesita data/ (python -m src.data) y third_party/commavq (ver README).
Uso, desde la raíz del repo: python pre-entrega/generar_imagenes.py [--out pre-entrega/img]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from src.common import BLOCK, BOS, CTX_FRAMES, TPF, device, with_bos  # noqa: E402
from src.evaluate import load_model  # noqa: E402
from src.viz import decode, load_decoder  # noqa: E402

SEG, START = 0, 600
N_GEN, N_RACE = 20, 50


@torch.no_grad()
def rollout(model, ctx, n_new):
  """ctx (19, 128) -> (n_new, 128). Genera de a un frame y corre la ventana."""
  from utils.gpt import KVCache
  c, dev = model.config, ctx.device
  L = BLOCK + 8 - BLOCK % 8
  for b in model.transformer.h:
    b.attn.kv_cache = KVCache(1, L, c.n_head, c.head_dim, dtype=torch.float32).to(dev)
  model.causal_mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev)).view(1, 1, L, L)
  model.max_batch_size, model.max_seq_length = 1, L
  seq, frames = ctx, []
  for _ in range(n_new):
    new = model.generate(with_bos(seq).view(-1), TPF)[1:].long()
    new[new == BOS] = seq[-1][new == BOS]  # si sale un BOS en una celda, se repite la del frame anterior
    frames.append(new)
    seq = torch.cat([seq[1:], new[None]], 0)
  return torch.stack(frames)


@torch.no_grad()
def top5(model, ctx, gen0):
  """Probabilidades de gpt2m para cada celda de gen0, dado el contexto (una sola pasada)."""
  seq = torch.cat([with_bos(ctx), torch.tensor([BOS] + gen0.tolist(), device=ctx.device)])  # 20*129 tokens
  logits = model(seq[None])[0].float()
  start = (CTX_FRAMES - 1) * TPF  # posición del BOS del frame generado
  probs = torch.softmax(logits[start:start + 128], -1).cpu()
  top = probs.topk(5, -1)
  g = gen0.cpu()
  return [[[int(i), round(float(p), 3)] for i, p in zip(top.indices[j], top.values[j])] +
          [[int(g[j]), round(float(probs[j, g[j]]), 3)]] for j in range(128)]


def sprite(imgs, path, quality):
  """Frames apilados en vertical: el HTML los recorre con translateY."""
  Image.fromarray(np.concatenate(imgs, 0)).save(path, quality=quality, optimize=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="pre-entrega/img")
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)

  val = np.load("data/val_tokens.npy").astype(np.int64)
  ctx = val[SEG, START - (CTX_FRAMES - 1):START]
  dev = device()
  model = load_model("gpt2m", dev)

  torch.manual_seed(args.seed)
  t0 = time.time()
  ctx_t = torch.from_numpy(ctx).to(dev)
  gen = rollout(model, ctx_t, N_GEN).cpu().numpy()
  print(f"rollout: {N_GEN} frames en {time.time() - t0:.0f} s", file=sys.stderr)
  probs = top5(model, ctx_t, torch.from_numpy(gen[0]).to(dev))

  dec = load_decoder()
  sprite(decode(dec, ctx), out / "anim_real.jpg", 88)
  sprite(decode(dec, gen), out / "anim_gen.jpg", 88)
  sprite(decode(dec, val[SEG, START:START + N_RACE]), out / "race_real.jpg", 85)
  print(f"imágenes en {out}/", file=sys.stderr)

  js = lambda x: json.dumps(x, separators=(",", ":"))
  print(f"  const TOK_IN = {js(ctx[-1].tolist())};")
  print(f"  const TOK_OUT = {js(gen[0].tolist())};")
  print(f"  const TOP5 = {js(probs)};")


if __name__ == "__main__":
  main()
