"""Entrena el GPT chico sobre ventanas de 20 frames, con presupuesto de tiempo fijo.

Cada muestra: segmento al azar, frame inicial al azar, 21 frames con BOS -> 2581 tokens.
x = primeros 2580, y = corridos 1 (mismo formato que el benchmark).
LR: warmup lineal + coseno sobre la fracción de tiempo transcurrido.
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.common import BLOCK, CTX_FRAMES, device, with_bos
from src.evaluate import evaluate
from src.model import GPT, Config


def batch(train, bs, rng, dev):
  segs = rng.integers(0, len(train), bs)
  starts = rng.integers(0, train.shape[1] - CTX_FRAMES, bs)
  w = np.stack([train[s, f:f + CTX_FRAMES + 1] for s, f in zip(segs, starts)]).astype(np.int64)
  toks = with_bos(torch.from_numpy(w).to(dev))[:, :BLOCK + 1]
  return toks[:, :-1], toks[:, 1:]


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="checkpoints/gpt_mini.pt")
  ap.add_argument("--minutes", type=float, default=25)
  ap.add_argument("--max-steps", type=int, default=None)
  ap.add_argument("--bs", type=int, default=8)
  ap.add_argument("--lr", type=float, default=1e-3)
  ap.add_argument("--n-layer", type=int, default=6)
  ap.add_argument("--n-head", type=int, default=6)
  ap.add_argument("--dim", type=int, default=384)
  ap.add_argument("--factored-pos", action="store_true")
  ap.add_argument("--eval-every", type=int, default=200)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  torch.manual_seed(args.seed)
  rng = np.random.default_rng(args.seed)
  dev = device()
  train = np.load("data/train_tokens.npy")
  val = np.load("data/val_tokens.npy")
  model = GPT(Config(n_layer=args.n_layer, n_head=args.n_head, dim=args.dim, factored_pos=args.factored_pos)).to(dev)
  print(f"params: {model.n_params() / 1e6:.1f}M", flush=True)

  decay = [p for p in model.parameters() if p.dim() >= 2]
  no_decay = [p for p in model.parameters() if p.dim() < 2]
  opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                          lr=args.lr, betas=(0.9, 0.95))
  budget, warmup = args.minutes * 60, 100
  log, t0, step = [], time.time(), 0
  while True:
    frac = (time.time() - t0) / budget
    if frac >= 1 or (args.max_steps and step >= args.max_steps):
      break
    lr = args.lr * min(1, (step + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
    for g in opt.param_groups:
      g["lr"] = lr
    x, y = batch(train, args.bs, rng, dev)
    loss = F.cross_entropy(model(x).view(-1, Config.vocab_size), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    step += 1
    if step % 20 == 0:
      el = time.time() - t0
      print(f"step {step} loss {loss.item():.4f} lr {lr:.2e} {el:.0f}s ({el / step:.2f}s/step)", flush=True)
      log.append({"step": step, "seconds": el, "train_loss": loss.item()})
    if step % args.eval_every == 0:
      model.eval()
      r = evaluate(model, val, dev, segments=5, log=False)  # monitoreo rápido, no es el benchmark
      model.train()
      print(f"  val(5 seg) ce_all {r['ce_all']:.4f}", flush=True)
      log[-1]["val_ce_all_5seg"] = r["ce_all"]

  tokens_seen = step * args.bs * BLOCK
  model.save(args.out, steps=step, tokens_seen=tokens_seen, minutes=(time.time() - t0) / 60)
  json.dump({"args": vars(args), "params": model.n_params(), "steps": step, "tokens_seen": tokens_seen, "log": log},
            open(args.out.replace(".pt", "_log.json"), "w"), indent=1)
  print(f"listo: {step} steps, {tokens_seen / 1e6:.0f}M tokens", flush=True)
