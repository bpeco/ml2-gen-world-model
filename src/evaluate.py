"""Benchmark fijo: protocolo de notebooks/eval.ipynb de comma (commit a1adab98).

Por segmento de val: BOS antes de cada frame, primeros 1180 frames en 59 ventanas
no solapadas de 20 frames (2580 tokens). Target = la ventana corrida 1 token.
ce_all incluye los targets BOS (comparable con el 2,02 que reporta comma para gpt2m).
ce_tok, bits_tok y acc_tok excluyen los targets BOS.

Uso: python -m src.evaluate --model gpt2m [--segments N]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.common import BLOCK, BOS, CTX_FRAMES, device, with_bos

N_FRAMES = 1200
N_WIN = (N_FRAMES - CTX_FRAMES) // CTX_FRAMES  # 59


GPT2M = "data/raw/pytorch_model.bin"  # commaai/commavq-gpt2m, bajado por src.data


def load_gpt2m(dev):
  sys.path.insert(0, "third_party/commavq")
  from utils.gpt import GPT, GPTConfig
  m = GPT(GPTConfig())
  # reusa el loader de comma (transpone pesos de HF) pero leyendo el archivo local
  hub_load = torch.hub.load_state_dict_from_url
  torch.hub.load_state_dict_from_url = lambda url, **kw: torch.load(GPT2M, map_location="cpu", weights_only=True)
  try:
    m.load_state_dict_from_url()
  finally:
    torch.hub.load_state_dict_from_url = hub_load
  return m


def load_model(name, dev):
  if name != "gpt2m":
    raise ValueError(f"modelo no soportado: {name}")
  return load_gpt2m(dev).to(dev).eval()


@torch.no_grad()
def evaluate(model, val, dev, bs=10, segments=None, log=True):
  val = val[:segments] if segments else val
  seg_ce, seg_ce_tok, n_tok, n_correct = [], [], 0, 0
  t0 = time.time()
  for i, seg in enumerate(val):
    toks = with_bos(torch.from_numpy(np.asarray(seg, dtype=np.int64)).to(dev))  # (1200*129,)
    idx = torch.arange(N_WIN * BLOCK, device=dev).view(N_WIN, BLOCK)
    ce_sum, ce_tok_sum, tok_count = 0.0, 0.0, 0
    for j in range(0, N_WIN, bs):
      ii = idx[j:j + bs]
      x, y = toks[ii], toks[ii + 1]
      logits = model(x).float()
      ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
      tok = y.reshape(-1) != BOS
      ce_sum += ce.sum().item()
      ce_tok_sum += ce[tok].sum().item()
      tok_count += tok.sum().item()
      n_correct += (logits[..., :BOS].argmax(-1).reshape(-1)[tok] == y.reshape(-1)[tok]).sum().item()
    seg_ce.append(ce_sum / (N_WIN * BLOCK))
    seg_ce_tok.append(ce_tok_sum / tok_count)
    n_tok += tok_count
    if log and ((i + 1) % 10 == 0 or i + 1 == len(val)):
      print(f"{i + 1}/{len(val)} ce_all={np.mean(seg_ce):.4f} ({time.time() - t0:.0f}s)", flush=True)
  ce_tok = float(np.mean(seg_ce_tok))
  return {
    "ce_all": float(np.mean(seg_ce)),
    "ce_tok": ce_tok,
    "bits_tok": ce_tok / math.log(2),
    "ppl_tok": math.exp(ce_tok),
    "acc_tok": n_correct / n_tok,
    "segments": len(val),
    "seconds": time.time() - t0,
  }


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--name", default=None)
  ap.add_argument("--segments", type=int, default=None)
  ap.add_argument("--bs", type=int, default=10)
  args = ap.parse_args()
  dev = device()
  name = args.name or Path(args.model).stem
  model = load_model(args.model, dev)
  val = np.load("data/val_tokens.npy")
  res = {"model": name, **evaluate(model, val, dev, args.bs, args.segments)}
  out = Path(f"results/eval_{name}" + (f"_seg{args.segments}" if args.segments else "") + ".json")
  out.parent.mkdir(exist_ok=True)
  out.write_text(json.dumps(res, indent=1))
  print(json.dumps(res, indent=1))
