"""Benchmark fijo: protocolo de notebooks/eval.ipynb de comma (commit a1adab98).

Por segmento de val: BOS antes de cada frame, primeros 1180 frames en 59 ventanas
no solapadas de 20 frames (2580 tokens). Target = la ventana corrida 1 token.
ce_all incluye los targets BOS (comparable con el 2,02 que reporta comma para gpt2m).
ce_tok, bits_tok y acc_tok excluyen los targets BOS.
ce_last / acc_last: solo los 128 códigos del último frame de cada ventana (dados los 19 anteriores),
renormalizados sobre los 1024 códigos. Es lo único que se puede medir en modelos sin verosimilitud
(MaskGIT): ahí ce_all, ce_tok, etc. quedan en null.

Uso: python -m src.evaluate --model SPEC [--segments N] [--dtype fp32|fp16|bf16]
  SPEC: gpt2m, unigram, copy, random:..., o ruta a un checkpoint (ver src.models.load_model)
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.common import BLOCK, BOS, CTX_FRAMES, TPF, device, device_info, log_device, strip_bos, with_bos
from src.models import load_model as _load_model

N_FRAMES = 1200
N_WIN = (N_FRAMES - CTX_FRAMES) // CTX_FRAMES  # 59
LAST = (CTX_FRAMES - 1) * TPF                   # posición del BOS del último frame de la ventana
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def load_model(name, dev):
  """Compatibilidad con pre-entrega/generar_imagenes.py, que devuelve el GPT crudo de comma."""
  m = _load_model(name, dev)
  return m.gpt if name == "gpt2m" else m


@torch.no_grad()
def evaluate(model, val, dev, bs=10, segments=None, log=True):
  val = val[:segments] if segments else val
  lik = model.has_likelihood
  rows, t0 = [], time.time()  # por segmento: [ce_all, ce_tok, n_tok, n_correct, ce_last, n_correct_last]
  idx = torch.arange(N_WIN * BLOCK, device=dev).view(N_WIN, BLOCK)
  for i, seg in enumerate(val):
    toks = with_bos(torch.from_numpy(np.asarray(seg, dtype=np.int64)).to(dev))  # (1200*129,)
    acc = torch.zeros(6, dtype=torch.float64, device=dev)
    for j in range(0, N_WIN, bs):
      ii = idx[j:j + bs]
      x, y = toks[ii], toks[ii + 1]
      if lik:
        logits = model.logits(x).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
        tok = y.reshape(-1) != BOS
        hit = logits[..., :BOS].argmax(-1).reshape(-1) == y.reshape(-1)
        lg_last = logits[:, LAST:LAST + 128, :BOS]
        acc[:4] += torch.stack([ce.sum(), ce[tok].sum(), tok.sum(), (hit & tok).sum()]).double()
      else:
        lg_last = model.last_frame_logits(strip_bos(x)).float()
      y_last = y[:, LAST:LAST + 128]
      acc[4] += F.cross_entropy(lg_last.reshape(-1, BOS), y_last.reshape(-1), reduction="sum").double()
      acc[5] += (lg_last.argmax(-1) == y_last).sum()
    rows.append(acc.tolist())  # un .item() por segmento
    if log and ((i + 1) % 10 == 0 or i + 1 == len(val)):
      r = np.array(rows)
      print(f"{i + 1}/{len(val)} ce_last={np.mean(r[:, 4] / (N_WIN * 128)):.4f}"
            + (f" ce_all={np.mean(r[:, 0] / (N_WIN * BLOCK)):.4f}" if lik else "")
            + f" ({time.time() - t0:.0f}s)", flush=True)
  r = np.array(rows)
  n_last = len(val) * N_WIN * 128
  res = dict.fromkeys(["ce_all", "ce_tok", "bits_tok", "ppl_tok", "acc_tok"])
  if lik:
    ce_tok = float(np.mean(r[:, 1] / r[:, 2]))
    res = {"ce_all": float(np.mean(r[:, 0] / (N_WIN * BLOCK))), "ce_tok": ce_tok, "bits_tok": ce_tok / math.log(2),
           "ppl_tok": math.exp(ce_tok), "acc_tok": float(r[:, 3].sum() / r[:, 2].sum())}
  res.update(ce_last=float(np.mean(r[:, 4] / (N_WIN * 128))), acc_last=float(r[:, 5].sum() / n_last),
             segments=len(val), seconds=time.time() - t0)
  # por segmento, para tests pareados entre modelos (mismos segmentos de val)
  res["per_segment"] = {"ce_last": (r[:, 4] / (N_WIN * 128)).tolist()}
  if lik:
    res["per_segment"].update(ce_all=(r[:, 0] / (N_WIN * BLOCK)).tolist(), ce_tok=(r[:, 1] / r[:, 2]).tolist())
  return res


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--name", default=None)
  ap.add_argument("--segments", type=int, default=None)
  ap.add_argument("--bs", type=int, default=10)
  ap.add_argument("--dtype", default="fp32", choices=list(DTYPES))
  args = ap.parse_args()
  dev = device()
  log_device(dev)
  model = _load_model(args.model, dev).to(DTYPES[args.dtype]).eval()
  name = args.name or model.name
  val = np.load("data/val_tokens.npy")
  res = {"model": name, "dtype": args.dtype, "device": device_info(dev), **evaluate(model, val, dev, args.bs, args.segments)}
  out = Path(f"results/eval_{name}" + (f"_seg{args.segments}" if args.segments else "")
             + (f"_{args.dtype}" if args.dtype != "fp32" else "") + ".json")
  out.parent.mkdir(exist_ok=True)
  out.write_text(json.dumps(res, indent=1))
  print(json.dumps(res, indent=1))
