"""CE según la posición dentro de la ventana de 20 frames -> results/ce_por_posicion.json.

- Modelos con verosimilitud (GPT, gpt2m, baselines): CE de los 128 códigos de cada frame f de
  la ventana, con el protocolo de comma (ventanas que arrancan en los frames 0, 20, 40... del segmento).
- MaskGIT: CE con 1 paso del frame f dados los f frames anteriores limpios (last_frame_logits).

Uso: python scripts/ce_por_posicion.py [--segments 10]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from src.common import BLOCK, TPF, device, with_bos  # noqa: E402
from src.models import load_model  # noqa: E402

LIK = ["gpt2m", "gpt_s_abs", "gpt_s", "gpt_m_kd", "transicion"]
MG = ["maskgit_s", "maskgit_s_nopc", "maskgit_s_abs", "maskgit_m"]


@torch.no_grad()
def by_frame(m, val, dev, segs):
  tot, n = torch.zeros(20, device=dev), 0
  for s in segs:
    toks = with_bos(torch.from_numpy(val[s].astype(np.int64)).to(dev))
    idx = torch.arange(59 * BLOCK, device=dev).view(59, BLOCK)
    for j in range(0, 59, 10):
      ii = idx[j:j + 10]
      lg = m.logits(toks[ii]).float()
      ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), toks[ii + 1].reshape(-1), reduction="none")
      tot += ce.view(len(ii), 20, TPF)[:, :, :128].sum((0, 2))
      n += len(ii) * 128
  return (tot / n).tolist()


@torch.no_grad()
def by_context(m, val, dev, n_win=64, seed=0):
  rng = np.random.default_rng(seed)
  segs, starts = rng.integers(0, len(val), n_win), rng.integers(0, 1200 - 20, n_win)
  W = torch.from_numpy(np.stack([val[s, t:t + 20] for s, t in zip(segs, starts)]).astype(np.int64)).to(dev)
  out = []
  for f in range(20):
    ce = 0.0
    for i in range(0, n_win, 16):
      fr = W[i:i + 16, :f + 1]
      ce += F.cross_entropy(m.last_frame_logits(fr).float().reshape(-1, 1024), fr[:, -1].reshape(-1), reduction="sum").item()
    out.append(ce / (n_win * 128))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--segments", type=int, default=10)
  a = ap.parse_args()
  dev = device()
  val = np.load("data/val_tokens.npy")
  segs = list(range(0, len(val), len(val) // a.segments))[:a.segments]
  res = {"segments": segs, "por_frame": {}, "por_contexto": {}}
  for name in LIK:
    res["por_frame"][name] = by_frame(load_model(name, dev).eval(), val, dev, segs)
    print(name, " ".join(f"{v:.2f}" for v in res["por_frame"][name]), flush=True)
  for name in MG:
    res["por_contexto"][name] = by_context(load_model(name, dev).eval(), val, dev)
    print(name, " ".join(f"{v:.2f}" for v in res["por_contexto"][name]), flush=True)
  Path("results/ce_por_posicion.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
  main()
