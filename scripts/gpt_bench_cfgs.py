"""Checkpoints de GPT con pesos al azar para medir latencia por configuración (fase 1c del plan de movimiento).
  python scripts/gpt_bench_cfgs.py 6x384 4x512 ...   -> checkpoints/_bench/gpt_<L>x<D>.pt"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import build  # noqa: E402

out = Path("checkpoints/_bench")
out.mkdir(parents=True, exist_ok=True)
for cfg in sys.argv[1:]:
  L, D = map(int, cfg.split("x"))
  m = build("gpt", {"n_layer": L, "dim": D, "n_head": D // 64}, "s")
  n = sum(p.numel() for p in m.parameters())
  torch.save({"kind": "gpt", "config": {**m.cfg.__dict__}, "model": m.state_dict(), "name": f"gpt_{cfg}"},
             out / f"gpt_{cfg}.pt")
  print(cfg, f"{n / 1e6:.1f} M")
