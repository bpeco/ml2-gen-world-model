"""Entrena el decoder de visualización de un LeWM ya entrenado (post-hoc, sobre el modelo congelado).

  python -m src.train_decoder --run lewm
  python -m src.train_decoder --run lewm --steps 4000 --bs 64 --config '{"layers": 6}'

El LeWM se congela (eval, sin gradiente: sus BatchNorm usan las estadísticas guardadas); los frames se generan en la GPU
con el decoder de comma (src/pixels.py), se codifican con el LeWM y el decoder los reconstruye. Pérdida MSE entre
la imagen reconstruida y la del render, ambas en [-1, 1]. Solo para visualizar: el paper (App. D) no publica el
código del decoder, así que arquitectura, MSE, AdamW (lr 3e-4, wd 0.01, clip 1.0, warmup 100, coseno a 0, bf16 en
CUDA) y presupuesto (--steps x --bs frames) son propios.

Escribe checkpoints/<run>/decoder.pt y results/train_decoder_<run>.json.
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch

from src import schedules as S
from src.common import device, device_info, log_device
from src.models import load_model
from src.models.lewm_decoder import DecoderConfig, LeWMDecoder, save_decoder, target_of
from src.pixels import render
from src.windows import Windows


def parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--run", required=True, help="corrida de LeWM (nombre registrado, checkpoints/<run>/final.pt o ruta)")
  p.add_argument("--data", default="data/train_tokens.npy")
  p.add_argument("--config", default="{}", help="JSON con campos de DecoderConfig")
  p.add_argument("--steps", type=int, default=3000)
  p.add_argument("--bs", type=int, default=64, help="frames por paso")
  p.add_argument("--lr", type=float, default=3e-4)
  p.add_argument("--wd", type=float, default=0.01)
  p.add_argument("--clip", type=float, default=1.0)
  p.add_argument("--warmup", type=int, default=100)
  p.add_argument("--amp", choices=["bf16", "none"], default="bf16", help="solo en CUDA")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--eval-every", type=int, default=500)
  p.add_argument("--log-every", type=int, default=50)
  p.add_argument("--ckpt-dir", default="checkpoints")
  p.add_argument("--results-dir", default="results")
  return p.parse_args(argv)


def run_name(spec: str) -> str:
  """Nombre de la corrida: el mismo spec, o el de la carpeta/archivo si es una ruta a un .pt."""
  p = Path(spec)
  if spec.endswith(".pt"):
    return p.parent.name if p.stem in ("final", "last") else p.stem
  return spec


def decoder_loss(model, dec, tokens):
  """MSE del decoder sobre frames (B, 128) tokens (o (B, H, W, 3) píxeles). El LeWM no recibe gradiente."""
  with torch.no_grad():
    img = tokens if tokens.dim() == 4 else render(tokens)
    z = model.encode(img[:, None])[:, 0]
  return (dec(z.detach()).float() - target_of(img)).square().mean()


@torch.no_grad()
def holdout_mse(model, dec, windows, bs, dev, autocast):
  was = dec.training
  dec.eval()
  x = windows.monitor(n_windows=32, n_frames=1)[:, 0]
  tot = 0.0
  for i in range(0, len(x), bs):
    with autocast():
      tot += decoder_loss(model, dec, x[i:i + bs].to(dev)).item() * len(x[i:i + bs])
  dec.train(was)
  return tot / len(x)


def main(argv=None):
  a = parse(argv)
  dev = device()
  log_device(dev)
  torch.manual_seed(a.seed)
  model = load_model(a.run, dev).eval()
  if not hasattr(model, "encode"):
    raise SystemExit(f"{a.run}: no es un modelo latente (LeWM)")
  for p_ in model.parameters():
    p_.requires_grad_(False)
  cfg = DecoderConfig(**{"dim_in": model.cfg.dim, "patch": model.cfg.patch, "img_h": model.cfg.img_h,
                         "img_w": model.cfg.img_w, **json.loads(a.config)})
  dec = LeWMDecoder(cfg).to(dev).train()
  opt = torch.optim.AdamW(dec.parameters(), lr=0.0, weight_decay=a.wd, fused=dev.type == "cuda")
  segs = [S.linear(0.0, a.lr, pasos=a.warmup, nombre="warmup"), S.cosine(a.lr, 0.0)]
  adtype = torch.bfloat16 if (a.amp == "bf16" and dev.type == "cuda") else None
  autocast = (lambda: torch.autocast(dev.type, dtype=adtype)) if adtype else (lambda: torch.autocast(dev.type, enabled=False))
  windows = Windows(a.data, seed=a.seed)
  print(f"decoder de {a.run}: {dec.num_params() / 1e6:.2f}M params, {a.steps} pasos x {a.bs} frames", flush=True)

  t0, hist, loss_acc, n_acc = time.time(), [], 0.0, 0
  for step in range(a.steps):
    torch.manual_seed(a.seed * 1_000_003 + step)
    lr = S.value(segs, step, a.steps)
    for g in opt.param_groups:
      g["lr"] = lr
    x = windows.sample(step, a.bs, 1)[:, 0].to(dev, non_blocking=True)
    with autocast():
      loss = decoder_loss(model, dec, x)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dec.parameters(), a.clip)
    opt.step()
    opt.zero_grad(set_to_none=True)
    loss_acc = loss_acc + loss.detach()
    n_acc += 1
    if (step + 1) % a.log_every == 0 or step + 1 == a.steps:
      print(f"step {step + 1} lr {lr:.3g} mse {float(loss_acc) / n_acc:.5f}", flush=True)
      loss_acc, n_acc = 0.0, 0
    if (step + 1) % a.eval_every == 0 and step + 1 < a.steps:
      hist.append({"step": step + 1, "holdout_mse": holdout_mse(model, dec, windows, a.bs, dev, autocast)})
      print("[eval]", hist[-1], flush=True)
  hist.append({"step": a.steps, "holdout_mse": holdout_mse(model, dec, windows, a.bs, dev, autocast)})
  assert all(p_.grad is None for p_ in model.parameters()), "el LeWM recibió gradiente"

  name = run_name(a.run)
  run_dir = Path(a.ckpt_dir) / name
  run_dir.mkdir(parents=True, exist_ok=True)
  save_decoder(run_dir / "decoder.pt", dec, lewm_run=a.run, steps=a.steps)
  res = {"run": name, "lewm": a.run, "decoder": dataclasses.asdict(cfg), "params": dec.num_params(), "steps": a.steps,
         "frames": a.steps * a.bs, "seconds": time.time() - t0, "holdout_mse": hist[-1]["holdout_mse"],
         "history": hist, "args": vars(a), "device_info": device_info(dev),
         "nota": "decoder de visualización post-hoc sobre el LeWM congelado; arquitectura y receta propias (el paper no las publica)"}
  Path(a.results_dir).mkdir(parents=True, exist_ok=True)
  (Path(a.results_dir) / f"train_decoder_{name}.json").write_text(json.dumps(res, indent=1))
  print(f"listo: holdout MSE {res['holdout_mse']:.5f} -> {run_dir / 'decoder.pt'}")
  return res


if __name__ == "__main__":
  main()
