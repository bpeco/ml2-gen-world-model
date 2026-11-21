"""Entrena el decoder de visualización de V-JEPA 2-AC sobre un encoder `vjepa2` congelado (post-hoc).

  python -m src.models.vjepa.train_decoder --run vjepa2_s --steps 3000 --bs 64

Carga la corrida (registro/manifiesto, ruta .pt o checkpoints/<run>/final.pt), codifica cada frame solo (tubelet duplicado,
target_encoder, layer_norm) sin gradiente y entrena FrameDecoder con MSE contra el frame. Guarda
checkpoints/<run>_viz/decoder.pt ({"config", "model", "encoder_run", ...}); `cargar(path)` lo reconstruye.
Receta (paper, Fig. 15: AdamW wd 0.1, clip 1.0, warmup 2000 a 5e-4 y coseno; batch 1024 y 150k pasos son presupuesto):
warmup lineal (2000 pasos, o el 10% si hay menos) y coseno a 0.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from src import pixels
from src.common import device
from src.models import load_model
from src.models.vjepa.decoder import FrameDecoder
from src.models.vjepa.patch import normalizar_cthw
from src.models.vjepa.pretrain import layer_norm_canal
from src.windows import Windows


def cargar(path, dev="cpu"):
  ck = torch.load(path, map_location="cpu", weights_only=False)
  d = FrameDecoder(**ck["config"])
  d.load_state_dict(ck["model"])
  return d.to(dev).eval()


def lr_en(step, total, peak=5e-4, warm=2000):
  warm = min(warm, max(1, total // 10))
  if step < warm:
    return peak * (step + 1) / warm
  return 0.5 * peak * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def entrenar(encoder_model, frames_fn, dev, steps=3000, bs=64, dim=256, depth=6, heads=4, wd=0.1, clip=1.0, peak=5e-4,
             log=print):
  """frames_fn(step, bs) -> píxeles uint8 (bs, H, W, 3) en el device. Devuelve (decoder, [mse...])."""
  cfg = encoder_model.cfg
  dec = FrameDecoder(encoder_model.encoder.embed_dim, dim, depth, heads, (cfg.height // cfg.patch, cfg.width // cfg.patch),
                     cfg.patch).to(dev)
  dec.config = dict(embed_dim=encoder_model.encoder.embed_dim, dim=dim, depth=depth, heads=heads,
                    grid_hw=(cfg.height // cfg.patch, cfg.width // cfg.patch), patch=cfg.patch)
  opt = torch.optim.AdamW([{"params": [p for p in dec.parameters() if p.ndim >= 2], "weight_decay": wd},
                           {"params": [p for p in dec.parameters() if p.ndim < 2], "weight_decay": 0.0}],
                          lr=peak, betas=(0.9, 0.999))
  # historial en un buffer del device: guardar loss.detach() retenía todo el buffer elementwise del mse_loss (el escalar
  # comparte storage con él: ~24 MiB por paso con bs 64, y OOM en el paso ~1000), y .item() por paso sincronizaría
  hist = torch.zeros(steps, device=dev)
  for step in range(steps):
    for g in opt.param_groups:
      g["lr"] = lr_en(step, steps, peak)
    img = frames_fn(step, bs)  # (B, H, W, 3) uint8
    x = normalizar_cthw(img.float().div(255).permute(0, 3, 1, 2)[:, :, None])  # (B, 3, 1, H, W)
    with torch.no_grad():
      z = layer_norm_canal(encoder_model.target_encoder.encode_frames(x))  # (B, N, D)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
      pred = dec(z)
    loss = F.mse_loss(pred.float(), x[:, :, 0])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dec.parameters(), clip)
    opt.step()
    hist[step] = loss.detach()
    if (step + 1) % 100 == 0 or step + 1 == steps:
      log(f"step {step + 1}/{steps} mse {loss.item():.4f}")
  return dec, hist.tolist()


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--run", required=True, help="corrida vjepa2 (encoder congelado)")
  p.add_argument("--data", default="data/train_tokens.npy")
  p.add_argument("--steps", type=int, default=3000)
  p.add_argument("--bs", type=int, default=64)
  p.add_argument("--dim", type=int, default=256)
  p.add_argument("--depth", type=int, default=6)
  p.add_argument("--heads", type=int, default=4)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--ckpt-dir", default="checkpoints")
  a = p.parse_args(argv)
  dev = device()
  m = load_model(a.run, dev).eval()
  for q in m.parameters():
    q.requires_grad_(False)
  w = Windows(a.data, seed=a.seed)

  def frames_fn(step, bs):  # un frame al azar por muestra (el decoder es por frame)
    return pixels.render(w.sample(step, bs, 1).to(dev))[:, 0]

  dec, hist = entrenar(m, frames_fn, dev, a.steps, a.bs, a.dim, a.depth, a.heads)
  out = Path(a.ckpt_dir) / f"{a.run}_viz"
  out.mkdir(parents=True, exist_ok=True)
  torch.save({"config": dec.config, "model": dec.state_dict(), "encoder_run": a.run, "steps": a.steps,
              "final_mse": hist[-1]}, out / "decoder.pt")
  (out / "log.json").write_text(json.dumps({"mse": hist[::10], "args": vars(a)}))
  print(f"listo: {out / 'decoder.pt'} mse final {hist[-1]:.4f}")


if __name__ == "__main__":
  main()
