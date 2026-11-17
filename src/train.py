"""Entrenamiento de los world models propios (gpt, maskgit, convlstm), con o sin destilación.

  python -m src.train --kind gpt --size s --run gpt_s_scratch --tokens 100e6
  python -m src.train --kind gpt --size s --run gpt_s_kd --tokens 100e6 --kd

Los datos dependen solo de (seed, step): scratch y KD con los mismos --tokens/--seed/--bs ven los
mismos batches. Reanuda solo desde <ckpt-dir>/<run>/last.pt (Colab corta las sesiones).
"""
import argparse
import dataclasses
import json
import math
import os
import time
import warnings
from pathlib import Path

import torch

from src.common import TPF, device, device_info, log_device
from src.models import KINDS, build, load_model
from src.windows import Windows


def parse(argv=None):
  p = argparse.ArgumentParser()
  p.add_argument("--kind", required=True, choices=list(KINDS))
  p.add_argument("--size", choices=["s", "m", "l"])
  p.add_argument("--config", default="{}", help="JSON con campos de Config, p. ej. '{\"dropout\":0.1}'")
  p.add_argument("--run", required=True)
  p.add_argument("--tokens", type=float, default=100e6)
  p.add_argument("--bs", type=int, default=16)
  p.add_argument("--accum", type=int, default=1)
  p.add_argument("--lr", type=float, default=1e-3)
  p.add_argument("--min-lr-frac", type=float, default=0.1)
  p.add_argument("--warmup", type=int, default=200)
  p.add_argument("--wd", type=float, default=0.1)
  p.add_argument("--clip", type=float, default=1.0)
  p.add_argument("--amp", choices=["fp16", "bf16", "none"], default=None, help="default: fp16 en CUDA, none en el resto")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--frames", type=int, default=20)
  p.add_argument("--kd", action="store_true")
  p.add_argument("--alpha", type=float, default=0.5)
  p.add_argument("--tau", type=float, default=1.0)
  p.add_argument("--data", default="data/train_tokens.npy")
  p.add_argument("--poses", default="data/train_poses.npy", help="poses por frame (solo modelos con uses_poses)")
  p.add_argument("--ckpt-dir", default="checkpoints")
  p.add_argument("--ckpt-every-min", type=float, default=15)
  p.add_argument("--eval-every", type=int, default=250)
  p.add_argument("--log-every", type=int, default=25)
  p.add_argument("--compile", action="store_true")
  p.add_argument("--max-steps", type=int)
  p.add_argument("--results-dir", default="results")
  return p.parse_args(argv)


def lr_at(step, a, total):
  """Warmup lineal y coseno hasta min_lr_frac * lr."""
  if step < a.warmup:
    return a.lr * (step + 1) / a.warmup
  prog = (step - a.warmup) / max(1, total - a.warmup)
  return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))


def make_optimizer(model, a, dev):
  # weight decay en todo lo de >= 2 dimensiones (matrices, convs, embeddings); no en biases ni normas
  decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
  other = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
  groups = [{"params": decay, "weight_decay": a.wd}, {"params": other, "weight_decay": 0.0}]
  return torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95), fused=dev.type == "cuda")


def atomic_save(obj, path):
  tmp = str(path) + ".tmp"
  torch.save(obj, tmp)
  os.replace(tmp, path)


@torch.no_grad()
def monitor_ce(model, windows, bs, autocast, dev):
  """CE (sin maestro) sobre las ventanas fijas del holdout. No altera el RNG del entrenamiento."""
  rng = torch.get_rng_state()
  cuda_rng = torch.cuda.get_rng_state_all() if dev.type == "cuda" else None
  torch.manual_seed(1234)
  was_training = model.training
  model.eval()
  x, tot, n = windows.monitor(), 0.0, 0
  for i in range(0, len(x), bs):
    xb = x[i:i + bs].to(dev)
    with autocast():
      loss, met = model.train_loss(xb)
    # los modelos latentes (jepa, lewm) suman otras pérdidas: se monitorea la CE de su lectura a tokens
    tot += met.get("ce", loss).item() * len(xb)
    n += len(xb)
  model.train(was_training)
  torch.set_rng_state(rng)
  if cuda_rng is not None:
    torch.cuda.set_rng_state_all(cuda_rng)
  return tot / n


def main(argv=None):
  a = parse(argv)
  dev = device()
  log_device(dev)
  run_dir = Path(a.ckpt_dir) / a.run
  run_dir.mkdir(parents=True, exist_ok=True)
  last_path = run_dir / "last.pt"
  tok_per_step = a.bs * a.accum * a.frames * TPF
  total = math.ceil(a.tokens / tok_per_step)
  stop_at = min(total, a.max_steps) if a.max_steps else total

  torch.manual_seed(a.seed)
  model = build(a.kind, json.loads(a.config), a.size).to(dev)
  # num_params(): solo lo que se usa en inferencia (los latentes no cuentan su encoder target EMA)
  n_params = model.num_params() if hasattr(model, "num_params") else sum(p.numel() for p in model.parameters())
  use_poses = getattr(model, "uses_poses", False)
  windows = Windows(a.data, seed=a.seed, poses=a.poses if use_poses else None)
  if use_poses:  # normalización de las acciones; viaja en el checkpoint como buffer
    model.set_pose_stats(*windows.pose_stats())
  opt = make_optimizer(model, a, dev)
  amp = a.amp or ("fp16" if dev.type == "cuda" else "none")
  adtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
  scaler = torch.amp.GradScaler(dev.type, enabled=amp == "fp16" and dev.type == "cuda")
  autocast = (lambda: torch.autocast(dev.type, dtype=adtype)) if adtype else (lambda: torch.autocast(dev.type, enabled=False))

  step, tokens, elapsed, history = 0, 0, 0.0, []
  if last_path.exists():
    ck = torch.load(last_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    scaler.load_state_dict(ck["scaler"])
    step, tokens, elapsed, history = ck["step"], ck["tokens"], ck["elapsed"], ck["history"]
    print(f"reanudando {last_path} en el paso {step}/{total}", flush=True)
  print(f"{a.kind} {a.size or ''} {n_params / 1e6:.2f}M params, {total} pasos, {tok_per_step} tokens/paso, amp={amp}", flush=True)

  teacher = None
  if a.kd:
    teacher = load_model("gpt2m", dev).to(torch.float16).eval()
    for p in teacher.parameters():
      p.requires_grad_(False)

  loss_fn = model.train_loss
  if a.compile:
    try:
      loss_fn = torch.compile(model.train_loss)
    except Exception as e:
      warnings.warn(f"torch.compile falló, sigo en eager: {e}")
  log_f = open(run_dir / "log.jsonl", "a")

  def log(rec):
    log_f.write(json.dumps(rec) + "\n")
    log_f.flush()

  t0 = time.time()  # inicio de esta sesión; elapsed acumula las anteriores

  def save_last():
    atomic_save({"kind": a.kind, "config": dataclasses.asdict(model.cfg), "model": model.state_dict(),
                 "optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "step": step, "tokens": tokens,
                 "args": vars(a), "name": a.run, "elapsed": elapsed + time.time() - t0, "history": history,
                 "best": min((h["monitor_ce"] for h in history), default=None)}, last_path)

  def evaluate():
    ce = monitor_ce(model, windows, a.bs, autocast, dev)
    history.append({"step": step, "tokens": tokens, "monitor_ce": ce})
    log({"step": step, "tokens": tokens, "monitor_ce": ce})
    print(f"[eval] step {step} monitor_ce {ce:.4f}", flush=True)

  model.train()
  t_log, tok_log, t_ckpt = t0, tokens, t0
  acc, n_acc = {}, 0
  while step < stop_at:
    torch.manual_seed(a.seed * 1_000_003 + step)  # dropout / máscaras de MaskGIT reproducibles al reanudar
    lr = lr_at(step, a, total)
    for g in opt.param_groups:
      g["lr"] = lr
    for i in range(a.accum):
      kw = dict(teacher=teacher, alpha=a.alpha, tau=a.tau)
      if use_poses:  # mismos tokens que sin poses: las poses salen de los mismos segmentos y frames
        x, p = windows.sample(step * a.accum + i, a.bs, a.frames, with_poses=True)
        kw["poses"] = p.to(dev, non_blocking=True)
      else:
        x = windows.sample(step * a.accum + i, a.bs, a.frames)
      x = x.to(dev, non_blocking=True)
      with autocast():
        try:
          loss, met = loss_fn(x, **kw)
        except Exception as e:
          if loss_fn is model.train_loss:
            raise
          warnings.warn(f"torch.compile falló al ejecutar, sigo en eager: {e}")
          loss_fn = model.train_loss
          loss, met = loss_fn(x, **kw)
      scaler.scale(loss / a.accum).backward()
      for k, v in {"loss": loss.detach(), **met}.items():
        acc[k] = acc.get(k, 0.0) + v.float() / a.accum
    n_acc += 1
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    step += 1
    tokens += tok_per_step
    if hasattr(model, "after_step"):  # EMA del encoder target (jepa, lewm_s_ema)
      model.after_step(step, total)

    if step % a.log_every == 0 or step == stop_at:
      now = time.time()
      rec = {"step": step, "tokens": tokens, "lr": lr, **{k: v.item() / n_acc for k, v in acc.items()},
             "tok_s": (tokens - tok_log) / max(now - t_log, 1e-9), "elapsed": elapsed + now - t0}
      acc, n_acc = {}, 0
      print(" ".join(f"{k} {v:.4g}" for k, v in rec.items()), flush=True)
      log(rec)
      t_log, tok_log = time.time(), tokens
    if step % a.eval_every == 0 and step < stop_at:
      evaluate()
      t_log, tok_log = time.time(), tokens
    if (time.time() - t_ckpt) / 60 >= a.ckpt_every_min:
      save_last()
      t_ckpt = time.time()

  if not history or history[-1]["step"] != step:
    evaluate()
  save_last()
  elapsed += time.time() - t0
  atomic_save({"kind": a.kind, "config": dataclasses.asdict(model.cfg), "model": model.state_dict(),
               "name": a.run, "step": step, "tokens": tokens}, run_dir / "final.pt")
  res = {"run": a.run, "kind": a.kind, "size": a.size, "config": dataclasses.asdict(model.cfg), "params": n_params,
         "tokens": tokens, "steps": step, "seconds": elapsed, "tokens_per_s": tokens / max(elapsed, 1e-9),
         "final_monitor_ce": history[-1]["monitor_ce"], "monitor_history": history, "args": vars(a),
         "device_info": device_info(dev)}
  Path(a.results_dir).mkdir(parents=True, exist_ok=True)
  (Path(a.results_dir) / f"train_{a.run}.json").write_text(json.dumps(res, indent=1))
  print(f"listo: {step} pasos, {tokens} tokens, {res['tokens_per_s']:.0f} tok/s, monitor CE {res['final_monitor_ce']:.4f}")
  return res


if __name__ == "__main__":
  main()
