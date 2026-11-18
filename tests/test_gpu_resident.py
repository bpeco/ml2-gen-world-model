"""Entrenamiento residente en la GPU: lotes idénticos al camino CPU, sin sincronizaciones ni copias por paso."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import src
from src import pixels
from src.train import main
from src.windows import Windows

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="sin CUDA")
SRC_DIR = str(Path(src.__file__).parent)


def synth(n=12, t=40, seed=0):
  return np.random.default_rng(seed).integers(0, 1024, (n, t, 128)).astype(np.int16)


def synth_poses(n=12, t=40, seed=1):
  p = np.random.default_rng(seed).normal(size=(n, t, 6)).astype(np.float32)
  p[np.random.default_rng(seed + 1).random(p.shape) < 0.05] = np.nan  # los NaN deben viajar intactos
  return p


# ---------------------------------------------------------------- (a) índices precalculados == camino de antes

@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("seed", [0, 7])
def test_precomputed_equals_legacy_cpu(k, seed):
  tok, pos = synth(), synth_poses()
  legacy = Windows(tok, holdout=2, seed=seed, poses=pos)
  fast = Windows(tok, holdout=2, seed=seed, poses=pos, device="cpu")
  B, n, first = 4, 6, 5
  fast.prepare(20, B, n, k, first_step=first)
  for step in (first, first + 1, first + 9, first + 19):
    assert torch.equal(legacy.sample(step, B, n, frameskip=k), fast.sample(step, B, n, frameskip=k))
    (xl, pl), (xf, pf) = (legacy.sample(step, B, n, with_poses=True, frameskip=k),
                          fast.sample(step, B, n, with_poses=True, frameskip=k))
    assert xf.dtype == torch.long and pf.dtype == torch.float32 and pf.shape == (B, n, 6 * k)
    assert torch.equal(xl, xf)
    assert torch.equal(torch.nan_to_num(pl, nan=-999.0), torch.nan_to_num(pf, nan=-999.0))  # NaN en las mismas posiciones
  # fuera del plan o con otros parámetros: cae al camino de antes, mismo resultado
  assert torch.equal(legacy.sample(first + 50, B, n, frameskip=k), fast.sample(first + 50, B, n, frameskip=k))
  assert torch.equal(legacy.sample(first, B + 1, n, frameskip=k), fast.sample(first, B + 1, n, frameskip=k))
  assert torch.equal(legacy.monitor(6, 5, frameskip=k), fast.monitor(6, 5, frameskip=k))


# ---------------------------------------------------------------- (c) sin .item/.cpu/.numpy/.tolist en el paso (CPU)

def test_no_host_sync_in_train_steps(tmp_path, monkeypatch):
  monkeypatch.setenv("ML2_DEVICE", "cpu")
  data = tmp_path / "tok.npy"
  np.save(data, synth(60, 30))
  armed = {"on": False, "steps": 0}

  def guard(name):
    orig = getattr(torch.Tensor, name)

    def wrapped(self, *a, **kw):
      caller = sys._getframe(1).f_code.co_filename
      if armed["on"] and caller.startswith(SRC_DIR):  # solo el código del repo; el interior de torch (optimizador) puede
        raise AssertionError(f"Tensor.{name} dentro de un paso de entrenamiento ({caller})")
      return orig(self, *a, **kw)
    monkeypatch.setattr(torch.Tensor, name, wrapped)

  for name in ("item", "cpu", "numpy", "tolist"):
    guard(name)
  orig_sample, orig_zero = Windows.sample, torch.optim.Optimizer.zero_grad

  def sample(self, *a, **kw):  # arma al empezar el paso (antes del primer sample) ...
    armed["on"] = True
    return orig_sample(self, *a, **kw)

  def zero_grad(self, *a, **kw):  # ... y desarma tras el optimizador (siguen el log y la evaluación, que sí sincronizan)
    out = orig_zero(self, *a, **kw)
    armed["on"] = False
    armed["steps"] += 1
    return out

  monkeypatch.setattr(Windows, "sample", sample)
  monkeypatch.setattr(torch.optim.Optimizer, "zero_grad", zero_grad)
  args = ["--kind", "gpt", "--config", json.dumps({"n_layer": 2, "n_head": 2, "dim": 32}), "--run", "g", "--tokens", str(1032 * 400), "--max-steps", "5",
          "--bs", "2", "--frames", "4", "--lr", "3e-3", "--eval-every", "1000", "--log-every", "1000",
          "--data", str(data), "--ckpt-dir", str(tmp_path / "ck"), "--results-dir", str(tmp_path / "res")]
  res = main(args)
  assert res["steps"] == 5 and armed["steps"] == 5


# ---------------------------------------------------------------- pixels: render en el device

def test_render_float_equals_render_cpu():
  tok = torch.from_numpy(synth(1, 2)[0].astype(np.int64))[None]
  img, f = pixels.render(tok), pixels.render_float(tok)
  assert img.dtype == torch.uint8 and f.dtype == torch.float32 and f.shape == img.shape and f.device == img.device
  assert torch.equal(f, img.float() / 255)
  mean, std = torch.tensor([0.4, 0.5, 0.6]), torch.tensor([0.2, 0.3, 0.4])
  assert torch.allclose(pixels.render_float(tok, mean=mean, std=std), (f - mean) / std, atol=1e-6)


# ---------------------------------------------------------------- (b) CUDA: cero Memcpy HtoD/DtoH en régimen

def copies(prof):
  return [e.name for e in prof.events() if "Memcpy HtoD" in e.name or "Memcpy DtoH" in e.name]


def steady_state_copies(step_fn, warm=3, n=7):
  from torch.profiler import ProfilerActivity, profile
  for i in range(warm):  # warmup: allocator, cuDNN, estado del optimizador (paso en device), codec
    step_fn(i)
  torch.cuda.synchronize()
  with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for i in range(warm, warm + n):
      step_fn(i)
    torch.cuda.synchronize()  # el sync del propio test no copia datos
  return copies(prof)


@needs_cuda
def test_gpt_train_steps_no_memcpy_cuda():
  from src.models import gpt
  dev = torch.device("cuda")
  torch.manual_seed(0)
  model = gpt.GPT(gpt.GPT.Config(n_layer=2, n_head=2, dim=64, dropout=0.0)).to(dev).train()
  opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
  scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16/fp32: deshabilitado, sin sync
  w = Windows(synth(30, 30), holdout=2, device=dev).prepare(20, 2, 4)

  def step_fn(i):
    for g in opt.param_groups:
      g["lr"] = 1e-3 * (i + 1) / 20
    x = w.sample(i, 2, 4)
    with torch.autocast("cuda", dtype=torch.bfloat16):
      loss, met = model.train_loss(x)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)

  assert w.sample(0, 2, 4).device.type == "cuda"
  assert steady_state_copies(step_fn) == []


@needs_cuda
def test_pixel_model_train_steps_no_memcpy_cuda():
  dev = torch.device("cuda")
  torch.manual_seed(0)
  net = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 8, stride=8), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1),
                            torch.nn.Flatten(), torch.nn.Linear(8, 4)).to(dev).train()
  opt = torch.optim.AdamW(net.parameters(), lr=1e-3, fused=True)
  w = Windows(synth(30, 30), holdout=2, device=dev).prepare(20, 2, 2)
  mean, std = torch.tensor([0.45] * 3, device=dev), torch.tensor([0.25] * 3, device=dev)

  def step_fn(i):
    x = w.sample(i, 2, 2)
    img = pixels.render_float(x, mean=mean, std=std)  # (B, 2, 128, 256, 3) en la GPU
    assert img.device.type == "cuda"
    y = net(img.reshape(-1, 128, 256, 3).permute(0, 3, 1, 2))
    loss = y.pow(2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

  assert steady_state_copies(step_fn) == []
