"""Tests con datos sintéticos de Windows y src/train.py (CPU, pequeños)."""
import json

import numpy as np
import pytest
import torch

from src.models import load_checkpoint
from src.train import main
from src.windows import Windows


def synth(n=12, t=40, seed=0):
  return np.random.default_rng(seed).integers(0, 1024, (n, t, 128)).astype(np.int16)


def test_windows_deterministic():
  tok = synth()
  a, b = Windows(tok, holdout=2, seed=3), Windows(tok, holdout=2, seed=3)
  assert a.sample(5, 4, 10).shape == (4, 10, 128) and a.sample(5, 4, 10).dtype == torch.long
  assert torch.equal(a.sample(5, 4, 10), b.sample(5, 4, 10))
  assert not torch.equal(a.sample(5, 4, 10), a.sample(6, 4, 10))
  assert not torch.equal(a.sample(5, 4, 10), Windows(tok, holdout=2, seed=4).sample(5, 4, 10))
  # train no toca el holdout; monitor sí lo usa y es fijo
  w = Windows(tok, holdout=2)
  hold = {tuple(tok[i, :, 0]) for i in (10, 11)}
  m = w.monitor(6, 10)
  assert torch.equal(m, w.monitor(6, 10)) and m.shape == (6, 10, 128)
  assert all(tuple(r.tolist()) in {h[s:s + 10] for h in hold for s in range(31)} for r in m[:, :, 0])
  for step in range(30):
    x = w.sample(step, 4, 10)
    assert not any(tuple(r.tolist()) in {h[s:s + 10] for h in hold for s in range(31)} for r in x[:, :, 0])


def overfit(model, steps=100, lr=3e-3):
  torch.manual_seed(0)
  x = torch.from_numpy(synth(1, 4).astype(np.int64))  # un solo segmento: el frame 0 no tiene contexto
  opt = torch.optim.AdamW(model.parameters(), lr=lr)
  first = None
  for _ in range(steps):
    loss, _ = model.train_loss(x)
    first = first if first is not None else loss.item()
    opt.zero_grad()
    loss.backward()
    opt.step()
  return first, loss.item()


def test_overfit_gpt():
  gpt = pytest.importorskip("src.models.gpt")
  first, last = overfit(gpt.GPT(gpt.GPT.Config(n_layer=2, n_head=2, dim=64, dropout=0.0)), steps=150)
  assert last < 0.5 * first


def test_kd_alpha1_equals_ce():
  gpt = pytest.importorskip("src.models.gpt")
  cfg = gpt.GPT.Config(n_layer=1, n_head=2, dim=32, dropout=0.0)
  torch.manual_seed(0)
  student, teacher = gpt.GPT(cfg), gpt.GPT(cfg).eval()
  x = torch.from_numpy(synth(2, 3).astype(np.int64))
  ce, _ = student.train_loss(x)
  kd, met = student.train_loss(x, teacher=teacher, alpha=1.0, tau=2.0)
  assert "kl" in met and torch.allclose(ce, kd, atol=1e-6)


def run_args(tmp_path, data, run, extra=()):
  return ["--kind", "gpt", "--config", json.dumps({"n_layer": 2, "n_head": 2, "dim": 32, "dropout": 0.0}), "--run", run,
          "--receta", "legacy_cosine", "--tokens", str(1032 * 6),
          "--bs", "2", "--frames", "4", "--warmup", "2", "--lr", "3e-3", "--eval-every", "2", "--log-every", "1",
          "--data", str(data), "--ckpt-dir", str(tmp_path / "ck"), "--results-dir", str(tmp_path / "res"), *extra]


def test_train_resume_reproducible(tmp_path, monkeypatch):
  monkeypatch.setenv("ML2_DEVICE", "cpu")
  data = tmp_path / "tok.npy"
  np.save(data, synth(60, 30))
  full = main(run_args(tmp_path, data, "full"))
  assert full["steps"] == 6
  main(run_args(tmp_path, data, "split", ["--max-steps", "3"]))
  split = main(run_args(tmp_path, data, "split"))  # reanuda desde last.pt (paso 3)
  assert split["steps"] == 6
  a = load_checkpoint(tmp_path / "ck" / "full" / "final.pt").state_dict()
  b = load_checkpoint(tmp_path / "ck" / "split" / "final.pt").state_dict()
  for k in a:
    assert torch.allclose(a[k], b[k], atol=1e-6), k
  assert abs(full["final_monitor_ce"] - split["final_monitor_ce"]) < 1e-5
  assert (tmp_path / "res" / "train_full.json").exists()
