"""Tests con datos sintéticos de Windows, ConvLSTM y src/train.py (CPU, pequeños)."""
import json

import numpy as np
import pytest
import torch

from src.models import load_checkpoint
from src.models.base import FrameModel
from src.models.convlstm import ConvLSTM
from src.train import main
from src.windows import Windows


def synth(n=12, t=40, seed=0):
  return np.random.default_rng(seed).integers(0, 1024, (n, t, 128)).astype(np.int16)


def tiny_convlstm():
  return ConvLSTM(ConvLSTM.Config(emb=8, hidden=16, n_layers=2, kernel=3))


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


def test_convlstm_causal():
  torch.manual_seed(0)
  m = tiny_convlstm().eval()
  x = torch.randint(0, 1024, (2, 6, 128))
  y = x.clone()
  y[:, 3] = torch.randint(0, 1024, (2, 128))
  la, lb = m.frame_logits(x), m.frame_logits(y)
  assert la.shape == (2, 6, 128, 1024)
  assert torch.allclose(la[:, :4], lb[:, :4], atol=1e-6)      # frames <= 3 no ven el frame 3
  assert not torch.allclose(la[:, 4], lb[:, 4], atol=1e-4)    # el frame 4 sí


def test_convlstm_stateful_generate_matches_stateless():
  torch.manual_seed(0)
  m = tiny_convlstm().eval()
  ctx = torch.randint(0, 1024, (2, 3, 128))
  fast = m.rollout(ctx, 4, temperature=0)
  slow = torch.stack(list(FrameModel_generate(m, ctx, 4)), 1)
  assert torch.equal(fast, slow)
  # con temperatura > 0 y mismo generator, mismos frames
  g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
  assert torch.equal(m.rollout(ctx, 3, generator=g1), torch.stack(list(FrameModel_generate(m, ctx, 3, 1.0, g2)), 1))


def FrameModel_generate(m, ctx, n, temperature=0, generator=None):
  """Camino sin estado de la clase base (next_frame repetido)."""
  from src.models.base import WorldModel
  return WorldModel.generate(m, ctx, n, temperature, None, generator)


def test_convlstm_no_kd():
  with pytest.raises(ValueError):
    tiny_convlstm().train_loss(torch.zeros(1, 3, 128, dtype=torch.long), teacher=object())


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


def test_overfit_convlstm():
  first, last = overfit(ConvLSTM(ConvLSTM.Config(emb=16, hidden=32)))
  assert first > 6.5 and last < 0.5 * first


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
  return ["--kind", "convlstm", "--config", json.dumps({"emb": 8, "hidden": 16}), "--run", run, "--tokens", str(1032 * 6),
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
