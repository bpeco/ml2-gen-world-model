"""Tests con datos sintéticos de los baselines y del layout de FrameModel."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.common import BOS, TPF, VOCAB, with_bos
from src.models.baselines import CopyMix, Transition, Unigram, fit_stats, load_baseline


def synth(n=8, f=30, repeat=0.9, seed=0):
  """Segmentos donde cada celda repite la del frame anterior con prob `repeat`."""
  rng = np.random.default_rng(seed)
  a = rng.integers(0, 1024, (n, f, 128))
  for t in range(1, f):
    keep = rng.random((n, 128)) < repeat
    a[:, t] = np.where(keep, a[:, t - 1], a[:, t])
  return a.astype(np.int16)


@pytest.fixture(scope="module")
def stats():
  return fit_stats(synth())


def test_fit_stats(stats):
  uni, p, trans, alpha = stats
  assert trans.shape == (1024, 1024) and trans.dtype == np.float32 and alpha > 0
  assert np.allclose(np.exp(trans).sum(1), 1, atol=1e-4)
  assert uni.shape == (1024,) and abs(uni.sum() - 1) < 1e-9
  assert 0.8 < p < 0.97  # ~0.9 + algo por coincidencias


def test_copy_prefers_repeating(stats):
  m = CopyMix(stats[0], stats[1])
  fr = torch.from_numpy(synth(2, 6).astype(np.int64))
  lg = m.frame_logits(fr)
  assert lg.shape == (2, 6, 128, 1024)
  assert torch.allclose(lg.exp().sum(-1), torch.ones(2, 6, 128), atol=1e-4)  # distribuciones válidas
  assert (lg[:, 1:].argmax(-1) == fr[:, :-1]).all()
  assert torch.allclose(lg[:, 0], m.logp.expand(2, 128, 1024))               # frame 0: unigram
  assert m.last_frame_logits(fr).shape == (2, 128, 1024)


def test_framemodel_layout_bos_certain(stats):
  m = CopyMix(stats[0], stats[1])
  fr = torch.from_numpy(synth(1, 3).astype(np.int64))
  x = with_bos(fr)
  lg = m.logits(x)
  assert lg.shape == (1, 3 * TPF, VOCAB)
  lp = F.log_softmax(lg.float(), -1)
  slot = torch.arange(3) * TPF + 128                      # el último token predice el BOS del frame siguiente
  assert (lp[0, slot, BOS].exp() > 0.999).all()
  assert (lp[0].exp().sum(-1) - 1).abs().max() < 1e-4
  assert (lp[0, ~torch.isin(torch.arange(3 * TPF), slot), BOS].exp() < 1e-6).all()


def test_unigram_uniform_ce():
  rng = np.random.default_rng(1)
  uni, _ = fit_stats(rng.integers(0, 1024, (40, 20, 128)).astype(np.int16))[:2]
  m = Unigram(uni)
  fr = torch.from_numpy(rng.integers(0, 1024, (2, 20, 128)))
  x = with_bos(fr)
  lg = m.logits(x)[:, :-1]
  ce = F.cross_entropy(lg.reshape(-1, VOCAB), x[:, 1:].reshape(-1), reduction="none").view(2, -1)
  tok = (x[:, 1:] != BOS)
  assert abs(ce[tok].mean().item() - (np.log(1024) + np.log(129 / 128))) < 0.1
  assert abs(ce[~tok].mean().item() - np.log(129)) < 1e-4
  assert m.last_frame_logits(fr).shape == (2, 128, 1024)


@pytest.mark.parametrize("kind", ["unigram", "transicion", "copy_mix"])
def test_next_frame(stats, kind):
  m = {"unigram": lambda: Unigram(stats[0]), "transicion": lambda: Transition(stats[0], stats[2]),
       "copy_mix": lambda: CopyMix(stats[0], stats[1])}[kind]()
  ctx = torch.from_numpy(synth(3, 5).astype(np.int64))
  g = lambda s: torch.Generator().manual_seed(s)
  a, b = m.next_frame(ctx, generator=g(0)), m.next_frame(ctx, generator=g(0))
  assert a.shape == (3, 128) and a.min() >= 0 and a.max() < 1024
  assert torch.equal(a, b)
  assert not torch.equal(a, m.next_frame(ctx, generator=g(1)))
  assert m.rollout(ctx, 4, generator=g(0)).shape == (3, 4, 128)
  if kind in ("transicion", "copy_mix"):  # con p_copy ~0.9, la mayoría de celdas repiten
    assert (m.next_frame(ctx, generator=g(0)) == ctx[:, -1]).float().mean() > 0.7


def test_half_precision(stats):
  m = CopyMix(stats[0], stats[1]).to(torch.float16)
  fr = torch.from_numpy(synth(1, 3).astype(np.int64))
  assert torch.isfinite(m.logits(with_bos(fr)).float()[..., :1024].max())


def test_transition_learns_mapping():
  """Si cada celda pasa siempre de x a (x + 1) % 1024, la tabla lo aprende."""
  rng = np.random.default_rng(3)
  a = np.zeros((40, 20, 128), np.int64)
  a[:, 0] = rng.integers(0, 1024, (40, 128))
  for t in range(1, 20):
    a[:, t] = (a[:, t - 1] + 1) % 1024
  uni, _, trans, alpha = fit_stats(a.astype(np.int16))
  m = Transition(uni, trans)
  fr = torch.from_numpy(a[:2])
  lg = m.frame_logits(fr)
  assert (lg[:, 1:].argmax(-1) == (fr[:, :-1] + 1) % 1024).all()
  assert torch.allclose(lg[:, 0], m.logp.expand(2, 128, 1024))
  assert torch.allclose(lg.exp().sum(-1), torch.ones(2, 20, 128), atol=1e-3)


def test_load_baseline_validates_name():
  with pytest.raises(ValueError):
    load_baseline("nope")


def test_load_baseline_aliases(tmp_path, monkeypatch, stats):
  """"copy" (nombre viejo de la tabla de transición) sigue cargando con un aviso; "congelado" es copy_mix."""
  from src.models import baselines
  monkeypatch.setattr(baselines, "STATS", tmp_path / "b.npz")
  baselines.save_stats(*stats, path=tmp_path / "b.npz")
  with pytest.warns(DeprecationWarning, match="transicion"):
    old = load_baseline("copy")
  assert isinstance(old, Transition) and old.name == "transicion"
  assert isinstance(load_baseline("transicion"), Transition)
  fro = load_baseline("congelado")
  assert isinstance(fro, CopyMix) and fro.name == "congelado"
  ctx = torch.from_numpy(synth(2, 4).astype(np.int64))
  assert torch.equal(fro.next_frame(ctx, temperature=0), ctx[:, -1])  # temperatura 0: repite el último frame
