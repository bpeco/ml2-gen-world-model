"""Modelo plantilla (src/models/_plantilla.py): lo propio de su arquitectura. El contrato común está en test_contrato.py."""
import pytest
import torch

from src.models import KINDS, build, load_model
from src.models._plantilla import FIDELIDAD, Plantilla


def tiny(**kw):
  torch.manual_seed(0)
  return Plantilla(Plantilla.Config(dim=32, **kw)).eval()


def frames(B=2, F=5, seed=1):
  return torch.randint(0, 1024, (B, F, 128), generator=torch.Generator().manual_seed(seed))


def test_registrada_y_propia():
  assert "plantilla" in KINDS and isinstance(build("plantilla", {"dim": 16}), Plantilla)
  assert isinstance(load_model("random:plantilla"), Plantilla)
  assert {f["categoría"] for f in FIDELIDAD} <= {"propio", "datos", "presupuesto"}
  assert any(f["categoría"] == "propio" for f in FIDELIDAD)


def test_forma_y_loss_sobre_frames_con_contexto():
  m = tiny()
  x = frames()
  assert m.frame_logits(x).shape == (2, 5, 128, 1024)
  loss, met = m.train_loss(x)
  assert torch.allclose(loss, met["ce"]) and loss.item() > 6   # ~ ln(1024) sin entrenar
  with pytest.raises(ValueError):
    m.train_loss(x[:, :1])
  with pytest.raises(ValueError):
    m.train_loss(x, teacher=object())


def test_solo_ve_los_ultimos_lags_frames():
  m = tiny(lags=2)
  x = frames()
  x2 = x.clone()
  x2[:, 0] = (x2[:, 0] + 1) % 1024    # el frame 0 sale de la ventana de los frames >= 3 (lags = 2, más el resumen del anterior)
  a, b = m.frame_logits(x), m.frame_logits(x2)
  assert not torch.equal(a[:, 1], b[:, 1]) and torch.allclose(a[:, 3:], b[:, 3:], atol=1e-6)


def test_rollout_forma_y_determinismo_a_T0():
  m = tiny()
  ctx = frames(2, 3)
  r = m.rollout(ctx, 4, temperature=0)
  assert r.shape == (2, 4, 128) and torch.equal(r, m.rollout(ctx, 4, temperature=0))
