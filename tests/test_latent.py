"""Tests de los world models latentes: LeWM (LeWorldModel, desde píxeles) y las piezas comunes de latent.py."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from src import store
from src.models import load_checkpoint, load_model
from src.models.base import WorldModel, lm_loss
from src.models.latent import SIGReg, cem, ema_momentum, ema_update
from src.models.lewm import LeWM
from src.train import main
from src.windows import Windows, disjoint_holdout

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="sin CUDA")
FRAMES = 6  # capacidad del predictor de los modelos de juguete


def fake_render(tok):
  """Cámara de juguete (sin los pesos del decoder de comma): cada token pinta su parche de 16x16."""
  pal = (torch.arange(1024)[:, None] * torch.tensor([37, 91, 53]) % 256).to(torch.uint8).to(tok.device)
  img = pal[tok.long()].view(*tok.shape[:-1], 8, 16, 3)
  return img.repeat_interleave(16, -3).repeat_interleave(16, -2).contiguous()


@pytest.fixture(autouse=True)
def camera(monkeypatch):
  """LeWM ve píxeles: en los tests, los tokens se renderizan con la cámara de juguete."""
  import src.models.lewm as lewm
  monkeypatch.setattr(lewm, "render_tokens", fake_render)


def tiny(**kw):
  torch.manual_seed(0)
  cfg = dict(dim=32, enc_layers=1, enc_heads=2, proj_hidden=64, pred_dim=32, pred_layers=2, pred_heads=4,
             pred_mlp=64, pred_dropout=0.0, dec_dim=32, dec_layers=1, dec_heads=2, dec_kv=2, frames=FRAMES,
             num_proj=64)
  m = LeWM(LeWM.Config(**{**cfg, **kw}))
  m.ctx_frames = FRAMES - 1
  return m


def wake(m):
  """Saca a AdaLN-zero de la identidad (si no, la acción y el contexto no influyen al inicializar)."""
  g = torch.Generator().manual_seed(3)
  for n, p in m.named_parameters():
    if ".ada." in n:
      p.data = torch.randn(p.shape, generator=g) * 0.2
  return m


def frames(B=2, F=4, seed=1):
  return torch.randint(0, 1024, (B, F, 128), generator=torch.Generator().manual_seed(seed))


def test_sigreg_gaussian_vs_collapsed():
  torch.manual_seed(0)
  s = SIGReg(num_proj=256)
  gauss = s(torch.randn(4000, 16)).item()
  narrow = s(0.1 * torch.randn(4000, 16)).item()
  collapsed = s(torch.ones(4000, 16) * 0.5).item()
  assert gauss < 0.5 < narrow < collapsed


def test_no_future_leak():
  m = wake(tiny()).eval()
  x = frames(1, 4)
  p = torch.randn(1, 4, 6)
  y = x.clone()
  y[0, 2] = torch.randint(0, 1024, (128,))
  with torch.no_grad():
    a, b = m.frame_logits(x, p), m.frame_logits(y, p)
  assert a.shape == (1, 4, 128, 1024)
  assert (a[:, :3] - b[:, :3]).abs().max() == 0          # el frame 2 solo influye en la predicción del 3
  assert (a[:, 3] - b[:, 3]).abs().max() > 1e-5
  q = p.clone()
  q[0, 2] = torch.randn(6)                               # la acción del frame 2 no cambia los frames 0 y 1
  with torch.no_grad():
    c = m.frame_logits(x, q)
  assert (a[:, :2] - c[:, :2]).abs().max() == 0
  assert (a[:, 2] - c[:, 2]).abs().max() > 1e-6


def test_readout_does_not_train_encoder():
  """El gradiente del encoder es el de pred_mse + λ·SIGReg: la CE de la lectura (sobre ẑ y z detach) no lo toca."""
  m = tiny().train()
  m.cfg.p_null = 0.0
  x = frames()
  torch.manual_seed(5)  # SIGReg proyecta en direcciones al azar: misma semilla en los dos cálculos
  loss, met = m.train_loss(x)
  assert {"pred_mse", "sigreg", "ce", "ce_techo", "z_std"} <= set(met)
  loss.backward()
  g_total = [p.grad.clone() for p in m.encoder.parameters() if p.grad is not None]
  assert all(p.grad is not None for p in m.decoder.parameters())  # el decoder a tokens sí aprende de la CE
  m.zero_grad()
  torch.manual_seed(5)
  z, zh = m.predict_frames(x)
  ref = (zh.float() - z.float()).square().mean() + m.cfg.lam * m.sigreg(z.reshape(1, -1, z.shape[-1]))
  ref.backward()
  g_pred = [p.grad.clone() for p in m.encoder.parameters() if p.grad is not None]
  assert len(g_total) == len(g_pred) > 0
  for a, b in zip(g_total, g_pred):
    assert torch.allclose(a, b, atol=1e-6)


def test_ema_helpers():
  """ema_momentum / ema_update / make_target / after_step: piezas genéricas (las reutilizará V-JEPA 2)."""
  assert ema_momentum(0, 10, 0.996, 1.0) == pytest.approx(0.996) and ema_momentum(10, 10, 0.996, 1.0) == 1.0
  assert 0.996 < ema_momentum(5, 10, 0.996, 1.0) < 1.0
  tgt, onl = nn.Linear(3, 3), nn.Linear(3, 3)
  before = [p.clone() for p in tgt.parameters()]
  ema_update(tgt, onl, 1.0)  # momento 1: no cambia
  assert all(torch.equal(a, b) for a, b in zip(before, tgt.parameters()))
  ema_update(tgt, onl, 0.0)  # momento 0: target = online
  assert all(torch.equal(a, b) for a, b in zip(onl.parameters(), tgt.parameters()))

  m = tiny()
  m.tgt = m.make_target()
  assert all(not p.requires_grad for p in m.tgt.parameters())
  m.cfg = SimpleNamespace(**{**vars(m.cfg), "ema": (0.0, 0.0)})
  with torch.no_grad():
    for p in m.encoder.parameters():
      p.add_(1.0)
  m.after_step(1, 10)  # momento 0: target = online
  for a, b in zip(m.tgt.parameters(), m.encoder.parameters()):
    assert torch.equal(a, b)
  assert m.num_params() < sum(p.numel() for p in m.parameters())  # el target no cuenta


def test_adaln_zero_init():
  m = tiny().eval()
  z = torch.randn(2, 3, 32)
  with torch.no_grad():
    a = m.predict(z, m.actions_norm(torch.randn(2, 3, 6)))
    b = m.predict(z, m.actions_norm(torch.randn(2, 3, 6)))
  assert torch.equal(a, b)


def test_generate_cache_matches_generic():
  m = wake(tiny()).eval()
  ctx = frames(2, 3)
  fast = m.rollout(ctx, 4, temperature=0)
  slow = torch.stack(list(WorldModel.generate(m, ctx, 4, 0.0, None, None)), 1)
  assert fast.shape == (2, 4, 128) and torch.equal(fast, slow)
  g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
  assert torch.equal(m.rollout(ctx, 2, generator=g1), torch.stack(list(WorldModel.generate(m, ctx, 2, 1.0, None, g2)), 1))
  out = list(m.generate(ctx, 3, temperature=0, poses=torch.randn(2, 6, 6)))  # con acciones
  assert len(out) == 3 and out[0].shape == (2, 128)
  m.cfg.rollout = "latent"
  assert m.rollout(ctx, 3, temperature=0).shape == (2, 3, 128)


@needs_cuda
def test_cuda_graphs_match_eager():
  m = wake(tiny()).cuda().eval()
  ctx = frames(2, 5).cuda()  # contexto lleno: ahí se usan los graphs
  eager = m.rollout(ctx, 4, temperature=0)
  m.use_cuda_graphs(True)
  assert torch.equal(m.rollout(ctx, 4, temperature=0), eager)


def test_cem_recovers_minimum():
  target = torch.tensor([[0.5, -1.0, 2.0], [1.5, 0.0, -0.5]])
  plan = cem(lambda c: (c - target[:, None]).square().sum(-1).sum(-1), 2, (1, 3), n_samples=200, n_elites=20, iters=20,
             generator=torch.Generator().manual_seed(0))
  assert torch.allclose(plan[:, 0], target, atol=0.05)


def test_plan_and_features():
  m = wake(tiny()).eval()
  x, p = frames(2, 4), torch.randn(2, 4, 6)
  plan = m.plan_cem(x[:, :3], x[:, 3], horizon=2, ctx_poses=p[:, :3], n_samples=16, n_elites=4, iters=2,
                    generator=torch.Generator().manual_seed(0))
  assert plan.shape == (2, 2, 6) and torch.isfinite(plan).all()
  z, zh = m.features(x, p)
  assert z.shape[:2] == (2, 4) and zh.shape[:2] == (2, 4)
  assert m.latent_error(x, p).shape == (2, 4)


def test_poses_nan_and_dropout_use_null_action():
  m = tiny().eval()
  B, T = 2, 3
  p = torch.randn(B, T, 6)
  p[0, 1] = float("nan")
  a = m.actions(p, B, T, "cpu")
  assert torch.equal(a[0, 1], m.null_act) and not torch.equal(a[0, 0], m.null_act)
  assert torch.equal(m.actions(None, B, T, "cpu")[1, 2], m.null_act)
  m.train()
  m.cfg.p_null = 1.0
  assert torch.equal(m.actions(torch.randn(B, T, 6), B, T, "cpu"), m.null_act.expand(B, T, -1))


def test_windows_poses():
  tok = np.random.default_rng(0).integers(0, 1024, (12, 40, 128)).astype(np.int16)
  pos = np.random.default_rng(1).normal(size=(12, 40, 6)).astype(np.float32)
  pos[:, -1] = np.nan
  w = Windows(tok, holdout=2, poses=pos)
  x, p = w.sample(5, 4, 10, with_poses=True)
  assert torch.equal(x, w.sample(5, 4, 10)) and p.shape == (4, 10, 6)
  for i in range(4):  # cada ventana de poses sale del mismo segmento y frames que la de tokens
    s = next(s for s in range(10) for t in range(31) if np.array_equal(tok[s, t:t + 10], x[i].numpy()))
    t = next(t for t in range(31) if np.array_equal(tok[s, t:t + 10], x[i].numpy()))
    assert np.allclose(pos[s, t:t + 10], p[i].numpy(), equal_nan=True)
  mean, std = w.pose_stats()
  assert np.isfinite(mean).all() and np.isfinite(std).all()
  keys = [f"r{i // 2}_{i}" for i in range(10)] + ["r4_10", "nueva_11"]  # r4 comparte ruta con train
  assert disjoint_holdout(keys, holdout=2) == [11]


def test_train_resume_register_and_load(tmp_path, monkeypatch):
  kind = "lewm"  # capacidad de 20 frames: monitor_ce usa ventanas de 20
  cfg = dict(dim=32, enc_layers=1, enc_heads=2, proj_hidden=64, pred_dim=32, pred_layers=2, pred_heads=4,
             pred_mlp=64, dec_dim=32, dec_layers=1, dec_heads=2, dec_kv=2, num_proj=32)
  monkeypatch.setenv("ML2_DEVICE", "cpu")
  tok, pos = tmp_path / "tok.npy", tmp_path / "pos.npy"
  np.save(tok, np.random.default_rng(0).integers(0, 1024, (60, 30, 128)).astype(np.int16))
  p = np.random.default_rng(1).normal(size=(60, 30, 6)).astype(np.float32)
  p[:, -1] = np.nan
  np.save(pos, p)

  def args(run, extra=()):
    return ["--kind", kind, "--config", json.dumps(cfg), "--run", run, "--tokens", str(1032 * 6), "--bs", "2",
            "--frames", "4", "--warmup", "2", "--lr", "3e-3", "--eval-every", "2", "--log-every", "1",
            "--data", str(tok), "--poses", str(pos), "--ckpt-dir", str(tmp_path / "checkpoints"),
            "--results-dir", str(tmp_path / "results"), *extra]

  full = main(args("full"))
  main(args("split", ["--max-steps", "3"]))
  split = main(args("split"))  # reanuda: mismos batches, dropout de acción y proyecciones de SIGReg
  assert full["steps"] == split["steps"] == 6
  a = load_checkpoint(tmp_path / "checkpoints" / "full" / "final.pt")
  b = load_checkpoint(tmp_path / "checkpoints" / "split" / "final.pt")
  for k, v in a.state_dict().items():
    assert torch.allclose(v, b.state_dict()[k], atol=1e-5), k
  assert a.pose_std.ne(1).any()  # la normalización de poses viaja en el checkpoint
  assert full["params"] == a.num_params()

  # guardado para reutilizar: register -> load_model("<corrida>") por nombre
  monkeypatch.setattr(store, "CKPT_ROOT", tmp_path / "checkpoints")
  monkeypatch.setattr(store, "RESULTS", tmp_path / "results")
  monkeypatch.setattr(store, "MANIFEST", tmp_path / "results" / "checkpoints.json")
  e = store.register("full")
  assert e["kind"] == kind and e["params"] == full["params"]
  c = load_model("full")
  x = frames(1, 3)
  with torch.no_grad():
    assert torch.equal(a.frame_logits(x), c.frame_logits(x))


def test_lm_loss_mask_unaffected():
  """La CE de la lectura usa el lm_loss común (sanity: mismas convenciones que el resto)."""
  lg = torch.randn(2, 3, 128, 1024)
  y = torch.randint(0, 1024, (2, 3, 128))
  assert lm_loss(lg, y)[0].item() > 0
