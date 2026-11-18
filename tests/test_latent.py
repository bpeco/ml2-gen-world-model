"""Tests de LeWM (LeWorldModel, desde píxeles, fiel al código oficial) y de su decoder de visualización."""
import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from src import recipes as R
from src import schedules as S
from src import store
from src import train_decoder
from src.models import load_checkpoint, load_model
from src.models.latent import ema_momentum, ema_update
from src.models.lewm import FIDELIDAD, ARPredictor, ConditionalBlock, LeWM, SIGReg, cem
from src.models.lewm_decoder import DecoderConfig, LeWMDecoder, load_decoder, save_decoder
from src.train import main
from src.windows import Windows

TINY = dict(dim=32, enc_layers=1, enc_heads=2, enc_mlp=64, proj_hidden=64, pred_layers=2, pred_heads=4,
            pred_dim_head=8, pred_mlp=64, pred_dropout=0.0, num_proj=64)
DEC = dict(dec_dim=32, layers=1, heads=2, n_kv=2)


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
  monkeypatch.setattr(train_decoder, "render", fake_render)


def tiny(**kw):
  torch.manual_seed(0)
  return LeWM(LeWM.Config(**{**TINY, **kw}))


def wake(m):
  """Saca a AdaLN-zero de la identidad (si no, la acción no influye al inicializar)."""
  g = torch.Generator().manual_seed(3)
  for n, p in m.named_parameters():
    if "adaLN_modulation" in n:
      p.data = torch.randn(p.shape, generator=g) * 0.2
  return m


def frames(B=2, F=4, seed=1):
  return torch.randint(0, 1024, (B, F, 128), generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------- tamaños y estructura oficial

def test_param_counts_official_size():
  """El tamaño oficial: predictor ≈ 10-11M (ARPredictor de module.py), y el total que se informa."""
  m = LeWM(LeWM.Config())
  n = lambda mod: sum(p.numel() for p in mod.parameters())  # noqa: E731
  total = n(m)
  print(f"LeWM base: total {total / 1e6:.2f}M, encoder {n(m.encoder) / 1e6:.2f}M, projector {n(m.projector) / 1e6:.2f}M, "
        f"predictor {n(m.predictor) / 1e6:.2f}M, pred_proj {n(m.pred_proj) / 1e6:.2f}M, acción {n(m.action_encoder) / 1e6:.2f}M")
  assert 10e6 < n(m.predictor) < 11e6
  assert 5e6 < n(m.encoder) < 6e6                      # ViT-tiny
  assert n(m.projector) == n(m.pred_proj) == 192 * 2048 + 2048 + 2 * 2048 + 2048 * 192 + 192
  assert total == m.num_params() and 17e6 < total < 19e6
  assert m.cfg.history == 3 and m.cfg.num_preds == 1 and m.cfg.frameskip == 5 and m.cfg.n_pose == 30


def test_predictor_matches_module_py():
  p = ARPredictor(num_frames=3, depth=6, heads=16, mlp_dim=2048, dim=192, dim_head=64, dropout=0.1)
  blk = p.transformer.layers[0]
  assert len(p.transformer.layers) == 6 and p.pos_embedding.shape == (1, 3, 192)
  assert blk.attn.to_qkv.weight.shape == (3 * 1024, 192) and blk.attn.to_qkv.bias is None      # QKV sin bias
  assert blk.mlp.net[1].out_features == 2048 and blk.attn.dropout == 0.1 and blk.mlp.net[3].p == 0.1
  assert blk.adaLN_modulation[-1].weight.shape == (6 * 192, 192)                                 # 6 vectores (con gate)
  assert not blk.norm1.elementwise_affine and p.dropout.p == 0.0 and isinstance(p.transformer.norm, nn.LayerNorm)


def test_projector_and_embedder_match_official():
  m = LeWM(LeWM.Config())
  lin, bn, act, lout = m.projector.net
  assert (lin.in_features, lin.out_features, lout.out_features) == (192, 2048, 192)
  assert isinstance(bn, nn.BatchNorm1d) and isinstance(act, nn.GELU)
  assert isinstance(m.pred_proj.net[1], nn.BatchNorm1d) and m.pred_proj is not m.projector
  ae = m.action_encoder
  assert ae.patch_embed.kernel_size == (1,) and ae.patch_embed.in_channels == 30 and ae.embed[0].out_features == 4 * 192
  assert isinstance(ae.embed[1], nn.SiLU)
  assert m.action_embed(torch.randn(2, 4, 30), 2, 4, "cpu").shape == (2, 4, 192)


# ---------------------------------------------------------------- SIGReg

def test_sigreg_gaussian_vs_collapsed():
  torch.manual_seed(0)
  s = SIGReg(num_proj=256)
  gauss = s(torch.randn(2, 128, 16)).item()
  narrow = s(0.1 * torch.randn(2, 128, 16)).item()
  collapsed = s(torch.ones(2, 128, 16) * 0.5).item()
  assert gauss < 2 and 20 * gauss < narrow < collapsed        # el estadístico va multiplicado por B: gaussianas ~ 1
  assert s(torch.randn(2, 4000, 16)).item() / 4000 < 1e-3    # y por muestra tiende a 0 (≈ 0 para embeddings gaussianos)
  assert collapsed / 128 > 0.1


def test_sigreg_is_applied_per_time_step():
  """(T, B, D): cada paso de tiempo se mide a través del batch; el resultado es el promedio sobre el tiempo."""
  torch.manual_seed(1)
  g, c = torch.randn(1, 64, 8), torch.ones(1, 64, 8) * 0.5
  s = SIGReg(num_proj=128)
  both = torch.cat([g, c])

  def at(x):
    torch.manual_seed(7)  # mismas proyecciones
    return s(x).item()

  assert at(both) == pytest.approx((at(g) + at(c)) / 2, rel=1e-4)
  mixed = torch.cat([g, c], 1)                          # el mismo contenido mezclado en un solo paso: otro valor
  assert abs(at(mixed) - at(both)) > 1e-3


def test_sigreg_resamples_projections_and_scales_with_batch():
  s = SIGReg(num_proj=64)
  x = torch.randn(3, 32, 8)
  assert s(x).item() != s(x).item()                      # proyecciones nuevas en cada forward
  assert not any(p.requires_grad for p in s.parameters()) and s.t.shape == (17,) and s.t[-1] == 3
  torch.manual_seed(0)
  one = s(torch.ones(2, 16, 8)).item()                   # colapso: err no depende de B; la estadística crece con B
  torch.manual_seed(0)
  two = s(torch.ones(2, 32, 8)).item()
  assert two == pytest.approx(2 * one, rel=1e-4)


# ---------------------------------------------------------------- predictor

def test_no_future_leak_in_predictor():
  m = wake(tiny()).eval()
  z, a = torch.randn(1, 3, 32), torch.randn(1, 3, 32)
  z2, a2 = z.clone(), a.clone()
  z2[0, 2] = torch.randn(32)
  a2[0, 2] = torch.randn(32)
  with torch.no_grad():
    base, dz, da = m.predict(z, a), m.predict(z2, a), m.predict(z, a2)
  assert (base[:, :2] - dz[:, :2]).abs().max() == 0 and (base[:, 2] - dz[:, 2]).abs().max() > 1e-5
  assert (base[:, :2] - da[:, :2]).abs().max() == 0 and (base[:, 2] - da[:, 2]).abs().max() > 1e-5
  z3 = z.clone()
  z3[0, 0] = torch.randn(32)                              # el pasado sí influye en el presente
  with torch.no_grad():
    assert (m.predict(z3, a)[:, 1:] - base[:, 1:]).abs().max() > 1e-6


def test_adaln_zero_init_is_identity():
  blk = ConditionalBlock(32, 4, 8, 64, 0.0).eval()
  x, c = torch.randn(2, 3, 32), torch.randn(2, 3, 32)
  assert torch.equal(blk(x, c), x)                        # gate 0: el bloque no hace nada al inicializar
  m = tiny().eval()
  z = torch.randn(2, 3, 32)
  with torch.no_grad():                                   # y el predictor ignora la acción
    a = m.predict(z, torch.randn(2, 3, 32))
    b = m.predict(z, torch.randn(2, 3, 32))
  assert torch.equal(a, b)
  w = wake(tiny()).eval()                                 # con AdaLN despierta, la acción sí cambia la predicción
  with torch.no_grad():
    assert (w.predict(z, torch.randn(2, 3, 32)) - w.predict(z, torch.randn(2, 3, 32))).abs().max() > 1e-4


def test_dropout_only_in_training():
  m = wake(tiny(pred_dropout=0.5))
  z, a = torch.randn(2, 3, 32), torch.randn(2, 3, 32)
  m.train()
  assert (m.predict(z, a) - m.predict(z, a)).abs().max() > 0
  m.eval()
  assert torch.equal(m.predict(z, a), m.predict(z, a))


# ---------------------------------------------------------------- pérdida y acciones

def test_train_loss_is_mse_plus_lambda_sigreg():
  m = wake(tiny()).train()
  x, p = frames(4, 4), torch.randn(4, 4, 30)
  torch.manual_seed(5)
  loss, met = m.train_loss(x, poses=p)
  assert {"pred_mse", "sigreg", "z_std"} <= set(met) and "ce" not in met
  assert loss.item() == pytest.approx(met["pred_mse"].item() + 0.09 * met["sigreg"].item(), rel=1e-5)
  torch.manual_seed(5)  # recomputa a mano: ctx = z[:, :3], objetivo = z[:, 1:], SIGReg sobre z como (T, B, D)
  z = m.encode(x)
  pred = m.predict(z[:, :3], m.action_embed(p, 4, 4, "cpu")[:, :3])
  ref = (pred - z[:, 1:]).pow(2).mean() + 0.09 * m.sigreg(z.transpose(0, 1))
  assert loss.item() == pytest.approx(ref.item(), rel=1e-4)


def test_no_stop_grad_and_no_ema():
  """El objetivo recibe gradiente (sin stop-grad) y no hay encoder target."""
  m = wake(tiny()).train()
  m.cfg.lam = 0.0
  loss, _ = m.train_loss(frames(4, 4), poses=torch.randn(4, 4, 30))
  loss.backward()
  for mod in (m.encoder, m.projector, m.predictor, m.pred_proj, m.action_encoder):
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in mod.parameters() if p.requires_grad)
  assert not hasattr(m, "tgt") and not hasattr(m, "start") and not hasattr(m, "null_act") and not hasattr(m, "decoder")
  # con el objetivo detach el gradiente del encoder sería distinto
  g_full = m.encoder.patch_embed.weight.grad.clone()
  m.zero_grad()
  z = m.encode(frames(4, 4))
  pred = m.predict(z[:, :3], m.action_embed(torch.randn(4, 4, 30), 4, 4, "cpu")[:, :3])
  (pred - z[:, 1:].detach()).pow(2).mean().backward()
  assert not torch.allclose(g_full, m.encoder.patch_embed.weight.grad)


def test_train_loss_uses_first_four_frames_of_longer_windows():
  """train.py monitorea con ventanas de 20: el modelo usa solo history + num_preds."""
  m = wake(tiny()).eval()
  x = frames(2, 6)
  torch.manual_seed(1)
  a = m.train_loss(x)[0]
  torch.manual_seed(1)
  b = m.train_loss(x[:, :4])[0]
  assert a.item() == b.item()


def test_poses_nan_become_zero_action_and_none_is_zero():
  m = tiny().eval()
  m.set_pose_stats(torch.ones(30) * 2, torch.ones(30) * 3)
  p = torch.randn(2, 3, 30)
  p[0, 1, 4] = float("nan")
  a = m.action_embed(p, 2, 3, "cpu")
  assert torch.isfinite(a).all()
  q = p.clone()
  q[0, 1, 4] = 2.0                                        # NaN -> 0 tras normalizar = el valor de la media
  assert torch.allclose(a, m.action_embed(q, 2, 3, "cpu"))
  zeros = m.action_embed(torch.full((2, 3, 30), 2.0), 2, 3, "cpu")
  assert torch.allclose(m.action_embed(None, 2, 3, "cpu"), zeros, atol=1e-6)
  assert torch.allclose(m.action_embed(torch.full((2, 3, 6), 2.0), 2, 3, "cpu"), zeros, atol=1e-6)  # (.., 6) se repite k veces


def test_windows_frameskip_action_stacking():
  tok = np.random.default_rng(0).integers(0, 1024, (12, 60, 128)).astype(np.int16)
  pos = np.random.default_rng(1).normal(size=(12, 60, 6)).astype(np.float32)
  w = Windows(tok, holdout=2, poses=pos)
  x, p = w.sample(3, 4, 4, with_poses=True, frameskip=5)
  assert x.shape == (4, 4, 128) and p.shape == (4, 4, 30)
  for i in range(4):  # la acción del frame j son las 5 poses crudas desde ese frame
    s, t = next((s, t) for s in range(10) for t in range(41) if np.array_equal(tok[s, t:t + 16:5], x[i].numpy()))
    assert np.allclose(p[i].numpy().reshape(4, 5, 6), pos[s, t:t + 20].reshape(4, 5, 6))
  mean, std = w.pose_stats(5)
  assert mean.shape == std.shape == (30,)
  m = tiny()
  m.set_pose_stats(mean, std)
  assert m.pose_std.ne(1).any() and m.train_loss(x, poses=p)[0].isfinite()


# ---------------------------------------------------------------- features, CEM

def test_features_and_latent_error_shapes():
  m = wake(tiny()).eval()
  x, p = frames(2, 4), torch.randn(2, 4, 30)
  z, zh = m.features(x, p)
  assert z.shape == (2, 4, 32) and zh.shape == (2, 3, 32)
  assert m.latent_error(x, p).shape == (2, 3)
  zl, zhl = m.features(frames(2, 6), None)               # ventanas largas: contexto deslizante de 3 frames
  assert zl.shape == (2, 6, 32) and zhl.shape == (2, 5, 32)
  with torch.no_grad():
    assert torch.allclose(zhl[:, 3], m.predict(zl[:, 1:4], m.action_embed(None, 2, 3, "cpu"))[:, -1], atol=1e-5)


def test_cem_recovers_minimum():
  target = torch.tensor([[0.5, -1.0, 2.0], [1.5, 0.0, -0.5]])
  plan = cem(lambda c: (c - target[:, None]).square().sum(-1).sum(-1), 2, (1, 3), n_samples=200, n_elites=20, iters=20,
             generator=torch.Generator().manual_seed(0))
  assert torch.allclose(plan[:, 0], target, atol=0.05)


def test_cem_defaults_are_the_official_ones():
  import inspect
  d = {k: v.default for k, v in inspect.signature(cem).parameters.items()}
  pd = {k: v.default for k, v in inspect.signature(LeWM.plan_cem).parameters.items()}
  assert (d["n_samples"], d["n_elites"], d["iters"], d["init_std"]) == (300, 30, 10, 1.0) == tuple(pd[k] for k in ("n_samples", "n_elites", "iters", "init_std"))
  assert pd["horizon"] == 5


def test_plan_cem_rollout_and_cost():
  m = wake(tiny()).eval()
  x, p = frames(2, 4), torch.randn(2, 4, 30)
  plan = m.plan_cem(x[:, :3], x[:, 3], horizon=2, ctx_poses=p[:, :3], n_samples=16, n_elites=4, iters=2,
                    generator=torch.Generator().manual_seed(0))
  assert plan.shape == (2, 2, 30) and torch.isfinite(plan).all()
  # rollout_latent = pasos sucesivos del predictor con la historia deslizante de 3
  z = m.encode(x[:, :3])
  a_ctx, fut = m.action_embed(p[:, :2], 2, 2, "cpu"), m.action_embed(torch.randn(2, 2, 30), 2, 2, "cpu")
  out = m.rollout_latent(z, a_ctx, fut)
  a = torch.cat([a_ctx, fut], 1)
  z1 = m._next(z, a[:, :3])
  z2 = m._next(torch.cat([z, z1[:, None]], 1)[:, 1:], a[:, 1:4])
  assert out.shape == (2, 2, 32) and torch.allclose(out[:, 0], z1) and torch.allclose(out[:, 1], z2, atol=1e-6)


def test_plan_cem_cost_prefers_the_goal():
  """Con un modelo que predice z_{t+1} = z_t + b·a (acción lineal), el CEM encuentra la acción que llega a la meta."""
  m = wake(tiny()).eval()
  x = frames(1, 3)
  goal_tok = frames(1, 1, seed=4)[:, 0]
  zg = m.encode(goal_tok[:, None])[:, 0]
  N = 64
  cands = torch.randn(N, 1, 1, 30, generator=torch.Generator().manual_seed(2))
  z0 = m.encode(x)
  ac = m.action_embed(None, 1, 2, "cpu")

  def cost(c):
    zh = m.rollout_latent(z0.expand(len(c), -1, -1), ac.expand(len(c), -1, -1), m.action_encoder(c.reshape(len(c), 1, 30)))
    return (zh[:, -1] - zg).square().sum(-1)[:, None]

  best = cands[cost(cands).argmin()]
  plan = cem(lambda c: cost(c.reshape(-1, 1, 30)).reshape(-1, 1), 1, (1, 30), n_samples=64, n_elites=8, iters=6,
             generator=torch.Generator().manual_seed(0))
  assert cost(plan.reshape(1, 1, 30)).item() < cost(torch.zeros(1, 1, 30)).item() * 1.01 or cost(plan.reshape(1, 1, 30)).item() < cost(best[None, :, 0]).item()


def test_generate_raises_no_tokens():
  m = tiny()
  with pytest.raises(NotImplementedError):
    m.rollout(frames(1, 3), 2)
  assert not m.has_likelihood and m.uses_poses and m.ctx_frames == 3


# ---------------------------------------------------------------- decoder de visualización

def test_decoder_shapes_decode_and_roundtrip(tmp_path):
  dec = LeWMDecoder(DecoderConfig(dim_in=32, **DEC)).eval()
  z = torch.randn(2, 3, 32)
  img = dec(z)
  assert img.shape == (2, 3, 128, 256, 3) and dec.decode(z).dtype == torch.uint8
  save_decoder(tmp_path / "decoder.pt", dec, lewm_run="x")
  back = load_decoder(tmp_path / "decoder.pt")
  assert torch.equal(back(z), img)
  assert len(dec.queries) == 8 * 16                      # una query aprendida por parche


def test_decoder_training_leaves_lewm_without_grads():
  m = wake(tiny()).eval()
  for p in m.parameters():
    p.requires_grad_(False)
  dec = LeWMDecoder(DecoderConfig(dim_in=32, **DEC)).train()
  loss = train_decoder.decoder_loss(m, dec, frames(2, 1)[:, 0])
  loss.backward()
  assert all(p.grad is None for p in m.parameters()) and all(p.grad is not None for p in dec.parameters())
  # aun sin congelar, el embedding entra al decoder con detach: el LeWM tampoco recibe gradiente
  m2 = wake(tiny()).eval()
  train_decoder.decoder_loss(m2, LeWMDecoder(DecoderConfig(dim_in=32, **DEC)), frames(2, 1)[:, 0]).backward()
  assert all(p.grad is None for p in m2.parameters())


def test_train_decoder_end_to_end(tmp_path, monkeypatch):
  monkeypatch.setenv("ML2_DEVICE", "cpu")
  tok = tmp_path / "tok.npy"
  np.save(tok, np.random.default_rng(0).integers(0, 1024, (60, 30, 128)).astype(np.int16))
  m = tiny()
  torch.save({"kind": "lewm", "config": m.cfg.__dict__, "model": m.state_dict(), "name": "lw"}, tmp_path / "lw.pt")
  res = train_decoder.main(["--run", str(tmp_path / "lw.pt"), "--data", str(tok), "--config", json.dumps(DEC),
                            "--steps", "3", "--bs", "2", "--warmup", "1", "--eval-every", "2", "--log-every", "1",
                            "--ckpt-dir", str(tmp_path / "ck"), "--results-dir", str(tmp_path / "res")])
  assert res["steps"] == 3 and np.isfinite(res["holdout_mse"])
  assert (tmp_path / "ck" / "lw" / "decoder.pt").exists() and (tmp_path / "res" / "train_decoder_lw.json").exists()
  assert load_decoder(tmp_path / "ck" / "lw" / "decoder.pt").cfg.dim_in == 32


# ---------------------------------------------------------------- receta y fidelidad

def test_recipe_is_the_official_one():
  r = LeWM.recipe()
  assert r.optimizer == "adamw" and tuple(r.betas) == (0.9, 0.999) and r.wd_ini == 1e-3 and r.wd_on == "todos"
  assert r.amp == "bf16" and r.grad_clip == 1.0 and (r.batch, r.accum, r.frames, r.frameskip) == (128, 1, 4, 5)
  assert r.presupuesto == 780_000 and r.ema_segs is None and S.peak(r.lr_segs) == 5e-5
  total = -(-780_000 // (128 * 4))                       # 1524 pasos
  warm = S.lengths(r.lr_segs, total)[0][1]
  assert warm == round(0.01 * total) == 15               # 1 % de los pasos, sin el piso de 100
  assert r.lr(0, total) < r.lr(warm - 1, total) == pytest.approx(5e-5) and r.lr(total - 1, total) < 1e-9
  assert r.lr(warm + (total - warm) // 2, total) == pytest.approx(2.5e-5, rel=0.02)   # coseno a 0
  assert all(c in R.CATEGORIAS for _, _, c in r.forzado) and LeWM.recipe("xs").hash() == r.hash()


def test_sizes_base_and_xs():
  base, xs = LeWM.Config.sized("base"), LeWM.Config.sized("xs")
  assert base == LeWM.Config() and xs.dim == 96 and xs.pred_layers == 4
  n = lambda c: sum(p.numel() for p in LeWM(c).parameters())  # noqa: E731
  assert n(xs) < n(base) / 4


def test_fidelidad_table():
  assert all({"campo", "fuente_valor", "aquí", "categoría", "fuente"} == set(r) for r in FIDELIDAD)
  assert {r["categoría"] for r in FIDELIDAD} == {"fiel", "datos", "presupuesto", "propio"}
  assert all(str(v).strip() for r in FIDELIDAD for v in r.values())
  assert R.fidelidad("lewm") is FIDELIDAD and R.fidelidad_md("lewm").count("\n") == len(FIDELIDAD) + 1
  text = " ".join(r["campo"] for r in FIDELIDAD)
  for k in ("Encoder", "Projector", "Predictor", "Embedder", "Frameskip", "Loss", "SIGReg", "Optimizador", "CEM", "Decoder"):
    assert k in text


# ---------------------------------------------------------------- GPU-residente, entrenar / reanudar / registrar

def test_no_host_roundtrips(monkeypatch):
  """train_loss (+ backward), el paso del predictor y un paso del stream no sincronizan con el host."""
  from src.infer.stream import make_stream
  m = wake(tiny()).train()
  x, p = frames(2, 4), torch.randn(2, 4, 30)
  s = make_stream(wake(tiny()), graphs=False)
  img = fake_render(frames(1, 1)[:, 0])
  pose = torch.randn(1, 6)

  def boom(*a, **k):
    raise AssertionError("viaje GPU->CPU")
  for n in ("item", "cpu", "tolist", "numpy", "__bool__", "__int__", "__float__"):
    monkeypatch.setattr(torch.Tensor, n, boom)
  loss, _ = m.train_loss(x, poses=p)
  loss.backward()
  m.eval()
  m.predict(torch.randn(2, 3, 32), torch.randn(2, 3, 32))
  for _ in range(6):                                     # observe (1 de cada 5 codifica) + predict + step
    s.observe_pixels(img, pose)
  s.predict()
  s.predict(k=2, pose=pose)
  s.step()


def test_sigreg_projections_generated_on_the_device(monkeypatch):
  seen = []
  orig = torch.randn

  def spy(*a, **k):
    seen.append(k.get("device"))
    return orig(*a, **k)
  monkeypatch.setattr(torch, "randn", spy)
  SIGReg(num_proj=8)(torch.randn(2, 4, 8))
  assert seen[-1] is not None and torch.device(seen[-1]) == torch.device("cpu")


def test_ema_helpers():
  """ema_momentum / ema_update: piezas genéricas de latent.py (las reutilizará V-JEPA 2)."""
  assert ema_momentum(0, 10, 0.996, 1.0) == pytest.approx(0.996) and ema_momentum(10, 10, 0.996, 1.0) == 1.0
  tgt, onl = nn.Linear(3, 3), nn.Linear(3, 3)
  before = [p.clone() for p in tgt.parameters()]
  ema_update(tgt, onl, 1.0)
  assert all(torch.equal(a, b) for a, b in zip(before, tgt.parameters()))
  ema_update(tgt, onl, 0.0)
  assert all(torch.equal(a, b) for a, b in zip(onl.parameters(), tgt.parameters()))


def test_train_resume_register_and_load(tmp_path, monkeypatch):
  cfg = {k: v for k, v in TINY.items() if k != "pred_dropout"}
  monkeypatch.setenv("ML2_DEVICE", "cpu")
  tok, pos = tmp_path / "tok.npy", tmp_path / "pos.npy"
  np.save(tok, np.random.default_rng(0).integers(0, 1024, (60, 30, 128)).astype(np.int16))
  p = np.random.default_rng(1).normal(size=(60, 30, 6)).astype(np.float32)
  p[:, -1] = np.nan
  np.save(pos, p)

  def args(run, extra=()):
    return ["--kind", "lewm", "--config", json.dumps(cfg), "--run", run, "--tokens", str(1032 * 6), "--bs", "2",
            "--warmup", "2", "--lr", "3e-3", "--eval-every", "2", "--log-every", "1",
            "--data", str(tok), "--poses", str(pos), "--ckpt-dir", str(tmp_path / "checkpoints"),
            "--results-dir", str(tmp_path / "results"), *extra]

  full = main(args("full"))                              # frames 4 y frameskip 5 salen de la receta
  main(args("split", ["--max-steps", "3"]))
  split = main(args("split"))                            # reanuda: mismos batches, dropout y proyecciones de SIGReg
  assert full["steps"] == split["steps"] == 6 and full["receta"]["frameskip"] == 5 and full["receta"]["frames"] == 4
  a = load_checkpoint(tmp_path / "checkpoints" / "full" / "final.pt")
  b = load_checkpoint(tmp_path / "checkpoints" / "split" / "final.pt")
  assert a.cfg.n_pose == 30
  for k, v in a.state_dict().items():
    assert torch.allclose(v.float(), b.state_dict()[k].float(), atol=1e-5), k
  assert a.pose_std.ne(1).any() and a.pose_std.shape == (30,)  # la normalización de poses viaja en el checkpoint
  assert full["params"] == a.num_params()

  monkeypatch.setattr(store, "CKPT_ROOT", tmp_path / "checkpoints")
  monkeypatch.setattr(store, "RESULTS", tmp_path / "results")
  monkeypatch.setattr(store, "MANIFEST", tmp_path / "results" / "checkpoints.json")
  e = store.register("full")
  assert e["kind"] == "lewm" and e["params"] == full["params"]
  c = load_model("full")
  x = frames(1, 4)
  with torch.no_grad():
    assert torch.equal(a.encode(x), c.encode(x))
