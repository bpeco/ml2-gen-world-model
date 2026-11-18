import json

import numpy as np
import pytest
import torch

import src.models.convlstm_px as cp
from src.infer.stream import ConvLSTMPxStream, make_stream
from src.models import build, load_model
from src.models.convlstm_px import FIDELIDAD, ConvLSTMCell, ConvLSTMPx

H, W = 16, 32


def tiny(**kw):
  torch.manual_seed(0)
  cfg = dict(height=H, width=W, patch=4, hidden=(8, 6), kernel=3, n_in=3, n_out=2)
  cfg.update(kw)
  m = ConvLSTMPx(ConvLSTMPx.Config(**cfg)).eval()
  for p in m.parameters():
    p.data.add_(0.3 * torch.randn_like(p))
  return m


def rand_px(B, F, seed=1):
  g = torch.Generator().manual_seed(seed)
  return torch.randint(0, 256, (B, F, H, W, 3), generator=g, dtype=torch.uint8)


def np_conv_same(x, w):
  """x (Cin, h, w), w (Cout, Cin, k, k) -> (Cout, h, w), correlación con padding cero (como nn.Conv2d)."""
  Co, Ci, k, _ = w.shape
  p = k // 2
  xp = np.pad(x, ((0, 0), (p, p), (p, p)))
  out = np.zeros((Co, x.shape[1], x.shape[2]))
  for i in range(x.shape[1]):
    for j in range(x.shape[2]):
      out[:, i, j] = np.einsum("ockl,ckl->o", w, xp[:, i:i + k, j:j + k])
  return out


def test_cell_equals_numpy_reference():
  torch.manual_seed(0)
  cell = ConvLSTMCell(3, 4, 3).double()
  for p in (cell.w_ci, cell.w_cf, cell.w_co):
    p.data.normal_()
  x, h, c = torch.randn(1, 3, 5, 6).double(), torch.randn(1, 4, 5, 6).double(), torch.randn(1, 4, 5, 6).double()
  h2, c2 = cell(x, (h, c))
  Wt, b = cell.conv.weight.detach().numpy(), cell.conv.bias.detach().numpy()
  Wx, Wh = Wt[:, :3], Wt[:, 3:]  # W_x*X + W_h*H por separado, como en la Eq. 3
  z = np_conv_same(x[0].numpy(), Wx) + np_conv_same(h[0].numpy(), Wh) + b[:, None, None]
  sig = lambda v: 1 / (1 + np.exp(-v))
  pc = lambda w: w.detach().numpy()[:, None, None]
  c0 = c[0].numpy()
  i = sig(z[0:4] + pc(cell.w_ci) * c0)
  f = sig(z[4:8] + pc(cell.w_cf) * c0)
  cn = f * c0 + i * np.tanh(z[8:12])
  o = sig(z[12:16] + pc(cell.w_co) * cn)
  hn = o * np.tanh(cn)
  assert np.allclose(c2[0].detach().numpy(), cn, atol=1e-10)
  assert np.allclose(h2[0].detach().numpy(), hn, atol=1e-10)


def test_cell_without_input_ignores_x():
  cell = ConvLSTMCell(0, 4, 3)
  h, c = torch.randn(2, 4, 5, 5), torch.randn(2, 4, 5, 5)
  a, _ = cell(None, (h, c))
  assert a.shape == h.shape and cell.conv.in_channels == 4


def test_param_count_paper():
  """Moving-MNIST de Shi (parches 16x16x16, 128/64/64, 5x5): 7 585 296 parámetros (Tab. 1)."""
  m = ConvLSTMPx(ConvLSTMPx.Config(in_ch=1, height=64, width=64, patch=4, hidden=(128, 64, 64)))
  assert m.num_params() == 7_585_296


def test_forecaster_starts_from_encoder_state(monkeypatch):
  m = tiny()
  seen = []
  orig = m.forecast
  monkeypatch.setattr(m, "forecast", lambda st, n: (seen.append([(h.clone(), c.clone()) for h, c in st]), orig(st, n))[1])
  win = m.to_float(rand_px(2, 3))
  m.probs(win, 2)
  x = torch.nn.functional.pixel_unshuffle(win.flatten(0, 1), 4).unflatten(0, (2, 3))
  ref = m.encode(x)
  for (h, c), (h2, c2) in zip(seen[0], ref):
    assert torch.equal(h, h2) and torch.equal(c, c2)
  # y el primer frame pronosticado cambia si cambia el estado del encoder
  other = m.to_float(rand_px(2, 3, seed=2))
  assert not torch.allclose(m.probs(win, 1), m.probs(other, 1))


def test_causal():
  m = tiny()
  win = m.to_float(rand_px(1, 3))
  x = torch.nn.functional.pixel_unshuffle(win.flatten(0, 1), 4).unflatten(0, (1, 3))
  s2 = m.encode(x[:, :2])
  win2 = win.clone()
  win2[:, 2] = torch.rand_like(win2[:, 2])  # cambia solo el último frame de entrada
  x2 = torch.nn.functional.pixel_unshuffle(win2.flatten(0, 1), 4).unflatten(0, (1, 3))
  s2b = m.encode(x2[:, :2])
  assert all(torch.equal(a[0], b[0]) for a, b in zip(s2, s2b))      # el estado tras 2 frames no ve el 3.º
  assert not torch.allclose(m.probs(win, 2), m.probs(win2, 2))
  # el pronóstico del paso 1 no depende de cuántos pasos se pidan
  assert torch.allclose(m.probs(win, 1)[:, 0], m.probs(win, 2)[:, 0], atol=1e-6)
  # y la pérdida no depende de nada posterior a su objetivo: cambiar el frame objetivo 2 no mueve el logit 1
  assert torch.equal(m.logits(win, 1)[:, 0], m.logits(win, 2)[:, 0])


def test_train_loss_bce_matches_manual():
  m = tiny()
  fr = rand_px(2, 5)
  loss, met = m.train_loss(fr)
  fl = m.to_float(fr)
  p = m.probs(fl[:, :3], 2)
  t = fl[:, 3:]
  ref = -(t * torch.log(p) + (1 - t) * torch.log(1 - p)).mean()
  assert torch.allclose(loss, ref, atol=1e-4)
  assert set(met) == {"bce", "mse"}
  with pytest.raises(ValueError):
    m.train_loss(rand_px(1, 3))


def test_train_from_tokens_uses_render(monkeypatch):
  m = tiny()
  px = rand_px(2, 5)
  monkeypatch.setattr(cp, "render", lambda tok: px)
  a, _ = m.train_loss(torch.zeros(2, 5, 128, dtype=torch.long))
  b, _ = m.train_loss(px)
  assert torch.equal(a, b)


def test_overfits_tiny_batch():
  torch.manual_seed(0)
  m = ConvLSTMPx(ConvLSTMPx.Config(height=H, width=W, hidden=(8, 8), kernel=3, n_in=3, n_out=2)).train()
  fr = (rand_px(2, 1) > 127).to(torch.uint8).repeat(1, 5, 1, 1, 1) * 255  # video binario estático
  opt = torch.optim.RMSprop(m.parameters(), lr=3e-3, alpha=0.9)
  first = None
  for _ in range(300):
    loss, _ = m.train_loss(fr)
    first = first if first is not None else loss.item()
    opt.zero_grad()
    loss.backward()
    opt.step()
  assert loss.item() < 0.5 * first


def test_generate_blocks_and_shapes():
  m = tiny()
  ctx = rand_px(2, 4)
  out = m.rollout(ctx, 5)
  assert out.shape == (2, 5, H, W, 3) and out.dtype == torch.uint8
  # 2.º bloque: la ventana de entrada son los n_in últimos frames (ventana + bloque 1)
  win = m.to_float(ctx[:, -3:])
  b1 = m.probs(win, 2)
  win2 = torch.cat([win, b1], 1)[:, -3:]
  b2 = m.probs(win2, 2)
  assert torch.equal(out[:, 2], m.to_uint8(b2[:, 0]))


def test_stream_equals_generate():
  m = tiny()
  ctx = rand_px(2, 6)
  s = make_stream(m)
  assert isinstance(s, ConvLSTMPxStream)
  for t in range(6):
    s.observe_pixels(ctx[:, t])
  nxt = s.predict()
  fut = [s.predict(k) for k in (1, 2)]
  ref = m.rollout(ctx, 7)
  assert torch.equal(nxt, ref[:, 0])
  assert torch.equal(fut[1], ref[:, 1]) or (fut[1].int() - ref[:, 1].int()).abs().max() <= 1
  got = torch.stack(list(s.imagine(7)), 1)
  assert torch.equal(got, ref)
  assert s.n_seen == 6 + 7


def test_stream_observe_tokens_and_partial_window(monkeypatch):
  m = tiny()
  px = rand_px(1, 3)
  monkeypatch.setattr("src.pixels.render", lambda tok: px[:, 0])
  s = make_stream(m)
  s.observe(torch.zeros(1, 128, dtype=torch.long))   # tokens -> render
  assert s.win.shape[1] == 1 and not s.full
  assert s.predict().shape == (1, H, W, 3)            # ventana incompleta: camino sin graphs


def test_no_host_roundtrips(monkeypatch):
  m = tiny()
  px = rand_px(2, 5)
  s = make_stream(m)
  for t in range(3):
    s.observe_pixels(px[:, t])

  def boom(*a, **k):
    raise AssertionError("viaje al host")
  for name in ("item", "cpu", "numpy", "tolist"):
    monkeypatch.setattr(torch.Tensor, name, boom)
  m.train()
  loss, met = m.train_loss(px)
  loss.backward()
  s.predict()
  s.step()
  s.step()
  s.step()  # cambia de bloque


def test_fidelidad_schema_and_recipe():
  assert FIDELIDAD and all({"campo", "fuente_valor", "aquí", "categoría", "fuente"} <= set(d) for d in FIDELIDAD)
  assert {d["categoría"] for d in FIDELIDAD} <= {"fiel", "datos", "presupuesto", "propio"}
  r = ConvLSTMPx.recipe()
  assert r.optimizer == "rmsprop" and r.alpha == 0.9 and r.lr(0, 100) == r.lr(99, 100) == 1e-3
  assert r.frames == 20 and r.presupuesto == 780_000 and r.amp is None
  from src.recipes import Recipe
  assert Recipe.from_dict(r.to_dict()).hash() == r.hash()
  m = build("convlstm_px")
  assert m.cfg.hidden == (128, 64, 64) and m.output == "pixels" and not m.has_likelihood
  assert load_model("random:convlstm_px").name == "random_convlstm_px"


# ---------------------------------------------------------------- rollouts y métricas de píxeles

def test_rollout_pixel_path_writes_and_reads(tmp_path, monkeypatch):
  import src.rollouts as R
  m = tiny(n_in=2)
  monkeypatch.setattr(R, "OUT", tmp_path)
  px = rand_px(3, 2)
  out = np.lib.format.open_memmap(tmp_path / "x.npy", mode="w+", dtype=np.uint8, shape=(3, 4, H, W, 3))
  out[:] = m.rollout(px, 4).numpy()
  out.flush()
  back = np.load(tmp_path / "x.npy", mmap_mode="r")
  from src.metrics_video import fake_frames, is_pixels
  assert is_pixels(back) and fake_frames(back[:2], lambda t: pytest.fail("no se decodifica")).shape == (2, 4, H, W, 3)
  assert not is_pixels(np.zeros((2, 4, 128), np.int16))


def test_rollouts_main_pixel_model(tmp_path, monkeypatch):
  """src.rollouts de punta a punta con un modelo de píxeles y datos falsos (sin decoder ni val_tokens)."""
  import src.rollouts as R
  m = tiny(n_in=2)
  m.name = "fake_px"
  monkeypatch.setattr(R, "OUT", tmp_path)
  monkeypatch.setattr(np, "load", lambda *a, **k: np.zeros((100, 1000, 128), np.int16))
  monkeypatch.setattr("src.models.load_model", lambda spec, dev: m)
  monkeypatch.setattr("src.models.convlstm_px.render", lambda tok: rand_px(tok.shape[0], tok.shape[1]))
  monkeypatch.setattr(R, "HORIZON", 4)
  R.main(["--model", "x", "--bs", "2", "--limit", "2"])
  arr = np.lib.format.open_memmap(tmp_path / "fake_px.npy", mode="r")
  assert arr.dtype == np.uint8 and arr.shape == (2, 4, H, W, 3)
  assert json.loads((tmp_path / "fake_px.json").read_text())["output"] == "pixels"


def test_metrics_stats_with_pixel_rollouts(tmp_path, monkeypatch):
  import src.metrics_video as MV
  from types import SimpleNamespace
  monkeypatch.setattr(MV, "RESULTS", tmp_path)
  monkeypatch.setattr(MV, "OUT", tmp_path)
  rng = np.random.default_rng(0)
  n = 12
  lpm = rng.random((n, 60))
  ff = {k: rng.normal(size=(n, 5)) for k in MV.WINDOWS}
  rf = {k: rng.normal(size=(n, 5)) for k in MV.WINDOWS}
  a = SimpleNamespace(name="px", boot=20, boot_fvd=5, fvd_model="i3d")
  res = MV.write_stats(a, lpm, ff, rf, "h", torch.device("cpu"), 0.0)
  assert (tmp_path / "video_px.json").exists() and set(res["fvd"]) == set(MV.WINDOWS)
