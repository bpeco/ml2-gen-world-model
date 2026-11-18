"""Tests de la inferencia de punta a punta (src/infer): el estado por frame da lo mismo que generate.

En CPU, igualdad exacta de tokens contra model.generate con la misma semilla (temperatura 0 y 1). En CUDA,
el camino con graphs contra el mismo sin graphs. El codec necesita los pesos de data/raw (si no, se saltea).
"""
import pytest
import torch

from src.common import TPF
from src.infer.stream import ConvLSTMStream, LatentStream, MaskGITStream, TokenStream, make_stream
from src.models.convlstm import ConvLSTM
from src.models.jepa import JEPA
from src.models.lewm import LeWM
from src.models.maskgit import MaskGIT

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="sin CUDA")
FRAMES = 6


def frames(B, F, seed=1, dev="cpu"):
  g = torch.Generator().manual_seed(seed)
  return torch.randint(0, 1024, (B, F, 128), generator=g).to(dev)


def shake(m, scale=4.0, noise=0.0, seed=3):
  """Logits menos planos (y AdaLN-zero fuera de la identidad) para que el test discrimine."""
  g = torch.Generator().manual_seed(seed)
  for p in m.parameters():
    p.data.mul_(scale)
    if noise:
      p.data.add_(torch.randn(p.shape, generator=g) * noise)
  return m


def tiny(kind, dev="cpu", **kw):
  torch.manual_seed(0)
  if kind == "maskgit":
    m = shake(MaskGIT(MaskGIT.Config(n_layer=2, n_head=4, dim=64, block_size=FRAMES * TPF, **kw)))
  elif kind == "convlstm":
    m = shake(ConvLSTM(ConvLSTM.Config(emb=8, hidden=16, n_layers=2)), scale=3.0)
    m.ctx_frames = FRAMES - 1
  elif kind == "jepa":
    m = shake(JEPA(JEPA.Config(n_layer=3, n_head=2, dim=32, n_enc=1, block_size=FRAMES * TPF, num_proj=64,
                               sigreg_n=256, **kw)), scale=1.0, noise=0.05)
    m.ctx_frames = FRAMES - 1
  else:
    m = shake(LeWM(LeWM.Config(dim=32, enc_layers=1, enc_heads=2, proj_hidden=64, pred_dim=32, pred_layers=2,
                               pred_heads=4, pred_mlp=64, pred_dropout=0.0, dec_dim=32, dec_layers=1, dec_heads=2,
                               dec_kv=2, frames=FRAMES, num_proj=64, **kw)), scale=1.0, noise=0.05)
    m.ctx_frames = FRAMES - 1
  return m.to(dev).eval()


def via_stream(m, ctx, n, temperature, steps=None, seed=0, graphs=False, **kw):
  s = make_stream(m, temperature=temperature, steps=steps, seed=seed, graphs=graphs, **kw)
  for f in range(ctx.shape[1]):
    s.observe(ctx[:, f])
  return torch.stack(list(s.imagine(n)), 1), s


def via_generate(m, ctx, n, temperature, steps=None, seed=0):
  g = torch.Generator(ctx.device).manual_seed(seed)
  return m.rollout(ctx, n, temperature=temperature, steps=steps, generator=g)


CASES = [("maskgit", {}), ("convlstm", {}), ("jepa", {}), ("lewm", {}), ("lewm", {"rollout": "latent"})]


@pytest.mark.parametrize("kind,kw", CASES)
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_stream_equals_generate_cpu(kind, kw, temperature):
  m = tiny(kind, **kw)
  ctx = frames(2, FRAMES - 1)
  steps = 3 if kind == "maskgit" else None
  ref = via_generate(m, ctx, 4, temperature, steps)
  got, s = via_stream(m, ctx, 4, temperature, steps)
  assert got.shape == ref.shape == (2, 4, 128)
  assert torch.equal(got, ref)


def test_adapters():
  assert isinstance(make_stream(tiny("maskgit")), MaskGITStream)
  assert isinstance(make_stream(tiny("convlstm")), ConvLSTMStream)
  assert isinstance(make_stream(tiny("jepa")), LatentStream)
  from src.models.baselines import Unigram
  assert isinstance(make_stream(Unigram(torch.zeros(1024))), TokenStream)


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_maskgit_fused_frame_equals_next_frame(temperature):
  """El frame entero de MaskGITStream (lo que se captura en un graph) da lo mismo que next_frame."""
  m = tiny("maskgit")
  s = MaskGITStream(m, temperature=temperature, steps=4, seed=7, graphs=False)
  ctx = frames(2, FRAMES - 1)
  got = s._frame(ctx, *s._noise(2))
  ref = m.next_frame(ctx, temperature, 4, torch.Generator().manual_seed(7))
  assert torch.equal(got, ref)


@pytest.mark.parametrize("kind", ["maskgit", "convlstm", "lewm"])
def test_predict_k_does_not_change_state(kind):
  m = tiny(kind)
  ctx = frames(1, FRAMES - 1)
  s = make_stream(m, temperature=0.0, steps=2, graphs=False)
  for f in range(ctx.shape[1]):
    s.observe(ctx[:, f])
  far = s.predict(k=3)
  roll = torch.stack(list(s.imagine(3)), 1)
  assert torch.equal(far, roll[:, -1])          # el frame t+3 de predict es el tercero del rollout
  assert torch.equal(s.predict(), s.predict())  # predict no avanza el estado


def test_convlstm_reset_every_matches_window():
  """reset_every = ctx_frames + 1: el estado se recalcula con la ventana de los últimos ctx_frames frames en cada
  frame, que es justo lo que hace generate con un contexto más largo que ctx_frames."""
  m = tiny("convlstm")
  long = frames(1, 3 * FRAMES)
  s = make_stream(m, temperature=0.0, graphs=False, reset_every=m.ctx_frames + 1)
  for f in range(long.shape[1]):
    s.observe(long[:, f])
  ref = m.next_frame(long, 0.0)
  assert torch.equal(s.predict(), ref)


def test_step_commits_frame():
  """step() muestrea y deja el frame en el estado: igual a imagine, también llamado de a uno."""
  m = tiny("maskgit")
  ctx = frames(1, FRAMES - 1)
  a, _ = via_stream(m, ctx, 3, 1.0, steps=2, seed=4)
  s = make_stream(m, temperature=1.0, steps=2, seed=4, graphs=False)
  for f in range(ctx.shape[1]):
    s.observe(ctx[:, f])
  b = torch.stack([s.step() for _ in range(3)], 1)
  assert torch.equal(a, b)


def test_convlstm_step_onnx(tmp_path):
  """El paso de ConvLSTM exportado a ONNX (src/infer/export_codec.py) da lo mismo que el stream."""
  pytest.importorskip("onnxruntime")
  import onnxruntime as ort
  from src.infer.export_codec import ConvLSTMStep, export
  m = tiny("convlstm")
  st = [torch.randn(1, m.cfg.hidden, 8, 16) * 0.1 for _ in range(2 * m.cfg.n_layers)]
  x = m._embed(frames(1, 1))[:, 0].detach()
  names = ["x"] + [f"s{i}" for i in range(len(st))]
  p = export(ConvLSTMStep(m), (x, *st), names, ["logits"] + [f"o{i}" for i in range(len(st))], tmp_path / "s.onnx")
  got = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]).run(
    None, {n: t.numpy() for n, t in zip(names, (x, *st))})
  with torch.no_grad():
    ref = ConvLSTMStream(m, graphs=False)._step(x, *st)
  for g, r in zip(got, ref):
    assert abs(g - r.numpy()).max() < 1e-4


def test_random_run_spec_uses_manifest_config():
  """random:<corrida> arma la arquitectura registrada en results/checkpoints.json (sin bajar el checkpoint)."""
  from src import store
  from src.models import load_model
  man = store.load_manifest()
  if "jepa_xs" not in man:
    pytest.skip("sin results/checkpoints.json")
  m = load_model("random:jepa_xs")
  assert m.cfg.dim == man["jepa_xs"]["config"]["dim"] and m.num_params() == man["jepa_xs"]["params"]


# ---------------------------------------------------------------- LeWM desde píxeles

def fake_render(tok):
  """Cámara de juguete para los tests (sin los pesos del decoder): cada token pinta su parche de 16x16."""
  pal = (torch.arange(1024)[:, None] * torch.tensor([37, 91, 53]) % 256).to(torch.uint8).to(tok.device)
  img = pal[tok.long()].view(*tok.shape[:-1], 8, 16, 3)
  return img.repeat_interleave(16, -3).repeat_interleave(16, -2).contiguous()


@pytest.fixture
def px(monkeypatch):
  import src.models.lewm as lewm
  monkeypatch.setattr(lewm, "render_tokens", fake_render)
  return lambda **kw: tiny("lewm", input="pixels", **kw)


def test_lewm_pixels_encode_tokens_equals_pixels(px):
  m = px()
  tok = frames(2, 3)
  assert torch.equal(m.encode(tok), m.encode(fake_render(tok)))
  assert m.encode(tok).shape == (2, 3, m.cfg.dim)
  assert all(not n.startswith("encoder.in_proj") for n, _ in m.named_parameters())  # sin el codebook a la entrada


def test_lewm_pixels_train_step(px):
  m = px().train()
  loss, met = m.train_loss(frames(2, FRAMES))
  loss.backward()
  assert torch.isfinite(loss) and m.encoder.patch.weight.grad is not None


@pytest.mark.parametrize("rollout", ["token", "latent"])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_lewm_pixels_stream_equals_generate(px, rollout, temperature):
  m = px(rollout=rollout)
  ctx = frames(2, FRAMES - 1)
  ref = via_generate(m, ctx, 4, temperature)
  got, _ = via_stream(m, ctx, 4, temperature)
  assert torch.equal(got, ref)


def test_lewm_pixels_observe_pixels_equals_tokens(px):
  """Desde la cámara (observe_pixels) da lo mismo que desde los tokens de esos frames."""
  m = px()
  ctx = frames(1, FRAMES - 1)
  a = make_stream(m, temperature=0.0, graphs=False)
  b = make_stream(m, temperature=0.0, graphs=False)
  for f in range(ctx.shape[1]):
    a.observe(ctx[:, f])
    b.observe_pixels(fake_render(ctx[:, f]))
  assert b.pixels and torch.equal(a.predict(), b.predict())
  with pytest.raises(AssertionError):
    make_stream(tiny("lewm"), graphs=False).observe_pixels(fake_render(ctx[:, 0]))


@needs_cuda
@pytest.mark.parametrize("kind", ["maskgit", "convlstm", "jepa", "lewm"])
def test_graphs_equal_eager_cuda(kind):
  m = tiny(kind, dev="cuda")
  ctx = frames(1, FRAMES - 1, dev="cuda")
  a, _ = via_stream(m, ctx, 6, 1.0, steps=3, seed=5, graphs=False)
  b, _ = via_stream(m, ctx, 6, 1.0, steps=3, seed=5, graphs=True)
  assert (a == b).float().mean() >= 0.999


# ---------------------------------------------------------------- codec

def _codec_weights():
  from src.infer.codec import DECODER, ENCODER
  return ENCODER.exists() and DECODER.exists()


needs_codec = pytest.mark.skipif(not _codec_weights(), reason="sin pesos del encoder/decoder en data/raw")


@needs_codec
def test_codec_roundtrip_cpu():
  from src.infer.codec import Codec
  c = Codec("cpu", "fp32", graphs=False)
  tok = torch.randint(0, 1024, (1, 128), generator=torch.Generator().manual_seed(0))
  img = c.decode(tok)
  assert img.dtype == torch.uint8 and img.shape == (1, 128, 256, 3)
  # el decoder replica a src.viz.decode
  from src.viz import decode, load_decoder
  ref = decode(load_decoder(), tok.numpy())
  assert (img.numpy().astype(int) - ref.astype(int)).__abs__().max() <= 1
  back = c.encode(img)
  assert back.shape == (1, 128) and back.dtype == torch.long
  assert (back == tok).float().mean() > 0.5  # tokens al azar no son una imagen natural, pero mayormente vuelven


@needs_codec
@needs_cuda
def test_codec_graphs_equal_eager_cuda():
  from src.infer.codec import Codec
  tok = torch.randint(0, 1024, (1, 128), generator=torch.Generator().manual_seed(0)).cuda()
  e, g = Codec("cuda", "fp16", graphs=False), Codec("cuda", "fp16", graphs=True)
  img_e, img_g = e.decode(tok).clone(), g.decode(tok).clone()
  assert torch.equal(img_e, img_g)
  assert torch.equal(e.encode(img_e), g.encode(img_g))
