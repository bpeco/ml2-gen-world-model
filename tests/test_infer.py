"""Tests de la inferencia de punta a punta (src/infer): el estado por frame da lo mismo que generate.

En CPU, igualdad exacta de tokens contra model.generate con la misma semilla (temperatura 0 y 1). En CUDA,
el camino con graphs contra el mismo sin graphs. El codec necesita los pesos de data/raw (si no, se saltea).
"""
import pytest
import torch

from src.common import TPF
from src.infer.stream import GPTStream, LatentStream, MaskGITStream, TokenStream, make_stream
from src.models.gpt import GPT
from src.models.lewm import LeWM
from src.models.maskgit import MaskGIT

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="sin CUDA")
FRAMES = 6


def frames(B, F, seed=1, dev="cpu"):
  g = torch.Generator().manual_seed(seed)
  return torch.randint(0, 1024, (B, F, 128), generator=g).to(dev)


def fake_render(tok):
  """Cámara de juguete para los tests (sin los pesos del decoder): cada token pinta su parche de 16x16."""
  pal = (torch.arange(1024)[:, None] * torch.tensor([37, 91, 53]) % 256).to(torch.uint8).to(tok.device)
  img = pal[tok.long()].view(*tok.shape[:-1], 8, 16, 3)
  return img.repeat_interleave(16, -3).repeat_interleave(16, -2).contiguous()


@pytest.fixture(autouse=True)
def px(monkeypatch):
  """LeWM ve píxeles: los tests de este módulo usan la cámara de juguete en vez del decoder de comma."""
  import src.models.lewm as lewm
  monkeypatch.setattr(lewm, "render_tokens", fake_render)
  return lambda **kw: tiny("lewm", **kw)


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
  elif kind == "gpt":
    m = shake(GPT(GPT.Config(n_layer=2, n_head=4, dim=64, block_size=FRAMES * TPF, **kw)))
    m.ctx_frames = FRAMES - 1   # el frame nuevo cabe en el bloque (el de verdad usa 19 de 20)
  elif kind == "gpt2m":         # la forma de gpt2m (bloque de 20 frames, 19 de contexto) con 2 capas de 64
    import src.models.gpt2m as g2   # agrega third_party/commavq al path
    from utils.gpt import GPTConfig
    m = g2.Gpt2m(GPTConfig(n_layer=2, n_head=4, dim=64, intermediate_size=256))
  else:
    m = shake(LeWM(LeWM.Config(dim=32, enc_layers=1, enc_heads=2, enc_mlp=64, proj_hidden=64, pred_layers=2,
                               pred_heads=4, pred_dim_head=8, pred_mlp=64, pred_dropout=0.0, num_proj=64, **kw)),
              scale=1.0, noise=0.05)
  return m.to(dev).eval()


def via_stream(m, ctx, n, temperature, steps=None, seed=0, graphs=False, **kw):
  s = make_stream(m, temperature=temperature, steps=steps, seed=seed, graphs=graphs, **kw)
  for f in range(ctx.shape[1]):
    s.observe(ctx[:, f])
  return torch.stack(list(s.imagine(n)), 1), s


def via_generate(m, ctx, n, temperature, steps=None, seed=0):
  g = torch.Generator(ctx.device).manual_seed(seed)
  return m.rollout(ctx, n, temperature=temperature, steps=steps, generator=g)


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_stream_equals_generate_cpu(temperature):
  m = tiny("maskgit")
  ctx = frames(2, FRAMES - 1)
  ref = via_generate(m, ctx, 4, temperature, 3)
  got, s = via_stream(m, ctx, 4, temperature, 3)
  assert got.shape == ref.shape == (2, 4, 128)
  assert torch.equal(got, ref)


def test_adapters():
  assert isinstance(make_stream(tiny("maskgit")), MaskGITStream)
  assert isinstance(make_stream(tiny("lewm")), LatentStream)
  assert isinstance(make_stream(tiny("gpt")), GPTStream)
  assert isinstance(make_stream(tiny("gpt2m")), GPTStream)
  from src.models.baselines import Unigram
  s = make_stream(Unigram(torch.zeros(1024)))
  assert isinstance(s, TokenStream) and not isinstance(s, GPTStream)


# ---------------------------------------------------------------- GPT y gpt2m: el frame entero en un graph

def ctx_for(m, B=2, seed=1):
  """Un contexto lleno para el stream del modelo (los frames que caben en el bloque)."""
  return frames(B, make_stream(m, graphs=False).ctx_frames, seed)


def draw_noise(B, seed, dev="cpu"):
  """El ruido de un frame tal como sample_logits lo saca del generador: una tirada (B, 1024) por token, en orden."""
  g = torch.Generator(dev).manual_seed(seed)
  buf = torch.empty(128, B, 1024, device=dev)
  for i in range(128):
    buf[i].exponential_(1, generator=g)
  return buf


@pytest.mark.parametrize("kind", ["gpt", "gpt2m"])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_gpt_next_frame_noise_equals_generator(kind, temperature):
  """next_frame(noise=...) con el ruido tirado a mano en el orden del generador da los mismos tokens que next_frame con el
  generador: el ruido como dato no cambia el muestreo."""
  m = tiny(kind)
  ctx = ctx_for(m)
  ref = m.next_frame(ctx, temperature, generator=torch.Generator().manual_seed(7))
  got = m.next_frame(ctx, temperature, noise=draw_noise(2, 7))
  assert ref.shape == (2, 128) and torch.equal(got, ref)
  if temperature:
    assert not torch.equal(got, m.next_frame(ctx, temperature, noise=draw_noise(2, 8)))   # el ruido sí se usa


@pytest.mark.parametrize("kind", ["gpt", "gpt2m"])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_gpt_fused_frame_equals_next_frame(kind, temperature):
  """El frame entero de GPTStream (lo que se captura en un graph) da lo mismo que next_frame con el mismo generador."""
  m = tiny(kind)
  s = GPTStream(m, temperature=temperature, seed=7, graphs=False)
  ctx = ctx_for(m)
  got = s._frame(ctx, *s._noise(2))
  ref = m.next_frame(ctx, temperature, generator=torch.Generator().manual_seed(7))
  assert torch.equal(got, ref)


@pytest.mark.parametrize("kind", ["gpt", "gpt2m"])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_gpt_stream_equals_generate_cpu(kind, temperature):
  m = tiny(kind)
  ctx = ctx_for(m, B=1)
  ref = via_generate(m, ctx, 3, temperature)
  got, s = via_stream(m, ctx, 3, temperature)
  assert isinstance(s, GPTStream) and got.shape == ref.shape == (1, 3, 128)
  assert torch.equal(got, ref)


# ---------------------------------------------------------------- atención contra la caché (cached_attention)

def attn_case(T, dev="cpu", dtype=torch.float32, B=2, H=3, S=40, hd=8):
  """q, k, v al azar y una máscara bool (T, S) sin filas enteras tapadas."""
  g = torch.Generator().manual_seed(T)
  q, k, v = (torch.randn(B, H, n, hd, generator=g).to(dev, dtype) for n in (T, S, S))
  mask = torch.rand(T, S, generator=g) > 0.3
  mask[:, 0] = True
  return q, k, v, mask.to(dev)


@pytest.mark.parametrize("T", [1, 5, TPF])
def test_cached_attention_equals_sdpa_cpu(T):
  """La atención explícita del decode es la misma cuenta que SDPA con la máscara bool, con el bias (T, S) del Backbone
  y con el (1, 1, T, S) de gpt2m."""
  import torch.nn.functional as F
  from src.models.backbone import additive_mask, cached_attention
  q, k, v, mask = attn_case(T)
  ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
  bias = additive_mask(mask, q.dtype)
  assert torch.allclose(cached_attention(q, k, v, bias), ref, atol=1e-5)
  assert torch.allclose(cached_attention(q, k, v, bias[None, None]), ref, atol=1e-5)


@needs_cuda
@pytest.mark.parametrize("T", [1, TPF])
def test_cached_attention_equals_sdpa_cuda_fp16(T):
  import torch.nn.functional as F
  from src.models.backbone import additive_mask, cached_attention
  q, k, v, mask = attn_case(T, "cuda", torch.float16, S=2584, hd=64)
  ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
  got = cached_attention(q, k, v, additive_mask(mask, q.dtype))
  assert (got.float() - ref.float()).abs().max() < 2e-3


def test_gpt2m_next_frame_restores_comma_attention():
  """El forward de la atención de comma se reemplaza solo mientras la caché está enganchada: después, logits() es el
  de comma, con su máscara bool."""
  m = tiny("gpt2m")
  seq = torch.cat([frames(1, 2)[:, 0], frames(1, 2)[:, 1]], 1)
  before = m.logits(seq)
  m.next_frame(ctx_for(m, B=1), 1.0, generator=torch.Generator().manual_seed(0))
  assert all("forward" not in b.attn.__dict__ for b in m.gpt.transformer.h)
  assert m.gpt.causal_mask.dtype == torch.bool
  assert torch.equal(m.logits(seq), before)


def test_gpt_stream_ctx_frames_fit_the_block():
  """Como MaskGITStream: el frame nuevo tiene que caber en el bloque del modelo."""
  m = tiny("gpt")
  m.ctx_frames = 19    # la clase dice 19, el bloque chico solo admite FRAMES - 1
  assert GPTStream(m, graphs=False).ctx_frames == FRAMES - 1
  assert GPTStream(tiny("gpt2m"), graphs=False).ctx_frames == 19


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_maskgit_fused_frame_equals_next_frame(temperature):
  """El frame entero de MaskGITStream (lo que se captura en un graph) da lo mismo que next_frame."""
  m = tiny("maskgit")
  s = MaskGITStream(m, temperature=temperature, steps=4, seed=7, graphs=False)
  ctx = frames(2, FRAMES - 1)
  got = s._frame(ctx, *s._noise(2))
  ref = m.next_frame(ctx, temperature, 4, torch.Generator().manual_seed(7))
  assert torch.equal(got, ref)


def test_predict_k_does_not_change_state():
  m = tiny("maskgit")
  ctx = frames(1, FRAMES - 1)
  s = make_stream(m, temperature=0.0, steps=2, graphs=False)
  for f in range(ctx.shape[1]):
    s.observe(ctx[:, f])
  far = s.predict(k=3)
  roll = torch.stack(list(s.imagine(3)), 1)
  assert torch.equal(far, roll[:, -1])          # el frame t+3 de predict es el tercero del rollout
  assert torch.equal(s.predict(), s.predict())  # predict no avanza el estado


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


def test_random_run_spec_uses_manifest_config():
  """random:<corrida> arma la arquitectura registrada en results/checkpoints.json (sin bajar el checkpoint)."""
  from src import store
  from src.models import load_model
  man = store.load_manifest()
  if "gpt_m" not in man:
    pytest.skip("sin results/checkpoints.json")
  m = load_model("random:gpt_m")
  assert m.cfg.dim == man["gpt_m"]["config"]["dim"] and sum(p.numel() for p in m.parameters()) == man["gpt_m"]["params"]


# ---------------------------------------------------------------- LeWM desde píxeles (LatentStream)

K = 5  # frameskip de LeWM: un paso del modelo cada 5 frames de cámara


def poses(B, n, seed=2):
  return torch.randn(B, n, 6, generator=torch.Generator().manual_seed(seed))


def feed(s, tok, pose=None, pixels=False):
  """Alimenta el stream con n frames de cámara (tokens (B, n, 128), poses (B, n, 6))."""
  for f in range(tok.shape[1]):
    p = None if pose is None else pose[:, f]
    s.observe_pixels(fake_render(tok[:, f]), p) if pixels else s.observe(tok[:, f], p)


def test_lewm_pixels_encode_tokens_equals_pixels(px):
  m = px()
  tok = frames(2, 3)
  assert torch.equal(m.encode(tok), m.encode(fake_render(tok)))
  assert m.encode(tok).shape == (2, 3, m.cfg.dim)


def test_lewm_pixels_train_step(px):
  m = px().train()
  loss, met = m.train_loss(frames(2, 4), poses=torch.randn(2, 4, 30))
  loss.backward()
  assert torch.isfinite(loss) and m.encoder.patch_embed.weight.grad is not None


def test_lewm_stream_steps_every_frameskip_camera_frames(px):
  """Solo 1 de cada 5 frames de cámara se codifica; las acciones del estado son las 5 poses apiladas del bloque."""
  m = px()
  tok, pose = frames(1, 3 * K + 1), poses(1, 3 * K + 1)
  s = make_stream(m, graphs=False)
  moved = [s.observe(tok[:, f], pose[:, f]) for f in range(tok.shape[1])]
  assert moved == [f % K == 0 for f in range(tok.shape[1])] and s.n_seen == 4 and s.full
  ref = m.encode(tok[:, ::K])                                          # los frames 0, 5, 10, 15
  assert torch.allclose(s.z, ref[:, -3:], atol=1e-6) and s.acts.shape == (1, 2, m.cfg.dim)
  stacked = torch.stack([pose[:, K * i:K * i + K].reshape(1, -1) for i in range(1, 3)], 1)   # bloques 1 y 2
  assert torch.allclose(s.acts, m.action_embed(stacked, 1, 2, "cpu"), atol=1e-6)


def test_lewm_stream_predict_equals_model(px):
  """predict() = el paso del modelo sobre los últimos 3 frames con sus acciones y la acción pedida."""
  m = px()
  tok, pose = frames(2, 3 * K), poses(2, 3 * K)
  s = make_stream(m, graphs=False)
  feed(s, tok, pose)
  nxt = poses(2, 1, seed=9)[:, 0]
  got = s.predict(pose=nxt)
  z = m.encode(tok[:, ::K])
  stack = torch.cat([torch.stack([pose[:, K * i:K * i + K].reshape(2, -1) for i in range(2)], 1),
                     nxt.repeat(1, K)[:, None]], 1)                    # la pose pedida se mantiene los 5 frames
  assert torch.allclose(got, m._next(z, m.action_embed(stack, 2, 3, "cpu")), atol=1e-6)
  assert got.shape == (2, m.cfg.dim)


def test_lewm_pixels_observe_pixels_equals_tokens(px):
  """Desde la cámara (observe_pixels) da lo mismo que desde los tokens de esos frames."""
  m = px()
  tok = frames(1, 2 * K)
  a, b = make_stream(m, graphs=False), make_stream(m, graphs=False)
  feed(a, tok, pixels=False)
  feed(b, tok, pixels=True)
  assert torch.equal(a.predict(), b.predict())


def test_lewm_stream_predict_k_does_not_change_state(px):
  m = px()
  s = make_stream(m, graphs=False)
  feed(s, frames(1, 3 * K), poses(1, 3 * K))
  z, acts = s.z.clone(), s.acts.clone()
  far = s.predict(k=3)
  assert torch.equal(s.z, z) and torch.equal(s.acts, acts) and torch.equal(s.predict(), s.predict())
  roll = torch.stack(list(s.imagine(3)), 1)                            # imagine: cada ẑ entra al estado
  assert torch.allclose(far, roll[:, -1], atol=1e-6) and s.n_seen == 6


def test_lewm_stream_works_with_fewer_than_history_frames(px):
  m = px()
  s = make_stream(m, graphs=False)
  feed(s, frames(1, K), poses(1, K))
  assert not s.full and s.z.shape[1] == 1 and s.predict().shape == (1, m.cfg.dim)


def test_lewm_stream_predict_pixels_with_decoder(px):
  from src.models.lewm_decoder import DecoderConfig, LeWMDecoder
  m = px()
  dec = LeWMDecoder(DecoderConfig(dim_in=32, dec_dim=32, layers=1, heads=2, n_kv=2)).eval()
  s = make_stream(m, graphs=False)
  s.decoder = dec
  feed(s, frames(1, K))
  img = s.predict_pixels()
  assert img.dtype == torch.uint8 and img.shape == (1, 128, 256, 3)


@needs_cuda
def test_lewm_stream_graphs_equal_eager_cuda():
  m = tiny("lewm", dev="cuda")
  tok, pose = frames(1, 4 * K, dev="cuda"), poses(1, 4 * K).cuda()
  a, b = make_stream(m, graphs=False), make_stream(m, graphs=True)
  for s in (a, b):
    for f in range(tok.shape[1]):
      s.observe_pixels(fake_render(tok[:, f]), pose[:, f])
  assert torch.allclose(a.predict(), b.predict(), atol=1e-4)


@needs_cuda
@pytest.mark.parametrize("kind", ["maskgit"])
def test_graphs_equal_eager_cuda(kind):
  m = tiny(kind, dev="cuda")
  ctx = frames(1, FRAMES - 1, dev="cuda")
  a, _ = via_stream(m, ctx, 6, 1.0, steps=3, seed=5, graphs=False)
  b, _ = via_stream(m, ctx, 6, 1.0, steps=3, seed=5, graphs=True)
  assert (a == b).float().mean() >= 0.999


@needs_cuda
@pytest.mark.parametrize("kind", ["gpt", "gpt2m"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_gpt_frame_graph_equals_eager_cuda(kind, dtype, temperature):
  """Con graphs el frame entero es un solo graph (s._g) y da los mismos tokens que sin graphs."""
  m = tiny(kind, dev="cuda").to(dtype)
  ctx = ctx_for(m, B=1).cuda()
  a, _ = via_stream(m, ctx, 4, temperature, seed=5, graphs=False)
  b, s = via_stream(m, ctx, 4, temperature, seed=5, graphs=True)
  assert len(s._g) == 1                        # el graph del frame se capturó y se usó
  assert (a == b).float().mean() >= 0.999


@needs_cuda
def test_gpt_frame_graph_is_rebuilt_when_the_cache_is_dropped():
  """m.logits() tira la caché KV del GPT: el graph viejo apunta a ella, así que el stream rehace el graph."""
  m = tiny("gpt", dev="cuda")
  ctx = frames(1, FRAMES - 1, dev="cuda")
  outs = []
  for graphs in (False, True):
    s = make_stream(m, temperature=1.0, seed=5, graphs=graphs)
    for f in range(ctx.shape[1]):
      s.observe(ctx[:, f])
    first = s.step()
    if graphs:
      (old, _), = s._g.values()
    m.logits(torch.cat([ctx[:, 0], ctx[:, 1]], 1))     # tira la caché y los graphs del modelo
    outs.append(torch.stack([first, s.step()], 1))
  assert (outs[0] == outs[1]).float().mean() >= 0.999
  # el graph reescribe la caché entera en cada replay, así que los tokens solos no delatan un graph viejo: se mira que
  # sea otro y que apunte a las cachés nuevas del modelo
  (new, graphs_dict), = s._g.values()
  assert new is not old and graphs_dict is m._graphs


@pytest.mark.parametrize("kind", ["gpt", "gpt2m"])
def test_gpt_fused_frames_chain_equals_generate_cpu(kind):
  """Varios frames por el camino del ruido (_noise + _frame, el del graph) con el contexto deslizándose: el generador del
  stream sigue en el mismo punto que el de generate, así que el segundo frame y los siguientes también coinciden."""
  m = tiny(kind)
  ctx = ctx_for(m, B=1)
  ref = via_generate(m, ctx, 3, 1.0)
  s = GPTStream(m, temperature=1.0, seed=0, graphs=False)
  win, out = ctx, []
  for _ in range(3):
    new = s._frame(win, *s._noise(1))
    out.append(new)
    win = torch.cat([win, new[:, None]], 1)[:, -s.ctx_frames:]
  assert torch.equal(torch.stack(out, 1), ref)


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
