import pytest
import torch

from src.common import BOS, TPF, with_bos
from src.models import SIZES, build, load_model
from src.models.backbone import Backbone, BackboneConfig
from src.models.gpt import GPT

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="sin CUDA")


def tiny_net(attn, frames=3, dev="cpu"):
  torch.manual_seed(0)
  cfg = BackboneConfig(n_layer=2, n_head=4, dim=64, block_size=frames * TPF, attn=attn)
  return Backbone(cfg).to(dev).eval()


def tiny_gpt(frames=5, dev="cpu"):
  torch.manual_seed(0)
  m = GPT(GPT.Config(n_layer=2, n_head=4, dim=64, block_size=frames * TPF)).to(dev).eval()
  for p in m.parameters():  # init pequeña = logits casi planos; los agrandamos para que argmax sea informativo
    p.data.mul_(4.0)
  return m


def rand_frames(B, F, dev="cpu", seed=1):
  g = torch.Generator().manual_seed(seed)
  return torch.randint(0, 1024, (B, F, 128), generator=g).to(dev)


@pytest.mark.parametrize("dev", DEVS)
def test_cache_causal(dev):
  net = tiny_net("causal", dev=dev)
  x = with_bos(rand_frames(2, 3, dev))
  with torch.no_grad():
    full = net(x)
    net.setup_cache(2, device=dev)
    P = 200
    parts = [net(x[:, :P], torch.arange(P, device=dev))]
    for t in range(P, 260):
      parts.append(net(x[:, t:t + 1], torch.tensor([t], device=dev)))
    inc = torch.cat(parts, 1)
  assert (inc - full[:, :260]).abs().max() < 1e-4


@pytest.mark.parametrize("dev", DEVS)
def test_cache_block(dev):
  net = tiny_net("block", dev=dev)
  x = with_bos(rand_frames(2, 3, dev))
  with torch.no_grad():
    full = net(x)
    net.setup_cache(2, device=dev)
    inc = torch.cat([net(x[:, f * TPF:(f + 1) * TPF], torch.arange(f * TPF, (f + 1) * TPF, device=dev))
                     for f in range(3)], 1)
  assert (inc - full).abs().max() < 1e-4


@pytest.mark.parametrize("dev", DEVS)
def test_leakage_causal(dev):
  net = tiny_net("causal", dev=dev)
  x = with_bos(rand_frames(1, 3, dev))
  t = 150
  x2 = x.clone()
  x2[0, t] = (x2[0, t] + 1) % 1024
  with torch.no_grad():
    a, b = net(x), net(x2)
  assert (a[:, :t] - b[:, :t]).abs().max() < 1e-5
  assert (a[:, t:] - b[:, t:]).abs().max() > 1e-4


@pytest.mark.parametrize("dev", DEVS)
def test_leakage_block(dev):
  net = tiny_net("block", dev=dev)
  x = with_bos(rand_frames(1, 3, dev))
  t = TPF + 40  # frame 1
  x2 = x.clone()
  x2[0, t] = (x2[0, t] + 1) % 1024
  with torch.no_grad():
    a, b = net(x), net(x2)
  assert (a[:, :TPF] - b[:, :TPF]).abs().max() < 1e-5               # frame 0 intacto
  assert (a[:, TPF:t] - b[:, TPF:t]).abs().max() > 1e-4              # posiciones previas del frame 1 cambian
  assert (a[:, 2 * TPF:] - b[:, 2 * TPF:]).abs().max() > 1e-4        # frames posteriores cambian


@pytest.mark.parametrize("dev", DEVS)
def test_next_frame(dev):
  m = tiny_gpt(5, dev)
  ctx = rand_frames(3, 4, dev)
  def run(seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    return m.next_frame(ctx, 1.0, generator=g)
  a, b, c = run(0), run(0), run(1)
  assert a.shape == (3, 128) and a.dtype == torch.long
  assert a.min() >= 0 and a.max() <= 1023
  assert torch.equal(a, b) and not torch.equal(a, c)
  # greedy == forwards completos repetidos
  g = m.next_frame(ctx, temperature=0.0)
  seq = torch.cat([with_bos(ctx), torch.full((3, 1), BOS, device=dev)], 1)
  out = []
  with torch.no_grad():
    for _ in range(128):
      nxt = m.logits(seq)[:, -1, :BOS].argmax(-1)
      out.append(nxt)
      seq = torch.cat([seq, nxt[:, None]], 1)
  assert torch.equal(g, torch.stack(out, 1))


@pytest.mark.parametrize("dev", DEVS)
def test_next_frame_window(dev):
  # más de 19 frames de contexto: solo cuentan los últimos 19 (bloque completo de 20 frames)
  m = tiny_gpt(20, dev)
  ctx = rand_frames(1, 21, dev)
  a = m.next_frame(ctx, temperature=0.0)
  b = m.next_frame(ctx[:, -19:], temperature=0.0)
  assert torch.equal(a, b)


@needs_cuda
def test_cuda_graph_equals_eager():
  m = tiny_gpt(5, "cuda")
  ctx = rand_frames(2, 4, "cuda")
  for temp in (1.0, 0.0):
    g1 = torch.Generator(device="cuda").manual_seed(3)
    g2 = torch.Generator(device="cuda").manual_seed(3)
    m.use_cuda_graphs(False)
    a = m.next_frame(ctx, temp, generator=g1)
    m.use_cuda_graphs(True)
    b = m.next_frame(ctx, temp, generator=g2)
    c = m.next_frame(ctx, temp, generator=torch.Generator(device="cuda").manual_seed(3))  # reutiliza el graph
    assert torch.equal(a, b) and torch.equal(a, c)


@needs_cuda
def test_fp16():
  m32 = tiny_gpt(5, "cuda")
  m16 = tiny_gpt(5, "cuda").to(torch.float16)
  ctx = rand_frames(2, 4, "cuda")
  x = with_bos(ctx)
  with torch.no_grad():
    l32, l16 = m32.logits(x).float(), m16.logits(x).float()
  assert l16.dtype == torch.float32 and (l32 - l16).abs().max() < 0.5
  assert (l32.argmax(-1) == l16.argmax(-1)).float().mean() > 0.9
  m16.use_cuda_graphs(True)
  out = m16.next_frame(ctx, 1.0, generator=torch.Generator(device="cuda").manual_seed(0))
  assert out.shape == (2, 128) and out.max() <= 1023


def test_params_and_registry():
  counts = {}
  for s in SIZES:
    counts[s] = build("gpt", size=s).net.num_params()
  print("params:", counts)
  assert 11e6 < counts["s"] < 14e6 and 25e6 < counts["m"] < 30e6 and 80e6 < counts["l"] < 95e6
  m = load_model("random:gpt_s", "cpu")
  assert isinstance(m, GPT) and m.name == "random_gpt_s"
  assert m.logits(with_bos(rand_frames(1, 1))).shape == (1, TPF, 1025)


def test_train_loss():
  m = tiny_gpt(3)
  m.train()
  frames = rand_frames(2, 3)
  loss, met = m.train_loss(frames)
  loss.backward()
  assert torch.isfinite(loss) and "ce" in met
  loss2, met2 = m.train_loss(frames, teacher=tiny_gpt(3), alpha=0.5)
  assert torch.isfinite(loss2) and "kl" in met2
