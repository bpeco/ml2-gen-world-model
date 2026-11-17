import pytest
import torch

from src.common import BOS, MASK, TPF, with_bos
from src.models import build, load_model
from src.models.base import lm_loss
from src.models.maskgit import MaskGIT, n_masked

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="sin CUDA")


def tiny(frames=4, dev="cpu", **kw):
  torch.manual_seed(0)
  m = MaskGIT(MaskGIT.Config(n_layer=2, n_head=4, dim=64, block_size=frames * TPF, **kw)).to(dev).eval()
  for p in m.parameters():  # init pequeña = logits casi planos; los agrandamos
    p.data.mul_(4.0)
  return m


def rand_frames(B, F, dev="cpu", seed=1):
  g = torch.Generator().manual_seed(seed)
  return torch.randint(0, 1024, (B, F, 128), generator=g).to(dev)


@pytest.mark.parametrize("dev", DEVS)
def test_block_leakage(dev):
  m = tiny(dev=dev)
  x = with_bos(rand_frames(1, 3, dev))
  x2 = x.clone()
  x2[0, 2 * TPF + 5] = MASK
  with torch.no_grad():
    a, b = m.logits(x), m.logits(x2)
  assert (a[:, :2 * TPF] - b[:, :2 * TPF]).abs().max() < 1e-5
  assert (a[:, 2 * TPF:] - b[:, 2 * TPF:]).abs().max() > 1e-4
  x3 = x.clone()  # dentro del frame es bidireccional: un token posterior cambia uno anterior
  x3[0, TPF + 100] = MASK
  with torch.no_grad():
    c = m.logits(x3)
  assert (a[:, TPF + 3] - c[:, TPF + 3]).abs().max() > 1e-4


@pytest.mark.parametrize("dev", DEVS)
def test_cache_consistency(dev):
  m = tiny(dev=dev)
  ctx = rand_frames(2, 3, dev)
  fr = torch.full((2, TPF), MASK, device=dev)
  fr[:, 0] = BOS
  fr[:, 1:60] = rand_frames(2, 1, dev, seed=5)[:, 0, :59]
  with torch.no_grad():
    seq = with_bos(ctx)
    full = m.logits(torch.cat([seq, fr], 1))[:, -TPF:]
    m._ensure_cache(2, torch.float32, torch.device(dev))
    P = seq.shape[1]
    pos = torch.arange(m.cfg.block_size, device=dev)
    m.net(seq, pos[:P])
    inc = m._frame_pass(fr, pos[P:P + TPF])
    inc2 = m._frame_pass(fr, pos[P:P + TPF])  # reescribir el frame es idempotente
  assert (inc - full).abs().max() < 1e-4
  assert (inc2 - full).abs().max() < 1e-4


def test_schedule():
  for S in (1, 2, 4, 8, 16):
    ns = [128] + [n_masked(s, S) for s in range(1, S + 1)]
    assert ns[-1] == 0
    assert all(a >= b for a, b in zip(ns, ns[1:]))
  assert n_masked(1, 1) == 0


@pytest.mark.parametrize("S", [1, 2, 4, 8])
def test_pass_count(S):
  m = tiny()
  calls = []
  m.net.register_forward_hook(lambda mod, a, o: calls.append(a[0].shape[1]))
  m.next_frame(rand_frames(2, 3), steps=S)
  assert calls == [3 * TPF] + [TPF] * S


@pytest.mark.parametrize("dev", DEVS)
def test_output(dev):
  m = tiny(dev=dev)
  ctx = rand_frames(3, 3, dev)
  for S in (1, 3, 8):
    out = m.next_frame(ctx, steps=S, generator=torch.Generator(dev).manual_seed(0))
    assert out.shape == (3, 128) and out.dtype == torch.long
    assert out.min() >= 0 and out.max() <= 1023
  g = lambda s=7: torch.Generator(dev).manual_seed(s)
  assert (m.next_frame(ctx, steps=4, generator=g()) == m.next_frame(ctx, steps=4, generator=g())).all()
  assert (m.next_frame(ctx, steps=4, generator=g()) != m.next_frame(ctx, steps=4, generator=g(8))).any()
  o0 = m.next_frame(ctx, temperature=0, steps=4)
  assert (o0 == m.next_frame(ctx, temperature=0, steps=4)).all()
  assert o0.max() <= 1023
  assert m.next_frame(rand_frames(1, 1, dev), steps=2).shape == (1, 128)  # contexto corto
  assert m.next_frame(rand_frames(1, 6, dev), steps=2).shape == (1, 128)  # contexto largo: se recorta a block_size - 1 frames


@pytest.mark.parametrize("dev", DEVS)
def test_one_step_is_argmax_of_masked_pass(dev):
  m = tiny(dev=dev)
  ctx = rand_frames(2, 3, dev)
  out = m.next_frame(ctx, temperature=0, steps=1)
  lg = m.last_frame_logits(torch.cat([ctx, torch.zeros_like(ctx[:, :1])], 1))
  assert lg.shape == (2, 128, 1024)
  assert (out == lg.argmax(-1)).all()


def test_steps_reuse_confident_tokens():
  # con temperature 0 los tokens fijados no cambian: el resultado de S pasos coincide con un
  # seguimiento a mano con el mismo criterio de selección
  m = tiny()
  ctx = rand_frames(1, 3)
  S = 4
  out = m.next_frame(ctx, temperature=0, steps=S)
  seq = with_bos(ctx)
  tok = torch.full((1, TPF), MASK)
  tok[:, 0] = BOS
  fixed = torch.zeros(1, 128, dtype=torch.bool)
  n_prev = 128
  with torch.no_grad():
    for s in range(1, S + 1):
      lg = m.logits(torch.cat([seq, tok], 1))[:, -TPF + 1:].float()
      lp = torch.log_softmax(lg, -1)
      samp = lg.argmax(-1)
      conf = lp.gather(-1, samp[..., None])[..., 0].masked_fill(fixed, float("-inf"))
      k = n_prev - n_masked(s, S)
      fixed.scatter_(1, conf.topk(k, -1).indices, True)
      tok[:, 1:] = torch.where(fixed & (tok[:, 1:] == MASK), samp, tok[:, 1:])
      n_prev -= k
  assert fixed.all() and (tok[:, 1:] == out).all()


@needs_cuda
@pytest.mark.parametrize("S", [1, 4])
def test_cuda_graphs_equal_eager(S):
  m = tiny(dev="cuda")
  ctx = rand_frames(2, 3, "cuda")
  g = lambda: torch.Generator("cuda").manual_seed(3)
  ref = [m.next_frame(ctx, steps=S, generator=g()) for _ in range(2)]
  m.use_cuda_graphs(True)
  got = [m.next_frame(ctx, steps=S, generator=g()) for _ in range(2)]
  assert all((a == b).all() for a, b in zip(ref, got))
  ctx2 = torch.cat([ctx, got[0][:, None]], 1)  # otra longitud de contexto reutiliza el graph
  m.use_cuda_graphs(False)
  r2 = m.next_frame(ctx2, steps=S, generator=g())
  m.use_cuda_graphs(True)
  assert (m.next_frame(ctx2, steps=S, generator=g()) == r2).all()
  m.next_frame(rand_frames(1, 3, "cuda"), steps=S)  # otro batch: caché nueva, graph nuevo


def masks_for(m, B, Fr, seed):
  """Reconstruye la máscara de train_loss (mismo orden de RNG)."""
  torch.manual_seed(seed)
  r = m._mask_rates(B, Fr, torch.device("cpu"))
  return torch.rand(B, Fr, 128) < r[..., None]


def test_train_loss_only_masked():
  m = tiny()
  fr = rand_frames(2, 3)
  torch.manual_seed(0)
  loss, met = m.train_loss(fr)
  mk = masks_for(m, 2, 3, 0)
  assert met["mask_rate"].item() == pytest.approx(mk.float().mean().item())
  x = with_bos(fr).view(2, 3, TPF).clone()
  x[:, :, 1:] = torch.where(mk, torch.full_like(fr, MASK), fr)
  with torch.no_grad():
    lg = m.logits(x.view(2, -1)).view(2, 3, TPF, -1)[:, :, 1:].float()
  ce = torch.nn.functional.cross_entropy(lg.reshape(-1, 1024), fr.reshape(-1), reduction="none").view(2, 3, 128)
  assert torch.allclose(ce[mk].mean(), loss.detach(), atol=1e-5)
  # cambiar el target de una posición no enmascarada no cambia la pérdida; de una enmascarada sí
  y2 = fr.clone()
  b, f, c = (~mk).nonzero()[0].tolist()
  y2[b, f, c] = (y2[b, f, c] + 1) % 1024
  assert torch.allclose(lm_loss(lg, y2, mask=mk)[0], loss.detach(), atol=1e-6)
  b, f, c = mk.nonzero()[0].tolist()
  y2[b, f, c] = (y2[b, f, c] + 1) % 1024
  assert not torch.allclose(lm_loss(lg, y2, mask=mk)[0], loss.detach(), atol=1e-6)


def test_train_loss_teacher_raises():
  with pytest.raises(ValueError):
    tiny().train_loss(rand_frames(1, 2), teacher=object())


@pytest.mark.parametrize("sched,lo,hi,mean", [("cosine", 0.0, 1.0, 2 / torch.pi), ("uniform", 0.5, 1.0, 0.75)])
def test_mask_rates(sched, lo, hi, mean):
  m = tiny(mask_schedule=sched)
  torch.manual_seed(0)
  r = m._mask_rates(64, 20, torch.device("cpu"))
  assert r.min() >= lo - 1e-6 and r.max() <= hi + 1e-6
  assert r.max() - r.min() > 0.8 * (hi - lo)
  assert abs(r.mean().item() - mean) < 0.03
  _, met = m.train_loss(rand_frames(64, 2))
  assert abs(met["mask_rate"].item() - mean) < 0.1


@pytest.mark.parametrize("dev", DEVS)
def test_overfit(dev):
  m = tiny(frames=3, dev=dev).train()
  fr = rand_frames(4, 3, dev)
  opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
  first = None
  for i in range(100):
    loss, _ = m.train_loss(fr)
    opt.zero_grad()
    loss.backward()
    opt.step()
    first = first if first is not None else loss.item()
  assert loss.item() < 0.5 * first


def test_registry():
  for size in "sm":
    m = build("maskgit", size=size)
    print("params", size, sum(p.numel() for p in m.parameters()))
  m = load_model("random:maskgit_s", "cpu")
  assert m.name == "random_maskgit_s" and not m.has_likelihood
