import pytest
import torch

from src.common import MASK
from src.infer.stream import GenieStream, make_stream
from src.models import build, load_model
from src.models.base import lm_loss
from src.models.genie import FIDELIDAD, Genie, n_masked

T = 4


def tiny(frames=T, **kw):
  torch.manual_seed(0)
  m = Genie(Genie.Config(n_layer=2, n_head=4, dim=64, frames=frames, **kw)).eval()
  for p in m.parameters():  # logits menos planos
    p.data.mul_(4.0)
  return m


def rand_frames(B, F, seed=1):
  g = torch.Generator().manual_seed(seed)
  return torch.randint(0, 1024, (B, F, 128), generator=g)


def test_first_frame_never_masked_and_rate():
  m = tiny(frames=16)
  torch.manual_seed(0)
  mk = m.mask_for(256, 16, torch.device("cpu"))
  assert not mk[:, 0].any()
  per = mk[:, 1:].float().mean((1, 2))  # tasa por secuencia ~ U[0.5, 1]
  assert per.min() > 0.4 and per.max() <= 1.0 and per.max() - per.min() > 0.4
  assert abs(per.mean().item() - 0.75) < 0.03
  # la tasa es constante entre los frames de una secuencia (una tasa por muestra)
  fr = mk[:, 1:].float().mean(2)
  assert (fr - per[:, None]).abs().max() < 0.2


def test_train_loss_only_masked():
  m = tiny()
  fr = rand_frames(2, 3)
  torch.manual_seed(0)
  loss, met = m.train_loss(fr)
  torch.manual_seed(0)
  mk = m.mask_for(2, 3, torch.device("cpu"))
  assert met["mask_rate"].item() == pytest.approx(mk.float().mean().item())
  with torch.no_grad():
    lg = m.net(torch.where(mk, torch.full_like(fr, MASK), fr)).float()
  ce = torch.nn.functional.cross_entropy(lg.reshape(-1, 1024), fr.reshape(-1), reduction="none").view(2, 3, 128)
  assert torch.allclose(ce[mk].mean(), loss.detach(), atol=1e-5)
  y2 = fr.clone()  # el target de una posición no enmascarada no cuenta; el de una enmascarada sí
  b, f, c = (~mk).nonzero()[0].tolist()
  y2[b, f, c] = (y2[b, f, c] + 1) % 1024
  assert torch.allclose(lm_loss(lg, y2, mask=mk)[0], loss.detach(), atol=1e-6)
  b, f, c = mk.nonzero()[0].tolist()
  y2[b, f, c] = (y2[b, f, c] + 1) % 1024
  assert not torch.allclose(lm_loss(lg, y2, mask=mk)[0], loss.detach(), atol=1e-6)


def test_train_loss_guards():
  with pytest.raises(ValueError):
    tiny().train_loss(rand_frames(1, 2), teacher=object())
  with pytest.raises(ValueError):
    tiny().train_loss(rand_frames(1, T + 1))


def test_temporal_attention_is_causal():
  m = tiny()
  x = rand_frames(1, 4)
  x2 = x.clone()
  x2[0, 2, 7] = MASK  # cambia el frame 2: los frames 0 y 1 no se enteran, el 2 y el 3 sí
  x2[0, 3, 5] = (x2[0, 3, 5] + 1) % 1024
  with torch.no_grad():
    a, b = m.net(x), m.net(x2)
  assert (a[:, :2] - b[:, :2]).abs().max() < 1e-5
  assert (a[:, 2] - b[:, 2]).abs().max() > 1e-4
  assert (a[:, 3] - b[:, 3]).abs().max() > 1e-4


def test_spatial_attention_is_within_frame():
  m = tiny()
  x = rand_frames(1, 3)
  x2 = x.clone()
  x2[0, 1, 100] = MASK
  with torch.no_grad():
    a, b = m.net(x), m.net(x2)
  assert (a[:, 0] - b[:, 0]).abs().max() < 1e-5          # otro frame (pasado): no cambia
  assert (a[:, 1, 3] - b[:, 1, 3]).abs().max() > 1e-4    # misma frame, celda anterior: bidireccional
  assert (a[:, 2, 3] - b[:, 2, 3]).abs().max() > 1e-4    # el futuro sí ve el cambio por la atención temporal


def test_cache_equals_full_pass():
  """El paso con caché temporal (prefill + frame nuevo) da lo mismo que la secuencia entera."""
  m = tiny()
  ctx = rand_frames(2, 3)
  new = rand_frames(2, 1, seed=5)[:, 0].clone()
  new[:, 40:] = MASK
  with torch.no_grad():
    full = m.net(torch.cat([ctx, new[:, None]], 1))[:, -1]
    _, kvs = m.net(ctx, 0, kv=True)
    inc = m._step(new, *[t for kv in kvs for t in kv])
  assert (inc - full).abs().max() < 1e-4


def test_schedule_and_output():
  for S in (1, 2, 5, 25):
    ns = [128] + [n_masked(s, S) for s in range(1, S + 1)]
    assert ns[-1] == 0 and all(a >= b for a, b in zip(ns, ns[1:]))
  m = tiny()
  ctx = rand_frames(3, 3)
  out = m.next_frame(ctx, temperature=1.0, steps=3, generator=torch.Generator().manual_seed(0))
  assert out.shape == (3, 128) and out.dtype == torch.long and 0 <= out.min() and out.max() <= 1023
  assert m.next_frame(rand_frames(1, 1), steps=2).shape == (1, 128)   # contexto corto
  assert m.next_frame(rand_frames(1, 9), steps=2).shape == (1, 128)   # largo: se recorta a T-1 frames
  assert m.next_frame(ctx[:, :0], steps=2).shape == (3, 128)          # sin contexto
  g = lambda s=7: torch.Generator().manual_seed(s)
  assert (m.next_frame(ctx, 1.0, 4, g()) == m.next_frame(ctx, 1.0, 4, g())).all()
  assert (m.next_frame(ctx, 1.0, 4, g()) != m.next_frame(ctx, 1.0, 4, g(8))).any()


def test_temperature_zero_deterministic():
  m = tiny()
  ctx = rand_frames(2, 3)
  a = m.next_frame(ctx, temperature=0, steps=4)
  assert (a == m.next_frame(ctx, temperature=0, steps=4, generator=torch.Generator().manual_seed(1))).all()
  assert (a == m.next_frame(ctx, temperature=0, steps=4)).all()


def test_one_step_is_argmax_of_masked_pass():
  m = tiny()
  ctx = rand_frames(2, 3)
  out = m.next_frame(ctx, temperature=0, steps=1)
  lg = m.last_frame_logits(torch.cat([ctx, torch.zeros_like(ctx[:, :1])], 1))
  assert (out == lg.argmax(-1)).all()


def test_pass_count():
  m = tiny()
  calls = []
  m.net.register_forward_hook(lambda mod, a, o: calls.append(a[0].shape[1]))
  m.next_frame(rand_frames(2, 3), steps=5)
  assert calls == [3] + [1] * 5   # un prefill del contexto y S pasos de un solo frame


@pytest.mark.parametrize("temperature", [0.0, 1.0, 2.0])
def test_stream_equals_generate(temperature):
  m = tiny()
  ctx = rand_frames(2, T - 1)
  ref = m.rollout(ctx, 4, temperature=temperature, steps=3, generator=torch.Generator().manual_seed(0))
  s = make_stream(m, temperature=temperature, steps=3, seed=0, graphs=False)
  assert isinstance(s, GenieStream)
  for f in range(ctx.shape[1]):
    s.observe(ctx[:, f])
  assert torch.equal(torch.stack(list(s.imagine(4)), 1), ref)


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_stream_fused_frame_equals_next_frame(temperature):
  m = tiny()
  s = GenieStream(m, temperature=temperature, steps=4, seed=7, graphs=False)
  ctx = rand_frames(2, T - 1)
  got = s._frame(ctx, *s._noise(2))
  assert torch.equal(got, m.next_frame(ctx, temperature, 4, torch.Generator().manual_seed(7)))


def test_overfit():
  m = tiny(frames=3).train()
  fr = rand_frames(4, 3)
  opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
  first = None
  for _ in range(100):
    loss, _ = m.train_loss(fr)
    opt.zero_grad()
    loss.backward()
    opt.step()
    first = first if first is not None else loss.item()
  assert loss.item() < 0.5 * first


def test_registry_and_fidelidad():
  for size in "sm":
    m = build("genie", size=size)
    print("params", size, sum(p.numel() for p in m.parameters()))
  m = load_model("random:genie_s", "cpu")
  assert m.name == "random_genie_s" and not m.has_likelihood and m.ctx_frames == 15
  assert all(set(r) == {"campo", "fuente_valor", "aquí", "categoría", "fuente"} for r in FIDELIDAD)
  assert {r["categoría"] for r in FIDELIDAD} <= {"fiel", "datos", "presupuesto", "propio"}


def test_no_host_roundtrips(monkeypatch):
  """train_loss y next_frame no sincronizan con el host (.item/.cpu/.tolist/.numpy)."""
  m = tiny().train()
  fr = rand_frames(2, 3)

  def boom(*a, **k):
    raise AssertionError("viaje GPU->CPU")
  for n in ("item", "cpu", "tolist", "numpy", "__bool__", "__int__", "__float__"):
    monkeypatch.setattr(torch.Tensor, n, boom)
  loss, _ = m.train_loss(fr)
  loss.backward()
  m.eval()
  m.next_frame(fr, temperature=2.0, steps=3, generator=torch.Generator().manual_seed(0))
  m.next_frame(fr, temperature=0, steps=3)
