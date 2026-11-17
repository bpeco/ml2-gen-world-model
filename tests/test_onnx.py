import pytest
import torch

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from src.common import BOS, MASK, TPF, with_bos
from src.export_onnx import explicit_attention, export
from src.models.maskgit import CODES, MaskGIT, n_masked
from src.ort_model import OrtMaskGIT

K = 5


def tiny():
  torch.manual_seed(0)
  m = MaskGIT(MaskGIT.Config(n_layer=2, n_head=4, dim=64, block_size=K * TPF)).eval()
  for p in m.parameters():  # init pequeña = logits casi planos; los agrandamos
    p.data.mul_(4.0)
  return m


def window(B=1, seed=1):
  g = torch.Generator().manual_seed(seed)
  return torch.randint(0, 1024, (B, K - 1, 128), generator=g)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
  m = tiny()
  path = tmp_path_factory.mktemp("onnx") / "tiny_k5.onnx"
  export(m, K, path)
  return m, path


def test_roundtrip_ort_vs_torch(exported):
  m, path = exported
  onnx.checker.check_model(onnx.load(str(path)))
  x = torch.cat([with_bos(window()), torch.full((1, TPF), MASK)], 1)
  x[:, -TPF] = BOS
  with torch.no_grad(), explicit_attention():
    ref = m.net(x)
  sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
  lg = torch.from_numpy(sess.run(None, {"tokens": x.numpy()})[0])
  assert lg.shape == ref.shape == (1, K * TPF, 1024)
  assert (lg - ref).abs().max() < 1e-3
  assert torch.equal(lg.argmax(-1), ref.argmax(-1))


def torch_full_window(m, ctx, steps):
  """Referencia: S pasadas completas de PyTorch sin caché, temperatura 0, mismo esquema coseno."""
  seq = torch.cat([with_bos(ctx[:, -(K - 1):]), torch.full((1, TPF), MASK)], 1)
  seq[:, -TPF] = BOS
  fixed = torch.zeros(1, CODES, dtype=torch.bool)
  n_prev = CODES
  for s in range(1, steps + 1):
    with torch.no_grad():
      lg = m.net(seq)[:, -CODES:].float()
    samp = lg.argmax(-1)
    lp = torch.log_softmax(lg, -1).gather(-1, samp[..., None])[..., 0].masked_fill(fixed, float("-inf"))
    n_s = n_masked(s, steps)
    if n_prev - n_s > 0:
      fixed.scatter_(1, lp.topk(n_prev - n_s, -1).indices, True)
    seq[:, -CODES:] = torch.where(fixed & (seq[:, -CODES:] == MASK), samp, seq[:, -CODES:])
    n_prev = n_s
  return seq[:, -CODES:]


@pytest.mark.parametrize("steps", [1, 4])
def test_next_frame_matches_full_window_reference(exported, steps):
  m, path = exported
  om = OrtMaskGIT(path)
  assert om.ctx_frames == K - 1 and not om.has_likelihood
  ctx = window()
  got = om.next_frame(ctx, temperature=0, steps=steps)
  assert got.shape == (1, 128)
  assert torch.equal(got, torch_full_window(m, ctx, steps))


def test_matches_cached_maskgit_with_same_context(exported):
  m, path = exported
  ctx = window()
  with torch.no_grad():
    ref = m.next_frame(ctx, temperature=0, steps=4)
  got = OrtMaskGIT(path).next_frame(ctx, temperature=0, steps=4)
  assert (got == ref).float().mean() > 0.99  # puede haber empates a fp32 entre confianzas


def test_needs_enough_context(exported):
  with pytest.raises(ValueError):
    OrtMaskGIT(exported[1]).next_frame(window()[:, :2], temperature=0)
