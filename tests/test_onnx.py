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


def tiny_genie():
  from src.models.genie import Genie
  torch.manual_seed(0)
  m = Genie(Genie.Config(n_layer=2, n_head=4, dim=64, frames=4)).eval()
  for p in m.parameters():
    p.data.mul_(4.0)
  return m


def test_paso_de_genie_ort_vs_torch(tmp_path):
  from src.export_onnx import genie_paso
  net, path, feed = genie_paso(tiny_genie(), 4, tmp_path / "paso.onnx")
  onnx.checker.check_model(onnx.load(str(path)))
  with torch.no_grad(), explicit_attention():
    ref = net._step(*feed)
  sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
  assert [i.name for i in sess.get_inputs()][:3] == ["cur", "k0", "v0"] and len(sess.get_inputs()) == 1 + 2 * 2
  lg = torch.from_numpy(sess.run(None, {i.name: t.numpy() for i, t in zip(sess.get_inputs(), feed)})[0])
  assert lg.shape == (1, 128, 1024) and (lg - ref).abs().max() < 1e-3 and torch.equal(lg.argmax(-1), ref.argmax(-1))


def test_ort_model_mide_un_paso_de_genie(tmp_path, monkeypatch):
  import json
  from types import SimpleNamespace
  from src import ort_model
  from src.export_onnx import genie_paso
  monkeypatch.setattr(ort_model, "OUT", tmp_path)
  _, path, _ = genie_paso(tiny_genie(), 4, tmp_path / "paso.onnx")
  ort_model.main_paso(SimpleNamespace(onnx=path, provider="cpu", n=3, warmup=1), ort_model.make_session(path, "cpu"))
  r = json.loads((tmp_path / "ort_paso__cpu.json").read_text())
  assert r["unit"] == "paso" and r["median_ms"] > 0 and r["contexto"] == [3]
  assert r["vs_cpu"]["acuerdo_argmax"] == 1.0 and r["vs_cpu"]["max_abs_diff"] == 0.0


def test_prefill_de_genie_alimenta_al_paso(tmp_path):
  """El prefill exportado produce el (k, v) que consume el paso exportado: encadenados en ORT dan lo mismo que PyTorch."""
  from src.export_onnx import GeniePrefill, genie_paso, genie_prefill
  m = tiny_genie()
  net, pre_path, (ctx,) = genie_prefill(m, 4, tmp_path / "prefill.onnx")
  _, paso_path, (cur, *past) = genie_paso(m, 4, tmp_path / "paso.onnx")
  with torch.no_grad(), explicit_attention():
    ref_kv = GeniePrefill(net)(ctx)
    ref = net._step(cur, *past)
  pre = ort.InferenceSession(str(pre_path), providers=["CPUExecutionProvider"])
  assert [i.name for i in pre.get_inputs()] == ["ctx"] and [o.name for o in pre.get_outputs()] == ["k0", "v0", "k1", "v1"]
  kv = pre.run(None, {"ctx": ctx.numpy()})
  assert len(kv) == len(ref_kv) == 4 and kv[0].shape == (128, 4, 3, 16)   # (B*N, H, K-1, hd)
  assert max(abs(torch.from_numpy(a) - b).max() for a, b in zip(kv, ref_kv)) < 1e-3
  paso = ort.InferenceSession(str(paso_path), providers=["CPUExecutionProvider"])
  lg = torch.from_numpy(paso.run(None, {"cur": cur.numpy(), **{o.name: a for o, a in zip(pre.get_outputs(), kv)}})[0])
  assert (lg - ref).abs().max() < 1e-3 and torch.equal(lg.argmax(-1), ref.argmax(-1))


def test_ort_model_mide_el_prefill_de_genie(tmp_path, monkeypatch):
  import json
  from types import SimpleNamespace
  from src import ort_model
  from src.export_onnx import genie_prefill
  monkeypatch.setattr(ort_model, "OUT", tmp_path)
  _, path, _ = genie_prefill(tiny_genie(), 4, tmp_path / "prefill.onnx")
  ort_model.main_paso(SimpleNamespace(onnx=path, provider="cpu", n=3, warmup=1), ort_model.make_session(path, "cpu"))
  r = json.loads((tmp_path / "ort_prefill__cpu.json").read_text())
  assert r["unit"] == "prefill" and r["contexto"] == [3] and r["median_ms"] > 0
  assert r["vs_cpu"]["acuerdo_argmax"] is None and r["vs_cpu"]["rel_l2"] == 0.0


def test_needs_enough_context(exported):
  with pytest.raises(ValueError):
    OrtMaskGIT(exported[1]).next_frame(window()[:, :2], temperature=0)
