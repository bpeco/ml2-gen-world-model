"""Exporta UNA pasada de forma fija de un MaskGIT o GPT a ONNX, sin caché KV (extra D8, NPU).

  python -m src.export_onnx --model SPEC --frames K --out checkpoints/onnx/<name>_k<K>.onnx [--fp16]
Entrada: tokens int64 (1, K*129) (K-1 frames de contexto + el frame que se genera); salida: logits
(1, K*129, V). La atención se exporta con ops explícitas (matmul + máscara + softmax), no SDPA.
Después compara ORT (CPU) contra PyTorch sobre una ventana real de val.

Genie (--model random:genie_s): exporta UN paso de decodificación con el contexto de forma fija (K-1 frames; por defecto
los 15 de la config): cur int64 (1, 128) con MASK y el (k, v) temporal de cada capa (128, H, K-1, hd) -> logits
(1, 128, 1024). Es lo que corre S veces por frame; el prefill del contexto y el muestreo quedan en PyTorch. Para medirlo
en la NPU o la iGPU: python -m src.ort_model --onnx <archivo> --provider npu|dml|cpu.
Con --parte prefill exporta en cambio el prefill: ctx int64 (1, K-1, 128) -> el (k, v) temporal de cada capa (128, H, K-1, hd),
que es lo que el paso recibe como contexto. Se mide igual.
"""
import argparse
import contextlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.common import MASK, TPF, with_bos
from src.models import load_model
from src.models.genie import Genie
from src.models.maskgit import MaskGIT

VAL = Path("data/val_tokens.npy")


def manual_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **_):
  """SDPA con ops explícitas: matmul, máscara aditiva y softmax. attn_mask bool: True = puede mirar."""
  scores = q @ k.transpose(-2, -1) * (q.shape[-1] ** -0.5)
  if is_causal:
    attn_mask = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool).tril()
  if attn_mask is not None:
    neg = torch.finfo(scores.dtype).min
    scores = scores + torch.zeros_like(attn_mask, dtype=scores.dtype).masked_fill(~attn_mask, neg)
  return torch.softmax(scores, -1) @ v


@contextlib.contextmanager
def explicit_attention():
  orig = F.scaled_dot_product_attention
  F.scaled_dot_product_attention = manual_sdpa
  try:
    yield
  finally:
    F.scaled_dot_product_attention = orig


class Window(torch.nn.Module):
  """tokens (1, L) -> logits (1, L, V) en fp32, sin caché."""

  def __init__(self, net):
    super().__init__()
    self.net = net

  def forward(self, tokens):
    return self.net(tokens).float()


class GenieStep(torch.nn.Module):
  """cur (1, 128), k0, v0, k1, v1... -> logits (1, 128, 1024) fp32: un paso de Genie contra el contexto en caché."""

  def __init__(self, model):
    super().__init__()
    self.m = model

  def forward(self, cur, *past):
    return self.m._step(cur, *past)


class GeniePrefill(torch.nn.Module):
  """ctx (1, P, 128) -> k0, v0, k1, v1... (128, H, P, hd): el prefill de Genie, que es el contexto en caché de cada paso."""

  def __init__(self, model):
    super().__init__()
    self.m = model

  def forward(self, ctx):
    return tuple(t for kv in self.m.net(ctx, 0, kv=True)[1] for t in kv)


def _contexto(model, K, fp16):
  """(modelo en el dtype pedido, ctx (1, K-1, 128)): frames reales de val, o aleatorios si no hay data/."""
  P = K - 1
  if not 0 < P <= model.ctx_frames:
    raise SystemExit(f"Genie tiene contexto de 1 a {model.ctx_frames} frames, no {P}")
  model = (model.half() if fp16 else model.float()).eval()
  if VAL.exists():
    ctx = torch.as_tensor(np.array(np.load(VAL, mmap_mode="r")[0, 100:100 + P]), dtype=torch.long)[None]
  else:
    ctx = torch.randint(0, 1024, (1, P, 128), generator=torch.Generator().manual_seed(0))
  return model, ctx


def genie_prefill(model, K, out, fp16=False):
  """Escribe el .onnx del prefill de Genie (K-1 frames de contexto) y devuelve (modelo, ruta, entradas de ejemplo)."""
  model, ctx = _contexto(model, K, fp16)
  out = Path(out)
  out.parent.mkdir(parents=True, exist_ok=True)
  names = [f"{kv}{i}" for i in range(len(model.net.blocks)) for kv in "kv"]
  with explicit_attention(), torch.no_grad():
    torch.onnx.export(GeniePrefill(model), (ctx,), str(out), input_names=["ctx"], output_names=names,
                      opset_version=17, dynamo=False)
  return model, out, (ctx,)


def genie_paso(model, K, out, fp16=False):
  """Escribe el .onnx del paso de Genie con K-1 frames de contexto y devuelve (modelo, ruta, entradas de ejemplo)."""
  model, ctx = _contexto(model, K, fp16)
  with torch.no_grad():
    _, kvs = model.net(ctx, 0, kv=True)
  past = [t for kv in kvs for t in kv]
  cur = torch.full((1, 128), MASK, dtype=torch.long)
  names = ["cur"] + [f"{kv}{i}" for i in range(len(kvs)) for kv in "kv"]
  out = Path(out)
  out.parent.mkdir(parents=True, exist_ok=True)
  with explicit_attention(), torch.no_grad():
    torch.onnx.export(GenieStep(model), (cur, *past), str(out), input_names=names, output_names=["logits"],
                      opset_version=17, dynamo=False)
  return model, out, (cur, *past)


def load_window(K, kind):
  """Ventana real de val (1, K*129): K frames con BOS; en MaskGIT el último frame va enmascarado."""
  val = np.load(VAL, mmap_mode="r")
  fr = torch.as_tensor(np.array(val[0, 100:100 + K]), dtype=torch.long)[None]
  seq = with_bos(fr)
  if kind == "maskgit":
    seq[:, -(TPF - 1):] = MASK
  return seq


def export(model, K, out, fp16=False):
  """Escribe el .onnx y devuelve (net, ruta). net es el backbone usado (en dtype de la exportación)."""
  if not hasattr(model, "net"):
    raise SystemExit("solo MaskGIT y GPT (con backbone .net)")
  net = model.net.eval()
  L = K * TPF
  if L > net.cfg.block_size:
    raise SystemExit(f"{K} frames = {L} tokens > block_size {net.cfg.block_size}")
  net = net.half() if fp16 else net.float()
  net.clear_cache()
  dummy = torch.randint(0, 1024, (1, L))
  out = Path(out)
  out.parent.mkdir(parents=True, exist_ok=True)
  with explicit_attention(), torch.no_grad():
    torch.onnx.export(Window(net), (dummy,), str(out), input_names=["tokens"], output_names=["logits"],
                      opset_version=17, dynamo=False)
  return net, out


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True, help="spec de load_model (random:maskgit_s, ruta.pt)")
  ap.add_argument("--frames", type=int, help="K: frames de la ventana (K-1 de contexto + 1 nuevo); default 5, o los de la config en Genie")
  ap.add_argument("--out", help="default: checkpoints/onnx/<name>_k<K>.onnx")
  ap.add_argument("--fp16", action="store_true", help="pesos en fp16 (ORT CPU no lo acelera)")
  ap.add_argument("--parte", choices=["paso", "prefill"], default="paso", help="solo Genie: un paso de decodificación o el prefill del contexto")
  a = ap.parse_args(argv)
  import onnx
  import onnxruntime as ort
  model = load_model(a.model, "cpu").eval()
  if isinstance(model, Genie):
    return main_genie(model, a, onnx, ort)
  a.frames = a.frames or 5
  kind = "maskgit" if isinstance(model, MaskGIT) else "gpt"
  out = a.out or f"checkpoints/onnx/{model.name}_k{a.frames}.onnx"
  net, path = export(model, a.frames, out, a.fp16)
  onnx.checker.check_model(onnx.load(str(path)))
  print(f"{path}: {path.stat().st_size / 2**20:.1f} MB, onnx.checker ok")
  if not VAL.exists():
    print(f"sin {VAL}: no se compara contra PyTorch")
    return path
  x = load_window(a.frames, kind)
  with torch.no_grad():
    ref = net(x).float()
  sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
  lg = torch.from_numpy(sess.run(None, {"tokens": x.numpy()})[0])
  print(f"ORT CPU vs PyTorch: max abs diff {(lg - ref).abs().max():.2e}, "
        f"acuerdo de argmax {(lg.argmax(-1) == ref.argmax(-1)).float().mean() * 100:.2f} %")
  return path


def main_genie(model, a, onnx, ort):
  K = a.frames or model.cfg.frames
  out = a.out or f"checkpoints/onnx/{model.name}_{a.parte}_k{K}.onnx"
  net, path, feed = (genie_prefill if a.parte == "prefill" else genie_paso)(model, K, out, a.fp16)
  onnx.checker.check_model(onnx.load(str(path)))
  print(f"{path}: {path.stat().st_size / 2**20:.1f} MB, onnx.checker ok")
  with explicit_attention(), torch.no_grad():
    ref = (GeniePrefill(net) if a.parte == "prefill" else GenieStep(net))(*feed)
  ref = [ref] if isinstance(ref, torch.Tensor) else list(ref)
  sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
  got = [torch.from_numpy(o) for o in sess.run(None, {i.name: t.numpy() for i, t in zip(sess.get_inputs(), feed)})]
  msg = f"ORT CPU vs PyTorch: max abs diff {max((g - r).abs().max().item() for g, r in zip(got, ref)):.2e}"
  if a.parte == "paso":
    msg += f", acuerdo de argmax {(got[0].argmax(-1) == ref[0].argmax(-1)).float().mean() * 100:.2f} %"
  print(msg)
  return path


if __name__ == "__main__":
  main()
