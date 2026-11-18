"""Exporta a ONNX las piezas de la cadena para correrlas con ONNX Runtime en la NPU o la iGPU (plan, fase 7).

  python -m src.infer.export_codec [--out checkpoints/onnx] [--convlstm convlstm] [--fp16]

Piezas (todas con batch 1 y forma fija):
- codec_encoder_body.onnx: imagen float (1, 3, 128, 256) RGB en [0, 255] -> latentes antes del cuantizador
  (1, 128, 256). El argmin contra el codebook va en la CPU (numpy, src/infer/bench_ort.py): cruza 128 KB y la
  NPU no suele soportar ArgMin con esas formas.
- codec_decoder_body.onnx: embeddings del codebook (1, 128, 256) (el lookup va en la CPU) -> imagen float
  (1, 3, 128, 256) en [0, 255], sin recortar.
- <corrida>_step.onnx: un paso de ConvLSTM, entrada x (1, emb, 8, 16) y (h, c) por capa -> logits (1, 128, 1024) y
  el estado nuevo. x es el embedding del frame (lookup en la CPU con <corrida>_tok_emb.npy) o, en el primer paso,
  el embedding aprendido de inicio (<corrida>_start.npy). MaskGIT ya tiene su export (src/export_onnx.py).
Cada pieza se valida contra PyTorch con ONNX Runtime en la CPU. --fp16 convierte los pesos a fp16 (entradas y
salidas quedan en float32).
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.infer.codec import ensure_weights, load_decoder, load_encoder
from src.infer.stream import convlstm_step

OUT = Path("checkpoints/onnx")


class _PassQuant(nn.Module):
  """En lugar del cuantizador: el encoder devuelve los latentes y el decoder recibe embeddings."""

  def forward(self, h):
    return None, h

  def decode(self, z):
    return z, None


class EncoderBody(nn.Module):
  def __init__(self):
    super().__init__()
    self.enc = load_encoder("cpu")
    self.codebook = self.enc.quantize.cb.clone()
    self.enc.quantize = _PassQuant()

  def forward(self, img):           # (1, 3, 128, 256) float [0, 255] -> (1, 128, 256)
    return self.enc(img)


class DecoderBody(nn.Module):
  def __init__(self):
    super().__init__()
    self.dec = load_decoder("cpu")
    self.codebook = self.dec.quantize.cb.clone()
    self.dec.quantize = _PassQuant()

  def forward(self, z):             # (1, 128, 256) -> (1, 3, 128, 256) float [0, 255]
    return self.dec(z)


class ConvLSTMStep(nn.Module):
  def __init__(self, model):
    super().__init__()
    self.m = model

  def forward(self, x, *state):     # x (1, emb, 8, 16)
    return convlstm_step(self.m, x, *state)


def export(module, inputs, names_in, names_out, path, fp16=False):
  path.parent.mkdir(parents=True, exist_ok=True)
  module = module.eval()
  torch.onnx.export(module, inputs, str(path), input_names=names_in, output_names=names_out, opset_version=17,
                    dynamo=False)
  if fp16:
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    m = convert_float_to_float16(onnx.load(str(path)), keep_io_types=True)
    onnx.save(m, str(path))
  return path


def check(module, inputs, path, names_in, atol):
  import onnxruntime as ort
  sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
  with torch.no_grad():
    ref = module(*inputs)
  ref = ref if isinstance(ref, tuple) else (ref,)
  got = sess.run(None, {n: x.numpy() for n, x in zip(names_in, inputs)})
  err = max(float(np.abs(g - r.numpy()).max()) for g, r in zip(got, ref))
  print(f"  {path.name}: error máximo ORT (CPU) vs PyTorch {err:.2e}", flush=True)
  assert err < atol, f"{path.name}: {err} >= {atol}"
  return err


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default=str(OUT))
  ap.add_argument("--convlstm", default="convlstm", help="spec de load_model de un ConvLSTM (o 'none')")
  ap.add_argument("--fp16", action="store_true")
  a = ap.parse_args()
  out = Path(a.out)
  sfx = "_fp16" if a.fp16 else ""
  ensure_weights()
  torch.manual_seed(0)

  enc = EncoderBody()
  img = torch.rand(1, 3, 128, 256) * 255
  p = export(enc, (img,), ["img"], ["latents"], out / f"codec_encoder_body{sfx}.onnx", a.fp16)
  check(enc, (img,), p, ["img"], atol=0.5 if a.fp16 else 1e-2)
  np.save(out / "codec_codebook.npy", enc.codebook.numpy())

  dec = DecoderBody()
  z = dec.codebook[torch.randint(0, 1024, (1, 128))]
  p = export(dec, (z,), ["emb"], ["img"], out / f"codec_decoder_body{sfx}.onnx", a.fp16)
  check(dec, (z,), p, ["emb"], atol=2.0 if a.fp16 else 1e-1)

  if a.convlstm != "none":
    from src.models import load_model
    m = load_model(a.convlstm, "cpu").eval()
    step = ConvLSTMStep(m)
    Hd = m.cfg.hidden
    state = [torch.randn(1, Hd, 8, 16) * 0.1 for _ in range(2 * m.cfg.n_layers)]
    x = m._embed(torch.randint(0, 1024, (1, 1, 128)))[:, 0].detach()
    names_in = ["x"] + [f"{k}{l}" for l in range(m.cfg.n_layers) for k in ("h", "c")]
    names_out = ["logits"] + [f"{k}{l}_out" for l in range(m.cfg.n_layers) for k in ("h", "c")]
    p = export(step, (x, *state), names_in, names_out, out / f"{m.name}_step{sfx}.onnx", a.fp16)
    check(step, (x, *state), p, names_in, atol=0.5 if a.fp16 else 1e-3)
    np.save(out / f"{m.name}_start.npy", m.start.detach().numpy())        # (emb, 8, 16): entrada del paso 0
    np.save(out / f"{m.name}_tok_emb.npy", m.tok.weight.detach().numpy())  # (1024, emb)
  print("->", out)


if __name__ == "__main__":
  main()
