"""Exporta a ONNX las piezas de la cadena para correrlas con ONNX Runtime en la NPU o la iGPU (plan, fase 7).

  python -m src.infer.export_codec [--out checkpoints/onnx] [--fp16]

Piezas (todas con batch 1 y forma fija):
- codec_encoder_body.onnx: imagen float (1, 3, 128, 256) RGB en [0, 255] -> latentes antes del cuantizador
  (1, 128, 256). El argmin contra el codebook va en la CPU (numpy, src/infer/bench_ort.py): cruza 128 KB y la
  NPU no suele soportar ArgMin con esas formas.
- codec_decoder_body.onnx: embeddings del codebook (1, 128, 256) (el lookup va en la CPU) -> imagen float
  (1, 3, 128, 256) en [0, 255], sin recortar.
  Los modelos de mundo tienen su export aparte (MaskGIT: src/export_onnx.py).
Cada pieza se valida contra PyTorch con ONNX Runtime en la CPU. --fp16 convierte los pesos a fp16 (entradas y
salidas quedan en float32).
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.infer.codec import ensure_weights, load_decoder, load_encoder

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

  print("->", out)


if __name__ == "__main__":
  main()
