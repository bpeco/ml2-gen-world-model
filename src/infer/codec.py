"""Encoder y decoder del VQ-VAE de comma, residentes en el device: uint8 (B, 128, 256, 3) <-> tokens (B, 128).

  codec = Codec(dev, dtype="fp16", graphs=True)
  tok = codec.encode(img)    # img uint8 NHWC RGB en el device -> (B, 128) long
  img = codec.decode(tok)    # (B, 128) long -> uint8 NHWC RGB

- Preprocesado del encoder (notebooks/encode.ipynb de commaVQ y encoder.onnx): RGB en [0, 255] como float, NCHW,
  sin normalizar. El recorte y la escala del video de cámara a 128x256 están en src/infer/video.py.
- Salida del decoder: ((h + 1) / 2) * 255, recortada a [0, 255] y truncada a uint8, igual que src.viz.decode
  (así los frames coinciden con los de las métricas de video ya calculadas).
- fp16: las convoluciones y la atención van en fp16, pero el cuantizador queda en fp32. En el encoder, la
  distancia al codebook con la misma fórmula que el original (el argmin es sensible al redondeo); en el decoder,
  un lookup del codebook en fp32 (equivale al one-hot @ codebook del original, sin redondeo) y recién ahí fp16.
- graphs=True: cada etapa se captura en un CUDA graph por tamaño de batch. Lo que devuelven encode y decode es
  el buffer de salida del graph: hay que consumirlo (o clonarlo) antes de la próxima llamada a la misma etapa.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cudagraph import GraphedStep

sys.path.insert(0, "third_party/commavq")
from utils.vqvae import CompressorConfig, Decoder, Encoder  # noqa: E402

ENCODER = Path("data/raw/encoder_pytorch_model.bin")
DECODER = Path("data/raw/decoder_pytorch_model.bin")
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
H, W = 128, 256


def ensure_weights():
  """Baja los pesos del encoder y del decoder si faltan (src.hf_fetch, con sha256)."""
  from src.hf_fetch import fetch
  for p in (ENCODER, DECODER):
    if not p.exists():
      fetch("commaai/commavq-gpt2m", p.name, p.parent)


class _EncQuant(nn.Module):
  """Reemplaza a VectorQuantizer.forward en el encoder: misma distancia que el original, en fp32."""

  def __init__(self, codebook: torch.Tensor):
    super().__init__()
    self.cb = codebook.float()            # atributo común: model.to(fp16) no lo toca
    self.cb2 = self.cb.square().sum(1)

  def forward(self, h):                   # (B, S, C) -> (None, (B, S))
    x = h.float()
    d = x.square().sum(-1, keepdim=True) + self.cb2 - 2 * x @ self.cb.t()
    return None, d.argmin(-1)


class _DecQuant(nn.Module):
  """Reemplaza a VectorQuantizer.decode en el decoder: lookup en fp32 y cast al dtype del decoder."""

  def __init__(self, codebook: torch.Tensor, dtype):
    super().__init__()
    self.cb, self.dtype = codebook.float(), dtype

  def decode(self, idx):                  # (B, S) -> ((B, S, C), (B, S))
    return F.embedding(idx, self.cb).to(self.dtype), idx


def load_encoder(dev="cpu", dtype=torch.float32) -> Encoder:
  e = Encoder(CompressorConfig())
  e.load_state_dict(torch.load(ENCODER, map_location="cpu", weights_only=True))
  cb = e.quantize._embedding.weight.detach().to(dev)
  e = e.eval().to(dev, dtype)
  e.quantize = _EncQuant(cb)
  return e


def load_decoder(dev="cpu", dtype=torch.float32) -> Decoder:
  d = Decoder(CompressorConfig())
  d.load_state_dict(torch.load(DECODER, map_location="cpu", weights_only=True))
  cb = d.quantize._embedding.weight.detach().to(dev)
  d = d.eval().to(dev, dtype)
  d.quantize = _DecQuant(cb, dtype)
  return d


class Codec:
  """Encoder + decoder de comma en un device, con dtype y CUDA graphs opcionales.

  encoder/decoder=False no los carga (p. ej. el modo imaginación no necesita el encoder después del primer frame,
  pero conviene tenerlo igual para medir). graphs solo en CUDA (NVIDIA o ROCm con soporte de hipGraph).
  """

  def __init__(self, dev=None, dtype="fp16", graphs: bool = True, encoder: bool = True, decoder: bool = True):
    self.dev = torch.device(dev) if dev is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.dtype = DTYPES[dtype] if isinstance(dtype, str) else dtype
    self.graphs = graphs and self.dev.type == "cuda"
    ensure_weights()
    self.enc = load_encoder(self.dev, self.dtype) if encoder else None
    self.dec = load_decoder(self.dev, self.dtype) if decoder else None
    self._g = {}

  # ---- pasos puros (sin estado): lo que se captura
  def _encode(self, img):
    x = img.permute(0, 3, 1, 2).to(self.dtype)
    return self.enc(x).view(img.shape[0], -1)

  def _decode(self, tok):
    y = self.dec(tok)
    return y.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()

  def _run(self, key, fn, x):
    if not self.graphs:
      with torch.no_grad():
        return fn(x)
    k = (key, x.shape[0])
    g = self._g.get(k)
    if g is None:
      g = self._g[k] = GraphedStep(fn, x)
    return g(x)

  def encode(self, img: torch.Tensor) -> torch.Tensor:
    """img uint8 (B, 128, 256, 3) RGB en el device -> tokens (B, 128) long."""
    assert img.dtype == torch.uint8 and img.shape[1:] == (H, W, 3), img.shape
    return self._run("enc", self._encode, img)

  def decode(self, tok: torch.Tensor) -> torch.Tensor:
    """tokens (B, 128) long en el device -> uint8 (B, 128, 256, 3) RGB."""
    return self._run("dec", self._decode, tok.long())

  def num_params(self):
    n = lambda m: sum(p.numel() for p in m.parameters()) if m is not None else 0
    return {"encoder": n(self.enc), "decoder": n(self.dec)}
