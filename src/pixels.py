"""Render compartido de tokens a píxeles con el decoder de comma congelado, y caché en disco para val.

render(tokens) es la "cámara" de LeWM desde píxeles (entrenamiento y evaluación con tokens). No es un submódulo
de ningún modelo: no entra en checkpoints, parámetros ni optimizador.

Caché opcional (memmap uint8 en data/cache/pixels_<nombre>_<hash>.npy; el hash cubre los tokens y el decoder):
  frames(tokens, "val", seg, ini, fin) -> (fin - ini, 128, 256, 3), renderiza el archivo entero la primera vez.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch

CACHE = Path("data/cache")
_RENDER = {}


def _codec(dev):
  from src.infer.codec import Codec
  key = (dev.type, dev.index)
  if key not in _RENDER:
    _RENDER[key] = Codec(dev, "fp16" if dev.type == "cuda" else "fp32", graphs=False, encoder=False)
  return _RENDER[key]


def _chunks(tok, chunk, fn):
  """Aplica fn(uint8 (n, 128, 256, 3)) a cada trozo de frames y concatena, todo en el device de `tok`."""
  tok = torch.as_tensor(tok)
  codec = _codec(tok.device)
  flat = tok.reshape(-1, tok.shape[-1]).long()
  with torch.autocast(tok.device.type, enabled=False):
    out = torch.cat([fn(codec.decode(flat[i:i + chunk])) for i in range(0, len(flat), chunk)])
  return out.view(*tok.shape[:-1], *out.shape[1:])


@torch.no_grad()
def render(tok, chunk=32):
  """tokens (..., 128) -> imágenes uint8 (..., 128, 256, 3) con el decoder de comma congelado.

  fp16 sin autocast en CUDA, fp32 en CPU; por trozos de `chunk` frames para acotar la memoria. La salida queda en
  el device de `tok` (con tokens CUDA no hay .cpu() ni transferencias).
  """
  return _chunks(tok, chunk, lambda y: y)


@torch.no_grad()
def render_float(tok, chunk=32, mean=None, std=None, dtype=torch.float32):
  """Como render pero devuelve float en [0, 1] (o normalizado con mean/std por canal) en el device de `tok`.

  Para entrenar desde píxeles: la conversión a float ocurre en la GPU, trozo a trozo, sin pasar por el host. Se
  conserva el truncado a uint8 del decoder (mismo valor que render()/255) para que train y evaluación vean las
  mismas imágenes. Forma (..., 128, 256, 3) (NHWC). mean/std: 3 valores en escala [0, 1]; pasalos
  como tensores ya en el device (si son listas se suben en cada llamada: una transferencia HtoD por trozo).
  """
  def conv(y):
    y = y.to(dtype) / 255
    if mean is not None:
      y = (y - torch.as_tensor(mean, dtype=dtype, device=y.device)) / torch.as_tensor(std, dtype=dtype, device=y.device)
    return y
  return _chunks(tok, chunk, conv)


def _hash(tokens) -> str:
  """Huella barata: forma, una muestra espaciada del contenido y el tamaño del decoder."""
  from src.infer.codec import DECODER
  t = np.asarray(tokens).reshape(-1)
  h = hashlib.sha1(repr(tuple(np.shape(tokens))).encode())
  h.update(np.ascontiguousarray(t[::4099]).tobytes())
  h.update(str(DECODER.stat().st_size if DECODER.exists() else 0).encode())
  return h.hexdigest()[:10]


def cache_path(name, tokens, cache_dir=CACHE) -> Path:
  return Path(cache_dir) / f"pixels_{name}_{_hash(tokens)}.npy"


def cached(tokens, name, cache_dir=CACHE, chunk=32, dev=None) -> np.ndarray:
  """Render de todo `tokens` (N, T, 128) como memmap uint8 (N, T, 128, 256, 3); lo crea si no existe.

  Ojo con el tamaño: ~98 KB por frame (el val de 100 segmentos pesa ~12 GB)."""
  path = cache_path(name, tokens, cache_dir)
  if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    N, T = tokens.shape[:2]
    dev = torch.device(dev or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(N, T, 128, 256, 3))
    for i in range(N):
      out[i] = render(torch.as_tensor(np.asarray(tokens[i])).to(dev), chunk).cpu().numpy()
    out.flush()
    del out
    tmp.replace(path)
  return np.load(path, mmap_mode="r")


def frames(tokens, name, seg, start, stop, cache_dir=CACHE, **kw) -> np.ndarray:
  """Píxeles uint8 (stop - start, 128, 256, 3) del segmento `seg`, frames [start, stop), desde la caché."""
  return np.asarray(cached(tokens, name, cache_dir, **kw)[seg, start:stop])
