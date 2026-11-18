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


@torch.no_grad()
def render(tok, chunk=32):
  """tokens (..., 128) -> imágenes uint8 (..., 128, 256, 3) con el decoder de comma congelado.

  fp16 sin autocast en CUDA, fp32 en CPU; por trozos de `chunk` frames para acotar la memoria.
  """
  from src.infer.codec import Codec
  tok = torch.as_tensor(tok)
  dev = tok.device
  key = (dev.type, dev.index)
  if key not in _RENDER:
    _RENDER[key] = Codec(dev, "fp16" if dev.type == "cuda" else "fp32", graphs=False, encoder=False)
  codec = _RENDER[key]
  flat = tok.reshape(-1, tok.shape[-1]).long()
  with torch.autocast(dev.type, enabled=False):
    out = torch.cat([codec.decode(flat[i:i + chunk]) for i in range(0, len(flat), chunk)])
  return out.view(*tok.shape[:-1], *out.shape[1:])


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
