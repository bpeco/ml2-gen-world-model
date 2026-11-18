"""Video de entrada y de salida para la inferencia de punta a punta.

- sample_video(): el video de cámara de ejemplo de commaVQ (third_party/commavq/examples/sample_video_ecamera.hevc,
  1200 frames de 1208x1928 a 20 FPS) recortado y escalado a 128x256 con el mismo transform_img que usó comma
  para tokenizar el dataset. Se guarda en data/sample_video_128x256.npy (uint8, 118 MB) para no decodificar
  el HEVC cada vez. Sus tokens de referencia (los que calculó comma) están en examples/tokens.npy.
- read_video(path): cualquier video propio, llevado a 128x256 (queda fuera de la distribución de entrenamiento:
  otra cámara, otro recorte).
- write_mp4(frames, path): uint8 (N, H, W, 3) RGB -> mp4.
"""
import sys
from pathlib import Path

import numpy as np

COMMAVQ = Path("third_party/commavq")
SAMPLE = COMMAVQ / "examples/sample_video_ecamera.hevc"
SAMPLE_TOKENS = COMMAVQ / "examples/tokens.npy"
CACHE = Path("data/sample_video_128x256.npy")


def _utils():
  sys.path.insert(0, str(COMMAVQ))
  from utils import video
  return video


def read_video(path, limit=None, comma_crop=True) -> np.ndarray:
  """(N, 128, 256, 3) uint8 RGB. comma_crop: recorte de la cámara de comma (transform_img); si no, solo resize."""
  import cv2
  v = _utils()
  cap = cv2.VideoCapture(str(path))
  out = []
  while limit is None or len(out) < limit:
    ok, img = cap.read()
    if not ok:
      break
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out.append(v.transform_img(img) if comma_crop else cv2.resize(img, (256, 128), interpolation=cv2.INTER_AREA))
  cap.release()
  if not out:
    raise RuntimeError(f"no se pudo leer {path}")
  return np.stack(out)


def sample_video() -> np.ndarray:
  if CACHE.exists():
    return np.load(CACHE)
  frames = read_video(SAMPLE)
  CACHE.parent.mkdir(parents=True, exist_ok=True)
  np.save(CACHE, frames)
  return frames


def sample_tokens() -> np.ndarray:
  """(1200, 128) tokens de comma para sample_video()."""
  return np.load(SAMPLE_TOKENS).astype(np.int64)


def write_mp4(frames, path, fps=20, scale=1):
  """uint8 (N, H, W, 3) RGB -> mp4 (mp4v). scale: agrandar con vecino más cercano para que se vea."""
  import cv2
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  h, w = frames.shape[1:3]
  size = (w * scale, h * scale)
  vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
  for f in frames:
    f = cv2.resize(f, size, interpolation=cv2.INTER_NEAREST) if scale != 1 else f
    vw.write(np.ascontiguousarray(f[..., ::-1]))
  vw.release()
  return path
