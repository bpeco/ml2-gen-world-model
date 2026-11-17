import os
import platform

import torch

BOS = 1024
VOCAB = 1025          # 1024 códigos del VQ-VAE + BOS
MASK = 1025           # token de máscara de MaskGIT (solo entrada, fuera del vocabulario de comma)
TPF = 129             # tokens por frame (BOS + 8x16)
CTX_FRAMES = 20       # 1 s a 20 FPS, igual que gpt2m
BLOCK = CTX_FRAMES * TPF
FPS = 20


def device():
  """CUDA (NVIDIA, o AMD con ROCm: misma API torch.cuda), MPS o CPU.

  ML2_DEVICE fuerza uno, por ejemplo ML2_DEVICE=cpu para comparar.
  """
  forced = os.environ.get("ML2_DEVICE")
  if forced:
    return torch.device(forced)
  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def device_info(dev=None) -> dict:
  """Qué hardware se usó, para anotar en cada results/*.json."""
  dev = torch.device(dev) if dev is not None else device()
  info = {"device": dev.type, "torch": torch.__version__, "platform": platform.platform(),
          "cpu": platform.processor()}
  if dev.type == "cuda":
    i = dev.index if dev.index is not None else torch.cuda.current_device()
    info["gpu"] = torch.cuda.get_device_name(i)
    info["backend"] = "rocm" if torch.version.hip else "cuda"
    info["backend_version"] = torch.version.hip or torch.version.cuda
  return info


def log_device(dev=None):
  info = device_info(dev)
  print("device:", info.get("gpu", info["device"]), f"({info.get('backend', info['device'])})", flush=True)
  return info


def with_bos(frames):
  """(..., F, 128) tokens -> (..., F*129) con BOS al inicio de cada frame."""
  bos = torch.full((*frames.shape[:-1], 1), BOS, dtype=frames.dtype, device=frames.device)
  return torch.cat([bos, frames], -1).flatten(-2)


def strip_bos(seq):
  """(..., F*129) secuencia comma -> (..., F, 128) frames. Inversa de with_bos."""
  return seq.unflatten(-1, (-1, TPF))[..., 1:]
