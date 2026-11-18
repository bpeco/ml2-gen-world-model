"""Embedding de tubelets 3D de V-JEPA 2 y normalización de la entrada.

Fuente: github.com/facebookresearch/vjepa2 (main).
  - PatchEmbed3D: src/models/utils/patch_embed.py L26-52 (Conv3d con kernel = stride = (tubelet, parche, parche),
    con bias, flatten(2).transpose(1, 2) -> (B, N, D); el orden de los tokens es t-mayor, luego h, luego w).
  - Normalización: app/vjepa/transforms.py L21 y L48 (media/std de ImageNet), L139-152 (`_tensor_normalize_inplace`:
    sin autoaumento, mean y std se multiplican por 255 y se aplican sobre uint8 -> equivale a x/255, (x-mean)/std).
  - Ruta de un solo frame de V-JEPA 2-AC: app/vjepa_droid/train.py L408-412 (`unsqueeze(2).repeat(1, 1, 2, 1, 1)`).
"""
from functools import lru_cache

import torch
import torch.nn as nn

IMAGENET_MEDIA = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PatchEmbed3D(nn.Module):
  """(B, 3, T, H, W) -> (B, (T/tubelet)*(H/parche)*(W/parche), D)."""

  def __init__(self, patch_size=16, tubelet_size=2, in_chans=3, embed_dim=768):
    super().__init__()
    self.patch_size = patch_size
    self.tubelet_size = tubelet_size
    self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(tubelet_size, patch_size, patch_size),
                          stride=(tubelet_size, patch_size, patch_size))

  def forward(self, x):
    return self.proj(x).flatten(2).transpose(1, 2)


@lru_cache(maxsize=None)
def _media_std(device):
  """mean/std (3, 1, 1, 1) en `device`, una vez por dispositivo (sin copias host->device por paso)."""
  f = lambda v: torch.tensor(v, device=device).view(3, 1, 1, 1)
  return f(IMAGENET_MEDIA), f(IMAGENET_STD)


def normalizar(frames):
  """Frames RGB uint8 (..., T, H, W, 3) -> float normalizado con ImageNet, (..., 3, T, H, W).

  Acepta (T, H, W, 3) o (B, T, H, W, 3). Un float ya en [0, 1] se normaliza igual (sin dividir por 255).
  """
  x = frames.float() / 255.0 if frames.dtype == torch.uint8 else frames.float()
  mean, std = _media_std(x.device)
  x = x.movedim(-1, -4)  # (..., 3, T, H, W)
  return (x - mean) / std


def normalizar_cthw(x01):
  """Ya en [0, 1] y (B, 3, T, H, W) -> media/std de ImageNet (mismo orden que `normalizar`)."""
  mean, std = _media_std(x01.device)
  return (x01 - mean) / std


def frame_a_tubelet(x, tubelet_size=2):
  """Ruta de un frame (V-JEPA 2-AC): (B, 3, H, W) -> (B, 3, tubelet, H, W) duplicando el frame.

  train.py L410: c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1).
  """
  return x.unsqueeze(2).repeat(1, 1, tubelet_size, 1, 1)


def clip_a_frames(x, tubelet_size=2):
  """(B, 3, T, H, W) -> (B*T, 3, tubelet, H, W): cada frame del clip como un tubelet duplicado (AC, L410)."""
  return frame_a_tubelet(x.permute(0, 2, 1, 3, 4).flatten(0, 1), tubelet_size)
