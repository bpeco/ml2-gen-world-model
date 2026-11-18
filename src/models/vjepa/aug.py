"""Aumentos de V-JEPA 2 en la GPU (random-resized-crop + flip horizontal), sin lecturas a la CPU.

Fuente: configs/train/vitl16/pretrain-256px-16f.yaml L29-38 (random_resize_scale [0.3, 1], random_resize_aspect_ratio
[0.75, 1.35], sin autoaugment, sin motion shift, reprob 0; el flip entra por `random_horizontal_flip` por defecto,
app/vjepa/transforms.py L48-57) y src/datasets/utils/video/transforms.py L470-543:
  - `_get_param_spatial_crop` (L470-508): hasta 10 intentos; área = U(scale)*H*W; razón de aspecto log-uniforme sobre
    [ratio]; w = round(sqrt(area*r)), h = round(sqrt(area/r)); se acepta el primero con 0 < w <= W y 0 < h <= H, con
    posición uniforme; si ninguno cabe, recorte central según la razón de la imagen (L511-523).
  - `random_resized_crop` (L525-543): el MISMO recorte para todo el clip, y F.interpolate bilineal (align_corners=False,
    sin antialias) al tamaño de salida.
  - flip horizontal con p = 0.5, el mismo para todo el clip (app/vjepa/transforms.py L108-109).
Aquí: los 10 intentos se sortean juntos y se elige el primero válido (argmax), el recorte y el remuestreo bilineal
son un solo `grid_sample` (equivalente a recortar y a interpolar con align_corners=False); sin Python por muestra.
Diferencia declarada (datos): el oficial recorta fuentes ~16:9 a un cuadrado, donde la razón mide el recorte en píxeles
de la fuente; aquí la fuente ya es 1:2 y la salida también, así que la razón se expresa respecto del aspecto de la
fuente (w/h = r * W/H): r = 1 significa "el recorte conserva el aspecto del frame".
"""
import math

import torch
import torch.nn.functional as F

INTENTOS = 10


def parametros_recorte(B, H, W, scale=(0.3, 1.0), ratio=(0.75, 1.35), device="cpu", generator=None):
  """(i, j, h, w) float por muestra, forma (B,), del recorte (L470-508)."""
  asp = W / H
  r = torch.rand(B, INTENTOS, 4, device=device, generator=generator, dtype=torch.float64)
  area = (scale[0] + r[..., 0] * (scale[1] - scale[0])) * H * W
  lr = math.log(ratio[0] * asp), math.log(ratio[1] * asp)
  ar = torch.exp(lr[0] + r[..., 1] * (lr[1] - lr[0]))
  w = torch.round(torch.sqrt(area * ar))
  h = torch.round(torch.sqrt(area / ar))
  ok = (w > 0) & (w <= W) & (h > 0) & (h <= H)
  k = ok.int().argmax(1)  # primer intento válido (0 si ninguno: se corrige con el recorte central)
  g = lambda t: t.gather(1, k[:, None])[:, 0]
  w, h, u_i, u_j = g(w), g(h), g(r[..., 2]), g(r[..., 3])
  # recorte central (L511-523)
  lo, hi = ratio[0] * asp, ratio[1] * asp
  if asp < lo:
    fw, fh = W, round(W / lo)
  elif asp > hi:
    fh, fw = H, round(H * hi)
  else:
    fw, fh = W, H
  has = ok.any(1)
  w = torch.where(has, w, torch.full_like(w, fw))
  h = torch.where(has, h, torch.full_like(h, fh))
  i = torch.where(has, torch.floor(u_i * (H - h + 1)), torch.full_like(h, (H - fh) // 2))
  j = torch.where(has, torch.floor(u_j * (W - w + 1)), torch.full_like(h, (W - fw) // 2))
  return i, j, h, w


def recortar_y_voltear(x, scale=(0.3, 1.0), ratio=(0.75, 1.35), flip=True, size=None, generator=None):
  """x (B, C, H, W) float (los frames del clip apilados en C) -> (B, C, *size). Mismo recorte y flip para todo C."""
  B, C, H, W = x.shape
  Ho, Wo = size or (H, W)
  i, j, h, w = (t.to(x.dtype) for t in parametros_recorte(B, H, W, scale, ratio, x.device, generator))
  sgn = torch.ones(B, dtype=x.dtype, device=x.device)
  if flip:
    sgn = torch.where(torch.rand(B, device=x.device, generator=generator) < 0.5, -sgn, sgn)
  theta = torch.zeros(B, 2, 3, dtype=x.dtype, device=x.device)
  theta[:, 0, 0] = sgn * w / W
  theta[:, 0, 2] = (2 * j + w) / W - 1
  theta[:, 1, 1] = h / H
  theta[:, 1, 2] = (2 * i + h) / H - 1
  grid = F.affine_grid(theta, (B, C, Ho, Wo), align_corners=False)
  return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
