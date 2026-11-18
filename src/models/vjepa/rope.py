"""3D-RoPE de V-JEPA 2 (rotación de q y k por posición temporal, vertical y horizontal).

Fuente: src/models/utils/modules.py, clase RoPEAttention (la de V-JEPA 2 de vídeo, no la ACRoPEAttention).
  - L290-292: d_dim = h_dim = w_dim = 2*((head_dim//3)//2); L363-370: se rotan los tres trozos por separado
    ([t | h | w]) y el resto de head_dim (head_dim - 3*d_dim) queda sin rotar y se concatena al final.
  - L296-329 `separate_positions`: de un id plano (t-mayor, luego h, luego w) salen frame = id // (H*W),
    h = (id % (H*W)) // W, w = el resto. Los ids vienen de `mask` (los índices de los tokens que se conservan)
    o de arange(T*H*W) si no hay máscara: la posición RoPE de un token es la de su lugar original en la rejilla
    completa, no su lugar en la secuencia recortada.
  - L26-55 `rotate_queries_or_keys`: omega_j = 1/10000^(j/(D/2)), j = 0..D/2-1; ángulo = pos * omega; pares de
    elementos contiguos (y1, y2) -> (-y2, y1); salida = x*cos + rot(x)*sin.

Particularidad de la versión oficial (L39-46): sin y cos se expanden con `.repeat(1, 1, 1, 2)` (las D/2 frecuencias
se *concatenan* dos veces), no con `repeat_interleave(2)`. Así el par (x_{2p}, x_{2p+1}) se gira con dos ángulos
distintos (omega_{2p mod D/2} y omega_{2p+1 mod D/2}) y la transformación deja de ser una rotación: el producto q·k
ya no depende solo de la posición relativa. El propio código lo marca como bug conocido y dice que corregirlo rompe
la compatibilidad con los checkpoints publicados. Aquí `oficial=True` (por defecto) reproduce lo publicado y
`oficial=False` aplica la corrección (repeat_interleave), que sí es invariante a desplazamientos.
"""
from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def _omega(D, dtype, device):
  """omega_j = 1/10000^(j/(D/2)) (L30-33), una vez por (D, dtype, device): sin copias host->device por paso."""
  omega = torch.arange(D // 2, dtype=dtype, device=device) / (D / 2.0)
  return 1.0 / 10000 ** omega


@lru_cache(maxsize=None)
def ids_rejilla(T, H, W, device):
  """arange(T*H*W) en `device` (L342-345, caso sin máscara), cacheado."""
  return torch.arange(T * H * W, device=device)


def rotar(x, pos, oficial=True):
  """x: (B, h, N, D) con D par; pos: (B, N) o (N,) float. Rota los pares contiguos de x según pos."""
  D = x.shape[-1]
  assert D % 2 == 0, "D debe ser par"
  omega = _omega(D, x.dtype, x.device)  # (D/2,)
  freq = pos.to(x.dtype)[..., None] * omega  # (..., N, D/2)
  sin, cos = freq.sin(), freq.cos()
  if oficial:  # modules.py L43-44
    sin, cos = torch.cat([sin, sin], -1), torch.cat([cos, cos], -1)
  else:  # modules.py L45-46 (la corrección propuesta en el propio archivo)
    sin, cos = sin.repeat_interleave(2, -1), cos.repeat_interleave(2, -1)
  if pos.ndim == 2:  # (B, N, D) -> (B, 1, N, D) para difundir sobre las cabezas
    sin, cos = sin[:, None], cos[:, None]
  y1, y2 = x.unflatten(-1, (-1, 2)).unbind(-1)
  y = torch.stack((-y2, y1), -1).flatten(-2)
  return x * cos + y * sin


def dim_rope(head_dim):
  """Tamaño de cada uno de los tres trozos rotados (t, h, w): 2*((head_dim//3)//2)."""
  return 2 * ((head_dim // 3) // 2)


def coords_desde_ids(ids, H, W):
  """ids planos (..., N) long de una rejilla (T, H, W) t-mayor -> (t, h, w) float, cada uno (..., N)."""
  por_frame = H * W
  t = ids // por_frame
  r = ids - t * por_frame
  h = r // W
  w = r - h * W
  return t.float(), h.float(), w.float()


def rope_3d(q, k, coords, oficial=True):
  """Aplica 3D-RoPE a q y k (B, h, N, hd). coords = (t, h, w), cada uno (B, N) float.

  Rota [0:d), [d:2d), [2d:3d) con t, h, w (d = dim_rope(hd)); el resto de hd queda igual (L363-370).
  """
  hd = q.shape[-1]
  d = dim_rope(hd)
  salida = []
  for x in (q, k):
    partes = [rotar(x[..., i * d:(i + 1) * d], c, oficial) for i, c in enumerate(coords)]
    if 3 * d < hd:
      partes.append(x[..., 3 * d:])
    salida.append(torch.cat(partes, -1))
  return salida[0], salida[1]
