"""Decoder de visualización de V-JEPA 2-AC: de las representaciones de UN frame a píxeles.

Fuente: V-JEPA 2 paper, decoder de visualización (Fig. 15 y apéndice): ViT feedforward por frame (cada frame se decodifica
solo, sin atención entre frames) sobre las representaciones del encoder congelado (con layer_norm), pérdida MSE contra
los píxeles, entrenado a posteriori sin tocar el encoder ni el predictor. Receta del paper: ViT-L, salida 256x256x3,
AdamW wd 0.1, clip 1.0, batch 1024, 150k pasos, warmup 2000 a 5e-4 y coseno.
Aquí (presupuesto): ViT propio chico (bloques RoPE de vit.py, cada token de entrada -> un parche de 16x16x3), salida
128x256x3; la pérdida es el MSE sobre píxeles normalizados con media/std de ImageNet (el paper no dice el espacio; propio).
"""
import torch.nn as nn

from src.models.vjepa.rope import ids_rejilla
from src.models.vjepa.vit import LN, Block, _init


class FrameDecoder(nn.Module):
  def __init__(self, embed_dim, dim=256, depth=6, heads=4, grid_hw=(8, 16), patch=16):
    super().__init__()
    self.grid_hw, self.patch = grid_hw, patch
    self.inp = nn.Linear(embed_dim, dim)
    self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
    self.norm = LN(dim)
    self.head = nn.Linear(dim, patch * patch * 3)
    _init(self, self.blocks)

  def forward(self, z):
    """z (..., N, E) -> (..., 3, H, W) en el espacio normalizado de entrada del encoder."""
    lead = z.shape[:-2]
    z = z.reshape(-1, *z.shape[-2:])
    H, W = self.grid_hw
    ids = ids_rejilla(1, H, W, z.device)[None]
    x = self.inp(z)
    for b in self.blocks:
      x = b(x, ids, H, W)
    x = self.head(self.norm(x)).view(len(z), H, W, 3, self.patch, self.patch)
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(len(z), 3, H * self.patch, W * self.patch)
    return x.view(*lead, *x.shape[1:])
