"""Base común de los world models latentes (hoy LeWorldModel, src/models/lewm.py).

Un latente no emite tokens: predice el embedding del frame siguiente. Esta base solo trae lo que no depende de
la arquitectura: las acciones (poses crudas apiladas, normalizadas con estadísticas que viajan en el checkpoint
como buffers), los CUDA graphs de inferencia, el conteo de parámetros y los helpers de EMA (genéricos, para
V-JEPA 2).

Convenciones:
- Acción del frame f = las k poses crudas desde f hasta antes del siguiente frame tomado (6·k dimensiones, k =
  frameskip; src/windows.py las apila así). Una pose NaN (bordes del segmento) pasa a 0 tras normalizar, como en
  el código oficial de LeWM (`torch.nan_to_num(batch["action"], 0.0)`). poses=None = acción 0 (la media).
- Las subclases definen encode(frames) -> z (B, F, D), predict(z, act_emb) -> ẑ (B, F, D) y action_embed(poses).
"""
import copy
import math

import torch
import torch.nn as nn

from src.models.base import WorldModel
from src.models.cudagraph import GraphedStep


def ema_momentum(step: int, total: int, m0: float, m1: float) -> float:
  """Momento de la EMA: de m0 a m1 con coseno (V-JEPA / I-JEPA)."""
  prog = min(1.0, step / max(1, total))
  return m1 - (m1 - m0) * (math.cos(math.pi * prog) + 1) / 2


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, m: float):
  """target <- m * target + (1 - m) * online. Los buffers (estadísticas de BatchNorm) se copian."""
  for pt, po in zip(target.parameters(), online.parameters()):
    pt.lerp_(po.detach().to(pt.dtype), 1.0 - m)
  for bt, bo in zip(target.buffers(), online.buffers()):
    bt.copy_(bo)


class LatentWM(WorldModel):
  """Base de los latentes: acciones normalizadas, CUDA graphs, parámetros."""

  uses_poses = True
  has_likelihood = False

  def _init_common(self, n_pose: int):
    self.register_buffer("pose_mean", torch.zeros(n_pose))
    self.register_buffer("pose_std", torch.ones(n_pose))
    self._graphs_on = False
    self._graphs = {}

  # ---- a definir por cada familia
  def encode(self, frames):
    raise NotImplementedError

  def predict(self, z_in, acts):
    raise NotImplementedError

  def action_embed(self, poses, B, T, device):
    raise NotImplementedError

  # ---- acciones
  def set_pose_stats(self, mean, std):
    self.pose_mean.copy_(torch.as_tensor(mean, dtype=self.pose_mean.dtype))
    self.pose_std.copy_(torch.as_tensor(std, dtype=self.pose_std.dtype))

  def normalize_poses(self, poses):
    """Poses crudas (..., n_pose) -> normalizadas, con los NaN en 0 (la acción media)."""
    p = poses.float()
    return torch.nan_to_num((p - self.pose_mean.float()) / self.pose_std.float(), nan=0.0, posinf=0.0, neginf=0.0)

  # ---- CUDA graphs
  def use_cuda_graphs(self, enable: bool = True):
    self._graphs_on = enable
    self._graphs = {}

  def _graphed(self, key, fn, *inputs):
    if not (self._graphs_on and inputs[0].device.type == "cuda"):
      return fn(*inputs)
    g = self._graphs.get(key)
    if g is None:
      g = self._graphs[key] = GraphedStep(fn, *inputs)
    return g(*inputs)

  # ---- entrenamiento
  def after_step(self, step: int, total: int):
    if getattr(self, "tgt", None) is not None:
      ema_update(self.tgt, self.encoder, ema_momentum(step, total, *self.cfg.ema))

  def num_params(self) -> int:
    """Parámetros de inferencia: sin el encoder target (solo entrenamiento, V-JEPA 2)."""
    return sum(p.numel() for n, p in self.named_parameters() if not n.startswith("tgt."))
