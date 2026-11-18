"""Ventanas de frames de train, deterministas por (seed, step) para poder reanudar y comparar corridas."""
import json
from pathlib import Path

import numpy as np
import torch

from src.data import route_of


class Windows:
  """Tokens de train (N, T, 128). Los últimos `holdout` segmentos quedan aparte para monitorear.

  poses (opcional): (N, T, 6) del mismo shard y en el mismo orden (data/train_poses.npy). Con
  with_poses=True, sample() devuelve también las poses de los mismos frames, crudas (con NaN); el
  modelo las normaliza con pose_stats() y trata los NaN como acción nula.

  frameskip k: se toma 1 de cada k frames crudos (t, t+k, ...). La acción de cada frame tomado son las k
  poses crudas desde ese frame hasta antes del siguiente tomado, apiladas (6·k dimensiones, como en LeWM).
  Con k = 1 todo es idéntico a antes.
  """

  def __init__(self, tokens="data/train_tokens.npy", holdout: int = 50, seed: int = 0, poses=None):
    self.tokens = tokens if isinstance(tokens, np.ndarray) else np.load(tokens)
    self.n_train = len(self.tokens) - holdout
    assert self.n_train > 0, "holdout más grande que el dataset"
    self.holdout, self.seed = holdout, seed
    self.poses = None
    if poses is not None:
      self.poses = poses if isinstance(poses, np.ndarray) else np.load(poses)
      assert self.poses.shape[:2] == self.tokens.shape[:2], "poses y tokens no coinciden"

  def _gather(self, segs, starts, n_frames, k=1):
    out = np.stack([self.tokens[s, t:t + (n_frames - 1) * k + 1:k] for s, t in zip(segs, starts)])
    return torch.from_numpy(out.astype(np.int64))

  def _gather_poses(self, segs, starts, n_frames, k=1):
    out = np.stack([self.poses[s, t:t + n_frames * k].reshape(n_frames, -1) for s, t in zip(segs, starts)])
    return torch.from_numpy(out.astype(np.float32))

  def _draw(self, rng, n, lo, hi, n_frames, k=1):
    T = self.tokens.shape[1]
    span = n_frames * k  # frames crudos que cubre la ventana (incluye las k poses del último frame)
    return rng.integers(lo, hi, n), rng.integers(0, T - span + 1, n)

  def sample(self, step: int, batch_size: int, n_frames: int = 20, with_poses: bool = False, frameskip: int = 1):
    """(B, n_frames, 128) LongTensor en CPU; depende solo de (seed, step).

    with_poses: devuelve (tokens, poses (B, n_frames, 6·frameskip)) con los mismos segmentos y frames; los
    tokens son idénticos a los de with_poses=False (mismas extracciones del RNG).
    """
    rng = np.random.default_rng([self.seed, step])
    segs, starts = self._draw(rng, batch_size, 0, self.n_train, n_frames, frameskip)
    x = self._gather(segs, starts, n_frames, frameskip)
    pin = torch.cuda.is_available()
    x = x.pin_memory() if pin else x
    if not with_poses:
      return x
    assert self.poses is not None, "Windows sin poses"
    p = self._gather_poses(segs, starts, n_frames, frameskip)
    return x, (p.pin_memory() if pin else p)

  def monitor(self, n_windows: int = 32, n_frames: int = 20, frameskip: int = 1) -> torch.Tensor:
    """Ventanas fijas del holdout (seed fija, no depende de self.seed)."""
    rng = np.random.default_rng([12345, 0])
    n_frames = min(n_frames, self.tokens.shape[1] // frameskip)  # con frameskip la ventana cubre n_frames * k crudos
    segs, starts = self._draw(rng, n_windows, self.n_train, len(self.tokens), n_frames, frameskip)
    return self._gather(segs, starts, n_frames, frameskip)

  def pose_stats(self, frameskip: int = 1):
    """(media, desvío) por dimensión sobre los segmentos de train (sin holdout), ignorando NaN.

    Con frameskip k se repiten k veces (las acciones apiladas tienen 6·k dimensiones)."""
    assert self.poses is not None, "Windows sin poses"
    p = self.poses[:self.n_train].reshape(-1, self.poses.shape[-1])
    mean, std = np.nanmean(p, 0).astype(np.float32), np.maximum(np.nanstd(p, 0), 1e-6).astype(np.float32)
    return np.tile(mean, frameskip), np.tile(std, frameskip)


def disjoint_holdout(keys="data/train_keys.json", holdout: int = 50):
  """Índices de los segmentos del holdout de Windows cuya ruta no aparece en los de train.

  El holdout de Windows son los últimos `holdout` segmentos del shard; algunos pueden compartir ruta
  (mismo viaje) con segmentos de train. Para probes y planificación se usan solo los de rutas nuevas.
  """
  keys = keys if isinstance(keys, list) else json.loads(Path(keys).read_text())
  n_train = len(keys) - holdout
  train_routes = {route_of(k) for k in keys[:n_train]}
  return [i for i in range(n_train, len(keys)) if route_of(keys[i]) not in train_routes]
