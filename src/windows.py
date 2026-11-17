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

  def _gather(self, segs, starts, n_frames):
    out = np.stack([self.tokens[s, t:t + n_frames] for s, t in zip(segs, starts)])
    return torch.from_numpy(out.astype(np.int64))

  def _gather_poses(self, segs, starts, n_frames):
    out = np.stack([self.poses[s, t:t + n_frames] for s, t in zip(segs, starts)])
    return torch.from_numpy(out.astype(np.float32))

  def _draw(self, rng, n, lo, hi, n_frames):
    T = self.tokens.shape[1]
    return rng.integers(lo, hi, n), rng.integers(0, T - n_frames + 1, n)

  def sample(self, step: int, batch_size: int, n_frames: int = 20, with_poses: bool = False):
    """(B, n_frames, 128) LongTensor en CPU; depende solo de (seed, step).

    with_poses: devuelve (tokens, poses (B, n_frames, 6)) con los mismos segmentos y frames; los
    tokens son idénticos a los de with_poses=False (mismas extracciones del RNG).
    """
    rng = np.random.default_rng([self.seed, step])
    segs, starts = self._draw(rng, batch_size, 0, self.n_train, n_frames)
    x = self._gather(segs, starts, n_frames)
    pin = torch.cuda.is_available()
    x = x.pin_memory() if pin else x
    if not with_poses:
      return x
    assert self.poses is not None, "Windows sin poses"
    p = self._gather_poses(segs, starts, n_frames)
    return x, (p.pin_memory() if pin else p)

  def monitor(self, n_windows: int = 32, n_frames: int = 20) -> torch.Tensor:
    """Ventanas fijas del holdout (seed fija, no depende de self.seed)."""
    rng = np.random.default_rng([12345, 0])
    segs, starts = self._draw(rng, n_windows, self.n_train, len(self.tokens), n_frames)
    return self._gather(segs, starts, n_frames)

  def pose_stats(self):
    """(media, desvío) por dimensión sobre los segmentos de train (sin holdout), ignorando NaN."""
    assert self.poses is not None, "Windows sin poses"
    p = self.poses[:self.n_train].reshape(-1, self.poses.shape[-1])
    return np.nanmean(p, 0).astype(np.float32), np.maximum(np.nanstd(p, 0), 1e-6).astype(np.float32)


def disjoint_holdout(keys="data/train_keys.json", holdout: int = 50):
  """Índices de los segmentos del holdout de Windows cuya ruta no aparece en los de train.

  El holdout de Windows son los últimos `holdout` segmentos del shard; algunos pueden compartir ruta
  (mismo viaje) con segmentos de train. Para probes y planificación se usan solo los de rutas nuevas.
  """
  keys = keys if isinstance(keys, list) else json.loads(Path(keys).read_text())
  n_train = len(keys) - holdout
  train_routes = {route_of(k) for k in keys[:n_train]}
  return [i for i in range(n_train, len(keys)) if route_of(keys[i]) not in train_routes]
