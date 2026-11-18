"""Baselines de conteo: los primeros WorldModel, validan el evaluador.

  unigram: sin contexto, p(y) = frecuencia del código en train (suavizado add-one).
  transicion: p(y | x) con x = código de la misma celda en el frame anterior (tabla 1024x1024 de conteos
           sobre train, suavizada hacia el unigram): una tabla de transición de primer orden por celda. Se llamaba
           "copy"; ese nombre sigue cargando, pero está obsoleto (no copia nada).
  copy_mix / congelado: p(y) = p_copy * 1[y == misma celda del frame anterior] + (1 - p_copy) * unigram(y).
           Con temperatura 0 repite el último frame: es el "frame congelado" (alias congelado).

Uso: python -m src.models.baselines   (ajusta sobre train y guarda checkpoints/baselines.npz)
"""
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from src.common import VOCAB
from src.models.base import FrameModel, WorldModel, sample_logits

STATS = Path("checkpoints/baselines.npz")
TRAIN = "data/train_tokens.npy"
N_CODES = 1024


def fit_stats(train, chunk=50, n_held=100):
  """train (N, F, 128) -> (unigram (1024,), p_copy, tabla de transición log-probs (1024, 1024), alpha).

  Un solo recorrido en chunks. alpha (suavizado de la tabla hacia el unigram) se elige por máxima
  verosimilitud en los últimos n_held segmentos de train (si hay más de 2 * n_held), nunca en val.
  """
  n_held = n_held if len(train) > 2 * n_held else max(len(train) // 5, 1)
  trans = np.zeros(N_CODES * N_CODES, np.int64)  # trans[x * 1024 + y]: x en frame t-1, y en frame t (misma celda)
  held = np.zeros_like(trans)
  counts = np.zeros(N_CODES, np.int64)    # frecuencia de cada código
  matches = np.zeros(N_CODES, np.int64)   # veces que el código repite la celda del frame anterior
  n_pairs = 0
  for i in range(0, len(train), chunk):
    a = np.asarray(train[i:i + chunk]).astype(np.int64)
    counts += np.bincount(a.ravel(), minlength=N_CODES)
    prev, cur = a[:, :-1], a[:, 1:]
    matches += np.bincount(cur[prev == cur], minlength=N_CODES)
    n_pairs += cur.size
    k = min(max(len(train) - n_held - i, 0), len(a))  # segmentos del chunk que no son held-out
    pk = np.bincount((prev[:k] * N_CODES + cur[:k]).ravel(), minlength=N_CODES ** 2)
    pa = np.bincount((prev[k:] * N_CODES + cur[k:]).ravel(), minlength=N_CODES ** 2)
    trans += pk
    held += pa
  uni = (counts + 1) / (counts.sum() + N_CODES)  # add-one
  # máxima verosimilitud de p_copy: acierto -> log(p + (1-p) u_c); fallo -> log((1-p) u_y) (u_y no depende de p)
  n_hit, n_miss = matches.sum(), n_pairs - matches.sum()

  def ll(p):
    return (matches * np.log(p + (1 - p) * uni)).sum() + n_miss * np.log(1 - p)

  grid = np.linspace(1e-3, 1 - 1e-3, 999)
  p = grid[np.argmax([ll(g) for g in grid])]
  fine = np.linspace(max(p - 1e-3, 1e-6), min(p + 1e-3, 1 - 1e-6), 201)
  p_copy = float(fine[np.argmax([ll(g) for g in fine])])
  alpha = fit_alpha(trans, held, uni)
  return uni, p_copy, transition_logp(trans + held, uni, alpha), alpha


def transition_logp(table, uni, a):
  """Conteos (1024*1024,) -> log p(y | x) = log((c[x,y] + a * 1024 * u(y)) / (rowsum[x] + a * 1024))."""
  c = table.reshape(N_CODES, N_CODES).astype(np.float64)
  p = (c + a * N_CODES * uni[None]) / (c.sum(1, keepdims=True) + a * N_CODES)
  return np.log(p).astype(np.float32)


def fit_alpha(train_counts, held_counts, uni, grid=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)):
  """alpha de máxima verosimilitud sobre los conteos de held-out."""
  h = held_counts.reshape(N_CODES, N_CODES)
  ll = [(h * transition_logp(train_counts, uni, a)).sum() for a in grid]
  return float(grid[int(np.argmax(ll))])


class Unigram(WorldModel):
  """Sin contexto: la misma distribución en todas las posiciones (no conoce las posiciones)."""
  name = "unigram"

  def __init__(self, uni):
    super().__init__()
    lp = torch.log(torch.as_tensor(uni, dtype=torch.float32))
    seq = torch.cat([lp + np.log(128 / 129), torch.full((1,), np.log(1 / 129), dtype=torch.float32)])
    self.register_buffer("logp", lp)         # (1024,) log-probs de los códigos
    self.register_buffer("seq_logp", seq)    # (1025,) con BOS = 1/129 y códigos = 128/129 * unigram

  def logits(self, x):
    return self.seq_logp.expand(*x.shape, VOCAB)

  def last_frame_logits(self, frames):
    return self.logp.expand(frames.shape[0], 128, N_CODES)

  @torch.no_grad()
  def next_frame(self, ctx, temperature=1.0, steps=None, generator=None):
    return sample_logits(self.logp.expand(ctx.shape[0], 128, N_CODES), temperature, generator)


class CopyMix(FrameModel):
  """Mezcla de copiar la celda del frame anterior con el unigram. El frame 0 usa solo el unigram."""
  name = "copy_mix"

  def __init__(self, uni, p_copy):
    super().__init__()
    self.register_buffer("logp", torch.log(torch.as_tensor(uni, dtype=torch.float32)))
    self.register_buffer("p_copy", torch.tensor(float(p_copy), dtype=torch.float32))

  def frame_logits(self, frames):
    B, Fr, C = frames.shape
    p, lp = self.p_copy.float(), self.logp.float()
    out = (lp + torch.log1p(-p)).expand(B, Fr, C, N_CODES).clone()   # (1-p) * unigram
    prev = frames[:, :-1]
    at_prev = out[:, 1:].gather(-1, prev[..., None])                  # log((1-p) u[prev])
    hit = torch.logaddexp(at_prev, torch.log(p).expand_as(at_prev))   # log(p + (1-p) u[prev])
    out[:, 1:].scatter_(-1, prev[..., None], hit)
    out[:, 0] = lp                                                    # sin frame previo
    return out


class Transition(FrameModel):
  """p(y | x): x = código de la misma celda en el frame anterior, tabla de conteos de train."""
  name = "transicion"

  def __init__(self, uni, trans_logp):
    super().__init__()
    self.register_buffer("logp", torch.log(torch.as_tensor(uni, dtype=torch.float32)))
    self.register_buffer("table", torch.as_tensor(trans_logp, dtype=torch.float32))  # (1024, 1024)

  def frame_logits(self, frames):
    B, Fr, C = frames.shape
    out = self.logp.float().expand(B, Fr, C, N_CODES).clone()   # frame 0: unigram
    out[:, 1:] = self.table.float()[frames[:, :-1]]
    return out


def save_stats(uni, p_copy, trans_logp, alpha, path=STATS):
  path.parent.mkdir(exist_ok=True)
  np.savez(path, unigram=uni, p_copy=p_copy, trans_logp=trans_logp, alpha=alpha)


BASELINES = ("unigram", "transicion", "copy_mix", "congelado")
DEPRECATED = {"copy": "transicion"}  # nombre viejo de la tabla de transición


def load_baseline(name, dev="cpu"):
  if name in DEPRECATED:
    warnings.warn(f'baseline "{name}" está obsoleto: usar "{DEPRECATED[name]}"', DeprecationWarning, stacklevel=2)
    name = DEPRECATED[name]
  if name not in BASELINES:
    raise ValueError(f"baseline desconocido: {name}")
  if not STATS.exists() or "trans_logp" not in np.load(STATS):
    print("ajustando baselines sobre train...", flush=True)
    save_stats(*fit_stats(np.load(TRAIN, mmap_mode="r")))
  s = np.load(STATS)
  if name == "unigram":
    m = Unigram(s["unigram"])
  elif name == "transicion":
    m = Transition(s["unigram"], s["trans_logp"])
  else:
    m = CopyMix(s["unigram"], float(s["p_copy"]))
    m.name = name
  return m.to(dev).eval()


if __name__ == "__main__":
  t0 = time.time()
  uni, p_copy, trans, alpha = fit_stats(np.load(TRAIN, mmap_mode="r"))
  save_stats(uni, p_copy, trans, alpha)
  ent = -(uni * np.log(uni)).sum()
  print(f"guardado en {STATS} ({time.time() - t0:.0f}s)")
  print(f"p_copy = {p_copy:.4f}   entropía del unigram = {ent:.4f} nats   min/max p = {uni.min():.2e}/{uni.max():.2e}")
  print(f"alpha (tabla de transición, elegido en held-out de train) = {alpha}")
