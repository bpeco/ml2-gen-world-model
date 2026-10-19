"""Baselines de conteo, estimados sobre train.

unigram: P(token), ignora el contexto.
markov:  P(token | token en el mismo slot del frame anterior), es decir 129 posiciones atrás.
Ambos se exponen como modelos que devuelven logits (B, T, V) para usar el mismo benchmark.
"""
import numpy as np
import torch
import torch.nn as nn

from src.common import BOS, TPF, VOCAB

COUNTS = "results/baseline_counts.npz"


def fit(train: np.ndarray, alpha=0.1):
  """train: (N, 1200, 128) int16."""
  t = train.astype(np.int64)
  uni = np.bincount(t.ravel(), minlength=VOCAB).astype(np.float64)
  uni[BOS] = uni.sum() / (TPF - 1)  # un BOS cada 128 tokens
  prev, cur = t[:, :-1].ravel(), t[:, 1:].ravel()
  pair = np.bincount(prev * VOCAB + cur, minlength=VOCAB * VOCAB).reshape(VOCAB, VOCAB).astype(np.float64)
  pair[BOS, BOS] = 1e12  # el BOS siempre está 129 posiciones después de otro BOS (P ~ 1)
  logp_uni = np.log(uni / uni.sum())
  logp_mk = np.log((pair + alpha) / (pair + alpha).sum(1, keepdims=True))
  same = (t[:, 1:] == t[:, :-1]).mean()
  return logp_uni, logp_mk, float(same)


class Unigram(nn.Module):
  def __init__(self, logp):
    super().__init__()
    self.register_buffer("logp", torch.tensor(logp, dtype=torch.float32))

  def forward(self, idx):
    return self.logp.expand(*idx.shape, -1)


class Markov(nn.Module):
  """Para predecir x[t+1] mira x[t+1-129] = x[t-128]. Sin frame anterior cae al unigrama."""
  def __init__(self, logp_uni, logp_mk):
    super().__init__()
    self.register_buffer("uni", torch.tensor(logp_uni, dtype=torch.float32))
    self.register_buffer("mk", torch.tensor(logp_mk, dtype=torch.float32))

  def forward(self, idx):
    B, T = idx.shape
    out = self.uni.expand(B, T, -1).clone()
    out[:, TPF - 1:] = self.mk[idx[:, : T - (TPF - 1)]]
    return out


def load(name):
  z = np.load(COUNTS)
  return Unigram(z["uni"]) if name == "unigram" else Markov(z["uni"], z["markov"])


if __name__ == "__main__":
  train = np.load("data/train_tokens.npy", mmap_mode="r")
  uni, mk, same = fit(np.asarray(train))
  np.savez(COUNTS, uni=uni, markov=mk)
  print(f"tokens iguales al mismo slot del frame anterior (train): {same:.3f}")
