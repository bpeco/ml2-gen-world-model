"""GPT chico estilo nanoGPT: mismo esquema que utils/gpt.py de comma (pos. aprendidas, pre-LN, GELU)."""
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common import BLOCK, TPF, VOCAB


@dataclass
class Config:
  n_layer: int = 6
  n_head: int = 6
  dim: int = 384
  block_size: int = BLOCK
  vocab_size: int = VOCAB
  factored_pos: bool = False  # posición = slot dentro del frame (0-128) + índice de frame


class Block(nn.Module):
  def __init__(self, c: Config):
    super().__init__()
    self.n_head = c.n_head
    self.ln_1 = nn.LayerNorm(c.dim)
    self.c_attn = nn.Linear(c.dim, 3 * c.dim)
    self.attn_proj = nn.Linear(c.dim, c.dim)
    self.ln_2 = nn.LayerNorm(c.dim)
    self.mlp = nn.Sequential(nn.Linear(c.dim, 4 * c.dim), nn.GELU(approximate="tanh"), nn.Linear(4 * c.dim, c.dim))

  def forward(self, x):
    B, T, C = x.shape
    q, k, v = self.c_attn(self.ln_1(x)).split(C, dim=-1)
    q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    x = x + self.attn_proj(y.transpose(1, 2).reshape(B, T, C))
    return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
  def __init__(self, c: Config):
    super().__init__()
    self.config = c
    self.wte = nn.Embedding(c.vocab_size, c.dim)
    if c.factored_pos:
      self.wpe_slot = nn.Embedding(TPF, c.dim)
      self.wpe_frame = nn.Embedding(c.block_size // TPF, c.dim)
    else:
      self.wpe = nn.Embedding(c.block_size, c.dim)
    self.h = nn.ModuleList(Block(c) for _ in range(c.n_layer))
    self.ln_f = nn.LayerNorm(c.dim)
    self.lm_head = nn.Linear(c.dim, c.vocab_size, bias=False)
    self.lm_head.weight = self.wte.weight  # weight tying
    self.apply(self._init)

  @staticmethod
  def _init(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
      nn.init.normal_(m.weight, std=0.02)
    if isinstance(m, nn.Linear) and m.bias is not None:
      nn.init.zeros_(m.bias)

  def n_params(self):
    return sum(p.numel() for p in self.parameters())

  def forward(self, idx):
    pos = torch.arange(idx.shape[1], device=idx.device)
    if self.config.factored_pos:
      x = self.wte(idx) + self.wpe_slot(pos % TPF) + self.wpe_frame(pos // TPF)
    else:
      x = self.wte(idx) + self.wpe(pos)
    for b in self.h:
      x = b(x)
    return self.lm_head(self.ln_f(x))

  def save(self, path, **extra):
    torch.save({"config": asdict(self.config), "state_dict": self.state_dict(), **extra}, path)

  @classmethod
  def load(cls, path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=True)
    m = cls(Config(**ck["config"]))
    m.load_state_dict(ck["state_dict"])
    return m
