"""Transformer propio (estilo nanoGPT), compartido por el GPT alumno y MaskGIT.

attn="causal": máscara causal token a token. attn="block": causal entre frames de TPF tokens y
bidireccional dentro de cada frame. Con caché KV estática (setup_cache) todo es apto para CUDA graphs.
"""
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common import BLOCK, TPF, VOCAB


@dataclass
class BackboneConfig:
  n_layer: int = 6
  n_head: int = 6
  dim: int = 384
  block_size: int = BLOCK
  vocab_in: int = VOCAB    # MaskGIT: 1026 (+ MASK). 0: sin embedding de tokens (entrada continua, forward(emb=...))
  vocab_out: int = VOCAB   # MaskGIT: 1024. 0: sin cabeza, devuelve el estado oculto normalizado (jepa, lewm)
  attn: str = "causal"     # "causal" | "block"
  dropout: float = 0.0
  pos: str = "abs"         # "abs": un embedding por posición | "frame": frame + celda (factorizado)


def mult8(n: int) -> int:
  """Largo de caché/máscara alineado a 8 (como comma y gpt-fast): SDPA con máscara lo prefiere."""
  return n + (-n) % 8


def make_mask(n: int, attn: str) -> torch.Tensor:
  """(n, n) bool, mask[i, j] = la posición i puede mirar a la j."""
  if attn == "causal":
    return torch.tril(torch.ones(n, n, dtype=torch.bool))
  if attn == "block":
    f = torch.arange(n) // TPF
    return f[None, :] <= f[:, None]
  raise ValueError(f"attn no soportado: {attn}")


class Attention(nn.Module):
  def __init__(self, cfg: BackboneConfig):
    super().__init__()
    assert cfg.dim % cfg.n_head == 0
    self.n_head, self.hd, self.dropout = cfg.n_head, cfg.dim // cfg.n_head, cfg.dropout
    self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim)
    self.proj = nn.Linear(cfg.dim, cfg.dim)
    self.k_cache = self.v_cache = None  # tensores simples (no buffers): no entran al state_dict

  def forward(self, x, attn_mask=None, input_pos=None, is_causal=False):
    B, T, C = x.shape
    q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.hd).permute(2, 0, 3, 1, 4)
    if self.k_cache is not None:
      # escritura in place en las posiciones nuevas; se atiende sobre toda la caché estática
      self.k_cache.index_copy_(2, input_pos, k)
      self.v_cache.index_copy_(2, input_pos, v)
      k, v = self.k_cache, self.v_cache
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal,
                                       dropout_p=self.dropout if self.training else 0.0)
    return self.proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
  def __init__(self, cfg: BackboneConfig):
    super().__init__()
    self.ln1, self.ln2 = nn.LayerNorm(cfg.dim), nn.LayerNorm(cfg.dim)
    self.attn = Attention(cfg)
    self.fc = nn.Linear(cfg.dim, 4 * cfg.dim)
    self.proj = nn.Linear(4 * cfg.dim, cfg.dim)
    self.drop = nn.Dropout(cfg.dropout)

  def forward(self, x, **kw):
    x = x + self.drop(self.attn(self.ln1(x), **kw))
    return x + self.drop(self.proj(F.gelu(self.fc(self.ln2(x)), approximate="tanh")))


class Backbone(nn.Module):
  def __init__(self, cfg: BackboneConfig):
    super().__init__()
    self.cfg = cfg
    self.wte = nn.Embedding(cfg.vocab_in, cfg.dim) if cfg.vocab_in else None
    if cfg.pos == "abs":
      self.wpe = nn.Embedding(cfg.block_size, cfg.dim)
    elif cfg.pos == "frame":  # la misma celda comparte embedding en todos los frames
      self.wpe_f = nn.Embedding(math.ceil(cfg.block_size / TPF), cfg.dim)
      self.wpe_c = nn.Embedding(TPF, cfg.dim)
    else:
      raise ValueError(f"pos no soportado: {cfg.pos}")
    self.drop = nn.Dropout(cfg.dropout)
    self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
    self.ln_f = nn.LayerNorm(cfg.dim)
    self.head = nn.Linear(cfg.dim, cfg.vocab_out, bias=False) if cfg.vocab_out else None
    # alineada a 8; las posiciones >= block_size nunca se escriben y quedan "en el futuro" (enmascaradas)
    self.register_buffer("mask", make_mask(mult8(cfg.block_size), cfg.attn), persistent=False)
    self.cache_len = 0
    self.apply(self._init)
    for b in self.blocks:  # proyecciones residuales escaladas (GPT-2)
      for lin in (b.attn.proj, b.proj):
        nn.init.normal_(lin.weight, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

  @staticmethod
  def _init(m):
    if isinstance(m, nn.Linear):
      nn.init.normal_(m.weight, 0.0, 0.02)
      if m.bias is not None:
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
      nn.init.normal_(m.weight, 0.0, 0.02)

  def pos_emb(self, pos):
    if self.cfg.pos == "abs":
      return self.wpe(pos)
    return self.wpe_f(pos // TPF) + self.wpe_c(pos % TPF)

  def num_params(self) -> int:
    return sum(p.numel() for p in self.parameters())

  def setup_cache(self, batch_size: int, max_len: Optional[int] = None, dtype=None, device=None):
    """Reserva cachés KV estáticas (B, n_head, max_len, head_dim) por capa."""
    c = self.cfg
    p = next(self.parameters())
    dtype, device = dtype or p.dtype, device or p.device
    self.cache_len = mult8(max_len or c.block_size)
    for b in self.blocks:
      shape = (batch_size, c.n_head, self.cache_len, c.dim // c.n_head)
      b.attn.k_cache = torch.zeros(shape, dtype=dtype, device=device)
      b.attn.v_cache = torch.zeros(shape, dtype=dtype, device=device)

  def clear_cache(self):
    self.cache_len = 0
    for b in self.blocks:
      b.attn.k_cache = b.attn.v_cache = None

  def forward(self, idx: Optional[torch.Tensor] = None, input_pos: Optional[torch.Tensor] = None,
              emb: Optional[torch.Tensor] = None, hidden: bool = False) -> torch.Tensor:
    """idx (B, T) ids, o emb (B, T, dim) entrada continua (sin wte). hidden=True devuelve ln_f(x)
    (B, T, dim) en vez de los logits; sin cabeza (vocab_out=0) siempre devuelve eso."""
    T = (emb if emb is not None else idx).shape[1]
    if self.blocks[0].attn.k_cache is not None:
      assert input_pos is not None, "con caché hay que pasar input_pos"
      kw = dict(input_pos=input_pos, attn_mask=self.mask[input_pos, :self.cache_len])
    elif input_pos is None:
      input_pos = torch.arange(T, device=self.mask.device)
      kw = dict(is_causal=True) if self.cfg.attn == "causal" else dict(attn_mask=self.mask[:T, :T])
    else:
      kw = dict(attn_mask=self.mask[input_pos[:, None], input_pos[None, :]])
    x = emb if emb is not None else self.wte(idx)
    x = self.drop(x + self.pos_emb(input_pos))
    for b in self.blocks:
      x = b(x, **kw)
    x = self.ln_f(x)
    return x if hidden or self.head is None else self.head(x)
