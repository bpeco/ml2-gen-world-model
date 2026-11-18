"""Decoder de visualización de LeWM: embedding del frame (CLS proyectado) -> imagen RGB 128 x 256.

El paper (App. D) lo describe así: un transformer liviano en el que el CLS entra por cross-attention con una query
aprendida por parche, y las salidas son los parches de la imagen. Sirve solo para ver lo que el modelo sabe; se
entrena aparte, sobre el LeWM congelado (src/train_decoder.py), y no toca ni al encoder ni al predictor. El código
oficial no lo publica: profundidad, ancho, pérdida (MSE) y optimizador son propios (ver FIDELIDAD en lewm.py).

Como un solo vector no admite cross-attention útil (el softmax sobre una sola clave es la identidad), el embedding se
despliega en `n_kv` tokens de memoria con una proyección lineal; las queries (una por parche, con su posición) los
atienden y se atienden entre sí.
"""
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DecoderConfig:
  dim_in: int = 192      # embedding de LeWM
  dec_dim: int = 384
  layers: int = 4
  heads: int = 6
  n_kv: int = 8          # tokens de memoria en los que se despliega el CLS
  mlp_ratio: int = 4
  patch: int = 16
  img_h: int = 128
  img_w: int = 256


class DecBlock(nn.Module):
  """Pre-LN: auto-atención entre las queries, cross-attention a la memoria y MLP."""

  def __init__(self, dim, heads, mlp_ratio):
    super().__init__()
    self.heads = heads
    self.ln_s, self.ln_q, self.ln_kv, self.ln_m = (nn.LayerNorm(dim) for _ in range(4))
    self.qkv_s, self.proj_s = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
    self.q, self.kv, self.proj_c = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
    self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))

  def _heads(self, t, n):
    return t.view(t.shape[0], n, self.heads, -1).transpose(1, 2)

  def forward(self, x, mem):
    B, Q, C = x.shape
    q, k, v = self.qkv_s(self.ln_s(x)).view(B, Q, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
    x = x + self.proj_s(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, Q, C))
    k, v = self.kv(self.ln_kv(mem)).view(B, mem.shape[1], 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
    q = self._heads(self.q(self.ln_q(x)), Q)
    x = x + self.proj_c(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, Q, C))
    return x + self.mlp(self.ln_m(x))


class LeWMDecoder(nn.Module):
  """z (..., dim_in) -> imagen (..., img_h, img_w, 3) en [-1, 1] (sin saturar): float; decode() da uint8."""

  def __init__(self, cfg: DecoderConfig = None):
    super().__init__()
    self.cfg = cfg = cfg or DecoderConfig()
    self.gh, self.gw = cfg.img_h // cfg.patch, cfg.img_w // cfg.patch
    self.mem = nn.Linear(cfg.dim_in, cfg.n_kv * cfg.dec_dim)
    self.queries = nn.Parameter(torch.randn(self.gh * self.gw, cfg.dec_dim) * 0.02)  # una por parche (con su posición)
    self.blocks = nn.ModuleList(DecBlock(cfg.dec_dim, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.layers))
    self.norm = nn.LayerNorm(cfg.dec_dim)
    self.head = nn.Linear(cfg.dec_dim, cfg.patch * cfg.patch * 3)

  def forward(self, z):
    c = self.cfg
    sh = z.shape[:-1]
    z = z.reshape(-1, z.shape[-1])
    mem = self.mem(z).view(len(z), c.n_kv, c.dec_dim)
    x = self.queries.to(mem.dtype).expand(len(z), -1, -1)
    for b in self.blocks:
      x = b(x, mem)
    p = self.head(self.norm(x)).view(len(z), self.gh, self.gw, c.patch, c.patch, 3)
    img = p.permute(0, 1, 3, 2, 4, 5).reshape(len(z), c.img_h, c.img_w, 3)
    return img.view(*sh, c.img_h, c.img_w, 3)

  @torch.no_grad()
  def decode(self, z):
    """z (..., dim_in) -> uint8 (..., img_h, img_w, 3)."""
    return ((self(z).float() + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)

  def num_params(self) -> int:
    return sum(p.numel() for p in self.parameters())


def target_of(img):
  """uint8 (..., H, W, 3) -> float en [-1, 1], el objetivo de la MSE."""
  return img.float() / 127.5 - 1


def save_decoder(path, dec: LeWMDecoder, **meta):
  torch.save({"config": asdict(dec.cfg), "model": dec.state_dict(), **meta}, path)


def load_decoder(path, dev="cpu") -> LeWMDecoder:
  ck = torch.load(path, map_location="cpu", weights_only=False)
  dec = LeWMDecoder(DecoderConfig(**ck["config"]))
  dec.load_state_dict(ck["model"])
  return dec.to(dev).eval()
