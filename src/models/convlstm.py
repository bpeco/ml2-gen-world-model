"""ConvLSTM (Shi et al. 2015) sobre la grilla 8x16 de tokens: baseline E1, sin atención."""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from src.models.base import FrameModel, lm_loss, sample_logits

H, W = 8, 16


class ConvLSTM(FrameModel):
  name = "convlstm"
  ctx_frames = 19

  @dataclass
  class Config:
    emb: int = 64
    hidden: int = 256
    n_layers: int = 2
    kernel: int = 3

  def __init__(self, cfg: Optional["ConvLSTM.Config"] = None):
    super().__init__()
    self.cfg = cfg = cfg or self.Config()
    pad = cfg.kernel // 2
    self.tok = nn.Embedding(1024, cfg.emb)
    self.start = nn.Parameter(torch.zeros(cfg.emb, H, W))  # entrada del paso 0 (aprendida)
    # gates = conv([x, h]) = conv_x(x) + conv_h(h); así conv_x se aplica a todos los pasos de una vez
    self.conv_x = nn.ModuleList([nn.Conv2d(cfg.emb if i == 0 else cfg.hidden, 4 * cfg.hidden, cfg.kernel, padding=pad)
                                 for i in range(cfg.n_layers)])
    self.conv_h = nn.ModuleList([nn.Conv2d(cfg.hidden, 4 * cfg.hidden, cfg.kernel, padding=pad, bias=False)
                                 for _ in range(cfg.n_layers)])
    self.head = nn.Conv2d(cfg.hidden, 1024, 1)
    nn.init.normal_(self.tok.weight, std=0.1)
    nn.init.normal_(self.start, std=0.1)
    for c in self.conv_x:
      nn.init.zeros_(c.bias)
      c.bias.data[cfg.hidden:2 * cfg.hidden] = 1.0  # forget gate abierta al inicio

  def _embed(self, frames):
    """(B, F, 128) -> (B, F, emb, 8, 16)"""
    B, Fr, _ = frames.shape
    return self.tok(frames.reshape(B, Fr, H, W)).permute(0, 1, 4, 2, 3)

  def _run(self, x, state=None):
    """x (B, F, emb, 8, 16) -> logits (B, F, 128, 1024), estado [(h, c)] por capa."""
    B, Fr = x.shape[:2]
    Hd = self.cfg.hidden
    new_state = []
    for l in range(self.cfg.n_layers):
      gx = self.conv_x[l](x.flatten(0, 1)).unflatten(0, (B, Fr))
      h, c = state[l] if state is not None else (gx.new_zeros(B, Hd, H, W), gx.new_zeros(B, Hd, H, W))
      hs = []
      for t in range(Fr):
        i, f, g, o = (gx[:, t] + self.conv_h[l](h)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        hs.append(h)
      x = torch.stack(hs, 1)
      new_state.append((h, c))
    lg = self.head(x.flatten(0, 1))  # (B*F, 1024, 8, 16)
    lg = lg.permute(0, 2, 3, 1).reshape(B, Fr, H * W, 1024)
    return lg, new_state

  def _start(self, B):
    return self.start[None, None].expand(B, 1, -1, -1, -1)

  def frame_logits(self, frames):
    x = torch.cat([self._start(frames.shape[0]), self._embed(frames[:, :-1])], 1)
    return self._run(x)[0]

  def train_loss(self, frames, teacher=None, alpha=0.5, tau=1.0):
    if teacher is not None:
      raise ValueError("ConvLSTM no usa destilación")
    return lm_loss(self.frame_logits(frames), frames)

  @torch.no_grad()
  def generate(self, ctx, n_frames, temperature=1.0, steps=None, generator=None):
    """Recorre el contexto una vez y luego cada frame nuevo es un paso recurrente (O(1) por frame)."""
    ctx = ctx[:, -self.ctx_frames:]
    x = torch.cat([self._start(ctx.shape[0]), self._embed(ctx)], 1)
    lg, state = self._run(x)
    for k in range(n_frames):
      new = sample_logits(lg[:, -1], temperature, generator)
      yield new
      if k + 1 < n_frames:
        lg, state = self._run(self._embed(new[:, None]), state)
