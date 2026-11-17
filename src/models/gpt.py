"""GPT alumno autoregresivo: mismo formato de secuencia que gpt2m (BOS antes de cada frame)."""
from dataclasses import dataclass

import torch

from src.common import BLOCK, BOS, TPF, VOCAB, with_bos
from src.models.backbone import Backbone, BackboneConfig
from src.models.base import WorldModel, lm_loss, sample_logits
from src.models.cudagraph import GraphedStep


class GPT(WorldModel):
  name = "gpt"
  ctx_frames = 19
  has_likelihood = True

  @dataclass
  class Config:
    n_layer: int = 6
    n_head: int = 6
    dim: int = 384
    dropout: float = 0.0
    block_size: int = BLOCK
    pos: str = "abs"                # embedding de posición: "abs" | "frame" (frame + celda)

  def __init__(self, cfg: "GPT.Config"):
    super().__init__()
    self.cfg = cfg
    self.net = Backbone(BackboneConfig(n_layer=cfg.n_layer, n_head=cfg.n_head, dim=cfg.dim,
                                       block_size=cfg.block_size, vocab_in=VOCAB, vocab_out=VOCAB,
                                       attn="causal", dropout=cfg.dropout, pos=cfg.pos))
    self._graphs_on = False
    self._graphs = {}
    self._cache_key = None

  def _drop_cache(self):
    if self._cache_key is not None:  # el camino sin caché no admite cachés activas
      self.net.clear_cache()
      self._cache_key, self._graphs = None, {}

  def logits(self, x):
    self._drop_cache()
    return self.net(x)

  def train_loss(self, frames, teacher=None, alpha=0.5, tau=1.0):
    seq = with_bos(frames)
    x, y = seq[:, :-1], seq[:, 1:]
    lg = self.logits(x)
    tl = None
    if teacher is not None:
      with torch.no_grad():
        tl = teacher.logits(x)
    return lm_loss(lg, y, tl, alpha, tau)

  def use_cuda_graphs(self, enable: bool = True):
    self._graphs_on = enable
    self._graphs = {}

  def _ensure_cache(self, B, dtype, device):
    key = (B, dtype, device)
    if key != self._cache_key:
      self.net.setup_cache(B, dtype=dtype, device=device)
      self._cache_key = key
      self._graphs = {}  # los graphs apuntan a las cachés viejas

  def _decode(self, tok, pos):
    return self.net(tok, pos)[:, -1]

  @torch.no_grad()
  def next_frame(self, ctx, temperature=1.0, steps=None, generator=None):
    B = ctx.shape[0]
    p = next(self.parameters())
    self._ensure_cache(B, p.dtype, p.device)
    seq = with_bos(ctx[:, -self.ctx_frames:])
    seq = torch.cat([seq, torch.full_like(seq[:, :1], BOS)], 1)  # BOS del frame a generar
    P = seq.shape[1]
    pos_all = torch.arange(self.cfg.block_size, device=p.device)
    lg = self.net(seq, pos_all[:P])[:, -1]  # prefill; el contexto se re-prefillea en cada frame
    out = []
    use_graph = self._graphs_on and p.device.type == "cuda"
    for i in range(TPF - 1):
      tok = sample_logits(lg[:, :BOS], temperature, generator)  # nunca emite BOS
      out.append(tok)
      if i == TPF - 2:
        break
      tok, pos = tok[:, None], pos_all[P + i:P + i + 1]
      if use_graph:
        g = self._graphs.get(B)
        if g is None:
          g = self._graphs[B] = GraphedStep(self._decode, tok, pos)
        lg = g(tok, pos)
      else:
        lg = self._decode(tok, pos)
    return torch.stack(out, 1)
