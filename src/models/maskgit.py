"""MaskGIT sobre tokens de comma: predice los 128 tokens de un frame en paralelo, con S pasadas.

Mismo formato de secuencia que el GPT (BOS + 128 códigos por frame), pero la salida en la posición
p predice el token EN p (no el siguiente). Atención por bloques: causal entre frames y bidireccional
dentro de cada frame (Genie). El BOS nunca se enmascara.
"""
import math
from dataclasses import dataclass

import torch

from src.common import BOS, MASK, TPF, with_bos
from src.models.backbone import Backbone, BackboneConfig
from src.models.base import WorldModel, lm_loss
from src.models.cudagraph import GraphedStep

CODES = TPF - 1


def n_masked(s: int, S: int) -> int:
  """Tokens que siguen enmascarados tras el paso s de S (cosine; n_S = 0)."""
  return 0 if s >= S else int(math.floor(CODES * math.cos(math.pi / 2 * s / S)))


def decode_frame(logits_fn, steps, temperature, choice_temp, cur, generator=None, noise=None):
  """Decodificación iterativa por confianza (el sampler de MaskGIT), única para PyTorch, el stream y ONNX.

  cur (B, 128): los códigos del frame, todo MASK al empezar. logits_fn(cur) -> logits (B, 128, 1024) float32 de
  una pasada con los códigos parciales. En cada paso s de `steps` se muestrea cada celda, se fija en las más
  confiables (con ruido Gumbel de choice_temp, que baja linealmente a 0) hasta dejar n_masked(s, steps) en MASK.
  noise: None (el ruido sale de generator, por paso: (B, 128, 1024) del muestreo y (B, 128) del Gumbel) o la
  lista ya generada en ese mismo orden (para capturar el frame entero en un CUDA graph). Devuelve (B, 128).
  """
  B = cur.shape[0]
  fixed = torch.zeros(B, CODES, dtype=torch.bool, device=cur.device)
  n_prev, j = CODES, 0
  for s in range(1, steps + 1):
    lg = logits_fn(cur)
    if temperature == 0:
      samp = lg.argmax(-1)
    else:
      q = noise[j] if noise is not None else torch.empty_like(lg).exponential_(1, generator=generator)
      j += 1
      samp = (torch.softmax(lg / temperature, -1) / q).argmax(-1)
    lp = torch.log_softmax(lg / temperature if temperature > 0 else lg, -1).gather(-1, samp[..., None])[..., 0]
    if temperature > 0 and choice_temp > 0 and s < steps:
      q = noise[j] if noise is not None else torch.empty_like(lp).exponential_(1, generator=generator)
      j += 1
      lp = lp + choice_temp * (1 - s / steps) * -torch.log(q)
    n_s = n_masked(s, steps)
    k = n_prev - n_s
    lp = lp.masked_fill(fixed, float("-inf"))
    if k > 0:
      fixed = fixed.scatter(1, lp.topk(k, dim=-1).indices, True)
    cur = torch.where(fixed & (cur == MASK), samp, cur)
    n_prev = n_s
  return cur


class MaskGIT(WorldModel):
  name = "maskgit"
  ctx_frames = 19
  has_likelihood = False

  @dataclass
  class Config:
    n_layer: int = 6
    n_head: int = 6
    dim: int = 384
    dropout: float = 0.0
    block_size: int = 20 * TPF
    mask_schedule: str = "cosine"   # "cosine": r = cos(pi/2 u) | "uniform": r ~ U[mask_min, mask_max] (Genie)
    mask_min: float = 0.5
    mask_max: float = 1.0
    default_steps: int = 8
    choice_temp: float = 1.0        # ruido Gumbel sobre la confianza, baja linealmente a 0
    pos: str = "abs"                # embedding de posición: "abs" | "frame" (frame + celda)
    p_clean: float = 0.0            # prob. de dejar un frame entero sin máscara al entrenar (contexto limpio, como en inferencia)

  def __init__(self, cfg: "MaskGIT.Config"):
    super().__init__()
    self.cfg = cfg
    self.net = Backbone(BackboneConfig(n_layer=cfg.n_layer, n_head=cfg.n_head, dim=cfg.dim,
                                       block_size=cfg.block_size, vocab_in=1026, vocab_out=1024,
                                       attn="block", dropout=cfg.dropout, pos=cfg.pos))
    self._graphs_on = False
    self._graphs = {}
    self._cache_key = None

  def _drop_cache(self):
    if self._cache_key is not None:  # el camino sin caché no admite cachés activas
      self.net.clear_cache()
      self._cache_key, self._graphs = None, {}

  def logits(self, x):
    """(B, L) secuencia con MASK -> (B, L, 1024): la posición p predice el token en p."""
    self._drop_cache()
    return self.net(x)

  def _mask_rates(self, B, Fr, dev):
    c = self.cfg
    u = torch.rand(B, Fr, device=dev)
    if c.mask_schedule == "cosine":
      return torch.cos(math.pi / 2 * u)
    if c.mask_schedule == "uniform":
      return c.mask_min + (c.mask_max - c.mask_min) * u
    raise ValueError(f"mask_schedule no soportado: {c.mask_schedule}")

  def train_loss(self, frames, teacher=None, alpha=0.5, tau=1.0):
    if teacher is not None:
      raise ValueError("MaskGIT no se destila: teacher tiene que ser None")
    B, Fr, _ = frames.shape
    r = self._mask_rates(B, Fr, frames.device)
    if self.cfg.p_clean > 0:  # en inferencia el contexto está limpio: que en entrenamiento también aparezca
      r = torch.where(torch.rand(B, Fr, device=frames.device) < self.cfg.p_clean, torch.zeros_like(r), r)
    masked = torch.rand(B, Fr, CODES, device=frames.device) < r[..., None]  # el BOS no entra
    seq = with_bos(frames).view(B, Fr, TPF)
    x = seq.clone()
    x[:, :, 1:] = torch.where(masked, torch.full_like(frames, MASK), frames)
    lg = self.logits(x.view(B, -1)).view(B, Fr, TPF, -1)[:, :, 1:]
    loss, m = lm_loss(lg, frames, mask=masked)
    m["mask_rate"] = masked.float().mean().detach()
    return loss, m

  def last_frame_logits(self, frames):
    """Último frame entero enmascarado, una pasada: (B, 128, 1024). Es la "CE con 1 paso"."""
    seq = with_bos(frames)
    seq[:, -CODES:] = MASK
    return self.logits(seq)[:, -CODES:]

  def use_cuda_graphs(self, enable: bool = True):
    self._graphs_on = enable
    self._graphs = {}

  def _ensure_cache(self, B, dtype, device):
    key = (B, dtype, device)
    if key != self._cache_key:
      self.net.setup_cache(B, dtype=dtype, device=device)
      self._cache_key = key
      self._graphs = {}  # los graphs apuntan a las cachés viejas

  def _frame_pass(self, tok, pos):
    return self.net(tok, pos)

  @torch.no_grad()
  def next_frame(self, ctx, temperature=1.0, steps=None, generator=None):
    S = steps or self.cfg.default_steps
    B = ctx.shape[0]
    p = next(self.parameters())
    dev = p.device
    self._ensure_cache(B, p.dtype, dev)
    seq = with_bos(ctx[:, -min(self.ctx_frames, self.cfg.block_size // TPF - 1):])  # el frame nuevo tiene que caber
    P = seq.shape[1]
    pos_all = torch.arange(self.cfg.block_size, device=dev)
    self.net(seq, pos_all[:P])  # prefill del contexto en la caché; cada frame lo re-prefillea
    pos = pos_all[P:P + TPF]
    tok = torch.full((B, TPF), MASK, dtype=torch.long, device=dev)
    tok[:, 0] = BOS
    use_graph = self._graphs_on and dev.type == "cuda"

    def logits_fn(cur):
      tok[:, 1:] = cur
      if use_graph:
        g = self._graphs.get(B)
        if g is None:
          g = self._graphs[B] = GraphedStep(self._frame_pass, tok, pos)
        return g(tok, pos)[:, 1:].float()
      return self._frame_pass(tok, pos)[:, 1:].float()

    cur = torch.full((B, CODES), MASK, dtype=torch.long, device=dev)
    return decode_frame(logits_fn, S, temperature, self.cfg.choice_temp, cur, generator).clone()
