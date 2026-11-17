"""JEPA estilo V-JEPA / V-JEPA 2-AC sobre los tokens de comma: predice el latente espacial del próximo frame.

- Entrada: cada token -> su vector del codebook del VQ-VAE (congelado) -> Linear -> encoder transformer
  bidireccional dentro del frame -> z (128 celdas x dim), con LayerNorm sin parámetros.
- Predictor: transformer causal entre frames y bidireccional dentro (la máscara de MaskGIT), 129 lugares por
  frame: el 0 es la acción (pose) y los 1..128 el latente. La salida del frame f predice el latente del frame f
  (la entrada va corrida un frame: [start, z_0, ..., z_{F-2}]).
- Objetivos (cfg.objective):
    "ema"      V-JEPA: L1 contra un encoder target EMA del encoder, con stop-grad.
    "sigreg"   L1 contra el mismo encoder sin stop-grad + λ·SIGReg sobre las celdas (el antiaplastamiento de LeWM).
    "codebook" L1 contra el vector del codebook (normalizado) de cada token del frame: no puede colapsar.
    "ce"       control: sin pérdida latente, la CE de la lectura entrena todo.
- Lectura: LayerNorm + Linear dim -> 1024 por celda, entrenada con la CE sobre ẑ.detach() (y sobre el target,
  para medir el "techo"): no le pasa gradiente al encoder ni al predictor (salvo en "ce").
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common import BLOCK, TPF
from src.models.backbone import Backbone, BackboneConfig
from src.models.base import lm_loss
from src.models.latent import LatentWM, SIGReg

CELLS = TPF - 1


class CellEncoder(nn.Module):
  """Frame de tokens -> latentes por celda (B, F, 128, dim)."""

  def __init__(self, dim, n_layer, n_head, dropout):
    super().__init__()
    self.in_proj = nn.Linear(256, dim)
    self.net = Backbone(BackboneConfig(n_layer=n_layer, n_head=n_head, dim=dim, block_size=CELLS, vocab_in=0,
                                       vocab_out=0, attn="block", dropout=dropout, pos="abs"))

  def forward(self, frames, codebook):
    B, Fr, C = frames.shape
    e = codebook[frames.reshape(B * Fr, C)].to(self.in_proj.weight.dtype)
    x = self.net(emb=self.in_proj(e))
    return F.layer_norm(x, x.shape[-1:]).view(B, Fr, C, -1)


class JEPA(LatentWM):
  name = "jepa"
  has_likelihood = True

  @dataclass
  class Config:
    n_layer: int = 6           # encoder + predictor
    n_head: int = 6
    dim: int = 384
    n_enc: int = 2             # capas del encoder; el predictor tiene n_layer - n_enc
    dropout: float = 0.0
    block_size: int = BLOCK    # lugares del predictor: 20 frames x 129
    objective: str = "ema"     # "ema" | "sigreg" | "codebook" | "ce"
    ema: tuple = (0.996, 1.0)  # momento de la EMA, de [0] a [1] con coseno
    lam: float = 0.09          # peso de SIGReg (objective="sigreg")
    num_proj: int = 1024
    knots: int = 17
    sigreg_n: int = 2048       # celdas muestreadas por paso para SIGReg
    p_null: float = 0.25       # dropout de acción al entrenar
    rollout: str = "token"     # "token" | "latent"
    n_pose: int = 6

  def __init__(self, cfg: "JEPA.Config"):
    super().__init__()
    assert cfg.objective in ("ema", "sigreg", "codebook", "ce"), cfg.objective
    assert 0 < cfg.n_enc < cfg.n_layer
    self.cfg = cfg
    self._init_common(cfg.n_pose)
    D = cfg.dim
    self.encoder = CellEncoder(D, cfg.n_enc, cfg.n_head, cfg.dropout)
    self.pred = Backbone(BackboneConfig(n_layer=cfg.n_layer - cfg.n_enc, n_head=cfg.n_head, dim=D,
                                        block_size=cfg.block_size, vocab_in=0, vocab_out=0, attn="block",
                                        dropout=cfg.dropout, pos="frame"))
    self.tdim = 256 if cfg.objective == "codebook" else D
    self.pred_head = nn.Linear(D, self.tdim)
    self.start = nn.Parameter(torch.randn(CELLS, D) * 0.02)
    self.act_embed = nn.Sequential(nn.Linear(cfg.n_pose, D), nn.GELU(approximate="tanh"), nn.Linear(D, D))
    self.null_act = nn.Parameter(torch.randn(D) * 0.02)
    self.read = nn.Sequential(nn.LayerNorm(self.tdim), nn.Linear(self.tdim, 1024))
    self.tgt = self.make_target() if cfg.objective == "ema" else None
    self.sigreg = SIGReg(cfg.knots, cfg.num_proj) if cfg.objective == "sigreg" else None

  # ---- interfaz de LatentWM
  def encode(self, frames, target=False):
    return (self.tgt if target else self.encoder)(frames, self.codebook)

  def predict(self, z_in, acts):
    B, Fr, C, D = z_in.shape
    seq = torch.cat([acts[:, :, None].to(z_in.dtype), z_in], 2).view(B, Fr * TPF, D)
    h = self.pred(emb=seq).view(B, Fr, TPF, D)[:, :, 1:]
    return self.pred_head(h)

  def readout(self, zh):
    return self.read(zh)

  @torch.no_grad()
  def targets(self, frames):
    obj = self.cfg.objective
    if obj == "ema":
      return self.encode(frames, target=True)
    if obj == "codebook":
      cb = self.codebook[frames]
      return F.layer_norm(cb, cb.shape[-1:])
    return self.encode(frames)

  # ---- entrenamiento
  def train_loss(self, frames, teacher=None, alpha=0.5, tau=1.0, poses=None):
    if teacher is not None:
      raise ValueError("JEPA no se destila: teacher tiene que ser None")
    z, zh = self.predict_frames(frames, poses)
    obj = self.cfg.objective
    z_std = z.detach().float().reshape(-1, z.shape[-1]).std(0).mean()
    if obj == "ce":
      loss, met = lm_loss(self.readout(zh), frames)
      return loss, {**met, "z_std": z_std}
    tz = z if obj == "sigreg" else self.targets(frames)  # sigreg: sin stop-grad
    pred = (zh.float() - tz.float()).abs().mean()
    loss, met = pred, {"jepa_l1": pred.detach()}
    if obj == "sigreg":
      flat = z.reshape(-1, z.shape[-1])
      idx = torch.randint(len(flat), (min(self.cfg.sigreg_n, len(flat)),), device=flat.device)
      sr = self.sigreg(flat[idx])
      loss = loss + self.cfg.lam * sr
      met["sigreg"] = sr.detach()
    ce, _ = lm_loss(self.readout(zh.detach()), frames)
    ce_t, _ = lm_loss(self.readout(tz.detach()), frames)
    met.update(ce=ce.detach(), ce_techo=ce_t.detach(), z_std=z_std)
    return loss + ce + ce_t, met
