"""LeWorldModel (LeWM): un embedding por frame, SIGReg y predictor con AdaLN, desde tokens de comma o desde píxeles.

Receta de Maes, Le Lidec, Scieur, LeCun y Balestriero, "LeWorldModel: Stable End-to-End Joint-Embedding
Predictive Architecture from Pixels" (arXiv 2603.19312; código oficial github.com/lucas-maes/le-wm, MIT):
- Encoder ViT-Tiny (12 capas, 3 cabezas, dim 192) -> CLS -> projector MLP con BatchNorm -> z (dim 192).
  cfg.input = "tokens" (E5, adaptación): en vez de patches de píxeles, cada token (8 x 16) entra con su vector del
  codebook del VQ-VAE (congelado); decodificar todo train a píxeles de antemano costaba ~16 ms/frame y ~295 GB.
  En inferencia necesita el encoder de comma delante (cámara -> tokens).
  cfg.input = "pixels" (como el paper): ViT sobre parches de 16x16 de la imagen RGB de 128x256 (la misma grilla de
  8 x 16). El dataset solo tiene tokens, así que al entrenar los frames se generan al vuelo con el decoder de comma
  congelado (fp16, sin gradiente; src/infer/codec.py): el modelo ve las reconstrucciones del VQ-VAE, que están a
  LPIPS 0,10 de la cámara real. encode() acepta imágenes uint8 (B, F, 128, 256, 3) o tokens (B, F, 128), que
  decodifica primero (así la evaluación, los rollouts y el bench, que trabajan con tokens, no cambian). En
  inferencia desde la cámara no hace falta el encoder de comma.
- Predictor: transformer causal en el tiempo (6 capas, 16 cabezas, dropout 0,1) con AdaLN-zero condicionado
  en la acción (pose del frame), y pred_proj MLP con BatchNorm. Ancho 320 para quedar en los ~10M que da la
  sección 3.1 del paper (el apéndice habla de un ViT-S de 384; con AdaLN serían ~16M).
- Pérdida: MSE(ẑ, z) + λ·SIGReg(z), λ = 0,09, sin stop-grad ni EMA (cfg.objective = "sigreg").
  Ablaciones: "ema" (target EMA + stop-grad, estilo V-JEPA) y "ce" (control: la CE del decoder entrena todo).
- Decoder a tokens: el decoder de visualización del paper (queries aprendidas que atienden al embedding con
  cross-attention), con una query por celda y salida de 1024 logits. Se entrena sobre ẑ.detach() y
  z.detach(): no toca al JEPA. Es lo que permite medir CE, LPIPS y FVD como al resto.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common import CTX_FRAMES, TPF
from src.models.backbone import Backbone, BackboneConfig
from src.models.base import lm_loss
from src.models.latent import MLP, ConditionalBlock, Embedder, LatentWM, SIGReg

CELLS = TPF - 1


class FrameEncoder(nn.Module):
  """ViT sobre los 128 tokens de un frame -> CLS -> projector (MLP + BatchNorm): (B, F, dim)."""

  def __init__(self, dim, n_layer, n_head, proj_hidden):
    super().__init__()
    self.in_proj = nn.Linear(256, dim)
    self.cls = nn.Parameter(torch.randn(dim) * 0.02)
    self.net = Backbone(BackboneConfig(n_layer=n_layer, n_head=n_head, dim=dim, block_size=TPF, vocab_in=0,
                                       vocab_out=0, attn="block", pos="abs"))
    self.projector = MLP(dim, proj_hidden, dim)

  def forward(self, frames, codebook):
    B, Fr, C = frames.shape
    e = self.in_proj(codebook[frames.reshape(B * Fr, C)].to(self.in_proj.weight.dtype))
    x = torch.cat([self.cls.to(e.dtype).expand(B * Fr, 1, -1), e], 1)
    return self.projector(self.net(emb=x)[:, 0]).view(B, Fr, -1)


class PixelFrameEncoder(nn.Module):
  """ViT sobre parches de la imagen RGB (128 x 256 -> 8 x 16 parches de 16 px) -> CLS -> projector: (B, F, dim)."""

  def __init__(self, dim, n_layer, n_head, proj_hidden, patch=16):
    super().__init__()
    self.patch = nn.Conv2d(3, dim, patch, stride=patch)
    self.cls = nn.Parameter(torch.randn(dim) * 0.02)
    self.net = Backbone(BackboneConfig(n_layer=n_layer, n_head=n_head, dim=dim, block_size=TPF, vocab_in=0,
                                       vocab_out=0, attn="block", pos="abs"))
    self.projector = MLP(dim, proj_hidden, dim)

  def forward(self, img, codebook=None):
    """img (B, F, H, W, 3) uint8 o float en [0, 255]."""
    B, Fr, H, W, _ = img.shape
    x = img.reshape(B * Fr, H, W, 3).permute(0, 3, 1, 2).to(self.patch.weight.dtype) / 127.5 - 1
    e = self.patch(x).flatten(2).transpose(1, 2)                       # (B*F, 128, dim)
    x = torch.cat([self.cls.to(e.dtype).expand(B * Fr, 1, -1), e], 1)
    return self.projector(self.net(emb=x)[:, 0]).view(B, Fr, -1)


_RENDER = {}


@torch.no_grad()
def render_tokens(tok, chunk=32):
  """tokens (..., 128) -> imágenes uint8 (..., 128, 256, 3) con el decoder de comma congelado (fp16 en CUDA).

  Es la "cámara" de LeWM desde píxeles al entrenar y al evaluar con tokens. No es un submódulo del modelo: no
  entra en el checkpoint, ni en los parámetros, ni en el optimizador.
  """
  from src.infer.codec import Codec
  dev = tok.device
  key = (dev.type, dev.index)
  if key not in _RENDER:
    _RENDER[key] = Codec(dev, "fp16" if dev.type == "cuda" else "fp32", graphs=False, encoder=False)
  codec = _RENDER[key]
  flat = tok.reshape(-1, tok.shape[-1]).long()
  with torch.autocast(dev.type, enabled=False):
    out = torch.cat([codec.decode(flat[i:i + chunk]) for i in range(0, len(flat), chunk)])
  return out.view(*tok.shape[:-1], *out.shape[1:])


class CrossBlock(nn.Module):
  def __init__(self, dim, heads):
    super().__init__()
    self.heads = heads
    self.ln_q, self.ln_kv, self.ln_mlp = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
    self.q, self.kv, self.proj = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
    self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"), nn.Linear(4 * dim, dim))

  def forward(self, x, mem):
    B, Q, C = x.shape
    H = self.heads
    q = self.q(self.ln_q(x)).view(B, Q, H, C // H).transpose(1, 2)
    k, v = self.kv(self.ln_kv(mem)).view(B, mem.shape[1], 2, H, C // H).permute(2, 0, 3, 1, 4)
    y = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, Q, C)
    x = x + self.proj(y)
    return x + self.mlp(self.ln_mlp(x))


class TokenDecoder(nn.Module):
  """Embedding de un frame (..., dim) -> logits de sus 128 celdas (..., 128, 1024)."""

  def __init__(self, dim, dec_dim, layers, heads, n_kv):
    super().__init__()
    self.n_kv, self.dec_dim = n_kv, dec_dim
    self.mem = nn.Linear(dim, n_kv * dec_dim)
    self.queries = nn.Parameter(torch.randn(CELLS, dec_dim) * 0.02)
    self.blocks = nn.ModuleList(CrossBlock(dec_dim, heads) for _ in range(layers))
    self.norm = nn.LayerNorm(dec_dim)
    self.head = nn.Linear(dec_dim, 1024)

  def forward(self, z):
    sh = z.shape[:-1]
    z = z.reshape(-1, z.shape[-1])
    mem = self.mem(z).view(len(z), self.n_kv, self.dec_dim)
    x = self.queries.to(mem.dtype).expand(len(z), -1, -1)
    for b in self.blocks:
      x = b(x, mem)
    return self.head(self.norm(x)).view(*sh, CELLS, 1024)


class LeWM(LatentWM):
  name = "lewm"
  has_likelihood = True

  @dataclass
  class Config:
    dim: int = 192             # embedding por frame (paper: 192)
    enc_layers: int = 12       # ViT-Tiny
    enc_heads: int = 3
    proj_hidden: int = 2048
    pred_dim: int = 320        # 6 capas con AdaLN ≈ 11M: los ~10M de la sección 3.1 del paper
    pred_layers: int = 6
    pred_heads: int = 16
    pred_mlp: int = 1280
    pred_dropout: float = 0.1
    dec_dim: int = 256
    dec_layers: int = 3
    dec_heads: int = 4
    dec_kv: int = 8            # tokens de memoria en los que se despliega el embedding para el decoder
    frames: int = CTX_FRAMES   # largo máximo de la secuencia del predictor (start + 19 frames)
    objective: str = "sigreg"  # "sigreg" | "ema" | "ce"
    lam: float = 0.09
    num_proj: int = 1024
    knots: int = 17
    ema: tuple = (0.996, 1.0)
    p_null: float = 0.25
    rollout: str = "token"     # "token" | "latent"
    n_pose: int = 6
    input: str = "tokens"      # "tokens" (E5: vectores del codebook) | "pixels" (como el paper: parches de la imagen)
    patch: int = 16

  def __init__(self, cfg: "LeWM.Config"):
    super().__init__()
    assert cfg.objective in ("sigreg", "ema", "ce"), cfg.objective
    self.cfg = cfg
    self._init_common(cfg.n_pose)
    assert cfg.input in ("tokens", "pixels"), cfg.input
    enc = PixelFrameEncoder if cfg.input == "pixels" else FrameEncoder
    self.encoder = enc(cfg.dim, cfg.enc_layers, cfg.enc_heads, cfg.proj_hidden,
                       **({"patch": cfg.patch} if cfg.input == "pixels" else {}))
    P = cfg.pred_dim
    self.pred_in = nn.Linear(cfg.dim, P)
    self.pos = nn.Parameter(torch.randn(cfg.frames, P) * 0.02)
    self.blocks = nn.ModuleList(ConditionalBlock(P, cfg.pred_heads, cfg.pred_mlp, cfg.pred_dropout)
                                for _ in range(cfg.pred_layers))
    self.pred_norm = nn.LayerNorm(P)
    self.pred_out = nn.Linear(P, cfg.dim)
    self.pred_proj = MLP(cfg.dim, cfg.proj_hidden, cfg.dim)
    self.act_embed = Embedder(cfg.n_pose, P)
    self.null_act = nn.Parameter(torch.randn(P) * 0.02)
    self.start = nn.Parameter(torch.randn(cfg.dim) * 0.02)
    self.decoder = TokenDecoder(cfg.dim, cfg.dec_dim, cfg.dec_layers, cfg.dec_heads, cfg.dec_kv)
    self.tgt = self.make_target() if cfg.objective == "ema" else None
    self.sigreg = SIGReg(cfg.knots, cfg.num_proj) if cfg.objective == "sigreg" else None

  # ---- interfaz de LatentWM
  def encode(self, frames, target=False):
    """frames: tokens (B, F, 128) o, con input="pixels", también imágenes uint8 (B, F, 128, 256, 3)."""
    enc = self.tgt if target else self.encoder
    if self.cfg.input == "pixels":
      return enc(frames if frames.dim() == 5 else render_tokens(frames))
    return enc(frames, self.codebook)

  def predict(self, z_in, acts):
    T = z_in.shape[1]
    x = self.pred_in(z_in) + self.pos[:T].to(z_in.dtype)
    c = acts.to(x.dtype)
    for b in self.blocks:
      x = b(x, c)
    return self.pred_proj(self.pred_out(self.pred_norm(x)))

  def readout(self, zh):
    return self.decoder(zh)

  @torch.no_grad()
  def targets(self, frames):
    return self.encode(frames, target=self.cfg.objective == "ema")

  # ---- entrenamiento
  def train_loss(self, frames, teacher=None, alpha=0.5, tau=1.0, poses=None):
    if teacher is not None:
      raise ValueError("LeWM no se destila: teacher tiene que ser None")
    z, zh = self.predict_frames(frames, poses)
    obj = self.cfg.objective
    z_std = z.detach().float().reshape(-1, z.shape[-1]).std(0).mean()
    if obj == "ce":
      loss, met = lm_loss(self.readout(zh), frames)
      return loss, {**met, "z_std": z_std}
    tz = z if obj == "sigreg" else self.targets(frames)  # LeWM: sin stop-grad
    pred = (zh.float() - tz.float()).square().mean()
    loss, met = pred, {"pred_mse": pred.detach()}
    if obj == "sigreg":
      sr = self.sigreg(z.reshape(1, -1, z.shape[-1]))  # los B·F embeddings de la ventana
      loss = loss + self.cfg.lam * sr
      met["sigreg"] = sr.detach()
    ce, _ = lm_loss(self.readout(zh.detach()), frames)
    ce_t, _ = lm_loss(self.readout(tz.detach()), frames)
    met.update(ce=ce.detach(), ce_techo=ce_t.detach(), z_std=z_std)
    return loss + ce + ce_t, met
