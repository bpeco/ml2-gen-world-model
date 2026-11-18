"""V-JEPA 2: preentrenamiento autosupervisado (kind `vjepa2`). arXiv 2506.09985; código github.com/facebookresearch/vjepa2.

Un paso (app/vjepa/train.py L424-447, con los conjuntos de máscara de utils/wrappers.py):
  clip (16 frames a 4 fps = 1 de cada 5 de los 20 FPS) -> render con el decoder de comma congelado (GPU) -> recorte
  aleatorio + flip (aug.py) -> media/std de ImageNet -> (B, 3, 16, 128, 256).
  target  h = layer_norm_canal(target_encoder(clip completo))            sin gradiente (EMA del encoder)       L430-433
  por cada conjunto de máscara s: z_s = encoder(clip, contexto_s); p_s = predictor(z_s, contexto_s, objetivo_s)  L440-441
  loss = media_s( mean |p_s - h[objetivo_s]| )   L1 solo sobre los tokens enmascarados, promedio entre conjuntos  L439-449
  tras el optimizador: target <- m*target + (1-m)*encoder, m = 0.99925 constante                                 L299-301

FIDELIDAD_PRETRAIN y FIDELIDAD (de vit.py) listan cada decisión con su fuente y su categoría:
fiel (del paper/código), datos (lo obliga el dato de comma), presupuesto (lo obliga el cómputo) o propio.

Sincronizaciones con la CPU: ninguna en train_loss ni en el backward salvo la del banco de máscaras. Las formas de las
máscaras dependen del mínimo del lote (masks.py), así que cada `cfg.bank` llamadas a train_loss se sortean de una vez
`cfg.bank` lotes de máscaras y se lee UN vector de enteros (una sincronización por banco, no por paso).
"""
import copy
import dataclasses
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import pixels
from src import schedules as S
from src.models.base import WorldModel
from src.models.vjepa import aug
from src.models.vjepa.masks import MultiSeqMasks
from src.models.vjepa.patch import normalizar_cthw
from src.models.vjepa.vit import FIDELIDAD, PRESETS, VJEPAEncoder, VJEPAPredictor, apply_masks
from src.recipes import Recipe

PAPER = "V-JEPA 2 (arXiv 2506.09985)"
YAML = "configs/train/vitl16/pretrain-256px-16f.yaml"
WARM = 12_000 / 252_000  # fracción de calentamiento y de enfriamiento de la corrida de 252k pasos (Tab. 9)

FIDELIDAD_PRETRAIN = FIDELIDAD + [
  dict(campo="Optimizador", paper="AdamW β=(0.9, 0.999), eps 1e-8, sin grad-clip", aquí="igual", categoría="fiel",
       fuente=f"app/vjepa/utils.py L224-225, L243; train.py (sin clip); {YAML} L92"),
  dict(campo="Weight decay", paper="0.04 constante, sin wd en bias ni en parámetros 1-D", aquí="igual (wd_on=matrices: "
       "solo ndim >= 2)", categoría="fiel", fuente="app/vjepa/utils.py L205-216"),
  dict(campo="LR", paper="1e-4 → 5.25e-4 en 12k pasos de 252k, constante, enfriamiento lineal a 1e-6 en los últimos 12k",
       aquí="las mismas fracciones (12k/252k = 0.0476) del total de pasos de la corrida; piso de 100 pasos de warmup",
       categoría="fiel", fuente=f"{PAPER} Tab. 9; {YAML} L91-97"),
  dict(campo="Batch", paper="3072 clips", aquí="64 clips por paso (estimado para ~11 GB con checkpointing de "
       "activaciones, sin medir), sin acumulación", categoría="presupuesto", fuente=f"{PAPER} Tab. 9"),
  dict(campo="Pasos / presupuesto", paper="252k pasos", aquí="el presupuesto de la receta (8x los 780k frames de "
       "referencia ≈ 6.1k pasos); con tan pocos pasos el EMA de 0.99925 (constante de tiempo ~1.3k pasos) deja al "
       "target cerca de la inicialización al principio", categoría="presupuesto", fuente=f"{PAPER} Tab. 9"),
  dict(campo="Precisión", paper="bf16", aquí="bf16 (autocast del entrenador, solo CUDA)", categoría="fiel",
       fuente=f"{YAML} L69"),
  dict(campo="EMA del target", paper="0.99925 constante", aquí="igual, actualizado con _foreach sobre los parámetros "
       "en el device (after_step)", categoría="fiel", fuente=f"{YAML} L87-89; train.py L299-301"),
  dict(campo="Clip de entrada", paper="16 frames a 4 fps, recorte 256x256", aquí="16 frames = 1 de cada 5 de los 20 FPS "
       "(4 fps), frames de 128x256 renderizados del token con el decoder de comma", categoría="datos",
       fuente=f"{YAML} L18-25"),
  dict(campo="Aumentos", paper="random-resized-crop escala [0.3, 1], aspecto [0.75, 1.35] y flip horizontal",
       aquí="igual, en la GPU; el aspecto se mide respecto del aspecto 1:2 del frame (ver aug.py)",
       categoría="datos", fuente=f"{YAML} L29-38; transforms.py L470-543"),
  dict(campo="Checkpointing de activaciones", paper="use_activation_checkpointing: true", aquí="igual (cfg.grad_ckpt)",
       categoría="fiel", fuente=f"{YAML} L82"),
  dict(campo="Encoder final", paper="se evalúa el target_encoder (EMA)", aquí="igual: V-JEPA 2-AC carga target_encoder",
       categoría="fiel", fuente="configs/eval (checkpoint_key: target_encoder)"),
  dict(campo="Datos de preentrenamiento", paper="22M videos + imágenes de internet", aquí="clips de conducción de comma "
       "(solo se compara la arquitectura, no los datos)", categoría="datos", fuente=f"{PAPER} §2"),
]


@dataclass
class Config:
  encoder: str = "small"       # preset de vit.PRESETS ("tiny" | "small"); el paper usa ViT-L
  embed_dim: int = 0           # 0 = el del preset (los tests lo bajan)
  depth: int = 0
  num_heads: int = 0
  pred_dim: int = 384
  pred_depth: int = 12
  pred_heads: int = 12
  frames: int = 16             # frames por clip (a 4 fps)
  height: int = 128
  width: int = 256
  patch: int = 16
  tubelet: int = 2
  ema_momentum: float = 0.99925
  crop_scale: tuple = (0.3, 1.0)
  crop_ratio: tuple = (0.75, 1.35)
  flip: bool = True
  oficial_rope: bool = True    # RoPE como el código publicado (con su bug de frecuencias duplicadas)
  grad_ckpt: bool = True
  bank: int = 16               # lotes de máscaras por sorteo (una sincronización por banco)
  mask_seed: int = 0


def layer_norm_canal(h):
  """F.layer_norm sin parámetros sobre el canal (train.py L433, L447)."""
  return F.layer_norm(h, (h.size(-1),))


class VJEPA2(WorldModel):
  name = "vjepa2"
  has_likelihood = False
  uses_poses = False
  ctx_frames = 16

  Config = Config

  @classmethod
  def recipe(cls, size=None):
    return Recipe(
      name="vjepa2_oficial", fuente=f"{PAPER} Tab. 9; {YAML}", optimizer="adamw", betas=(0.9, 0.999), eps=1e-8,
      lr_segs=[S.linear(1e-4, 5.25e-4, frac=WARM, nombre="warmup"), S.constant(5.25e-4, nombre="constante"),
               S.linear(5.25e-4, 1e-6, frac=WARM, nombre="decay")],
      wd_ini=0.04, wd_fin=None, wd_on="matrices", ema_segs=None, grad_clip=None, amp="bf16",
      batch=64, frames=16, frameskip=5, accum=1, presupuesto=8 * 780_000, paper_steps=252_000,
      forzado=[("batch", "3072 clips es inalcanzable: 64 por paso (estimado para ~11 GB con checkpointing)", "presupuesto"),
               ("presupuesto", "6.1k pasos en vez de 252k; el warmup y el enfriamiento conservan sus fracciones", "presupuesto"),
               ("frames/frameskip", "16 frames a 4 fps = 1 de cada 5 de los 20 FPS", "datos"),
               ("encoder", "ViT small (384, 12 capas) en vez de ViT-L", "presupuesto")])

  def __init__(self, cfg: Config):
    super().__init__()
    self.cfg = cfg
    p = dict(PRESETS[cfg.encoder])
    for k in ("embed_dim", "depth", "num_heads"):
      if getattr(cfg, k):
        p[k] = getattr(cfg, k)
    kw = dict(patch_size=cfg.patch, tubelet_size=cfg.tubelet, oficial=cfg.oficial_rope)
    self.encoder = VJEPAEncoder(**p, **kw)
    self.predictor = VJEPAPredictor(embed_dim=p["embed_dim"], grid_hw=(cfg.height // cfg.patch, cfg.width // cfg.patch),
                                    predictor_dim=cfg.pred_dim, depth=cfg.pred_depth, num_heads=cfg.pred_heads,
                                    oficial=cfg.oficial_rope)
    self.target_encoder = copy.deepcopy(self.encoder)
    for q in self.target_encoder.parameters():
      q.requires_grad_(False)
    self.encoder.grad_ckpt = self.predictor.grad_ckpt = cfg.grad_ckpt
    self.grid = (cfg.frames // cfg.tubelet, cfg.height // cfg.patch, cfg.width // cfg.patch)
    self._sampler, self._bank, self._bank_key, self._refills = None, [], None, 0

  # ---- datos
  def num_params(self):
    return sum(q.numel() for q in self.encoder.parameters()) + sum(q.numel() for q in self.predictor.parameters())

  def clip(self, frames):
    """tokens (B, F, 128) o píxeles uint8 (B, F, H, W, 3) -> clip normalizado (B, 3, T, H, W), con aumentos si training."""
    c = self.cfg
    if frames.dim() == 3:
      with torch.no_grad():
        frames = pixels.render(frames[:, :c.frames])
    x = frames[:, :c.frames].float().div(255)  # (B, T, H, W, 3)
    B, T = x.shape[:2]
    x = x.permute(0, 1, 4, 2, 3).reshape(B, T * 3, *x.shape[2:4])
    if self.training:
      x = aug.recortar_y_voltear(x, c.crop_scale, c.crop_ratio, c.flip, (c.height, c.width))
    elif x.shape[-2:] != (c.height, c.width):
      x = F.interpolate(x, size=(c.height, c.width), mode="bilinear", align_corners=False)
    return normalizar_cthw(x.view(B, T, 3, c.height, c.width).permute(0, 2, 1, 3, 4))

  def masks(self, B, device):
    """(masks_enc, masks_pred): listas por conjunto de (B, K) long. Del banco; se rellena cada cfg.bank llamadas."""
    if not self.training:  # monitoreo: máscaras fijas, no tocan el banco de entrenamiento
      g = torch.Generator(device=device)
      g.manual_seed(1234)
      return MultiSeqMasks(self.grid, device=device)(B, g)
    if self._sampler is None or self._bank_key != (B, torch.device(device)):
      self._sampler, self._bank, self._bank_key = MultiSeqMasks(self.grid, device=device), [], (B, torch.device(device))
    if not self._bank:
      g = torch.Generator(device=device)
      g.manual_seed(self.cfg.mask_seed * 1_000_003 + self._refills)
      self._refills += 1
      self._bank = self._sampler.banco(self.cfg.bank, B, g)[::-1]  # única sincronización con la CPU
    return self._bank.pop()

  # ---- entrenamiento
  def train_loss(self, frames, teacher=None, alpha=0.5, tau=1.0, poses=None):
    if teacher is not None:
      raise ValueError("V-JEPA 2 no se destila: teacher tiene que ser None")
    x = self.clip(frames)
    m_enc, m_pred = self.masks(len(x), x.device)
    with torch.no_grad():
      h = layer_norm_canal(self.target_encoder(x))
    losses = []
    for me, mp in zip(m_enc, m_pred):
      z = self.encoder(x, me)
      p = self.predictor(z, me, mp)
      losses.append((p.float() - apply_masks(h, mp).float()).abs().mean())
    loss = torch.stack(losses).mean()
    met = {"l1": loss.detach(), "l1_A": losses[0].detach(), "l1_B": losses[-1].detach(),
           "target_std": h.detach().float().std(1).mean(), "pred_std": p.detach().float().std(1).mean()}
    return loss, met

  @torch.no_grad()
  def after_step(self, step, total):
    """EMA del target (L299-301): target <- m*target + (1-m)*encoder, con una sola llamada foreach."""
    tp = [q for q in self.target_encoder.parameters()]
    ep = [q.detach() for q in self.encoder.parameters()]
    m = self.cfg.ema_momentum
    torch._foreach_mul_(tp, m)
    torch._foreach_add_(tp, ep, alpha=1 - m)

  @torch.no_grad()
  def features(self, clip, target=True):
    """Representaciones del clip completo (B, N, D) con layer_norm; por defecto las del target encoder (el que se evalúa)."""
    return layer_norm_canal((self.target_encoder if target else self.encoder)(clip))


def config_dict(model):
  return dataclasses.asdict(model.cfg)

# Tabla de fidelidad completa de este modelo (la que leen src.recipes --fidelidad y las guías).
FIDELIDAD = FIDELIDAD_PRETRAIN
