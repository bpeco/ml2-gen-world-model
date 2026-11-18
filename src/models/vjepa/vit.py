"""Encoder ViT y predictor de V-JEPA 2 (bloques de construcción; sin bucle de entrenamiento, EMA ni loss).

Fuente: github.com/facebookresearch/vjepa2 (main)
  - Encoder: src/models/vision_transformer.py (VisionTransformer, con use_rope=True como en la receta, yaml L84).
    Bloque pre-norm (src/models/utils/modules.py Block L505-568): x + attn(norm1(x)); x + mlp(norm2(x)); MLP fc1-GELU-fc2
    con mlp_ratio 4; qkv con bias; LayerNorm eps 1e-6 (vision_transformer.py L283); sin cls token; norm final (L109, L210-211).
    Sin posición absoluta: solo 3D-RoPE (L81-82). Init: trunc_normal std 0.02 en Linear y Conv3d, bias 0 (L130-145) y
    reescalado de attn.proj y mlp.fc2 por 1/sqrt(2*(i+1)) (L147-153).
  - Máscaras en el encoder: L190-193: se recortan los tokens con apply_masks y los ids planos de esos tokens se pasan a
    los bloques como posiciones RoPE (L203). Con una lista de máscaras, las salidas se concatenan en el batch
    (src/masks/utils.py L62-74, cada conjunto es una secuencia independiente).
  - Predictor: src/models/predictor.py (VisionTransformerPredictor): predictor_embed Linear(enc -> 384) (L57), mask token
    único inicializado en 0 (L64-66, zero_init_mask_tokens), contexto + mask tokens en las posiciones objetivo, orden
    por posición (L222-225), bloques con RoPE sobre esas posiciones (L236-240), predictor_norm (L241), selección de los
    tokens objetivo (L246-250) y predictor_proj Linear(384 -> enc) (L252). Receta: depth 12, 384, 12 cabezas, MLP 1536
    (yaml L77-85).
  - Los presets `tiny` (192, 12, 3) y `small` (384, 12, 6) existen en el código oficial (vision_transformer.py L334-359);
    el paper no los usa (el más chico es ViT-L, Tab. 12).

API (entradas ya normalizadas, ver patch.normalizar):
  enc = VJEPAEncoder(**PRESETS["tiny"]);  z = enc(x)                 x (B, 3, T, H, W) -> z (B, N, D)
  z = enc(x, masks)   masks: (B, K) o lista de S tensores (B, K_s) -> z (S*B, K, D)  (el clip se replica por conjunto)
  z = enc.encode_frames(x)   x (B, 3, T, H, W): cada frame como tubelet duplicado (V-JEPA 2-AC) -> (B, T*Hp*Wp, D)
  pred = VJEPAPredictor(embed_dim=D, grid_hw=(Hp, Wp));  p = pred(z_ctx, masks_ctx, masks_tgt)  -> (S*B, K_tgt, D)
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from src.models.vjepa.patch import PatchEmbed3D, clip_a_frames
from src.models.vjepa.rope import coords_desde_ids, ids_rejilla, rope_3d

FIDELIDAD = [
  dict(campo="Tamaño del encoder", paper="ViT-L: 1024, 24 capas, 16 cabezas (300M)",
       aquí="tiny 192/12/3 o small 384/12/6", categoría="presupuesto",
       fuente="paper Tab. 12; vision_transformer.py L334-359 (presets tiny/small del código oficial)"),
  dict(campo="Entrada", paper="16 frames 256x256 a 4 fps -> 8x16x16 = 2048 tokens",
       aquí="16 frames 128x256 (render del decoder de comma) -> 8x8x16 = 1024 tokens", categoría="datos",
       fuente="yaml L18-25"),
  dict(campo="Predictor", paper="12 capas, 384, 12 cabezas, MLP 1536, mask tokens en 0",
       aquí="igual", categoría="fiel", fuente="yaml L77-85; paper Tab. 12; predictor.py L64-66"),
  dict(campo="Mask tokens", paper="num_mask_tokens=2 por defecto; con una sola longitud de clip solo se usa el índice 0",
       aquí="1 mask token compartido por los dos conjuntos", categoría="fiel",
       fuente="predictor.py L42, L202-203; utils/wrappers.py L42 (mask_index = índice del grupo de fpc, no del conjunto)"),
  dict(campo="Posiciones RoPE del predictor", paper="decodifica ids con rejilla cuadrada (grid_size x grid_size) porque "
       "sus bloques se llaman sin H/W (predictor.py L240, modules.py L296-329)",
       aquí="se pasa la rejilla real (H=8, W=16), como ya hace su encoder",
       categoría="datos", fuente="modules.py L316-329; vision_transformer.py L178-181"),
  dict(campo="3D-RoPE", paper="expansión sin/cos con repeat (frecuencias duplicadas, bug conocido, compatible con "
       "los checkpoints)", aquí="oficial=True por defecto (igual); oficial=False aplica la corrección",
       categoría="fiel", fuente="modules.py L39-46"),
  dict(campo="Regularización del modelo", paper="sin dropout, sin drop-path, atención SDPA", aquí="igual (no implementados)",
       categoría="fiel", fuente="yaml (no define drop_rate/drop_path_rate); vision_transformer.py L35-37"),
  dict(campo="Máscaras", paper="A: 8 bloques escala 0.15; B: 2 bloques escala 0.7; aspecto [0.75, 1.5]; tiempo completo",
       aquí="igual sobre la rejilla 8x16 (bloques de menos tokens por la rejilla más chica)",
       categoría="fiel", fuente="yaml L41-67; multiseq_multiblock3d.py L145-269"),
  dict(campo="Normalización de entrada", paper="media/std de ImageNet sobre [0, 1]", aquí="igual",
       categoría="fiel", fuente="app/vjepa/transforms.py L21, L48, L139-152"),
]

PRESETS = {
  "tiny": dict(embed_dim=192, depth=12, num_heads=3),
  "small": dict(embed_dim=384, depth=12, num_heads=6),
}
PRED_PRESET = dict(predictor_dim=384, depth=12, num_heads=12)  # mlp 4x = 1536

LN = partial(nn.LayerNorm, eps=1e-6)


def apply_masks(x, masks):
  """x (B, N, D), masks lista de (B, K) -> (S*B, K, D); como src/masks/utils.py L62-74 (concat en la dim del batch)."""
  if torch.is_tensor(masks):
    masks = [masks]
  return torch.cat([torch.gather(x, 1, m.unsqueeze(-1).expand(-1, -1, x.size(-1))) for m in masks], 0)


def _cat(masks):
  return torch.cat(masks, 0) if isinstance(masks, (list, tuple)) else masks


class RoPEAttention(nn.Module):
  """Atención multi-cabeza con 3D-RoPE en q y k (modules.py L266-387)."""

  def __init__(self, dim, num_heads, oficial=True):
    super().__init__()
    self.num_heads = num_heads
    self.oficial = oficial
    self.qkv = nn.Linear(dim, dim * 3, bias=True)
    self.proj = nn.Linear(dim, dim)

  def _qkv(self, x, ids, H, W):
    q, k, v = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)  # (B, h, N, hd)
    q, k = rope_3d(q, k, coords_desde_ids(ids, H, W), self.oficial)
    return q, k, v

  def scores(self, x, ids, H, W):
    """Logits q·k/sqrt(hd) antes del softmax, (B, h, N, N) (para tests)."""
    q, k, _ = self._qkv(x, ids, H, W)
    return q @ k.transpose(-2, -1) * q.shape[-1] ** -0.5

  def forward(self, x, ids, H, W):
    B, N, C = x.shape
    q, k, v = self._qkv(x, ids, H, W)
    x = F.scaled_dot_product_attention(q, k, v)
    return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
  def __init__(self, dim, num_heads, mlp_ratio=4.0, oficial=True):
    super().__init__()
    self.norm1 = LN(dim)
    self.attn = RoPEAttention(dim, num_heads, oficial)
    self.norm2 = LN(dim)
    self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
    self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)

  def forward(self, x, ids, H, W):
    x = x + self.attn(self.norm1(x), ids, H, W)
    return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


def _run(blk, ckpt, x, ids, H, W):
  """Un bloque, con checkpointing de activaciones si se pide (yaml L82 use_activation_checkpointing) y hay gradiente."""
  if ckpt and torch.is_grad_enabled() and x.requires_grad:
    return torch.utils.checkpoint.checkpoint(blk, x, ids, H, W, use_reentrant=False)
  return blk(x, ids, H, W)


def _init(modelo, bloques, std=0.02):
  """Init de vision_transformer.py L130-153 / predictor.py L157-172."""
  for m in modelo.modules():
    if isinstance(m, (nn.Linear, nn.Conv3d)):
      nn.init.trunc_normal_(m.weight, std=std, a=-2.0, b=2.0)
      if m.bias is not None:
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
      nn.init.ones_(m.weight)
      nn.init.zeros_(m.bias)
  for i, b in enumerate(bloques):
    b.attn.proj.weight.data.div_(math.sqrt(2.0 * (i + 1)))
    b.fc2.weight.data.div_(math.sqrt(2.0 * (i + 1)))


class VJEPAEncoder(nn.Module):
  def __init__(self, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, patch_size=16, tubelet_size=2,
               in_chans=3, oficial=True):
    super().__init__()
    self.embed_dim, self.patch_size, self.tubelet_size = embed_dim, patch_size, tubelet_size
    self.patch_embed = PatchEmbed3D(patch_size, tubelet_size, in_chans, embed_dim)
    self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, oficial) for _ in range(depth)])
    self.norm = LN(embed_dim)
    self.grad_ckpt = False
    _init(self, self.blocks)

  @classmethod
  def preset(cls, nombre, **kw):
    return cls(**{**PRESETS[nombre], **kw})

  def forward(self, x, masks=None):
    """x (B, 3, T, H, W) normalizado; masks None | (B, K) | lista de (B, K_s). Devuelve (B, N, D) o (S*B, K, D)."""
    _, _, T, H, W = x.shape
    T, H, W = T // self.tubelet_size, H // self.patch_size, W // self.patch_size
    x = self.patch_embed(x)
    if masks is None:
      ids = ids_rejilla(T, H, W, x.device)[None]  # (1, N): posición de cada token en la rejilla completa
    else:
      masks = [masks] if torch.is_tensor(masks) else list(masks)
      x = apply_masks(x, masks)
      ids = torch.cat(masks, 0)
    for blk in self.blocks:
      x = _run(blk, self.grad_ckpt, x, ids, H, W)
    return self.norm(x)

  def encode_frames(self, x):
    """Ruta de V-JEPA 2-AC (train.py L408-412): x (B, 3, T, H, W) -> (B, T*Hp*Wp, D), cada frame como tubelet duplicado."""
    B, _, T = x.shape[:3]
    h = self(clip_a_frames(x, self.tubelet_size))  # (B*T, Hp*Wp, D)
    return h.view(B, T, -1, h.size(-1)).flatten(1, 2)


class VJEPAPredictor(nn.Module):
  """Predictor estrecho. grid_hw = (Hp, Wp) de la rejilla de tokens (para decodificar los ids planos)."""

  def __init__(self, embed_dim=192, grid_hw=(8, 16), predictor_dim=384, depth=12, num_heads=12, mlp_ratio=4.0,
               oficial=True):
    super().__init__()
    self.grid_hw = grid_hw
    self.predictor_embed = nn.Linear(embed_dim, predictor_dim)
    self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))  # en 0 (L64-66)
    self.blocks = nn.ModuleList([Block(predictor_dim, num_heads, mlp_ratio, oficial) for _ in range(depth)])
    self.norm = LN(predictor_dim)
    self.proj = nn.Linear(predictor_dim, embed_dim)
    self.grad_ckpt = False
    _init(self, self.blocks)

  def forward(self, z, masks_ctx, masks_tgt):
    """z (S*B, Nc, D): salida del encoder con masks_ctx. masks_ctx/masks_tgt: (S*B, K) o listas de S tensores (B, K_s)
    (se concatenan en el batch; cada conjunto es una secuencia independiente). Devuelve (S*B, Nt, D)."""
    mc, mt = _cat(masks_ctx), _cat(masks_tgt)
    x = self.predictor_embed(z)
    Nc = x.shape[1]
    x = torch.cat([x, self.mask_token.expand(len(mt), mt.shape[1], -1)], 1)
    ids = torch.cat([mc, mt], 1)
    orden = torch.argsort(ids, 1)  # L222-225
    ids = torch.gather(ids, 1, orden)
    x = torch.gather(x, 1, orden.unsqueeze(-1).expand(-1, -1, x.size(-1)))
    H, W = self.grid_hw
    for blk in self.blocks:
      x = _run(blk, self.grad_ckpt, x, ids, H, W)
    x = self.norm(x)
    x = torch.gather(x, 1, torch.argsort(orden, 1).unsqueeze(-1).expand(-1, -1, x.size(-1)))  # L248-249
    return self.proj(x[:, Nc:])
