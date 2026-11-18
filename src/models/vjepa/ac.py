"""Predictor condicionado a acciones de V-JEPA 2-AC (bloque-causal por frame; RoPE 3D en parches, temporal en acción/estado).

Fuente: github.com/facebookresearch/vjepa2
  src/models/ac_predictor.py (VisionTransformerPredictorAC):
    - L53-56 predictor_embed, action_encoder y state_encoder son Linear separados (afines, con bias) a predictor_dim.
    - L145-153 por frame se concatenan [acción, estado, parches] -> T*(2 + H*W) tokens.
    - L109-117 + modules.py L12-23 (build_action_block_causal_attention_mask): máscara bool por bloques de frame: el frame
      t1 ve todos los tokens de los frames t2 <= t1 (ventana = T). True = atiende (F.scaled_dot_product_attention).
    - L183-190 se descartan los tokens de acción y estado; predictor_norm y predictor_proj.
    - L119-134 init: trunc_normal std 0.02, bias 0; attn.proj y mlp.fc2 divididos por sqrt(2*(i+1)).
  src/models/utils/modules.py ACRoPEAttention (L114-263):
    - L138-140 d_dim = h_dim = w_dim = 2*((head_dim//3)//2) (como RoPEAttention); L174-181 y L212-234: los tokens de
      parches rotan [t | h | w] y el resto de head_dim queda sin rotar.
    - L179-181 "snap spatial positions to grid size": h *= grid_size/H y w *= grid_size/W, con grid_size = img_height //
      patch (ac_predictor.py L83, L68). En su receta la rejilla es cuadrada y el factor es 1; con nuestra rejilla 8x16 las
      posiciones w se escalan por 8/16 (se conserva tal cual: es lo que hace el código publicado con una entrada así).
    - L184-206 acción y estado: q y k se rotan SOLO en los primeros d_dim canales, con la posición temporal arange(T); v sin
      rotar; el resto de canales queda igual. Las mismas posiciones temporales para acción y estado de un mismo frame.
Sin dropout ni drop-path (receta del yaml); GELU, MLP 4x, LayerNorm eps 1e-6.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from src.models.vjepa.rope import coords_desde_ids, dim_rope, ids_rejilla, rope_3d, rotar
from src.models.vjepa.vit import LN


def mascara_bloque_causal(T, n_tok, device):
  """(T*n_tok, T*n_tok) bool: el token del frame t ve todos los tokens de los frames <= t (modules.py L12-23)."""
  f = torch.arange(T * n_tok, device=device) // n_tok
  return f[:, None] >= f[None, :]


class ACAttention(nn.Module):
  def __init__(self, dim, num_heads, grid_size, oficial=True):
    super().__init__()
    self.num_heads, self.grid_size, self.oficial = num_heads, grid_size, oficial
    self.qkv = nn.Linear(dim, dim * 3, bias=True)
    self.proj = nn.Linear(dim, dim)

  def forward(self, x, T, H, W, attn_mask):
    B, N, C = x.shape
    n_tok = 2 + H * W
    q, k, v = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)  # (B, h, N, hd)
    hd = q.shape[-1]
    d = dim_rope(hd)
    q, k = (t.reshape(B, self.num_heads, T, n_tok, hd) for t in (q, k))
    # acción y estado: temporal en los primeros d canales (L195-199)
    pos_t = torch.arange(T, device=x.device).repeat_interleave(2)
    qc, kc = (t[:, :, :, :2].reshape(B, self.num_heads, T * 2, hd) for t in (q, k))
    qc = torch.cat([rotar(qc[..., :d], pos_t, self.oficial), qc[..., d:]], -1)
    kc = torch.cat([rotar(kc[..., :d], pos_t, self.oficial), kc[..., d:]], -1)
    # parches: 3D-RoPE con las posiciones "snapeadas" a grid_size (L174-181)
    qp, kp = (t[:, :, :, 2:].reshape(B, self.num_heads, T * H * W, hd) for t in (q, k))
    pt, ph, pw = coords_desde_ids(ids_rejilla(T, H, W, x.device), H, W)
    qp, kp = rope_3d(qp, kp, (pt, ph * (self.grid_size / H), pw * (self.grid_size / W)), self.oficial)
    q = torch.cat([qc.view(B, self.num_heads, T, 2, hd), qp.view(B, self.num_heads, T, H * W, hd)], 3)
    k = torch.cat([kc.view(B, self.num_heads, T, 2, hd), kp.view(B, self.num_heads, T, H * W, hd)], 3)
    q, k = q.reshape(B, self.num_heads, N, hd), k.reshape(B, self.num_heads, N, hd)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return self.proj(y.transpose(1, 2).reshape(B, N, C))


class ACBlock(nn.Module):
  def __init__(self, dim, num_heads, grid_size, mlp_ratio=4.0, oficial=True):
    super().__init__()
    self.norm1, self.norm2 = LN(dim), LN(dim)
    self.attn = ACAttention(dim, num_heads, grid_size, oficial)
    self.fc1, self.fc2 = nn.Linear(dim, int(dim * mlp_ratio)), nn.Linear(int(dim * mlp_ratio), dim)

  def forward(self, x, T, H, W, attn_mask):
    x = x + self.attn(self.norm1(x), T, H, W, attn_mask)
    return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class ACPredictor(nn.Module):
  """x (B, T, N, E) representaciones de T frames, acciones (B, T, A) y estados (B, T, A) -> (B, T, N, E).

  La salida del frame t es la predicción del frame t+1 dado los frames <= t (con su acción a_t y estado s_t).
  grid_hw = (H, W) de la rejilla de parches de UN frame (la ruta de un frame duplicado da 8 x 16 con 128x256).
  """

  def __init__(self, embed_dim, action_dim=6, grid_hw=(8, 16), predictor_dim=384, depth=12, num_heads=12,
               mlp_ratio=4.0, oficial=True):
    super().__init__()
    self.grid_hw = grid_hw
    self.predictor_embed = nn.Linear(embed_dim, predictor_dim)
    self.action_encoder = nn.Linear(action_dim, predictor_dim)
    self.state_encoder = nn.Linear(action_dim, predictor_dim)
    self.blocks = nn.ModuleList([ACBlock(predictor_dim, num_heads, grid_hw[0], mlp_ratio, oficial) for _ in range(depth)])
    self.norm = LN(predictor_dim)
    self.proj = nn.Linear(predictor_dim, embed_dim)
    self.grad_ckpt = False
    for m in self.modules():  # L119-126
      if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02, a=-2.0, b=2.0)
        nn.init.zeros_(m.bias)
      elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    for i, b in enumerate(self.blocks):  # L128-134
      b.attn.proj.weight.data.div_(math.sqrt(2.0 * (i + 1)))
      b.fc2.weight.data.div_(math.sqrt(2.0 * (i + 1)))

  def forward(self, x, actions, states):
    B, T, N, _ = x.shape
    H, W = self.grid_hw
    assert N == H * W, f"se esperaban {H * W} tokens por frame, llegaron {N}"
    x = self.predictor_embed(x)
    a = self.action_encoder(actions.to(x.dtype))[:, :, None]
    s = self.state_encoder(states.to(x.dtype))[:, :, None]
    x = torch.cat([a, s, x], 2).flatten(1, 2)  # (B, T*(2+N), D)
    mask = mascara_bloque_causal(T, 2 + N, x.device)
    for blk in self.blocks:
      if self.grad_ckpt and torch.is_grad_enabled() and x.requires_grad:
        x = torch.utils.checkpoint.checkpoint(blk, x, T, H, W, mask, use_reentrant=False)
      else:
        x = blk(x, T, H, W, mask)
    x = x.view(B, T, 2 + N, -1)[:, :, 2:]
    return self.proj(self.norm(x))
