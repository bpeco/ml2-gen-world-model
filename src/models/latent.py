"""Piezas comunes de los world models latentes: JEPA estilo V-JEPA (src/models/jepa.py) y
LeWorldModel (src/models/lewm.py).

SIGReg, ConditionalBlock (AdaLN-zero), Embedder y MLP siguen el código oficial de LeWorldModel
(github.com/lucas-maes/le-wm, module.py, licencia MIT): Maes, Le Lidec, Scieur, LeCun y Balestriero,
"LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels", arXiv 2603.19312.

Convenciones de las dos familias (además de las de base.py):
- Son FrameModel: frame_logits(frames, poses=None) predice el frame f desde los frames < f, con una
  lectura (cabeza o decoder) que pasa del latente predicho a los logits de las 128 celdas.
- Un frame "vacío" aprendido (start) abre la ventana: el frame 0 se predice sin contexto.
- Acción del frame f = pose del frame f (velocidades de traslación y giro, data/train_poses.npy). poses=None
  o NaN = acción nula aprendida. Al entrenar, con prob. p_null la pose se cambia por la nula: el mismo modelo
  corre sin poses (val, comparación justa con el resto) y con poses (holdout, planificación).
- Las subclases definen encode(frames, target=False) -> z (B, F, *S), predict(z_in, acts) -> ẑ (B, F, *S),
  readout(ẑ) -> (B, F, 128, 1024), targets(frames) -> lo que el predictor imita, y los módulos
  start, null_act, act_embed y encoder (más tgt si usan EMA).
"""
import copy
import functools
import math
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common import CTX_FRAMES
from src.models.base import FrameModel, sample_logits
from src.models.cudagraph import GraphedStep

CODEBOOK = "data/raw/decoder_pytorch_model.bin"


@functools.lru_cache(maxsize=1)
def _read_codebook(path: str) -> Optional[torch.Tensor]:
  p = Path(path)
  if not p.exists():
    return None
  sd = torch.load(p, map_location="cpu", weights_only=True)
  return sd["quantize._embedding.weight"].float()


def load_codebook(path: str = CODEBOOK) -> torch.Tensor:
  """(1024, 256) vectores del codebook del VQ-VAE de comma (congelados). Va como buffer en el checkpoint,
  así que cargar un modelo entrenado no necesita data/raw. Sin el archivo (tests), uno aleatorio fijo."""
  cb = _read_codebook(str(path))
  if cb is None:
    warnings.warn(f"{path} no existe: codebook aleatorio (solo sirve para tests o latencia)")
    return torch.randn(1024, 256, generator=torch.Generator().manual_seed(0))
  return cb.clone()


# ---------------------------------------------------------------- piezas de LeWorldModel (module.py)

class SIGReg(nn.Module):
  """Sketched Isotropic Gaussian Regularizer (LeJEPA / LeWM): empuja los embeddings hacia N(0, I).

  Proyecta sobre num_proj direcciones al azar (renovadas en cada llamada) y en cada una mide la distancia
  de Epps-Pulley entre la función característica empírica y la de N(0, 1), integrada con trapecios en
  `knots` nodos de [0, 3] con peso gaussiano. x: (..., N, D), N muestras. Se calcula en float32.

  Adaptación: el código original multiplica por N (el batch, 128 en el paper). Acá N varía entre familias
  (B·T embeddings en LeWM, una muestra de celdas en JEPA), así que se multiplica por n_ref = 128 fijo: λ
  significa lo mismo que en el paper. n_ref=None reproduce el original.
  """

  def __init__(self, knots: int = 17, num_proj: int = 1024, n_ref: Optional[int] = 128):
    super().__init__()
    self.num_proj, self.n_ref = num_proj, n_ref
    t = torch.linspace(0, 3, knots, dtype=torch.float32)
    dt = 3 / (knots - 1)
    trap = torch.full((knots,), 2 * dt)  # la integral es simétrica en t: 2 x [0, 3]
    trap[0] = trap[-1] = dt
    window = torch.exp(-t.square() / 2.0)  # función característica de N(0, 1) y peso
    self.register_buffer("t", t, persistent=False)
    self.register_buffer("phi", window, persistent=False)
    self.register_buffer("weights", trap * window, persistent=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    with torch.autocast(x.device.type, enabled=False):
      x = x.float()
      A = torch.randn(x.shape[-1], self.num_proj, device=x.device)
      A = A / A.norm(p=2, dim=0)
      xt = (x @ A).unsqueeze(-1) * self.t                     # (..., N, M, K)
      err = (xt.cos().mean(-3) - self.phi).square() + xt.sin().mean(-3).square()  # (..., M, K)
      n = self.n_ref if self.n_ref else x.shape[-2]
      return ((err @ self.weights) * n).mean()


class MLP(nn.Module):
  """Linear -> norma -> GELU -> Linear (projector y pred_proj de LeWM: con BatchNorm)."""

  def __init__(self, input_dim: int, hidden_dim: int, output_dim: Optional[int] = None, norm: str = "bn"):
    super().__init__()
    self.fc1 = nn.Linear(input_dim, hidden_dim)
    self.norm = nn.BatchNorm1d(hidden_dim) if norm == "bn" else nn.LayerNorm(hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, output_dim or input_dim)

  def forward(self, x):
    sh = x.shape
    x = self.fc2(F.gelu(self.norm(self.fc1(x.reshape(-1, sh[-1])))))
    return x.view(*sh[:-1], x.shape[-1])


def modulate(x, shift, scale):
  return x * (1 + scale) + shift


class ConditionalBlock(nn.Module):
  """Bloque transformer causal en el tiempo con AdaLN-zero: la condición c (la acción) genera shift, scale y
  gate de atención y MLP. La modulación arranca en cero, así que al inicializar el bloque es la identidad."""

  def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.0):
    super().__init__()
    assert dim % heads == 0
    self.heads, self.dropout = heads, dropout
    self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
    self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
    self.qkv = nn.Linear(dim, 3 * dim)
    self.proj = nn.Linear(dim, dim)
    self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(approximate="tanh"), nn.Dropout(dropout),
                             nn.Linear(mlp_dim, dim), nn.Dropout(dropout))
    self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
    nn.init.zeros_(self.ada[-1].weight)
    nn.init.zeros_(self.ada[-1].bias)

  def attn(self, x):
    B, T, C = x.shape
    q, k, v = self.qkv(x).view(B, T, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
    return self.proj(y.transpose(1, 2).reshape(B, T, C))

  def forward(self, x, c):
    s1, sc1, g1, s2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
    x = x + g1 * self.attn(modulate(self.norm1(x), s1, sc1))
    return x + g2 * self.mlp(modulate(self.norm2(x), s2, sc2))


class Embedder(nn.Module):
  """Acción -> embedding: lineal por paso (el Conv1d de kernel 1 del original) y MLP con SiLU."""

  def __init__(self, input_dim: int, emb_dim: int, mlp_scale: int = 4):
    super().__init__()
    self.patch = nn.Linear(input_dim, input_dim)
    self.mlp = nn.Sequential(nn.Linear(input_dim, mlp_scale * emb_dim), nn.SiLU(), nn.Linear(mlp_scale * emb_dim, emb_dim))

  def forward(self, a):
    return self.mlp(self.patch(a))


# ---------------------------------------------------------------- EMA y CEM

def ema_momentum(step: int, total: int, m0: float, m1: float) -> float:
  """Momento de la EMA: de m0 a m1 con coseno (V-JEPA / I-JEPA)."""
  prog = min(1.0, step / max(1, total))
  return m1 - (m1 - m0) * (math.cos(math.pi * prog) + 1) / 2


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, m: float):
  """target <- m * target + (1 - m) * online. Los buffers (estadísticas de BatchNorm) se copian."""
  for pt, po in zip(target.parameters(), online.parameters()):
    pt.lerp_(po.detach().to(pt.dtype), 1.0 - m)
  for bt, bo in zip(target.buffers(), online.buffers()):
    bt.copy_(bo)


@torch.no_grad()
def cem(cost_fn, batch: int, shape, n_samples: int = 300, n_elites: int = 30, iters: int = 10,
        init_std: float = 1.0, init_mean=None, device=None, generator=None) -> torch.Tensor:
  """Método de entropía cruzada (como en LeWM): minimiza cost_fn(cand (n_samples, batch, *shape)) -> costos
  (n_samples, batch). Arranca en N(init_mean (o 0), init_std²). Devuelve la media final, (batch, *shape)."""
  mean = torch.zeros(batch, *shape, device=device)
  if init_mean is not None:
    mean = mean + init_mean.to(mean)
  std = torch.full_like(mean, init_std)
  for _ in range(iters):
    eps = torch.randn(n_samples, batch, *shape, device=device, generator=generator)
    cand = mean + std * eps
    cost = cost_fn(cand)
    idx = cost.topk(n_elites, dim=0, largest=False).indices                       # (E, batch)
    elites = cand.gather(0, idx.view(n_elites, batch, *([1] * len(shape))).expand(n_elites, batch, *shape))
    mean, std = elites.mean(0), elites.std(0).clamp(min=1e-4)
  return mean


# ---------------------------------------------------------------- clase base

class LatentWM(FrameModel):
  """Base de jepa y lewm: acciones, generación con caché de latentes, EMA, features y planificación."""

  uses_poses = True
  ctx_frames = CTX_FRAMES - 1

  def _init_common(self, n_pose: int):
    self.register_buffer("codebook", load_codebook())
    self.register_buffer("pose_mean", torch.zeros(n_pose))
    self.register_buffer("pose_std", torch.ones(n_pose))
    self._graphs_on = False
    self._graphs = {}

  # ---- a definir por cada familia
  def encode(self, frames, target=False):
    raise NotImplementedError

  def predict(self, z_in, acts):
    raise NotImplementedError

  def readout(self, zh):
    raise NotImplementedError

  def targets(self, frames):
    raise NotImplementedError

  # ---- acciones
  def set_pose_stats(self, mean, std):
    self.pose_mean.copy_(torch.as_tensor(mean, dtype=self.pose_mean.dtype))
    self.pose_std.copy_(torch.as_tensor(std, dtype=self.pose_std.dtype))

  def normalize_poses(self, poses):
    """Poses crudas (..., 6) -> (normalizadas sin NaN, válidas (...) bool)."""
    p = poses.float()
    valid = torch.isfinite(p).all(-1)
    pn = torch.nan_to_num((p - self.pose_mean.float()) / self.pose_std.float())
    return pn, valid

  def actions_norm(self, pn, valid=None):
    """Poses normalizadas (B, T, 6) -> embeddings de acción; donde valid es False, la acción nula."""
    a = self.act_embed(pn.to(self.null_act.dtype))
    if valid is None:
      return a
    return torch.where(valid[..., None], a, self.null_act.to(a.dtype).expand_as(a))

  def actions(self, poses, B, T, device):
    """(B, T, Da). poses None -> acción nula en todos lados; al entrenar, dropout de acción con p_null."""
    if poses is None:
      return self.null_act.expand(B, T, -1)
    pn, valid = self.normalize_poses(poses.to(device))
    if self.training and self.cfg.p_null > 0:
      valid = valid & (torch.rand(B, T, device=device) >= self.cfg.p_null)
    return self.actions_norm(pn, valid)

  # ---- predicción
  def shift(self, z):
    """[start, z_0, ..., z_{F-2}]: la entrada del predictor que predice los frames 0..F-1."""
    start = self.start.to(z.dtype).expand(z.shape[0], 1, *self.start.shape)
    return torch.cat([start, z[:, :-1]], 1)

  def predict_frames(self, frames, poses=None):
    """(z, ẑ): latentes de los frames y su predicción desde los frames anteriores."""
    B, Fr = frames.shape[:2]
    z = self.encode(frames)
    return z, self.predict(self.shift(z), self.actions(poses, B, Fr, frames.device))

  def frame_logits(self, frames, poses=None):
    return self.readout(self.predict_frames(frames, poses)[1])

  def _predict_next(self, z, acts):
    """z (B, Fc, *S) contexto, acts (B, Fc + 1, Da) acción de cada frame -> ẑ del frame nuevo (B, *S)."""
    start = self.start.to(z.dtype).expand(z.shape[0], 1, *self.start.shape)
    return self.predict(torch.cat([start, z], 1), acts)[:, -1]

  def _next(self, z, acts):
    """(ẑ del frame nuevo (B, *S), logits (B, 128, 1024))."""
    zh = self._predict_next(z, acts)
    return zh, self.readout(zh[:, None])[:, 0]

  def _encode_one(self, tok):
    return self.encode(tok[:, None])[:, 0]

  def use_cuda_graphs(self, enable: bool = True):
    self._graphs_on = enable
    self._graphs = {}

  def _graphed(self, key, fn, *inputs):
    if not (self._graphs_on and inputs[0].device.type == "cuda"):
      return fn(*inputs)
    g = self._graphs.get(key)
    if g is None:
      g = self._graphs[key] = GraphedStep(fn, *inputs)
    return g(*inputs)

  @torch.no_grad()
  def generate(self, ctx, n_frames, temperature=1.0, steps=None, generator=None, poses=None):
    """Rollout con caché de latentes: cada frame se codifica una sola vez.

    poses: None (acción nula) o (B, F_ctx + n_frames, 6) crudas, alineadas con [contexto, frames nuevos].
    cfg.rollout = "token": el frame muestreado se re-codifica; "latent": entra ẑ al contexto (sin cuantizar).
    """
    if self.cfg.rollout == "latent" and getattr(self.cfg, "objective", None) == "codebook":
      raise ValueError("rollout latente no aplica al objetivo codebook (ẑ está en el espacio del codebook)")
    ctx = ctx[:, -self.ctx_frames:]
    B, Fc = ctx.shape[:2]
    if poses is not None:
      poses = poses[:, -(Fc + n_frames):]
    acts = self.actions(poses, B, Fc + n_frames, ctx.device).to(self.null_act.dtype)
    z = self.encode(ctx)
    j0 = 0
    for _ in range(n_frames):
      a = acts[:, j0:j0 + z.shape[1] + 1].contiguous()
      full = z.shape[1] == self.ctx_frames
      zh, lg = (self._graphed(("next", B), self._next, z, a) if full else self._next(z, a))
      new = sample_logits(lg.float(), temperature, generator)
      yield new
      zn = zh.clone() if self.cfg.rollout == "latent" else self._graphed(("enc", B), self._encode_one, new)
      z = torch.cat([z, zn[:, None].to(z.dtype)], 1)
      if z.shape[1] > self.ctx_frames:
        z, j0 = z[:, 1:], j0 + 1

  # ---- entrenamiento
  def after_step(self, step: int, total: int):
    if getattr(self, "tgt", None) is not None:
      ema_update(self.tgt, self.encoder, ema_momentum(step, total, *self.cfg.ema))

  def num_params(self) -> int:
    """Parámetros de inferencia: sin el encoder target (solo entrenamiento)."""
    return sum(p.numel() for n, p in self.named_parameters() if not n.startswith("tgt."))

  def make_target(self):
    """Copia EMA del encoder, sin gradiente (V-JEPA y la ablación lewm_s_ema)."""
    tgt = copy.deepcopy(self.encoder)
    for p in tgt.parameters():
      p.requires_grad_(False)
    return tgt

  # ---- evaluación de las representaciones
  @staticmethod
  def pool(z):
    """(B, F, *S) -> (B, F, D): media sobre las celdas si el latente es espacial."""
    return z.mean(-2) if z.dim() == 4 else z

  @torch.no_grad()
  def features(self, frames, poses=None):
    """(z, ẑ) por frame, promediados sobre las celdas: para los probes de pose."""
    z, zh = self.predict_frames(frames, poses)
    return self.pool(z).float(), self.pool(zh).float()

  @torch.no_grad()
  def latent_error(self, frames, poses=None):
    """(B, F): error cuadrático medio entre ẑ_f y el objetivo del frame f (la "sorpresa")."""
    _, zh = self.predict_frames(frames, poses)
    tz = self.targets(frames)
    return (zh.float() - tz.float()).square().flatten(2).mean(-1)

  @torch.no_grad()
  def plan_cem(self, ctx, goal, horizon: int, ctx_poses=None, history: int = 3, generator=None, **cem_kw):
    """Planifica con CEM las poses de los `horizon` frames siguientes para llegar al latente de `goal`.

    ctx (B, F, 128) contexto (se usan los últimos `history` frames, como en LeWM), goal (B, 128) frame
    objetivo, ctx_poses (B, F, 6) crudas o None. Costo: ‖ẑ_H − z_meta‖² (rollout latente). Devuelve las
    poses planeadas en unidades crudas, (B, horizon, 6).

    Adaptación: LeWM arranca el CEM en N(0, 1) en el espacio normalizado de acciones; acá, si hay poses del
    contexto, arranca en la última pose conocida (seguir como venía), que es la referencia natural al manejar.
    """
    ctx = ctx[:, -history:]
    B, h = ctx.shape[:2]
    z0 = self.encode(ctx)
    zg = self.targets(goal[:, None])[:, 0].float()
    init = None
    if ctx_poses is not None:
      pn, valid = self.normalize_poses(ctx_poses[:, -h:].to(ctx.device))
      a_ctx = self.actions_norm(pn, valid)
      init = torch.where(valid[:, -1:, None], pn[:, -1:], torch.zeros_like(pn[:, -1:]))  # (B, 1, 6)
    else:
      a_ctx = self.null_act.expand(B, h, -1)
    n_pose = self.pose_mean.shape[0]

    def cost(cand):  # (N, B, H, 6) normalizadas
      N = cand.shape[0]
      z = z0.unsqueeze(0).expand(N, *z0.shape).reshape(N * B, *z0.shape[1:])
      ac = a_ctx.unsqueeze(0).expand(N, *a_ctx.shape).reshape(N * B, h, -1)
      af = self.actions_norm(cand.reshape(N * B, horizon, n_pose))
      acts = torch.cat([ac, af], 1)  # acción de cada frame: [contexto, futuro]
      j0 = 0
      for _ in range(horizon):
        zh = self._predict_next(z, acts[:, j0:j0 + z.shape[1] + 1])
        z = torch.cat([z, zh[:, None].to(z.dtype)], 1)
        if z.shape[1] > history:
          z, j0 = z[:, 1:], j0 + 1
      d = (zh.float() - zg.unsqueeze(0).expand(N, *zg.shape).reshape(N * B, *zg.shape[1:])).square()
      return d.flatten(1).sum(-1).view(N, B)

    plan = cem(cost, B, (horizon, n_pose), init_mean=init, device=ctx.device, generator=generator, **cem_kw)
    return plan * self.pose_std.float() + self.pose_mean.float()
