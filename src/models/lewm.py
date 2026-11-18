"""LeWorldModel (LeWM): un embedding por frame, SIGReg y un predictor AdaLN-zero, entrenado de punta a punta desde píxeles.

Reescritura fiel del código oficial (Maes, Le Lidec, Scieur, LeCun y Balestriero, "LeWorldModel: Stable End-to-End
Joint-Embedding Predictive Architecture from Pixels", arXiv 2603.19312; github.com/lucas-maes/le-wm, MIT: module.py,
jepa.py, train.py y config/). Lo que sigue es lo oficial; lo que cambia por los datos o el cómputo está en FIDELIDAD:
- Encoder: ViT-tiny (12 capas, 3 cabezas, 192; HF ViT: pre-LN con eps 1e-12, LayerNorm final) sin preentrenar; el
  embedding del frame es el CLS de la última capa -> projector MLP(192 -> 2048, BatchNorm1d, GELU -> 192).
- Predictor ARPredictor: 6 capas AdaLN-zero (16 cabezas x 64, MLP 2048, dropout 0.1, QKV sin bias), posición aprendida,
  causal, LayerNorm final -> pred_proj (otro MLP igual con BatchNorm).
- Acción = las k poses de cada frame tomado apiladas (6·k), Embedder = Conv1d(k=1) + MLP (4x, SiLU).
- Muestras de history + num_preds = 4 frames a frameskip 5 (4 Hz). ctx = z[:, :3] con sus acciones, objetivo = z[:, 1:]:
  loss = MSE(pred, objetivo) + λ·SIGReg(z) con λ = 0.09, sin stop-grad ni EMA. SIGReg se aplica por paso de tiempo a
  través del batch (tensor (T, B, D)) con 1024 proyecciones nuevas en cada forward.
- Planificación: CEM (300 muestras, 30 élites, horizonte 5, 10 iteraciones, varianza inicial 1) sobre las acciones
  normalizadas, costo = ‖ẑ_final − z_meta‖² (config/eval/solver/cem.yaml y JEPA.criterion).

No hay lectura a tokens ni a píxeles dentro del modelo (el decoder de visualización se entrena aparte, sobre el modelo
congelado: src/models/lewm_decoder.py, src/train_decoder.py). LeWM no emite tokens: no tiene CE ni verosimilitud.

Todo el camino de entrenamiento y de inferencia vive en el device: sin .item/.cpu/.numpy/.tolist (las proyecciones de
SIGReg se generan en el device).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import schedules as S
from src.models.latent import LatentWM
from src.pixels import render
from src.recipes import Recipe

LEWM = "LeWM (arXiv 2603.19312)"
REPO = "github.com/lucas-maes/le-wm"
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# El render con el decoder de comma congelado vive en src/pixels.py (compartido); alias por compatibilidad.
render_tokens = render


# ---------------------------------------------------------------- piezas de module.py

class SIGReg(nn.Module):
  """Sketched Isotropic Gaussian Regularizer (module.py L10-36). proj: (T, B, D), se aplica por paso de tiempo a
  través del batch: 1024 direcciones nuevas en cada llamada (generadas en el device), 17 nodos en [0, 3], trapecio con la
  ventana exp(-t²/2), multiplicado por B y promediado sobre tiempo y proyecciones. Se calcula en float32."""

  def __init__(self, knots: int = 17, num_proj: int = 1024):
    super().__init__()
    self.num_proj = num_proj
    t = torch.linspace(0, 3, knots, dtype=torch.float32)
    dt = 3 / (knots - 1)
    weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
    weights[[0, -1]] = dt
    window = torch.exp(-t.square() / 2.0)
    self.register_buffer("t", t, persistent=False)
    self.register_buffer("phi", window, persistent=False)
    self.register_buffer("weights", weights * window, persistent=False)

  def forward(self, proj: torch.Tensor) -> torch.Tensor:
    with torch.autocast(proj.device.type, enabled=False):
      proj = proj.float()
      A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
      A = A.div_(A.norm(p=2, dim=0))
      x_t = (proj @ A).unsqueeze(-1) * self.t                                    # (T, B, M, K)
      err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()  # (T, M, K)
      statistic = (err @ self.weights) * proj.size(-2)
      return statistic.mean()


def modulate(x, shift, scale):
  """AdaLN-zero."""
  return x * (1 + scale) + shift


class FeedForward(nn.Module):
  def __init__(self, dim, hidden_dim, dropout=0.0):
    super().__init__()
    self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                             nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

  def forward(self, x):
    return self.net(x)


class Attention(nn.Module):
  """Atención causal con su propio LayerNorm de entrada y QKV sin bias (como module.py)."""

  def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
    super().__init__()
    inner = dim_head * heads
    self.heads, self.dropout = heads, dropout
    self.norm = nn.LayerNorm(dim)
    self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
    self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

  def forward(self, x, causal=True):
    B, T, _ = x.shape
    x = self.norm(x)
    q, k, v = self.to_qkv(x).view(B, T, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=causal)
    return self.to_out(out.transpose(1, 2).reshape(B, T, -1))


class ConditionalBlock(nn.Module):
  """Bloque con AdaLN-zero: la condición c (la acción) da shift, scale y gate de la atención y del MLP; la modulación
  arranca en cero, así que al inicializar el bloque es la identidad."""

  def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
    super().__init__()
    self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
    self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
    self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
    self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
    self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
    nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
    nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

  def forward(self, x, c):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
    x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
    return x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))


class Transformer(nn.Module):
  """Pila de ConditionalBlock + LayerNorm final. Las tres dimensiones (entrada, oculta, salida) son iguales en LeWM,
  así que input_proj, cond_proj y output_proj de module.py son Identity y no están."""

  def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
    super().__init__()
    self.norm = nn.LayerNorm(dim)
    self.layers = nn.ModuleList([ConditionalBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)])

  def forward(self, x, c):
    for block in self.layers:
      x = block(x, c)
    return self.norm(x)


class Embedder(nn.Module):
  """Acción apilada (B, T, input_dim) -> (B, T, emb_dim): Conv1d de kernel 1 a smoothed_dim y MLP (4x, SiLU)."""

  def __init__(self, input_dim, smoothed_dim, emb_dim, mlp_scale=4):
    super().__init__()
    self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
    self.embed = nn.Sequential(nn.Linear(smoothed_dim, mlp_scale * emb_dim), nn.SiLU(),
                               nn.Linear(mlp_scale * emb_dim, emb_dim))

  def forward(self, x):
    x = x.to(self.patch_embed.weight.dtype).permute(0, 2, 1)  # el oficial hace x.float(): igual en fp32
    return self.embed(self.patch_embed(x).permute(0, 2, 1))


class MLP(nn.Module):
  """Linear -> BatchNorm1d -> GELU -> Linear (projector y pred_proj). La BatchNorm ve todos los frames del batch."""

  def __init__(self, input_dim, hidden_dim, output_dim=None):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(),
                             nn.Linear(hidden_dim, output_dim or input_dim))

  def forward(self, x):
    sh = x.shape
    y = self.net(x.reshape(-1, sh[-1]))
    return y.view(*sh[:-1], y.shape[-1])


class ARPredictor(nn.Module):
  """Predictor autorregresivo del embedding siguiente: posición aprendida (randn, como el oficial) + Transformer."""

  def __init__(self, *, num_frames, depth, heads, mlp_dim, dim, dim_head=64, dropout=0.0, emb_dropout=0.0):
    super().__init__()
    self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
    self.dropout = nn.Dropout(emb_dropout)
    self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

  def forward(self, x, c):
    """x (B, T, D), c (B, T, D) acciones embebidas."""
    x = x + self.pos_embedding[:, :x.size(1)].to(x.dtype)
    return self.transformer(self.dropout(x), c)


# ---------------------------------------------------------------- encoder: ViT-tiny (HF ViT)

class ViTBlock(nn.Module):
  def __init__(self, dim, heads, mlp, eps=1e-12):
    super().__init__()
    self.heads = heads
    self.ln1, self.ln2 = nn.LayerNorm(dim, eps=eps), nn.LayerNorm(dim, eps=eps)
    self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
    self.fc1, self.fc2 = nn.Linear(dim, mlp), nn.Linear(mlp, dim)

  def forward(self, x):
    B, T, C = x.shape
    q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
    x = x + self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, C))
    return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class ViT(nn.Module):
  """ViT sin preentrenar (init de HF: N(0, 0.02) truncada) que devuelve el CLS de la última capa (tras la LayerNorm final)."""

  def __init__(self, dim, layers, heads, mlp, patch, grid):
    super().__init__()
    self.patch_embed = nn.Conv2d(3, dim, patch, stride=patch)
    self.cls = nn.Parameter(torch.zeros(1, 1, dim))
    self.pos = nn.Parameter(torch.zeros(1, 1 + grid[0] * grid[1], dim))
    self.blocks = nn.ModuleList(ViTBlock(dim, heads, mlp) for _ in range(layers))
    self.norm = nn.LayerNorm(dim, eps=1e-12)
    self.apply(self._init)
    for p in (self.cls, self.pos):
      nn.init.trunc_normal_(p, std=0.02)

  @staticmethod
  def _init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
      nn.init.trunc_normal_(m.weight, std=0.02)
      nn.init.zeros_(m.bias)

  def forward(self, x):
    """x (N, 3, H, W) normalizada -> CLS (N, dim)."""
    e = self.patch_embed(x).flatten(2).transpose(1, 2)
    x = torch.cat([self.cls.to(e.dtype).expand(len(e), -1, -1), e], 1) + self.pos.to(e.dtype)
    for b in self.blocks:
      x = b(x)
    return self.norm(x)[:, 0]


# ---------------------------------------------------------------- CEM

@torch.no_grad()
def cem(cost_fn, batch: int, shape, n_samples: int = 300, n_elites: int = 30, iters: int = 10,
        init_std: float = 1.0, init_mean=None, device=None, generator=None) -> torch.Tensor:
  """Método de entropía cruzada (config/eval/solver/cem.yaml): minimiza cost_fn(cand (n_samples, batch, *shape)) ->
  costos (n_samples, batch). Arranca en N(init_mean (o 0), init_std²); init_std es un escalar o un tensor que se
  difunde sobre `shape` (el desvío de cada dimensión de la acción). Devuelve la media final, (batch, *shape)."""
  mean = torch.zeros(batch, *shape, device=device)
  if init_mean is not None:
    mean = mean + init_mean.to(mean)
  std = torch.zeros_like(mean) + torch.as_tensor(init_std, dtype=mean.dtype, device=mean.device)  # escalar o por dimensión
  for _ in range(iters):
    eps = torch.randn(n_samples, batch, *shape, device=device, generator=generator)
    cand = mean + std * eps
    cost = cost_fn(cand)
    idx = cost.topk(n_elites, dim=0, largest=False).indices                       # (E, batch)
    elites = cand.gather(0, idx.view(n_elites, batch, *([1] * len(shape))).expand(n_elites, batch, *shape))
    mean, std = elites.mean(0), elites.std(0).clamp(min=1e-4)
  return mean


# ---------------------------------------------------------------- el modelo

FIDELIDAD = [
  dict(campo="Encoder", fuente_valor="ViT-tiny: 12 capas, 3 cabezas, 192; CLS de la última capa; sin preentrenar",
       aquí="igual (HF ViT: pre-LN, eps 1e-12, LayerNorm final, init N(0, 0.02) truncada)", categoría="fiel",
       fuente=f"{LEWM} §3; {REPO} config/train/model/lewm.yaml L3-9"),
  dict(campo="Imagen y parche", fuente_valor="224 x 224, parche 14 (grilla 16 x 16)",
       aquí="RGB 128 x 256, parche 16 (grilla 8 x 16): lo que da la cámara de comma", categoría="datos",
       fuente=f"{REPO} lewm.yaml L3-9 (img_size 224)"),
  dict(campo="Normalización de la imagen", fuente_valor="media y desvío de ImageNet", aquí="igual", categoría="fiel",
       fuente=f"{REPO} utils.py get_img_preprocessor"),
  dict(campo="Origen de los frames", fuente_valor="video RGB", aquí="el dataset solo tiene tokens VQ: los frames se "
       "renderizan al vuelo con el decoder de comma congelado (fp16, sin gradiente, en el device); el modelo ve las "
       "reconstrucciones del VQ-VAE", categoría="datos", fuente="commaVQ; src/pixels.py"),
  dict(campo="Projector y pred_proj", fuente_valor="Linear(192 → 2048), BatchNorm1d, GELU, Linear(2048 → 192); otro "
       "igual después del predictor", aquí="igual", categoría="fiel", fuente=f"{REPO} lewm.yaml L29-45; module.py MLP"),
  dict(campo="Predictor", fuente_valor="ARPredictor: 6 capas, 16 cabezas x dim_head 64 (inner 1024), MLP 2048, dropout "
       "0.1, emb_dropout 0, dim 192, QKV sin bias, AdaLN-zero (6 vectores con gate), posición aprendida, causal, "
       "LayerNorm final", aquí="igual", categoría="fiel", fuente=f"{REPO} lewm.yaml L11-22; module.py L68, L88-111, L244-285"),
  dict(campo="Embedder de la acción", fuente_valor="Conv1d(k=1) a smoothed_dim 10 (el default de module.py) + MLP 4x con "
       "SiLU; input_dim = frameskip x action_dim", aquí="igual con input_dim = 5 x 6 = 30", categoría="fiel",
       fuente=f"{REPO} module.py L189-214; data/pusht.yaml L3"),
  dict(campo="Acción", fuente_valor="acciones de PushT (2D), z-score por columna; NaN → 0",
       aquí="las 6 velocidades de traslación y giro de la pose de comma por frame crudo, z-score con las estadísticas de "
       "train (buffers del checkpoint); NaN → 0", categoría="datos", fuente=f"{REPO} train.py L24, utils.py"),
  dict(campo="Historia y predicciones", fuente_valor="history_size 3, num_preds 1 (4 frames por muestra)",
       aquí="igual", categoría="fiel", fuente=f"{REPO} lewm.yaml L15, L36; train.py L32-36"),
  dict(campo="Frameskip", fuente_valor="5", aquí="5: de 20 fps a 4 Hz (un paso del modelo = 250 ms)", categoría="fiel",
       fuente=f"{REPO} data/pusht.yaml L3"),
  dict(campo="Loss", fuente_valor="MSE(pred, objetivo) + λ·SIGReg, λ 0.09, sin stop-grad ni EMA", aquí="igual",
       categoría="fiel", fuente=f"{REPO} train.py L39-41; lewm.yaml L38-40"),
  dict(campo="SIGReg", fuente_valor="(T, B, D) por paso de tiempo a través del batch; 1024 proyecciones nuevas por "
       "forward; 17 nodos en [0, 3]; trapecio con ventana exp(-t²/2); por B; promedio sobre tiempo y proyecciones",
       aquí="igual; en float32 y con las proyecciones generadas en el device", categoría="fiel",
       fuente=f"{REPO} module.py L10-36 (la precisión: propio; el oficial corre bajo autocast bf16)"),
  dict(campo="Optimizador", fuente_valor="AdamW lr 5e-5, wd 1e-3, betas por defecto", aquí="igual, wd en todos los "
       "parámetros", categoría="fiel", fuente=f"{REPO} lewm.yaml L31-34"),
  dict(campo="Variante lewm_lrx3", fuente_valor="lr 5e-5", aquí="la corrida fiel (lewm) usa el lr del paper y con ~1.500 pasos "
       "apenas superó a copiar el último latente. El humo de 300 pasos (fuente, x3, x10, x30: monitor 0,261 / 0,222 / 0,250 / "
       "0,266, sin colapso; results/humo_lewm.json) eligió x3, así que se entrena además lewm_lrx3 con "
       "--set receta.lr_peak=1.5e-4 (misma forma: warmup 1 % y coseno a 0). Las dos van a la tabla", categoría="presupuesto",
       fuente=f"{REPO} lewm.yaml L31-34"),
  dict(campo="Schedule", fuente_valor="LinearWarmupCosineAnnealingLR por época: warmup 1 % desde 0, coseno a 0",
       aquí="igual pero por paso: warmup lineal sobre el 1 % de los pasos (piso 1) y coseno a 0", categoría="presupuesto",
       fuente=f"{REPO} train.py L87-94; con 1 500 pasos no hay épocas que contar"),
  dict(campo="Precisión y clip", fuente_valor="bf16, clip 1.0", aquí="igual (bf16 solo en CUDA)", categoría="fiel",
       fuente=f"{REPO} lewm.yaml L17-29"),
  dict(campo="Batch", fuente_valor="128 muestras de 4 frames", aquí="128 x 4, sin acumulación", categoría="fiel",
       fuente=f"{REPO} lewm.yaml L17-29; si la GPU de 12 GB no alcanza, accum rompe la BatchNorm y SIGReg: ajustar bs"),
  dict(campo="Presupuesto", fuente_valor="100 épocas en la config, 10 en el paper (PushT)", aquí="780k frames "
       "renderizados vistos (~1 500 pasos de 128 x 4)", categoría="presupuesto", fuente=f"{LEWM} App. E; {REPO} lewm.yaml"),
  dict(campo="Datos de las muestras", fuente_valor="trayectorias de PushT, split 90/10 por episodio", aquí="ventanas de "
       "20 frames crudos (4 a frameskip 5) de los segmentos de comma; holdout aparte", categoría="datos",
       fuente="src/windows.py"),
  dict(campo="CEM", fuente_valor="300 muestras, 30 élites, horizonte 5, 10 iteraciones (non-PushT), media 0 y "
       "varianza inicial 1 en el espacio normalizado", aquí="igual; la acción de cada paso son las 5 poses apiladas "
       "(30 dims)", categoría="fiel", fuente=f"{REPO} config/eval/solver/cem.yaml, config/eval/pusht.yaml; {LEWM} App. D"),
  dict(campo="Costo del CEM", fuente_valor="MSE sumado del embedding final predicho contra el de la meta",
       aquí="igual: rollout latente de `horizonte` pasos con la historia de 3", categoría="fiel", fuente=f"{REPO} jepa.py criterion"),
  dict(campo="Elites del CEM", fuente_valor="no leído del código del solver (stable-worldmodel)", aquí="media y desvío "
       "de los élites, desvío con piso 1e-4", categoría="propio", fuente="CEM estándar"),
  dict(campo="Evaluación latente", fuente_valor="el planificador (CEM, arriba); lo demás no está en las fuentes leídas",
       aquí="src/eval_latent.py: probes de velocidad y giro contra un encoder sin entrenar y contra píxeles; error latente "
       "por horizonte contra copiar el último latente; sorpresa; recuperación de acciones con CEM contra repetir la última "
       "acción. Los detalles propios van en el bloque `propio` del JSON", categoría="propio", fuente="src/eval_latent.py"),
  dict(campo="Cosas propias que se quitaron", fuente_valor="no existen", aquí="sin dropout de acción ni acción nula, sin "
       "frame start aprendido, sin rollout de tokens, sin lectura a tokens ni CE", categoría="fiel", fuente=f"{REPO} jepa.py"),
  dict(campo="Streaming", fuente_valor="no existe (evaluación offline en PushT)", aquí="LatentStream: cámara → ViT → "
       "predictor en la GPU, un paso del modelo cada 5 frames de cámara, sin codec", categoría="propio", fuente="src/infer/stream.py"),
  dict(campo="Decoder de visualización", fuente_valor="transformer liviano: CLS → cross-attention con una query "
       "aprendida por parche → parches; solo para visualizar; código no publicado", aquí="src/models/lewm_decoder.py, "
       "entrenado aparte sobre el modelo congelado (profundidad, ancho, MSE y optimizador: propio)", categoría="propio",
       fuente=f"{LEWM} App. D"),
  dict(campo="Tamaño xs", fuente_valor="no existe", aquí="reducción propia (dim 96, encoder de 6 capas, predictor de 4 "
       "capas y 8 cabezas x 32): opcional, el oficial es base", categoría="propio", fuente="Config.sized('xs')"),
]


def _xs():
  return dict(dim=96, enc_layers=6, enc_heads=3, enc_mlp=384, proj_hidden=1024, pred_layers=4, pred_heads=8,
              pred_dim_head=32, pred_mlp=1024)


SIZES_LEWM = {"base": {}, "xs": _xs()}


class LeWM(LatentWM):
  name = "lewm"
  has_likelihood = False

  @dataclass
  class Config:
    dim: int = 192             # embedding por frame
    enc_layers: int = 12       # ViT-tiny
    enc_heads: int = 3
    enc_mlp: int = 768
    patch: int = 16            # oficial 14 sobre 224; aquí 16 sobre 128 x 256 (grilla 8 x 16)
    img_h: int = 128
    img_w: int = 256
    proj_hidden: int = 2048
    pred_layers: int = 6
    pred_heads: int = 16
    pred_dim_head: int = 64
    pred_mlp: int = 2048
    pred_dropout: float = 0.1
    emb_dropout: float = 0.0
    act_smooth: int = 10       # smoothed_dim del Embedder (default de module.py)
    act_mlp_scale: int = 4
    history: int = 3           # history_size
    num_preds: int = 1
    frameskip: int = 5         # frames crudos por paso del modelo (la receta lo pasa a Windows; train.py fija n_pose)
    n_pose: int = 30           # 6 poses x frameskip
    lam: float = 0.09
    num_proj: int = 1024
    knots: int = 17

    @classmethod
    def sized(cls, size: str = "base", **kw) -> "LeWM.Config":
      return cls(**{**SIZES_LEWM[size], **kw})

  @classmethod
  def recipe(cls, size=None):
    """Receta del repo oficial (lewm.yaml, train.py). `size` ("base" o "xs") no cambia la receta."""
    return Recipe(
      name="lewm_paper", fuente=f"{LEWM}; {REPO} config/train/lewm.yaml, train.py", optimizer="adamw",
      betas=(0.9, 0.999), eps=1e-8,
      lr_segs=[S.linear(0.0, 5e-5, frac=0.01, nombre="warmup", min_pasos=1), S.cosine(5e-5, 0.0)],
      wd_ini=1e-3, wd_on="todos", grad_clip=1.0, amp="bf16", batch=128, accum=1, frames=4, frameskip=5,
      presupuesto=780_000, paper_steps=None, batch_fuente=128,
      forzado=[("lr_segs", "el oficial planifica por época (warmup en pasos); acá todo por paso: warmup lineal sobre el 1 % de "
                "los pasos (piso 1, sin el piso de 100) y coseno a 0", "presupuesto"),
               ("presupuesto", "780k frames renderizados vistos (~1 500 pasos); el oficial son 10-100 épocas de PushT", "presupuesto"),
               ("batch", "128 muestras de 4 frames sin acumulación (BatchNorm y SIGReg dependen del batch); si la GPU no "
                "alcanza, bajar bs en vez de acumular", "presupuesto"),
               ("frames", "4 por muestra (history 3 + num_preds 1): la unidad de presupuesto son los frames renderizados", "datos")])

  def __init__(self, cfg: "LeWM.Config"):
    super().__init__()
    self.cfg = cfg
    self.ctx_frames = cfg.history
    self._init_common(cfg.n_pose)
    grid = (cfg.img_h // cfg.patch, cfg.img_w // cfg.patch)
    self.encoder = ViT(cfg.dim, cfg.enc_layers, cfg.enc_heads, cfg.enc_mlp, cfg.patch, grid)
    self.projector = MLP(cfg.dim, cfg.proj_hidden, cfg.dim)
    self.predictor = ARPredictor(num_frames=cfg.history, depth=cfg.pred_layers, heads=cfg.pred_heads,
                                 mlp_dim=cfg.pred_mlp, dim=cfg.dim, dim_head=cfg.pred_dim_head, dropout=cfg.pred_dropout,
                                 emb_dropout=cfg.emb_dropout)
    self.action_encoder = Embedder(cfg.n_pose, cfg.act_smooth, cfg.dim, cfg.act_mlp_scale)
    self.pred_proj = MLP(cfg.dim, cfg.proj_hidden, cfg.dim)
    self.sigreg = SIGReg(cfg.knots, cfg.num_proj)
    self.register_buffer("img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
    self.register_buffer("img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

  # ---- encoder y predictor (jepa.py)
  def encode(self, frames):
    """frames: imágenes uint8 (B, F, 128, 256, 3) o tokens (B, F, 128), que se renderizan primero -> z (B, F, D)."""
    img = frames if frames.dim() == 5 else render_tokens(frames)
    B, Fr, H, W, _ = img.shape
    x = img.reshape(B * Fr, H, W, 3).permute(0, 3, 1, 2).to(self.encoder.patch_embed.weight.dtype) / 255
    x = (x - self.img_mean.to(x.dtype)) / self.img_std.to(x.dtype)
    return self.projector(self.encoder(x)).view(B, Fr, -1)

  def predict(self, z_in, acts):
    """z_in (B, T, D) y acts (B, T, D) acciones embebidas -> ẑ (B, T, D): la predicción del frame siguiente a cada uno."""
    return self.pred_proj(self.predictor(z_in, acts.to(z_in.dtype)))

  def action_embed(self, poses, B, T, device):
    """Poses crudas (B, T, n_pose) -> embeddings (B, T, D). None = acción 0 (la media). Acepta (B, T, 6) y las repite
    frameskip veces (las 6 poses de un frame crudo)."""
    if poses is None:
      return self.action_encoder(torch.zeros(B, T, self.cfg.n_pose, device=device))
    poses = poses.to(device)
    if poses.shape[-1] != self.cfg.n_pose and self.cfg.n_pose % poses.shape[-1] == 0:
      poses = poses.repeat(1, 1, self.cfg.n_pose // poses.shape[-1])
    return self.action_encoder(self.normalize_poses(poses))

  def _next(self, z, acts):
    """z (B, n, D), acts (B, n, D) -> ẑ del frame siguiente (B, D)."""
    return self.predict(z, acts)[:, -1]

  def predict_frames(self, frames, poses=None):
    """(z (B, F, D), ẑ (B, F-1, D)): ẑ[:, i] es la predicción del frame i + 1 desde los (hasta `history`) anteriores."""
    z = self.encode(frames)
    B, Fr = z.shape[:2]
    a = self.action_embed(poses, B, Fr, z.device)
    h = self.cfg.history
    if Fr - 1 <= h:
      return z, self.predict(z[:, :-1], a[:, :-1])
    zh = torch.cat([self.predict(z[:, max(0, f - h):f], a[:, max(0, f - h):f])[:, -1:] for f in range(1, Fr)], 1)
    return z, zh

  # ---- entrenamiento
  def train_loss(self, frames, teacher=None, alpha=0.5, tau=1.0, poses=None):
    """frames (B, F, 128) tokens (o píxeles) y poses (B, F, n_pose) crudas -> (loss, métricas), como lejepa_forward.

    Con más de history + num_preds frames (la evaluación de train.py usa ventanas de 20) se usan los primeros."""
    if teacher is not None:
      raise ValueError("LeWM no se destila: teacher tiene que ser None")
    c = self.cfg
    n = c.history + c.num_preds
    frames = frames[:, :n]
    B = frames.shape[0]
    emb = self.encode(frames)                                           # (B, n, D)
    act = self.action_embed(None if poses is None else poses[:, :n], B, n, emb.device)
    pred = self.predict(emb[:, :c.history], act[:, :c.history])
    tgt = emb[:, c.num_preds:]                                          # sin stop-grad
    pred_loss = (pred.float() - tgt.float()).pow(2).mean()
    sig = self.sigreg(emb.transpose(0, 1))                              # (T, B, D): por paso de tiempo, a través del batch
    loss = pred_loss + c.lam * sig
    z_std = emb.detach().float().reshape(-1, emb.shape[-1]).std(0).mean()
    return loss, {"pred_mse": pred_loss.detach(), "sigreg": sig.detach(), "z_std": z_std}

  # ---- evaluación de las representaciones
  @torch.no_grad()
  def features(self, frames, poses=None):
    """(z (B, F, D), ẑ (B, F-1, D) alineada con z[:, 1:]) en float32: para los probes de pose."""
    z, zh = self.predict_frames(frames, poses)
    return z.float(), zh.float()

  @torch.no_grad()
  def latent_error(self, frames, poses=None):
    """(B, F-1): error cuadrático medio entre ẑ_f y z_f del frame f = 1..F-1 (la "sorpresa")."""
    z, zh = self.predict_frames(frames, poses)
    return (zh.float() - z[:, 1:].float()).square().mean(-1)

  def generate(self, *a, **k):
    raise NotImplementedError("LeWM no emite tokens: predice embeddings (rollout_latent, plan_cem, LatentStream) y se "
                              "ve con el decoder de src/models/lewm_decoder.py")

  next_frame = last_frame_logits = logits = generate

  @torch.no_grad()
  def rollout_latent(self, z, acts, future_acts):
    """Rollout de embeddings. z (B, h, D) contexto; acts (B, h-1, D) acciones ya tomadas entre sus frames;
    future_acts (B, H, D): la acción tras el último frame del contexto y las siguientes. Devuelve ẑ (B, H, D)."""
    h = z.shape[1]
    a = torch.cat([acts, future_acts], 1)
    zf, out = z, []
    for t in range(future_acts.shape[1]):
      zh = self._next(zf[:, t:t + h], a[:, t:t + h])
      out.append(zh)
      zf = torch.cat([zf, zh[:, None].to(zf.dtype)], 1)
    return torch.stack(out, 1)

  @torch.no_grad()
  def plan_cem(self, ctx, goal, horizon: int = 5, ctx_poses=None, generator=None, n_samples: int = 300,
               n_elites: int = 30, iters: int = 10, init_std: float = 1.0):
    """Planifica con CEM las `horizon` acciones (poses apiladas) que llevan el último frame de `ctx` al de `goal`.

    ctx (B, F, 128) tokens (o píxeles; se usan los últimos `history` frames), goal (B, 128) (o (B, H, W, 3)),
    ctx_poses (B, F, n_pose) crudas (se usan las acciones ya tomadas entre los frames del contexto) o None (acción 0).
    CEM sobre las acciones normalizadas, de N(0, 1), como en la config de evaluación. Costo: ‖ẑ_horizonte − z_meta‖².
    Devuelve las poses planeadas en unidades crudas, (B, horizon, n_pose)."""
    h = self.cfg.history
    ctx = ctx[:, -h:]
    B = ctx.shape[0]
    z0 = self.encode(ctx)
    zg = self.encode(goal[:, None])[:, 0].float()
    a_ctx = self.action_embed(None if ctx_poses is None else ctx_poses[:, -h:-1], B, h - 1, z0.device)  # (B, h-1, D)
    A = self.cfg.n_pose

    def cost(cand):  # (N, B, horizon, A) normalizadas
      N = cand.shape[0]
      z = z0.unsqueeze(0).expand(N, *z0.shape).reshape(N * B, *z0.shape[1:])
      ac = a_ctx.unsqueeze(0).expand(N, *a_ctx.shape).reshape(N * B, h - 1, -1)
      af = self.action_encoder(cand.reshape(N * B, horizon, A))
      zh = self.rollout_latent(z, ac, af)[:, -1].float()
      return (zh - zg.unsqueeze(0).expand(N, *zg.shape).reshape(N * B, -1)).square().sum(-1).view(N, B)

    plan = cem(cost, B, (horizon, A), n_samples, n_elites, iters, init_std, device=ctx.device, generator=generator)
    return plan * self.pose_std.float() + self.pose_mean.float()

  # ---- lectura para src/eval_latent.py. Entrada: una ventana de píxeles uint8 (B, F, H, W, 3) ya a 4 Hz (un frame por
  # paso del modelo) y sus poses crudas (B, F, n_pose) (las `frameskip` poses de cada paso apiladas). Todo en el device.
  lat_aplica = ("probes", "horizonte", "sorpresa", "plan")
  lat_frames_por_paso = 1         # frames de la ventana que resume cada paso de las features (para la referencia de píxeles)
  lat_convencion = "promedio_bloque"  # la pose de un paso del plan es el promedio de sus `frameskip` poses
  lat_cem = dict(n_samples=300, n_elites=30, iters=10)  # config/eval/solver/cem.yaml (non-PushT)
  lat_pooling = "CLS proyectado de cada frame (sin pooling: el embedding ya es uno por frame)"

  @torch.no_grad()
  def lat_features(self, img, n=3):
    """{"obs": z de los últimos n frames (B, n·D), "pred": ẑ de esos frames predicho desde su historia, sin acciones}.

    Sin acciones: la pose es lo que el probe quiere leer, no lo que se le da al predictor."""
    z, zh = self.features(img[:, -(n + self.cfg.history):])
    return {"obs": z[:, -n:].flatten(1), "pred": zh[:, -n:].flatten(1)}

  @torch.no_grad()
  def lat_horizonte(self, img, poses, ctx=3, H=8, accion_nula=False):
    """Rollout latente de H pasos desde `ctx` frames reales con las acciones reales (o la acción media si accion_nula).

    Devuelve errores (B, H) contra el latente real: mse y l1 del modelo, y los de "copiar el último latente" (z del
    último frame de contexto repetido H veces)."""
    z = self.encode(img).float()
    B, Fr = z.shape[:2]
    a = self.action_embed(None if accion_nula else poses, B, Fr, z.device)
    zh = self.rollout_latent(z[:, :ctx], a[:, :ctx - 1], a[:, ctx - 1:ctx - 1 + H]).float()
    tgt = z[:, ctx:ctx + H]
    cop = z[:, ctx - 1:ctx].expand(-1, tgt.shape[1], -1)
    d, dc = zh - tgt, cop - tgt
    return {"mse": d.square().mean(-1), "l1": d.abs().mean(-1), "copia_mse": dc.square().mean(-1), "copia_l1": dc.abs().mean(-1)}

  @torch.no_grad()
  def lat_sorpresa(self, img, poses):
    """Error latente del frame f = 1..F-1 predicho desde los anteriores (B, F-1) y el de copiar el latente del anterior."""
    z, zh = self.predict_frames(img, poses)
    z = z.float()
    return {"mse": (zh.float() - z[:, 1:]).square().mean(-1), "copia_mse": (z[:, :-1] - z[:, 1:]).square().mean(-1)}

  @torch.no_grad()
  def lat_desvio(self, poses):
    """Desvío por dimensión de la acción normalizada (n_pose,), de poses (N, F, n_pose) de train: el desvío de muestreo del CEM."""
    x = ((poses.float() - self.pose_mean.float()) / self.pose_std.float()).reshape(-1, poses.shape[-1])
    ok = torch.isfinite(x)
    n = ok.sum(0).clamp(min=1)
    x0 = torch.where(ok, x, torch.zeros_like(x))
    mu = x0.sum(0) / n
    return ((torch.where(ok, x - mu, torch.zeros_like(x)).square().sum(0) / n).sqrt()).clamp(min=1e-3)

  @torch.no_grad()
  def lat_plan(self, img, poses, ctx=3, P=5, init_std=1.0, generator=None):
    """Planifica P pasos hacia el frame `ctx - 1 + P` de la ventana desde sus primeros `ctx`. Devuelve la pose
    (velocidad, giro) de cada paso, en unidades crudas (B, P, 2): el promedio de las `frameskip` poses apiladas del paso."""
    pl = self.plan_cem(img[:, :ctx], img[:, ctx - 1 + P], P, ctx_poses=poses[:, :ctx], generator=generator,
                       init_std=init_std, **self.lat_cem)
    k = self.cfg.frameskip
    return pl.view(len(pl), P, k, -1).mean(2)[..., [0, 5]]


# ---------------------------------------------------------------- export a ONNX (K5)

class _EncoderOnnx(nn.Module):
  """Un frame uint8 (B, H, W, 3) -> embedding (B, D): normalización de ImageNet + ViT + projector (BatchNorm en eval)."""

  def __init__(self, m):
    super().__init__()
    self.m = m

  def forward(self, img):
    return self.m.encode(img[:, None])[:, 0]


class _PredictorOnnx(nn.Module):
  """Un paso del predictor: z (B, h, D) y poses (B, h, n_pose) crudas SIN NaN -> ẑ del frame siguiente (B, D).

  La normalización de poses (sin el nan_to_num de normalize_poses: el llamador reemplaza los NaN) y el embedder de la
  acción van dentro; es lo mismo que `LeWM._next(z, action_embed(poses))`."""

  def __init__(self, m):
    super().__init__()
    self.m = m

  def forward(self, z, poses):
    p = (poses.float() - self.m.pose_mean.float()) / self.m.pose_std.float()
    return self.m._next(z, self.m.action_encoder(p))


def exportar_onnx(model, parte, ruta, B=1, opset=17):
  """Exporta `encoder` o `predictor` (un paso) de un LeWM a ONNX (atención con ops explícitas, sin SDPA).

  Devuelve {"entradas": {nombre: ndarray}, "salida": ndarray de PyTorch} con una entrada de ejemplo, para comparar con ORT."""
  from pathlib import Path

  from src.export_onnx import explicit_attention
  c = model.cfg
  model = model.eval().float().cpu()
  g = torch.Generator().manual_seed(0)
  if parte == "encoder":
    mod = _EncoderOnnx(model).eval()
    ej = {"img": torch.randint(0, 256, (B, c.img_h, c.img_w, 3), dtype=torch.uint8, generator=g)}
  elif parte == "predictor":
    mod = _PredictorOnnx(model).eval()
    ej = {"z": torch.randn(B, c.history, c.dim, generator=g), "poses": torch.randn(B, c.history, c.n_pose, generator=g)}
  else:
    raise ValueError(f"parte {parte!r}: encoder o predictor")
  ruta = Path(ruta)
  ruta.parent.mkdir(parents=True, exist_ok=True)
  with explicit_attention(), torch.no_grad():
    ref = mod(*ej.values())
    torch.onnx.export(mod, tuple(ej.values()), str(ruta), input_names=list(ej), output_names=["salida"],
                      opset_version=opset, dynamo=False)
  return {"entradas": {k: v.numpy() for k, v in ej.items()}, "salida": ref.numpy()}


ACCIONES = ("receta", "entrenar", "continuar", "evaluar", "bench", "demo", "exportar")
EVALUACION = "latente"
