"""V-JEPA 2-AC: world model condicionado a acciones sobre un encoder V-JEPA 2 congelado (kind `vjepa2_ac`).

Fuente: V-JEPA 2 paper §3 y github.com/facebookresearch/vjepa2 app/vjepa_droid/train.py:
  - L408-415 forward_target: cada frame del clip se codifica por separado, duplicado en un tubelet de 2
    (`unsqueeze(2).repeat(1, 1, 2, 1, 1)`), con el encoder congelado y sin gradiente; F.layer_norm sobre el canal.
  - L417-437 forward_predictions: (1) teacher forcing: el predictor ve los frames 0..T-2 con sus acciones y estados y predice
    1..T-1; cada salida pasa por layer_norm antes de la loss y de realimentarse. (2) rollout: [frame 0 real, frame 1
    predicho por teacher forcing] -> el predictor, con acciones[:2] y estados[:2], predice el frame 2; con auto_steps=2 el
    bucle da UN paso recurrente. z_ar = [frame1_pred_tf, frame2_pred_ar] (con gradiente por ese paso).
  - L439-449 loss: L1 media (loss_exp=1) de z_tf contra h[:, 1:T] y de z_ar contra h[:, 1:3], sumadas 1:1.
  - El encoder no está en el optimizador (solo el predictor).
Estado y acción (datos): en DROID el estado es la pose del efector y la acción es el cambio de estado entre frames
(acciones (B, T-1, 7), estados (B, T, 7)). Aquí el estado de cada frame tomado (1 de cada 5) es su pose (6 velocidades,
las primeras 6 de las 6·5 que apila windows.sample) normalizada con las estadísticas de pose, y la acción a_t = s_{t+1} - s_t
entre frames consecutivos tomados (en unidades normalizadas; donde falta una pose, acción y estado en 0).

Cargar el encoder: `encoder_run` es el nombre de una corrida `vjepa2` (registro/manifiesto, ruta .pt o checkpoints/<run>/final.pt);
se copia su `target_encoder` (el EMA, el que evalúa el paper) y queda congelado y fuera del optimizador. El checkpoint de este
modelo ya lleva los pesos del encoder: al guardarse cfg.cargar_encoder es False y no hace falta la corrida madre para recargarlo.
"""
import dataclasses
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import pixels
from src import schedules as S
from src.models.base import WorldModel
from src.models.vjepa.ac import ACPredictor
from src.models.vjepa.patch import normalizar_cthw
from src.models.vjepa.plan import planificar
from src.models.vjepa.pretrain import PAPER, layer_norm_canal
from src.models.vjepa.vit import PRESETS, VJEPAEncoder
from src.recipes import Recipe

YAML_AC = "configs/train/vitg16/droid-256px-8f.yaml"
WARM_AC = 4500 / 94500

FIDELIDAD_AC = [
  dict(campo="Predictor", paper="24 capas, 16 cabezas, dim 1024, GELU (~300M), frame-causal, RoPE",
       aquí="12 capas, 12 cabezas, dim 384 (GELU, MLP 4x)", categoría="presupuesto",
       fuente=f"{PAPER} §3.1; ac_predictor.py; {YAML_AC} L51-60"),
  dict(campo="Entrada por frame", paper="[acción, estado, parches]; acción y estado con Linear separados; RoPE 3D en parches "
       "y temporal en acción y estado", aquí="igual", categoría="fiel", fuente="ac_predictor.py L53-56, L145-153; modules.py L184-206"),
  dict(campo="Atención", paper="block-causal por frame", aquí="igual", categoría="fiel", fuente="ac_predictor.py L109-117"),
  dict(campo="Snap de posiciones espaciales", paper="h *= grid/H, w *= grid/W con grid = alto de la rejilla; factor 1 en "
       "su rejilla cuadrada", aquí="se conserva el código: con 8x16 las posiciones w se escalan por 8/16",
       categoría="datos", fuente="modules.py L179-181"),
  dict(campo="Codificación de un frame", paper="frame duplicado en un tubelet de 2; encoder congelado", aquí="igual; el encoder es "
       "el target_encoder de la corrida vjepa2 elegida (small, no ViT-g)", categoría="fiel", fuente="train.py L408-412"),
  dict(campo="Encoder", paper="ViT-g (V-JEPA 2 de 1B), preentrenado con 22M videos", aquí="el de nuestra corrida vjepa2 "
       "(small/tiny), preentrenada con los clips de comma", categoría="presupuesto", fuente=f"{PAPER} §3.1"),
  dict(campo="Normalización", paper="layer_norm sobre el canal en targets y salidas del predictor", aquí="igual",
       categoría="fiel", fuente="train.py L413-414, L421-423"),
  dict(campo="Loss", paper="L1 media, teacher forcing (T=15) + rollout de 2 pasos, 1:1", aquí="igual (auto_steps=2)",
       categoría="fiel", fuente="train.py L439-449; paper Eq. 2-4"),
  dict(campo="Frames por muestra", paper="clips de 4 s a 4 fps (16 frames según el paper; el yaml dice 8)",
       aquí="16 frames = 1 de cada 5 de los 20 FPS, T = 15 pasos de teacher forcing", categoría="fiel",
       fuente=f"{PAPER} §3.1; {YAML_AC} L7-19"),
  dict(campo="Estado y acción", paper="pose del efector (7D) y su cambio entre frames", aquí="estado = pose del frame (6 "
       "velocidades normalizadas), acción = diferencia de estados entre frames tomados", categoría="datos",
       fuente="app/vjepa_droid/train.py L390-394; datos de comma"),
  dict(campo="Aumentos", paper="random-resized-crop de escala [0.9, 1] en DROID", aquí="ninguno (el recorte mueve la geometría "
       "que las velocidades de pose suponen fija)", categoría="propio", fuente=f"{YAML_AC}"),
  dict(campo="Optimizador / LR", paper="AdamW (0.9, 0.999), wd 0.04, sin clip; warmup 7.5e-5 → 4.25e-4 en 4500 de 94.5k pasos, "
       "constante, decay lineal a 0 en los últimos 4500", aquí="igual con las mismas fracciones (4500/94500)", categoría="fiel",
       fuente=f"{PAPER} §3.1; {YAML_AC} L61-70"),
  dict(campo="Batch", paper="256", aquí="256 efectivo = 16 x accum 16", categoría="presupuesto", fuente=f"{PAPER} §3.1"),
  dict(campo="Presupuesto", paper="94.5k pasos", aquí="4x los 780k frames de referencia ≈ 760 pasos de 256 clips de 16 frames",
       categoría="presupuesto", fuente=f"{PAPER} §3.1"),
  dict(campo="Planificador CEM", paper="800 muestras, 10 iteraciones, top-10, horizonte 1, N(0, 1), energía L1",
       aquí="igual (plan.py), en el espacio normalizado de acciones", categoría="fiel", fuente=f"{PAPER} §3.2; mpc_utils.py L17-140"),
  dict(campo="Decoder de visualización", paper="ViT-L feedforward por frame, MSE, 150k pasos, batch 1024", aquí="ViT propio "
       "chico, mismo objetivo (MSE), batch y pasos menores (train_decoder.py)", categoría="presupuesto", fuente=f"{PAPER} Fig. 15, App."),
]


@dataclass
class Config:
  encoder_run: str = ""        # corrida vjepa2 de la que sale el encoder ("" = encoder aleatorio, solo para pruebas)
  cargar_encoder: bool = True  # pasa a False tras cargar: el checkpoint ya lleva los pesos
  encoder: str = "small"
  embed_dim: int = 0
  depth: int = 0
  num_heads: int = 0
  patch: int = 16
  tubelet: int = 2
  pred_dim: int = 384
  pred_depth: int = 12
  pred_heads: int = 12
  frames: int = 16
  height: int = 128
  width: int = 256
  n_pose: int = 30             # 6·frameskip: train.py lo fija con frameskip 5; solo las 6 primeras (pose del frame) son el estado
  action_dim: int = 6
  auto_steps: int = 2
  oficial_rope: bool = True
  grad_ckpt: bool = True


class VJEPA2AC(WorldModel):
  name = "vjepa2_ac"
  has_likelihood = False
  uses_poses = True
  ctx_frames = 15

  Config = Config

  @classmethod
  def recipe(cls, size=None):
    return Recipe(
      name="vjepa2_ac_oficial", fuente=f"{PAPER} §3.1; {YAML_AC}", optimizer="adamw", betas=(0.9, 0.999), eps=1e-8,
      lr_segs=[S.linear(7.5e-5, 4.25e-4, frac=WARM_AC, nombre="warmup"), S.constant(4.25e-4, nombre="constante"),
               S.linear(4.25e-4, 0.0, frac=WARM_AC, nombre="decay")],
      wd_ini=0.04, wd_fin=None, wd_on="matrices", ema_segs=None, grad_clip=None, amp="bf16",
      batch=16, frames=16, frameskip=5, accum=16, presupuesto=4 * 780_000, paper_steps=94_500,
      forzado=[("batch", "256 efectivo con batch 16 x accum 16 (memoria)", "presupuesto"),
               ("presupuesto", "≈760 pasos en vez de 94.5k; las fracciones de warmup y decay se conservan", "presupuesto"),
               ("predictor", "12 x 384 en vez de 24 x 1024", "presupuesto"),
               ("frames/frameskip", "16 frames a 4 fps = 1 de cada 5 de los 20 FPS", "datos")])

  def __init__(self, cfg: Config):
    super().__init__()
    self.cfg = cfg
    enc_cfg = None
    parent = None
    if cfg.cargar_encoder and cfg.encoder_run:
      from src.models import load_model
      parent = load_model(cfg.encoder_run)
      pc = parent.cfg
      for k in ("encoder", "embed_dim", "depth", "num_heads", "patch", "tubelet", "oficial_rope"):
        setattr(cfg, k, getattr(pc, k))
    p = dict(PRESETS[cfg.encoder])
    for k in ("embed_dim", "depth", "num_heads"):
      if getattr(cfg, k):
        p[k] = getattr(cfg, k)
    self.encoder = VJEPAEncoder(**p, patch_size=cfg.patch, tubelet_size=cfg.tubelet, oficial=cfg.oficial_rope)
    if parent is not None:
      self.encoder.load_state_dict(parent.target_encoder.state_dict())
      cfg.cargar_encoder = False  # el checkpoint de este modelo ya trae los pesos
    for q in self.encoder.parameters():
      q.requires_grad_(False)
    self.encoder.eval()
    self.grid = (cfg.height // cfg.patch, cfg.width // cfg.patch)
    self.predictor = ACPredictor(p["embed_dim"], cfg.action_dim, self.grid, cfg.pred_dim, cfg.pred_depth, cfg.pred_heads,
                                 oficial=cfg.oficial_rope)
    self.predictor.grad_ckpt = cfg.grad_ckpt
    self.register_buffer("pose_mean", torch.zeros(cfg.n_pose))
    self.register_buffer("pose_std", torch.ones(cfg.n_pose))

  def train(self, mode=True):
    super().train(mode)
    self.encoder.eval()  # congelado: nunca en modo entrenamiento
    return self

  def num_params(self):
    return sum(q.numel() for q in self.predictor.parameters())

  # ---- poses
  def set_pose_stats(self, mean, std):
    self.pose_mean.copy_(torch.as_tensor(mean, dtype=self.pose_mean.dtype))
    self.pose_std.copy_(torch.as_tensor(std, dtype=self.pose_std.dtype))

  def estados_acciones(self, poses, B=None, T=None, device=None):
    """poses crudas (B, T, n_pose) o None -> (estados (B, T, A), acciones (B, T-1, A)) normalizados.

    Estado = las A primeras componentes (la pose del frame) normalizadas; NaN -> 0. Acción = diferencia de estados
    consecutivos, 0 donde falta alguno de los dos. None: estados y acciones nulos (el monitor no lleva poses)."""
    A = self.cfg.action_dim
    if poses is None:
      return torch.zeros(B, T, A, device=device), torch.zeros(B, T - 1, A, device=device)
    p = poses[..., :A].float()
    ok = torch.isfinite(p).all(-1)
    s = torch.nan_to_num((p - self.pose_mean[:A].float()) / self.pose_std[:A].float())
    s = torch.where(ok[..., None], s, torch.zeros_like(s))
    a = torch.where((ok[:, 1:] & ok[:, :-1])[..., None], s[:, 1:] - s[:, :-1], torch.zeros_like(s[:, 1:]))
    return s, a

  # ---- representaciones
  def clip(self, frames):
    """tokens (B, F, 128) o píxeles uint8 (B, F, H, W, 3) -> (B, 3, T, H, W) normalizado, sin aumentos."""
    c = self.cfg
    if frames.dim() == 3:
      with torch.no_grad():
        frames = pixels.render(frames[:, :c.frames])
    x = frames[:, :c.frames].float().div(255)
    return normalizar_cthw(x.permute(0, 4, 1, 2, 3))

  @torch.no_grad()
  def codificar(self, clip):
    """(B, 3, T, H, W) normalizado -> (B, T, N, D), cada frame por separado (tubelet duplicado) y con layer_norm."""
    B, T = clip.shape[0], clip.shape[2]
    h = layer_norm_canal(self.encoder.encode_frames(clip))
    return h.view(B, T, -1, h.size(-1))

  def predecir(self, z, acciones, estados):
    """(B, T, N, D), acciones (B, T, A), estados (B, T, A) -> (B, T, N, D) con layer_norm: el frame t+1 para cada t."""
    return layer_norm_canal(self.predictor(z, acciones, estados))

  def predicciones(self, z, a, s):
    """(teacher forcing (B, T-1, N, D), rollout (B, auto_steps, N, D)) de train.py L417-437.

    tf: frames 1..T-1 desde los reales 0..T-2. ar: [frame 1 de tf, frame 2 predicho a partir de (frame 0 real, frame 1
    predicho)] y, con auto_steps > 2, un frame más por paso recurrente, siempre con acciones[:n+1] y estados[:n+1]."""
    tf = self.predecir(z[:, :-1], a, s[:, :-1])
    zr = torch.cat([z[:, :1], tf[:, :1]], 1)
    for n in range(1, self.cfg.auto_steps):
      zr = torch.cat([zr, self.predecir(zr, a[:, :n + 1], s[:, :n + 1])[:, -1:]], 1)
    return tf, zr[:, 1:]

  # ---- entrenamiento (train.py L417-449)
  def train_loss(self, frames, teacher=None, alpha=0.5, tau=1.0, poses=None):
    if teacher is not None:
      raise ValueError("V-JEPA 2-AC no se destila: teacher tiene que ser None")
    x = self.clip(frames)
    z = self.codificar(x)  # (B, T, N, D)
    B, T = z.shape[:2]
    s, a = self.estados_acciones(poses, B, T, z.device)
    tf, ar = self.predicciones(z, a, s)
    jloss = (tf.float() - z[:, 1:1 + tf.shape[1]].float()).abs().mean()
    sloss = (ar.float() - z[:, 1:1 + ar.shape[1]].float()).abs().mean()
    loss = jloss + sloss
    return loss, {"l1_tf": jloss.detach(), "l1_ar": sloss.detach()}

  @torch.no_grad()
  def planear(self, z, acciones, estados, objetivo, horizonte=1, **kw):
    """CEM (800 muestras, top-10, 10 iteraciones por defecto): acciones normalizadas (B, horizonte, A). Ver plan.py."""
    return planificar(self, z, acciones, estados, objetivo, horizonte, **kw)

  def a_pose(self, accion_norm, estado_norm):
    """Estado siguiente en unidades normalizadas (estado + acción), para encadenar planes."""
    return estado_norm + accion_norm


def config_dict(model):
  return dataclasses.asdict(model.cfg)

# Tabla de fidelidad completa de este modelo (la que leen src.recipes --fidelidad y las guías).
FIDELIDAD = FIDELIDAD_AC
