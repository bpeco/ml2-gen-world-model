"""MaskGIT sobre tokens de comma: predice los 128 tokens de un frame en paralelo, con S pasadas.

Mismo formato de secuencia que el GPT (BOS + 128 códigos por frame), pero la salida en la posición
p predice el token EN p (no el siguiente). Atención por bloques: causal entre frames y bidireccional
dentro de cada frame (Genie). El BOS nunca se enmascara.

Es una ABLACIÓN PROPIA, no una réplica de MaskGIT: el entrenamiento (posición factorizada, p_clean, tasa de máscara por
frame) es del repo; solo la decodificación sigue a MaskGIT. FIDELIDAD (abajo) separa lo que sigue al paper de lo propio.
El modelo fiel a Genie está en genie.py.
"""
import math
from dataclasses import dataclass

import torch

from src.common import BOS, MASK, TPF, with_bos
from src.models.backbone import Backbone, BackboneConfig
from src.models.base import WorldModel, lm_loss
from src.models.cudagraph import GraphedStep
from src.recipes import legacy_cosine

CODES = TPF - 1
MG = "MaskGIT (arXiv 2202.04200; github.com/google-research/maskgit)"

FIDELIDAD = [
  dict(campo="Calendario de máscara en inferencia", fuente_valor="coseno: quedan floor(128·cos(π/2·s/S)) enmascarados",
       aquí="igual", categoría="fiel", fuente=f"{MG} §3.2; libml/mask_schedule.py"),
  dict(campo="Decodificación por confianza", fuente_valor="se muestrea cada celda, se fijan las más confiables con ruido "
       "Gumbel sobre log p que baja a 0", aquí="igual (choice_temp 1.0 por defecto, el de MaskGIT es 4.5)",
       categoría="fiel", fuente=f"{MG} libml/parallel_decode.py L29-53; choice_temp: propio"),
  dict(campo="Pasos de decodificación", fuente_valor="8 en el paper", aquí="8 por defecto (default_steps)",
       categoría="fiel", fuente=f"{MG} §3.2"),
  dict(campo="Loss", fuente_valor="cross-entropy solo en los tokens enmascarados", aquí="igual, sin label smoothing "
       "(MaskGIT usa 0.1)", categoría="fiel", fuente=f"{MG} Eq. 1; config L41-42"),
  dict(campo="Máscara en entrenamiento", fuente_valor="una tasa coseno por muestra de imagen (r ~ U[0,1), cos(π/2·r))",
       aquí="una tasa por frame (coseno o U[0.5,1]), independiente entre frames", categoría="propio",
       fuente="MaskGIT genera una imagen; el video por frames no está en el paper"),
  dict(campo="p_clean", fuente_valor="no existe", aquí="probabilidad de dejar un frame limpio al entrenar, para que el "
       "contexto se parezca al de inferencia", categoría="propio", fuente="decisión del repo"),
  dict(campo="Atención", fuente_valor="bidireccional sobre la imagen",
       aquí="bloques: causal entre frames, bidireccional dentro del frame", categoría="propio",
       fuente="extensión a video (la de Genie es el ST-transformer de genie.py)"),
  dict(campo="Posición", fuente_valor="aprendida", aquí="pos='abs' (por defecto) o 'frame' (frame + celda, el de las corridas "
       "propias)", categoría="propio", fuente=f"{MG} config L55-59; variante factorizada: repo"),
  dict(campo="Arquitectura y tamaño", fuente_valor="24 capas, 768, MLP 3072 (bidireccional, sobre 256 tokens de imagen)",
       aquí="S = 6 x 384, M = 8 x 512, L = 12 x 768 (backbone propio compartido con el GPT)", categoría="presupuesto",
       fuente=f"{MG} config L55-59"),
  dict(campo="Receta de entrenamiento", fuente_valor="Adam (0.9, 0.96), lr 1e-4 constante tras warmup 5k, wd 4.5e-2, "
       "batch 256, dropout 0.1 (código de entrenamiento no publicado)",
       aquí="receta 'maskgit_propia': AdamW (0.9, 0.95), warmup + coseno, wd 0.1, clip 1, fp16, sin dropout",
       categoría="propio", fuente="receta heredada del repo; MaskGIT no publica el código de entrenamiento"),
  dict(campo="Datos", fuente_valor="ImageNet 256x256, tokens VQGAN 16x16", aquí="tokens de comma, 128 por frame",
       categoría="datos", fuente="commaVQ"),
]

# Capacidades que expone la CLI (src/cli.py, src/toolkit.py): solo las acciones que este modelo soporta.
ACCIONES = ("receta", "entrenar", "continuar", "evaluar", "bench", "demo", "exportar")
EVALUACION = "video"  # "video": CE + rollouts + LPIPS/FVD; "latente": src.eval_latent


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
    # v2, experimentos de movimiento (apagados por defecto: los checkpoints del paper no cambian)
    peso_cambio: float = 1.0        # peso de la CE en los tokens que cambian respecto del frame anterior (1 = sin pesar)
    p_self: float = 0.0             # prob. de que un frame de entrada sea una predicción del propio modelo (scheduled sampling)
    motion_in: bool = False         # embedding de "igual / distinto al frame anterior" sumado a cada token visible
    motion_cb: bool = False         # proyección de cb[token] - cb[token anterior] (codebook de commaVQ, congelado)

  @classmethod
  def recipe(cls, size=None):
    """Receta propia de la ablación (la heredada del repo): MaskGIT no publica su código de entrenamiento."""
    return legacy_cosine(name="maskgit_propia", fuente="propia (ablación del repo; MaskGIT no publica el entrenamiento)",
                         forzado=[("toda la receta", "ablación propia: no hay receta publicada que seguir", "propio")]
                         ).replace(batch_fuente=16)  # la fuente es la receta propia: su batch es el de referencia (no se escala)

  def __init__(self, cfg: "MaskGIT.Config"):
    super().__init__()
    self.cfg = cfg
    self.net = Backbone(BackboneConfig(n_layer=cfg.n_layer, n_head=cfg.n_head, dim=cfg.dim,
                                       block_size=cfg.block_size, vocab_in=1026, vocab_out=1024,
                                       attn="block", dropout=cfg.dropout, pos=cfg.pos))
    if cfg.motion_in:  # 0 = sin dato (MASK, BOS o primer frame), 1 = igual al anterior, 2 = distinto; arranca en cero
      self.motion = torch.nn.Embedding(3, cfg.dim)
      torch.nn.init.zeros_(self.motion.weight)
    if cfg.motion_cb:  # la diferencia de vectores del codebook predice el desplazamiento (results/v2/latente_movimiento.json)
      from src.infer.codec import ENCODER, ensure_weights
      ensure_weights()
      cb = torch.load(ENCODER, map_location="cpu", weights_only=True)["quantize._embedding.weight"].float()
      self.register_buffer("cb", cb / cb.norm(dim=-1).mean())  # escala ~1 por vector; queda en el checkpoint
      self.motion_proj = torch.nn.Linear(cb.shape[1], cfg.dim)
      torch.nn.init.zeros_(self.motion_proj.weight)
      torch.nn.init.zeros_(self.motion_proj.bias)
    self._graphs_on = False
    self._graphs = {}
    self._cache_key = None

  def _drop_cache(self):
    if self._cache_key is not None:  # el camino sin caché no admite cachés activas
      self.net.clear_cache()
      self._cache_key, self._graphs = None, {}

  @staticmethod
  def prev_of(x):
    """(B, F*129) -> el token de la misma celda en el frame anterior (-1 en el primer frame)."""
    prev = torch.full_like(x, -1)
    prev[:, TPF:] = x[:, :-TPF]
    return prev

  def run(self, x, input_pos=None, prev=None):
    """Pasada de la red. Con motion_in suma el embedding de movimiento: prev (mismo shape que x) es el token de la
    misma celda en el frame anterior; por defecto se toma de la propia secuencia (prev_of)."""
    c = self.cfg
    if not (c.motion_in or c.motion_cb):
      return self.net(x, input_pos)
    prev = self.prev_of(x) if prev is None else prev
    valid = (x < BOS) & (prev >= 0) & (prev < BOS)
    emb = self.net.wte(x)
    if c.motion_in:
      emb = emb + self.motion(torch.where(valid, 1 + (x != prev).long(), torch.zeros_like(x)))
    if c.motion_cb:
      d = self.cb[x.clamp(max=BOS - 1)] - self.cb[prev.clamp(0, BOS - 1)]
      emb = emb + self.motion_proj((d * valid[..., None]).to(emb.dtype))
    return self.net(input_pos=input_pos, emb=emb)

  def logits(self, x):
    """(B, L) secuencia con MASK -> (B, L, 1024): la posición p predice el token en p."""
    self._drop_cache()
    return self.run(x)

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
    c = self.cfg
    src = frames
    if c.p_self > 0 and self.training:  # scheduled sampling (no en el monitor: su CE queda comparable)
      src = self._self_frames(frames)
    r = self._mask_rates(B, Fr, frames.device)
    if c.p_clean > 0:  # en inferencia el contexto está limpio: que en entrenamiento también aparezca
      r = torch.where(torch.rand(B, Fr, device=frames.device) < c.p_clean, torch.zeros_like(r), r)
    masked = torch.rand(B, Fr, CODES, device=frames.device) < r[..., None]  # el BOS no entra
    seq = with_bos(src).view(B, Fr, TPF)
    x = seq.clone()
    x[:, :, 1:] = torch.where(masked, torch.full_like(frames, MASK), src)
    lg = self.logits(x.view(B, -1)).view(B, Fr, TPF, -1)[:, :, 1:]
    if c.peso_cambio == 1.0:
      loss, m = lm_loss(lg, frames, mask=masked)
    else:  # más peso a acertar lo que cambia: copiar el frame anterior deja de ser la salida barata
      ce = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), frames.reshape(-1),
                                             reduction="none").view(B, Fr, CODES)
      cambia = torch.zeros_like(masked)
      cambia[:, 1:] = frames[:, 1:] != frames[:, :-1]
      w = masked.float() * torch.where(cambia, c.peso_cambio, 1.0)
      loss = (ce * w).sum() / w.sum().clamp(min=1)
      mf = masked.float()
      m = {"ce": ((ce * mf).sum() / mf.sum().clamp(min=1)).detach(),
           "ce_cambia": ((ce * mf * cambia).sum() / (mf * cambia).sum().clamp(min=1)).detach()}
    m["mask_rate"] = masked.float().mean().detach()
    return loss, m

  @torch.no_grad()
  def _self_frames(self, frames):
    """Copia de frames donde cada frame, con prob. p_self, se reemplaza por la reconstrucción del modelo en una pasada
    (máscara U[0.5, 1], muestreo a T = 1) desde la entrada real. Imita los errores que el contexto tiene en inferencia."""
    B, Fr, _ = frames.shape
    r = 0.5 + 0.5 * torch.rand(B, Fr, 1, device=frames.device)
    masked = torch.rand(B, Fr, CODES, device=frames.device) < r
    x = with_bos(frames).view(B, Fr, TPF).clone()
    x[:, :, 1:] = torch.where(masked, torch.full_like(frames, MASK), frames)
    lg = self.logits(x.view(B, -1)).view(B, Fr, TPF, -1)[:, :, 1:].float()
    samp = torch.distributions.Categorical(logits=lg).sample()
    hat = torch.where(masked, samp, frames)
    usar = torch.rand(B, Fr, 1, device=frames.device) < self.cfg.p_self
    return torch.where(usar, hat, frames)

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

  def _frame_pass(self, tok, pos, prev):
    return self.run(tok, pos, prev)

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
    self.run(seq, pos_all[:P])  # prefill del contexto en la caché; cada frame lo re-prefillea
    pos = pos_all[P:P + TPF]
    tok = torch.full((B, TPF), MASK, dtype=torch.long, device=dev)
    tok[:, 0] = BOS
    prev = seq[:, -TPF:]  # el último frame del contexto: con motion_in, el "anterior" de cada celda del frame nuevo
    use_graph = self._graphs_on and dev.type == "cuda"

    def logits_fn(cur):
      tok[:, 1:] = cur
      if use_graph:
        g = self._graphs.get(B)
        if g is None:
          g = self._graphs[B] = GraphedStep(self._frame_pass, tok, pos, prev)
        return g(tok, pos, prev)[:, 1:].float()
      return self._frame_pass(tok, pos, prev)[:, 1:].float()

    cur = torch.full((B, CODES), MASK, dtype=torch.long, device=dev)
    return decode_frame(logits_fn, S, temperature, self.cfg.choice_temp, cur, generator).clone()
