"""ConvLSTM desde píxeles (Shi et al. 2015, arXiv 1506.04214): encoder-forecaster de ConvLSTM con peepholes.

Código propio y autocontenido. Es el ConvLSTM "de verdad" de la comparación: ve píxeles RGB y predice píxeles (no
tokens), como en el paper (Moving-MNIST y radar). Los frames se generan al vuelo con el decoder de comma congelado
(src.pixels.render, en el device) a partir de los tokens del dataset; train_loss y generate aceptan también imágenes
uint8 (B, F, 128, 256, 3) directamente (la cámara).

Arquitectura (§3.1, Eq. 3, Fig. 3; variante de mejor resultado en Tab. 1: 3 capas 128/64/64, kernels 5x5):
- Celda con peepholes: i = σ(W_xi*X + W_hi*H + W_ci∘C_{t-1} + b_i), f = σ(W_xf*X + W_hf*H + W_cf∘C_{t-1} + b_f),
  C = f∘C_{t-1} + i∘tanh(W_xc*X + W_hc*H + b_c), o = σ(W_xo*X + W_ho*H + W_co∘C + b_o), H = o∘tanh(C).
- Encoder: 3 ConvLSTM apiladas sobre los 10 frames de entrada (estado inicial 0). Forecaster: 3 ConvLSTM con la misma
  forma, estados iniciales = los últimos (H, C) del encoder; la capa 1 no recibe entrada, las siguientes reciben el H
  de la capa de abajo. Salida de cada paso: los H de todas las capas concatenados -> conv 1x1 -> parches -> frame.
- Loss: sigmoide + entropía cruzada binaria por píxel con valores en [0, 1].
Salidas determinísticas: el modelo no tiene temperatura ni verosimilitud (has_likelihood = False); output = "pixels".

FIDELIDAD (abajo) lista cada decisión con su fuente y categoría: fiel, datos, presupuesto o propio.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import schedules as S
from src.models.base import WorldModel
from src.pixels import render
from src.recipes import Recipe

SHI = "Shi et al. 2015 (arXiv 1506.04214)"

FIDELIDAD = [
  dict(campo="Celda", fuente_valor="ConvLSTM con peepholes: W_ci∘C_{t-1}, W_cf∘C_{t-1}, W_co∘C_t (Hadamard); transiciones "
       "entrada-estado y estado-estado convolucionales", aquí="igual; W_x*X y W_h*H se calculan con una sola conv sobre "
       "concat([X, H]) (matemáticamente idéntico a dos convs, con un solo sesgo)", categoría="fiel", fuente=f"{SHI} §3.1, Eq. 3"),
  dict(campo="Peepholes", fuente_valor="el paper no dice su forma", aquí="un vector por canal (broadcast en el espacio). "
       "Deducido: con parches 16x16x16, capas 128/64/64 y kernels 5x5 esta forma reproduce exactamente los 7 585 296 "
       "parámetros de la Tab. 1 (test_param_count_paper)", categoría="fiel", fuente=f"{SHI} Tab. 1; deducción propia del conteo"),
  dict(campo="Estructura", fuente_valor="encoder y forecaster apilados; el forecaster arranca del último estado del "
       "encoder; salidas de todas las capas concatenadas -> conv 1x1", aquí="igual; el forecaster conserva el orden de "
       "capas del encoder (el conteo de parámetros de la Tab. 1 solo cuadra así)", categoría="fiel",
       fuente=f"{SHI} §3.1, Fig. 3"),
  dict(campo="Entrada del forecaster", fuente_valor="Fig. 3: la red de pronóstico no recibe frames", aquí="la capa 1 "
       "no tiene entrada (sin pesos de entrada, como exige el conteo de la Tab. 1); no se realimenta la predicción "
       "dentro de los 10 pasos", categoría="fiel", fuente=f"{SHI} Fig. 3, Tab. 1"),
  dict(campo="Capas y canales", fuente_valor="3 capas 128/64/64, kernels 5x5 (mejor de la Tab. 1: 3670,85 vs 3733,56 "
       "con 2x128 y 3887,94 con 1x256)", aquí="128/64/64, kernel 5x5, padding 'same' (el paper no dice el borde)",
       categoría="fiel", fuente=f"{SHI} §4.1, Tab. 1 (padding: propio)"),
  dict(campo="Parches de entrada", fuente_valor="parches 4x4: 64x64x1 -> 16x16x16", aquí="parches 4x4: 128x256x3 -> "
       "32x64x48 (pixel_unshuffle; la salida usa pixel_shuffle)", categoría="fiel", fuente=f"{SHI} §4.1"),
  dict(campo="Imágenes", fuente_valor="Moving-MNIST 64x64 en escala de grises, dígitos que rebotan", aquí="video de "
       "conducción RGB 128x256 (reconstrucción del VQ-VAE de comma, renderizada con su decoder congelado)",
       categoría="datos", fuente="commaVQ; los tokens son lo único que hay en el dataset"),
  dict(campo="Frames de entrada y salida", fuente_valor="10 -> 10", aquí="10 -> 10 a 20 FPS (ventana de 20 frames)",
       categoría="fiel", fuente=f"{SHI} §4.1 (frame rate: datos, comma graba a 20 FPS)"),
  dict(campo="Horizonte de 3 s (60 frames)", fuente_valor="no existe: el forecaster pronostica 10 frames",
       aquí="se pronostican 10, y esos 10 pasan a ser la ventana de entrada del siguiente bloque de 10 (ventana "
       "deslizante, estado del encoder desde cero cada vez, como se entrenó). La predicción se realimenta como "
       "probabilidades en [0, 1] sin cuantizar", categoría="propio", fuente="el paper no hace rollouts largos"),
  dict(campo="Estado en línea (stream)", fuente_valor="el paper no habla de inferencia en línea", aquí="el stream "
       "guarda la ventana de los últimos 10 frames y re-codifica los 10 por cada predicción (igual que en "
       "entrenamiento); no arrastra el estado del encoder frame a frame", categoría="propio", fuente="fidelidad a lo entrenado"),
  dict(campo="Loss", fuente_valor="entropía cruzada binaria por píxel (suma sobre la secuencia), valores en [0, 1]",
       aquí="sigmoide + BCE con logits, promedio sobre píxeles, canales y frames (la suma solo cambia la escala, "
       "que RMSProp normaliza; el clip sí la ve)", categoría="fiel", fuente=f"{SHI} §4.1 (media en vez de suma: propio)"),
  dict(campo="Objetivos", fuente_valor="píxeles binarios/intensidades en [0, 1]", aquí="RGB uint8 / 255 (objetivos "
       "continuos en [0, 1])", categoría="datos", fuente=f"{SHI} §4.1"),
  dict(campo="Inicialización", fuente_valor="no publicada", aquí="Conv2d por defecto de PyTorch (Kaiming uniforme), "
       "sesgos por defecto, peepholes en 0", categoría="propio", fuente="el paper no la publica"),
  dict(campo="Precisión", fuente_valor="no publicada", aquí="fp32 (sin AMP)", categoría="propio", fuente="el paper no la publica"),
  dict(campo="Optimizador", fuente_valor="RMSProp, lr 1e-3, decay 0,9", aquí="RMSprop(lr=1e-3, alpha=0.9), lr "
       "constante, sin momentum ni weight decay", categoría="fiel", fuente=f"{SHI} §4.1 (eps 1e-8 de PyTorch: propio)"),
  dict(campo="Batch, clip, épocas, paciencia", fuente_valor="no publicados", aquí="batch 4 x 20 frames, clip de "
       "norma 1.0, pasos por presupuesto", categoría="propio", fuente="elegidos: el batch por memoria (12 GB) y los "
       "pasos que da; el clip porque BPTT sobre 20 pasos con RMSProp puede explotar"),
  dict(campo="Early stopping", fuente_valor="parar con el monitor de validación", aquí="se entrena el presupuesto "
       "completo (780k frames) y se evalúa el monitor (BCE en las ventanas del holdout) cada eval_every; train.py "
       "guarda el último estado, no el mejor", categoría="presupuesto", fuente=f"{SHI} §4.1; src/train.py"),
  dict(campo="Presupuesto", fuente_valor="no publicado (paper: 10 000 secuencias, hasta converger)", aquí="780k "
       "frames vistos (frames renderizados)", categoría="presupuesto", fuente="presupuesto común de la comparación"),
]


class ConvLSTMCell(nn.Module):
  """Celda ConvLSTM con peepholes (Eq. 3). in_ch = 0: sin entrada (capa 1 del forecaster)."""

  def __init__(self, in_ch, hid, k):
    super().__init__()
    self.in_ch, self.hid = in_ch, hid
    self.conv = nn.Conv2d(in_ch + hid, 4 * hid, k, padding=k // 2)
    self.w_ci, self.w_cf, self.w_co = (nn.Parameter(torch.zeros(hid)) for _ in range(3))

  def zero_state(self, B, h, w, like):
    z = like.new_zeros(B, self.hid, h, w)
    return z, z.clone()

  def forward(self, x, state):
    """x (B, in_ch, h, w) o None, state (H, C) -> (H', C'). Pura: el estado entra y sale explícito."""
    h, c = state
    z = self.conv(h if x is None else torch.cat([x, h], 1))
    gi, gf, gg, go = z.chunk(4, 1)
    p = lambda w: w.view(1, -1, 1, 1).to(c.dtype)
    i = torch.sigmoid(gi + p(self.w_ci) * c)
    f = torch.sigmoid(gf + p(self.w_cf) * c)
    c = f * c + i * torch.tanh(gg)
    o = torch.sigmoid(go + p(self.w_co) * c)
    return o * torch.tanh(c), c


class ConvLSTMPx(WorldModel):
  name = "convlstm_px"
  has_likelihood = False
  output = "pixels"

  @dataclass
  class Config:
    in_ch: int = 3                  # canales de la imagen
    height: int = 128
    width: int = 256
    patch: int = 4                  # parches de entrada (Shi §4.1)
    hidden: tuple = (128, 64, 64)   # mejor variante de Moving-MNIST (Tab. 1)
    kernel: int = 5
    n_in: int = 10
    n_out: int = 10

  @classmethod
  def recipe(cls, size=None):
    return Recipe(
      name="convlstm_px_shi2015", fuente=f"{SHI} §4.1", optimizer="rmsprop", alpha=0.9, momentum=0.0, eps=1e-8,
      lr_segs=[S.constant(1e-3)], wd_ini=0.0, wd_on="todos", ema_segs=None, grad_clip=1.0, amp=None,
      batch=4, frames=20, frameskip=1, accum=1, presupuesto=780_000,
      forzado=[("batch 4, clip 1.0, eps 1e-8", "el paper no los publica; batch por la memoria de 12 GB", "propio"),
               ("early stopping", "se entrena el presupuesto completo; train.py guarda el último estado", "presupuesto"),
               ("RGB 128x256, 20 FPS, parches 4x4 -> 32x64x48", "el dato de comma obliga", "datos"),
               ("fp32", "el paper no publica la precisión", "propio")])

  def __init__(self, cfg: "ConvLSTMPx.Config"):
    super().__init__()
    cfg.hidden = tuple(cfg.hidden)
    self.cfg = cfg
    self.ctx_frames = cfg.n_in
    assert cfg.height % cfg.patch == 0 and cfg.width % cfg.patch == 0
    self.pc = cfg.in_ch * cfg.patch ** 2  # canales por parche
    k, hs = cfg.kernel, cfg.hidden
    self.enc = nn.ModuleList(ConvLSTMCell(self.pc if i == 0 else hs[i - 1], hs[i], k) for i in range(len(hs)))
    self.dec = nn.ModuleList(ConvLSTMCell(0 if i == 0 else hs[i - 1], hs[i], k) for i in range(len(hs)))
    self.head = nn.Conv2d(sum(hs), self.pc, 1)

  # ---- núcleo (puro, estado explícito)
  def encode(self, x):
    """x (B, T, pc, h, w) parches -> estados [(H, C)] por capa tras T pasos desde cero."""
    B, T, _, h, w = x.shape
    st = [c.zero_state(B, h, w, x) for c in self.enc]
    for t in range(T):
      inp = x[:, t]
      for l, cell in enumerate(self.enc):
        st[l] = cell(inp, st[l])
        inp = st[l][0]
    return st

  def forecast(self, st, n):
    """Estados del encoder -> logits (B, n, pc, h, w) de los n frames siguientes (en parches)."""
    st, outs = list(st), []
    for _ in range(n):
      inp = None
      for l, cell in enumerate(self.dec):
        st[l] = cell(inp, st[l])
        inp = st[l][0]
      outs.append(self.head(torch.cat([s[0] for s in st], 1)))
    return torch.stack(outs, 1)

  def logits(self, win, n=None):
    """win (B, T, C, H, W) float en [0, 1] -> logits (B, n, C, H, W) de los n frames siguientes."""
    n = n or self.cfg.n_out
    B, T = win.shape[:2]
    p = self.cfg.patch
    x = F.pixel_unshuffle(win.flatten(0, 1), p).unflatten(0, (B, T))
    lg = self.forecast(self.encode(x), n)
    return F.pixel_shuffle(lg.flatten(0, 1), p).unflatten(0, (B, n))

  def probs(self, win, n=None):
    """Como logits, con sigmoide: frames pronosticados en [0, 1] (B, n, C, H, W)."""
    return torch.sigmoid(self.logits(win, n))

  # ---- conversiones
  def to_float(self, px):
    """uint8 (..., H, W, 3) -> float (..., 3, H, W) en [0, 1] con el dtype del modelo."""
    return px.movedim(-1, -3).to(self.head.weight.dtype) / 255

  @staticmethod
  def to_uint8(p):
    """float (..., 3, H, W) en [0, 1] -> uint8 (..., H, W, 3)."""
    return (p.float() * 255).round().clamp(0, 255).to(torch.uint8).movedim(-3, -1)

  def pixels(self, x):
    """Tokens (B, F, 128) -> píxeles (render con el decoder de comma, en el device); uint8 (B, F, H, W, 3) pasa igual."""
    return x if x.dim() == 5 else render(x)

  # ---- entrenamiento
  def train_loss(self, frames, teacher=None, alpha=0.5, tau=1.0, poses=None):
    if teacher is not None:
      raise ValueError("ConvLSTM no se destila: teacher tiene que ser None")
    c = self.cfg
    fl = self.to_float(self.pixels(frames))
    if fl.shape[1] <= c.n_in:
      raise ValueError(f"hacen falta más de {c.n_in} frames (entrada + objetivo), llegaron {fl.shape[1]}")
    tgt = fl[:, c.n_in:]
    lg = self.logits(fl[:, :c.n_in], tgt.shape[1]).float()
    loss = F.binary_cross_entropy_with_logits(lg, tgt.float())
    mse = (torch.sigmoid(lg) - tgt.float()).square().mean()
    return loss, {"bce": loss.detach(), "mse": mse.detach()}

  def last_frame_logits(self, frames):
    raise NotImplementedError("ConvLSTM predice píxeles: no hay logits de tokens (has_likelihood = False)")

  # ---- generación
  @torch.no_grad()
  def generate(self, ctx, n_frames, temperature=None, steps=None, generator=None):
    """Rollout determinístico en bloques de n_out frames: el bloque pronosticado pasa a ser la ventana de entrada
    (propio). ctx: tokens (B, F, 128) o píxeles uint8 (B, F, H, W, 3); usa los últimos n_in. Itera uint8 (B, H, W, 3)."""
    c = self.cfg
    win = self.to_float(self.pixels(ctx)[:, -c.n_in:])
    done = 0
    while done < n_frames:
      pr = self.probs(win, c.n_out)
      for j in range(min(c.n_out, n_frames - done)):
        yield self.to_uint8(pr[:, j])
      done += c.n_out
      win = torch.cat([win, pr], 1)[:, -c.n_in:]

  @torch.no_grad()
  def next_frame(self, ctx, temperature=None, steps=None, generator=None):
    """Frame siguiente en píxeles uint8 (B, H, W, 3)."""
    return next(self.generate(ctx, 1))

  def use_cuda_graphs(self, enable: bool = True):
    """Sin efecto: el graph lo captura ConvLSTMPxStream (src/infer/stream.py); existe por la interfaz de bench y rollouts."""

  def num_params(self):
    return sum(p.numel() for p in self.parameters())
