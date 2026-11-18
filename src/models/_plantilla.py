"""PLANTILLA de un world model: un predictor de tokens mínimo pero real, para copiar (CONTRIBUTING, "Sumar un modelo").

Qué hace: dado el contexto (frames de 128 tokens VQ de commaVQ), predice los 128 tokens del frame siguiente. Cada celda mira
la misma celda de los últimos `lags` frames (una proyección por retardo) y un resumen global del frame anterior, y pasa por un
MLP. Es chico (~0,3M parámetros), entrena en CPU en segundos y no tiene atención: lo justo para mostrar todo lo que un
modelo tiene que declarar.

Para sumar tu modelo:
  1. Copiá este archivo a src/models/<tu_modelo>.py y su test a tests/test_<tu_modelo>.py (cambiá `Plantilla` por tu clase).
  2. Reemplazá `Plantilla.__init__` y `frame_logits` por tu arquitectura. Si tu modelo no es de verosimilitud por celda,
     heredá de WorldModel y definí `logits`, `next_frame` y `train_loss` (ver src/models/base.py y src/models/genie.py);
     si es latente, mirá src/models/latent.py.
  3. Completá `recipe()` con la receta de tu FUENTE (optimizador, lr, wd, batch, presupuesto) y anotá en `forzado` lo que la fuente
     no fija o que cambiaste, con su categoría: datos, presupuesto o propio.
  4. Completá FIDELIDAD: una fila por decisión, con la fuente (paper, sección, tabla). Un diseño propio se declara "propio".
  5. Declará ACCIONES (qué acciones de la CLI soporta) y EVALUACION ("video" o "latente").
  6. Declará CONTRATO (abajo): la configuración mínima con la que corren los tests de contrato.
  7. Registralo en src/models/__init__.py (KINDS) y corré `python -m src.cli verificar` y
     `python -m pytest tests/test_contrato.py tests/test_<tu_modelo>.py`. La CLI, la cola, la evaluación y el reporte lo
     toman del registro; no hay que tocar nada más.

Formato: frames (B, F, 128) con tokens 0..1023, en el device del modelo, sin BOS (src.common.with_bos lo agrega).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from src import schedules as S
from src.common import TPF
from src.models.base import FrameModel, lm_loss
from src.recipes import Recipe

CODES = TPF - 1  # 128 tokens por frame (8 x 16)
FUENTE = "propio: plantilla del repo (src/models/_plantilla.py); no reproduce ningún paper"

# CAMBIAR: una fila por decisión de diseño o de entrenamiento, con su fuente. Categorías: fiel (del paper), datos (lo obliga
# el dato de comma), presupuesto (lo obliga el cómputo) o propio (la fuente no lo fija y lo elegimos nosotros).
FIDELIDAD = [
  dict(campo="Arquitectura", fuente_valor="no hay fuente: modelo de ejemplo", aquí="embedding por token + una proyección por "
       "retardo + resumen global del frame anterior + MLP de 2 capas", categoría="propio", fuente=FUENTE),
  dict(campo="Entrada y salida", fuente_valor="no hay fuente", aquí="128 tokens VQ por frame, logits (128, 1024) del frame siguiente",
       categoría="datos", fuente="commaVQ: 128 tokens por frame, vocabulario de 1024"),
  dict(campo="Contexto", fuente_valor="no hay fuente", aquí="los últimos `lags` = 2 frames", categoría="propio", fuente=FUENTE),
  dict(campo="Loss", fuente_valor="no hay fuente", aquí="cross-entropy sobre los frames 1..F-1 (el frame 0 no tiene contexto)",
       categoría="propio", fuente=FUENTE),
  dict(campo="Optimizador y lr", fuente_valor="no hay fuente", aquí="AdamW (0,9, 0,95), wd 0,01, lr 3e-3 con warmup y coseno",
       categoría="propio", fuente=FUENTE),
  dict(campo="Batch y presupuesto", fuente_valor="no hay fuente", aquí="16 x 8 frames por paso; 780k frames como todos",
       categoría="presupuesto", fuente="presupuesto común del repo: 780k frames vistos (100M tokens)"),
]

# Capacidades que expone la CLI (src/cli.py, src/toolkit.py): solo las acciones que este modelo soporta.
ACCIONES = ("receta", "entrenar", "continuar", "evaluar", "bench", "demo")
EVALUACION = "video"  # "video": CE + rollouts + LPIPS/FVD; "latente": src.eval_latent

# Configuración mínima del contrato (tests/test_contrato.py): chica, para que corra en CPU en segundos.
# `config` son campos de Plantilla.Config; el resto de las claves del contrato son opcionales (ver tests/test_contrato.py).
CONTRATO = dict(config=dict(dim=32, lags=2))


class Plantilla(FrameModel):
  """Predictor de tokens mínimo. FrameModel da gratis logits() (convención comma), last_frame_logits() y next_frame() a partir de
  frame_logits(); generate() y rollout() vienen de WorldModel."""
  name = "plantilla"
  has_likelihood = True

  @dataclass
  class Config:
    dim: int = 64        # CAMBIAR: los hiperparámetros de tu arquitectura
    lags: int = 2        # frames de contexto que ve cada predicción
    mlp_ratio: int = 2

  @classmethod
  def recipe(cls, size=None):
    """CAMBIAR: la receta de tu fuente. `forzado` anota lo que la fuente no fija o que tuvo que cambiar (campo, motivo, categoría)."""
    return Recipe(
      name="plantilla_propia", fuente=FUENTE, optimizer="adamw", betas=(0.9, 0.95), eps=1e-8,
      lr_segs=[S.linear(0.0, 3e-3, frac=0.05, nombre="warmup"), S.cosine(3e-3, 0.1)],
      wd_ini=0.01, wd_on="matrices", grad_clip=1.0, amp=None, batch=16, accum=1, frames=8, frameskip=1,
      presupuesto=780_000, batch_fuente=16,   # batch_fuente: el batch efectivo de la fuente (acá no hay: el propio)
      forzado=[("receta entera", "no hay paper: valores razonables para un modelo chico", "propio"),
               ("batch 16 x 8 frames", "cabe en cualquier máquina; con 780k frames da ~6.000 pasos", "presupuesto")])

  def __init__(self, cfg: "Plantilla.Config"):
    super().__init__()
    self.cfg = cfg
    self.ctx_frames = cfg.lags  # next_frame recorta el contexto a esto
    d = cfg.dim
    self.tok = nn.Embedding(1024, d)
    self.pos = nn.Parameter(torch.zeros(CODES, d))
    self.lag = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(cfg.lags))  # una proyección por retardo
    self.glob = nn.Linear(d, d)
    self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.mlp_ratio * d), nn.GELU(), nn.Linear(cfg.mlp_ratio * d, d))
    self.out = nn.LayerNorm(d)
    self.head = nn.Linear(d, 1024)
    nn.init.normal_(self.pos, std=0.02)

  def frame_logits(self, frames):
    """(B, F, 128) -> (B, F, 128, 1024): predicción del frame f dados SOLO los frames < f (el 0, sin contexto).

    La causalidad es por construcción: el retardo l mira el frame f - l (corrido l lugares, con ceros al principio)."""
    e = self.tok(frames)                                      # (B, F, 128, D)
    h = self.pos.expand_as(e[:, :1]).expand(-1, e.shape[1], -1, -1)
    for l, proj in enumerate(self.lag, start=1):
      past = torch.cat([torch.zeros_like(e[:, :l]), e[:, :-l]], 1) if e.shape[1] > l else torch.zeros_like(e)
      h = h + proj(past)
    prev = torch.cat([torch.zeros_like(e[:, :1]), e[:, :-1]], 1)  # frame anterior
    h = h + self.glob(prev.mean(2, keepdim=True))                 # resumen global del frame anterior
    return self.head(self.out(h + self.mlp(h)))

  def train_loss(self, frames, teacher=None, alpha=0.5, tau=1.0):
    """frames (B, F, 128) -> (loss escalar, métricas sin gradiente). CAMBIAR si tu loss es otra (máscaras, latentes...)."""
    if teacher is not None:
      raise ValueError("la plantilla no se destila: teacher tiene que ser None")
    if frames.shape[1] < 2:
      raise ValueError("hacen falta al menos 2 frames (contexto + objetivo)")
    lg = self.frame_logits(frames)[:, 1:]
    return lm_loss(lg, frames[:, 1:])
