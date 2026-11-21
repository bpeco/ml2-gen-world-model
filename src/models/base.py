"""Interfaz común de los world models y utilidades compartidas.

Evaluación (CE), rollouts y bench solo usan esta interfaz: no saben qué modelo miden.

Convenciones:
- frames: LongTensor (B, F, 128) con tokens 0..1023, en el device del modelo.
- secuencia comma: LongTensor (B, F*129) con BOS antes de cada frame (src.common.with_bos).
  La posición p predice el token p+1 (el último token de un frame predice el BOS del siguiente).
- Precisión de inferencia: fp16/bf16 = model.to(dtype) (se castean los pesos). Las cachés se crean
  con el dtype de los parámetros. Las entradas son ids (long); los logits salen en el dtype del
  modelo y quien los consume hace .float().
- Entrenamiento: train_loss(frames, teacher=None, alpha, tau) -> (loss, métricas). teacher es un
  WorldModel con logits() (gpt2m en fp16); el alumno le pasa la misma entrada que usa él.
"""
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common import BOS, CTX_FRAMES, TPF, VOCAB, strip_bos, with_bos

NEG = -1e9  # logit "imposible" en float32


class WorldModel(nn.Module):
  """Base de todos los modelos.

  Atributos:
    name: nombre para results/*.json.
    ctx_frames: máximo de frames de contexto que usa next_frame. gpt2m: 19, porque el frame
      nuevo ocupa el lugar 20 del bloque de 20 x 129 tokens.
    has_likelihood: True si logits() define la verosimilitud del protocolo de comma
      (GPT, baselines). MaskGIT: False (solo last_frame_logits).
  """
  name = "base"
  ctx_frames = CTX_FRAMES - 1
  has_likelihood = True

  def logits(self, x: torch.Tensor) -> torch.Tensor:
    """(B, L) secuencia comma -> (B, L, VOCAB) logits del token siguiente en cada posición."""
    raise NotImplementedError

  def last_frame_logits(self, frames: torch.Tensor) -> torch.Tensor:
    """(B, F, 128) -> (B, 128, 1024): logits de frames[:, -1] dado frames[:, :-1].

    Default para modelos con verosimilitud: teacher forcing con logits(). Para un GPT, el token i
    del último frame ve los tokens < i reales del mismo frame. MaskGIT lo redefine (frame entero
    enmascarado, una pasada): es la "CE con 1 paso".
    """
    lg = self.logits(with_bos(frames))
    start = (frames.shape[1] - 1) * TPF  # posición del BOS del último frame
    return lg[:, start:start + 128, :BOS]

  def next_frame(self, ctx: torch.Tensor, temperature: float = 1.0, steps: Optional[int] = None,
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """(B, F, 128) contexto -> (B, 128) tokens del frame siguiente (0..1023)."""
    raise NotImplementedError

  def train_loss(self, frames: torch.Tensor, teacher: Optional["WorldModel"] = None, alpha: float = 0.5,
                 tau: float = 1.0):
    """frames (B, F, 128) de train -> (loss escalar, dict de métricas sin gradiente)."""
    raise NotImplementedError

  @torch.no_grad()
  def generate(self, ctx: torch.Tensor, n_frames: int, temperature: float = 1.0, steps: Optional[int] = None,
               generator: Optional[torch.Generator] = None) -> Iterator[torch.Tensor]:
    """Rollout frame a frame: el frame nuevo entra al contexto y sale el más viejo. Itera (B, 128).

    Los modelos con caché reutilizable pueden redefinirlo.
    """
    ctx = ctx[:, -self.ctx_frames:]
    for _ in range(n_frames):
      new = self.next_frame(ctx, temperature, steps, generator)
      yield new
      ctx = torch.cat([ctx, new[:, None]], 1)[:, -self.ctx_frames:]

  @torch.no_grad()
  def rollout(self, ctx: torch.Tensor, n_frames: int, **kw) -> torch.Tensor:
    """(B, F, 128) contexto -> (B, n_frames, 128) frames generados."""
    return torch.stack(list(self.generate(ctx, n_frames, **kw)), 1)

  @property
  def device(self) -> torch.device:
    for t in self.parameters():
      return t.device
    for t in self.buffers():
      return t.device
    return torch.device("cpu")


class FrameModel(WorldModel):
  """Modelo que predice las 128 celdas de un frame independientes, dados los frames anteriores
  (baselines). Alcanza con definir frame_logits()."""

  def frame_logits(self, frames: torch.Tensor) -> torch.Tensor:
    """(B, F, 128) -> (B, F, 128, 1024): predicción del frame f dados los frames < f.
    El frame 0 se predice sin contexto."""
    raise NotImplementedError

  def logits(self, x):
    assert x.shape[-1] % TPF == 0, "la secuencia tiene que tener frames enteros"
    return frame_logits_to_seq(self.frame_logits(strip_bos(x)))

  def last_frame_logits(self, frames):
    return self.frame_logits(frames)[:, -1]

  @torch.no_grad()
  def next_frame(self, ctx, temperature=1.0, steps=None, generator=None):
    ctx = ctx[:, -self.ctx_frames:]
    dummy = torch.zeros_like(ctx[:, :1])  # la predicción del frame F solo depende de los frames < F
    lg = self.frame_logits(torch.cat([ctx, dummy], 1))[:, -1]
    return sample_logits(lg, temperature, generator)


def frame_logits_to_seq(frame_logits: torch.Tensor) -> torch.Tensor:
  """(B, F, 128, 1024) logits por celda -> (B, F*129, VOCAB) en la convención comma.

  En el frame f, la posición j < 128 (BOS o token j-1) predice la celda j; la posición 128
  (último token) predice el BOS del frame siguiente, con certeza.
  """
  B, Fr, C, V = frame_logits.shape
  out = frame_logits.new_full((B, Fr, TPF, VOCAB), NEG, dtype=torch.float32)
  out[:, :, :C, :V] = frame_logits.float()
  out[:, :, C, BOS] = 0.0
  return out.view(B, Fr * TPF, VOCAB)


def sample_logits(logits: torch.Tensor, temperature: float = 1.0,
                  generator: Optional[torch.Generator] = None,
                  noise: Optional[torch.Tensor] = None) -> torch.Tensor:
  """logits (..., V) -> índices (...). temperature 0 = argmax.

  Muestrea con el truco de la exponencial (argmax de p / q con q ~ Exp(1)), que equivale a
  muestrear de p sin sincronizar con la GPU. Si llega `noise` (float32, la forma de los logits), es el q ya
  tirado: el mismo cálculo sin tocar el generador, así el muestreo cabe dentro de un CUDA graph.
  """
  if temperature == 0:
    return logits.argmax(-1)
  probs = torch.softmax(logits.float() / temperature, -1)
  q = noise if noise is not None else torch.empty_like(probs).exponential_(1, generator=generator)
  return (probs / q).argmax(-1)


def frame_noise(n: int, B: int, device, generator: Optional[torch.Generator] = None, V: int = 1024) -> torch.Tensor:
  """Las n tiradas Exp(1) de un frame autorregresivo (n, B, V), en una sola llamada al generador: noise[i] es el q
  del token i en sample_logits. Una llamada por token son n lanzamientos que la CPU hace antes del graph del frame, y
  la GPU los espera (GPTStream)."""
  return torch.empty(n, B, V, device=device).exponential_(1, generator=generator)


def lm_loss(logits: torch.Tensor, targets: torch.Tensor, teacher_logits: Optional[torch.Tensor] = None,
            alpha: float = 0.5, tau: float = 1.0, mask: Optional[torch.Tensor] = None):
  """CE con hard labels y, si hay maestro, destilación (Hinton 2015):

    loss = alpha * CE + (1 - alpha) * tau^2 * KL(maestro || alumno), con softmax a temperatura tau.

  logits y teacher_logits: (..., V); targets: (...). mask (...) bool opcional: posiciones que cuentan.
  Devuelve (loss, dict de métricas como tensores sin gradiente).
  """
  V = logits.shape[-1]
  lg = logits.reshape(-1, V).float()
  y = targets.reshape(-1)
  ce = F.cross_entropy(lg, y, reduction="none")
  m = mask.reshape(-1).float() if mask is not None else None
  mean = (lambda t: (t * m).sum() / m.sum().clamp(min=1)) if m is not None else (lambda t: t.mean())
  ce = mean(ce)
  if teacher_logits is None:
    return ce, {"ce": ce.detach()}
  tl = teacher_logits.reshape(-1, teacher_logits.shape[-1]).float()
  log_ps = F.log_softmax(lg / tau, -1)
  log_pt = F.log_softmax(tl / tau, -1)
  kl = mean((log_pt.exp() * (log_pt - log_ps)).sum(-1))
  loss = alpha * ce + (1 - alpha) * tau ** 2 * kl
  return loss, {"ce": ce.detach(), "kl": kl.detach()}
