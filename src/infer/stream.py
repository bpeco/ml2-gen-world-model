"""Estado de inferencia por frame, residente en el device: el world model como un proceso en línea.

  s = make_stream(model, temperature=1.0, steps=4, seed=0)
  s.observe(tok)           # tokens (B, 128) de un frame real: entra al estado
  nxt = s.predict()        # muestra del frame siguiente, sin tocar el estado
  far = s.predict(k=5)     # el frame t+5: 5 pasos del modelo sobre una copia del estado
  for tok in s.imagine(60):  # rollout: cada frame generado entra al estado
    ...

Adaptadores (make_stream elige):
- TokenStream (genérico: GPT, gpt2m, baselines, cualquier WorldModel): buffer de los últimos ctx_frames
  frames de tokens en el device y model.next_frame.
- MaskGITStream: como TokenStream, pero con el contexto lleno el frame entero (prefill + S pasadas + muestreo)
  es un solo CUDA graph. El ruido del muestreo se genera afuera, en el mismo orden que next_frame, y entra
  como dato: mismos tokens que model.next_frame con el mismo generador.
- LatentStream (LeWM): caché de latentes de los últimos ctx_frames frames; cada frame se codifica una vez.
  Pose opcional por frame (acción); sin pose, la acción nula. LeWM ve píxeles: observe(tokens) los renderiza con
  el decoder de comma, y observe_pixels(img) acepta la imagen de la cámara directo, sin el encoder de comma.

Los pasos que se capturan son funciones puras (estado de entrada y de salida explícitos): el calentamiento de
GraphedStep los ejecuta varias veces. Los tensores que devuelven predict e imagine son nuevos (se pueden guardar).
Con el estado vacío (menos de ctx_frames frames) se usa el camino sin graphs: es el primer segundo del stream.
"""
from typing import Iterator, Optional

import torch

from src.common import BOS, MASK, TPF, with_bos
from src.models.base import sample_logits
from src.models.cudagraph import GraphedStep
from src.models.maskgit import CODES, decode_frame


class Stream:
  """Base: subclases definen reset, _push (un frame entra al estado), _sample_next, _save y _load."""

  def __init__(self, model, temperature: float = 1.0, steps: Optional[int] = None, seed: int = 0,
               graphs: bool = True):
    self.m = model.eval()
    self.dev = model.device
    self.temperature, self.steps = temperature, steps
    self.graphs = graphs and self.dev.type == "cuda"
    self.gen = torch.Generator(self.dev).manual_seed(seed)
    self.ctx_frames = model.ctx_frames
    self.n_seen = 0
    self.reset()

  # ---- API
  def reset(self):
    raise NotImplementedError

  @torch.no_grad()
  def observe(self, tok: torch.Tensor, pose: Optional[torch.Tensor] = None):
    """Un frame real (B, 128) entra al estado. pose (B, 6) cruda, solo para los modelos con acciones."""
    self._push(tok.to(self.dev).long(), None, pose)
    self.n_seen += 1

  @torch.no_grad()
  def predict(self, k: int = 1, pose: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Muestra del frame t+k (B, 128) sin modificar el estado (los k-1 intermedios se descartan)."""
    if k == 1:
      return self._sample_next(pose)[0].clone()
    snap = self._save()
    for i in range(k):
      tok, aux = self._sample_next(pose if i == 0 else None)
      if i + 1 < k:
        self._push(tok, aux, None)
    self._load(snap)
    return tok.clone()

  @torch.no_grad()
  def step(self) -> torch.Tensor:
    """Un frame imaginado (B, 128): se muestrea y entra al estado."""
    tok, aux = self._sample_next(None)
    tok = tok.clone()
    self._push(tok, aux, None)
    self.n_seen += 1
    return tok

  def imagine(self, n: int) -> Iterator[torch.Tensor]:
    """Rollout de n frames: cada uno entra al estado. Itera tokens (B, 128)."""
    for _ in range(n):
      yield self.step()

  @property
  def full(self) -> bool:
    raise NotImplementedError

  # ---- a definir
  def _push(self, tok, aux, pose):
    raise NotImplementedError

  def _sample_next(self, pose):
    """(tokens (B, 128), aux para _push o None)."""
    raise NotImplementedError

  def _save(self):
    raise NotImplementedError

  def _load(self, snap):
    raise NotImplementedError


# ---------------------------------------------------------------- tokens (genérico)

class TokenStream(Stream):
  def reset(self):
    self.ctx = None   # (B, F, 128), F <= ctx_frames
    self.n_seen = 0

  @property
  def full(self):
    return self.ctx is not None and self.ctx.shape[1] == self.ctx_frames

  def _push(self, tok, aux, pose):
    new = tok[:, None]
    self.ctx = new if self.ctx is None else torch.cat([self.ctx, new], 1)[:, -self.ctx_frames:]

  def _sample_next(self, pose):
    return self.m.next_frame(self.ctx, self.temperature, self.steps, self.gen), None

  def _save(self):
    return self.ctx

  def _load(self, snap):
    self.ctx = snap


class MaskGITStream(TokenStream):
  """El frame entero en un CUDA graph cuando el contexto está lleno. Misma lógica que MaskGIT.next_frame."""

  def __init__(self, model, **kw):
    super().__init__(model, **kw)
    self.CODES = CODES
    self.S = self.steps or model.cfg.default_steps
    self.ctx_frames = min(self.ctx_frames, model.cfg.block_size // TPF - 1)  # como next_frame: el frame nuevo cabe
    self._g = {}

  def _noise(self, B):
    """Ruido de muestreo en el orden en que lo pide next_frame: por paso, (B, 128, 1024) y (B, 128)."""
    t, S, c = self.temperature, self.S, self.m.cfg.choice_temp
    if t == 0:
      return []
    out = []
    for s in range(1, S + 1):
      out.append(torch.empty(B, self.CODES, 1024, device=self.dev).exponential_(1, generator=self.gen))
      if c > 0 and s < S:
        out.append(torch.empty(B, self.CODES, device=self.dev).exponential_(1, generator=self.gen))
    return out

  def _frame(self, ctx, *noise):
    """ctx (B, ctx_frames, 128) -> tokens (B, 128). Puro salvo la caché KV (se reescribe entera)."""
    m, S, t = self.m, self.S, self.temperature
    B = ctx.shape[0]
    m._ensure_cache(B, next(m.parameters()).dtype, ctx.device)  # no-op salvo la primera vez (antes de capturar)
    seq = with_bos(ctx)
    P = seq.shape[1]
    pos_all = torch.arange(m.cfg.block_size, device=ctx.device)
    m.net(seq, pos_all[:P])
    pos = pos_all[P:P + TPF]
    tok = torch.full((B, TPF), MASK, dtype=torch.long, device=ctx.device)
    tok[:, 0] = BOS

    def logits_fn(cur):
      tok[:, 1:] = cur
      return m.net(tok, pos)[:, 1:].float()

    cur = torch.full((B, self.CODES), MASK, dtype=torch.long, device=ctx.device)
    return decode_frame(logits_fn, S, t, m.cfg.choice_temp, cur, noise=list(noise)).clone()

  def _sample_next(self, pose):
    if not (self.graphs and self.full):
      return self.m.next_frame(self.ctx, self.temperature, self.S, self.gen), None
    B = self.ctx.shape[0]
    p = next(self.m.parameters())
    self.m._ensure_cache(B, p.dtype, self.dev)
    noise = self._noise(B)
    g = self._g.get(B)
    if g is None or self.m._graphs is not g[1]:  # _ensure_cache rehízo la caché: el graph apunta a la vieja
      g = self._g[B] = (GraphedStep(self._frame, self.ctx, *noise), self.m._graphs)
    return g[0](self.ctx, *noise), None


class GenieStream(TokenStream):
  """Como MaskGITStream, para Genie: prefill del contexto + S pasos + muestreo en un CUDA graph con el contexto lleno."""

  def __init__(self, model, **kw):
    super().__init__(model, **kw)
    self.S = self.steps or model.cfg.default_steps
    self._g = {}

  def _noise(self, B):
    from src.models.genie import make_noise
    return make_noise(B, self.temperature, self.S, self.m.cfg.choice_temp, self.gen, self.dev)

  def _frame(self, ctx, *noise):
    """ctx (B, ctx_frames, 128) -> tokens (B, 128). Puro: no hay estado de caché."""
    return self.m.decode(ctx, self.temperature, self.S, noise=list(noise))

  def _sample_next(self, pose):
    if not (self.graphs and self.full):
      return self.m.next_frame(self.ctx, self.temperature, self.S, self.gen), None
    B = self.ctx.shape[0]
    noise = self._noise(B)
    g = self._g.get(B)
    if g is None:
      g = self._g[B] = GraphedStep(self._frame, self.ctx, *noise)
    return g(self.ctx, *noise), None


# ---------------------------------------------------------------- LeWM

class LatentStream(Stream):
  def reset(self):
    self.z = None       # (B, F, *S) latentes de los últimos ctx_frames frames
    self.acts = None    # (B, F, Da) acción de cada frame del contexto
    self.n_seen = 0

  @property
  def full(self):
    return self.z is not None and self.z.shape[1] == self.ctx_frames

  def _act(self, pose, B):
    m = self.m
    dt = m.null_act.dtype
    if pose is None:
      return m.null_act.expand(B, 1, -1).to(dt)
    pn, valid = m.normalize_poses(pose.to(self.dev).view(B, 1, -1))
    return m.actions_norm(pn, valid).to(dt)

  @torch.no_grad()
  def observe_pixels(self, img: torch.Tensor, pose: Optional[torch.Tensor] = None):
    """Un frame de cámara uint8 (B, 128, 256, 3) en el device entra al estado."""
    m, B = self.m, img.shape[0]
    zn = m._graphed(("encpx", B), self._encode_px, img) if self.graphs else self._encode_px(img)
    self._append(zn, pose)
    self.n_seen += 1

  def _encode_px(self, img):
    return self.m.encode(img[:, None])[:, 0]

  def _push(self, tok, aux, pose):
    m, B = self.m, tok.shape[0]
    if aux is not None and m.cfg.rollout == "latent":
      zn = aux                                   # ẑ entra al contexto sin re-codificar
    else:
      zn = m._graphed(("enc", B), m._encode_one, tok) if self.graphs else m._encode_one(tok)
    self._append(zn, pose)

  def _append(self, zn, pose):
    m, B = self.m, zn.shape[0]
    zn = zn[:, None].to(next(m.parameters()).dtype).clone()
    a = self._act(pose, B)
    if self.z is None:
      self.z, self.acts = zn, a
    else:
      self.z = torch.cat([self.z, zn], 1)[:, -self.ctx_frames:]
      self.acts = torch.cat([self.acts, a], 1)[:, -self.ctx_frames:]

  def _sample_next(self, pose):
    m, B = self.m, self.z.shape[0]
    a = torch.cat([self.acts, self._act(pose, B)], 1).contiguous()
    if self.graphs and self.full:
      zh, lg = m._graphed(("next", B), m._next, self.z, a)
    else:
      zh, lg = m._next(self.z, a)
    return sample_logits(lg.float(), self.temperature, self.gen), zh.clone()

  def _save(self):
    return self.z, self.acts

  def _load(self, snap):
    self.z, self.acts = snap


def make_stream(model, **kw) -> Stream:
  """El adaptador que corresponde a la familia del modelo. kw: temperature, steps, seed, graphs."""
  from src.models.latent import LatentWM
  from src.models.genie import Genie
  from src.models.maskgit import MaskGIT
  if isinstance(model, Genie):
    return GenieStream(model, **kw)
  if isinstance(model, LatentWM):
    if kw.get("graphs", True):
      model.use_cuda_graphs(True)
    return LatentStream(model, **kw)
  if isinstance(model, MaskGIT):
    return MaskGITStream(model, **kw)
  if kw.get("graphs", True) and hasattr(model, "use_cuda_graphs"):
    model.use_cuda_graphs(True)
  return TokenStream(model, **kw)
