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
- ConvLSTMStream: un paso recurrente por frame, con (h, c) explícitos. El estado arrastra todo el stream
  (no solo los últimos 19 frames); reset_every reinicia cada tanto desde los últimos ctx_frames frames.
- LatentStream (JEPA, LeWM): caché de latentes de los últimos ctx_frames frames; cada frame se codifica una
  vez. Pose opcional por frame (acción); sin pose, la acción nula. LeWM desde píxeles (cfg.input = "pixels")
  también acepta la imagen de la cámara directo: observe_pixels(img), sin el encoder de comma.

Los pasos que se capturan son funciones puras (estado de entrada y de salida explícitos): el calentamiento de
GraphedStep los ejecuta varias veces. Los tensores que devuelven predict e imagine son nuevos (se pueden guardar).
Con el estado vacío (menos de ctx_frames frames) se usa el camino sin graphs: es el primer segundo del stream.
"""
from typing import Iterator, Optional

import torch

from src.common import BOS, MASK, TPF, with_bos
from src.models.base import sample_logits
from src.models.cudagraph import GraphedStep


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
    from src.models.maskgit import CODES, n_masked
    self.CODES, self.n_masked = CODES, n_masked
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
    m, S, t, CODES = self.m, self.S, self.temperature, self.CODES
    B = ctx.shape[0]
    m._ensure_cache(B, next(m.parameters()).dtype, ctx.device)  # no-op salvo la primera vez (antes de capturar)
    seq = with_bos(ctx)
    P = seq.shape[1]
    pos_all = torch.arange(m.cfg.block_size, device=ctx.device)
    m.net(seq, pos_all[:P])
    pos = pos_all[P:P + TPF]
    tok = torch.full((B, TPF), MASK, dtype=torch.long, device=ctx.device)
    tok[:, 0] = BOS
    fixed = torch.zeros(B, CODES, dtype=torch.bool, device=ctx.device)
    n_prev, j = CODES, 0
    for s in range(1, S + 1):
      lg = m.net(tok, pos)[:, 1:].float()
      if t == 0:
        samp = lg.argmax(-1)
      else:
        probs = torch.softmax(lg / t, -1)
        samp = (probs / noise[j]).argmax(-1)
        j += 1
      lp = torch.log_softmax(lg / t if t > 0 else lg, -1).gather(-1, samp[..., None])[..., 0]
      if t > 0 and m.cfg.choice_temp > 0 and s < S:
        lp = lp + m.cfg.choice_temp * (1 - s / S) * -torch.log(noise[j])
        j += 1
      n_s = self.n_masked(s, S)
      k = n_prev - n_s
      lp = lp.masked_fill(fixed, float("-inf"))
      if k > 0:
        fixed = fixed.scatter(1, lp.topk(k, dim=-1).indices, True)
      tok[:, 1:] = torch.where(fixed.logical_and(tok[:, 1:] == MASK), samp, tok[:, 1:])
      n_prev = n_s
    return tok[:, 1:].clone()

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


# ---------------------------------------------------------------- ConvLSTM

def convlstm_step(m, x, *state):
  """Un paso de ConvLSTM: x (B, emb, 8, 16), state [h, c] por capa -> (logits (B, 128, 1024), *state nuevo).
  La misma cuenta que ConvLSTM._run para un frame (también se exporta a ONNX en src/infer/export_codec.py)."""
  out = []
  for l in range(m.cfg.n_layers):
    h, c = state[2 * l], state[2 * l + 1]
    i, f, g, o = (m.conv_x[l](x) + m.conv_h[l](h)).chunk(4, 1)
    c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
    h = torch.sigmoid(o) * torch.tanh(c)
    out += [h, c]
    x = h
  B = x.shape[0]
  lg = m.head(x).permute(0, 2, 3, 1).reshape(B, -1, 1024)
  return (lg, *out)


class ConvLSTMStream(Stream):
  """reset_every: cada cuántos frames observados se recalcula el estado desde los últimos ctx_frames frames
  (como en el entrenamiento, que usa ventanas de 20). None = el estado arrastra todo el stream."""

  def __init__(self, model, reset_every: Optional[int] = None, **kw):
    self.reset_every = reset_every
    super().__init__(model, **kw)
    self._g = {}

  def reset(self):
    self.state = None   # [h0, c0, h1, c1, ...]
    self.lg = None      # logits del frame siguiente (B, 128, 1024)
    self.hist = None    # últimos ctx_frames tokens, para reset_every
    self.n_state = 0    # frames dentro del estado actual
    self.n_seen = 0

  @property
  def full(self):
    return self.n_state >= self.ctx_frames

  def _step(self, x, *state):
    return convlstm_step(self.m, x, *state)

  def _run_step(self, x):
    if self.state is None:
      B, Hd = x.shape[0], self.m.cfg.hidden
      z = x.new_zeros(B, Hd, *x.shape[-2:])
      self.state = [z] * (2 * self.m.cfg.n_layers)
    if self.graphs:
      g = self._g.get(x.shape[0])
      if g is None:
        g = self._g[x.shape[0]] = GraphedStep(self._step, x, *self.state)
      out = g(x, *self.state)
    else:
      out = self._step(x, *self.state)
    self.lg, self.state = out[0].clone(), [t.clone() for t in out[1:]]

  def _start_x(self, B):
    return self.m.start[None].expand(B, -1, -1, -1).to(next(self.m.parameters()).dtype)

  def _embed(self, tok):
    return self.m._embed(tok[:, None])[:, 0]

  def _push(self, tok, aux, pose):
    if self.lg is None:  # primer frame: el paso de "start" predice el frame 0
      self._run_step(self._start_x(tok.shape[0]))
    self.hist = tok[:, None] if self.hist is None else torch.cat([self.hist, tok[:, None]], 1)[:, -self.ctx_frames:]
    self._run_step(self._embed(tok))
    self.n_state += 1
    if self.reset_every and self.n_state >= self.reset_every:
      self._rewarm()

  def _rewarm(self):
    """Estado desde cero con los últimos ctx_frames frames (lo que vería el modelo en una ventana nueva)."""
    hist = self.hist
    self.state, self.lg, self.n_state = None, None, 0
    self._run_step(self._start_x(hist.shape[0]))
    for f in range(hist.shape[1]):
      self._run_step(self._embed(hist[:, f]))
      self.n_state += 1

  def _sample_next(self, pose):
    if self.lg is None:
      raise RuntimeError("observe() al menos un frame antes de predecir")
    return sample_logits(self.lg.float(), self.temperature, self.gen), None

  def _save(self):
    return self.state, self.lg, self.hist, self.n_state

  def _load(self, snap):
    self.state, self.lg, self.hist, self.n_state = snap


# ---------------------------------------------------------------- JEPA y LeWM

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

  @property
  def pixels(self) -> bool:
    """El modelo recibe la imagen (LeWM con input="pixels"): observe_pixels no necesita el encoder de comma."""
    return getattr(self.m.cfg, "input", "tokens") == "pixels"

  @torch.no_grad()
  def observe_pixels(self, img: torch.Tensor, pose: Optional[torch.Tensor] = None):
    """Un frame de cámara uint8 (B, 128, 256, 3) en el device entra al estado (solo modelos desde píxeles)."""
    assert self.pixels, "observe_pixels es para modelos con input='pixels'; los demás usan observe(tokens)"
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
  """El adaptador que corresponde a la familia del modelo. kw: temperature, steps, seed, graphs (y reset_every
  para ConvLSTM)."""
  from src.models.convlstm import ConvLSTM
  from src.models.latent import LatentWM
  from src.models.maskgit import MaskGIT
  if isinstance(model, ConvLSTM):
    return ConvLSTMStream(model, **kw)
  kw.pop("reset_every", None)
  if isinstance(model, LatentWM):
    if kw.get("graphs", True):
      model.use_cuda_graphs(True)
    return LatentStream(model, **kw)
  if isinstance(model, MaskGIT):
    return MaskGITStream(model, **kw)
  if kw.get("graphs", True) and hasattr(model, "use_cuda_graphs"):
    model.use_cuda_graphs(True)
  return TokenStream(model, **kw)
