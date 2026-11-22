"""Estado de inferencia por frame, residente en el device: el world model como un proceso en línea.

  s = make_stream(model, temperature=1.0, steps=4, seed=0)
  s.observe(tok)           # tokens (B, 128) de un frame real: entra al estado
  nxt = s.predict()        # muestra del frame siguiente, sin tocar el estado
  far = s.predict(k=5)     # el frame t+5: 5 pasos del modelo sobre una copia del estado
  for tok in s.imagine(60):  # rollout: cada frame generado entra al estado
    ...

Adaptadores (make_stream elige):
- TokenStream (genérico: baselines, cualquier WorldModel): buffer de los últimos ctx_frames
  frames de tokens en el device y model.next_frame.
- GPTStream (GPT, gpt2m): como TokenStream, pero con el contexto lleno el frame entero (prefill, 127 pasos de
  decode y el muestreo de los 128 tokens) es un solo CUDA graph. El ruido entra como dato, en el orden de next_frame.
- MaskGITStream: como TokenStream, pero con el contexto lleno el frame entero (prefill + S pasadas + muestreo)
  es un solo CUDA graph. El ruido del muestreo se genera afuera, en el mismo orden que next_frame, y entra
  como dato: mismos tokens que model.next_frame con el mismo generador.
- LatentStream (LeWM): caché de latentes de los últimos ctx_frames frames; cada frame se codifica una vez.
  Pose opcional por frame (acción); sin pose, la acción nula. LeWM ve píxeles: observe(tokens) los renderiza con
  el decoder de comma, y observe_pixels(img) acepta la imagen de la cámara directo, sin el encoder de comma.

- ConvLSTMPxStream (ConvLSTM de Shi 2015, píxeles -> píxeles): ventana de los últimos 10 frames en el device; observe_pixels
  (cámara) o observe (tokens, renderizados); predict/step devuelven uint8 (B, 128, 256, 3), sin codec de comma.

Los pasos que se capturan son funciones puras (estado de entrada y de salida explícitos): el calentamiento de
GraphedStep los ejecuta varias veces. Los tensores que devuelven predict e imagine son nuevos (se pueden guardar).
Con el estado vacío (menos de ctx_frames frames) se usa el camino sin graphs: es el primer segundo del stream.
"""
from typing import Iterator, Optional

import torch

from src.common import BOS, MASK, TPF, with_bos
from src.models.base import frame_noise, sample_logits
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
    m.run(seq, pos_all[:P])
    pos = pos_all[P:P + TPF]
    tok = torch.full((B, TPF), MASK, dtype=torch.long, device=ctx.device)
    tok[:, 0] = BOS
    hist = m.hist_of(seq)  # v2: los frames del contexto que usa la entrada de movimiento

    def logits_fn(cur):
      tok[:, 1:] = cur
      return m.run(tok, pos, hist)[:, 1:].float()

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


class GPTStream(TokenStream):
  """GPT y gpt2m: con el contexto lleno el frame entero (prefill, 127 pasos de decode y los 128 muestreos) es un solo
  CUDA graph. Antes solo el paso de decode de un token era graph y el bucle de 128 tokens, el muestreo y las copias a
  los buffers estáticos corrían en Python. El ruido del muestreo se genera afuera, en una sola llamada como en
  next_frame, y entra como dato: mismos tokens que model.next_frame con el mismo generador."""

  def __init__(self, model, **kw):
    super().__init__(model, **kw)
    cfg = getattr(model, "cfg", None)
    if cfg is not None:   # como next_frame: el frame nuevo cabe en el bloque (un GPT chico de los tests)
      self.ctx_frames = min(self.ctx_frames, cfg.block_size // TPF - 1)
    self._g = {}

  def _noise(self, B):
    """Las tiradas Exp(1) de los 128 tokens (128, B, 1024), en una sola llamada (frame_noise), como next_frame: un
    lanzamiento de la CPU antes del graph del frame, no 128. Con temperatura 0 no hace falta."""
    if self.temperature == 0:
      return []
    return [frame_noise(CODES, B, self.dev, self.gen)]

  def _frame(self, ctx, *noise):
    """ctx (B, ctx_frames, 128) -> tokens (B, 128). Puro salvo la caché KV (se reescribe entera)."""
    return self.m.next_frame(ctx, self.temperature, noise=noise[0] if noise else None, graphs=False)

  def _sample_next(self, pose):
    if not (self.graphs and self.full):
      return self.m.next_frame(self.ctx, self.temperature, self.steps, self.gen), None
    B = self.ctx.shape[0]
    p = next(self.m.parameters())
    self.m._ensure_cache(B, p.dtype, p.device)  # no-op salvo la primera vez (antes de capturar)
    noise = self._noise(B)
    key = (B, self.ctx.shape[1])   # el graph tiene la forma del contexto con que se capturó
    g = self._g.get(key)
    if g is None or self.m._graphs is not g[1]:  # _ensure_cache rehízo la caché: el graph apunta a la vieja
      g = self._g[key] = (GraphedStep(self._frame, self.ctx, *noise), self.m._graphs)
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
  """LeWM en línea, todo en el device: cámara (píxeles) -> ViT -> predictor, sin el codec de comma.

  La cámara va a 20 fps y el modelo da un paso cada `frameskip` frames de cámara (5: 250 ms). observe_pixels(img, pose)
  se llama en cada frame de cámara: solo el primero de cada bloque se codifica (el resto solo acumula su pose, que
  forma la acción del bloque: las k poses apiladas). El estado son los últimos `history` embeddings y las acciones
  ya tomadas entre ellos. predict() devuelve el embedding ẑ del próximo paso del modelo (B, dim): LeWM no emite tokens;
  con `decoder` (src/models/lewm_decoder.py) predict_pixels() lo pasa a imagen.

  pose de predict/step: la acción que se va a tomar tras el último frame codificado, (B, 6) cruda (se mantiene los k
  frames de cámara) o (B, 6k) apilada; None = acción 0 (la media).
  """

  def __init__(self, model, temperature: float = 1.0, steps=None, seed: int = 0, graphs: bool = True, decoder=None):
    self.decoder = decoder
    super().__init__(model, temperature=temperature, steps=steps, seed=seed, graphs=graphs)
    self.k = model.cfg.frameskip

  def reset(self):
    self.z = None       # (B, n, D) embeddings de los últimos `history` pasos del modelo
    self.acts = None    # (B, n - 1, D) acción embebida entre cada par de frames del estado
    self._poses = []    # poses (B, 6) crudas del bloque de cámara en curso
    self._cam = 0       # frames de cámara vistos (fase del bloque)
    self.n_seen = 0

  @property
  def full(self):
    return self.z is not None and self.z.shape[1] == self.ctx_frames

  # ---- acciones
  def _act(self, pose, B):
    """Acción embebida (B, 1, D) de una pose (B, 6) / (B, 6k) cruda; None = acción 0."""
    m = self.m
    if pose is None:
      return m.action_embed(None, B, 1, self.dev)
    pose = pose.to(self.dev).float().view(B, 1, -1)
    return m.action_embed(pose, B, 1, self.dev)

  # ---- observar
  @torch.no_grad()
  def observe_pixels(self, img: torch.Tensor, pose: Optional[torch.Tensor] = None) -> bool:
    """Un frame de cámara uint8 (B, 128, 256, 3) en el device y su pose cruda (B, 6) (None: sin pose).

    Devuelve True si el modelo avanzó un paso (cada `frameskip` frames de cámara)."""
    B = img.shape[0]
    new_block = self._cam % self.k == 0
    if new_block:
      self._close_block(B)
      zn = self.m._graphed(("encpx", B), self._encode_px, img) if self.graphs else self._encode_px(img)
      self._append(zn)
    p = torch.full((B, 6), float("nan"), device=self.dev) if pose is None else pose.to(self.dev).float().view(B, -1)
    self._poses.append(p)
    self._cam += 1
    self.n_seen += int(new_block)
    return new_block

  @torch.no_grad()
  def observe(self, tok: torch.Tensor, pose: Optional[torch.Tensor] = None) -> bool:
    """Como observe_pixels, desde los tokens del frame de cámara (se renderizan con el decoder de comma)."""
    from src.models.lewm import render_tokens
    tok = tok.to(self.dev).long()
    img = render_tokens(tok) if self._cam % self.k == 0 else tok.new_zeros(tok.shape[0], 1, 1, 3, dtype=torch.uint8)
    return self.observe_pixels(img, pose)

  def _encode_px(self, img):
    return self.m.encode(img[:, None])[:, 0]

  def _close_block(self, B):
    """Termina el bloque anterior: sus k poses apiladas son la acción entre el frame anterior y el que entra."""
    if self.z is None or not self._poses:
      self._poses = []
      return
    a = self.m.action_embed(torch.cat(self._poses, -1)[:, None], B, 1, self.dev)
    self.acts = torch.cat([self.acts, a.to(self.acts.dtype)], 1)[:, -(self.ctx_frames - 1):]
    self._poses = []

  def _append(self, zn):
    zn = zn[:, None].to(next(self.m.parameters()).dtype).clone()
    if self.z is None:
      self.z = zn
      self.acts = zn.new_zeros(zn.shape[0], 0, zn.shape[-1])
    else:
      self.z = torch.cat([self.z, zn], 1)[:, -self.ctx_frames:]

  # ---- predecir
  def _step_from(self, z, acts, pose):
    """(ẑ del paso siguiente (B, D), acción tras el último frame (B, 1, D))."""
    a_last = self._act(pose, z.shape[0]).to(acts.dtype)
    a = torch.cat([acts, a_last], 1).contiguous()
    full = z.shape[1] == self.ctx_frames
    zh = self.m._graphed(("next", z.shape[0]), self.m._next, z, a) if (self.graphs and full) else self.m._next(z, a)
    return zh, a_last

  @torch.no_grad()
  def predict(self, k: int = 1, pose: Optional[torch.Tensor] = None) -> torch.Tensor:
    """ẑ del paso k del modelo (B, D) sin tocar el estado; los pasos intermedios usan la acción 0."""
    z, acts = self.z, self.acts
    for i in range(k):
      zh, a_last = self._step_from(z, acts, pose if i == 0 else None)
      zh = zh.clone()
      if i + 1 < k:
        z = torch.cat([z, zh[:, None].to(z.dtype)], 1)[:, -self.ctx_frames:]
        acts = torch.cat([acts, a_last], 1)[:, -(self.ctx_frames - 1):]
    return zh

  @torch.no_grad()
  def predict_pixels(self, k: int = 1, pose: Optional[torch.Tensor] = None) -> torch.Tensor:
    """predict() pasado por el decoder de visualización: uint8 (B, 128, 256, 3)."""
    assert self.decoder is not None, "LatentStream sin decoder (src/models/lewm_decoder.py)"
    return self.decoder.decode(self.predict(k, pose))

  @torch.no_grad()
  def step(self, pose: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Un paso imaginado: ẑ (B, D) entra al estado (rollout latente)."""
    zh, a_last = self._step_from(self.z, self.acts, pose)
    zh = zh.clone()
    self.acts = torch.cat([self.acts, a_last], 1)[:, -(self.ctx_frames - 1):]
    self._append(zh)
    self.n_seen += 1
    return zh

  def _save(self):
    return self.z, self.acts

  def _load(self, snap):
    self.z, self.acts = snap


# ---------------------------------------------------------------- ConvLSTM desde píxeles

class ConvLSTMPxStream(Stream):
  """ConvLSTM de Shi 2015 (output = "pixels"): entra y sale píxeles uint8 (B, 128, 256, 3), sin codec de comma.

  El estado es la ventana de los últimos n_in frames (float en [0, 1], en el device). Cada predicción re-codifica la
  ventana desde estado 0 (como se entrenó) y pronostica; es una función pura de la ventana, así que se captura en un
  CUDA graph con la ventana llena. step() pronostica n_out frames de una vez, los va entregando de a uno y cada uno
  entra a la ventana (el mismo rollout por bloques que model.generate). observe() descarta lo que quede por entregar.
  Sin .item()/.cpu() en la ruta por frame; observe también acepta tokens (B, 128), que renderiza en el device.
  """

  def reset(self):
    self.win = None     # (B, F, 3, H, W), F <= n_in
    self.queue = None   # (B, K, 3, H, W) pronóstico pendiente de entregar
    self.qi = 0
    self.n_seen = 0
    self._g = {}

  @property
  def full(self):
    return self.win is not None and self.win.shape[1] == self.ctx_frames

  def _enter(self, p):
    p = p[:, None].to(self.win.dtype if self.win is not None else next(self.m.parameters()).dtype)
    self.win = p if self.win is None else torch.cat([self.win, p], 1)[:, -self.ctx_frames:]

  @torch.no_grad()
  def observe_pixels(self, img: torch.Tensor, pose=None):
    """Un frame de cámara uint8 (B, 128, 256, 3) en el device entra a la ventana."""
    self._enter(self.m.to_float(img.to(self.dev)))
    self.queue = None
    self.n_seen += 1

  @torch.no_grad()
  def observe(self, tok, pose=None):
    if tok.dim() == 4:   # imagen
      return self.observe_pixels(tok)
    from src.pixels import render
    self.observe_pixels(render(tok.to(self.dev).long()))

  def _probs(self, n):
    """Pronóstico de n frames desde la ventana (graph con la ventana llena)."""
    if not (self.graphs and self.full):
      return self.m.probs(self.win, n)
    key = (self.win.shape[0], n)
    g = self._g.get(key)
    if g is None:
      g = self._g[key] = GraphedStep(lambda w: self.m.probs(w, n), self.win)
    return g(self.win)

  @torch.no_grad()
  def predict(self, k: int = 1, pose=None) -> torch.Tensor:
    """Frame t+k (B, 128, 256, 3) uint8: los primeros k pasos del forecaster; no modifica el estado."""
    return self.m.to_uint8(self._probs(k)[:, -1])

  @torch.no_grad()
  def step(self) -> torch.Tensor:
    if self.queue is None or self.qi >= self.queue.shape[1]:
      self.queue, self.qi = self._probs(self.m.cfg.n_out).clone(), 0
    p = self.queue[:, self.qi]
    self.qi += 1
    self._enter(p)
    self.n_seen += 1
    return self.m.to_uint8(p)


# ---------------------------------------------------------------- V-JEPA 2-AC

class VJEPAStream:
  """V-JEPA 2-AC en línea: cámara a 20 Hz -> un paso del predictor cada 5 frames (4 Hz, 0.25 s), sin codec.

    s = make_stream(model)               # model: VJEPA2AC
    s.observe_pixels(img, pose)          # img uint8 (B, 128, 256, 3) en el device; pose (B, n_pose) cruda o None
                                         # devuelve True si ese frame entró al estado (1 de cada `cada` = 5)
    zh = s.predict(accion)               # (B, N, D) frame imaginado 0.25 s adelante (layer_norm); acción normalizada (B, A)
    zh = s.step(accion)                  # como predict, y el frame imaginado entra al estado (rollout)

  Frame de cámara -> encoder (un frame en un tubelet duplicado) -> latentes del estado; el predictor ve las últimas
  `ctx` (15) representaciones con sus estados (pose normalizada) y acciones (diferencia de estados observados; la del
  último frame es la que se pasa a predict). Todo es tensor en el device, con contadores de Python: sin .item/.cpu. Con la
  ventana llena el encoder y el paso del predictor son CUDA graphs (GraphedStep) de forma estática. La latencia de un paso
  es la del predictor (más un encoder cada 5 frames de cámara): ver latencia_ms().
  """

  def __init__(self, model, graphs: bool = True, cada: int = 5, ctx: Optional[int] = None, **_):
    self.m = model.eval()
    self.dev = model.device
    self.graphs = graphs and self.dev.type == "cuda"
    self.cada = cada
    self.ctx = ctx or model.ctx_frames
    self._g = {}
    self.reset()

  @property
  def full(self):
    return self.z is not None and self.z.shape[1] == self.ctx

  def reset(self):
    self.z = self.s = self.a = None  # (B, F, N, D), (B, F, A), (B, F-1, A)
    self.n_cam = 0

  def _enc(self, img):
    from src.models.vjepa.patch import normalizar_cthw
    x = normalizar_cthw(img.float().div(255).permute(0, 3, 1, 2)[:, :, None])
    return self.m.codificar(x)[:, 0]

  def _pred(self, z, a, s):
    return self.m.predecir(z, a, s)[:, -1]

  def _run(self, key, fn, *args):
    if not self.graphs:
      return fn(*args)
    g = self._g.get(key)
    if g is None:
      g = self._g[key] = GraphedStep(fn, *args)
    return g(*args)

  def _push(self, z, s, a):
    if self.z is None:
      self.z, self.s, self.a = z, s, s[:, :0].clone()
      return
    self.z = torch.cat([self.z, z], 1)[:, -self.ctx:]
    self.a = torch.cat([self.a, a], 1)[:, -(self.ctx - 1):] if self.ctx > 1 else self.a
    self.s = torch.cat([self.s, s], 1)[:, -self.ctx:]

  @torch.no_grad()
  def observe_pixels(self, img, pose=None):
    n = self.n_cam
    self.n_cam += 1
    if n % self.cada:
      return False
    B = img.shape[0]
    z = self._run(("enc", B), self._enc, img.to(self.dev))[:, None].clone()
    s, _ = self.m.estados_acciones(None if pose is None else pose.to(self.dev)[:, None], B, 1, self.dev)
    self._push(z, s, None if self.s is None else s - self.s[:, -1:])
    return True

  @torch.no_grad()
  def predict(self, accion=None):
    B = self.z.shape[0]
    acc = torch.zeros(B, 1, self.s.shape[-1], device=self.dev) if accion is None else accion.to(self.dev).view(B, 1, -1)
    a = torch.cat([self.a, acc], 1)
    return self._run(("pred", B), self._pred, self.z, a, self.s).clone() if self.full else self._pred(self.z, a, self.s)

  @torch.no_grad()
  def step(self, accion=None):
    B = self.z.shape[0]
    acc = torch.zeros(B, 1, self.s.shape[-1], device=self.dev) if accion is None else accion.to(self.dev).view(B, 1, -1)
    zh = self.predict(accion)
    self._push(zh[:, None], self.s[:, -1:] + acc, acc)
    return zh


def latencia_ms(stream, img, n: int = 20) -> float:
  """Milisegundos por paso del predictor (con la ventana ya llena); sincroniza solo al medir."""
  import time
  for _ in range(3):
    stream.predict()
  if stream.dev.type == "cuda":
    torch.cuda.synchronize()
  t0 = time.perf_counter()
  for _ in range(n):
    stream.predict()
  if stream.dev.type == "cuda":
    torch.cuda.synchronize()
  return (time.perf_counter() - t0) / n * 1000


def make_stream(model, **kw) -> Stream:
  """El adaptador que corresponde a la familia del modelo. kw: temperature, steps, seed, graphs."""
  from src.models.latent import LatentWM
  from src.models.genie import Genie
  from src.models.vjepa.action import VJEPA2AC
  if isinstance(model, VJEPA2AC):
    return VJEPAStream(model, **kw)
  from src.models.maskgit import MaskGIT
  from src.models.convlstm_px import ConvLSTMPx
  if isinstance(model, ConvLSTMPx):
    return ConvLSTMPxStream(model, **kw)
  if isinstance(model, Genie):
    return GenieStream(model, **kw)
  if isinstance(model, LatentWM):
    if kw.get("graphs", True):
      model.use_cuda_graphs(True)
    return LatentStream(model, **kw)
  if isinstance(model, MaskGIT):
    return MaskGITStream(model, **kw)
  if kw.get("graphs", True) and hasattr(model, "use_cuda_graphs"):
    model.use_cuda_graphs(True)   # el primer segundo (contexto incompleto) sigue por el decode con un graph por paso
  from src.models.gpt import GPT
  if isinstance(model, GPT) or type(model).__name__ == "Gpt2m":   # gpt2m se importa recién al usarlo (third_party/commavq)
    return GPTStream(model, **kw)
  return TokenStream(model, **kw)
