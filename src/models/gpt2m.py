"""gpt2m (commaai/commavq-gpt2m): el GPT de comma como WorldModel. Referencia y maestro."""
import sys
import types

import torch
import torch.nn.functional as F

from src.common import BLOCK, BOS, TPF, with_bos
from src.models.backbone import additive_mask, cached_attention
from src.models.base import WorldModel, frame_noise, sample_logits
from src.models.cudagraph import GraphedStep

sys.path.insert(0, "third_party/commavq")
from utils.gpt import GPT, GPTConfig, KVCache, find_multiple  # noqa: E402

FIDELIDAD = [
  dict(campo="Modelo", fuente_valor="commaai/commavq-gpt2m: 24 capas, 16 cabezas, 1024, MLP 4096, vocab 1025, bloque 2580",
       aquí="el código y los pesos de comma, sin cambios (third_party/commavq/utils/gpt.py)", categoría="fiel",
       fuente="commaVQ utils/gpt.py L26-123; config.json"),
  dict(campo="Pesos", fuente_valor="pytorch_model.bin de Hugging Face", aquí="data/raw/pytorch_model.bin, bajado por src.data",
       categoría="fiel", fuente="HF commaai/commavq-gpt2m"),
  dict(campo="Muestreo y KV caché", fuente_valor="generación autoregresiva con KV caché; el decode fuerza el backend "
       "math de SDPA (atención explícita)",
       aquí="la misma con caché estática propia (y CUDA graphs opcional); mismas distribuciones. La atención de decode "
       "es la cuenta explícita (cached_attention), como la del backend math que fuerza comma; SDPA sin forzar elige "
       "mem-efficient, que con una query deja la GPU casi vacía", categoría="propio",
       fuente="commaVQ utils/gpt.py L162-171 (decode_n_tokens); src/models/gpt2m.py: adaptación a la interfaz WorldModel"),
  dict(campo="Receta de entrenamiento", fuente_valor="'a similar recipe to IRIS' (no publicada)", aquí="no se entrena aquí; "
       "es la referencia y el maestro de la destilación", categoría="fiel", fuente="commaVQ README L354"),
]

WEIGHTS = "data/raw/pytorch_model.bin"  # commaai/commavq-gpt2m, bajado por src.data


def _attn_forward(self, x, mask, input_pos=None):
  """Attention.forward de comma (third_party/commavq/utils/gpt.py L81-99) con la caché enganchada: lo mismo (split,
  kv_cache.update, c_proj), salvo que con pocas queries (decode: 1) la atención es cached_attention en vez de SDPA.
  Es la cuenta del backend math, el que fuerza decode_n_tokens de comma. mask: aditiva (Gpt2m._setup)."""
  bsz, seqlen, _ = x.shape
  c = self.config
  q, k, v = self.c_attn(x).split([c.dim, c.dim, c.dim], dim=-1)
  q, k, v = (t.view(bsz, seqlen, c.n_head, c.head_dim).transpose(1, 2) for t in (q, k, v))
  k, v = self.kv_cache.update(input_pos, k, v)
  if seqlen <= TPF:
    y = cached_attention(q, k, v, mask)
  else:   # prefill: SDPA, como comma
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
  return self.c_proj(y.transpose(1, 2).contiguous().view(bsz, seqlen, c.dim))


class Gpt2m(WorldModel):
  name = "gpt2m"
  ctx_frames = 19
  has_likelihood = True

  def __init__(self, config=None):
    """config: solo para los tests (un GPT chico con la misma forma de secuencia); el modelo real es GPTConfig()."""
    super().__init__()
    self.gpt = GPT(config or GPTConfig())
    self._caches = None   # lista de KVCache; solo se enganchan al modelo durante next_frame
    self._mask = None
    self._graphs_on, self._graphs = False, {}

  @classmethod
  def load(cls, dev="cpu", path=WEIGHTS):
    m = cls()
    # reusa el loader de comma (transpone pesos de HF) pero leyendo el archivo local
    hub_load = torch.hub.load_state_dict_from_url
    torch.hub.load_state_dict_from_url = lambda url, **kw: torch.load(path, map_location="cpu", weights_only=True)
    try:
      m.gpt.load_state_dict_from_url()
    finally:
      torch.hub.load_state_dict_from_url = hub_load
    return m.to(dev).eval()

  def logits(self, x):
    """(B, L) -> (B, L, 1025), sin caché (las cachés solo viven dentro de next_frame)."""
    return self.gpt(x)

  def _setup(self, B, L, dtype, dev):
    """Cachés KV y máscara causal de tamaño L, aditiva en el dtype de la caché (para SDPA es lo mismo que la bool, y
    evita convertirla en cada capa); se reutilizan si no cambia batch, dtype ni device."""
    c = self.gpt.config
    key = (B, dtype, dev)
    if self._caches is None or self._caches[0] != key:
      kv = [KVCache(B, L, c.n_head, c.head_dim, dtype=dtype).to(dev) for _ in self.gpt.transformer.h]
      mask = additive_mask(torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev)), dtype).view(1, 1, L, L)
      self._caches = (key, kv, mask)
      self._graphs = {}  # los graphs apuntan a las cachés viejas
    return self._caches[1], self._caches[2]

  def use_cuda_graphs(self, enable: bool = True):
    """El paso de decode de un token (B, 1) corre como CUDA graph; el prefill queda eager."""
    self._graphs_on, self._graphs = enable, {}

  def _decode(self, tok, pos):
    return self.gpt(tok, pos)[:, -1, :BOS]

  def _ensure_cache(self, B, dtype, dev):
    """Reserva las cachés KV antes de capturar el frame entero en un graph (GPTStream); no-op si ya están."""
    self._setup(B, find_multiple(BLOCK, 8), dtype, dev)

  @torch.no_grad()
  def next_frame(self, ctx, temperature=1.0, steps=None, generator=None, noise=None, graphs=None):
    """(B, F, 128) -> (B, 128). Prefill del contexto + BOS y 128 pasos de decode, en batch.

    noise: (128, B, 1024) float32, las tiradas Exp(1) de los 128 tokens. Sin noise (y con temperatura), se tiran acá
    de una vez con el generador (frame_noise), como hace GPTStream fuera del graph del frame. graphs: None = lo que
    dice use_cuda_graphs; False fuerza el decode sin graph por paso (los graphs no se anidan)."""
    ctx = ctx[:, -self.ctx_frames:]
    B, dev = ctx.shape[0], ctx.device
    if noise is None and temperature > 0:
      noise = frame_noise(128, B, dev, generator)
    dtype = next(self.gpt.parameters()).dtype
    L = find_multiple(BLOCK, 8)
    kv, mask = self._setup(B, L, dtype, dev)
    old_mask = self.gpt.causal_mask
    for blk, c in zip(self.gpt.transformer.h, kv):
      blk.attn.kv_cache = c
      blk.attn.forward = types.MethodType(_attn_forward, blk.attn)   # solo mientras la caché está enganchada
    self.gpt.causal_mask = mask
    try:
      seq = torch.cat([with_bos(ctx), torch.full((B, 1), BOS, dtype=ctx.dtype, device=dev)], 1)
      P = seq.shape[1]
      pos_all = torch.arange(L, device=dev)
      lg = self.gpt(seq, pos_all[:P])[:, -1, :BOS]  # nunca emite BOS
      out = [sample_logits(lg, temperature, generator, None if noise is None else noise[0])]
      use_graph = (self._graphs_on if graphs is None else graphs) and dev.type == "cuda"
      for k in range(1, 128):
        tok, pos = out[-1][:, None], pos_all[P + k - 1:P + k]
        if use_graph:
          g = self._graphs.get(B)
          if g is None:
            g = self._graphs[B] = GraphedStep(self._decode, tok, pos)
          lg = g(tok, pos)
        else:
          lg = self._decode(tok, pos)
        out.append(sample_logits(lg, temperature, generator, None if noise is None else noise[k]))
      return torch.stack(out, 1)
    finally:  # logits() vuelve a funcionar sin caché, con el forward de comma
      for blk in self.gpt.transformer.h:
        blk.attn.kv_cache = None
        blk.attn.__dict__.pop("forward", None)
      self.gpt.causal_mask = old_mask
