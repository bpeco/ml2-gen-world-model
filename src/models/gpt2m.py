"""gpt2m (commaai/commavq-gpt2m): el GPT de comma como WorldModel. Referencia y maestro."""
import sys

import torch

from src.common import BLOCK, BOS, with_bos
from src.models.base import WorldModel, sample_logits
from src.models.cudagraph import GraphedStep

sys.path.insert(0, "third_party/commavq")
from utils.gpt import GPT, GPTConfig, KVCache, find_multiple  # noqa: E402

FIDELIDAD = [
  dict(campo="Modelo", fuente_valor="commaai/commavq-gpt2m: 24 capas, 16 cabezas, 1024, MLP 4096, vocab 1025, bloque 2580",
       aquí="el código y los pesos de comma, sin cambios (third_party/commavq/utils/gpt.py)", categoría="fiel",
       fuente="commaVQ utils/gpt.py L26-123; config.json"),
  dict(campo="Pesos", fuente_valor="pytorch_model.bin de Hugging Face", aquí="data/raw/pytorch_model.bin, bajado por src.data",
       categoría="fiel", fuente="HF commaai/commavq-gpt2m"),
  dict(campo="Muestreo y KV caché", fuente_valor="generación autoregresiva con KV caché",
       aquí="la misma con caché estática propia (y CUDA graphs opcional); mismas distribuciones", categoría="propio",
       fuente="src/models/gpt2m.py: adaptación a la interfaz WorldModel"),
  dict(campo="Receta de entrenamiento", fuente_valor="'a similar recipe to IRIS' (no publicada)", aquí="no se entrena aquí; "
       "es la referencia y el maestro de la destilación", categoría="fiel", fuente="commaVQ README L354"),
]

WEIGHTS = "data/raw/pytorch_model.bin"  # commaai/commavq-gpt2m, bajado por src.data


class Gpt2m(WorldModel):
  name = "gpt2m"
  ctx_frames = 19
  has_likelihood = True

  def __init__(self):
    super().__init__()
    self.gpt = GPT(GPTConfig())
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
    """Cachés KV y máscara causal de tamaño L; se reutilizan si no cambia batch, dtype ni device."""
    c = self.gpt.config
    key = (B, dtype, dev)
    if self._caches is None or self._caches[0] != key:
      kv = [KVCache(B, L, c.n_head, c.head_dim, dtype=dtype).to(dev) for _ in self.gpt.transformer.h]
      mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev)).view(1, 1, L, L)
      self._caches = (key, kv, mask)
      self._graphs = {}  # los graphs apuntan a las cachés viejas
    return self._caches[1], self._caches[2]

  def use_cuda_graphs(self, enable: bool = True):
    """El paso de decode de un token (B, 1) corre como CUDA graph; el prefill queda eager."""
    self._graphs_on, self._graphs = enable, {}

  def _decode(self, tok, pos):
    return self.gpt(tok, pos)[:, -1, :BOS]

  @torch.no_grad()
  def next_frame(self, ctx, temperature=1.0, steps=None, generator=None):
    """(B, F, 128) -> (B, 128). Prefill del contexto + BOS y 128 pasos de decode, en batch."""
    ctx = ctx[:, -self.ctx_frames:]
    B, dev = ctx.shape[0], ctx.device
    dtype = next(self.gpt.parameters()).dtype
    L = find_multiple(BLOCK, 8)
    kv, mask = self._setup(B, L, dtype, dev)
    old_mask = self.gpt.causal_mask
    for blk, c in zip(self.gpt.transformer.h, kv):
      blk.attn.kv_cache = c
    self.gpt.causal_mask = mask
    try:
      seq = torch.cat([with_bos(ctx), torch.full((B, 1), BOS, dtype=ctx.dtype, device=dev)], 1)
      P = seq.shape[1]
      pos_all = torch.arange(L, device=dev)
      lg = self.gpt(seq, pos_all[:P])[:, -1, :BOS]  # nunca emite BOS
      out = [sample_logits(lg, temperature, generator)]
      use_graph = self._graphs_on and dev.type == "cuda"
      for k in range(1, 128):
        tok, pos = out[-1][:, None], pos_all[P + k - 1:P + k]
        if use_graph:
          g = self._graphs.get(B)
          if g is None:
            g = self._graphs[B] = GraphedStep(self._decode, tok, pos)
          lg = g(tok, pos)
        else:
          lg = self._decode(tok, pos)
        out.append(sample_logits(lg, temperature, generator))
      return torch.stack(out, 1)
    finally:  # logits() vuelve a funcionar sin caché
      for blk in self.gpt.transformer.h:
        blk.attn.kv_cache = None
      self.gpt.causal_mask = old_mask
