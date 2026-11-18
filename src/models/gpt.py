"""GPT alumno autoregresivo: el modelo de mundo de commaVQ a menor escala (mismo formato: BOS antes de cada frame).

Arquitectura de commaVQ (utils/gpt.py): pre-LN, GELU, bias, posición absoluta aprendida, sin weight tying, vocab 1025,
bloque de 20 frames x 129. Receta: la de IRIS, que commaVQ dice seguir ("a similar recipe to IRIS"). FIDELIDAD (abajo)
lista cada decisión con su fuente y categoría: fiel | datos | presupuesto | propio.
"""
from dataclasses import dataclass

import torch

from src.common import BLOCK, BOS, TPF, VOCAB, with_bos
from src.models.backbone import Backbone, BackboneConfig
from src.models.base import WorldModel, lm_loss, sample_logits
from src.models.cudagraph import GraphedStep
from src import schedules as S
from src.recipes import Recipe


COMMA = "commaVQ (utils/gpt.py, commaai/commavq-gpt2m)"
IRIS = "IRIS (arXiv 2209.00588; github.com/eloialonso/iris)"

FIDELIDAD = [
  dict(campo="Arquitectura", fuente_valor="pre-LN, GELU (tanh), MLP 4x, bias en Linear, ln_f, cabeza sin bias",
       aquí="igual", categoría="fiel", fuente=f"{COMMA} L63-123"),
  dict(campo="Posición", fuente_valor="absoluta aprendida (wpe) sobre 2580 posiciones", aquí="igual (pos='abs')",
       categoría="fiel", fuente=f"{COMMA} L117; {IRIS} world_model.py L38"),
  dict(campo="Weight tying", fuente_valor="sin tying (lm_head aparte)", aquí="sin tying", categoría="fiel",
       fuente=f"{COMMA} L123"),
  dict(campo="Formato de secuencia", fuente_valor="128 tokens + BOS (1024) por frame, bloque de 20 frames, vocab 1025",
       aquí="igual", categoría="fiel", fuente=f"{COMMA}; nanogpt/prepare.py L16, L29-31"),
  dict(campo="Tamaño", fuente_valor="24 capas, 16 cabezas, 1024 (comma); 10 capas, 256 (IRIS)",
       aquí="S = 6 x 384 (6 cabezas), M = 8 x 512 (8 cabezas), L = 12 x 768", categoría="presupuesto",
       fuente=f"{COMMA} config.json; {IRIS} world_model/default.yaml"),
  dict(campo="Posición factorizada (ablación)", fuente_valor="no existe en comma ni IRIS",
       aquí="pos='frame': embedding de frame + de celda; solo para correr las corridas viejas", categoría="propio",
       fuente="decisión del repo (corridas previas a la revisión de fidelidad)"),
  dict(campo="Dropout", fuente_valor="IRIS: 0.1 en embedding, residual y atención; comma: 0 (dropout_p=0.0)",
       aquí="0.1 en los tres (default de Config; el de IRIS, que es la receta que comma dice seguir)",
       categoría="fiel", fuente=f"{IRIS} world_model/default.yaml L8-10; {COMMA} L97"),
  dict(campo="Inicialización", fuente_valor="N(0, 0.02) en Linear y Embedding, bias 0, LayerNorm 1/0",
       aquí="igual, sin la escala 1/sqrt(2L) de GPT-2 en las proyecciones residuales (res_scale=False)",
       categoría="fiel", fuente=f"{IRIS} utils.py L52-59"),
  dict(campo="Optimizador", fuente_valor="AdamW, betas (0.9, 0.999)", aquí="igual, eps 1e-8", categoría="fiel",
       fuente=f"{IRIS} trainer.py L91, Tab. 5"),
  dict(campo="Learning rate", fuente_valor="1e-4 constante, sin warmup ni scheduler", aquí="igual", categoría="fiel",
       fuente=f"{IRIS} config/trainer.yaml L55"),
  dict(campo="Weight decay", fuente_valor="0.01 solo en el weight de nn.Linear y nn.Conv1d (no bias, LayerNorm ni "
       "Embedding, incluida la posición)", aquí="igual (wd_on='lineales'; incluye la cabeza)", categoría="fiel",
       fuente=f"{IRIS} trainer.yaml L66, utils.py L14-36"),
  dict(campo="Grad clip", fuente_valor="10.0", aquí="10.0", categoría="fiel", fuente=f"{IRIS} trainer.yaml L65"),
  dict(campo="Batch", fuente_valor="64 secuencias de 20 pasos", aquí="16 por paso, sin acumulación (~2.440 pasos con 780k frames); "
       "lr a confirmar en el humo", categoría="presupuesto", fuente=f"{IRIS} trainer.yaml, world_model/default.yaml L2-3"),
  dict(campo="Precisión", fuente_valor="fp32 (sin autocast)", aquí="fp32", categoría="fiel", fuente=f"{IRIS} trainer.py"),
  dict(campo="Pasos / presupuesto", fuente_valor="no publicado para comma", aquí="780k frames (100M tokens)",
       categoría="presupuesto", fuente="presupuesto de cómputo del repo"),
  dict(campo="Datos", fuente_valor="segmentos de 1200 frames a 20 fps", aquí="igual (ventanas de 20 frames)",
       categoría="datos", fuente=f"{COMMA} nanogpt/prepare.py"),
  dict(campo="Destilación (--kd)", fuente_valor="no existe en comma ni IRIS",
       aquí="loss = alpha CE + (1-alpha) tau^2 KL(maestro gpt2m || alumno), alpha 0.5, tau 1, misma receta que el alumno",
       categoría="propio", fuente="Hinton et al. 2015; extensión del repo"),
]

# Capacidades que expone la CLI (src/cli.py, src/toolkit.py): solo las acciones que este modelo soporta.
ACCIONES = ("receta", "entrenar", "continuar", "evaluar", "bench", "demo")
EVALUACION = "video"  # "video": CE + rollouts + LPIPS/FVD; "latente": src.eval_latent


class GPT(WorldModel):
  name = "gpt"
  ctx_frames = 19
  has_likelihood = True

  @dataclass
  class Config:
    n_layer: int = 6
    n_head: int = 6
    dim: int = 384
    dropout: float = 0.1            # IRIS: embedding, residual y atención (comma: 0)
    block_size: int = BLOCK
    pos: str = "abs"                # embedding de posición: "abs" | "frame" (frame + celda)

  @classmethod
  def recipe(cls, size=None):
    """Receta de IRIS (la que commaVQ dice seguir): AdamW (0.9, 0.999), lr 1e-4 constante, wd 0.01 solo en Linear,
    clip 10, batch 64 (aquí 16: ver forzado), fp32. Con teacher (--kd) es la misma más la pérdida de destilación."""
    return Recipe(
      name="gpt_iris", fuente=f"{IRIS}; commaVQ: 'a similar recipe to IRIS'", optimizer="adamw", betas=(0.9, 0.999),
      eps=1e-8, lr_segs=[S.constant(1e-4)], wd_ini=0.01, wd_on="lineales", grad_clip=10.0, amp=None,
      batch=16, accum=1, frames=20, extras={"dropout": 0.1}, paper_steps=None, batch_fuente=64,
      forzado=[("batch", "16 por paso en vez de 64 (IRIS): con el presupuesto fijo de 780k frames da ~2.440 pasos; "
                         "lr a confirmar en el humo", "presupuesto"),
               ("presupuesto", "780k frames (100M tokens) en vez de lo que haya usado comma (no publicado)", "presupuesto"),
               ("tamaño", "6x384 / 8x512 en vez de 24x1024", "presupuesto")])

  def __init__(self, cfg: "GPT.Config"):
    super().__init__()
    self.cfg = cfg
    self.net = Backbone(BackboneConfig(n_layer=cfg.n_layer, n_head=cfg.n_head, dim=cfg.dim,
                                       block_size=cfg.block_size, vocab_in=VOCAB, vocab_out=VOCAB,
                                       attn="causal", dropout=cfg.dropout, pos=cfg.pos,
                                       res_scale=False))
    self._graphs_on = False
    self._graphs = {}
    self._cache_key = None

  def _drop_cache(self):
    if self._cache_key is not None:  # el camino sin caché no admite cachés activas
      self.net.clear_cache()
      self._cache_key, self._graphs = None, {}

  def logits(self, x):
    self._drop_cache()
    return self.net(x)

  def train_loss(self, frames, teacher=None, alpha=0.5, tau=1.0):
    seq = with_bos(frames)
    x, y = seq[:, :-1], seq[:, 1:]
    lg = self.logits(x)
    tl = None
    if teacher is not None:
      with torch.no_grad():
        tl = teacher.logits(x)
    return lm_loss(lg, y, tl, alpha, tau)

  def use_cuda_graphs(self, enable: bool = True):
    self._graphs_on = enable
    self._graphs = {}

  def _ensure_cache(self, B, dtype, device):
    key = (B, dtype, device)
    if key != self._cache_key:
      self.net.setup_cache(B, dtype=dtype, device=device)
      self._cache_key = key
      self._graphs = {}  # los graphs apuntan a las cachés viejas

  def _decode(self, tok, pos):
    return self.net(tok, pos)[:, -1]

  @torch.no_grad()
  def next_frame(self, ctx, temperature=1.0, steps=None, generator=None):
    B = ctx.shape[0]
    p = next(self.parameters())
    self._ensure_cache(B, p.dtype, p.device)
    seq = with_bos(ctx[:, -self.ctx_frames:])
    seq = torch.cat([seq, torch.full_like(seq[:, :1], BOS)], 1)  # BOS del frame a generar
    P = seq.shape[1]
    pos_all = torch.arange(self.cfg.block_size, device=p.device)
    lg = self.net(seq, pos_all[:P])[:, -1]  # prefill; el contexto se re-prefillea en cada frame
    out = []
    use_graph = self._graphs_on and p.device.type == "cuda"
    for i in range(TPF - 1):
      tok = sample_logits(lg[:, :BOS], temperature, generator)  # nunca emite BOS
      out.append(tok)
      if i == TPF - 2:
        break
      tok, pos = tok[:, None], pos_all[P + i:P + i + 1]
      if use_graph:
        g = self._graphs.get(B)
        if g is None:
          g = self._graphs[B] = GraphedStep(self._decode, tok, pos)
        lg = g(tok, pos)
      else:
        lg = self._decode(tok, pos)
    return torch.stack(out, 1)
