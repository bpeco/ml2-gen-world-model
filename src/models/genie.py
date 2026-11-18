"""Modelo de dinámica de Genie (Bruce et al. 2024, arXiv 2402.15391) con decodificación MaskGIT, sin acciones.

Es el "video MaskGIT" de la comparación según el paper de Genie: ST-transformer (atención espacial, atención temporal
causal, un FFW por bloque) entrenado con máscara Bernoulli de tasa ~ U[0.5, 1] y decodificado por frame con pasos
MaskGIT. No lleva el modelo de acciones latentes de Genie (nuestros world models no ven acciones, como el GPT de
commaVQ). Código propio y autocontenido: no importa nada de maskgit.py, para que afinar uno no mueva al otro.

Formato: frames (B, F, 128) con tokens 0..1023, sin BOS (la posición espacial y temporal ya dice dónde está cada
token). La entrada puede llevar MASK (src.common). Cada frame de la secuencia ve a los anteriores (misma celda,
atención temporal causal) y a todo su propio frame (atención espacial).

FIDELIDAD (abajo) lista cada decisión con su fuente y su categoría: fiel (del paper), datos (forzado por los datos),
presupuesto (forzado por el cómputo) o propio (el paper no lo publica y lo elegimos nosotros).
"""
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import schedules as S
from src.common import MASK, TPF
from src.models.base import WorldModel, lm_loss
from src.models.cudagraph import GraphedStep
from src.recipes import Recipe

CODES = TPF - 1  # 128 tokens por frame (8 x 16)
GENIE = "Genie (arXiv 2402.15391)"
MASKGIT = "MaskGIT (arXiv 2202.04200)"

FIDELIDAD = [
  dict(campo="Bloque", fuente_valor="atención espacial sobre los H×W tokens de cada instante, atención temporal causal "
       "sobre los T tokens de la misma posición, un FFW por bloque (solo después de la temporal)",
       aquí="igual", categoría="fiel", fuente=f"{GENIE} §2 'ST-transformer', Fig. 4"),
  dict(campo="Máscara causal en el tiempo", fuente_valor="dinámica causal en el tiempo, estilo MaskGIT", aquí="igual",
       categoría="fiel", fuente=f"{GENIE} §2.3"),
  dict(campo="QK-norm", fuente_valor="se usa en la dinámica (no dice qué norma)", aquí="LayerNorm sobre head_dim en q y k",
       categoría="fiel", fuente=f"{GENIE} §3; tipo de norma: propio (como ViT-22B)"),
  dict(campo="Posición de las normas", fuente_valor="no publicada", aquí="pre-LN antes de cada subcapa, residuales, "
       "LayerNorm final", categoría="propio", fuente="Genie no lo publica; es la elección estándar (Jafar, no oficial, también)"),
  dict(campo="Posición", fuente_valor="no publicada", aquí="embedding aprendido espacial (128 celdas) + temporal "
       "(frames), sumados al token", categoría="propio", fuente="Genie no lo publica"),
  dict(campo="Token de máscara", fuente_valor="los tokens enmascarados se reemplazan por un token de máscara",
       aquí="vector aprendido", categoría="fiel", fuente=f"{GENIE} §2.3 (aprendido: propio, igual que Jafar)"),
  dict(campo="Acciones", fuente_valor="embeddings aditivos de acciones latentes", aquí="ninguna (action-free)",
       categoría="datos", fuente=f"{GENIE} §2.3; nuestros world models no ven acciones"),
  dict(campo="Frames por secuencia (T)", fuente_valor="16", aquí="16", categoría="fiel", fuente=f"{GENIE} §3"),
  dict(campo="Frame rate", fuente_valor="10 fps", aquí="20 fps", categoría="datos",
       fuente=f"{GENIE} §3; la tarea es video comma a 20 fps"),
  dict(campo="Tokens por frame / vocabulario", fuente_valor="tokenizador propio (1024 códigos, 16×16)",
       aquí="128 tokens (8×16), 1024 códigos de comma", categoría="datos", fuente="commaVQ"),
  dict(campo="Máscara en entrenamiento", fuente_valor="Bernoulli por token con tasa ~ U[0.5, 1], frame 1 sin máscara",
       aquí="igual, sobre los frames 2..T (leemos 'z_{2:T-1}' como 2..T: si no, el último frame nunca se predeciría)",
       categoría="fiel", fuente=f"{GENIE} §2.3"),
  dict(campo="Granularidad de la tasa", fuente_valor="no dice si es por secuencia o por frame", aquí="una tasa por "
       "secuencia (muestra)", categoría="propio", fuente="Genie no lo dice"),
  dict(campo="Loss", fuente_valor="cross-entropy sobre los tokens enmascarados", aquí="igual, sin label smoothing "
       "ni dropout", categoría="fiel", fuente=f"{GENIE} §2.3; {MASKGIT} Eq. 1 (regularización: propio, no publicada para Genie)"),
  dict(campo="Pasos de decodificación", fuente_valor="25 pasos MaskGIT por frame", aquí="25 por defecto (parámetro)",
       categoría="fiel", fuente=f"{GENIE} §3, Tab. 17"),
  dict(campo="Temperatura de muestreo", fuente_valor="2 (CoinRun: 1.0)", aquí="2.0 por defecto (parámetro)",
       categoría="fiel", fuente=f"{GENIE} §3, Tab. 17"),
  dict(campo="Calendario de máscara en inferencia", fuente_valor="coseno: quedan floor(128·cos(π/2·s/S)) enmascarados",
       aquí="igual", categoría="fiel", fuente=f"{MASKGIT} §3.2"),
  dict(campo="Choice temperature", fuente_valor="Genie no la publica; MaskGIT: 4.5·(1 − (t+1)/T), ruido Gumbel sobre log p",
       aquí="4.5 anillada linealmente a 0 (parámetro); la confianza es log p del token muestreado con softmax a "
       "temperatura 1, como MaskGIT", categoría="propio", fuente=f"{MASKGIT} (código oficial); combinación con la "
       "temperatura 2 de Genie: propio"),
  dict(campo="Contexto al generar", fuente_valor="los frames previos más el nuevo todo enmascarado",
       aquí="los últimos T−1 = 15 frames reales/generados + el frame nuevo todo MASK", categoría="fiel",
       fuente=f"{GENIE} §2.3 (T = 16); recorte de la ventana: propio"),
  dict(campo="Tamaño", fuente_valor="el más chico es 41M: 18 capas, 8 cabezas, d_model 512, k/q 64",
       aquí="S = 6 × 384 (6 cabezas), M = 8 × 512 (8 cabezas), MLP 4×", categoría="presupuesto",
       fuente=f"{GENIE} Tab. 10; mismos tamaños que el GPT y el MaskGIT propios"),
  dict(campo="Inicialización", fuente_valor="no publicada", aquí="N(0, 0.02) en Linear y Embedding, bias 0",
       categoría="propio", fuente="convención GPT-2/MaskGIT"),
  dict(campo="Precisión", fuente_valor="bf16", aquí="bf16 (autocast; solo en CUDA)", categoría="fiel",
       fuente=f"{GENIE} §3"),
  dict(campo="Optimizador", fuente_valor="AdamW, β = (0.9, 0.9)", aquí="igual, eps 1e-8", categoría="fiel",
       fuente=f"{GENIE} Tab. 9"),
  dict(campo="Weight decay", fuente_valor="1e-4", aquí="1e-4 en todo lo de >= 2 dimensiones (no biases ni normas)",
       categoría="fiel", fuente=f"{GENIE} Tab. 9; a qué parámetros se aplica: propio"),
  dict(campo="Learning rate: valores", fuente_valor="max 3e-5, min 3e-6", aquí="igual", categoría="fiel",
       fuente=f"{GENIE} Tab. 9"),
  dict(campo="Warmup", fuente_valor="5k pasos de 200k (fracción 0.025)", aquí="fracción 0.025 de la corrida, piso 100 pasos",
       categoría="presupuesto", fuente=f"{GENIE} Tab. 9, Tab. 10; nuestra corrida tiene muchos menos pasos"),
  dict(campo="Forma tras el warmup", fuente_valor="no publicada para la dinámica (el tokenizador usa cosine decay)",
       aquí="coseno de 3e-5 a 3e-6", categoría="propio", fuente=f"{GENIE} App. F.1; Jafar (no oficial) también usa coseno"),
  dict(campo="Batch", fuente_valor="256", aquí="32 por paso = 16 x accum 2 (~1.520 pasos con 780k frames); lr a confirmar en el humo", categoría="presupuesto",
       fuente=f"{GENIE} Tab. 10; micro-batch por la memoria de 12 GB"),
  dict(campo="Grad clip", fuente_valor="no publicado", aquí="1.0", categoría="propio", fuente="Genie no lo publica"),
  dict(campo="Pasos", fuente_valor="200k", aquí="los que da el presupuesto de 780k frames (~1.520 pasos de 32 x 16 frames)",
       categoría="presupuesto", fuente=f"{GENIE} Tab. 10"),
]

# Capacidades que expone la CLI (src/cli.py, src/toolkit.py): solo las acciones que este modelo soporta.
ACCIONES = ("receta", "entrenar", "continuar", "evaluar", "bench", "demo")
EVALUACION = "video"  # "video": CE + rollouts + LPIPS/FVD; "latente": src.eval_latent


def n_masked(s: int, S: int) -> int:
  """Tokens que siguen enmascarados tras el paso s de S (calendario coseno de MaskGIT; n_S = 0)."""
  return 0 if s >= S else int(math.floor(CODES * math.cos(math.pi / 2 * s / S)))


def make_noise(B, temperature, steps, choice_temp, generator, device):
  """Ruido del decodificador en el orden en que lo pide decode_frame: por paso (B, 128, 1024) y (B, 128)."""
  if temperature == 0:
    return []
  out = []
  for s in range(1, steps + 1):
    out.append(torch.empty(B, CODES, 1024, device=device).exponential_(1, generator=generator))
    if choice_temp > 0 and s < steps:
      out.append(torch.empty(B, CODES, device=device).exponential_(1, generator=generator))
  return out


def decode_frame(logits_fn, steps, temperature, choice_temp, cur, generator=None, noise=None):
  """Decodificación iterativa de MaskGIT para un frame (propia de Genie; sampler de MaskGIT §3.2).

  cur (B, 128): todo MASK al empezar. logits_fn(cur) -> (B, 128, 1024) float32. En cada paso se muestrea cada celda a
  `temperature` (0 = argmax), la confianza es log p (softmax a temperatura 1) del token muestreado más ruido Gumbel de
  choice_temp·(1 − s/S), y se fijan los más confiables hasta dejar n_masked(s, S) en MASK.
  noise: None (sale de generator) o la lista de make_noise (para capturar el frame entero en un CUDA graph).
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
    lp = torch.log_softmax(lg, -1).gather(-1, samp[..., None])[..., 0]
    if temperature > 0 and choice_temp > 0 and s < steps:
      q = noise[j] if noise is not None else torch.empty_like(lp).exponential_(1, generator=generator)
      j += 1
      lp = lp + choice_temp * (1 - s / steps) * -torch.log(q)
    k = n_prev - n_masked(s, steps)
    lp = lp.masked_fill(fixed, float("-inf"))
    if k > 0:
      fixed = fixed.scatter(1, lp.topk(k, dim=-1).indices, True)
    cur = torch.where(fixed & (cur == MASK), samp, cur)
    n_prev -= k
  return cur


class Attn(nn.Module):
  """Atención multi-cabeza con QK-norm sobre secuencias (G, L, D). Con `past` (k, v de L' pasos previos), los L pasos
  nuevos los ven a todos (se usa con L = 1: el frame nuevo mira el contexto)."""

  def __init__(self, dim, n_head):
    super().__init__()
    self.h, self.hd = n_head, dim // n_head
    self.qkv = nn.Linear(dim, 3 * dim)
    self.q_norm, self.k_norm = nn.LayerNorm(self.hd), nn.LayerNorm(self.hd)
    self.proj = nn.Linear(dim, dim)

  def forward(self, x, causal=False, past=None):
    G, L, D = x.shape
    q, k, v = self.qkv(x).view(G, L, 3, self.h, self.hd).unbind(2)
    q, k = self.q_norm(q), self.k_norm(k)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (G, H, L, hd)
    new = (k, v)
    if past is not None:
      k, v = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=causal and past is None)
    return self.proj(y.transpose(1, 2).reshape(G, L, D)), new


class Block(nn.Module):
  """Atención espacial → atención temporal causal → FFW (Genie §2, Fig. 4). Pre-LN con residuales."""

  def __init__(self, dim, n_head, mlp_ratio):
    super().__init__()
    self.ln_s, self.ln_t, self.ln_f = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
    self.spatial, self.temporal = Attn(dim, n_head), Attn(dim, n_head)
    self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))

  def forward(self, x, past=None):
    """x (B, T, N, D); past: (k, v) temporales del contexto (B*N, H, Tp, hd) o None. Devuelve (x, (k, v) propios)."""
    B, T, N, D = x.shape
    h = self.ln_s(x).reshape(B * T, N, D)
    x = x + self.spatial(h)[0].view(B, T, N, D)
    h = self.ln_t(x).transpose(1, 2).reshape(B * N, T, D)
    y, kv = self.temporal(h, causal=True, past=past)
    x = x + y.view(B, N, T, D).transpose(1, 2)
    return x + self.mlp(self.ln_f(x)), kv


class STTransformer(nn.Module):
  def __init__(self, cfg: "Genie.Config"):
    super().__init__()
    d = cfg.dim
    self.tok = nn.Embedding(1024, d)
    self.mask_tok = nn.Parameter(torch.zeros(d))
    self.pos_s = nn.Parameter(torch.zeros(CODES, d))
    self.pos_t = nn.Parameter(torch.zeros(cfg.frames, d))
    self.blocks = nn.ModuleList(Block(d, cfg.n_head, cfg.mlp_ratio) for _ in range(cfg.n_layer))
    self.ln_out = nn.LayerNorm(d)
    self.head = nn.Linear(d, 1024)
    self.apply(self._init)
    for p in (self.mask_tok, self.pos_s, self.pos_t):
      nn.init.normal_(p, std=0.02)

  @staticmethod
  def _init(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
      nn.init.normal_(m.weight, std=0.02)
      if isinstance(m, nn.Linear):
        nn.init.zeros_(m.bias)

  def forward(self, tok, t0=0, past=None, kv=False):
    """tok (B, T, 128) con MASK -> logits (B, T, 128, 1024). t0: posición temporal del primer frame.
    past: lista por capa de (k, v) del contexto. kv=True devuelve también los (k, v) de cada capa."""
    B, T, N = tok.shape
    x = torch.where((tok == MASK)[..., None], self.mask_tok.to(self.tok.weight.dtype),
                    self.tok(tok.clamp(max=1023)))
    x = x + self.pos_s[None, None] + self.pos_t[t0:t0 + T][None, :, None]
    kvs = []
    for i, blk in enumerate(self.blocks):
      x, kv_i = blk(x, None if past is None else past[i])
      kvs.append(kv_i)
    lg = self.head(self.ln_out(x))
    return (lg, kvs) if kv else lg


class Genie(WorldModel):
  name = "genie"
  has_likelihood = False

  @dataclass
  class Config:
    n_layer: int = 6
    n_head: int = 6
    dim: int = 384
    mlp_ratio: int = 4
    frames: int = 16               # T de entrenamiento (Genie: 16); al generar, T-1 de contexto + el frame nuevo
    mask_min: float = 0.5          # tasa de máscara ~ U[mask_min, mask_max] por secuencia (Genie §2.3)
    mask_max: float = 1.0
    default_steps: int = 25        # pasos MaskGIT por frame (Genie)
    temperature: float = 2.0       # muestreo (Genie); next_frame(temperature=None) usa este
    choice_temp: float = 4.5       # ruido Gumbel de la confianza, anillado a 0 (MaskGIT)

  @classmethod
  def recipe(cls, size=None):
    """Receta de Genie (Tab. 9): AdamW (0.9, 0.9), wd 1e-4, lr 3e-5 -> 3e-6, warmup 5k de 200k (fracción 0.025), bf16,
    batch 256, 16 frames. El decaimiento tras el warmup no se publica para la dinámica: coseno (propio)."""
    return Recipe(
      name="genie_paper", fuente=f"{GENIE} Tab. 9, 10, App. F.1; schedule y clip: propio", optimizer="adamw",
      betas=(0.9, 0.9), eps=1e-8,
      lr_segs=[S.linear(0.0, 3e-5, frac=0.025, nombre="warmup"), S.cosine(3e-5, 0.1)],
      wd_ini=1e-4, wd_on="matrices", grad_clip=1.0, amp="bf16", batch=16, accum=2, frames=16, paper_steps=200_000, batch_fuente=256,
      forzado=[("batch", "32 muestras por paso (16 x accum 2) en vez de 256: con el presupuesto fijo de 780k frames da ~1.520 pasos; "
                         "lr a confirmar en el humo", "presupuesto"),
               ("warmup", "0.025 de la corrida (5k de 200k) con piso de 100 pasos: el presupuesto da pocos pasos", "presupuesto"),
               ("lr_segs", "coseno hasta 3e-6 tras el warmup: Genie no publica la forma para la dinámica", "propio"),
               ("grad_clip", "1.0: Genie no lo publica", "propio")])

  def __init__(self, cfg: "Genie.Config"):
    super().__init__()
    self.cfg = cfg
    self.ctx_frames = cfg.frames - 1
    self.net = STTransformer(cfg)
    self._graphs_on = False
    self._graphs = {}

  # ---- entrenamiento / evaluación
  def logits(self, x):
    """(B, L) secuencia comma (BOS antes de cada frame, con MASK posible) -> (B, L, 1024); 0 en los BOS."""
    B, L = x.shape
    fr = x.view(B, L // TPF, TPF)[..., 1:]
    out = x.new_zeros(B, L // TPF, TPF, 1024, dtype=self.net.head.weight.dtype)
    out[:, :, 1:] = self.net(fr)
    return out.view(B, L, 1024)

  def mask_for(self, B, Fr, dev):
    """Máscara de entrenamiento (B, Fr, 128): Bernoulli con tasa ~ U[mask_min, mask_max] por secuencia; frame 1 limpio."""
    c = self.cfg
    r = c.mask_min + (c.mask_max - c.mask_min) * torch.rand(B, 1, 1, device=dev)
    m = torch.rand(B, Fr, CODES, device=dev) < r
    m[:, 0] = False
    return m

  def train_loss(self, frames, teacher=None, alpha=0.5, tau=1.0):
    if teacher is not None:
      raise ValueError("Genie no se destila: teacher tiene que ser None")
    B, Fr, _ = frames.shape
    if Fr > self.cfg.frames:
      raise ValueError(f"Genie entrena con a lo sumo {self.cfg.frames} frames, llegaron {Fr}")
    masked = self.mask_for(B, Fr, frames.device)
    lg = self.net(torch.where(masked, torch.full_like(frames, MASK), frames))
    loss, m = lm_loss(lg, frames, mask=masked)
    m["mask_rate"] = masked.float().mean().detach()
    return loss, m

  def last_frame_logits(self, frames):
    """Último frame entero enmascarado, una pasada: (B, 128, 1024). Es la "CE con 1 paso"."""
    x = frames[:, -self.cfg.frames:].clone()
    x[:, -1] = MASK
    return self.net(x)[:, -1]

  # ---- generación
  def use_cuda_graphs(self, enable: bool = True):
    self._graphs_on = enable
    self._graphs = {}

  def _step(self, cur, *past):
    """Un paso de decodificación: el frame nuevo (B, 128) con MASK, contra el contexto en caché -> (B, 128, 1024)."""
    P = past[0].shape[2] if past else 0
    pk = [(past[2 * i], past[2 * i + 1]) for i in range(len(past) // 2)] or None
    return self.net(cur[:, None], P, pk)[:, 0].float()

  def decode(self, ctx, temperature=None, steps=None, generator=None, noise=None):
    """Frame siguiente (B, 128) dado el contexto (B, P, 128), P <= ctx_frames. Puro (sin estado): lo captura el stream.

    Prefill del contexto una vez (k, v temporales de cada capa) y los S pasos solo sobre el frame nuevo."""
    c = self.cfg
    t = c.temperature if temperature is None else temperature
    S = steps or c.default_steps
    ctx = ctx[:, -self.ctx_frames:]
    B, P = ctx.shape[:2]
    if self._graphs_on and ctx.device.type == "cuda" and noise is None:
      # el frame entero (prefill + S pasos + muestreo) en un solo graph; el ruido se genera afuera, en el device
      noise = make_noise(B, t, S, c.choice_temp, generator, ctx.device)
      key = (B, P, t, S)
      g = self._graphs.get(key)
      if g is None:
        g = self._graphs[key] = GraphedStep(lambda x, *n: self._decode(x, t, S, list(n)), ctx, *noise)
      return g(ctx, *noise).clone()
    return self._decode(ctx, t, S, noise, generator)

  def _decode(self, ctx, t, S, noise=None, generator=None):
    B, P = ctx.shape[:2]
    past = []
    if P:
      _, kvs = self.net(ctx, 0, kv=True)
      past = [t_ for kv_ in kvs for t_ in kv_]
    cur = torch.full((B, CODES), MASK, dtype=torch.long, device=ctx.device)
    return decode_frame(lambda cur: self._step(cur, *past), S, t, self.cfg.choice_temp, cur, generator, noise).clone()

  @torch.no_grad()
  def next_frame(self, ctx, temperature=None, steps=None, generator=None):
    return self.decode(ctx, temperature, steps, generator)
