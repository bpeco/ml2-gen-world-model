"""Tests de sanidad de la métrica de giro (src/metrica_giro.py, docs/metrica_giro.md).

Usan RAFT-small y el decoder de commaVQ sobre 16 curvas reales (8 a cada lado) de results/clips_motion.json; se saltean sin CUDA o sin
los datos. Cada test construye un video con un movimiento conocido y comprueba lo que la métrica tiene que dar:
  - desplazamiento horizontal de k px por frame: flow horizontal ≈ k y giro lineal en k
  - el clip real invertido en el tiempo: giro ≈ -1
  - escena congelada (el último frame del contexto repetido): giro ≈ 0
  - parpadeo puro (tokens re-muestreados al azar sobre una escena fija): giro ≈ 0 aunque cambien muchos tokens
  - flip horizontal del clip (modelo y real): el giro no cambia (la métrica es simétrica izquierda/derecha)
  python -m pytest tests/test_metrica_giro.py -q
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "results/clips_motion.json"
N = 16

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="sin CUDA"),
              pytest.mark.skipif(not CLIPS.exists(), reason="falta results/clips_motion.json")]


@lru_cache(maxsize=1)
def entorno():
  import os
  os.chdir(ROOT)
  from src.metrics_flow import Flujo
  from src.viz import load_decoder
  dev = torch.device("cuda")
  d = json.loads(CLIPS.read_text())
  # balanceado: N/2 curvas a cada lado. Con un solo lado, cualquier sesgo horizontal constante del flow (RAFT sobre
  # artefactos deriva hacia un lado) se confunde con giro; ver docs/metrica_giro.md, limitaciones.
  cur = [c for c in d["clips"] if c["tipo"] == "curva" and c["yaw_3s"] > 0][:N // 2] + \
        [c for c in d["clips"] if c["tipo"] == "curva" and c["yaw_3s"] < 0][:N // 2]
  T = np.load(ROOT / d["tokens"], mmap_mode="r")
  toks = np.stack([T[c["segment"], c["start"] + 19:c["start"] + 80] for c in cur]).astype(np.int64)  # (N, 61, 128)
  return Flujo(dev), load_decoder(dev), toks


def rot(frames):
  """frames uint8 (N, 61, H, W, 3) -> flow horizontal (N, 60) en la franja de 64 filas."""
  fl = entorno()[0]
  return np.stack([fl(f[:-1], f[1:])["rot"] for f in frames])


def decodificar(toks):
  from src.viz import decode
  dec = entorno()[1]
  return np.stack([decode(dec, t, bs=32) for t in toks])


@lru_cache(maxsize=1)
def reales():
  toks = entorno()[2]
  fr = decodificar(toks)
  return fr, rot(fr)


def test_desplazamiento_lineal():
  from src.metrica_giro import giro
  base = reales()[0][:, 0]                                        # (N, H, W, 3): un frame real por clip
  ancho = base.shape[2]
  medidos = {}
  for k in (1, 2, 4):
    # cámara que gira: la imagen se corre k px por frame; se recorta de un lienzo más ancho (sin bordes envueltos)
    lienzo = np.concatenate([base[:, :, ::-1], base, base[:, :, ::-1]], 2)
    fr = np.stack([lienzo[:, :, ancho - k * t:2 * ancho - k * t] for t in range(61)], 1)
    medidos[k] = rot(fr)
    assert abs(medidos[k].mean() - k) < 0.15 * k, (k, medidos[k].mean())
  g2 = giro(medidos[2], medidos[1], 0, 60)
  g4 = giro(medidos[4], medidos[1], 0, 60)
  assert abs(g2 - 2) < 0.2 and abs(g4 - 4) < 0.4, (g2, g4)


def test_tiempo_invertido():
  from src.metrica_giro import giro
  fr, r = reales()
  rev = rot(fr[:, ::-1].copy())
  g = giro(rev, r, 0, 60)
  assert abs(g + 1) < 0.15, g


def test_congelado():
  from src.metrica_giro import giro
  fr, r = reales()
  quieto = np.repeat(fr[:, :1], 61, 1)
  g = giro(rot(quieto), r, 0, 60)
  assert abs(g) < 0.05, g


def test_parpadeo_puro():
  from src.metrica_giro import giro
  toks = entorno()[2]
  rng = np.random.default_rng(0)
  pf = np.repeat(toks[:, :1], 61, 1).copy()
  cambia = rng.random(pf[:, 1:].shape) < 0.3
  pf[:, 1:] = np.where(cambia, rng.integers(0, 1024, pf[:, 1:].shape), pf[:, 1:])
  tasa = (pf[:, 1:] != pf[:, :-1]).mean()
  assert tasa > 0.3                                                # ~0,5: muy por encima del piso del codec (0,135)
  # RAFT sobre tokens al azar da flow grande (~2 px) y distinto en cada clip: con 16 clips el valor puntual tiene
  # ruido de ~0,2, así que se pide que el IC 95 % por bootstrap contenga el 0 (no hay giro sistemático)
  g, lo, hi = giro(rot(decodificar(pf)), reales()[1], 0, 60, boot=2000)
  assert lo < 0 < hi and abs(g) < 0.5, (g, lo, hi, tasa)


def test_flip_simetrico():
  from src.metrica_giro import giro
  fr, r = reales()
  # "modelo" = el real con la mitad del movimiento: se repite cada frame dos veces (gira a la mitad de velocidad)
  lento = fr[:, np.minimum(np.arange(61) // 2, 60)]
  g = giro(rot(lento), r, 0, 60)
  g_flip = giro(rot(lento[:, :, :, ::-1].copy()), rot(fr[:, :, :, ::-1].copy()), 0, 60)
  assert 0.3 < g < 0.7, g
  assert abs(g - g_flip) < 0.1, (g, g_flip)
