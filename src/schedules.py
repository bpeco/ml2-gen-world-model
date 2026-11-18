"""Formas de schedule (lr, wd, momento de EMA) como listas de tramos; funciones puras de (paso, total).

Un tramo es un dict (serializable a JSON):
  nombre   etiqueta del tramo ("warmup", "constante", "decay"...)
  forma    "linear" | "constant" | "cosine"
  ini, fin valores de los extremos (constant usa ini; cosine usa ini y min_frac)
  min_frac cosine: el valor final es ini * min_frac
  frac     largo como fracción del total de pasos de la corrida
  pasos    largo absoluto en pasos (si están frac y pasos, manda pasos)
  min_pasos piso del largo cuando se escala con frac (warmup: 100 si no se indica)
Un tramo sin frac ni pasos ocupa el resto de la corrida (lo que dejan los demás; p. ej. la meseta entre
  warmup y decay). Los largos se recortan al total en orden.

Convenciones (las de train.py de siempre, para reproducirlo bit a bit):
  linear: valor(k) = ini + (fin - ini) * (k + 1) / L, k = 0..L-1 (llega a fin en el último paso del tramo)
  cosine: valor(k) = ini * (min_frac + (1 - min_frac) * 0.5 * (1 + cos(pi * min(1, k / max(1, L)))))
Pasada la corrida (paso >= total) el valor es el final del último tramo.
"""
import math

WARMUP_MIN = 100  # piso de pasos de un warmup definido como fracción


def linear(ini, fin, *, frac=None, pasos=None, nombre="", min_pasos=None):
  return _seg("linear", nombre, ini=ini, fin=fin, frac=frac, pasos=pasos, min_pasos=min_pasos)


def constant(valor, *, frac=None, pasos=None, nombre="constante"):
  return _seg("constant", nombre, ini=valor, fin=valor, frac=frac, pasos=pasos)


def cosine(ini, min_frac, *, frac=None, pasos=None, nombre="cosine"):
  return _seg("cosine", nombre, ini=ini, min_frac=min_frac, frac=frac, pasos=pasos)


def _seg(forma, nombre, **kw):
  d = {"nombre": nombre, "forma": forma}
  d.update({k: v for k, v in kw.items() if v is not None})
  return d


def seg_end(s):
  """Valor del tramo en su último paso."""
  if s["forma"] == "cosine":
    return s["ini"] * s["min_frac"]
  return s["fin"] if s["forma"] == "linear" else s["ini"]


def lengths(segs, total):
  """[(inicio, largo)] por tramo para una corrida de `total` pasos."""
  explicit = []
  for s in segs:
    if s.get("pasos") is not None:
      explicit.append(int(s["pasos"]))
    elif s.get("frac") is not None:
      floor = s.get("min_pasos")
      if floor is None:
        floor = WARMUP_MIN if s.get("nombre") == "warmup" else 0
      explicit.append(max(int(floor), round(s["frac"] * total)))
    else:
      explicit.append(None)  # el resto: lo que dejan los demás tramos
  rest = max(0, total - sum(L for L in explicit if L is not None))
  out, start = [], 0
  for L in explicit:
    L = rest if L is None else L
    L = max(0, min(L, total - start))
    out.append((start, L))
    start += L
  return out


def locate(segs, step, total):
  """(índice del tramo, k dentro del tramo, largo) del paso dado; pasada la corrida, el último tramo no vacío."""
  spans = lengths(segs, total)
  last = None
  for i, (st, L) in enumerate(spans):
    if L == 0:
      continue
    last = i
    if step < st + L:
      return i, step - st, L
  if last is None:
    return len(segs) - 1, 0, 0
  st, L = spans[last]
  return last, L - 1, L


def _shape(s, k, L):
  if s["forma"] == "constant":
    return s["ini"]
  if s["forma"] == "linear":
    return s["ini"] + (s["fin"] - s["ini"]) * min(k + 1, L) / max(1, L)
  m = s["min_frac"]
  prog = k / max(1, L)
  return s["ini"] * (m + (1 - m) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))


def value(segs, step, total):
  """Valor del schedule en `step` (0-indexado) de una corrida de `total` pasos."""
  i, k, L = locate(segs, step, total)
  s = segs[i]
  if L == 0:
    return seg_end(s)
  if step >= total:  # pasada la corrida: el final del último tramo (cosine con prog saturado)
    return _shape(s, L, L) if s["forma"] == "cosine" else _shape(s, L - 1, L)
  return _shape(s, k, L)


def segment_name(segs, step, total):
  i = locate(segs, step, total)[0]
  return segs[i].get("nombre") or segs[i]["forma"]


def is_decay(s):
  return s["forma"] == "cosine" or (s["forma"] == "linear" and s["fin"] < s["ini"])


def decay_start(segs, total):
  """Primer paso del último tramo si es de decaimiento (lineal descendente o coseno); None si no hay."""
  if not segs or not is_decay(segs[-1]):
    return None
  spans = lengths(segs, total)
  st, L = spans[-1]
  return st if L > 0 else None


def has_constant_before_decay(segs):
  """warmup -> constante -> decay: la forma que admite continuar con más presupuesto sin reiniciar."""
  return len(segs) >= 2 and is_decay(segs[-1]) and segs[-2]["forma"] == "constant"


def peak(segs):
  vals = []
  for s in segs:
    vals += [s["ini"]] + ([s["fin"]] if s["forma"] == "linear" else [])
  return max(vals) if vals else 0.0
