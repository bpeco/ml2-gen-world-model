"""Estadística de la evaluación de video (D14): bootstrap por segmento, piso de FVD, diferencias pareadas y semillas.

Los 400 clips de evaluación son 100 segmentos x 4 arranques (results/clips_eval.json). Los 4 clips de un segmento
comparten ruta, luz y escena, así que no son independientes: se remuestrean segmentos enteros, no clips. Todas las
funciones son puras (no leen disco) salvo las del final, que arman los JSON a partir de results/ y results/cache.

  python -m src.estadistica semillas --nombre gpt_s --runs gpt_s,gpt_s_s1,gpt_s_s2   -> results/semillas_<nombre>.json
  python -m src.estadistica piso [--b real_eval_b] [--n N]                         -> results/piso_fvd.json
  python -m src.estadistica diferencia --a gpt_s --b maskgit_s_s4 [--n N]           -> results/diferencia_<a>_vs_<b>.json

Piso de FVD comparable (n contra n): FVD(real de clips_eval, real de clips_eval_b) con el mismo número de clips por
lado que el modelo (400), con IC por segmento (`piso_completo`). clips_eval_b son los mismos 100 segmentos de val con otros
4 arranques sin solape (val no tiene más segmentos), así que el piso es algo optimista respecto de segmentos distintos.
Se crea con: rollouts --model real --clips eval_b --name real_eval_b; metrics_video --name real_eval_b --clips eval_b.

Piso con mitades (secundario, cuando no hay eval_b): real contra real con el mismo número de clips. Los clips reales que se evalúan son los mismos que
sirven de referencia, así que no hay otra muestra real independiente: el piso se estima con dos mitades disjuntas por
segmento (cada mitad con la mitad de los segmentos, sorteadas `reps` veces) y es el FVD de comparar n/2 clips contra
n/2 clips. Es un piso de muestra chica: el FVD baja al crecer la muestra, así que NO es comparable con un FVD de
n clips contra n. Para compararlo con un modelo en igualdad de condiciones, piso_fvd(..., fake=...) calcula también
el FVD del modelo con las mismas mitades ("modelo_mitades": FVD(real[A], fake[A]) y FVD(real[B], fake[B])).
"""
import argparse
import json
from pathlib import Path

import numpy as np

RESULTS = Path("results")
CACHE = RESULTS / "cache"
HORIZONTES = ("1", "2", "3")
Z95 = 1.96


# ---- bootstrap por segmento
def segmentos_de(clips):
  """Lista de clips (dicts con "segment") o dict de load_clips -> arreglo (N,) con el segmento de cada clip."""
  c = clips["clips"] if isinstance(clips, dict) else clips
  return np.array([x["segment"] for x in c], dtype=np.int64)


def grupos_de(seg):
  """Índices de clips de cada segmento: lista de arreglos, uno por segmento distinto."""
  seg = np.asarray(seg)
  return [np.flatnonzero(seg == s) for s in np.unique(seg)]


def remuestrear(grupos, rng):
  """Un remuestreo por segmento: sortea len(grupos) segmentos con reposición y junta todos sus clips."""
  g = rng.integers(0, len(grupos), len(grupos))
  return np.concatenate([grupos[i] for i in g])


def ic_segmento(stat, seg, reps=1000, seed=0, punto=None):
  """IC 95% de stat(idx) con bootstrap por segmento. idx son índices de clips.

  Sin punto: intervalo percentil (sirve para medias, como LPIPS). Con punto: normal, punto ± 1,96 · desvío del
  bootstrap (el de FVD: remuestrear repite clips, baja la diversidad y corre la distribución hacia arriba, así que
  el percentil no es confiable; el desvío sí mide la variabilidad por muestreo).
  Devuelve {"lo", "hi", "sd", "reps", "metodo"}.
  """
  grupos = grupos_de(seg)
  rng = np.random.default_rng(seed)
  v = np.array([stat(remuestrear(grupos, rng)) for _ in range(reps)], dtype=np.float64)
  sd = float(v.std(ddof=1))
  if punto is None:
    lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    metodo = "percentil, bootstrap por segmento"
  else:
    lo, hi = max(punto - Z95 * sd, 0.0), punto + Z95 * sd
    metodo = "normal (valor ± 1,96 · desvío), bootstrap por segmento"
  return {"lo": lo, "hi": hi, "sd": sd, "reps": reps, "metodo": metodo}


def ic_media_segmento(x, seg, reps=1000, seed=0):
  """Media de una métrica por clip x (N,) con su IC percentil por segmento."""
  x = np.asarray(x, dtype=np.float64)
  r = ic_segmento(lambda i: x[i].mean(), seg, reps, seed)
  return {"mean": float(x.mean()), **r}


def _fvd(fvd_fn):
  if fvd_fn is not None:
    return fvd_fn
  from src.metrics_video import fvd_from_feats  # import tardío: metrics_video importa este módulo
  return fvd_from_feats


def ic_fvd_segmento(real, fake, seg, reps=200, seed=1, fvd_fn=None):
  """FVD(real, fake) con su IC normal por segmento. real, fake (N, D) sobre los mismos clips."""
  f = _fvd(fvd_fn)
  v = f(real, fake)
  return {"value": float(v), **ic_segmento(lambda i: f(real[i], fake[i]), seg, reps, seed, punto=v)}


# ---- piso de FVD
def piso_fvd(real, seg, reps=20, seed=0, fvd_fn=None, fake=None):
  """Piso de FVD: real contra real con dos mitades disjuntas por segmento (ver el docstring del módulo).

  real (N, D) features de los clips reales; seg (N,) segmento de cada clip. Cada repetición sortea la mitad de los
  segmentos para A y otro tanto distinto para B (con un número impar de segmentos sobra uno). fake (N, D), si se da,
  son las features del modelo sobre los mismos clips: se agrega "modelo_mitades" (FVD con las mismas mitades).
  Devuelve {"value" (media de las repeticiones), "sd", "min", "max", "reps", "n_a", "n_b", "metodo"}.
  """
  f = _fvd(fvd_fn)
  grupos = grupos_de(seg)
  g = len(grupos)
  if g < 2:
    raise ValueError("el piso necesita al menos 2 segmentos")
  rng = np.random.default_rng(seed)
  h = g // 2
  vals, na, nb, mod = [], [], [], []
  for _ in range(reps):
    p = rng.permutation(g)
    a = np.concatenate([grupos[i] for i in p[:h]])
    b = np.concatenate([grupos[i] for i in p[h:2 * h]])
    vals.append(f(real[a], real[b]))
    if fake is not None:
      mod.append((f(real[a], fake[a]) + f(real[b], fake[b])) / 2)
    na.append(len(a))
    nb.append(len(b))
  v = np.array(vals)
  res = {"value": float(v.mean()), "sd": float(v.std(ddof=1)) if reps > 1 else 0.0, "min": float(v.min()),
         "max": float(v.max()), "reps": reps, "n_a": float(np.mean(na)), "n_b": float(np.mean(nb)),
         "metodo": "real contra real, dos mitades disjuntas por segmento (n/2 clips por lado), media de las repeticiones"}
  if fake is not None:
    res["modelo_mitades"] = float(np.mean(mod))
  return res


def piso_completo(real_a, real_b, seg, reps=200, seed=1, fvd_fn=None):
  """FVD(real A, real B) a n contra n, con IC normal por segmento. A y B son clips reales distintos de los mismos
  segmentos y en el mismo orden (clip i de A y de B comparten segmento)."""
  assert len(real_a) == len(real_b) == len(seg), "los dos sets reales deben tener los mismos n y segmentos"
  r = ic_fvd_segmento(real_a, real_b, seg, reps, seed, fvd_fn)
  r["n_a"] = r["n_b"] = len(real_a)
  r["metodo"] = "real (clips_eval) contra real (clips_eval_b), n contra n; " + r["metodo"]
  return r


# ---- diferencia pareada entre dos modelos
def diferencia_pareada(real, fake_a, fake_b, seg, reps=200, seed=2, fvd_fn=None):
  """FVD(real, A) - FVD(real, B) sobre los mismos clips, con bootstrap por segmento pareado.

  Cada remuestreo usa los mismos clips para A, B y la referencia. IC normal alrededor de la diferencia medida; p
  de dos colas: el doble de la fracción de remuestreos que cae del otro lado de cero (con corrección +1).
  Una diferencia negativa significa que A es mejor (menor FVD).
  """
  f = _fvd(fvd_fn)
  va, vb = f(real, fake_a), f(real, fake_b)
  d0 = float(va - vb)
  grupos = grupos_de(seg)
  rng = np.random.default_rng(seed)
  d = np.empty(reps)
  for k in range(reps):
    i = remuestrear(grupos, rng)
    d[k] = f(real[i], fake_a[i]) - f(real[i], fake_b[i])
  sd = float(d.std(ddof=1))
  cola = min(int((d >= 0).sum()), int((d <= 0).sum()))
  p = min(1.0, 2 * (cola + 1) / (reps + 1))
  return {"fvd_a": float(va), "fvd_b": float(vb), "diff": d0, "lo": d0 - Z95 * sd, "hi": d0 + Z95 * sd, "sd": sd,
          "p": float(p), "reps": reps, "metodo": "bootstrap pareado por segmento; IC normal; p de dos colas por bootstrap"}


def diferencia_media_pareada(x, y, seg, reps=1000, seed=3):
  """Diferencia pareada de la media de una métrica por clip (p. ej. LPIPS): x - y, IC percentil y p por segmento."""
  x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
  dif = x - y
  r = ic_segmento(lambda i: dif[i].mean(), seg, reps, seed)
  # p por bootstrap: fracción de remuestreos del otro lado de cero
  grupos = grupos_de(seg)
  rng = np.random.default_rng(seed)
  d = np.array([dif[remuestrear(grupos, rng)].mean() for _ in range(reps)])
  cola = min(int((d >= 0).sum()), int((d <= 0).sum()))
  return {"diff": float(dif.mean()), "lo": r["lo"], "hi": r["hi"], "p": float(min(1.0, 2 * (cola + 1) / (reps + 1))),
          "reps": reps, "metodo": "bootstrap pareado por segmento; IC percentil"}


# ---- semillas
def resumen_semillas(valores):
  """Lista de números (uno por corrida) -> {"mean", "std" (ddof 1), "n", "values"}. Sin None; con 1 valor, std = None."""
  v = np.array([x for x in valores if x is not None], dtype=np.float64)
  if len(v) == 0:
    return {"mean": None, "std": None, "n": 0, "values": []}
  return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else None, "n": int(len(v)), "values": v.tolist()}


def _get(d, *keys):
  for k in keys:
    if not isinstance(d, dict) or k not in d:
      return None
    d = d[k]
  return d


def agregar_semillas(nombre, runs, results=RESULTS):
  """Media ± desvío entre corridas de un mismo modelo (<run>, <run>_s1, <run>_s2).

  Lee results/video_<run>.json (FVD y LPIPS por horizonte) y results/eval_<run>.json (CE). Cada corrida debe tener
  al menos uno de los dos; las métricas que le falten a alguna se agregan con las que sí la tienen (queda `n`).
  Los video_*.json deben ser del mismo set de clips (clips_hash).
  """
  results = Path(results)
  videos, evals, faltan = {}, {}, []
  for r in runs:
    pv, pe = results / f"video_{r}.json", results / f"eval_{r}.json"
    if pv.exists():
      videos[r] = json.loads(pv.read_text(encoding="utf-8"))
    if pe.exists():
      evals[r] = json.loads(pe.read_text(encoding="utf-8"))
    if r not in videos and r not in evals:
      faltan.append(r)
  if faltan:
    raise FileNotFoundError(f"sin results/video_<run>.json ni eval_<run>.json para: {', '.join(faltan)}")
  hashes = {v.get("clips_hash") for v in videos.values()}
  if len(hashes) > 1:
    raise ValueError(f"los video_*.json son de sets de clips distintos: {sorted(map(str, hashes))}")
  out = {"nombre": nombre, "runs": list(runs), "n_corridas": len(runs), "clips_hash": next(iter(hashes), None),
         "video": {"fvd": {}, "lpips": {}}, "eval": {},
         "nota": "media ± desvío (ddof 1) entre corridas con distinta semilla de entrenamiento; n = corridas con la métrica"}
  for h in HORIZONTES:
    out["video"]["fvd"][h] = resumen_semillas([_get(v, "fvd", h, "value") for v in videos.values()])
  for h in ("0.25", "0.5", "1", "2", "3"):
    out["video"]["lpips"][h] = resumen_semillas([_get(v, "lpips", "at", h, "mean") for v in videos.values()])
  for k in ("ce_all", "ce_tok", "ce_last", "acc_last"):
    out["eval"][k] = resumen_semillas([_get(e, k) for e in evals.values()])
  return out


# ---- comandos
def _cargar_feats(name, n=None, results=RESULTS):
  """(real, fake, seg) desde results/cache. n recorta a los primeros n clips (rollouts con --limit)."""
  from src.rollouts import clips_hash, load_clips
  cache = Path(results) / "cache"
  h = clips_hash()
  zr = np.load(cache / f"fvd_real_i3d_{h}.npz")
  seg = segmentos_de(load_clips())
  fake = None
  if name is not None:
    z = np.load(cache / f"fvd_fake_i3d_{name}.npz")
    fake = {k: z[k] for k in HORIZONTES}
  m = n or (min(len(v) for v in fake.values()) if fake else len(seg))
  return {k: zr[k][:m] for k in HORIZONTES}, (None if fake is None else {k: v[:m] for k, v in fake.items()}), seg[:m]


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  sub = ap.add_subparsers(dest="cmd", required=True)
  s = sub.add_parser("semillas", help="media ± desvío entre corridas")
  s.add_argument("--nombre", required=True)
  s.add_argument("--runs", required=True, help="nombres separados por coma: gpt_s,gpt_s_s1,gpt_s_s2")
  s.add_argument("--results", default=str(RESULTS))
  p = sub.add_parser("piso", help="piso de FVD desde results/cache: n contra n con eval_b y, aparte, mitades por segmento")
  p.add_argument("--b", default="real_eval_b", help="rollout 'real' sobre clips_eval_b (ver el docstring); si falta su caché, solo mitades")
  p.add_argument("--boot", type=int, default=200)
  p.add_argument("--n", type=int, help="clips (prefijo); por defecto todos")
  p.add_argument("--reps", type=int, default=20)
  d = sub.add_parser("diferencia", help="bootstrap pareado de la diferencia de FVD entre dos modelos")
  d.add_argument("--a", required=True)
  d.add_argument("--b", required=True)
  d.add_argument("--n", type=int)
  d.add_argument("--reps", type=int, default=200)
  a = ap.parse_args(argv)

  if a.cmd == "semillas":
    res = agregar_semillas(a.nombre, [r.strip() for r in a.runs.split(",") if r.strip()], a.results)
    path = Path(a.results) / f"semillas_{a.nombre}.json"
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    f1 = res["video"]["fvd"]["1"]
    print(f"{path}: {res['n_corridas']} corridas; FVD@1s = "
          + (f"{f1['mean']:.1f} ± {f1['std']:.1f}" if f1["std"] is not None else str(f1["mean"])))
    return res
  if a.cmd == "piso":
    real, _, seg = _cargar_feats(None, a.n)
    res = {"n_clips": len(seg), "mitades": {h: piso_fvd(real[h], seg, a.reps) for h in HORIZONTES}, "piso": None, "b": None}
    zb = CACHE / f"fvd_fake_i3d_{a.b}.npz"
    if zb.exists():
      zb = np.load(zb)
      res["b"] = a.b
      res["piso"] = {h: piso_completo(real[h], zb[h][:len(seg)], seg, a.boot, fvd_fn=None) for h in HORIZONTES}
    else:
      print(f"sin {zb}: solo el piso con mitades (generar eval_b: ver el docstring)")
    path = RESULTS / "piso_fvd.json"
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(path, {h: round(v["value"], 1) for h, v in (res["piso"] or res["mitades"]).items()})
    return res
  real, fake_a, seg = _cargar_feats(a.a, a.n)
  _, fake_b, _ = _cargar_feats(a.b, len(seg))
  res = {"a": a.a, "b": a.b, "n_clips": len(seg),
         "fvd": {h: diferencia_pareada(real[h], fake_a[h], fake_b[h], seg, a.reps) for h in HORIZONTES}}
  path = RESULTS / f"diferencia_{a.a}_vs_{a.b}.json"
  path.write_text(json.dumps(res, indent=1), encoding="utf-8")
  print(path, {h: (round(v["diff"], 1), round(v["p"], 3)) for h, v in res["fvd"].items()})
  return res


if __name__ == "__main__":
  main()
