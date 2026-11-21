"""Movimiento en tokens: qué fracción de los 128 tokens cambia entre frames consecutivos, real contra rollout.

  python -m src.metrics_motion --name NOMBRE [--name OTRO ...]     # results/rollouts/<NOMBRE>.npy
  python -m src.metrics_motion --todos                             # todos los rollouts de results/rollouts/

Escribe results/motion/<nombre>.json y, con varios, imprime una tabla.

- cambio[t] = fracción de celdas con token(t) != token(t-1); el frame t = 0 se compara con el último del contexto.
- Se resume por segundo (frames 0-19, 20-39, 40-59) y en total; la referencia son los frames reales de los mismos clips.
- copia_ctx = fracción de celdas del frame 59 (3 s) iguales al último frame de contexto: cuánto "se quedó quieto".
- ratio = cambio del modelo / cambio real a 2-3 s: 1 es el ritmo real, < 1 un video más quieto que el real.

Es una métrica de diagnóstico, no de calidad: parte del cambio real es parpadeo del codec, no movimiento físico, y un
modelo que cambia tokens al azar también da ratio 1. Se lee junto al FVD.
Rollouts con menos clips (los *_ej) se comparan con los primeros N clips del set de eval, que es como se generan.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.rollouts import VAL, clip_tokens, load_clips

ROLL = Path("results/rollouts")
OUT = Path("results/motion")
TRAMOS = {"0-1s": (0, 20), "1-2s": (20, 40), "2-3s": (40, 60)}


def cambios(x, ctx_last):
  """x (N, H, 128), ctx_last (N, 128) -> (N, H) fracción de celdas que cambian respecto del frame anterior."""
  full = np.concatenate([ctx_last[:, None], x], 1)
  return (full[:, 1:] != full[:, :-1]).mean(-1)


def resumen(x, ctx_last):
  d = cambios(x, ctx_last)
  r = {k: float(d[:, a:b].mean()) for k, (a, b) in TRAMOS.items()}
  r["total"] = float(d.mean())
  r["copia_ctx"] = float((x[:, -1] == ctx_last).mean())
  r["curva"] = [round(float(v), 4) for v in d.mean(0)]
  return r


def medir(name, ctx, real):
  x = np.load(ROLL / f"{name}.npy", mmap_mode="r")
  if x.ndim != 3 or x.shape[1:] != real.shape[1:]:
    return None  # no son tokens (convlstm_px guarda píxeles) o es otro horizonte
  n = len(x)
  if n > len(real):
    return None
  meta_p = ROLL / f"{name}.json"
  meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
  if meta.get("clips", "eval") != "eval":
    return None
  m, r = resumen(np.asarray(x), ctx[:n, -1]), resumen(real[:n], ctx[:n, -1])
  res = {"name": name, "n_clips": n, "modelo": m, "real": r,
         "ratio_2_3s": m["2-3s"] / r["2-3s"] if r["2-3s"] else None}
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / f"{name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
  return res


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--name", action="append", default=[])
  ap.add_argument("--todos", action="store_true")
  a = ap.parse_args(argv)
  names = a.name or []
  if a.todos:
    names += sorted(p.stem for p in ROLL.glob("*.npy"))
  if not names:
    ap.error("--name o --todos")
  ctx, real = clip_tokens(np.load(VAL, mmap_mode="r"), load_clips())
  print(f"{'rollout':34s} {'n':>4s} {'0-1s':>6s} {'1-2s':>6s} {'2-3s':>6s} {'real':>6s} {'ratio':>6s} {'copia':>6s}")
  for n in dict.fromkeys(names):
    r = medir(n, ctx, real)
    if r is None:
      continue
    m = r["modelo"]
    print(f"{n:34s} {r['n_clips']:4d} {m['0-1s']:6.3f} {m['1-2s']:6.3f} {m['2-3s']:6.3f} {r['real']['2-3s']:6.3f} "
          f"{r['ratio_2_3s']:6.2f} {m['copia_ctx']:6.3f}")


if __name__ == "__main__":
  main()
