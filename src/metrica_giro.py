"""Métrica de giro: ¿el video generado gira (yaw) para el mismo lado y tanto como el auto real?

Definición (documentada en docs/metrica_giro.md):
  u_t      flow horizontal medio (RAFT-small) entre el frame t-1 y el t, en la franja superior de F filas (F = 64 de 128:
           lejos del capó, poca paralaje; ahí el giro de la cámara domina el flow horizontal)
  s        sentido del giro real del clip = signo(sum_t u_t real) en los 60 frames del rollout
  giro_t   = s * u_t                           (positivo = se mueve como el giro real; el parpadeo da 0 en promedio)
  giro[a,b) = media_{clips, t en [a,b)} giro_t(modelo) / media_{clips, t en [a,b)} giro_t(real)   (cociente de medias)
Solo en curvas (|yaw acumulado en los 3 s| >= 20° por la pose). IC 95 %: bootstrap por clip (percentiles), remuestreando
los mismos clips en numerador y denominador.
Además: lado correcto = fracción de curvas con signo(sum u modelo) == s.

  python -m src.metrica_giro --names gpt2m_motion gpt_s_kd_motion ...      tabla (desde las cachés de src.metrics_flow)
  python -m src.metrica_giro --sens --names ...                            sensibilidad a la franja (32/64/128 filas)
                                                                           y al umbral de curva (10/20/30°)
"""
import argparse
import json
from pathlib import Path

import numpy as np

CACHE = Path("results/cache")
CLIPS = Path("results/clips_motion.json")
TRAMOS = {"f1": (0, 1), "0-1s": (0, 20), "1-2s": (20, 40), "2-3s": (40, 60)}
OUT = Path("results/motion/giro.json")


def sentido(rot_real):
  """(N, T) flow horizontal real -> (N, 1) signo del giro acumulado."""
  return np.sign(rot_real.sum(1, keepdims=True))


def giro(rot_m, rot_r, a=40, b=60, boot=0, seed=0):
  """Giro alineado relativo al real en los frames [a, b). Con boot > 0 devuelve (valor, lo, hi)."""
  s = sentido(rot_r)
  m, r = (rot_m * s)[:, a:b].mean(1), (rot_r * s)[:, a:b].mean(1)
  v = float(m.mean() / r.mean())
  if not boot:
    return v
  rng = np.random.default_rng(seed)
  idx = rng.integers(0, len(m), (boot, len(m)))
  bs = m[idx].mean(1) / r[idx].mean(1)
  return v, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def lado_correcto(rot_m, rot_r):
  return float((np.sign(rot_m.sum(1, keepdims=True)) == sentido(rot_r)).mean())


def resumen(rot_m, rot_r, boot=1000):
  d = {k: dict(zip(("valor", "lo", "hi"), giro(rot_m, rot_r, a, b, boot=boot))) for k, (a, b) in TRAMOS.items()}
  d["lado_correcto"] = lado_correcto(rot_m, rot_r)
  return d


def clips_info():
  d = json.loads(CLIPS.read_text())
  return d["clips"]


def cargar(name, campo="rot", h=None):
  h = h or _hash()
  pref = "flow_strips_" if campo != "rot" else "flow_"
  return np.load(CACHE / f"{pref}{name}_{h}.npz")[campo]


def _hash():
  from src.rollouts import clips_hash, clip_set
  return clips_hash(clip_set("motion")[0])


def curvas(umbral=20.0):
  return np.array([c["tipo"] == "curva" and abs(c["yaw_3s"]) >= umbral for c in clips_info()])


def tabla(names, boot=1000):
  cur = curvas()
  real = cargar("real")[cur]
  out = {n: resumen(cargar(n)[cur], real, boot) for n in names}
  for n, d in out.items():
    print(f"{n:28s} " + "  ".join(f"{k} {d[k]['valor']:+.2f} [{d[k]['lo']:+.2f},{d[k]['hi']:+.2f}]" for k in TRAMOS)
          + f"  lado {d['lado_correcto']:.2f}")
  return out


# ------------------------------------------------------------------ sensibilidad a la franja

FRANJAS = (32, 64, 128)


def strips(name, dev):
  """Recalcula el flow horizontal medio con franjas de 32/64/128 filas (caché flow_strips_<name>_<hash>.npz)."""
  import torch
  from src.metrics_flow import Flujo, decodificar
  from src.rollouts import OUT as ROLL, clip_set, clip_tokens
  from src.viz import load_decoder
  h = _hash()
  p = CACHE / f"flow_strips_{name}_{h}.npz"
  if p.exists():
    return
  _, clips, tokens = clip_set("motion")
  ctx, fut = clip_tokens(tokens, clips)
  x = fut if name == "real" else np.load(ROLL / f"{name}.npy")
  fl, dec = Flujo(dev), load_decoder(dev)
  res = {f"rot{F}": np.zeros((len(x), 60), np.float32) for F in FRANJAS}
  full = np.concatenate([ctx[:, -1:], x], 1).astype(np.int64)
  for i in range(len(x)):
    fr = decodificar(dec, full[i], dev)
    a, b = fr[:-1], fr[1:]                                         # 60 pares, igual que Flujo
    with torch.no_grad():
      t = lambda z: torch.as_tensor(np.asarray(z), device=dev).permute(0, 3, 1, 2).float() / 127.5 - 1  # noqa: E731
      u = torch.cat([fl.net(t(a[j:j + 32]), t(b[j:j + 32]), num_flow_updates=12)[-1][:, 0]
                     for j in range(0, 60, 32)])
    for F in FRANJAS:
      res[f"rot{F}"][i] = u[:, :F].mean((1, 2)).cpu().numpy()
  np.savez(p, **res)


def _suave(name):
  """flow horizontal (N, 60) del set de 10-20° (clips_motion_suave.json), o None si no está."""
  from src.rollouts import clips_hash, CLIPS_MOTION_SUAVE
  if not CLIPS_MOTION_SUAVE.exists():
    return None, None
  h = clips_hash(CLIPS_MOTION_SUAVE)
  n = name.replace("_motion", "_suave")
  p = CACHE / f"flow_{'real' if name == 'real' else n}_{h}.npz"
  y = np.array([c["yaw_3s"] for c in json.loads(CLIPS_MOTION_SUAVE.read_text())["clips"]])
  return (np.load(p)["rot"] if p.exists() else None), y


def sensibilidad(names):
  """Ranking por giro sostenido (2-3 s) con la franja de 32/64/128 filas (curvas >= 20°) y con umbrales de curva de
  10° (curvas del set principal + las 50 de 10-20°), 20° y 30°."""
  cur20 = curvas(20)
  out = {"franja": {}, "umbral": {}}
  for F in FRANJAS:
    real = cargar("real", f"rot{F}")
    out["franja"][F] = {n: giro(cargar(n, f"rot{F}")[cur20], real[cur20]) for n in names}
  real = cargar("real")
  rs, ys = _suave("real")
  for u in (10.0, 20.0, 30.0):
    c = curvas(u)
    d = {"n": int(c.sum())}
    for n in names:
      m, r = cargar(n)[c], real[c]
      if u < 20:
        ms = _suave(n)[0]
        if ms is None or rs is None:
          continue
        sel = np.abs(ys) >= u
        m, r = np.concatenate([m, ms[sel]]), np.concatenate([r, rs[sel]])
        d["n"] = len(r)
      d[n] = giro(m, r)
    out["umbral"][u] = d
  for k, d in out.items():
    for p, v in d.items():
      orden = sorted([n for n in names if n in v], key=lambda n: -v[n])
      print(f"{k} {p}: " + " > ".join(f"{n} {v[n]:+.2f}" for n in orden) + (f"  (n={v['n']})" if "n" in v else ""))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--names", nargs="+", required=True)
  ap.add_argument("--sens", action="store_true")
  ap.add_argument("--boot", type=int, default=1000)
  a = ap.parse_args()
  OUT.parent.mkdir(parents=True, exist_ok=True)
  prev = json.loads(OUT.read_text()) if OUT.exists() else {}
  if a.sens:
    from src.common import device
    dev = device()
    for n in ["real"] + a.names:
      strips(n, dev)
    prev["sensibilidad"] = sensibilidad(a.names)
  else:
    prev.setdefault("modelos", {}).update(tabla(a.names, a.boot))
  OUT.write_text(json.dumps(prev, indent=1))


if __name__ == "__main__":
  main()
