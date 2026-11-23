"""Set de clips de movimiento: 100 rectas y 100 curvas del val con poses (data/valp_*, rutas fuera de train y del val oficial).

  python -m src.clips_motion            escribe results/clips_motion.json (determinista, semilla 0)

Cada clip son 20 frames de contexto + 60 de rollout, como clips_eval.json, uno por segmento y sin repetir rutas.
Poses (data/valp_poses.npy, (S, 1200, 6)): col 0 = velocidad de avance (m/s), col 5 = yaw rate (rad/s); el último frame
de cada segmento no tiene pose. yaw_3s = giro acumulado en los 60 frames del rollout, en grados (sum(yaw_rate) / 20 FPS).
  curva : |yaw_3s| >= CURVA_MIN y velocidad media >= V_MIN; segmentos y arranques al azar, 50 a cada lado.
  recta : |yaw_3s| < RECTA_MAX, |giro en el contexto| < RECTA_MAX y velocidad media >= V_MIN; a cada curva se le asigna
          la recta de velocidad más parecida (pares por velocidad: el |flow| depende mucho de la velocidad).
El set lo usan src.rollouts / src.metrics_video / src.metrics_flow con --clips motion.
"""
import json
from pathlib import Path

import numpy as np

from src.common import CTX_FRAMES, FPS

PATH = Path("results/clips_motion.json")
TOKENS = "data/valp_tokens.npy"
POSES = "data/valp_poses.npy"
KEYS = "data/valp_keys.json"
HORIZON = 60
N_LADO = 100
CURVA_MIN, RECTA_MAX, V_MIN = 20.0, 1.5, 8.0
PASO = 10


def candidatos(poses):
  """(S, n_starts) yaw_3s, yaw_ctx (grados) y velocidad media, para arranques cada PASO frames."""
  T = poses.shape[1] - 1                                   # el último frame no tiene pose
  starts = np.arange(0, T - CTX_FRAMES - HORIZON + 1, PASO)
  yaw = np.degrees(np.nan_to_num(poses[..., 5])) / FPS     # grados por frame
  cs = np.concatenate([np.zeros((len(poses), 1)), np.cumsum(yaw, 1)], 1)
  cv = np.concatenate([np.zeros((len(poses), 1)), np.cumsum(np.nan_to_num(poses[..., 0]), 1)], 1)
  a, b, c = starts, starts + CTX_FRAMES, starts + CTX_FRAMES + HORIZON
  return starts, cs[:, c] - cs[:, b], cs[:, b] - cs[:, a], (cv[:, c] - cv[:, b]) / HORIZON


def make(path=PATH, seed=0):
  rng = np.random.default_rng(seed)
  poses = np.load(POSES, mmap_mode="r")
  keys = json.loads(Path(KEYS).read_text())
  route = lambda k: Path(k).name.rsplit("_", 1)[0]
  starts, y3, yc, v = candidatos(np.asarray(poses))
  ok_v = v >= V_MIN
  # curvas: segmentos al azar con algún arranque de |yaw_3s| >= CURVA_MIN (un arranque al azar entre esos), 50 por lado
  curva_ok = ok_v & (np.abs(y3) >= CURVA_MIN)
  usadas, curvas = set(), {+1: [], -1: []}
  for seg in rng.permutation(len(poses)):
    js = np.flatnonzero(curva_ok[seg])
    if len(js) == 0 or route(keys[seg]) in usadas:
      continue
    j = int(rng.choice(js))
    lado = int(np.sign(y3[seg, j]))
    if len(curvas[lado]) >= N_LADO // 2:
      continue
    curvas[lado].append((int(seg), j))
    usadas.add(route(keys[seg]))
    if all(len(x) == N_LADO // 2 for x in curvas.values()):
      break
  assert all(len(x) == N_LADO // 2 for x in curvas.values()), {k: len(x) for k, x in curvas.items()}
  # rectas: un arranque recto al azar por segmento; a cada curva se le asigna la recta de velocidad más parecida
  # (sin repetir), para que |flow| de rectas y curvas sea comparable
  rect_ok = ok_v & (np.abs(y3) < RECTA_MAX) & (np.abs(yc) < RECTA_MAX)
  pool = []
  for seg in range(len(poses)):
    js = np.flatnonzero(rect_ok[seg])
    if len(js) and route(keys[seg]) not in usadas:
      j = int(rng.choice(js))
      pool.append((float(v[seg, j]), seg, j))
  rectas = []
  for seg_c, j_c in curvas[+1] + curvas[-1]:
    k = min(range(len(pool)), key=lambda i: abs(pool[i][0] - v[seg_c, j_c]) if route(keys[pool[i][1]]) not in usadas else 1e9)
    _, seg, j = pool.pop(k)
    rectas.append((seg, j))
    usadas.add(route(keys[seg]))
  clips = []
  for tipo, lista in (("recta", rectas), ("curva", curvas[+1] + curvas[-1])):
    for s, j in lista:
      clips.append({"clip": len(clips), "segment": int(s), "key": keys[s], "start": int(starts[j]), "tipo": tipo,
                    "yaw_3s": round(float(y3[s, j]), 2), "yaw_ctx": round(float(yc[s, j]), 2),
                    "v_media": round(float(v[s, j]), 2)})
  d = {"ctx": CTX_FRAMES, "horizon": HORIZON, "fps": FPS, "source": "valp (data-0001, sin rutas de train ni del val oficial)",
       "tokens": TOKENS, "poses": POSES, "seed": seed,
       "criterio": {"curva_min_deg": CURVA_MIN, "recta_max_deg": RECTA_MAX, "v_min_ms": V_MIN, "paso": PASO,
                    "yaw": "col 5 (rad/s) integrada en los 60 frames del rollout"},
       "clips": clips}
  Path(path).write_text(json.dumps(d, indent=1))
  return d


if __name__ == "__main__":
  d = make()
  c = d["clips"]
  for t in ("recta", "curva"):
    x = [k for k in c if k["tipo"] == t]
    y = np.array([k["yaw_3s"] for k in x])
    print(f"{t}: {len(x)} clips, |yaw_3s| mediana {np.median(np.abs(y)):.1f}° (min {np.abs(y).min():.1f}, max "
          f"{np.abs(y).max():.1f}), izq/der {int((y > 0).sum())}/{int((y < 0).sum())}, v media "
          f"{np.median([k['v_media'] for k in x]):.1f} m/s")
  print(f"{PATH}: {len(c)} clips, {len({k['key'].rsplit('_', 1)[0] for k in c})} rutas")
