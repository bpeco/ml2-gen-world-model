"""Visor del set de movimiento (results/clips_motion.json) para el probador: http://localhost:8765/movimiento

Muestra cómo se armó el set (criterio, dispersión giro/velocidad de los 200 clips) y, para cada clip, el video real
(20 frames de contexto + 60) al lado de los rollouts ya generados de los modelos (results/rollouts/<nombre>_motion.npy),
con el yaw rate de la pose y el flow horizontal por frame (cachés de src.metrics_flow) del real y de cada modelo.
No genera nada: solo decodifica tokens que ya están en disco.
"""
import json
from pathlib import Path

import numpy as np

from src.rollouts import CLIPS_MOTION, clips_hash

R = Path("results")
ROLL = R / "rollouts"
CACHE = R / "cache"
FPS = 20

# nombres legibles de los rollouts más importantes (el resto se lista con su nombre de archivo)
ETIQUETAS = {
  "gpt2m_motion": "GPT2M (maestro)",
  "gpt_s_kd_motion": "GPT-S kd (paper)",
  "gpt_s_d050_motion": "GPT-S d050 (paper)",
  "kdB_s0_motion": "kdB_s0 (finalista)",
  "kd4x512_B_motion": "GPT 4×512 + ajuste (finalista)",
  "kd3x768_B_motion": "GPT 3×768 + ajuste",
  "kdOP_ctrl_motion": "kd + 500 pasos KD",
  "kdB_1b_motion": "kdB sobre +780 k (A)",
  "kdP_s0_hold_motion": "pose s0 · hold (fase 3)",
  "kdP_s0_real_motion": "pose s0 · real (fase 3)",
  "kdPc_s0_motion": "control fase 3 s0",
  "mgA_pc08_p4_motion": "MaskGIT p_clean 0,8 · 4 pasos",
  "mg_p4_motion": "MaskGIT-S · 4 pasos (paper)",
  "mg_p8_motion": "MaskGIT-S · 8 pasos (paper)",
  "genie_s_p4_motion": "Genie-S · 4 pasos (paper)",
  "genie_m_p4_motion": "Genie-M · 4 pasos (paper)",
}

_set = None


def conjunto():
  global _set
  if _set is None:
    d = json.loads(CLIPS_MOTION.read_text(encoding="utf-8"))
    _set = dict(d=d, tok=np.load(d["tokens"], mmap_mode="r"), pose=np.load(d["poses"], mmap_mode="r"),
                hash=clips_hash(CLIPS_MOTION))
  return _set


def flujo(nombre):
  p = CACHE / f"flow_{nombre}_{conjunto()['hash']}.npz"
  return np.load(p)["rot"] if p.exists() else None


def modelos():
  n = len(conjunto()["d"]["clips"])
  out = []
  for p in sorted(ROLL.glob("*_motion.npy")):
    nombre = p.stem
    try:
      shp = np.load(p, mmap_mode="r").shape
    except Exception:
      continue
    if shp[:1] != (n,) or len(shp) != 3:  # solo rollouts de tokens del set completo
      continue
    out.append({"id": nombre, "nombre": ETIQUETAS.get(nombre, nombre.removesuffix("_motion")),
                "destacado": nombre in ETIQUETAS, "flow": (CACHE / f"flow_{nombre}_{conjunto()['hash']}.npz").exists()})
  orden = list(ETIQUETAS)
  out.sort(key=lambda m: (orden.index(m["id"]) if m["id"] in orden else len(orden), m["id"]))
  return out


def lista():
  s = conjunto()
  d = s["d"]
  giro = {}
  gj = R / "motion/giro.json"
  if gj.exists():
    g = json.loads(gj.read_text(encoding="utf-8")).get("modelos", {})
    giro = {k: {"giro": round(v["2-3s"]["valor"], 3), "lado": v.get("lado_correcto")} for k, v in g.items() if "2-3s" in v}
  return {"criterio": d["criterio"], "fuente": d["source"], "ctx": d["ctx"], "horizon": d["horizon"], "fps": d["fps"],
          "clips": [{k: c[k] for k in ("clip", "segment", "key", "start", "tipo", "yaw_3s", "yaw_ctx", "v_media")}
                    for c in d["clips"]],
          "modelos": modelos(), "giro": giro}


def datos(i, ids, decodificar, jpeg):
  """decodificar: tokens (N, 128) -> uint8 (N, 128, 256, 3); jpeg: imagen -> data URI. Llamar con la GPU tomada."""
  s = conjunto()
  d = s["d"]
  c = d["clips"][i]
  nc, h = d["ctx"], d["horizon"]
  tok = np.asarray(s["tok"][c["segment"], c["start"]:c["start"] + nc + h]).astype(np.int64)
  pose = np.asarray(s["pose"][c["segment"], c["start"]:c["start"] + nc + h], dtype=np.float64)
  validos = {m["id"] for m in modelos()}
  out = {"clip": c, "real": [jpeg(x, 80) for x in decodificar(tok)],
         "yaw_deg_s": np.round(np.degrees(np.nan_to_num(pose[:, 5])), 3).tolist(),
         "v_ms": np.round(np.nan_to_num(pose[:, 0]), 2).tolist(), "modelos": {}, "flow": {}}
  fr = flujo("real")
  if fr is not None:
    out["flow"]["real"] = np.round(fr[i], 3).tolist()
  for m in ids:
    if m not in validos:
      continue
    r = np.load(ROLL / f"{m}.npy", mmap_mode="r")[i].astype(np.int64)
    out["modelos"][m] = [jpeg(x, 80) for x in decodificar(r)]
    f = flujo(m)
    if f is not None:
      out["flow"][m] = np.round(f[i], 3).tolist()
  return out
