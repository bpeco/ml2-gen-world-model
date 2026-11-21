"""Escenario en línea del banco de pruebas (src/infer/probador.py) sobre muchos clips, para la figura del paper.

  python -m src.infer.online_eval [--clips 40] [--dur 10] [--antic 2] [--periodo 1]   # -> results/online.json

Cada modelo corre solo en la GPU (batch 1, fp16, CUDA graphs). Cada `periodo` s observa los frames reales que llegaron
y genera `antic` s hacia adelante (un plan); el plan queda vigente recién cuando termina de calcularse. Guarda, por
modelo y clip, el resumen (cobertura, edad de la predicción en pantalla, margen del frame +antic, LPIPS) y, para el
primer clip, la línea de tiempo de los planes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.infer import probador as P

MODELOS = [  # (id del catálogo, pasos, etiqueta)
  ("gpt_s_kd", None, "GPT-S destilado"),
  ("maskgit_s", 8, "MaskGIT-S, 8 pasos"),
  ("maskgit_s", 4, "MaskGIT-S, 4 pasos"),
  ("genie_s", 4, "Genie-S, 4 pasos"),
]


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--clips", type=int, default=40, help="clips equiespaciados de los 400 de evaluación")
  ap.add_argument("--dur", type=float, default=10)
  ap.add_argument("--antic", type=float, default=2)
  ap.add_argument("--periodo", type=float, default=1)
  ap.add_argument("--semilla", type=int, default=0)
  ap.add_argument("--salida", default="results/online.json")
  a = ap.parse_args(argv)
  n_tot = len(P.load_clips()["clips"])
  idx = [int(i) for i in np.linspace(0, n_tot, a.clips, endpoint=False)]
  out = {"config": {"dur_s": a.dur, "antic_s": a.antic, "periodo_s": a.periodo, "semilla": a.semilla, "clips": idx,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"},
         "modelos": [{"id": m, "pasos": s, "nombre": nom, "clips": []} for m, s, nom in MODELOS]}
  for c_n, i in enumerate(idx):
    pedido = {"clip": i, "semilla": a.semilla, "dur": a.dur, "antic": a.antic, "periodo": a.periodo,
              "modelos": [{"id": m, "pasos": s} for m, s, _ in MODELOS]}
    res = {}
    P.correr_online(pedido, lambda ev, d: res.update({ev: d}))
    for dst, m in zip(out["modelos"], res["online"]["modelos"]):
      fila = {"clip": i, **m["resumen"]}
      if c_n == 0:
        fila["linea"] = [{k: p[k] for k in ("ini", "listo", "n", "ms", "margen", "lpips_ant")} for p in m["planes"]]
        fila["edad"], fila["lpips"], fila["ref"] = m["edad"], m["lpips"], m["ref"]
      dst["clips"].append(fila)
    r = [f"{d['nombre']}: cob {d['clips'][-1]['cobertura']:.0%}" for d in out["modelos"]]
    print(f"[{c_n + 1}/{len(idx)}] clip {i}  " + " | ".join(r), flush=True)
  for d in out["modelos"]:
    cs = d["clips"]
    med = lambda k: float(np.mean([c[k] for c in cs if c[k] is not None])) if any(c[k] is not None for c in cs) else None
    d["resumen"] = {k: med(k) for k in ("cobertura", "edad_media", "edad_max", "margen_medio", "ms_plan", "ms_paso",
                                        "lpips_ahora", "ref_ahora", "lpips_ant", "planes")}
    d["resumen"]["margen_iqr"] = [float(np.percentile([c["margen_medio"] for c in cs], q)) for q in (25, 75)]
    d["resumen"]["cobertura_iqr"] = [float(np.percentile([c["cobertura"] for c in cs], q)) for q in (25, 75)]
  Path(a.salida).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
  for d in out["modelos"]:
    print(d["nombre"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d["resumen"].items()})
  print("->", a.salida)


if __name__ == "__main__":
  main()
