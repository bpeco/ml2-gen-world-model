"""Barrido de temperatura en el holdout de train (D14, E2): elige T por FVD a 3 s, con LPIPS a 3 s de desempate.

  python -m src.temperatura --model RUN [--temperaturas 0.7,0.8,0.9,1.0,1.1] [--limite N] [--steps S]
                            [--bs 50] [--dtype fp32|fp16|bf16] [--seed 0] [--tol 0.02] [--boot 50]
Escribe results/temperatura_<run>.json con la tabla y la `elegida`.

- Clips: los últimos 50 segmentos de train (el holdout de src.windows.Windows) x 4 arranques = 200 clips
  (results/clips_holdout.json, ver src.rollouts --clips holdout). Nunca val: la temperatura es un hiperparámetro.
- El barrido parte del default de cada modelo (cfg.temperature: 2 en genie por su paper, 1 en el resto). Sin
  --temperaturas, la grilla es el default x (0,7; 0,8; 0,9; 1,0; 1,1). Con --temperaturas son valores absolutos
  (el default del modelo se agrega siempre, para tener la referencia en la tabla).
- FVD a 3 s: ventana de 16 frames [44:60] del rollout contra los mismos frames reales decodificados (I3D, igual que
  src.metrics_video); LPIPS a 3 s: frame 59. IC del FVD por bootstrap por segmento (src.estadistica).
- Desempate: son candidatas las T con FVD <= (1 + tol) x el mejor; entre ellas gana la de menor LPIPS (y, a igualdad,
  la más cercana al default).
- No aplica a modelos sin muestreo (convlstm_px es determinístico) ni a los latentes (D12).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src import estadistica
from src.common import device, device_info, log_device
from src.rollouts import (CLIPS_HOLDOUT, DTYPES, HORIZON, clip_tokens, clips_hash, load_clips_holdout, rechazar_latente,
                          temperatura_default)

RESULTS = Path("results")
FACTORES = (0.7, 0.8, 0.9, 1.0, 1.1)
VENTANA = (44, 60)                 # frames del rollout que forman la ventana de 3 s
FRAME_LPIPS = HORIZON - 1          # frame 59 (3 s)


def grilla(default, temperaturas=None):
  """Temperaturas a probar, ordenadas y sin repetir. Sin `temperaturas`: default x FACTORES. El default siempre está."""
  ts = [round(default * f, 6) for f in FACTORES] if not temperaturas else [float(t) for t in temperaturas]
  ts.append(float(default))
  return sorted({round(t, 6) for t in ts})


def elegir(tabla, default, tol=0.02):
  """tabla: filas con "T", "fvd" y "lpips". Gana la T de menor FVD; las que quedan a menos de tol (relativo) de ese
  mínimo se desempatan por LPIPS, y luego por cercanía al default. Devuelve la fila elegida."""
  mejor = min(r["fvd"] for r in tabla)
  cand = [r for r in tabla if r["fvd"] <= mejor * (1 + tol)]
  return min(cand, key=lambda r: (r["lpips"], abs(r["T"] - default)))


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True, help="spec de load_model (corrida, ruta .pt, random:...)")
  ap.add_argument("--temperaturas", help="valores absolutos separados por coma; por defecto default x 0,7..1,1")
  ap.add_argument("--limite", type=int, help="usar solo los primeros N clips del holdout")
  ap.add_argument("--steps", type=int)
  ap.add_argument("--bs", type=int, default=50)
  ap.add_argument("--dtype", default="fp32", choices=list(DTYPES))
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--tol", type=float, default=0.02, help="tolerancia relativa de FVD para el desempate por LPIPS")
  ap.add_argument("--boot", type=int, default=50, help="remuestreos del IC del FVD (por segmento)")
  ap.add_argument("--out", help="carpeta de salida (por defecto results/)")
  ap.add_argument("--opt", action="append", help="campo=valor de la Config del modelo; repetible")
  a = ap.parse_args(argv)

  dev = device()
  log_device(dev)
  from src.models import apply_opts, load_model
  model = apply_opts(load_model(a.model, dev), a.opt).eval().to(DTYPES[a.dtype])
  rechazar_latente(model, "src.temperatura")
  if getattr(model, "output", "tokens") == "pixels":
    raise SystemExit(f"src.temperatura: {model.name} es determinístico (no muestrea): no tiene temperatura que barrer")
  t0 = temperatura_default(model)
  temps = grilla(t0, [x for x in a.temperaturas.split(",") if x.strip()] if a.temperaturas else None)

  clips, toks = load_clips_holdout()
  n = min(a.limite or len(clips["clips"]), len(clips["clips"]))
  clips = {**clips, "clips": clips["clips"][:n]}
  ctx_all, real_all = clip_tokens(toks, clips)
  seg = estadistica.segmentos_de(clips)

  import lpips
  from src.metrics_video import fvd_from_feats, feats, load_fvd_model
  from src.viz import decode, load_decoder
  dec = load_decoder(dev)
  lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
  fn = load_fvd_model("i3d", dev)
  v0, v1 = VENTANA

  def ventana(tok):                                     # (B, 60, 128) -> (B, 16, 128, 256, 3) uint8
    return decode(dec, tok[:, v0:v1].reshape(-1, 128), bs=16).reshape(len(tok), v1 - v0, 128, 256, 3)

  def to_t(x):
    return torch.as_tensor(x, device=dev).permute(0, 3, 1, 2).float() / 127.5 - 1

  real_f, real_last = [], []
  for i in range(0, n, a.bs):
    w = ventana(real_all[i:i + a.bs])
    real_f.append(feats(fn, w))
    real_last.append(w[:, -1])
  real_f, real_last = np.concatenate(real_f), np.concatenate(real_last)

  tabla, t_ini = [], time.time()
  for T in temps:
    ff, lpc = [], []
    t1 = time.time()
    for i in range(0, n, a.bs):
      ctx = torch.as_tensor(ctx_all[i:i + a.bs].astype(np.int64), device=dev)
      g = torch.Generator(device=dev).manual_seed(a.seed * 100003 + i)
      with torch.no_grad():
        r = model.rollout(ctx, HORIZON, temperature=T, steps=a.steps, generator=g).cpu().numpy().astype(np.int16)
      w = ventana(r)
      ff.append(feats(fn, w))
      with torch.no_grad():
        lpc.append(lp(to_t(w[:, -1]), to_t(real_last[i:i + a.bs])).flatten().cpu().numpy())
    ff, lpc = np.concatenate(ff), np.concatenate(lpc).astype(np.float64)
    f = estadistica.ic_fvd_segmento(real_f, ff, seg, a.boot, fvd_fn=fvd_from_feats)
    l = estadistica.ic_media_segmento(lpc, seg, 200)
    tabla.append({"T": T, "fvd": f["value"], "fvd_lo": f["lo"], "fvd_hi": f["hi"], "lpips": l["mean"],
                  "lpips_lo": l["lo"], "lpips_hi": l["hi"], "seconds": time.time() - t1})
    print(f"  T={T:g}: FVD@3s {f['value']:.1f} [{f['lo']:.1f}, {f['hi']:.1f}]  LPIPS@3s {l['mean']:.4f}  "
          f"({time.time() - t1:.0f} s)", flush=True)

  el = elegir(tabla, t0, a.tol)
  res = {"model": a.model, "name": model.name, "default": t0, "temperaturas": temps, "elegida": el["T"],
         "tabla": tabla, "criterio": f"menor FVD a 3 s; empate (FVD <= {1 + a.tol:g} x el mejor) por menor LPIPS a 3 s, luego cercanía al default",
         "tol": a.tol, "ventana": list(VENTANA), "frame_lpips": FRAME_LPIPS, "fvd_ci": "normal, bootstrap por segmento",
         "clips": "holdout", "n_clips": n, "n_segmentos": int(len(np.unique(seg))), "clips_hash": clips_hash(CLIPS_HOLDOUT),
         "steps": a.steps, "dtype": a.dtype, "seed": a.seed, "device": device_info(dev), "seconds": time.time() - t_ini}
  out = Path(a.out) if a.out else RESULTS
  out.mkdir(parents=True, exist_ok=True)
  path = out / f"temperatura_{model.name}.json"
  path.write_text(json.dumps(res, indent=1), encoding="utf-8")
  print(f"{path}: elegida T={el['T']:g} (default {t0:g})")
  return res


if __name__ == "__main__":
  main()
