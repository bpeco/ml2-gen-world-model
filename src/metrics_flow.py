"""Movimiento en píxeles con optical flow (RAFT-small) sobre los rollouts decodificados, real contra modelo.

  python -m src.metrics_flow --name NOMBRE [--name OTRO ...] [--clips motion]
  python -m src.metrics_flow --piso-codec [--clips motion]       # piso: cada frame real contra su re-codificación

Escribe results/flow/<nombre>.json (y la caché por clip en results/cache/flow_<nombre>_<hash>.npz). El real se calcula
una vez por set de clips (flow_real_<hash>.npz).

Por cada par de frames consecutivos (el frame 0 contra el último del contexto), RAFT-small (torchvision) sobre los
frames decodificados a 128x256, y por frame:
  mag  : |flow| medio (píxeles por frame)
  rot  : flow horizontal medio en la mitad de arriba de la imagen (lejos, poca paralaje): proxy del giro
  exp  : expansión media (flow proyectado en la dirección radial desde el centro): proxy del avance
  tok  : fracción de tokens que cambian respecto del frame anterior (la de src.metrics_motion)
Resumen por segundo de horizonte (0-1, 1-2, 2-3 s), por tipo de clip (recta / curva) y diferencias contra el real:
  - mag, rot y exp contra el real: cociente de medias y error absoluto medio por frame.
  - curvas: giro acumulado en 3 s (suma de rot) por clip; acierto de signo contra el real (¿dobla para el mismo lado?)
    y correlación con el yaw de las poses (yaw_3s de results/clips_motion.json).
  - error de dirección: ángulo entre el vector de flow medio (rot, flow vertical medio) del modelo y el del real, por frame.
Piso del codec (--piso-codec): x = dec(tok), x' = dec(enc(x)). La fracción de tokens distintos y el |flow(x, x')| son el
"movimiento" que aparece sin que nada se mueva: parpadeo del codec.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.common import device
from src.infer.codec import ensure_weights, load_encoder
from src.viz import decode, load_decoder
from src.rollouts import HORIZON, OUT as ROLL, clip_set, clip_tokens, clips_hash

RESULTS = Path("results")
OUTD = RESULTS / "flow"
CACHE = RESULTS / "cache"
TRAMOS = {"0-1s": (0, 20), "1-2s": (20, 40), "2-3s": (40, 60)}
CAMPOS = ("mag", "rot", "exp", "vy", "tok")


class Flujo:
  def __init__(self, dev):
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
    w = Raft_Small_Weights.DEFAULT
    self.net = raft_small(weights=w).eval().to(dev)
    self.dev = dev
    H, W = 128, 256
    yy, xx = torch.meshgrid(torch.arange(H, device=dev) - (H - 1) / 2, torch.arange(W, device=dev) - (W - 1) / 2,
                            indexing="ij")
    r = torch.sqrt(xx ** 2 + yy ** 2).clamp(min=1)
    self.ux, self.uy = xx / r, yy / r

  @torch.no_grad()
  def __call__(self, a, b, bs=32):
    """a, b uint8 (N, 128, 256, 3) -> dict de (N,) por par: mag, rot, exp, vy."""
    out = {k: [] for k in ("mag", "rot", "exp", "vy")}
    for i in range(0, len(a), bs):
      t = lambda x: torch.as_tensor(np.asarray(x[i:i + bs]), device=self.dev).permute(0, 3, 1, 2).float() / 127.5 - 1
      f = self.net(t(a), t(b), num_flow_updates=12)[-1]           # (B, 2, H, W): u (x), v (y)
      u, v = f[:, 0], f[:, 1]
      out["mag"].append(torch.sqrt(u ** 2 + v ** 2).mean((1, 2)))
      out["rot"].append(u[:, :64].mean((1, 2)))
      out["vy"].append(v[:, :64].mean((1, 2)))
      out["exp"].append((u * self.ux + v * self.uy).mean((1, 2)))
    return {k: torch.cat(x).cpu().numpy() for k, x in out.items()}


def decodificar(dec, tok, dev, bs=32):
  """tok (N, 128) -> uint8 (N, 128, 256, 3)."""
  return decode(dec, np.asarray(tok).reshape(-1, 128), bs=bs)


def medir_tokens(fl, dec, ctx_last, x, dev):
  """ctx_last (N, 128), x (N, 60, 128) -> dict de (N, 60) por campo."""
  N = len(x)
  res = {k: np.zeros((N, HORIZON), np.float32) for k in CAMPOS}
  full = np.concatenate([ctx_last[:, None], x], 1).astype(np.int64)
  res["tok"] = (full[:, 1:] != full[:, :-1]).mean(-1).astype(np.float32)
  for i in range(N):
    fr = decodificar(dec, full[i], dev)                           # (61, 128, 256, 3)
    m = fl(fr[:-1], fr[1:])
    for k, v in m.items():
      res[k][i] = v
    if (i + 1) % 20 == 0:
      print(f"  flow {i + 1}/{N}", flush=True)
  return res


def cache_path(name, h):
  return CACHE / f"flow_{name}_{h}.npz"


def resumen(m, r, clips):
  """m, r: dicts (N, 60) del modelo y del real; clips: lista de dicts del set."""
  tipo = np.array([c.get("tipo", "todos") for c in clips])
  yaw = np.array([c.get("yaw_3s", np.nan) for c in clips], dtype=np.float64)
  out = {}
  for grupo in ["todos"] + sorted(set(tipo) - {"todos"}):
    sel = np.ones(len(tipo), bool) if grupo == "todos" else tipo == grupo
    g = {}
    for k in CAMPOS:
      mk, rk = m[k][sel], r[k][sel]
      d = {"curva_modelo": mk.mean(0).round(4).tolist(), "curva_real": rk.mean(0).round(4).tolist()}
      for t, (a, b) in TRAMOS.items():
        d[t] = {"modelo": float(np.abs(mk[:, a:b]).mean() if k == "rot" else mk[:, a:b].mean()),
                "real": float(np.abs(rk[:, a:b]).mean() if k == "rot" else rk[:, a:b].mean()),
                "mae": float(np.abs(mk[:, a:b] - rk[:, a:b]).mean())}
        d[t]["ratio"] = d[t]["modelo"] / d[t]["real"] if d[t]["real"] else None
      g[k] = d
    # dirección: ángulo entre (rot, vy) medios del modelo y del real, por frame, donde el real se mueve algo
    am, ar = np.arctan2(m["vy"][sel], m["rot"][sel]), np.arctan2(r["vy"][sel], r["rot"][sel])
    ang = np.degrees(np.abs((am - ar + np.pi) % (2 * np.pi) - np.pi))
    mov = np.hypot(r["rot"][sel], r["vy"][sel]) > 0.05
    g["error_direccion_deg"] = {t: float(ang[:, a:b][mov[:, a:b]].mean()) if mov[:, a:b].any() else None
                                for t, (a, b) in TRAMOS.items()}
    # giro acumulado en 3 s por clip
    gm, gr = m["rot"][sel].sum(1), r["rot"][sel].sum(1)
    g["giro_3s"] = {"modelo_abs_medio": float(np.abs(gm).mean()), "real_abs_medio": float(np.abs(gr).mean()),
                    "acierto_signo": float((np.sign(gm) == np.sign(gr)).mean()),
                    "corr_con_real": float(np.corrcoef(gm, gr)[0, 1]) if sel.sum() > 2 else None}
    y = yaw[sel]
    if np.isfinite(y).all() and sel.sum() > 2:
      g["giro_3s"]["corr_con_yaw_pose"] = float(np.corrcoef(gm, y)[0, 1])
      g["giro_3s"]["corr_real_con_yaw_pose"] = float(np.corrcoef(gr, y)[0, 1])
    g["n"] = int(sel.sum())
    out[grupo] = g
  return out


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--name", action="append", default=[])
  ap.add_argument("--clips", default="motion", choices=["eval", "motion", "motion_suave"])
  ap.add_argument("--piso-codec", action="store_true")
  a = ap.parse_args(argv)
  dev = device()
  ensure_weights()
  dec = load_decoder(dev)
  fl = Flujo(dev)
  cp, clips, toks = clip_set(a.clips)
  h = clips_hash(cp)
  ctx, real = clip_tokens(toks, clips)
  ctx_last = ctx[:, -1]
  CACHE.mkdir(parents=True, exist_ok=True)
  OUTD.mkdir(parents=True, exist_ok=True)
  t0 = time.time()

  rp = cache_path("real", h)
  if rp.exists():
    r = dict(np.load(rp))
  else:
    r = medir_tokens(fl, dec, ctx_last, real, dev)
    np.savez(rp, **r)
    print(f"real: {time.time() - t0:.0f} s", flush=True)

  if a.piso_codec:
    enc = load_encoder(dev)
    pc = {k: np.zeros((len(real), HORIZON), np.float32) for k in CAMPOS}
    for i in range(len(real)):
      x = decodificar(dec, real[i].astype(np.int64), dev)
      with torch.no_grad():
        t2 = torch.cat([enc(torch.as_tensor(x[j:j + 30], device=dev).permute(0, 3, 1, 2).float()).view(-1, 128)
                        for j in range(0, len(x), 30)]).cpu().numpy()
      x2 = decodificar(dec, t2, dev)
      m = fl(x, x2)
      for k, v in m.items():
        pc[k][i] = v
      pc["tok"][i] = (t2 != real[i]).mean(-1)
    np.savez(cache_path("piso_codec", h), **pc)
    res = {"name": "piso_codec", "clips": a.clips, "clips_hash": h,
           "metodo": "x = dec(tok real), x' = dec(enc(x)): tokens distintos y RAFT(x, x') sin movimiento",
           "resumen": {k: {"media": float(v.mean()), "p90": float(np.percentile(v, 90))} for k, v in pc.items()},
           "real_consecutivo": {k: float(np.abs(r[k]).mean() if k == "rot" else r[k].mean()) for k in CAMPOS}}
    (OUTD / "piso_codec.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))

  # alineación de las poses: correlación entre rot real por frame y el yaw rate, con desfasajes
  if a.clips == "motion" and not (OUTD / "alineacion_pose.json").exists():
    poses = np.load(clips["poses"], mmap_mode="r")
    cl = clips["clips"]
    res = {}
    for lag in range(-4, 5):
      yr = np.stack([poses[c["segment"], c["start"] + 20 + lag:c["start"] + 80 + lag, 5] for c in cl])
      ok = np.isfinite(yr)
      res[lag] = float(np.corrcoef(r["rot"][ok], yr[ok])[0, 1])
    vel = np.stack([poses[c["segment"], c["start"] + 20:c["start"] + 80, 0] for c in cl])
    res_v = float(np.corrcoef(r["exp"].ravel(), vel.ravel())[0, 1])
    (OUTD / "alineacion_pose.json").write_text(json.dumps(
      {"corr_rot_yaw_por_desfasaje": res, "corr_exp_velocidad": res_v,
       "nota": "lag k: yaw del frame t + k contra el flow horizontal entre t-1 y t (frames del rollout)"}, indent=1))
    print("alineación rot vs yaw:", {k: round(v, 3) for k, v in res.items()}, "exp vs v:", round(res_v, 3))

  for name in a.name:
    x = np.load(ROLL / f"{name}.npy", mmap_mode="r")
    n = len(x)
    p = cache_path(name, h)
    if p.exists():
      m = dict(np.load(p))
    else:
      m = medir_tokens(fl, dec, ctx_last[:n], np.asarray(x), dev)
      np.savez(p, **m)
    rr = {k: v[:n] for k, v in r.items()}
    meta_p = ROLL / f"{name}.json"
    res = {"name": name, "clips": a.clips, "clips_hash": h, "n_clips": n,
           "rollouts_meta": json.loads(meta_p.read_text()) if meta_p.exists() else None,
           "resumen": resumen(m, rr, clips["clips"][:n])}
    (OUTD / f"{name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    g = res["resumen"]["todos"]
    print(f"{name:30s} mag {g['mag']['0-1s']['modelo']:.2f}/{g['mag']['2-3s']['modelo']:.2f} (real "
          f"{g['mag']['0-1s']['real']:.2f}/{g['mag']['2-3s']['real']:.2f})  tok {g['tok']['2-3s']['ratio']:.2f}  "
          f"signo giro (curvas) {res['resumen'].get('curva', g)['giro_3s']['acierto_signo']:.2f}  "
          f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
  main()
