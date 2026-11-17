"""LPIPS y FVD de los rollouts contra los tokens reales decodificados (plan 2.4).

  python -m src.metrics_video --name NAME [--rollouts results/rollouts/NAME.npy] [--bs 32] [--boot 1000]
                              [--boot-fvd 200] [--fvd-model i3d|videomae]
Escribe results/video_<name>.json.

- Referencia: los tokens reales decodificados con el mismo decoder (aísla al world model).
- LPIPS (alex) pareado frame a frame, 60 horizontes; IC 95% por bootstrap sobre clips.
- FVD: ventanas de 16 frames que terminan a 1, 2 y 3 s (frames [4:20], [24:40], [44:60] del rollout).
  Features = logits de I3D (400-d) como cd-fvd. Bootstrap pareado sobre clips (mismos índices en real y fake).
- Cache (results/cache, ~2.4 GB): frames reales decodificados real_frames_<hash>.npy y sus features
  I3D fvd_real_<modelo>_<hash>.npz; la primera corrida los crea para los 400 clips. Los rollouts pueden ser
  un prefijo de los clips (--limit).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import torch

from src.common import FPS, device, device_info
from src.rollouts import HORIZON, OUT, VAL, clip_tokens, clips_hash, load_clips
from src.viz import decode, load_decoder

RESULTS = Path("results")
CACHE = RESULTS / "cache"
AT = {"0.25": 4, "0.5": 9, "1": 19, "2": 39, "3": 59}              # horizonte (s) -> índice de frame
WINDOWS = {"1": (4, 20), "2": (24, 40), "3": (44, 60)}              # horizonte (s) -> ventana de 16 frames


def frechet(mu1, s1, mu2, s2):
  """Distancia de Fréchet entre dos gaussianas, tr sqrt(S1 S2) vía autovalores (igual a sqrtm, más estable)."""
  w, v = np.linalg.eigh(s1)
  r = (v * np.sqrt(np.clip(w, 0, None))) @ v.T               # S1^(1/2)
  ev = np.linalg.eigvalsh(r @ s2 @ r)                         # autovalores de S1^(1/2) S2 S1^(1/2)
  tr = np.sqrt(np.clip(ev, 0, None)).sum()
  d = np.square(mu1 - mu2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr
  return float(max(d, 0.0))                                   # recorta negativos numéricos


def frechet_sqrtm(mu1, s1, mu2, s2):
  """Versión de cd-fvd con scipy.linalg.sqrtm (para contrastar en los tests)."""
  s = scipy.linalg.sqrtm(s1 @ s2)
  return float(np.real(np.square(mu1 - mu2).sum() + np.trace(s1 + s2 - 2 * s)))


def fvd_from_feats(a, b):
  """a, b (N, D) features -> FVD."""
  return frechet(a.mean(0), np.cov(a, rowvar=False), b.mean(0), np.cov(b, rowvar=False))


def boot_ci(stat, n, reps, seed=0, point=None):
  """stat(idx) -> valor; IC 95% con idx remuestreado con reposición.

  Sin point: intervalo percentil (sirve para medias, como LPIPS). Con point: intervalo normal
  point ± 1,96 · desvío del bootstrap. Es el que usamos para FVD: remuestrear repite clips, baja la
  diversidad y sube el FVD, así que la distribución bootstrap queda corrida hacia arriba (a veces
  entera por encima del valor medido) y ni el intervalo percentil ni el "básico" son confiables. El
  desvío sí mide la variabilidad por muestreo, que es lo que importa al comparar modelos sobre los
  mismos clips.
  """
  rng = np.random.default_rng(seed)
  v = np.array([stat(rng.integers(0, n, n)) for _ in range(reps)])
  if point is None:
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
  sd = float(v.std(ddof=1))
  return max(point - 1.96 * sd, 0.0), point + 1.96 * sd


def load_fvd_model(kind, dev):
  if kind == "i3d":
    from cdfvd.third_party.i3d.utils import load_i3d_model, preprocess_i3d
    m = load_i3d_model(dev).eval()
    return lambda x: m(preprocess_i3d(x).to(dev))                 # logits (B, 400)
  from cdfvd.third_party.VideoMAEv2.utils import load_videomae_model, preprocess_videomae
  m = load_videomae_model(dev).eval().half()                      # ViT-g: fp16 para que quepa
  return lambda x: m.forward_features(preprocess_videomae(x).to(dev).half()).float()


@torch.no_grad()
def feats(fn, vids, bs=8):
  """vids (N, 16, H, W, 3) uint8 -> (N, D) float64."""
  return np.concatenate([fn(vids[i:i + bs]).float().cpu().numpy() for i in range(0, len(vids), bs)]).astype(np.float64)


def window_feats(fn, frames, bs=8):
  """frames (B, 60, H, W, 3) uint8 -> {horizonte: (B, D)}."""
  return {k: feats(fn, frames[:, a:b], bs) for k, (a, b) in WINDOWS.items()}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--name", required=True)
  ap.add_argument("--rollouts")
  ap.add_argument("--bs", type=int, default=32)
  ap.add_argument("--boot", type=int, default=1000)
  ap.add_argument("--boot-fvd", type=int, default=200)
  ap.add_argument("--fvd-model", default="i3d", choices=["i3d", "videomae"])
  ap.add_argument("--from-cache", action="store_true",
                  help="recalcula las estadísticas desde results/cache (LPIPS por clip y features I3D), sin decodificar")
  a = ap.parse_args()
  if a.from_cache:
    return stats_from_cache(a)

  import lpips
  t0 = time.time()
  dev = device()
  h = clips_hash()
  roll = np.load(a.rollouts or OUT / f"{a.name}.npy")
  n = len(roll)
  val = np.load(VAL, mmap_mode="r")

  dec = load_decoder(dev)
  lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
  fn = load_fvd_model(a.fvd_model, dev)
  # referencia: frames reales decodificados (memmap en disco) y sus features I3D, para los 400 clips (se crean una vez)
  allc = load_clips()
  _, real_all = clip_tokens(val, allc)
  N = len(real_all)
  tag = f"{a.fvd_model}_{h}"
  fr_path, cache = CACHE / f"real_frames_{h}.npy", CACHE / f"fvd_real_{tag}.npz"
  CACHE.mkdir(parents=True, exist_ok=True)

  def dec_frames(tok):                                            # (B, 60, 128) -> (B, 60, 128, 256, 3) uint8
    return decode(dec, tok.reshape(-1, 128), bs=16).reshape(len(tok), HORIZON, 128, 256, 3)

  if not (fr_path.exists() and fr_path.with_suffix(".ok").exists()):
    mm = np.lib.format.open_memmap(fr_path, mode="w+", dtype=np.uint8, shape=(N, HORIZON, 128, 256, 3))
    for i in range(0, N, a.bs):
      mm[i:i + a.bs] = dec_frames(real_all[i:i + a.bs])
      print(f"  decodificando referencia {min(i + a.bs, N)}/{N}  {time.time() - t0:.0f} s", flush=True)
    mm.flush()
    del mm
    fr_path.with_suffix(".ok").write_text("ok")
    if cache.exists():
      cache.unlink()
  frames_real = np.load(fr_path, mmap_mode="r")
  if cache.exists():
    z = np.load(cache)
    real_feats = {k: z[k] for k in WINDOWS}
  else:
    rf = {k: [] for k in WINDOWS}
    for i in range(0, N, a.bs):
      for k, v in window_feats(fn, np.asarray(frames_real[i:i + a.bs])).items():
        rf[k].append(v)
      print(f"  features reales {min(i + a.bs, N)}/{N}  {time.time() - t0:.0f} s", flush=True)
    real_feats = {k: np.concatenate(v) for k, v in rf.items()}
    np.savez(cache, **real_feats)
  real_feats = {k: v[:n] for k, v in real_feats.items()}
  ff = {k: [] for k in WINDOWS}
  lp_all = []

  def to_t(x):                                                    # uint8 NHWC -> float NCHW en [-1, 1]
    return torch.as_tensor(x, device=dev).permute(0, 3, 1, 2).float() / 127.5 - 1

  for i in range(0, n, a.bs):
    fk = dec_frames(roll[i:i + a.bs])
    rl = np.asarray(frames_real[i:i + a.bs])
    B = len(fk)
    with torch.no_grad():
      d = [lp(to_t(fk.reshape(-1, 128, 256, 3)[j:j + 120]), to_t(rl.reshape(-1, 128, 256, 3)[j:j + 120])).flatten().cpu()
           for j in range(0, B * HORIZON, 120)]
    lp_all.append(torch.cat(d).reshape(B, HORIZON).numpy())
    for k, v in window_feats(fn, fk).items():
      ff[k].append(v)
    print(f"  clips {i + B}/{n}  {time.time() - t0:.0f} s", flush=True)

  lpm = np.concatenate(lp_all)                                    # (N, 60)
  ff = {k: np.concatenate(v) for k, v in ff.items()}
  # por clip, para comparaciones pareadas entre modelos (mismos clips): LPIPS (N, 60) y features I3D
  np.save(CACHE / f"lpips_{a.name}.npy", lpm.astype(np.float32))
  np.savez(CACHE / f"fvd_fake_{a.fvd_model}_{a.name}.npz", **ff)
  write_stats(a, lpm, ff, real_feats, h, dev, t0)


def write_stats(a, lpm, ff, real_feats, h, dev, t0):
  n = len(lpm)
  lpips_res = {"curve": lpm.mean(0).tolist(), "at": {}}
  for k, f in AT.items():
    col = lpm[:, f]
    lo, hi = boot_ci(lambda idx: col[idx].mean(), n, a.boot)
    lpips_res["at"][k] = {"mean": float(col.mean()), "lo": lo, "hi": hi}
  fvd = {}
  for k in WINDOWS:
    r, f = real_feats[k][:n], ff[k]
    val_k = fvd_from_feats(r, f)
    lo, hi = boot_ci(lambda idx: fvd_from_feats(r[idx], f[idx]), n, a.boot_fvd, seed=1, point=val_k)
    fvd[k] = {"value": val_k, "lo": lo, "hi": hi}
  meta = OUT / f"{a.name}.json"
  res = {"name": a.name, "n_clips": n, "lpips": lpips_res, "fvd": fvd, "fvd_model": a.fvd_model,
         "fvd_ci": "normal: valor ± 1,96 · desvío bootstrap", "clips_hash": h,
         "rollouts_meta": json.loads(meta.read_text()) if meta.exists() else None,
         "device": device_info(dev), "seconds": time.time() - t0}
  (RESULTS / f"video_{a.name}.json").write_text(json.dumps(res, indent=1))
  print("LPIPS:", {k: round(v["mean"], 4) for k, v in lpips_res["at"].items()})
  print("FVD  :", {k: round(v["value"], 2) for k, v in fvd.items()}, f"({res['seconds']:.0f} s)")


def stats_from_cache(a):
  t0, h = time.time(), clips_hash()
  lpm = np.load(CACHE / f"lpips_{a.name}.npy").astype(np.float64)
  z = np.load(CACHE / f"fvd_fake_{a.fvd_model}_{a.name}.npz")
  zr = np.load(CACHE / f"fvd_real_{a.fvd_model}_{h}.npz")
  write_stats(a, lpm, {k: z[k] for k in WINDOWS}, {k: zr[k] for k in WINDOWS}, h, torch.device("cpu"), t0)


if __name__ == "__main__":
  main()
