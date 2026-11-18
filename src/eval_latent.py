"""Evaluaciones propias de los world models latentes (lewm), al estilo de LeWorldModel.

Todo sobre el holdout de train separado por ruta (src.windows.disjoint_holdout: 46 segmentos cuya ruta no
aparece en train), porque val.zip no tiene poses. Ventanas de 20 frames cada 60 frames de cada segmento.

  probes      ¿la representación sabe la pose? Probe lineal (ridge) y MLP desde las features del último frame
              de la ventana hasta su pose (velocidad y giro): MSE (poses normalizadas) y Pearson r en test
              (segmentos aparte). Latentes: ẑ (predicho desde el pasado, sin la acción del frame) y z
              (observado). Referencias: GPT-S (estado en el BOS del frame) y MaskGIT-S (frame enmascarado).
  sorpresa    violación de expectativa: error latente del frame 10 con la ventana original vs. cortada en
              el frame 10 hacia otro segmento ("teletransporte") o hacia 2 s más adelante ("salto"). t pareado.
  rectitud    coseno medio entre velocidades latentes consecutivas (z_{t+1} − z_t) de la ventana real.
  acciones    error latente con las poses reales vs. con la acción nula, pareado por ventana.
  plan        CEM (300 muestras, 30 élites, 10 iteraciones, historia 3) de las poses de los próximos H frames
              para llegar al latente del frame t+H; error contra las poses reales vs. repetir la última pose y
              vs. la pose media de train.

  python -m src.eval_latent --model lewm [--name lewm] [--no-plan]
  python -m src.eval_latent --model maskgit_s --probes-only
Salida: results/latent_<nombre>.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, ttest_rel

from src.common import CTX_FRAMES, MASK, TPF, device, device_info, log_device, with_bos
from src.models import load_model
from src.models.gpt import GPT
from src.models.latent import LatentWM
from src.models.maskgit import MaskGIT
from src.windows import disjoint_holdout

TOKENS, POSES, KEYS = "data/train_tokens.npy", "data/train_poses.npy", "data/train_keys.json"
W = CTX_FRAMES          # 20 frames por ventana
STRIDE = 60             # una ventana cada 3 s
CUT = 10                # frame de la perturbación
JUMP = 40               # salto de 2 s
H = 10                  # horizonte de planificación (0,5 s)
DIMS = {"velocidad": 0, "giro": 5}


def windows(segs, T):
  return [(s, t) for s in segs for t in range(0, T - W - JUMP - H, STRIDE)]


def gather(arr, idx, n, offset=0):
  return np.stack([arr[s, t + offset:t + offset + n] for s, t in idx])


@torch.no_grad()
def probe_features(model, x):
  """(B, F, 128) -> {"pred": (B, D)} (+ "obs" en los latentes): features del último frame de la ventana."""
  if isinstance(model, LatentWM):
    z, zh = model.features(x)  # sin acciones: la acción del frame es justo la pose que se quiere leer
    return {"pred": zh[:, -1], "obs": z[:, -1]}
  if isinstance(model, GPT):
    model._drop_cache()
    h = model.net(with_bos(x), hidden=True)
    return {"pred": h[:, (x.shape[1] - 1) * TPF].float()}  # el BOS del último frame solo ve el pasado
  if isinstance(model, MaskGIT):
    model._drop_cache()
    seq = with_bos(x)
    seq[:, -(TPF - 1):] = MASK
    return {"pred": model.net(seq, hidden=True)[:, -(TPF - 1):].float().mean(1)}
  raise SystemExit(f"{type(model).__name__}: sin features para el probe")


def ridge(Xtr, ytr, Xte, alpha=1.0):
  mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
  A, B = (Xtr - mu) / sd, (Xte - mu) / sd
  A1 = np.c_[A, np.ones(len(A))]
  w = np.linalg.solve(A1.T @ A1 + alpha * np.eye(A1.shape[1]), A1.T @ ytr)
  return np.c_[B, np.ones(len(B))] @ w


def mlp_probe(Xtr, ytr, Xte, seed=0, steps=1500):
  torch.manual_seed(seed)
  mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
  A = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
  B = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
  y = torch.tensor(ytr, dtype=torch.float32)
  net = torch.nn.Sequential(torch.nn.Linear(A.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, y.shape[1]))
  opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
  for _ in range(steps):
    loss = F.mse_loss(net(A), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
  with torch.no_grad():
    return net(B).numpy()


def probe_scores(feats, y, train_mask):
  out = {}
  for kind, fn in (("lineal", ridge), ("mlp", mlp_probe)):
    pred = fn(feats[train_mask], y[train_mask], feats[~train_mask])
    yt = y[~train_mask]
    out[kind] = {d: {"mse": float(np.mean((pred[:, i] - yt[:, i]) ** 2)), "pearson": float(pearsonr(pred[:, i], yt[:, i])[0])}
                 for d, i in DIMS.items()}
  return out


def paired_t(a, b):
  a, b = np.asarray(a), np.asarray(b)
  t = ttest_rel(b, a)
  return {"orig": float(a.mean()), "pert": float(b.mean()), "ratio": float(b.mean() / max(a.mean(), 1e-12)),
          "t": float(t.statistic), "p": float(t.pvalue), "n": int(len(a))}


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--name")
  ap.add_argument("--bs", type=int, default=16)
  ap.add_argument("--probes-only", action="store_true", help="solo probes (modelos de tokens: gpt, maskgit)")
  ap.add_argument("--no-plan", action="store_true")
  ap.add_argument("--plan-n", type=int, default=46, help="planes (uno por segmento del holdout)")
  ap.add_argument("--segments", type=int, help="usar solo los primeros N segmentos del holdout (pruebas rápidas)")
  ap.add_argument("--out", default="results")
  a = ap.parse_args(argv)
  dev = device()
  info = log_device(dev)
  torch.manual_seed(0)
  model = load_model(a.model, dev).eval()
  name = a.name or model.name
  tok = np.load(TOKENS, mmap_mode="r")
  pos = np.load(POSES, mmap_mode="r")
  segs = disjoint_holdout(KEYS)[:a.segments]
  idx = windows(segs, tok.shape[1])
  seg_ids = np.array([s for s, _ in idx])
  test_segs = set(segs[::3])  # un tercio de los segmentos para test del probe
  train_mask = np.array([s not in test_segs for s in seg_ids])
  X = torch.from_numpy(gather(tok, idx, W).astype(np.int64))
  P = torch.from_numpy(gather(pos, idx, W).astype(np.float32))
  print(f"{name}: {len(segs)} segmentos, {len(idx)} ventanas", flush=True)
  res = {"model": a.model, "name": name, "segments": len(segs), "windows": len(idx), "device_info": info}
  t0 = time.time()

  # ---- probes
  feats = {}
  for sl in range(0, len(X), a.bs):
    f = probe_features(model, X[sl:sl + a.bs].to(dev))
    for k, v in f.items():
      feats.setdefault(k, []).append(v.cpu().numpy())
  feats = {k: np.concatenate(v) for k, v in feats.items()}
  mean, std = (model.pose_mean.float().cpu().numpy(), model.pose_std.float().cpu().numpy()) if isinstance(model, LatentWM) \
    else (np.nanmean(pos[:-50].reshape(-1, 6), 0), np.nanstd(pos[:-50].reshape(-1, 6), 0))
  y = (P[:, -1].numpy() - mean) / std
  ok = np.isfinite(y).all(1)
  res["probes"] = {k: probe_scores(v[ok], y[ok], train_mask[ok]) for k, v in feats.items()}
  print("probes", json.dumps(res["probes"])[:300], f"({time.time() - t0:.0f}s)", flush=True)
  if a.probes_only or not isinstance(model, LatentWM):
    return save(res, a.out, name)

  # ---- sorpresa, rectitud y acciones
  def errs(frames, poses):
    out = []
    for i in range(0, len(frames), a.bs):
      p = None if poses is None else poses[i:i + a.bs].to(dev)
      out.append(model.latent_error(frames[i:i + a.bs].to(dev), p).cpu())
    return torch.cat(out).numpy()

  other = np.roll(np.arange(len(idx)), len(idx) // 2)  # otra ventana, de otro segmento
  assert (seg_ids != seg_ids[other]).all()
  tele = X.clone()
  tele[:, CUT:] = X[other, CUT:]
  jump = X.clone()
  jump[:, CUT:] = torch.from_numpy(gather(tok, idx, W - CUT, offset=CUT + JUMP).astype(np.int64))
  e_orig, e_tele, e_jump = errs(X, P), errs(tele, P), errs(jump, P)
  res["sorpresa"] = {"teletransporte": paired_t(e_orig[:, CUT], e_tele[:, CUT]),
                     "salto_2s": paired_t(e_orig[:, CUT], e_jump[:, CUT]),
                     "por_frame_orig": e_orig.mean(0).tolist()}
  e_null = errs(X, None)
  res["acciones"] = {"error_pose_real": float(e_orig[:, 1:].mean()), "error_accion_nula": float(e_null[:, 1:].mean()),
                     "pareado": paired_t(e_orig[:, 1:].mean(1), e_null[:, 1:].mean(1)),
                     "per_window_real": e_orig[:, 1:].mean(1).tolist(), "per_window_nula": e_null[:, 1:].mean(1).tolist()}
  cos = []
  with torch.no_grad():
    for i in range(0, len(X), a.bs):
      z = model.encode(X[i:i + a.bs].to(dev)).float().flatten(2)
      v = z[:, 1:] - z[:, :-1]
      cos.append(F.cosine_similarity(v[:, 1:], v[:, :-1], dim=-1).mean(1).cpu())
  res["rectitud"] = float(torch.cat(cos).mean())
  print("sorpresa", {k: round(v["ratio"], 3) for k, v in res["sorpresa"].items() if isinstance(v, dict)},
        "acciones", round(res["acciones"]["error_pose_real"], 4), round(res["acciones"]["error_accion_nula"], 4),
        "rectitud", round(res["rectitud"], 3), f"({time.time() - t0:.0f}s)", flush=True)

  # ---- planificación con CEM
  if not a.no_plan:
    plan_idx = [next(i for i, (s, _) in enumerate(idx) if s == seg) for seg in segs][:a.plan_n]
    real, planned, ms = [], [], []
    g = torch.Generator(device=dev).manual_seed(0)
    for i in plan_idx:
      s, t = idx[i]
      ctx = torch.from_numpy(tok[s, t:t + W].astype(np.int64))[None].to(dev)
      goal = torch.from_numpy(tok[s, t + W + H - 1].astype(np.int64))[None].to(dev)
      cp = torch.from_numpy(pos[s, t:t + W].astype(np.float32))[None]
      if dev.type == "cuda":
        torch.cuda.synchronize()
      t1 = time.perf_counter()
      pl = model.plan_cem(ctx, goal, H, ctx_poses=cp, history=3, generator=g, n_samples=300, n_elites=30, iters=10)
      if dev.type == "cuda":
        torch.cuda.synchronize()
      ms.append((time.perf_counter() - t1) * 1000)
      planned.append(pl[0].cpu().numpy())
      real.append(np.asarray(pos[s, t + W:t + W + H], dtype=np.float32))
    planned, real = np.stack(planned), np.stack(real)
    last = np.stack([np.repeat(np.asarray(pos[idx[i][0], idx[i][1] + W - 1])[None], H, 0) for i in plan_idx])
    mean_pose = np.broadcast_to(model.pose_mean.float().cpu().numpy(), real.shape)

    def mae(p):
      ok = np.isfinite(real).all(-1)
      return {d: float(np.abs(p[..., i] - real[..., i])[ok].mean()) for d, i in DIMS.items()}

    res["plan"] = {"horizonte": H, "n": len(plan_idx), "ms_por_plan": float(np.median(ms)), "cem": mae(planned),
                   "ultima_pose": mae(last), "pose_media": mae(mean_pose)}
    print("plan", json.dumps(res["plan"]), f"({time.time() - t0:.0f}s)", flush=True)
  return save(res, a.out, name)


def save(res, out, name):
  Path(out).mkdir(parents=True, exist_ok=True)
  p = Path(out) / f"latent_{name}.json"
  p.write_text(json.dumps(res, indent=1))
  print(p)
  return res


if __name__ == "__main__":
  main()
