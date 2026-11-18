"""Evaluación de los world models latentes (`lewm`, `vjepa2`, `vjepa2_ac`): la misma lectura para todos.

  python -m src.eval_latent --model <run | ruta.pt | random:kind> [--name N] [--limite N] [--no-plan] [--out results]
Escribe results/latent_<name>.json.

Datos (D13). Los probes se entrenan en segmentos de train (data/train_*.npy) y se evalúan en el val con poses
(data/valp_*.npy: 2212 segmentos sin ninguna ruta de train ni del val oficial). Los hiperparámetros (el α del ridge) se
eligen en el holdout de `src.windows.Windows` (sus últimos 50 segmentos, sin las rutas que están en train), nunca en val.
Los píxeles salen de src/pixels.py (tokens -> imagen con el decoder de comma). Una ventana son 16 pasos a 4 Hz (1 de cada
5 frames crudos, 3,75 s) con las 5 poses de cada paso apiladas (6·5 = 30 por paso), como las ve LeWM y V-JEPA 2-AC.

Qué se mide, cada cosa con su referencia al lado y con IC95 por bootstrap sobre segmentos (no ventanas):
  probes      ¿el latente sabe la pose (velocidad, giro) del último frame? Probe lineal (ridge) sobre las features de los
              últimos 3 pasos. Referencias: el mismo encoder sin entrenar (misma arquitectura, init aleatoria con semilla
              fija), píxeles crudos reducidos a 8x16 y la media de train. R² y MSE (poses normalizadas con train).
  horizonte   (lewm, vjepa2_ac) error del rollout latente a h = 1..8 pasos (0,25 .. 2 s) desde 3 frames reales con las
              acciones reales; referencias: copiar el último latente y la acción nula.
  sorpresa    (lewm, vjepa2_ac) error del primer paso tras cortar la ventana hacia otro segmento ("teletransporte") o
              2 s más adelante ("salto") contra el de la ventana original; test pareado a nivel segmento. Referencia: el
              mismo cociente con "copiar el último latente" (¿alcanza la distancia entre latentes para ver el corte?).
  cem         (lewm, vjepa2_ac) recuperación de las acciones con el planificador de cada paper, con el desvío de muestreo
              sacado del desvío de los datos de train; referencias: repetir la última acción y la acción media de train.
V-JEPA 2 (sin acciones ni predicción del futuro) solo tiene probes: lo demás sale con "aplica": false y el motivo.

Lo que viene de cada paper y lo que es propio está en los bloques "fuente" y "propio" del JSON. Cada modelo trae su propia
lectura (`lat_*` en lewm.py, vjepa/pretrain.py y vjepa/action.py): este archivo solo compara, mide y resume.
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import t as t_student

from src import pixels
from src.common import device, device_info, log_device
from src.data import route_of
from src.models import build, load_model
from src.models.lewm import LeWM
from src.models.vjepa.action import VJEPA2AC
from src.models.vjepa.pretrain import VJEPA2
from src.windows import disjoint_holdout

TRAIN = ("data/train_tokens.npy", "data/train_poses.npy", "data/train_keys.json")
VAL = ("data/valp_tokens.npy", "data/valp_poses.npy", "data/valp_keys.json")
HOLDOUT = 50            # segmentos de holdout de Windows
SKIP = 5                # frames crudos por paso (20 fps -> 4 Hz)
PASOS = 16              # pasos por ventana (3,75 s)
SPAN = PASOS * SKIP     # frames crudos que cubre una ventana, con las poses del último paso
SALTO = 8               # el "salto" de 2 s son 8 pasos
CORTE = 8               # primer paso perturbado en la sorpresa
N_PASOS = 3             # pasos que resumen las features de los probes
CTX = 3                 # frames reales de contexto en el rollout y en el CEM
H_MAX = 8               # horizonte del rollout (2 s)
DIMS = {"velocidad": 0, "giro": 5}
ALFAS = [10.0 ** e for e in range(-2, 6)]
KINDS = {LeWM: "lewm", VJEPA2: "vjepa2", VJEPA2AC: "vjepa2_ac"}
NA = {"aplica": False}

FUENTE = {
  "lewm": ["CEM: 300 muestras, 30 élites, horizonte 5, 10 iteraciones, N(0, 1) sobre las acciones normalizadas, costo = "
           "‖ẑ_final − z_meta‖² (LeWM arXiv 2603.19312 App. D; github.com/lucas-maes/le-wm config/eval/solver/cem.yaml)",
           "acción = las 5 poses de cada paso apiladas, normalizadas con train (le-wm train.py, module.py Embedder)"],
  "vjepa2": ["encoder evaluado = target_encoder (EMA) con layer_norm (V-JEPA 2 arXiv 2506.09985; vjepa2 configs/eval, "
             "checkpoint_key target_encoder)"],
  "vjepa2_ac": ["CEM: 800 muestras, top-10, 10 iteraciones, energía L1 entre la representación imaginada y la del objetivo, "
                "gaussiana inicial de media 0 (V-JEPA 2 §3.2 y App.; vjepa2 notebooks/utils/mpc_utils.py)",
                "estado = pose del frame, acción = diferencia de estados entre frames (vjepa2 app/vjepa_droid/train.py L390-394)",
                "error latente: L1 media sobre tokens y canales, la loss de entrenamiento (train.py L439-449); el MSE se "
                "reporta además"],
}
PROPIO = [
  dict(item="ventanas", detalle=f"{PASOS} pasos a 4 Hz con las {SKIP} poses de cada paso apiladas; 2 ventanas por segmento "
       "en posiciones aleatorias (semilla fija), dejando lugar para el salto de 2 s"),
  dict(item="probe", detalle="ridge lineal sobre features estandarizadas; α entre 1e-2 y 1e5 elegido en el holdout de Windows "
       "(segmentos con ruta ausente de train) y reajustado con todo train; sin MLP"),
  dict(item="features", detalle=f"las de los últimos {N_PASOS} pasos del modelo concatenadas (pooling por modelo en 'pooling'); "
       "'pred' = lo que el predictor da para esos pasos desde su historia, sin acciones (no se le da la pose que se lee)"),
  dict(item="referencia sin entrenar", detalle="la misma arquitectura y config con init aleatoria, semilla 0; para vjepa2_ac "
       "incluye el encoder; las estadísticas de pose se copian del modelo"),
  dict(item="referencia de píxeles", detalle="píxeles de los mismos frames, promediados en parches de 16x16 (8x16 por canal); "
       "para vjepa2 los 2 frames de cada tubelet se promedian"),
  dict(item="probes: pose objetivo", detalle="velocidad (dim 0) y giro (dim 5) de la pose del último frame; normalizadas con "
       "la media y el desvío de los targets de train"),
  dict(item="horizonte", detalle=f"{CTX} frames de contexto, {H_MAX} pasos, acciones reales de las poses; errores medios por "
       "paso en el espacio latente de cada modelo (no comparables entre modelos); acción nula = la acción media"),
  dict(item="sorpresa", detalle=f"corte en el paso {CORTE} de {PASOS}; teletransporte = frames del mismo paso de una ventana "
       f"de un segmento de otra ruta; salto = frames {SALTO} pasos (2 s) más adelante del mismo segmento; las poses no se "
       "tocan; una ventana por segmento; el test es pareado entre segmentos (t de Student sobre la diferencia por segmento)"),
  dict(item="cem: contexto", detalle=f"{CTX} frames de contexto con sus acciones ya tomadas; el objetivo es el frame a "
       "`horizonte` pasos; sin momento"),
  dict(item="cem: desvío de muestreo", detalle="el desvío por dimensión de las acciones de train (normalizadas para lewm; "
       "diferencias de estado normalizadas para vjepa2_ac), no el N(0, 1) fijo de los papers; media inicial 0"),
  dict(item="cem: horizontes", detalle="lewm: 5 pasos (el del repo); vjepa2_ac: 1 paso (el del paper) y 5 pasos (propio, para "
       "compararlo con lewm, con menos planes por su costo: --n-plan-largo); el paper de V-JEPA 2-AC no fija el largo del "
       "contexto; el predictor de AC corre en bf16 en CUDA (como se entrenó)"),
  dict(item="cem: lo que se compara", detalle="la pose (velocidad, giro) de cada paso del plan contra la real: lewm = promedio "
       "de las 5 poses del paso; vjepa2_ac = pose del frame al que se llega (estado + acciones); MAE en unidades crudas y "
       "en desvíos de los datos; un plan por segmento"),
  dict(item="IC95", detalle="percentiles 2,5 y 97,5 de 1000 remuestreos de segmentos con reposición (semilla fija); los "
       "mismos segmentos remuestreados para el modelo y sus referencias, así que las diferencias son pareadas"),
]


# ---------------------------------------------------------------- estadística

class Boot:
  """Bootstrap por segmento: los mismos remuestreos de segmentos para todo lo que se mida sobre las mismas ventanas."""

  def __init__(self, seg, n_boot=1000, seed=0):
    _, self.cod = np.unique(np.asarray(seg), return_inverse=True)
    self.n = int(self.cod.max()) + 1 if len(self.cod) else 0
    self.idx = np.random.default_rng(seed).integers(0, max(self.n, 1), (n_boot, max(self.n, 1)))

  def sums(self, cols):
    """cols (n, k) por ventana -> (suma (k,), sumas de cada remuestreo (k, n_boot))."""
    cols = np.asarray(cols, np.float64).reshape(len(self.cod), -1)
    S = np.stack([np.bincount(self.cod, cols[:, j], self.n) for j in range(cols.shape[1])], 1)  # (n_seg, k)
    return S.sum(0), S[self.idx].sum(1).T

  def ic(self, cols, fn):
    """{"valor", "ic95"} de fn(sumas) (fn indexa s[j]: funciona con (k,) y con (k, n_boot))."""
    s, b = self.sums(cols)
    v = np.asarray(fn(b))
    return {"valor": float(fn(s)), "ic95": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]}


def razon(a, b):
  return a / np.maximum(b, 1e-30)


def por_segmento(valores, seg):
  """Promedia por segmento: (valores (n,), seg (n,)) -> (promedios (n_seg,), n_seg)."""
  _, cod = np.unique(np.asarray(seg), return_inverse=True)
  v = np.asarray(valores, np.float64)
  n = cod.max() + 1
  return np.bincount(cod, v, n) / np.bincount(cod, minlength=n), int(n)


def sorpresa_pareada(orig, pert, seg, n_boot=1000, seed=0):
  """Efecto pareado a nivel segmento: cada segmento aporta el promedio de sus ventanas (no cada ventana).

  Devuelve original, perturbado, diferencia (pert − orig) y razón (pert / orig) con IC95, y el t de Student pareado.
  Si orig y pert son lo mismo, el efecto es exactamente 0 (razón 1, p = 1)."""
  a, n = por_segmento(orig, seg)
  b, _ = por_segmento(pert, seg)
  d = b - a
  sd = d.std(ddof=1) if n > 1 else 0.0
  if sd > 0:
    t = float(d.mean() / (sd / np.sqrt(n)))
    p = float(2 * t_student.sf(abs(t), n - 1))
  else:
    t, p = 0.0, (1.0 if d.mean() == 0 else 0.0)
  boot = Boot(np.arange(n), n_boot, seed)
  cols = np.stack([a, b, np.ones(n)], 1)
  return {"original": boot.ic(cols, lambda s: razon(s[0], s[2])), "perturbado": boot.ic(cols, lambda s: razon(s[1], s[2])),
          "dif": boot.ic(cols, lambda s: razon(s[1] - s[0], s[2])), "razon": boot.ic(cols, lambda s: razon(s[1], s[0])),
          "t": t, "p": p, "n_segmentos": n}


# ---------------------------------------------------------------- probes (ridge)

def ridge_fit(Xtr, ytr):
  """Ridge con features estandarizadas e intercepto sin penalizar; una SVD sirve para todos los α. -> pred(X, alpha)."""
  Xtr, ytr = np.asarray(Xtr, np.float64), np.asarray(ytr, np.float64)
  mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
  ym = ytr.mean(0)
  U, S, Vt = np.linalg.svd((Xtr - mu) / sd, full_matrices=False)
  Uy = U.T @ (ytr - ym)

  def pred(X, alpha):
    W = Vt.T @ ((S / (S ** 2 + alpha))[:, None] * Uy)
    return ((np.asarray(X, np.float64) - mu) / sd) @ W + ym
  return pred


def elegir_alpha(Ftr, ytr, Fho, yho, alfas=ALFAS):
  """α con menor MSE (medio entre dimensiones) en el HOLDOUT. No recibe nada de val. -> (alpha, {alpha: mse})."""
  pred = ridge_fit(Ftr, ytr)
  tabla = {a: float(((pred(Fho, a) - yho) ** 2).mean()) for a in alfas}
  return min(tabla, key=tabla.get), tabla


def ajustar_probe(Ftr, ytr, Fho, yho, Fva, alfas=ALFAS):
  """Elige α en el holdout, reajusta con train y predice val. Val entra solo como features: nunca decide un hiperparámetro."""
  alpha, tabla = elegir_alpha(Ftr, ytr, Fho, yho, alfas)
  return alpha, ridge_fit(Ftr, ytr)(Fva, alpha), tabla


def puntajes(pred, y, seg, boot):
  """{dim: {mse, r2 con IC, pearson}} de pred (n, 2) contra y (n, 2), normalizadas con train."""
  out = {}
  for d, j in enumerate(DIMS):
    e2 = (pred[:, d] - y[:, d]) ** 2
    cols = np.stack([e2, y[:, d], y[:, d] ** 2, np.ones(len(y))], 1)
    r2 = lambda s: 1 - s[0] / np.maximum(s[2] - s[1] ** 2 / np.maximum(s[3], 1), 1e-30)
    r = float(np.corrcoef(pred[:, d], y[:, d])[0, 1]) if pred[:, d].std() > 0 and y[:, d].std() > 0 else float("nan")
    out[j] = {"mse": boot.ic(cols, lambda s: s[0] / np.maximum(s[3], 1)), "r2": boot.ic(cols, r2), "pearson": r}
  return out


def dif_mse(pred_a, pred_b, y, boot):
  """MSE(a) − MSE(b) por dimensión, pareado (los mismos remuestreos); negativo = a mejor que b."""
  out = {}
  for d, j in enumerate(DIMS):
    cols = np.stack([(pred_a[:, d] - y[:, d]) ** 2, (pred_b[:, d] - y[:, d]) ** 2, np.ones(len(y))], 1)
    out[j] = boot.ic(cols, lambda s: (s[0] - s[1]) / np.maximum(s[2], 1))
  return out


# ---------------------------------------------------------------- datos y ventanas

def cargar(rutas):
  tok, pos, keys = rutas
  return np.load(tok, mmap_mode="r"), np.load(pos, mmap_mode="r"), json.loads(Path(keys).read_text())


def lista_ventanas(segs, T, por_seg, seed):
  """[(segmento, inicio, k)] con `por_seg` ventanas por segmento; cada ventana más el salto cabe en el segmento."""
  hi = T - SPAN - SALTO * SKIP
  assert hi >= 1, f"segmentos de {T} frames: no alcanzan para una ventana de {SPAN} + el salto de {SALTO * SKIP}"
  rng = np.random.default_rng([seed, T, len(segs)])
  return [(int(s), int(t), k) for s in segs for k, t in enumerate(np.sort(rng.integers(0, hi, por_seg)))]


def tomar(tok, pos, idx, corrimiento=0, desde=0):
  """Tokens (n, PASOS, 128) int64 y poses crudas (n, PASOS, 6·SKIP) float32 de las ventanas `idx`.

  corrimiento: pasos que se adelantan los frames `desde` en adelante (salto); las poses no se corren."""
  seg = np.array([s for s, _, _ in idx])[:, None]
  t0 = np.array([t for _, t, _ in idx])[:, None]
  paso = np.arange(PASOS)[None] * SKIP
  fr = t0 + paso + np.where(np.arange(PASOS)[None] >= desde, corrimiento * SKIP, 0)
  tk = np.asarray(tok[seg, fr]).astype(np.int64)
  raw = t0 + np.arange(SPAN)[None]
  ps = np.asarray(pos[seg, raw], np.float32).reshape(len(idx), PASOS, -1)
  return tk, ps


def render(tk):
  """tokens (B, F, 128) en el device -> uint8 (B, F, 128, 256, 3) con el decoder de comma."""
  return pixels.render(tk)


def estadisticas_pose(pos, n_train):
  p = np.asarray(pos[:n_train]).reshape(-1, pos.shape[-1])
  return np.nanmean(p, 0).astype(np.float32), np.maximum(np.nanstd(p, 0), 1e-6).astype(np.float32)


def poner_estadisticas_pose(model, tpos, n_tr):
  """Un modelo sin las estadísticas de pose de su entrenamiento (sin entrenar) usa las de train. -> de dónde salen."""
  if hasattr(model, "pose_std") and bool((model.pose_std == 1).all()) and bool((model.pose_mean == 0).all()):
    mean, std = estadisticas_pose(tpos, n_tr)
    k = model.pose_mean.numel() // 6
    model.set_pose_stats(np.tile(mean, k), np.tile(std, k))
    return "de train (el modelo no las traía)"
  return "del checkpoint"


def pix_features(img, frames_por_paso, n=N_PASOS):
  """Píxeles de los últimos n pasos reducidos a 8x16 por canal: (B, n·3·8·16). Referencia de los probes."""
  x = img[:, -n * frames_por_paso:].float().div(255)
  B, T = x.shape[:2]
  x = F.adaptive_avg_pool2d(x.permute(0, 1, 4, 2, 3).reshape(B * T, 3, *x.shape[2:4]), (8, 16))
  return x.reshape(B, n, frames_por_paso, -1).mean(2).flatten(1).cpu()


def modelo_sin_entrenar(model, seed=0):
  """Misma arquitectura y config, init aleatoria con semilla fija, mismas estadísticas de pose."""
  cfg = dataclasses.asdict(model.cfg)
  if isinstance(model, VJEPA2AC):
    cfg.update(cargar_encoder=False, encoder_run="")
  torch.manual_seed(seed)
  ref = build(KINDS[type(model)], cfg).to(model.device).eval()
  if hasattr(model, "pose_mean"):
    ref.set_pose_stats(model.pose_mean.cpu(), model.pose_std.cpu())
  return ref


def convencion_real(model, poses, ctx, P):
  """Pose (velocidad, giro) real de cada paso del plan, en la convención del modelo. poses (PASOS, 6·SKIP) crudas."""
  d = list(DIMS.values())
  if model.lat_convencion == "promedio_bloque":
    return poses.reshape(PASOS, SKIP, 6)[ctx - 1:ctx - 1 + P].mean(1)[:, d]
  return poses[ctx:ctx + P, :6][:, d]


def convencion_base(model, poses, ctx, P, media):
  """(repetir la última acción, acción media de train) en la misma convención. media: (media de pose, media de Δ) de train."""
  d = list(DIMS.values())
  if model.lat_convencion == "promedio_bloque":
    ultima = poses.reshape(PASOS, SKIP, 6)[ctx - 2].mean(0)[d]
    return np.repeat(ultima[None], P, 0), np.repeat(media[0][d][None], P, 0)
  s = poses[:, :6][:, d]
  paso = np.arange(1, P + 1)[:, None]
  return s[ctx - 1] + paso * (s[ctx - 1] - s[ctx - 2]), s[ctx - 1] + paso * media[1][d]


# ---------------------------------------------------------------- las etapas

def extraer(model, ref, img):
  """Features de un lote: {"obs"|"pred": {"modelo", "sin_entrenar"}, "pixeles"} en la CPU, float32."""
  out = {}
  for nombre, m in (("modelo", model), ("sin_entrenar", ref)):
    for k, v in m.lat_features(img, N_PASOS).items():
      out.setdefault(k, {})[nombre] = v.float().cpu().numpy()
  out["pixeles"] = pix_features(img, model.lat_frames_por_paso).numpy()
  return out


def juntar(partes):
  """Lista de dicts anidados de arrays -> el mismo dict con las partes concatenadas."""
  if isinstance(partes[0], dict):
    return {k: juntar([p[k] for p in partes]) for k in partes[0]}
  return np.concatenate(partes)


def split_features(model, ref, tok, pos, idx, bs, dev):
  """Features, targets (pose del último frame, cruda) y segmentos de un split (train u holdout)."""
  feats, ys = [], []
  for i in range(0, len(idx), bs):
    tk, ps = tomar(tok, pos, idx[i:i + bs])
    feats.append(extraer(model, ref, render(torch.from_numpy(tk).to(dev))))
    ys.append(ps[:, -1, :6][:, list(DIMS.values())])
  return juntar(feats), np.concatenate(ys), np.array([s for s, _, _ in idx])


def etapa_probes(model, ref, tr, ho, va, a):
  """Entrena los probes en train, elige α en holdout y mide en val, con sus referencias."""
  (Ftr, ytr, _), (Fho, yho, _), (Fva, yva, seg_va) = tr, ho, va
  ok = lambda y: np.isfinite(y).all(1)
  mtr, mho, mva = ok(ytr), ok(yho), ok(yva)
  mu, sd = ytr[mtr].mean(0), ytr[mtr].std(0) + 1e-9
  norm = lambda y, m: (y[m] - mu) / sd
  ytr_n, yho_n, yva_n = norm(ytr, mtr), norm(yho, mho), norm(yva, mva)
  boot = Boot(seg_va[mva], a.n_boot)
  hay_pred = "pred" in Fva
  res = {"normalizacion": "pose objetivo normalizada con la media y el desvío de train", "n_train": int(mtr.sum()),
         "n_holdout": int(mho.sum()), "n_val": int(mva.sum()), "alpha": {}}
  for tipo in (["obs", "pred"] if hay_pred else ["obs"]):
    fuentes = {"modelo": (Ftr[tipo]["modelo"], Fho[tipo]["modelo"], Fva[tipo]["modelo"]),
               "sin_entrenar": (Ftr[tipo]["sin_entrenar"], Fho[tipo]["sin_entrenar"], Fva[tipo]["sin_entrenar"]),
               "pixeles": (Ftr["pixeles"], Fho["pixeles"], Fva["pixeles"])}
    preds, bloque = {}, {}
    for nombre, (xtr, xho, xva) in fuentes.items():
      alpha, preds[nombre], _ = ajustar_probe(xtr[mtr], ytr_n, xho[mho], yho_n, xva[mva])
      res["alpha"][f"{tipo}/{nombre}"] = alpha
      bloque[nombre] = puntajes(preds[nombre], yva_n, seg_va[mva], boot)
    preds["media_train"] = np.zeros_like(yva_n)
    bloque["media_train"] = puntajes(preds["media_train"], yva_n, seg_va[mva], boot)
    for ref_ in ("sin_entrenar", "pixeles", "media_train"):
      bloque[f"dif_mse_modelo_menos_{ref_}"] = dif_mse(preds["modelo"], preds[ref_], yva_n, boot)
    res[tipo] = {"dimensiones": int(fuentes["modelo"][0].shape[1]), **bloque}
  return res


def resumen_horizonte(r, seg, boot_seed=0, n_boot=1000):
  """r: {"real"|"nula": {"mse","l1","copia_mse","copia_l1": (n, H)}} -> por horizonte, modelo y referencias con IC."""
  boot = Boot(seg, n_boot, boot_seed)
  H = r["real"]["mse"].shape[1]
  out = {"pasos": list(range(1, H + 1)), "segundos": [(h + 1) * SKIP / 20 for h in range(H)], "contexto": CTX}
  for m in ("mse", "l1"):
    cols = np.concatenate([r["real"][m], r["real"][f"copia_{m}"], r["nula"][m], np.ones((len(seg), 1))], 1)
    n = lambda s, k: s[k * H:(k + 1) * H] / np.maximum(s[3 * H], 1)
    s, b = boot.sums(cols)  # una sola vez: son (n_boot, segmentos, 3H + 1) números

    def lista(fn):
      lo, hi = np.percentile(fn(b), [2.5, 97.5], axis=-1)
      return [{"valor": float(v), "ic95": [float(l), float(h)]} for v, l, h in zip(fn(s), lo, hi)]
    out[m] = {"modelo": lista(lambda s: n(s, 0)), "copiar_ultimo": lista(lambda s: n(s, 1)),
              "accion_nula": lista(lambda s: n(s, 2)),
              "razon_modelo_sobre_copiar": lista(lambda s: razon(n(s, 0), n(s, 1))),
              "razon_accion_nula_sobre_real": lista(lambda s: razon(n(s, 2), n(s, 0)))}
  return out


def resumen_sorpresa(e, seg, a):
  """e: {"orig"|"tele"|"salto": {"mse","copia_mse": (n, PASOS-1)}} -> efectos pareados del corte, con su referencia."""
  c = CORTE - 1
  out = {"paso_del_corte": CORTE, "salto_pasos": SALTO}
  for nombre, k in (("teletransporte", "tele"), ("salto_2s", "salto")):
    out[nombre] = {"modelo": sorpresa_pareada(e["orig"]["mse"][:, c], e[k]["mse"][:, c], seg, a.n_boot),
                   "referencia_copiar_ultimo": sorpresa_pareada(e["orig"]["copia_mse"][:, c], e[k]["copia_mse"][:, c], seg, a.n_boot),
                   "por_paso_original": e["orig"]["mse"].mean(0).tolist(), "por_paso_perturbado": e[k]["mse"].mean(0).tolist()}
  return out


def etapa_val(model, ref, tok, pos, idx, bs, dev, partner):
  """Una pasada por val: features de los probes, error por horizonte y error de sorpresa (ventana 0 de cada segmento)."""
  feats, ys, hor, sor = [], [], {"real": [], "nula": []}, {"orig": [], "tele": [], "salto": []}
  seg_sor = []
  aplica = model.lat_aplica
  for i in range(0, len(idx), bs):
    lote = idx[i:i + bs]
    tk, ps = tomar(tok, pos, lote)
    img = render(torch.from_numpy(tk).to(dev))
    P = torch.from_numpy(ps).to(dev)
    feats.append(extraer(model, ref, img))
    ys.append(ps[:, -1, :6][:, list(DIMS.values())])
    if "horizonte" in aplica:
      for nombre, nula in (("real", False), ("nula", True)):
        hor[nombre].append({k: v.cpu().numpy() for k, v in model.lat_horizonte(img, P, CTX, H_MAX, nula).items()})
    if "sorpresa" in aplica:
      sel = [j for j, (_, _, k) in enumerate(lote) if k == 0]
      if sel:
        sub = [lote[j] for j in sel]
        tele_t = tk[sel].copy()
        otros = [partner[s] for s, _, _ in sub]
        tk_o, _ = tomar(tok, pos, [(o, t, 0) for o, (_, t, _) in zip(otros, sub)])
        tele_t[:, CORTE:] = tk_o[:, CORTE:]
        tk_j, _ = tomar(tok, pos, sub, SALTO, CORTE)
        Ps = P[sel]
        for nombre, im in (("orig", img[sel]), ("tele", render(torch.from_numpy(tele_t).to(dev))),
                           ("salto", render(torch.from_numpy(tk_j).to(dev)))):
          sor[nombre].append({k: v.cpu().numpy() for k, v in model.lat_sorpresa(im, Ps).items()})
        seg_sor += [s for s, _, _ in sub]
  seg = np.array([s for s, _, _ in idx])
  out = {"feats": juntar(feats), "y": np.concatenate(ys), "seg": seg}
  if "horizonte" in aplica:
    out["hor"] = {k: juntar(v) for k, v in hor.items()}
  if "sorpresa" in aplica:
    out["sor"] = {k: juntar(v) for k, v in sor.items()}
    out["seg_sor"] = np.array(seg_sor)
  return out


def etapa_cem(model, tok, pos, plan_idx, tr_poses, dev, a):
  """Un plan por segmento con el CEM del paper del modelo; MAE de la pose planeada contra la real y las dos referencias."""
  with torch.no_grad():
    desvio = model.lat_desvio(torch.from_numpy(tr_poses).to(dev))
  P_tr = tr_poses[..., :6].reshape(-1, 6)
  media = (np.nanmean(P_tr, 0), np.nanmean(np.diff(tr_poses[:, :, :6], axis=1).reshape(-1, 6), 0))
  sd_dato = np.nanstd(P_tr, 0)[list(DIMS.values())]
  g = torch.Generator(device=dev).manual_seed(0)
  res = {"aplica": True, "desvio_muestreo": [float(x) for x in desvio.cpu()], "convencion_pose": model.lat_convencion,
         "algoritmo": model.lat_cem, "n_planes": len(plan_idx), "contexto": CTX}
  for P in horizontes_plan(model):
    # el CEM de V-JEPA 2-AC a 5 pasos cuesta ~8 veces el de 1 paso (800 muestras por iteración): menos planes
    planes = plan_idx[:a.n_plan_largo] if (P > 1 and isinstance(model, VJEPA2AC)) else plan_idx
    errs = {"cem": [], "repetir_ultima": [], "media_train": []}
    ms = []
    for s, t, _ in planes:
      tk, ps = tomar(tok, pos, [(s, t, 0)])
      img = render(torch.from_numpy(tk).to(dev))
      if dev.type == "cuda":
        torch.cuda.synchronize()
      t1 = time.perf_counter()
      pl = model.lat_plan(img, torch.from_numpy(ps).to(dev), CTX, P, desvio, g)[0].float().cpu().numpy()
      if dev.type == "cuda":
        torch.cuda.synchronize()
      ms.append((time.perf_counter() - t1) * 1000)
      real = convencion_real(model, ps[0], CTX, P)
      ult, med = convencion_base(model, ps[0], CTX, P, media)
      for k, p in (("cem", pl), ("repetir_ultima", ult), ("media_train", med)):
        errs[k].append(np.abs(p - real).mean(0))  # (2,) por dimensión
    E = {k: np.stack(v) for k, v in errs.items()}
    boot = Boot(np.arange(len(planes)), a.n_boot)
    out = {"ms_por_plan": float(np.median(ms)), "pasos": P, "n_planes": len(planes)}
    for d, j in enumerate(DIMS):
      cols = np.stack([E["cem"][:, d], E["repetir_ultima"][:, d], E["media_train"][:, d], np.ones(len(E["cem"]))], 1)
      m = lambda k: (lambda s: s[k] / np.maximum(s[3], 1))
      out[j] = {"mae_cem": boot.ic(cols, m(0)), "mae_repetir_ultima": boot.ic(cols, m(1)), "mae_media_train": boot.ic(cols, m(2)),
                "mae_cem_en_desvios": float(E["cem"][:, d].mean() / sd_dato[d]),
                "dif_cem_menos_repetir_ultima": boot.ic(cols, lambda s: (s[0] - s[1]) / np.maximum(s[3], 1)),
                "dif_cem_menos_media_train": boot.ic(cols, lambda s: (s[0] - s[2]) / np.maximum(s[3], 1))}
    res[f"h{P}"] = out
  return res


# ---------------------------------------------------------------- main

def parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--model", required=True, help="corrida, ruta .pt o random:<kind> (lewm, vjepa2, vjepa2_ac)")
  p.add_argument("--name")
  p.add_argument("--limite", type=int, help="usar solo N segmentos de val (y 3N de train): para pruebas rápidas")
  p.add_argument("--no-plan", action="store_true", help="sin CEM")
  p.add_argument("--n-plan", type=int, default=100, help="planes del CEM (uno por segmento de val)")
  p.add_argument("--n-plan-largo", type=int, default=25, help="planes del CEM de 5 pasos de vjepa2_ac (cuesta ~8x el de 1 paso)")
  p.add_argument("--train-seg", type=int, default=1000, help="segmentos de train para entrenar los probes")
  p.add_argument("--por-seg", type=int, default=2, help="ventanas por segmento")
  p.add_argument("--bs", type=int, default=8, help="ventanas por lote")
  p.add_argument("--n-boot", type=int, default=1000)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--out", default="results")
  a = p.parse_args(argv)
  return a


def horizontes_plan(model):
  """Pasos del plan: lewm 5 (el del repo); vjepa2_ac 1 (el del paper) y 5 (propio, para compararlo con lewm)."""
  return (1, 5) if isinstance(model, VJEPA2AC) else (5,)


def main(argv=None):
  a = parse(argv)
  dev = device()
  info = log_device(dev)
  torch.manual_seed(a.seed)
  model = load_model(a.model, dev).eval()
  if type(model) not in KINDS:
    raise SystemExit(f"{a.model}: eval_latent evalúa lewm, vjepa2 y vjepa2_ac, no {type(model).__name__}")
  name = a.name or model.name
  kind = KINDS[type(model)]
  ttok, tpos, tkeys = cargar(TRAIN)
  vtok, vpos, vkeys = cargar(VAL)
  n_tr = len(ttok) - HOLDOUT
  stats = poner_estadisticas_pose(model, tpos, n_tr)
  ref = modelo_sin_entrenar(model, a.seed)

  nv = min(a.limite or len(vtok), len(vtok))
  tr_segs = np.unique(np.linspace(0, n_tr - 1, min(n_tr, a.limite * 3 if a.limite else a.train_seg)).astype(int))
  ho_segs = disjoint_holdout(tkeys, HOLDOUT) or list(range(n_tr, len(ttok)))
  if a.limite:
    ho_segs = ho_segs[:max(2, a.limite)]
  v_segs = np.unique(np.linspace(0, len(vtok) - 1, nv).astype(int))
  T = ttok.shape[1]
  assert vtok.shape[1] == T, "train y val con distinto largo de segmento"
  w_tr = lista_ventanas(tr_segs, T, a.por_seg, a.seed)
  w_ho = lista_ventanas(ho_segs, T, a.por_seg, a.seed + 1)
  w_va = lista_ventanas(v_segs, T, a.por_seg, a.seed + 2)
  print(f"{name} ({kind}): train {len(w_tr)} ventanas / {len(tr_segs)} segmentos, holdout {len(w_ho)}, val {len(w_va)} / "
        f"{len(v_segs)} segmentos", flush=True)
  t0 = time.time()

  tr = split_features(model, ref, ttok, tpos, w_tr, a.bs, dev)
  ho = split_features(model, ref, ttok, tpos, w_ho, a.bs, dev)
  print(f"features de train y holdout ({time.time() - t0:.0f}s)", flush=True)
  # el compañero de cada segmento en el teletransporte: otro segmento de otra ruta
  partner = {}
  for i, s in enumerate(v_segs):
    j = (i + len(v_segs) // 2) % len(v_segs)
    for _ in range(len(v_segs)):
      if route_of(vkeys[v_segs[j]]) != route_of(vkeys[s]):
        break
      j = (j + 1) % len(v_segs)
    partner[int(s)] = int(v_segs[j])
  hay_otra_ruta = len({route_of(vkeys[s]) for s in v_segs}) > 1
  v = etapa_val(model, ref, vtok, vpos, w_va, a.bs, dev, partner)
  print(f"val ({time.time() - t0:.0f}s)", flush=True)

  res = {"model": a.model, "name": name, "kind": kind, "device_info": info, "lectura": "latente", "semilla": a.seed,
         "datos": {"train_segmentos": len(tr_segs), "holdout_segmentos": len(ho_segs), "val_segmentos": len(v_segs),
                   "ventanas": {"train": len(w_tr), "holdout": len(w_ho), "val": len(w_va)}, "val": VAL[0],
                   "pasos_por_ventana": PASOS, "hz": 4, "estadisticas_de_pose": stats},
         "params": int(model.num_params()), "pooling": model.lat_pooling}
  res["probes"] = etapa_probes(model, ref, tr, ho, (v["feats"], v["y"], v["seg"]), a)
  res["horizonte"] = resumen_horizonte(v["hor"], v["seg"], a.seed, a.n_boot) if "hor" in v else {**NA, "motivo": "no predice el futuro"}
  if "sor" in v:
    res["sorpresa"] = resumen_sorpresa(v["sor"], v["seg_sor"], a)
    res["sorpresa"]["nota_teletransporte"] = None if hay_otra_ruta else "val con una sola ruta: el teletransporte usa otro segmento de la misma"
  else:
    res["sorpresa"] = {**NA, "motivo": "no predice el futuro"}
  if "plan" not in model.lat_aplica:
    res["cem"] = {**NA, "motivo": "no tiene acciones ni predictor"}
  elif a.no_plan:
    res["cem"] = {**NA, "motivo": "--no-plan"}
  else:
    plan_idx = [w for w in w_va if w[2] == 0][:a.n_plan]
    res["cem"] = etapa_cem(model, vtok, vpos, plan_idx, poses_train_a_4hz(tpos, tr_segs), dev, a)
  res["fuente"] = FUENTE[kind]
  res["propio"] = PROPIO
  res["segundos"] = time.time() - t0
  res["args"] = vars(a)
  Path(a.out).mkdir(parents=True, exist_ok=True)
  p = Path(a.out) / f"latent_{name}.json"
  p.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
  print(p, flush=True)
  return res


def poses_train_a_4hz(tpos, segs, n=300):
  """Poses de train (N, F, 6·SKIP) a 4 Hz para el desvío de muestreo del CEM: hasta n tramos de 30 pasos."""
  segs = np.asarray(segs)[:max(1, n // 4)]
  T = tpos.shape[1] // SKIP * SKIP
  p = np.asarray(tpos[segs, :T], np.float32)
  return p.reshape(len(segs), T // SKIP, SKIP * 6)


if __name__ == "__main__":
  main()
