"""Set fijo de clips de evaluación y rollouts de los modelos (plan 2.4).

Clips: 100 segmentos de val x 4 arranques = 400 clips; 20 frames de contexto (1 s) + 60 de rollout (3 s).
  python -m src.rollouts --make-clips        escribe results/clips_eval.json
  python -m src.rollouts --make-clips --clips holdout   escribe results/clips_holdout.json (E2)

Segundo set de val (--clips eval_b, results/clips_eval_b.json): para el piso de FVD real contra real a 400 clips por
lado. Val tiene exactamente 100 segmentos (los mismos del benchmark), así que son los mismos segmentos con otros 4
arranques (235, 505, 775, 1045), sin solaparse con los de clips_eval.json (cada clip usa 80 frames: 20 de contexto y 60).
  python -m src.rollouts --model real --clips eval_b --name real_eval_b   (y metrics_video --name real_eval_b --clips eval_b)

Clips del holdout (--clips holdout): los últimos 50 segmentos de data/train_tokens.npy (el holdout de src.windows.Windows),
4 arranques por segmento = 200 clips, mismo formato que clips_eval.json. Sirven para elegir hiperparámetros (la
temperatura, src.temperatura) sin tocar val. "segment" es el índice dentro del holdout (0..49) y "row" la fila en
train_tokens.npy. Sin --clips (o con --clips eval) todo es igual que antes.

Rollouts:
  python -m src.rollouts --model SPEC [--name NAME] [--bs 50] [--steps S] [--temperature 1.0]
                         [--dtype fp32|fp16|bf16] [--ctx-frames K] [--limit N] [--seed 0]
  SPEC es lo que entiende src.models.load_model, o "real" (los tokens reales como "rollout").
  Escribe results/rollouts/<name>.npy (N, 60, 128) int16 y <name>.json con los metadatos. Un modelo con
  output = "pixels" (convlstm_px) escribe en cambio uint8 (N, 60, 128, 256, 3): son los frames predichos, sin tokens.

Los modelos latentes (lewm, vjepa2, vjepa2_ac) no entran acá (D12): python -m src.eval_latent.
La temperatura por defecto es la del modelo (cfg.temperature: 2 en genie, 1 en el resto); --temperature la cambia.

Nombre por defecto: model.name + sufijos solo para lo que no es el valor por defecto:
  _s<steps> (--steps), _<dtype> (fp16/bf16), _ctx<K> (--ctx-frames), _t<temp> (temperatura distinta de la del modelo).
  Ejemplos: maskgit_s_s4, gpt_s_fp16, gpt_s_ctx8, gpt_s_t0.8. "real" se llama "real".
Cada lote usa un Generator con semilla seed * 100003 + índice del primer clip del lote.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.common import CTX_FRAMES, FPS, device, device_info, log_device

CLIPS = Path("results/clips_eval.json")
CLIPS_B = Path("results/clips_eval_b.json")
CLIPS_HOLDOUT = Path("results/clips_holdout.json")
CLIPS_MOTION_SUAVE = Path("results/clips_motion_suave.json")  # 50 curvas de 10-20° (src.clips_motion.make_suave)
CLIPS_MOTION = Path("results/clips_motion.json")   # 100 rectas + 100 curvas del val con poses (src.clips_motion)
TRAIN = "data/train_tokens.npy"
HOLDOUT = 50                                    # igual que Windows(holdout=50)
LATENTES = ("LeWM", "VJEPA2", "VJEPA2AC")       # clases sin tokens de salida: tabla propia (D12)
SPLIT = Path("results/val_split.json")
VAL = "data/val_tokens.npy"
OUT = Path("results/rollouts")
HORIZON = 60
STARTS = [100, 370, 640, 910]
STARTS_B = [235, 505, 775, 1045]    # sin solape con STARTS: [s, s + 80) disjuntos
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def make_clips(path=CLIPS, starts=STARTS, extra=None):
  keys = json.loads(SPLIT.read_text())["segments"]
  clips = [{"clip": i * len(starts) + j, "segment": i, "key": k, "start": s}
           for i, k in enumerate(keys) for j, s in enumerate(starts)]
  d = {"ctx": CTX_FRAMES, "horizon": HORIZON, "fps": FPS, "starts": list(starts), **(extra or {}), "clips": clips}
  Path(path).write_text(json.dumps(d, indent=1))
  return d


def make_clips_b(path=CLIPS_B):
  """Segundo set de val: mismos segmentos (val no tiene más de 100) con arranques distintos y disjuntos de los de eval."""
  n = len(np.load(VAL, mmap_mode="r"))
  assert n == len(json.loads(SPLIT.read_text())["segments"]), "val tiene segmentos de más: conviene usar otros"
  assert all(a + 80 <= b or b + 80 <= a for a in STARTS for b in STARTS_B)
  return make_clips(path, STARTS_B, {"source": "val_eval_b", "nota": "val tiene solo los 100 segmentos del benchmark: "
                    "mismos segmentos, otros arranques sin solape con clips_eval.json (cada clip usa 80 frames)"})


def make_clips_holdout(path=CLIPS_HOLDOUT, tokens=TRAIN, holdout=HOLDOUT):
  """Clips de los últimos `holdout` segmentos de train (los que Windows deja aparte), mismos arranques que val."""
  n = len(np.load(tokens, mmap_mode="r"))
  clips = [{"clip": i * len(STARTS) + j, "segment": i, "row": n - holdout + i, "start": s}
           for i in range(holdout) for j, s in enumerate(STARTS)]
  d = {"ctx": CTX_FRAMES, "horizon": HORIZON, "fps": FPS, "starts": STARTS, "source": "train_holdout",
       "tokens": str(tokens), "holdout": holdout, "clips": clips}
  Path(path).write_text(json.dumps(d, indent=1))
  return d


def load_clips_holdout(path=CLIPS_HOLDOUT, tokens=TRAIN, holdout=HOLDOUT):
  """(clips, tokens del holdout (holdout, T, 128) memmap). Crea el archivo si falta."""
  c = json.loads(Path(path).read_text()) if Path(path).exists() else make_clips_holdout(path, tokens, holdout)
  return c, np.load(c.get("tokens", tokens), mmap_mode="r")[-c.get("holdout", holdout):]


def clip_set(nombre):
  """(path, clips, tokens) del set de clips por nombre: eval | eval_b | holdout | motion."""
  if nombre == "holdout":
    c, t = load_clips_holdout()
    return CLIPS_HOLDOUT, c, t
  if nombre == "motion_suave":
    if not CLIPS_MOTION_SUAVE.exists():
      from src import clips_motion
      clips_motion.make_suave(CLIPS_MOTION_SUAVE)
    c = json.loads(CLIPS_MOTION_SUAVE.read_text())
    return CLIPS_MOTION_SUAVE, c, np.load(c["tokens"], mmap_mode="r")
  if nombre == "motion":
    if not CLIPS_MOTION.exists():
      from src import clips_motion
      clips_motion.make(CLIPS_MOTION)
    c = json.loads(CLIPS_MOTION.read_text())
    return CLIPS_MOTION, c, np.load(c["tokens"], mmap_mode="r")
  cp = CLIPS_B if nombre == "eval_b" else CLIPS
  return cp, load_clips(cp), np.load(VAL, mmap_mode="r")


def rechazar_latente(model, comando="rollouts"):
  """D12: los latentes no tienen tokens de salida, no entran en la tabla de video ni en CE."""
  if type(model).__name__ in LATENTES:
    raise SystemExit(f"{comando}: {type(model).__name__} es un modelo latente y no entra en la tabla de video (D12): "
                     "usá python -m src.eval_latent --model <modelo>")


def temperatura_default(model):
  """Temperatura de muestreo propia del modelo (Genie: 2 según su paper); 1 si la config no la tiene."""
  t = getattr(getattr(model, "cfg", None), "temperature", None)
  return 1.0 if t is None else float(t)


def load_clips(path=CLIPS):
  """Crea el archivo si falta."""
  path = Path(path)
  if path.exists():
    return json.loads(path.read_text())
  return make_clips_b(path) if path.name == CLIPS_B.name else make_clips(path)


def clips_hash(path=CLIPS):
  load_clips(path)
  return hashlib.sha1(path.read_bytes()).hexdigest()[:10]


def clip_tokens(val, clips):
  """val (S, 1200, 128), clips (dict de load_clips o lista) -> ctx (N, 20, 128), real (N, 60, 128) int16."""
  c = clips["clips"] if isinstance(clips, dict) else clips
  ctx_n = clips["ctx"] if isinstance(clips, dict) else CTX_FRAMES
  hor = clips["horizon"] if isinstance(clips, dict) else HORIZON
  ctx = np.stack([val[x["segment"], x["start"]:x["start"] + ctx_n] for x in c])
  real = np.stack([val[x["segment"], x["start"] + ctx_n:x["start"] + ctx_n + hor] for x in c])
  return ctx, real


def default_name(base, steps=None, dtype="fp32", ctx_frames=CTX_FRAMES, temperature=1.0, default_temperature=1.0):
  name = base
  if steps is not None:
    name += f"_s{steps}"
  if dtype != "fp32":
    name += f"_{dtype}"
  if ctx_frames != CTX_FRAMES:
    name += f"_ctx{ctx_frames}"
  if temperature != default_temperature:
    name += f"_t{temperature:g}"
  return name


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--make-clips", action="store_true")
  ap.add_argument("--model")
  ap.add_argument("--name")
  ap.add_argument("--bs", type=int, default=50)
  ap.add_argument("--steps", type=int)
  ap.add_argument("--temperature", type=float, help="por defecto la del modelo (2 en genie, 1 en el resto)")
  ap.add_argument("--clips", choices=["eval", "eval_b", "holdout", "motion", "motion_suave"], default="eval",
                  help="eval: 400 clips de val (results/clips_eval.json); eval_b: otros 400 de val (piso); holdout: 200 clips "
                       "del holdout de train; motion: 100 rectas + 100 curvas del val con poses (results/clips_motion.json)")
  ap.add_argument("--reanudar", action="store_true",
                  help="guarda results/rollouts/<name>.parcial.npy tras cada batch y, si existe (misma config), sigue desde ahí")
  ap.add_argument("--dtype", default="fp32", choices=list(DTYPES))
  ap.add_argument("--ctx-frames", type=int, default=CTX_FRAMES)
  ap.add_argument("--limit", type=int)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--graphs", action="store_true", help="CUDA graphs en el decode (mismos tokens, más rápido)")
  ap.add_argument("--opt", action="append", help="campo=valor de la Config del modelo (p. ej. rollout=latent); repetible")
  a = ap.parse_args(argv)

  hold = a.clips == "holdout"
  cpath = {"holdout": CLIPS_HOLDOUT, "eval_b": CLIPS_B, "motion": CLIPS_MOTION,
           "motion_suave": CLIPS_MOTION_SUAVE}.get(a.clips, CLIPS)
  if a.make_clips and a.clips == "motion":
    from src import clips_motion
    clips_motion.make(CLIPS_MOTION)
    print(f"{CLIPS_MOTION}: hash {clips_hash(CLIPS_MOTION)}")
    if not a.model:
      return CLIPS_MOTION
  elif a.make_clips:
    d = make_clips_holdout() if hold else (make_clips_b() if a.clips == "eval_b" else make_clips())
    print(f"{cpath}: {len(d['clips'])} clips, hash {clips_hash(cpath)}")
    if not a.model:
      return cpath
  assert a.model, "falta --model SPEC (o --make-clips)"

  _, clips, val = clip_set(a.clips)
  n = min(a.limit or len(clips["clips"]), len(clips["clips"]))
  clips = {**clips, "clips": clips["clips"][:n]}
  ctx_all, real_all = clip_tokens(val, clips)
  ctx_all = ctx_all[:, -a.ctx_frames:]
  dev = device()
  info = log_device(dev)

  real = a.model == "real"
  model = None
  if not real:
    from src.models import apply_opts, load_model
    model = apply_opts(load_model(a.model, dev), a.opt).eval().to(DTYPES[a.dtype])
    rechazar_latente(model, "src.rollouts")
    if a.graphs:
      model.use_cuda_graphs(True)
  t0_model = 1.0 if real else temperatura_default(model)
  temperature = t0_model if a.temperature is None else a.temperature
  base = "real" if real else model.name
  name = a.name or (base if real else default_name(base, a.steps, a.dtype, a.ctx_frames, temperature, t0_model))

  pix = model is not None and getattr(model, "output", "tokens") == "pixels"
  if pix:  # modelo de píxeles: uint8 (N, 60, 128, 256, 3) en disco (memmap), metrics_video lo usa sin decodificar
    OUT.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(OUT / f"{name}.npy", mode="w+", dtype=np.uint8, shape=(n, HORIZON, model.cfg.height, model.cfg.width, 3))
  else:
    out = np.zeros((n, HORIZON, 128), np.int16)
  t0 = time.time()
  n_batches = 0
  parcial, hecho = OUT / f"{name}.parcial.npy", 0
  firma = {"model": a.model, "steps": a.steps, "temperature": temperature, "dtype": a.dtype, "bs": a.bs, "seed": a.seed,
           "opt": a.opt, "clips": a.clips, "n": n}
  if a.reanudar and not pix and parcial.exists():
    pm = json.loads(parcial.with_suffix(".json").read_text())
    if pm["firma"] == firma:
      hecho = pm["hecho"]
      out[:hecho] = np.load(parcial)[:hecho]
      print(f"reanudando {name} desde el clip {hecho}/{n}", flush=True)
  for i in range(0, n, a.bs):
    if i < hecho:
      continue
    if real:
      out[i:i + a.bs] = real_all[i:i + a.bs]
      continue
    ctx = torch.as_tensor(ctx_all[i:i + a.bs].astype(np.int64), device=dev)
    g = torch.Generator(device=dev).manual_seed(a.seed * 100003 + i)
    r = model.rollout(ctx, HORIZON, temperature=temperature, steps=a.steps, generator=g)
    out[i:i + a.bs] = r.cpu().numpy() if pix else r.cpu().numpy().astype(np.int16)
    n_batches += 1
    print(f"  clips {i + len(ctx)}/{n}  {time.time() - t0:.1f} s", flush=True)
    if a.reanudar and not pix:
      OUT.mkdir(parents=True, exist_ok=True)
      np.save(parcial, out)
      parcial.with_suffix(".json").write_text(json.dumps({"firma": firma, "hecho": i + len(ctx)}))
  if dev.type == "cuda":
    torch.cuda.synchronize()
  sec = time.time() - t0

  OUT.mkdir(parents=True, exist_ok=True)
  if pix:
    out.flush()
  else:
    np.save(OUT / f"{name}.npy", out)
    for f in (parcial, parcial.with_suffix(".json")):
      f.unlink(missing_ok=True)
  meta = {"output": "pixels" if pix else "tokens", "model": a.model, "name": name, "opt": a.opt, "steps": a.steps, "temperature": temperature, "dtype": a.dtype,
          "bs": a.bs, "seed": a.seed, "graphs": a.graphs, "ctx_frames": a.ctx_frames, "n_clips": n, "seconds": sec,
          "ms_per_frame_batch": 1000 * sec / (max(n_batches, 1) * HORIZON) if not real else None,
          "device_info": info, "clips": a.clips, "clips_hash": clips_hash(cpath)}
  (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1))
  print(f"{OUT / name}.npy {out.shape}  {sec:.1f} s")
  return OUT / f"{name}.npy"


if __name__ == "__main__":
  main()
