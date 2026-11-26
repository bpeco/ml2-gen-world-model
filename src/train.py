"""Entrenamiento genérico de los world models propios (gpt, maskgit, lewm), con o sin destilación.

Cada modelo trae su receta (`Clase.recipe(size)`, ver src/recipes.py): optimizador, lr, wd, EMA, clip, AMP,
batch y presupuesto. La CLI solo la sobreescribe, y cada override queda anotado como desviación.

  python -m src.train --kind gpt --size s --run gpt_s_scratch --presupuesto 1x
  python -m src.train --kind gpt --size s --run gpt_s_kd --tokens 100e6 --kd
  python -m src.train --kind gpt --size s --run x --set receta.betas=[0.9,0.99] --set receta.extras.dropout=0.1
  python -m src.train --extend-from gpt_s_scratch --presupuesto 2x      # continúa en <run>_x2
  python -m src.status --watch                                          # progreso de las corridas

Presupuesto: frames vistos (780k frames = 100M tokens a 129 tokens/frame); --tokens es un alias.
Los datos dependen solo de (seed, step): scratch y KD con el mismo presupuesto/seed/batch ven los mismos
batches. Reanuda solo desde <ckpt-dir>/<run>/last.pt (Colab corta las sesiones).

Continuar: en recetas warmup -> constante -> decay se guarda pre_decay.pt al llegar al decay; --extend-from
retoma ahí, sigue la meseta hasta el nuevo inicio de decay y decae al final (equivale a entrenar el
presupuesto grande de una vez, con warmup de largo absoluto). Con coseno hace falta --restart (no equivalente).
"""
import argparse
import copy
import dataclasses
import json
import math
import os
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from src import recipes as R
from src import schedules as S
from src.common import TPF, amp_dtype, device, device_info, log_device
from src.models import KINDS, build, load_model, model_class
from src.status import write_status
from src.windows import Windows


def parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--kind", choices=list(KINDS))
  p.add_argument("--size", choices=["s", "m", "l"])
  p.add_argument("--config", default=None, help="JSON con campos de Config, p. ej. '{\"dropout\":0.1}'")
  p.add_argument("--run")
  p.add_argument("--receta", help=f"receta con nombre en vez de la del modelo ({', '.join(R.NAMED)})")
  p.add_argument("--set", action="append", default=[], metavar="receta.campo=valor", help="override genérico de la receta (JSON)")
  p.add_argument("--presupuesto", help="frames vistos, o Nx = N veces el presupuesto de la receta (default 780k frames)")
  p.add_argument("--tokens", type=float, help="alias de --presupuesto en tokens (129 por frame)")
  p.add_argument("--extend-from", help="corrida a continuar con más presupuesto (crea <corrida>_x<k>)")
  p.add_argument("--init-from", help="ajuste fino: arranca con los pesos de esa corrida (last.pt) y optimizador/schedule "
                 "nuevos; los módulos que la corrida no tenía quedan con su init (deben estar en cero para no cambiarla)")
  p.add_argument("--restart", action="store_true", help="con --extend-from y receta coseno: re-warmup y coseno nuevo (no equivalente)")
  p.add_argument("--bs", type=int)
  p.add_argument("--accum", type=int)
  p.add_argument("--frames", type=int)
  p.add_argument("--frameskip", type=int)
  p.add_argument("--lr", type=float, help="lr pico")
  p.add_argument("--min-lr-frac", type=float)
  p.add_argument("--warmup", type=int, help="pasos de warmup")
  p.add_argument("--wd", type=float)
  p.add_argument("--clip", type=float)
  p.add_argument("--amp", choices=["fp16", "bf16", "none"], default=None, help="default: el de la receta, solo en CUDA")
  p.add_argument("--seed", type=int)
  p.add_argument("--kd", action="store_true")
  p.add_argument("--alpha", type=float)
  p.add_argument("--tau", type=float)
  p.add_argument("--data")
  p.add_argument("--seq-maestro", help="secuencias generadas por el maestro (src/maestro_seq.py): una fracción del batch "
                 "se reemplaza por ventanas de ellas (destilación por secuencias)")
  p.add_argument("--seq-frac", type=float, default=0.5, help="fracción del batch de --seq-maestro")
  p.add_argument("--poses", help="poses por frame (solo modelos con uses_poses)")
  p.add_argument("--ckpt-dir", default="checkpoints")
  p.add_argument("--ckpt-every-min", type=float, default=15)
  p.add_argument("--eval-every", type=int, default=250)
  p.add_argument("--log-every", type=int, default=25)
  p.add_argument("--compile", action="store_true")
  p.add_argument("--max-steps", type=int)
  p.add_argument("--results-dir", default="results")
  a = p.parse_args(argv)
  if not a.extend_from and not (a.kind and a.run):
    p.error("--kind y --run son obligatorios (salvo con --extend-from)")
  if a.presupuesto and a.tokens:
    p.error("--presupuesto y --tokens son lo mismo: usá uno")
  if a.restart and not a.extend_from:
    p.error("--restart solo con --extend-from")
  return a


# ---------------------------------------------------------------- receta y presupuesto

def budget_frames(a, recipe):
  """Frames del presupuesto pedido (float): --presupuesto, --tokens o el default de la receta."""
  if a.presupuesto:
    t = str(a.presupuesto).strip().lower()
    return float(t[:-1]) * recipe.presupuesto if t.endswith("x") else float(t)
  if a.tokens:
    return a.tokens / TPF
  return float(recipe.presupuesto)


def total_steps(a, recipe):
  per_step = recipe.batch * recipe.accum * recipe.frames
  if a.tokens and not a.presupuesto:  # exacto como antes: ceil(tokens / tokens-por-paso)
    return math.ceil(a.tokens / (per_step * TPF))
  return math.ceil(budget_frames(a, recipe) / per_step)


def apply_overrides(r, a, desv):
  """Aplica los flags de la CLI y --set a la receta; lo que cambia queda en `desv` (lista de dicts)."""
  desv = list(desv)

  def put(campo, viejo, nuevo, fn):
    nonlocal r
    if nuevo is None or nuevo == viejo:
      return
    r = fn(r)
    desv.append({"campo": campo, "receta": viejo, "valor": nuevo})

  put("batch", r.batch, a.bs, lambda x: x.replace(batch=a.bs))
  put("accum", r.accum, a.accum, lambda x: x.replace(accum=a.accum))
  put("frames", r.frames, a.frames, lambda x: x.replace(frames=a.frames))
  put("frameskip", r.frameskip, a.frameskip, lambda x: x.replace(frameskip=a.frameskip))
  put("lr", S.peak(r.lr_segs), a.lr, lambda x: x.con_pico(a.lr))
  warm = next((s for s in r.lr_segs if s.get("nombre") == "warmup"), None)
  put("warmup", warm.get("pasos") if warm else None, a.warmup, lambda x: x.con_warmup(a.warmup))
  cos = [s for s in r.lr_segs if s["forma"] == "cosine"]
  put("min_lr_frac", cos[-1]["min_frac"] if cos else None, a.min_lr_frac, lambda x: x.con_min_frac(a.min_lr_frac))
  put("wd", r.wd_ini if r.wd_fin is None else (r.wd_ini, r.wd_fin), a.wd, lambda x: x.replace(wd_ini=a.wd, wd_fin=None))
  put("grad_clip", r.grad_clip, a.clip, lambda x: x.replace(grad_clip=a.clip))
  if a.amp is not None:
    nuevo = None if a.amp == "none" else a.amp
    put("amp", r.amp, nuevo, lambda x: x.replace(amp=nuevo))
  for item in a.set:
    k, eq, v = item.partition("=")
    if not eq or not k.startswith("receta."):
      raise SystemExit(f"--set {item}: usá receta.campo=valor")
    campo, valor = k[len("receta."):], R.parse_value(v)
    nuevo = R.apply_override(r, campo, valor)
    viejo = (r.extras.get(campo[7:]) if campo.startswith("extras.") else getattr(r, campo))
    if nuevo.to_dict() != r.to_dict():
      desv.append({"campo": "receta." + campo, "receta": viejo, "valor": valor})
      r = nuevo
  return r, desv


# ---------------------------------------------------------------- optimizador y EMA

def mezclar_seq(x, seq_t, a, micro, n_frames):
  """Reemplaza los últimos round(frac·B) ejemplos del batch por ventanas de n_frames de las secuencias del maestro.
  El offset va de 0 a L - n_frames: con 19 frames reales de contexto y G = 20 generados, la ventana trae entre 1 y 20
  frames generados. Depende solo de (seed, micro-paso), así que es reproducible al reanudar."""
  B = x.shape[0]
  k = int(round(a.seq_frac * B))
  if k == 0:
    return x
  g = torch.Generator().manual_seed(a.seed * 7_919 + micro)
  idx = torch.randint(0, seq_t.shape[0], (k,), generator=g)
  off = torch.randint(0, seq_t.shape[1] - n_frames + 1, (k,), generator=g)
  w = torch.stack([seq_t[j, o:o + n_frames] for j, o in zip(idx.tolist(), off.tolist())]).to(x.dtype)
  return torch.cat([x[:B - k], w], 0)


def make_optimizer(model, r, dev, lr0):
  params = [p for p in model.parameters() if p.requires_grad]
  if r.wd_on == "matrices":  # weight decay en todo lo de >= 2 dimensiones (matrices, convs, embeddings); no en biases ni normas
    groups = [{"params": [p for p in params if p.ndim >= 2], "weight_decay": r.wd_ini},
              {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}]
  elif r.wd_on == "todos":
    groups = [{"params": params, "weight_decay": r.wd_ini}]
  elif r.wd_on == "lineales":  # IRIS: solo el weight de nn.Linear y nn.Conv1d (no biases, normas ni embeddings, tampoco la posición)
    lin = {id(m.weight) for m in model.modules() if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d))}
    groups = [{"params": [p for p in params if id(p) in lin], "weight_decay": r.wd_ini},
              {"params": [p for p in params if id(p) not in lin], "weight_decay": 0.0}]
  else:
    raise ValueError(f"wd_on no soportado: {r.wd_on}")
  if r.optimizer == "adamw":
    return torch.optim.AdamW(groups, lr=lr0, betas=tuple(r.betas), eps=r.eps, fused=dev.type == "cuda")
  if r.optimizer == "adam":
    return torch.optim.Adam(groups, lr=lr0, betas=tuple(r.betas), eps=r.eps)
  if r.optimizer == "rmsprop":
    return torch.optim.RMSprop(groups, lr=lr0, alpha=r.alpha, eps=r.eps, momentum=r.momentum)
  raise ValueError(f"optimizador no soportado: {r.optimizer}")


def ema_init(model):
  return {n: p.detach().clone().float() for n, p in model.named_parameters() if p.requires_grad}


@torch.no_grad()
def ema_step(ema, model, m):
  for n, p in model.named_parameters():
    if n in ema:
      ema[n].mul_(m).add_(p.detach().float(), alpha=1 - m)


def ema_state_dict(model, ema):
  sd = {k: v.clone() if torch.is_tensor(v) else copy.deepcopy(v) for k, v in model.state_dict().items()}  # v: estado extra
  for n, v in ema.items():
    sd[n] = v.to(sd[n].dtype)
  return sd


def atomic_save(obj, path):
  tmp = str(path) + ".tmp"
  torch.save(obj, tmp)
  os.replace(tmp, path)


@torch.no_grad()
def monitor_ce(model, windows, bs, autocast, dev, frameskip=1, n_frames=20):
  """CE (sin maestro) sobre las ventanas fijas del holdout, del mismo largo que las de entrenamiento (n_frames de la
  receta: Genie y V-JEPA entrenan con 16, LeWM con 4). No altera el RNG del entrenamiento."""
  rng = torch.get_rng_state()
  cuda_rng = torch.cuda.get_rng_state_all() if dev.type == "cuda" else None
  mps_rng = torch.mps.get_rng_state() if dev.type == "mps" else None
  torch.manual_seed(1234)
  was_training = model.training
  model.eval()
  x, n = windows.monitor(n_frames=n_frames, frameskip=frameskip).to(dev), 0  # ya residente en CUDA: .to es no-op
  tot = torch.zeros((), device=dev, dtype=torch.float32)  # acumula en el device: un solo sync al final
  for i in range(0, len(x), bs):
    xb = x[i:i + bs]
    with autocast():
      loss, met = model.train_loss(xb)
    # los modelos latentes (lewm) suman otras pérdidas: se monitorea la CE de su lectura a tokens
    tot += met.get("ce", loss).detach().float() * len(xb)
    n += len(xb)
  tot = tot.item()
  model.train(was_training)
  torch.set_rng_state(rng)
  if cuda_rng is not None:
    torch.cuda.set_rng_state_all(cuda_rng)
  if mps_rng is not None:
    torch.mps.set_rng_state(mps_rng)
  return tot / n


# ---------------------------------------------------------------- continuar una corrida

def load_source(a):
  """Estado de la corrida a continuar: (checkpoint, archivo, sha256, receta). Elige pre_decay.pt o last.pt."""
  from src import store
  d = Path(a.ckpt_dir) / a.extend_from
  last = d / "last.pt"
  if not last.exists():
    raise SystemExit(f"--extend-from {a.extend_from}: no existe {last}")
  ck = torch.load(last, map_location="cpu", weights_only=False)
  r = R.Recipe.from_dict(ck["recipe"]) if "recipe" in ck else R.from_legacy_args(ck["args"])
  kind = r.kind_continuacion()
  if kind == "coseno" and not a.restart:
    raise SystemExit(f"--extend-from {a.extend_from}: la receta {r.name} decae en coseno sin meseta; continuarla no "
                     "equivale a entrenar más: agregá --restart (re-warmup + coseno nuevo, queda marcada no equivalente)")
  path = last
  if kind == "wcd" and not a.restart:
    path = d / "pre_decay.pt"
    if not path.exists():
      raise SystemExit(f"--extend-from {a.extend_from}: falta {path} (la corrida no llegó a su decay con esta versión)")
    del ck
    ck = torch.load(path, map_location="cpu", weights_only=False)
  return ck, path, store.sha256(path), r


def extension_name(a, r):
  b = budget_frames(a, r)
  k = float(str(a.presupuesto).strip().lower()[:-1]) if a.presupuesto and str(a.presupuesto).strip().lower().endswith("x") \
    else round(b / r.presupuesto, 2)
  return f"{a.extend_from}_x{k:g}"


def main(argv=None):
  a = parse(argv)
  if a.extend_from:  # los valores sin indicar salen de la corrida madre
    # el nombre depende del presupuesto relativo a la receta madre: se lee su last.pt (receta + args)
    probe = torch.load(Path(a.ckpt_dir) / a.extend_from / "last.pt", map_location="cpu", weights_only=False) \
      if (Path(a.ckpt_dir) / a.extend_from / "last.pt").exists() else None
    if probe is None:
      raise SystemExit(f"--extend-from {a.extend_from}: no existe checkpoints/{a.extend_from}/last.pt")
    r0 = R.Recipe.from_dict(probe["recipe"]) if "recipe" in probe else R.from_legacy_args(probe["args"])
    a.run = a.run or extension_name(a, r0)
    pa = probe["args"]
    a.kind = a.kind or probe["kind"]
    a.size = a.size or pa.get("size")
    a.config = a.config or json.dumps(probe["config"])
    a.seed = pa.get("seed", 0) if a.seed is None else a.seed
    a.data, a.poses = a.data or pa.get("data"), a.poses or pa.get("poses")
    a.kd, a.alpha, a.tau = a.kd or pa.get("kd", False), a.alpha or pa.get("alpha"), a.tau or pa.get("tau")
    del probe
  a.config = a.config or "{}"
  a.seed = 0 if a.seed is None else a.seed
  a.data = a.data or "data/train_tokens.npy"
  a.poses = a.poses or "data/train_poses.npy"
  a.alpha = 0.5 if a.alpha is None else a.alpha
  a.tau = 1.0 if a.tau is None else a.tau
  dev = device()
  log_device(dev)
  run_dir = Path(a.ckpt_dir) / a.run
  run_dir.mkdir(parents=True, exist_ok=True)
  last_path = run_dir / "last.pt"
  try:
    return _train(a, dev, run_dir, last_path)
  except BaseException as e:
    msg = "interrumpida" if isinstance(e, KeyboardInterrupt) else f"{type(e).__name__}: {e}"
    prev = {}
    try:
      prev = json.loads((run_dir / "status.json").read_text())
    except (OSError, json.JSONDecodeError):
      pass
    write_status(run_dir, **{**prev, "estado": "fallada", "error": msg, "run": a.run, "kind": a.kind,
                             "hora_txt": datetime.now().isoformat(timespec="seconds")})
    raise


def _train(a, dev, run_dir, last_path):
  cls = model_class(a.kind)
  src = own = None
  if last_path.exists():
    own = torch.load(last_path, map_location="cpu", weights_only=False)
  elif a.extend_from:
    src = load_source(a)

  # ---- receta: la guardada si se reanuda, la de la corrida madre si se continúa, o la del modelo
  desv, sched0, restart_pending = [], 0, False
  if own is not None and "recipe" in own:
    r, desv, sched0 = R.Recipe.from_dict(own["recipe"]), list(own["args"].get("desviaciones", [])), own.get("sched0", 0)
    for k in ("parent", "parent_sha256", "parent_file", "continuacion_no_equivalente"):
      if own["args"].get(k) is not None:
        setattr(a, k, own["args"][k])
  elif src is not None:
    sck, spath, ssha, r = src
    if not a.restart and r.kind_continuacion() == "wcd":
      r = r.congelar_warmup(sck.get("total", 0))
    a.parent, a.parent_sha256, a.parent_file = a.extend_from, ssha, spath.name
    a.continuacion_no_equivalente = bool(a.restart)
    restart_pending = a.restart
  elif a.receta:
    if a.receta not in R.NAMED:
      raise SystemExit(f"--receta {a.receta}: no existe ({', '.join(R.NAMED)})")
    r = R.NAMED[a.receta]()
  else:
    r = cls.recipe(a.size)
  r, desv = apply_overrides(r, a, desv)
  total = total_steps(a, r)

  if src is not None and own is None:
    s0 = src[0]["step"]
    if total <= s0:
      raise SystemExit(f"--extend-from: el presupuesto nuevo ({total} pasos) no supera lo ya entrenado ({s0})")
    if restart_pending:  # re-warmup a la mitad del pico y coseno nuevo sobre el presupuesto extra
      warm = next((L for s, (_, L) in zip(r.lr_segs, S.lengths(r.lr_segs, src[0].get("total") or total))
                   if s.get("nombre") == "warmup"), 100)
      r, sched0 = r.reinicio(total - s0, warm or 100), s0
    elif r.kind_continuacion() == "wcd" and (r.decay_start(total) or 0) < s0:
      raise SystemExit(f"--extend-from: con {total} pasos el decay empezaría en {r.decay_start(total)}, antes del punto de "
                       f"retoma ({s0}); pedí más presupuesto")
  stop_at = min(total, a.max_steps) if a.max_steps else total
  lt = total - sched0  # largo del schedule (en el paso local step - sched0)
  a.desviaciones, a.receta_hash, a.receta_nombre = desv, r.hash(), r.name
  frames_step = r.batch * r.accum * r.frames
  tok_per_step = frames_step * TPF
  ds = r.decay_start(lt)
  guarda_pre = r.kind_continuacion() == "wcd" and ds is not None and ds < lt and not a.restart

  torch.manual_seed(a.seed)
  cfg = json.loads(a.config)
  if getattr(cls, "uses_poses", False) and r.frameskip > 1:
    cfg.setdefault("n_pose", 6 * r.frameskip)  # la acción apila las k poses crudas de cada frame tomado
  model = build(a.kind, cfg, a.size).to(dev)
  if a.init_from and own is None and src is None:
    ip = Path(a.ckpt_dir) / a.init_from / "last.pt"
    if not ip.exists():  # corridas viejas (gpt_s_kd) solo guardaron final.pt
      ip = ip.with_name("final.pt")
    ick = torch.load(ip, map_location="cpu", weights_only=False)
    falta, sobra = model.load_state_dict(ick["model"], strict=False)
    if sobra:
      raise SystemExit(f"--init-from {a.init_from}: el checkpoint tiene pesos que este modelo no usa: {sobra[:5]}")
    from src import store
    a.parent, a.parent_sha256, a.parent_file = a.init_from, store.sha256(ip), "last.pt (init, ajuste fino)"
    print(f"ajuste fino desde {ip} (paso {ick.get('step')}); nuevos: {sorted({k.split('.')[0] for k in falta})}", flush=True)
    del ick
  # num_params(): solo lo que se usa en inferencia (los latentes no cuentan su encoder target EMA)
  n_params = model.num_params() if hasattr(model, "num_params") else sum(p.numel() for p in model.parameters())
  use_poses = getattr(model, "uses_poses", False)
  windows = Windows(a.data, seed=a.seed, poses=a.poses if use_poses else None, device=dev)  # en CUDA: tokens/poses residentes
  if use_poses:  # normalización de las acciones; viaja en el checkpoint como buffer
    model.set_pose_stats(*windows.pose_stats(r.frameskip))
  opt = make_optimizer(model, r, dev, r.lr(0, lt))
  # --amp manda en cualquier device; la receta solo se aplica en CUDA (como siempre: fp16 en CUDA, nada en el resto)
  amp = a.amp if a.amp is not None else ((r.amp or "none") if dev.type in ("cuda", "mps") else "none")
  adtype, nota_amp = amp_dtype(dev, amp)   # MPS: bf16 solo si el sistema lo soporta; si no, fp32 declarado
  if nota_amp:
    print(f"aviso: {nota_amp}", flush=True)
    amp = "none"
  # El scaler solo existe con fp16 (bf16/fp32 no lo necesitan: deshabilitado, scale/unscale_/step/update son no-ops
  # y step() va directo a opt.step()). Con fp16, scaler.step() hace un sync por paso que PyTorch exige: lee
  # found_inf (.item()) para decidir si saltea el paso; clip_grad_norm_ no sincroniza (error_if_nonfinite=False).
  scaler = torch.amp.GradScaler(dev.type, enabled=amp == "fp16" and dev.type == "cuda")
  autocast = (lambda: torch.autocast(dev.type, dtype=adtype)) if adtype else (lambda: torch.autocast(dev.type, enabled=False))
  ema = ema_init(model) if r.ema_segs is not None else None

  step, tokens, elapsed, history = 0, 0, 0.0, []

  def restore(ck):
    nonlocal step, tokens, elapsed, history, ema
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    scaler.load_state_dict(ck["scaler"])
    step, tokens, elapsed, history = ck["step"], ck["tokens"], ck["elapsed"], ck["history"]
    if ema is not None and ck.get("ema") is not None:
      ema = {k: v.to(dev) for k, v in ck["ema"].items()}

  if own is not None:
    restore(own)
    print(f"reanudando {last_path} en el paso {step}/{total}", flush=True)
    own_recipe_hash = own.get("recipe_hash")
    if own_recipe_hash and own_recipe_hash != r.hash():
      warnings.warn(f"la receta cambió desde el checkpoint ({own_recipe_hash} -> {r.hash()}): sigo con la nueva")
    del own
  elif src is not None:
    restore(src[0])
    print(f"continuando {a.extend_from} ({src[1].name}, paso {step}) hasta {total} pasos"
          + (" [restart: no equivalente]" if a.restart else ""), flush=True)
    src = None
  print(f"{a.kind} {a.size or ''} {n_params / 1e6:.2f}M params, {total} pasos, {tok_per_step} tokens/paso, amp={amp}, "
        f"receta {r.name} [{r.hash()}]" + (f", {len(desv)} desviaciones" if desv else ""), flush=True)

  teacher = None
  if a.kd:
    teacher = load_model("gpt2m", dev).to(torch.float16).eval()
    for p in teacher.parameters():
      p.requires_grad_(False)

  loss_fn = model.train_loss
  if a.compile:
    try:
      loss_fn = torch.compile(model.train_loss)
    except Exception as e:
      warnings.warn(f"torch.compile falló, sigo en eager: {e}")
  log_f = open(run_dir / "log.jsonl", "a")

  def log(rec):
    log_f.write(json.dumps(rec) + "\n")
    log_f.flush()

  t0 = time.time()  # inicio de esta sesión; elapsed acumula las anteriores
  last_loss = None

  def status(estado, step_rate=None):
    now = time.time()
    eta = 0.0 if estado == "lista" else (max(0, stop_at - step) * step_rate if step_rate else None)
    write_status(run_dir, estado=estado, paso=step, total=total, porcentaje=100.0 * step / max(1, total), loss=last_loss,
                 lr=r.lr(max(0, step - 1 - sched0), lt), tramo=r.tramo(max(0, step - 1 - sched0), lt),
                 frames_s=(frames_step / step_rate) if step_rate else None, eta_s=eta,
                 eta_hora=(datetime.fromtimestamp(now) + timedelta(seconds=eta)).isoformat(timespec="seconds") if eta is not None else None,
                 pid=os.getpid(), hora=now, hora_txt=datetime.fromtimestamp(now).isoformat(timespec="seconds"),
                 run=a.run, kind=a.kind)

  def state():
    sd = {"kind": a.kind, "config": dataclasses.asdict(model.cfg), "model": model.state_dict(),
          "optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "step": step, "tokens": tokens,
          "frames": tokens // TPF, "args": vars(a), "name": a.run, "elapsed": elapsed + time.time() - t0,
          "history": history, "best": min((h["monitor_ce"] for h in history), default=None),
          "recipe": r.to_dict(), "recipe_hash": r.hash(), "total": total, "sched0": sched0, "pos_datos": step}
    if ema is not None:
      sd["ema"] = ema
    return sd

  def save_last():
    atomic_save(state(), last_path)

  def evaluate():
    ce = monitor_ce(model, windows, r.batch, autocast, dev, r.frameskip, r.frames)
    history.append({"step": step, "tokens": tokens, "monitor_ce": ce})
    log({"step": step, "tokens": tokens, "monitor_ce": ce})
    print(f"[eval] step {step} monitor_ce {ce:.4f}", flush=True)

  # índices (segmento, inicio) de todos los pasos que faltan, calculados una vez con el mismo RNG y subidos al device
  seq_t = None
  if a.seq_maestro:  # (N, 19 + G, 128) int16, residente
    seq_t = torch.from_numpy(np.load(a.seq_maestro)).to(dev)
    print(f"secuencias del maestro: {tuple(seq_t.shape)}, fracción {a.seq_frac}", flush=True)
  windows.prepare((stop_at - step) * r.accum, r.batch, r.frames, r.frameskip, first_step=step * r.accum)
  model.train()
  status("en curso")
  t_log, tok_log, step_log, t_ckpt = t0, tokens, step, t0
  acc, n_acc = {}, 0
  while step < stop_at:
    if guarda_pre and step - sched0 == ds:  # entra al decay: este es el punto desde el que se puede continuar
      atomic_save({**state(), "decay_start": ds}, run_dir / "pre_decay.pt")
    torch.manual_seed(a.seed * 1_000_003 + step)  # dropout / máscaras de MaskGIT reproducibles al reanudar
    ls = step - sched0
    lr = r.lr(ls, lt)
    wd = r.wd(ls, lt)
    for i, g in enumerate(opt.param_groups):
      g["lr"] = lr
      if i == 0:  # grupo 0: lo que decae (los biases y normas, grupo 1, nunca)
        g["weight_decay"] = wd
    for i in range(r.accum):
      kw = dict(teacher=teacher, alpha=a.alpha, tau=a.tau)
      if use_poses:  # mismos tokens que sin poses: las poses salen de los mismos segmentos y frames
        x, p = windows.sample(step * r.accum + i, r.batch, r.frames, with_poses=True, frameskip=r.frameskip)
        kw["poses"] = p.to(dev, non_blocking=True)
      else:
        x = windows.sample(step * r.accum + i, r.batch, r.frames, frameskip=r.frameskip)
      x = x.to(dev, non_blocking=True)
      if seq_t is not None:
        x = mezclar_seq(x, seq_t, a, step * r.accum + i, r.frames)
      with autocast():
        try:
          loss, met = loss_fn(x, **kw)
        except Exception as e:
          if loss_fn is model.train_loss:
            raise
          warnings.warn(f"torch.compile falló al ejecutar, sigo en eager: {e}")
          loss_fn = model.train_loss
          loss, met = loss_fn(x, **kw)
      scaler.scale(loss / r.accum).backward()
      for k, v in {"loss": loss.detach(), **met}.items():
        acc[k] = acc.get(k, 0.0) + v.float() / r.accum
    n_acc += 1
    if r.grad_clip is not None:
      scaler.unscale_(opt)
      torch.nn.utils.clip_grad_norm_(model.parameters(), r.grad_clip)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    step += 1
    tokens += tok_per_step
    if ema is not None:
      ema_step(ema, model, r.ema(ls, lt))
    if hasattr(model, "after_step"):  # EMA del encoder target (los modelos latentes que la usen)
      model.after_step(step, total)

    if step % a.log_every == 0 or step == stop_at:
      now = time.time()
      rec = {"step": step, "tokens": tokens, "lr": lr, **{k: v.item() / n_acc for k, v in acc.items()},
             "tok_s": (tokens - tok_log) / max(now - t_log, 1e-9), "elapsed": elapsed + now - t0}
      acc, n_acc = {}, 0
      last_loss = rec.get("loss")
      print(" ".join(f"{k} {v:.4g}" for k, v in rec.items()), flush=True)
      log(rec)
      status("en curso", step_rate=(now - t_log) / max(1, step - step_log))
      t_log, tok_log, step_log = time.time(), tokens, step
    if step % a.eval_every == 0 and step < stop_at:
      evaluate()
      t_log, tok_log, step_log = time.time(), tokens, step
    if (time.time() - t_ckpt) / 60 >= a.ckpt_every_min:
      save_last()
      t_ckpt = time.time()

  if not history or history[-1]["step"] != step:
    evaluate()
  save_last()
  elapsed += time.time() - t0
  final_sd = ema_state_dict(model, ema) if ema is not None else model.state_dict()
  atomic_save({"kind": a.kind, "config": dataclasses.asdict(model.cfg), "model": final_sd,
               "name": a.run, "step": step, "tokens": tokens, "recipe": r.to_dict()}, run_dir / "final.pt")
  res = {"run": a.run, "kind": a.kind, "size": a.size, "config": dataclasses.asdict(model.cfg), "params": n_params,
         "tokens": tokens, "frames": tokens // TPF, "steps": step, "seconds": elapsed,
         "tokens_per_s": tokens / max(elapsed, 1e-9),
         "final_monitor_ce": history[-1]["monitor_ce"], "monitor_history": history, "args": vars(a),
         "receta": r.to_dict(), "receta_hash": r.hash(), "desviaciones": desv,
         "device_info": device_info(dev)}
  for k in ("parent", "parent_sha256", "parent_file", "continuacion_no_equivalente"):
    if getattr(a, k, None) is not None:
      res[k] = getattr(a, k)
  Path(a.results_dir).mkdir(parents=True, exist_ok=True)
  (Path(a.results_dir) / f"train_{a.run}.json").write_text(json.dumps(res, indent=1))
  status("lista")
  print(f"listo: {step} pasos, {tokens} tokens, {res['tokens_per_s']:.0f} tok/s, monitor CE {res['final_monitor_ce']:.4f}")
  return res


if __name__ == "__main__":
  main()
