"""Registro de modelos: un string -> un WorldModel listo para evaluar, hacer rollouts o medir.

Especificaciones que entiende load_model():
  "gpt2m"                 GPT de comma (referencia y maestro), data/raw/pytorch_model.bin
  "unigram", "transicion" baselines de conteo (se ajustan sobre train la primera vez); "transicion" es la
                          tabla de transición de primer orden por celda ("copy" es su nombre viejo, con aviso)
  "copy_mix", "congelado" mezcla copiar/unigrama: a temperatura 0 repite el último frame (frame congelado)
  "random:<kind>_<size>"  modelo sin entrenar, para medir latencia (p. ej. random:gpt_l, random:maskgit_m,
                          random:lewm). random:<corrida> (p. ej. random:gpt_s) arma la arquitectura de una
                          corrida de results/checkpoints.json sin bajar el checkpoint
  "<ruta>.pt"             checkpoint de src/train.py
  "<corrida>"             nombre de una corrida de results/checkpoints.json (p. ej. "gpt_s_kd"): si no está
                          en checkpoints/, src.store la baja del almacén compartido y verifica el sha256

Formato de checkpoint (lo escribe src/train.py):
  {"kind": "gpt" | "maskgit" | "lewm", "config": dict, "model": state_dict, "name": str, ...}
Cada clase entrenable tiene un dataclass Config y se construye con Clase(Config(**config)).
"""
import importlib
from pathlib import Path

import torch

from src.models.base import WorldModel

KINDS = {
  "gpt": ("src.models.gpt", "GPT"),
  "maskgit": ("src.models.maskgit", "MaskGIT"),
  "lewm": ("src.models.lewm", "LeWM"),      # LeWorldModel (un embedding por frame, SIGReg, AdaLN)
  "genie": ("src.models.genie", "Genie"),   # dinámica de Genie (ST-transformer + MaskGIT, sin acciones)
  "vjepa2": ("src.models.vjepa.pretrain", "VJEPA2"),       # V-JEPA 2: preentrenamiento autosupervisado (EMA, máscaras 3D)
  "vjepa2_ac": ("src.models.vjepa.action", "VJEPA2AC"),    # V-JEPA 2-AC: predictor con acciones sobre el encoder congelado
  "convlstm_px": ("src.models.convlstm_px", "ConvLSTMPx"),  # ConvLSTM de Shi 2015, píxeles -> píxeles
  "plantilla": ("src.models._plantilla", "Plantilla"),     # modelo mínimo para copiar al sumar uno (CONTRIBUTING)
}

# tamaños de los transformers propios (D3): capas, cabezas, ancho
SIZES = {
  "s": dict(n_layer=6, n_head=6, dim=384),    # ~12M
  "m": dict(n_layer=8, n_head=8, dim=512),    # ~27M
  "l": dict(n_layer=12, n_head=12, dim=768),  # ~88M
}


def model_class(kind: str):
  mod, cls = KINDS[kind]
  return getattr(importlib.import_module(mod), cls)


def build(kind: str, config: dict | None = None, size: str | None = None) -> WorldModel:
  """Modelo sin entrenar. size ("s", "m", "l") completa la config de los transformers."""
  cls = model_class(kind)
  cfg = dict(SIZES[size]) if size else {}
  cfg.update(config or {})
  return cls(cls.Config(**cfg))


def load_checkpoint(path, dev="cpu") -> WorldModel:
  ck = torch.load(path, map_location="cpu", weights_only=False)
  m = model_class(ck["kind"])(model_class(ck["kind"]).Config(**ck["config"]))
  m.load_state_dict(ck["model"])
  m.name = ck.get("name") or Path(path).parent.name
  return m.to(dev).eval()


def apply_opts(model: WorldModel, opts) -> WorldModel:
  """--opt clave=valor (rollouts, bench): cambia campos de model.cfg en inferencia, p. ej. rollout=latent."""
  import json
  for kv in opts or ():
    k, _, v = kv.partition("=")
    if not hasattr(model.cfg, k):
      raise SystemExit(f"--opt {k}: {type(model).__name__}.Config no tiene ese campo")
    try:
      v = json.loads(v)
    except json.JSONDecodeError:
      pass  # string sin comillas
    setattr(model.cfg, k, v)
  return model


def load_model(spec: str, dev="cpu") -> WorldModel:
  if spec == "gpt2m":
    from src.models.gpt2m import Gpt2m
    return Gpt2m.load(dev)
  if spec in ("unigram", "transicion", "copy", "copy_mix", "congelado"):
    from src.models.baselines import load_baseline
    return load_baseline(spec, dev)
  if spec.startswith("random:"):
    run = spec[len("random:"):]
    from src import store  # random:<corrida registrada>: la misma arquitectura (config del manifiesto), sin pesos
    man = store.load_manifest()
    if run in man:
      m = build(man[run]["kind"], man[run]["config"])
      m.name = spec.replace(":", "_")
      return m.to(dev).eval()
    kind, _, size = run.rpartition("_")
    if kind in ("gpt", "maskgit", "genie"):
      m = build(kind, size=size)
    else:  # random:lewm, random:convlstm_px
      m = build(run if run in KINDS else (kind if kind else size))
    m.name = spec.replace(":", "_")
    return m.to(dev).eval()
  if spec.endswith(".pt") or Path(spec).exists():
    return load_checkpoint(spec, dev)
  local = Path("checkpoints") / spec / "final.pt"
  from src import store  # nombre de una corrida registrada: se baja del almacén si no está local
  if spec in store.load_manifest():
    return load_checkpoint(store.ensure_local(spec), dev)
  if local.exists():  # corrida local todavía sin registrar
    return load_checkpoint(local, dev)
  raise ValueError(f"modelo no soportado: {spec}")
