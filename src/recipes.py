"""Receta de entrenamiento de un modelo: optimizador, schedules, batch y presupuesto, con su fuente.

train.py es genérico: toda la receta sale de `Clase.recipe(size)` (cada modelo define la suya, junto a su
código) más los overrides de la CLI, que quedan anotados como desviaciones. Lo que la receta de la fuente
no fija y hubo que elegir va en `forzado` con su motivo y categoría:
  datos        el dato de comma obliga (resolución, tokens VQ, 20 FPS...)
  presupuesto  el presupuesto de cómputo obliga (menos pasos, batch más chico...)
  propio       decisión del repo (no sale del paper)

  python -m src.recipes --table              tabla markdown de todos los modelos registrados
  python -m src.recipes gpt --size s         una receta (JSON + resumen + hash)
  python -m src.recipes --fidelidad genie    tabla de fidelidad (decisión, fuente, aquí, categoría) en markdown

Unidades: el presupuesto se mide en frames vistos (780k frames = 100M tokens a 129 tokens/frame).
"""
import argparse
import copy
import dataclasses
import hashlib
import json
import sys
from dataclasses import dataclass, field

from src import schedules as S

CATEGORIAS = ("datos", "presupuesto", "propio")
FID_CATEGORIAS = ("fiel", "datos", "presupuesto", "propio")   # las de las listas FIDELIDAD de cada modelo
FID_EXTRA = {"baselines": "src.models.baselines", "gpt2m": "src.models.gpt2m"}  # con FIDELIDAD pero fuera de KINDS
PRESUPUESTO_DEFAULT = 780_000  # frames (= 100M tokens)


@dataclass
class Recipe:
  name: str
  fuente: str
  optimizer: str = "adamw"            # "adamw" | "adam" | "rmsprop"
  betas: tuple = (0.9, 0.95)
  eps: float = 1e-8
  alpha: float = 0.99                 # rmsprop: factor del promedio de cuadrados
  momentum: float = 0.0               # rmsprop
  lr_segs: list = field(default_factory=list)   # tramos de src.schedules (valores absolutos de lr)
  wd_ini: float = 0.1                 # weight decay; con wd_fin es una rampa lineal wd_ini -> wd_fin
  wd_fin: float | None = None
  wd_on: str = "matrices"             # "matrices" (ndim >= 2) | "todos" | "lineales" (solo weight de Linear/Conv1d, como IRIS)
  ema_segs: list | None = None        # None sin EMA de pesos; si no, schedule del momento
  grad_clip: float | None = 1.0
  amp: str | None = "fp16"            # "fp16" | "bf16" | None; la receta solo se aplica en CUDA
  batch: int = 16
  frames: int = 20                    # frames (ya submuestreados) por muestra
  frameskip: int = 1                  # se toma 1 de cada k frames crudos
  accum: int = 1
  presupuesto: int = PRESUPUESTO_DEFAULT   # frames vistos del presupuesto de referencia
  extras: dict = field(default_factory=dict)  # label_smoothing, dropout...
  paper_steps: int | None = None      # pasos del paper (informativo)
  batch_fuente: int | None = None     # batch efectivo de la fuente en muestras (el humo escala el lr según batch/batch_fuente)
  forzado: list = field(default_factory=list)  # [(campo, motivo, categoría)]

  # ---- schedules
  def lr(self, step, total):
    return S.value(self.lr_segs, step, total)

  def wd(self, step, total):
    if self.wd_fin is None:
      return self.wd_ini
    return self.wd_ini + (self.wd_fin - self.wd_ini) * min(1.0, step / max(1, total))

  def ema(self, step, total):
    return None if self.ema_segs is None else S.value(self.ema_segs, step, total)

  def tramo(self, step, total):
    return S.segment_name(self.lr_segs, step, total)

  def decay_start(self, total):
    """Primer paso del último tramo de decaimiento; None si la forma no tiene."""
    return S.decay_start(self.lr_segs, total)

  def kind_continuacion(self):
    """'wcd' (warmup -> constante -> decay: se continúa desde pre_decay.pt), 'constante' (se continúa desde
    last.pt) o 'coseno' (decay sin meseta: solo con --restart)."""
    if not self.lr_segs or not S.is_decay(self.lr_segs[-1]):
      return "constante"
    return "wcd" if S.has_constant_before_decay(self.lr_segs) else "coseno"

  # ---- cambios (devuelven una receta nueva)
  def replace(self, **kw):
    return dataclasses.replace(copy.deepcopy(self), **kw)

  def con_pico(self, lr):
    """Cambia el lr pico; los valores de los tramos que lo igualan pasan a ser `lr` exacto, el resto escala."""
    old = S.peak(self.lr_segs)
    segs = copy.deepcopy(self.lr_segs)
    for s in segs:
      for k in ("ini", "fin"):
        if k in s:
          s[k] = lr if s[k] == old else (s[k] * lr / old if old else s[k])
    return self.replace(lr_segs=segs)

  def con_warmup(self, pasos):
    segs = copy.deepcopy(self.lr_segs)
    w = [s for s in segs if s.get("nombre") == "warmup"]
    if not w:
      raise ValueError("la receta no tiene tramo 'warmup'")
    w[0]["pasos"] = int(pasos)
    w[0].pop("frac", None)
    return self.replace(lr_segs=segs)

  def con_min_frac(self, m):
    segs = copy.deepcopy(self.lr_segs)
    c = [s for s in segs if s["forma"] == "cosine"]
    if not c:
      raise ValueError("la receta no tiene tramo cosine")
    c[-1]["min_frac"] = m
    return self.replace(lr_segs=segs)

  def congelar_warmup(self, total):
    """Warmup con largo absoluto (el que tendría con `total` pasos): imprescindible para continuar una corrida."""
    segs = copy.deepcopy(self.lr_segs)
    spans = S.lengths(segs, total)
    for s, (_, L) in zip(segs, spans):
      if s.get("nombre") == "warmup" and s.get("pasos") is None:
        s["pasos"] = L
        s.pop("frac", None)
    return self.replace(lr_segs=segs)

  def reinicio(self, pasos_extra, warm):
    """Receta de un restart: re-warmup a la mitad del pico y un coseno nuevo sobre el presupuesto extra,
    hasta el mismo lr final de la receta original."""
    pico = S.peak(self.lr_segs)
    fin = S.seg_end(self.lr_segs[-1])
    mitad = pico / 2
    warm = max(0, min(int(warm), pasos_extra // 2))
    return self.replace(lr_segs=[S.linear(0.0, mitad, pasos=warm, nombre="warmup"),
                                 S.cosine(mitad, min(1.0, fin / mitad) if mitad else 1.0, nombre="cosine")])

  # ---- serialización
  def to_dict(self):
    d = dataclasses.asdict(self)
    d["betas"] = list(self.betas)
    d["forzado"] = [list(f) for f in self.forzado]
    if d.get("batch_fuente") is None:  # no cambia el hash (ni el dict) de las recetas que no lo declaran
      d.pop("batch_fuente", None)
    return json.loads(json.dumps(d))

  @classmethod
  def from_dict(cls, d):
    d = dict(d)
    d["betas"] = tuple(d.get("betas", (0.9, 0.95)))
    d["forzado"] = [tuple(f) for f in d.get("forzado", [])]
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})

  def hash(self):
    return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]

  def resumen(self):
    segs = " | ".join(_seg_text(s) for s in self.lr_segs)
    return (f"{self.name} [{self.hash()}] {self.optimizer} betas={tuple(self.betas)} eps={self.eps:g} "
            f"lr: {segs} wd={_wd_text(self)} ema={'sí' if self.ema_segs else 'no'} clip={self.grad_clip} amp={self.amp} "
            f"batch={self.batch}x{self.frames}f/skip{self.frameskip}x{self.accum} presupuesto={self.presupuesto}f")


def _seg_text(s):
  largo = f"{s['pasos']}p" if s.get("pasos") is not None else (f"{s['frac']:g}" if s.get("frac") is not None else "resto")
  if s["forma"] == "constant":
    return f"{s['nombre']}: {s['ini']:g} ({largo})"
  if s["forma"] == "linear":
    return f"{s['nombre'] or 'lineal'}: {s['ini']:g}->{s['fin']:g} ({largo})"
  return f"{s['nombre']}: coseno {s['ini']:g}->{s['ini'] * s['min_frac']:g} ({largo})"


def _wd_text(r):
  return f"{r.wd_ini:g}" if r.wd_fin is None else f"{r.wd_ini:g}->{r.wd_fin:g}"


# ---------------------------------------------------------------- recetas con nombre

def legacy_cosine(lr=1e-3, warmup=200, min_lr_frac=0.1, wd=0.1, clip=1.0, bs=16, accum=1, frames=20,
                  amp="fp16", name="legacy_cosine", fuente="receta actual del repo", forzado=None):
  """Lo que hacía train.py antes de las recetas: AdamW (0.9, 0.95), wd solo en matrices, warmup lineal de
  `warmup` pasos y coseno hasta min_lr_frac * lr, clip, AMP fp16 en CUDA."""
  return Recipe(
    name=name, fuente=fuente, optimizer="adamw", betas=(0.9, 0.95), eps=1e-8,
    lr_segs=[S.linear(0.0, lr, pasos=warmup, nombre="warmup"), S.cosine(lr, min_lr_frac, nombre="cosine")],
    wd_ini=wd, wd_on="matrices", grad_clip=clip, amp=amp, batch=bs, frames=frames, accum=accum,
    forzado=[tuple(f) for f in (forzado if forzado is not None else
             [("toda la receta", "es la que usaba el repo antes de adoptar la del paper", "propio")])])


def from_legacy_args(args):
  """Receta de una corrida vieja a partir de los `args` de su checkpoint (lr, warmup, wd, clip, bs...)."""
  amp = args.get("amp")
  return legacy_cosine(lr=args["lr"], warmup=args["warmup"], min_lr_frac=args["min_lr_frac"], wd=args["wd"],
                       clip=args["clip"], bs=args["bs"], accum=args.get("accum", 1), frames=args.get("frames", 20),
                       amp=None if amp == "none" else (amp or "fp16"))


NAMED = {"legacy_cosine": legacy_cosine}


# ---------------------------------------------------------------- overrides

ALIAS = {"lr_peak": "con_pico", "warmup": "con_warmup", "min_lr_frac": "con_min_frac"}  # atajos: --set receta.lr_peak=3e-4


def apply_override(r, campo, valor):
  """campo: nombre de un campo de Recipe, un alias de ALIAS (lr_peak, warmup, min_lr_frac) o extras.<clave>. Devuelve una receta nueva."""
  if campo in ALIAS:
    return getattr(r, ALIAS[campo])(valor)
  if campo.startswith("extras."):
    ex = dict(r.extras)
    ex[campo[len("extras."):]] = valor
    return r.replace(extras=ex)
  names = {f.name for f in dataclasses.fields(Recipe)}
  if campo not in names:
    raise SystemExit(f"--set receta.{campo}: la receta no tiene ese campo ({', '.join(sorted(names | set(ALIAS)))})")
  if campo == "betas":
    valor = tuple(valor)
  if campo == "forzado":
    valor = [tuple(f) for f in valor]
  return r.replace(**{campo: valor})


def parse_value(text):
  try:
    return json.loads(text)
  except json.JSONDecodeError:
    return text


# ---------------------------------------------------------------- tabla / CLI

def registered():
  """[(etiqueta, Recipe)] de todos los modelos entrenables con receta."""
  from src.models import KINDS, SIZES, model_class
  out = []
  for kind in KINDS:
    cls = model_class(kind)
    if not hasattr(cls, "recipe"):
      continue
    sized = "n_layer" in {f.name for f in dataclasses.fields(cls.Config)}
    for size in (SIZES if sized else [None]):
      out.append((f"{kind}_{size}" if size else kind, cls.recipe(size)))
  return out


def table():
  rows = registered()
  head = ["modelo", "fuente", "opt", "betas", "eps", "lr", "wd", "wd_on", "ema", "clip", "amp", "batch x frames (skip) x accum",
          "presupuesto (frames)", "extras", "paper_steps"]
  lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
  for label, r in rows:
    ema = "sí" if r.ema_segs else "no"
    lines.append("| " + " | ".join(str(c).replace("|", "/") for c in [
      label, r.fuente, r.optimizer, tuple(r.betas), f"{r.eps:g}", "; ".join(_seg_text(s) for s in r.lr_segs),
      _wd_text(r), r.wd_on, ema, r.grad_clip, r.amp, f"{r.batch} x {r.frames} ({r.frameskip}) x {r.accum}",
      r.presupuesto, json.dumps(r.extras, ensure_ascii=False) if r.extras else "-", r.paper_steps or "-"]) + " |")
  lines += ["", "Valores forzados (campo, motivo, categoría):", ""]
  for label, r in rows:
    for campo, motivo, cat in r.forzado:
      lines.append(f"- {label}: {campo} - {motivo} ({cat})")
  return "\n".join(lines)


def fidelidad_kinds():
  from src.models import KINDS
  return list(KINDS) + list(FID_EXTRA)


def fidelidad(kind):
  """Lista FIDELIDAD de un kind registrado (o de 'baselines' / 'gpt2m')."""
  import importlib
  from src.models import KINDS
  mod = KINDS[kind][0] if kind in KINDS else FID_EXTRA[kind]
  return importlib.import_module(mod).FIDELIDAD


def fidelidad_md(kind):
  """Tabla markdown de la fidelidad de un modelo (la fuente del valor se llama 'paper' en algunos módulos)."""
  head = ["decisión", "fuente / paper", "aquí", "categoría", "fuente"]
  lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
  for f in fidelidad(kind):
    vals = [f["campo"], f.get("fuente_valor", f.get("paper")), f["aquí"], f["categoría"], f["fuente"]]
    lines.append("| " + " | ".join(str(v).replace("|", "/").replace(chr(10), " ") for v in vals) + " |")
  return chr(10).join(lines)


def main(argv=None):
  p = argparse.ArgumentParser(prog="python -m src.recipes", description=__doc__.split("\n")[0])
  p.add_argument("kind", nargs="?")
  p.add_argument("--size", choices=["s", "m", "l"])
  p.add_argument("--table", action="store_true")
  p.add_argument("--set", action="append", default=[], metavar="receta.campo=valor|config.campo=valor",
                 help="overrides (JSON): se aplican a la receta o a la Config y se muestran como desvíos")
  p.add_argument("--fidelidad", metavar="KIND", help="tabla de fidelidad en markdown (un kind, 'baselines' o 'gpt2m')")
  a = p.parse_args(argv)
  if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
  if a.table:
    print(table())
    return
  if a.fidelidad:
    if a.fidelidad not in fidelidad_kinds():
      p.error(f"--fidelidad: {a.fidelidad} no existe ({', '.join(fidelidad_kinds())})")
    print(fidelidad_md(a.fidelidad))
    return
  if not a.kind:
    p.error("indicá un kind o --table")
  from src.models import SIZES, model_class
  cls = model_class(a.kind)
  r = cls.recipe(a.size)
  cfg_ov, desv = {}, []
  for item in a.set:
    k, eq, v = item.partition("=")
    if not eq or k.split(".")[0] not in ("receta", "config"):
      p.error(f"--set {item}: usá receta.campo=valor o config.campo=valor")
    campo, valor = k.split(".", 1)[1], parse_value(v)
    if k.startswith("receta."):
      desv.append((k, valor))
      r = apply_override(r, campo, valor)
    else:
      campos = {f.name for f in dataclasses.fields(cls.Config)}
      if campo not in campos:
        p.error(f"--set {k}: la Config de {a.kind} no tiene ese campo ({', '.join(sorted(campos))})")
      desv.append((k, valor))
      cfg_ov[campo] = valor
  print(json.dumps(r.to_dict(), indent=1, ensure_ascii=False))
  print(r.resumen())
  sized = "n_layer" in {f.name for f in dataclasses.fields(cls.Config)}
  cfg = dataclasses.asdict(cls.Config(**{**(SIZES[a.size] if (a.size and sized) else {}), **cfg_ov}))
  print("config efectiva:", json.dumps(cfg, ensure_ascii=False))
  for k, v in desv:
    print(f"desvío: {k} = {v!r}")
  for campo, motivo, cat in r.forzado:
    print(f"forzado ({cat}): {campo} - {motivo}")


if __name__ == "__main__":
  main()
