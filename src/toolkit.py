"""Toolkit: una clase por modelo con todas las acciones del repo (entrenar, continuar, evaluar, bench, demo, exportar...).

Es la misma API para la CLI (`python -m src.cli <modelo> <acción> ...`, un argparse fino sobre esta clase) y para los
notebooks. Cada acción arma el argv de los scripts existentes (`python -m src.train ...`, `src.rollouts`, ...) y los corre
como subprocesos, con candado de GPU, cola y bitácora (src/ejecutor.py). Nada se ejecuta en el proceso del llamador.

  from src.toolkit import Toolkit
  tk = Toolkit("genie")
  tk.accion("entrenar", tamano="s", semilla=1, dry_run=True)        # imprime los comandos, no corre nada
  tk.entrenar(tamano="s", presupuesto="1x")                           # entrena; queda en results/bitacora.jsonl

Opciones comunes (los mismos nombres que la CLI, con `-` por `_`): run, tamano, semilla, variante, presupuesto, device,
limite, set, dry_run, esperar, encolar, permitir_sucio; y las propias de cada acción (ver OPCIONES). `extra` son los argumentos
que la CLI recibe después de `--` y que pasan tal cual al script envuelto.

Cada módulo de modelo declara `ACCIONES` y `EVALUACION` ("video" o "latente") junto a su FIDELIDAD. Las entradas que no son un
modelo entrenable (gpt2m, unigram, transicion, congelado, real, codec) están en ESPECIALES.
"""
import dataclasses
import importlib
import json
import re
from dataclasses import dataclass

from src import ejecutor, rutas
from src.ejecutor import Cmd, Plan
from src.infer import backends

PY = ejecutor.PY
ACCIONES_TODAS = ("receta", "entrenar", "continuar", "evaluar", "bench", "demo", "exportar", "estado")
GENERALES_SCRIPT = {"reporte": "src.report", "resultados": "src.resultados", "estadistica": "src.estadistica"}
ESPECIALES = {  # entradas que no son KINDS: spec de load_model -> acciones
  "gpt2m": ("evaluar", "bench"),
  "unigram": ("evaluar", "bench"),
  "transicion": ("evaluar", "bench"),
  "congelado": ("evaluar", "bench"),
  "real": ("evaluar",),
  "codec": ("bench", "exportar"),
}
CON_PASOS = ("maskgit", "genie")
FAMILIA_REF = {"gpt": "gpt_s_kd", "genie": "genie_s"}   # corrida cuyo barrido de temperatura (results/temperatura_<run>.json) elige la T de la familia
LEGACY = {"transicion": "copy", "congelado": "frozen"}    # nombres que tienen en disco los resultados viejos (--desde-cache)
LEGACY_CE = {"transicion": "copy", "congelado": "copy_mix"}
COLAPSO = ("vjepa2", "vjepa2_ac", "lewm")           # modelos a los que el humo les pasa --colapso (src.humo)
DTYPE_VIDEO = {"gpt": "fp16", "maskgit": "fp16", "genie": "fp16", "gpt2m": "fp16"}  # dtype de los rollouts de la tabla (en CUDA)
PASOS_HUMO = 300                           # pasos de cada entrenamiento del humo


def _multiplo(nombre):
  """'x30' -> 30.0 (candidato del humo: k veces el lr de la fuente); None si el nombre no tiene esa forma o k <= 0."""
  if not nombre.startswith("x"):
    return None
  try:
    k = float(nombre[1:])
  except ValueError:
    return None
  return k if k > 0 else None


class ToolkitError(Exception):
  """Pedido inválido (acción que el modelo no declara, --set con un campo que no existe...). La CLI lo muestra y sale con 2."""


@dataclass(frozen=True)
class Opcion:
  nombre: str                 # kwarg de Python
  flags: tuple                # la primera es la canónica
  tipo: str                   # "str" | "int" | "float" | "bool" | "lista"
  ayuda: str
  acciones: tuple | None = None   # solo estas acciones (None = todas)
  choices: tuple | None = None
  const: float | None = None  # float con nargs="?": valor si se pasa sin argumento


OPCIONES = [
  Opcion("run", ("--run",), "str", "nombre de la corrida (default: <modelo>_<tamaño>[_<variante>][_s<semilla>])"),
  Opcion("tamano", ("--tamaño", "--tamano"), "str", "tamaño del modelo (s, m, l; vjepa2: t o s)"),
  Opcion("semilla", ("--semilla",), "int", "semilla de entrenamiento (la de la corrida a evaluar si no es 0)"),
  Opcion("variante", ("--variante",), "str", "sufijo del nombre de la corrida (_<variante>)"),
  Opcion("presupuesto", ("--presupuesto",), "str", "frames vistos, o Nx = N veces el presupuesto de la receta",
         ("entrenar", "continuar", "receta")),
  Opcion("device", ("--device",), "str", "cuda, cpu, rocm, dml, npu o mps (Apple Silicon); default: el que elige el repo",
         choices=tuple(backends.nombres())),
  Opcion("limite", ("--limite",), "int", "recorta la acción: pasos (entrenar), segmentos/clips (evaluar), frames (bench, demo)"),
  Opcion("set", ("--set",), "lista", "receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)",
         ("receta", "entrenar", "continuar")),
  Opcion("dtype", ("--dtype",), "str", "precisión de la inferencia", ("evaluar", "bench", "demo", "exportar"),
         choices=("fp32", "fp16", "bf16")),
  Opcion("temperatura", ("--temperatura",), "float", "temperatura de muestreo", ("evaluar", "bench", "demo")),
  Opcion("pasos", ("--pasos",), "str", "pasos de decodificación, uno o varios separados por coma (maskgit, genie)",
         ("evaluar", "bench", "demo")),
  Opcion("ctx_frames", ("--ctx-frames",), "int", "frames de contexto en la inferencia (default 20 = 1 s)", ("evaluar", "bench")),
  Opcion("int8", ("--int8",), "bool", "cuantización int8 dinámica en CPU", ("bench",)),
  Opcion("aleatorio", ("--aleatorio",), "bool", "modelo sin entrenar con la arquitectura de la corrida (latencia)", ("bench", "demo", "exportar")),
  Opcion("humo", ("--humo",), "bool", "tres entrenamientos de 300 pasos (lr de la receta, por raíz y lineal) y elige", ("entrenar",)),
  Opcion("humo_candidatos", ("--humo-candidatos",), "str", "con --humo: candidatos a probar, p. ej. fuente,raiz,x3 (default: fuente,raiz,lineal,x3,x10)",
         ("entrenar",)),
  Opcion("destilar", ("--destilar",), "bool", "destilación del maestro gpt2m (--kd)", ("entrenar",)),
  Opcion("restart", ("--restart",), "bool", "continuar con re-warmup y coseno nuevo (no equivalente)", ("continuar",)),
  Opcion("encoder", ("--encoder",), "str", "corrida vjepa2 de la que sale el encoder congelado", ("entrenar", "continuar", "demo")),
  Opcion("barrer_temperatura", ("--barrer-temperatura",), "bool", "barrido de temperatura en el holdout", ("evaluar",)),
  Opcion("sin_plan", ("--no-plan",), "bool", "evaluación latente sin el plan con CEM", ("evaluar",)),
  Opcion("cadena", ("--cadena",), "bool", "bench: latencia de la cadena de punta a punta; evaluar: calidad en línea y con la cadena (src.infer.quality)",
         ("bench", "evaluar")),
  Opcion("desde_cache", ("--desde-cache",), "bool", "evaluar: no rehace los rollouts que ya existen; recalcula las métricas desde results/cache "
         "(nombres de los resultados viejos, T = 1)", ("evaluar",)),
  Opcion("forzar", ("--forzar",), "bool", "entrenar: seguir aunque la corrida ya tenga last.pt o final.pt (train.py la reanuda desde last.pt)",
         ("entrenar",)),
  Opcion("diagnostico", ("--diagnostico",), "bool", "con --cadena: el piso de ~20 ms a ritmo de cámara (bench_e2e --diagnostico)", ("bench",)),
  Opcion("clips", ("--clips",), "str", "set de clips de los rollouts (solo `real evaluar`: eval, holdout o eval_b)", ("evaluar",),
         choices=("eval", "holdout", "eval_b")),
  Opcion("modo", ("--modo",), "str", "con --cadena: online o imagine (default: los dos)", ("bench",), choices=("online", "imagine")),
  Opcion("ritmo", ("--ritmo",), "float", "con --cadena: milisegundos entre frames (0 = bucle cerrado)", ("bench",)),
  Opcion("tag", ("--tag",), "str", "nombre del dispositivo (radeon860m...) en los JSON, para no pisar los de otra máquina; con rocm es obligatorio "
         "en los benchs que escriben results/bench/", ("bench",)),
  Opcion("ort", ("--ort",), "bool", "codec bench: mide el codec exportado con ONNX Runtime (src.infer.bench_ort); con dml y npu es lo único que hay",
         ("bench",)),
  Opcion("entrenar_decoder", ("--entrenar-decoder",), "bool", "entrena el decoder de píxeles (encoder congelado) en vez del demo",
         ("demo",)),
  Opcion("vigilar", ("--vigilar",), "float", "estado: repite cada N segundos (default 10)", ("estado",), const=10.0),
  Opcion("dry_run", ("--dry-run",), "bool", "imprime los comandos que ejecutaría y no corre nada"),
  Opcion("esperar", ("--esperar",), "bool", "si la GPU está ocupada, espera en vez de encolar"),
  Opcion("encolar", ("--encolar",), "bool", "no correr ahora: dejar la acción en la cola"),
  Opcion("permitir_sucio", ("--permitir-sucio",), "bool", "ya no hace falta: una acción de GPU corre con cambios sin commitear (queda anotado; no cuenta como vigente)"),
]
POR_NOMBRE = {o.nombre: o for o in OPCIONES}
CONTROL = ("dry_run", "esperar", "encolar", "permitir_sucio")


# ---------------------------------------------------------------- registro

def kinds():
  from src.models import KINDS
  return list(KINDS)


def modelos():
  """Todos los nombres que acepta la CLI: modelos entrenables y entradas especiales."""
  return kinds() + list(ESPECIALES)


def _modulo(kind):
  from src.models import KINDS
  return importlib.import_module(KINDS[kind][0])


def acciones_de(modelo):
  """Acciones declaradas por el modelo. Falla con un mensaje claro si el módulo todavía no declara ACCIONES."""
  if modelo in ESPECIALES:
    return ESPECIALES[modelo]
  if modelo not in kinds():
    raise ToolkitError(f"el modelo {modelo!r} no existe; opciones: {', '.join(modelos())}")
  mod = _modulo(modelo)
  if not hasattr(mod, "ACCIONES"):
    raise ToolkitError(f"{mod.__name__} todavía no declara ACCIONES (ni EVALUACION): el modelo {modelo} no está en la CLI "
                       "hasta que lo haga (el paquete del modelo lo agrega junto a FIDELIDAD)")
  return tuple(mod.ACCIONES)


def evaluacion_de(modelo):
  if modelo in ESPECIALES:
    return "video"
  mod = _modulo(modelo)
  if not hasattr(mod, "EVALUACION"):
    raise ToolkitError(f"{mod.__name__} todavía no declara EVALUACION ('video' o 'latente')")
  return mod.EVALUACION


def catalogo():
  """{modelo: acciones} de todos; los que no declaran todavía figuran con una tupla vacía."""
  out = {}
  for m in modelos():
    try:
      out[m] = tuple(acciones_de(m)) + (("estado",) if m in kinds() else ())
    except ToolkitError:
      out[m] = ()
  return out


# ---------------------------------------------------------------- helpers de opciones

def _vacio(v):
  """None, False o lista vacía (0 y 0.0 son valores: --semilla 0, --ritmo 0)."""
  return v is None or v is False or (isinstance(v, list) and not v)


def _lista_csv(texto):
  try:
    return [int(x) for x in str(texto).replace(" ", "").split(",") if x]
  except ValueError:
    raise ToolkitError(f"--pasos {texto!r}: esperaba enteros separados por coma, p. ej. 1,2,4,8")


def argv_canonico(modelo, accion, opts, extra=()):
  """Los argumentos de `python -m src.cli` que equivalen a estas opciones (el que se anota en la bitácora desde un notebook)."""
  av = [modelo, accion]
  for o in OPCIONES:
    v = opts.get(o.nombre)
    if _vacio(v):
      continue
    if o.tipo == "bool":
      av.append(o.flags[0])
    elif o.tipo == "lista":
      for x in v:
        av += [o.flags[0], str(x)]
    else:
      av += [o.flags[0], str(v)]
  if extra:
    av += ["--", *extra]
  return av


def parsear_set(kind, sets, accion):
  """Valida --set contra Recipe y Config. Devuelve (sets_de_receta [str], dict de config). Un campo inválido lanza ToolkitError."""
  from src import recipes as R
  receta, config = [], {}
  for item in sets or []:
    k, eq, v = str(item).partition("=")
    if not eq or "." not in k or k.split(".")[0] not in ("receta", "config"):
      raise ToolkitError(f"--set {item}: usá receta.campo=valor o config.campo=valor")
    zona, campo = k.split(".", 1)
    if zona == "receta":
      campos = {f.name for f in dataclasses.fields(R.Recipe)} | set(R.ALIAS)
      if campo.split(".")[0] not in campos or (campo.startswith("extras") and not campo.startswith("extras.")):
        raise ToolkitError(f"--set {k}: la receta no tiene el campo {campo.split('.')[0]!r} (campos: {', '.join(sorted(campos))}; "
                           "extras.<clave> para los extras)")
      if "." in campo and not campo.startswith("extras."):
        raise ToolkitError(f"--set {k}: solo extras.<clave> admite un punto")
      receta.append(f"receta.{campo}={v}")
    else:
      if kind not in kinds():
        raise ToolkitError(f"--set {k}: {kind} no tiene Config")
      campos = {f.name for f in dataclasses.fields(_clase(kind).Config)}
      if campo not in campos:
        raise ToolkitError(f"--set {k}: la Config de {kind} no tiene el campo {campo!r} (campos: {', '.join(sorted(campos))})")
      config[campo] = R.parse_value(v)
  if (receta or config) and accion not in ("receta", "entrenar", "continuar"):
    raise ToolkitError(f"--set solo aplica a receta, entrenar y continuar (no a {accion})")
  return receta, config


ALIAS_TRAIN = {"lr_peak": "--lr", "warmup": "--warmup", "min_lr_frac": "--min-lr-frac"}  # los que train.py tiene como flag


def args_set(receta):
  """['receta.campo=valor'] -> argumentos de train.py: los alias de lr van como su flag (queda anotado como desvío igual)."""
  a = []
  for item in receta:
    campo, _, v = item[len("receta."):].partition("=")
    a += [ALIAS_TRAIN[campo], v] if campo in ALIAS_TRAIN else ["--set", item]
  return a


def _clase(kind):
  from src.models import model_class
  return model_class(kind)


def _es_sized(kind):
  return "n_layer" in {f.name for f in dataclasses.fields(_clase(kind).Config)}


def _validar_presupuesto(p):
  if p is not None and not re.fullmatch(r"\d+(\.\d+)?(e\d+)?x?", str(p).strip().lower()):
    raise ToolkitError(f"--presupuesto {p!r}: frames (780000, 780e3) o N veces el de la receta (10x)")


# ---------------------------------------------------------------- Toolkit

class Toolkit:
  """Todas las acciones de un modelo. `Toolkit("gpt").plan("entrenar", tamano="s")` solo arma el plan; `.accion(...)` lo ejecuta."""

  def __init__(self, modelo):
    if modelo not in modelos():
      raise ToolkitError(f"el modelo {modelo!r} no existe; opciones: {', '.join(modelos())}")
    self.modelo = modelo
    self._error = None
    self._meta = {}
    self._dev_forzado = None

  # ---- capacidades
  @property
  def acciones(self):
    return acciones_de(self.modelo)

  @property
  def evaluacion(self):
    return evaluacion_de(self.modelo)

  @property
  def es_kind(self):
    return self.modelo in kinds()

  def _validar(self, accion, opts):
    if accion not in ACCIONES_TODAS:
      raise ToolkitError(f"acción {accion!r} desconocida; opciones: {', '.join(ACCIONES_TODAS)}")
    if accion != "estado" or not self.es_kind:
      if accion not in self.acciones:
        raise ToolkitError(f"{self.modelo} no tiene la acción {accion!r}; declara: {', '.join(self.acciones)}")
    for nombre, v in opts.items():
      if nombre not in POR_NOMBRE:
        raise ToolkitError(f"opción desconocida: {nombre}")
      o = POR_NOMBRE[nombre]
      if _vacio(v):
        continue
      if o.acciones and accion not in o.acciones:
        raise ToolkitError(f"{o.flags[0]} solo aplica a {', '.join(o.acciones)} (no a {accion})")
      if o.choices and v not in o.choices:
        raise ToolkitError(f"{o.flags[0]} {v!r}: opciones {', '.join(o.choices)}")
    if opts.get("pasos") and self.modelo not in CON_PASOS:
      raise ToolkitError(f"--pasos: {self.modelo} no decodifica por pasos (solo {', '.join(CON_PASOS)})")
    _validar_presupuesto(opts.get("presupuesto"))
    if opts.get("tamano"):
      validos = ("t", "s") if self.modelo == "vjepa2" else (("s", "m", "l") if self.es_kind and _es_sized(self.modelo) else ())
      if opts["tamano"] not in validos:
        raise ToolkitError(f"--tamaño {opts['tamano']!r}: {self.modelo} " + (f"admite {', '.join(validos)}" if validos else "no tiene tamaños"))
    if opts.get("humo_candidatos") and not opts.get("humo"):
      raise ToolkitError("--humo-candidatos solo con --humo")
    if opts.get("humo") and (opts.get("run") or opts.get("limite")):
      raise ToolkitError("--humo fija sus corridas (<modelo>_humo_{fuente,raiz,lineal}) y sus 300 pasos: sin --run ni --limite")
    if opts.get("entrenar_decoder") and self.evaluacion != "latente":
      raise ToolkitError("--entrenar-decoder solo existe para los modelos latentes (lewm, vjepa2, vjepa2_ac)")
    if opts.get("encoder") and self.modelo != "vjepa2_ac":
      raise ToolkitError("--encoder solo aplica a vjepa2_ac")
    if opts.get("desde_cache") and (opts.get("cadena") or opts.get("barrer_temperatura")):
      raise ToolkitError("--desde-cache no va con --cadena ni --barrer-temperatura")
    if (opts.get("modo") or opts.get("ritmo") is not None or opts.get("diagnostico")) and not opts.get("cadena"):
      raise ToolkitError("--modo, --ritmo y --diagnostico solo con --cadena")
    if opts.get("clips") and self.modelo != "real":
      raise ToolkitError("--clips solo aplica a `real evaluar` (el piso de FVD)")
    if opts.get("ort") and self.modelo != "codec":
      raise ToolkitError("--ort solo aplica a `codec bench` (la cadena del codec con ONNX Runtime)")

  # ---- nombres
  def nombre_run(self, o):
    if o.get("run"):
      return o["run"]
    if not self.es_kind:
      return self.modelo
    m = self.modelo
    if m == "vjepa2":
      base = f"vjepa2_{o.get('tamano') or 's'}"
    elif _es_sized(m):
      base = f"{m}_{o.get('tamano') or 's'}"
    else:
      base = m
    if o.get("variante"):
      base += f"_{o['variante']}"
    if o.get("semilla"):
      base += f"_s{o['semilla']}"
    return base

  def _spec(self, o):
    run = self.nombre_run(o)
    return f"random:{run}" if o.get("aleatorio") else run

  # ---- receta
  def receta_efectiva(self, o):
    """Recipe del modelo con los --set de receta aplicados (para el hash y el humo)."""
    from src import recipes as R
    sets, _ = parsear_set(self.modelo, o.get("set"), "entrenar")
    r = _clase(self.modelo).recipe(self._tam(o))
    for s in sets:
      campo, _, v = s[len("receta."):].partition("=")
      r = R.apply_override(r, campo, R.parse_value(v))
    return r

  def _tam(self, o):
    """Tamaño que se le pasa a Clase.recipe y a train --size (s por defecto en los modelos con tamaños; None en el resto)."""
    return (o.get("tamano") or "s") if (self.es_kind and _es_sized(self.modelo)) else None

  def _config_json(self, o, config):
    cfg = {}
    if self.modelo == "vjepa2" and o.get("tamano"):
      cfg["encoder"] = {"t": "tiny", "s": "small"}[o["tamano"]]
    if o.get("encoder"):
      cfg["encoder_run"] = o["encoder"]
    cfg.update(config)
    return json.dumps(cfg, separators=(",", ":")) if cfg else None

  # ---- device
  def _device(self, accion, o):
    if accion in ("receta", "estado", "exportar"):
      return "cpu"
    return ejecutor.resolver_device(o.get("device"))

  @staticmethod
  def _env(o):
    d = o.get("device")
    if not d:
      return {}
    return {"ML2_DEVICE": _dev_torch(d)}

  # ---- plan
  def plan(self, accion, extra=(), cli_argv=None, **o):
    self._validar(accion, o)
    o = {k: v for k, v in o.items() if not _vacio(v)}
    avisos = []
    self._error, self._meta = None, {}
    cmds = getattr(self, f"_plan_{accion}")(o, list(extra or ()), avisos)
    error, self._error = self._error, None
    meta, self._meta = self._meta, {}
    dev_forzado, self._dev_forzado = self._dev_forzado, None
    run = self.nombre_run(o)
    gpu_dev = dev_forzado or self._device(accion, o)
    rh = None
    if accion in ("entrenar", "continuar", "receta") and self.es_kind and not o.get("humo"):
      try:
        rh = self.receta_efectiva(o).hash()
      except Exception:  # noqa: BLE001  (un --set inválido ya falló antes)
        rh = None
    return Plan(cmds, self.modelo, accion, run, gpu_dev, rh, cli_argv or argv_canonico(self.modelo, accion, o, extra), avisos,
                meta, error)

  def accion(self, accion, extra=(), cli_argv=None, desde_cola=False, salida=print, meta=None, **opts):
    """Planifica y ejecuta. dry_run=True imprime los comandos. Devuelve un ejecutor.Resultado."""
    ctrl = {k: opts.pop(k, False) for k in CONTROL}
    if ctrl["permitir_sucio"]:  # lo único de CONTROL que queda en el argv de la bitácora (para repetir tal cual)
      opts["permitir_sucio"] = True
    plan = self.plan(accion, extra=extra, cli_argv=cli_argv, **opts)
    if plan.error and not ctrl["dry_run"]:
      raise ToolkitError(plan.error)
    return ejecutor.ejecutar(plan, esperar=ctrl["esperar"], encolar=ctrl["encolar"], permitir_sucio=ctrl["permitir_sucio"],
                             dry_run=ctrl["dry_run"], desde_cola=desde_cola, salida=salida, meta=meta)

  # atajos: tk.entrenar(...) == tk.accion("entrenar", ...)
  def __getattr__(self, nombre):
    if nombre in ACCIONES_TODAS:
      return lambda **kw: self.accion(nombre, **kw)
    raise AttributeError(nombre)

  # ================================================================ acciones
  def _cmd(self, modulo, *args, o=None, env=True):
    return Cmd([PY, "-m", modulo, *[str(a) for a in args]], self._env(o or {}) if env else {})

  def _registrar(self, run, *extras):
    """Anota la corrida (y sus extras, p. ej. un decoder) en results/checkpoints.json y la sube al almacén del equipo
    (Hugging Face). Sin sesión de HF, o si la subida falla, el push avisa y sale con 0: nunca corta la cola."""
    reg = ["register", run] + [x for e in extras for x in ("--extra", e)]
    return [self._cmd("src.store", *reg, env=False), self._cmd("src.store", "push", run, "--si-hay-sesion", env=False)]

  def _plan_receta(self, o, extra, avisos):
    sets, config = parsear_set(self.modelo, o.get("set"), "receta")
    args = [self.modelo]
    if self._tam(o):
      args += ["--size", self._tam(o)]
    for item in o.get("set", []):
      args += ["--set", item]
    return [self._cmd("src.recipes", *args, *extra)]

  def _plan_estado(self, o, extra, avisos):
    args = ["--kind", self.modelo] + (["--watch", f"{o['vigilar']:g}"] if o.get("vigilar") else [])
    return [self._cmd("src.status", *args, *extra)]

  # ---- entrenar / continuar
  def _args_train(self, o, sets, config, run):
    a = ["--kind", self.modelo]
    if self._tam(o):
      a += ["--size", self._tam(o)]
    a += ["--run", run]
    if o.get("semilla") is not None:
      a += ["--seed", o["semilla"]]
    a += args_set(sets)
    cfg = self._config_json(o, config)
    if cfg:
      a += ["--config", cfg]
    if o.get("destilar"):
      a += ["--kd"]
    if o.get("limite"):
      a += ["--max-steps", o["limite"]]
    return a

  def _plan_entrenar(self, o, extra, avisos):
    sets, config = parsear_set(self.modelo, o.get("set"), "entrenar")
    if o.get("destilar") and self.modelo != "gpt":
      raise ToolkitError("--destilar solo existe para gpt (maestro gpt2m)")
    if o.get("humo"):
      if self.modelo == "vjepa2_ac" and not o.get("encoder") and "encoder_run" not in config:
        avisos.append("sin --encoder el humo usa un encoder aleatorio: sirve para elegir lr y batch, no para evaluar")
      return self._plan_humo(o, extra, sets, config, avisos)
    run = self.nombre_run(o)
    self._chequear_existente([run], o)
    if self.modelo == "vjepa2_ac" and not o.get("encoder") and "encoder_run" not in config:
      raise ToolkitError("vjepa2_ac entrenar necesita un encoder congelado: --encoder <corrida vjepa2> (o --set config.encoder_run=...)")
    a = self._args_train(o, sets, config, run)
    if o.get("presupuesto"):
      a += ["--presupuesto", o["presupuesto"]]
    cmds = [self._cmd("src.train", *a, *extra, o=o)]
    if not o.get("limite"):  # una corrida completa se anota en results/checkpoints.json (se carga después por nombre)
      cmds += self._registrar(run)
    return cmds

  def _chequear_existente(self, runs, o):
    """entrenar no pisa ni reanuda sin querer: si la corrida ya tiene last.pt o final.pt hace falta `continuar` o --forzar."""
    if o.get("forzar"):
      return
    for run in runs:
      d = rutas.ckpt() / run
      if (d / "last.pt").exists() or (d / "final.pt").exists():
        self._error = (f"la corrida {run} ya existe ({d}): `entrenar` no la pisa. Para retomarla usá `continuar --run {run}`; "
                       "para extenderla `continuar --presupuesto Nx`; para otra corrida cambiá --variante o --semilla; "
                       "--forzar deja pasar el pedido (train.py reanuda desde last.pt)")
        return

  def _plan_humo(self, o, extra, sets, config, avisos):
    from src import schedules as S
    r = self.receta_efectiva(o)
    pico = S.peak(r.lr_segs)
    batch = r.batch * r.accum
    frames = PASOS_HUMO * batch * r.frames
    cands = [("fuente", pico)]
    if r.batch_fuente:
      k = batch / r.batch_fuente
      cands += [("raiz", pico * k ** 0.5), ("lineal", pico * k)]
    else:
      avisos.append("la receta no declara batch_fuente: el humo no prueba los lr escalados por raíz ni lineal (solo fuente, x3 y x10)")
    cands += [("x3", pico * 3), ("x10", pico * 10)]
    if o.get("humo_candidatos"):
      pedidos = [c.strip() for c in o["humo_candidatos"].split(",") if c.strip()]
      # x<k>: k veces el lr de la fuente, con cualquier k > 0 (p. ej. x30 si el ganador quedó en el borde de la grilla)
      mult = {c: _multiplo(c) for c in pedidos if c not in ("fuente", "raiz", "lineal")}
      malos = [c for c, k in mult.items() if k is None] + [c for c in pedidos if c in ("raiz", "lineal") and not r.batch_fuente]
      if malos or "fuente" not in pedidos:
        raise ToolkitError(f"--humo-candidatos {o['humo_candidatos']!r}: opciones fuente, raiz, lineal y x<k> "
                           "(k veces el lr de la fuente, p. ej. x3, x10, x30); fuente siempre")
      base = dict(cands)
      cands = [(c, base[c] if c in base and c not in mult else pico * mult[c]) for c in pedidos]
    unicos, alias = [], {}
    for nombre, lr in cands:   # un solo entrenamiento por lr distinto
      igual = next((n for n, l in unicos if abs(l - lr) <= 1e-9 * abs(lr)), None)
      if igual:
        alias[nombre] = igual
      else:
        unicos.append((nombre, lr))
    for n, igual in alias.items():
      avisos.append(f"el lr de {n} es el de {igual}: se entrena una sola vez")
    runs = [f"{self.modelo}_humo_{n}" for n, _ in unicos]
    self._chequear_existente(runs, o)
    cmds, cand_args = [], []
    for (nombre, lr), run in zip(unicos, runs):
      a = self._args_train({**o, "semilla": o.get("semilla")}, sets, config, run)
      a += ["--presupuesto", frames]
      if nombre != "fuente":
        a += ["--lr", f"{lr:.6g}"]
      cmds.append(self._cmd("src.train", *a, *extra, o=o))
      cand_args += ["--cand", f"{nombre}:{run}:{lr:.6g}"]
    h = ["--kind", self.modelo, *cand_args, "--pasos", PASOS_HUMO, "--batch", batch]
    for n, igual in alias.items():
      h += ["--alias", f"{n}={igual}"]
    if r.batch_fuente:
      h += ["--batch-fuente", r.batch_fuente]
    if self.modelo in COLAPSO:
      h += ["--colapso"]
    cmds.append(self._cmd("src.humo", *h, env=False))
    return cmds

  def _guardado(self, run):
    """Los args y la receta que guardó la corrida en checkpoints/<run>/last.pt (None si no existe)."""
    f = rutas.ckpt() / run / "last.pt"
    if not f.exists():
      return None
    import torch
    ck = torch.load(f, map_location="cpu", weights_only=False)
    return {"args": ck.get("args", {}), "recipe": ck.get("recipe")}

  def _plan_continuar(self, o, extra, avisos):
    sets, config = parsear_set(self.modelo, o.get("set"), "continuar")
    run = self.nombre_run(o)
    g = self._guardado(run)
    if g is None:
      self._error = f"no existe {rutas.ckpt() / run / 'last.pt'}: no hay nada que continuar (entrená primero con `entrenar`)"
      avisos.append(self._error)
    from src import recipes as R
    rec = R.Recipe.from_dict(g["recipe"]) if (g and g["recipe"]) else self.receta_efectiva({k: v for k, v in o.items() if k != "set"})
    if o.get("presupuesto"):  # extender: <run>_x<k>, desde pre_decay.pt (o last.pt)
      a = ["--extend-from", run, "--presupuesto", o["presupuesto"]]
      restart = bool(o.get("restart"))
      if rec.kind_continuacion() == "coseno" and not restart:
        restart = True
        avisos.append(f"la receta {rec.name} decae en coseno sin meseta: continuar con más presupuesto no equivale a entrenar más; "
                      "uso --restart (re-warmup + coseno nuevo, queda marcada como no equivalente y fuera de la tabla principal)")
      if restart:
        a += ["--restart"]
      a += args_set(sets)
      cfg = self._config_json(o, config)
      if cfg:
        a += ["--config", cfg]
      if o.get("semilla") is not None:
        a += ["--seed", o["semilla"]]
      if o.get("limite"):
        a += ["--max-steps", o["limite"]]
      nueva = self._nombre_extension(run, o["presupuesto"], rec)
      cmds = [self._cmd("src.train", *a, *extra, o=o)]
      if not o.get("limite"):
        cmds += self._registrar(nueva)
      return cmds
    if o.get("restart"):
      raise ToolkitError("--restart necesita --presupuesto (es la forma de extender una receta coseno)")
    # retomar una corrida cortada: train.py reanuda solo desde last.pt, con la receta guardada, pero necesita
    # los mismos argumentos del presupuesto (kind, size, config, seed, presupuesto, kd, datos): salen de lo que guardó la corrida
    pa = (g or {}).get("args", {})
    a = ["--kind", pa.get("kind") or self.modelo]
    if pa.get("size"):
      a += ["--size", pa["size"]]
    elif g is None and self._tam(o):
      a += ["--size", self._tam(o)]
    a += ["--run", run]
    if pa.get("config") and pa["config"] != "{}":
      a += ["--config", pa["config"]]
    elif g is None and self._config_json(o, config):
      a += ["--config", self._config_json(o, config)]
    for k, flag in (("seed", "--seed"), ("presupuesto", "--presupuesto"), ("tokens", "--tokens"), ("alpha", "--alpha"),
                    ("tau", "--tau"), ("data", "--data"), ("poses", "--poses")):
      if pa.get(k) is not None and (k != "tokens" or not pa.get("presupuesto")):
        a += [flag, pa[k]]
    if pa.get("kd"):
      a += ["--kd"]
    if o.get("limite"):
      a += ["--max-steps", o["limite"]]
    cmds = [self._cmd("src.train", *a, *extra, o=o)]
    if not o.get("limite"):
      cmds += self._registrar(run)
    return cmds

  def _nombre_extension(self, run, presupuesto, rec):
    import argparse
    from src.train import extension_name
    return extension_name(argparse.Namespace(extend_from=run, presupuesto=presupuesto, tokens=None), rec)

  # ---- evaluar
  def _nombre_eval(self, run, steps, o, base_dtype, legacy=False):
    n = run
    if steps:
      n += f"_s{steps}" if legacy else f"_p{steps}"
    d = o.get("dtype")
    if d and d != base_dtype:
      n += f"_{d}"
    if o.get("ctx_frames") and o["ctx_frames"] != 20:
      n += f"_ctx{o['ctx_frames']}"
    if o.get("temperatura") is not None and o["temperatura"] != 1.0 and not legacy:
      n += f"_t{o['temperatura']:g}"
    return n

  def _dtype_rollouts(self, o, dev):
    """dtype efectivo de los rollouts: el pedido, o fp16 con graphs en CUDA para los transformers y fp32 en el resto."""
    if o.get("dtype"):
      return o["dtype"]
    return DTYPE_VIDEO.get(self.modelo, "fp32") if dev != "cpu" else "fp32"

  def _temperatura(self, o):
    """T de los rollouts y de dónde sale (queda en la bitácora y en los JSON): --temperatura, la elegida de la familia, o el default."""
    if self.modelo == "congelado":
      T, origen = 0.0, "congelado: T = 0 por definición (repite el último frame)"
    elif o.get("temperatura") is not None:
      T, origen = o["temperatura"], "--temperatura"
    else:
      T, origen = None, "default del modelo"
      ref = FAMILIA_REF.get(self.modelo)
      f = rutas.results() / f"temperatura_{ref}.json" if ref else None
      if f is not None and f.exists():
        try:
          T = float(json.loads(f.read_text(encoding="utf-8"))["elegida"])
          origen = f"results/temperatura_{ref}.json (barrido en el holdout, corrida de referencia de la familia {self.modelo})"
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
          pass
    self._meta["temperatura"] = {"valor": T, "origen": origen}
    return T

  def _plan_evaluar(self, o, extra, avisos):
    dev = self._device("evaluar", o)
    spec = self._spec(o)
    run = self.nombre_run(o)
    if o.get("barrer_temperatura"):
      if self.evaluacion != "video":
        raise ToolkitError("--barrer-temperatura solo existe para la evaluación de video")
      a = ["--model", spec] + (["--limite", o["limite"]] if o.get("limite") else [])
      return [self._cmd("src.temperatura", *a, *extra, o=o)]
    if self.evaluacion == "latente":
      if o.get("desde_cache") or o.get("cadena"):
        raise ToolkitError("--desde-cache y --cadena solo existen para la evaluación de video")
      a = ["--model", spec]
      if o.get("limite"):
        a += ["--limite", o["limite"]]
      if o.get("sin_plan"):
        a += ["--no-plan"]
      return [self._cmd("src.eval_latent", *a, *extra, o=o)]
    if o.get("sin_plan"):
      raise ToolkitError("--no-plan solo existe para la evaluación latente")
    cache = bool(o.get("desde_cache"))
    roll_dir = rutas.results() / "rollouts"
    if self.modelo == "real":
      cmds = []
      nreal = "real" if not o.get("clips") or o["clips"] == "eval" else f"real_{o['clips']}"
      if not (cache and (roll_dir / f"{nreal}.npy").exists()):
        cmds.append(self._cmd("src.rollouts", "--name", nreal, "--model", "real", *(["--clips", o["clips"]] if o.get("clips") else []),
                              *(["--limit", o["limite"]] if o.get("limite") else []), *extra, o=o))
      cmds.append(self._cmd("src.metrics_video", "--name", nreal, *(["--clips", o["clips"]] if o.get("clips") else []),
                            *(["--from-cache"] if cache else []), o=o))
      self._dev_forzado = "cpu" if (cache and len(cmds) == 1) else None
      return cmds
    base_dtype = DTYPE_VIDEO.get(self.modelo, "fp32")
    dt = self._dtype_rollouts(o, dev)
    tiene_ce = (not self.es_kind) or getattr(_clase(self.modelo), "output", "tokens") != "pixels"
    nombre_base = LEGACY.get(self.modelo, run) if cache else run
    temperatura = o.get("temperatura") if cache else self._temperatura(o)
    pasos = _lista_csv(o["pasos"]) if o.get("pasos") else [None]
    cmds = []
    if o.get("cadena"):  # calidad en línea (1 paso del modelo por frame real) y con la cadena (src.infer.quality)
      for s in pasos:
        nombre = self._nombre_eval(run, s, {**o, "temperatura": o.get("temperatura")}, base_dtype)
        qdt = o.get("dtype") or ("fp16" if dev != "cpu" else "fp32")
        base = ["--model", spec, "--dtype", qdt] + (["--steps", s] if s else []) + (
          ["--temperature", temperatura] if temperatura is not None else [])
        lim = ["--segments", o["limite"]] if o.get("limite") else []
        cmds.append(self._cmd("src.infer.quality", "online", *base, "--name", nombre, *lim, *extra, o=o))
        cmds.append(self._cmd("src.infer.quality", "imagine", *base, "--name", nombre, *extra, o=o))
        cmds.append(self._cmd("src.metrics_video", "--name", f"{nombre}_stream", o=o))
      return cmds
    ce_nombre = LEGACY_CE.get(self.modelo, run) if cache else run
    if tiene_ce and not (cache and (rutas.results() / f"eval_{ce_nombre}.json").exists()):
      ce = ["--model", spec, "--name", ce_nombre]
      if o.get("dtype"):
        ce += ["--dtype", o["dtype"]]
      if o.get("limite"):
        ce += ["--segments", o["limite"]]
      cmds.append(self._cmd("src.evaluate", *ce, o=o))
    n_pesados = len(cmds)
    for s in pasos:
      explicita = o.get("temperatura") if not cache else None
      nombre = self._nombre_eval(nombre_base, s, {**o, "temperatura": explicita}, base_dtype, legacy=cache)
      if not (cache and (roll_dir / f"{nombre}.npy").exists()):
        r = ["--name", nombre, "--model", spec]
        if s:
          r += ["--steps", s]
        T = temperatura if temperatura is not None else (1.0 if cache else None)
        if T is not None:
          r += ["--temperature", T]
        if dt != "fp32":
          r += ["--dtype", dt]
          if self.modelo in DTYPE_VIDEO or self.modelo in CON_PASOS:
            r += ["--graphs"]
        if o.get("ctx_frames"):
          r += ["--ctx-frames", o["ctx_frames"]]
        if self.modelo == "gpt2m":
          r += ["--bs", 16]
        if o.get("limite"):
          r += ["--limit", o["limite"]]
        cmds.append(self._cmd("src.rollouts", *r, *extra, o=o))
        n_pesados += 1
      cmds.append(self._cmd("src.metrics_video", "--name", nombre, *(["--from-cache"] if cache else []), o=o))
    if cache and n_pesados == 0:
      self._dev_forzado = "cpu"   # todo sale de results/cache: no usa la GPU
    return cmds

  # ---- bench
  def _plan_bench(self, o, extra, avisos):
    dev = self._device("bench", o)
    run = self.nombre_run(o)
    if self.modelo == "codec":
      if o.get("cadena"):
        raise ToolkitError("codec bench no tiene --cadena: mide el codec solo (src.infer.bench_codec)")
      d = o.get("device")
      if o.get("ort") or (d and not backends.obtener(d).torch_device):   # dml y npu: el codec exportado, con ONNX Runtime
        return self._plan_bench_ort(o, extra, d or "cpu")
      _exigir_tag(dev, o)
      tag = ["--tag", o["tag"]] if o.get("tag") else []
      a = (["--device", d] if d else []) + (["--frames", o["limite"]] if o.get("limite") else []) + tag
      cmds = [self._cmd("src.infer.bench_codec", *a, *extra, o=o)]
      for dt in (["fp32", "fp16"] if dev != "cpu" else ["fp32"]):  # el decoder del VQ-VAE solo (src.bench --decoder)
        cmds.append(self._cmd("src.bench", "--decoder", "--dtype", dt, *(["--device", d] if d else []), *tag, o=o))
      return cmds
    spec = self._spec(o)
    pasos = _lista_csv(o["pasos"]) if o.get("pasos") else [None]
    dt = o.get("dtype") or (DTYPE_VIDEO.get(self.modelo, "fp32") if dev != "cpu" else "fp32")
    d = o.get("device")
    if d and not backends.obtener(d).torch_device:   # dml y npu: ONNX Runtime, sin world model en la cadena
      raise ToolkitError(
        f"{d} corre con ONNX Runtime y el bench del world model (con o sin --cadena) corre con torch. Con {d} se mide la "
        f"cadena del codec: `codec bench --device {d}`, que corre `python -m src.infer.bench_ort --enc {d} --dec {d}` "
        "(exportarlo antes con `codec exportar`)")
    cmds = []
    for s in pasos:
      if o.get("cadena") and not o.get("modo"):   # protocolo único (D15): un JSON con modelo solo, cadena cerrada y cadena a ritmo
        a = ["--model", spec, "--name", run + (f"_p{s}" if s else ""), "--dtype", dt,
             "--diagnostico" if o.get("diagnostico") else "--protocolo"]
        if s:
          a += ["--steps", s]
        if o.get("temperatura") is not None:
          a += ["--temperature", o["temperatura"]]
        if o.get("ritmo"):
          a += ["--pace", f"{o['ritmo']:g}"]
        if o.get("limite"):
          a += ["--frames", o["limite"]]
        if o.get("device"):
          a += ["--device", o["device"]]
        if o.get("tag"):
          a += ["--tag", o["tag"]]
        cmds.append(self._cmd("src.infer.bench_e2e", *a, *extra, o=o))
        continue
      if o.get("cadena"):
        modos = [o["modo"]]
        for modo in modos:
          a = ["--model", spec, "--name", run + (f"_p{s}" if s else ""), "--mode", modo, "--dtype", dt]
          if s:
            a += ["--steps", s]
          if o.get("temperatura") is not None:
            a += ["--temperature", o["temperatura"]]
          if o.get("ritmo"):
            a += ["--pace", f"{o['ritmo']:g}"]
          if o.get("limite"):
            a += ["--frames", o["limite"]]
          if o.get("device"):
            a += ["--device", o["device"]]
          if o.get("tag"):
            a += ["--tag", o["tag"]]
          cmds.append(self._cmd("src.infer.bench_e2e", *a, *extra, o=o))
        continue
      if o.get("device") == "mps":
        raise ToolkitError("src.bench solo mide en cuda o cpu; para mps usá `bench --cadena`")
      _exigir_tag(dev, o)
      nombre = run + (f"_p{s}" if s else "")
      a = ["--model", spec, "--name", nombre, "--dtype", dt]
      if dt == "fp16" and (self.modelo in DTYPE_VIDEO or self.modelo in CON_PASOS):
        a += ["--graphs"]
      if s:
        a += ["--steps", s]
      if o.get("ctx_frames"):
        a += ["--ctx-frames", o["ctx_frames"]]
      if o.get("int8"):
        a += ["--int8"]
      if o.get("device"):
        a += ["--device", o["device"]]
      if o.get("limite"):
        a += ["--frames", o["limite"]]
      if o.get("tag"):
        a += ["--tag", o["tag"]]
      cmds.append(self._cmd("src.bench", *a, *extra, o=o))
    return cmds

  def _plan_bench_ort(self, o, extra, d):
    """codec bench con ONNX Runtime (dml, npu, o cpu/cuda con --ort): src.infer.bench_ort con el provider del backend en
    las dos etapas. Lo que se quiera distinto (--only enc, --profile, --dec dml) va después de `--`."""
    try:
      b = backends.resolver_ort(d)
    except ValueError as e:
      raise ToolkitError(f"codec bench --device {d}: {e}") from None
    dt = o.get("dtype") or "fp32"
    if dt not in ("fp32", "fp16"):
      raise ToolkitError(f"codec bench con ONNX Runtime mide fp32 o fp16, no {dt} (fp16 usa los ONNX de `codec exportar --dtype fp16`)")
    a = ["--enc", b.ort_provider, "--dec", b.ort_provider] + (["--fp16"] if dt == "fp16" else [])
    a += (["--frames", o["limite"]] if o.get("limite") else []) + (["--tag", o["tag"]] if o.get("tag") else [])
    self._dev_forzado = b.nombre   # el dispositivo de la bitácora y del candado es el del provider, no el de torch
    return [self._cmd("src.infer.bench_ort", *a, *extra, o=o, env=False)]   # sin ML2_DEVICE: dml y npu no son de torch

  # ---- demo
  def _plan_demo(self, o, extra, avisos):
    run = self.nombre_run(o)
    if o.get("entrenar_decoder"):
      if self.modelo == "lewm":
        a = ["--run", run]
        mod = "src.train_decoder"
        dueño, dec = run, rutas.ckpt() / run / "decoder.pt"
      else:
        enc = o.get("encoder") or (run if self.modelo == "vjepa2" else self._encoder_guardado(run))
        if not enc:
          raise ToolkitError(f"{self.modelo} demo --entrenar-decoder: indicá el encoder con --encoder <corrida vjepa2> "
                             f"(no lo encuentro en {rutas.ckpt() / run})")
        a = ["--run", enc]
        mod = "src.models.vjepa.train_decoder"
        dueño, dec = enc, rutas.ckpt() / f"{enc}_viz" / "decoder.pt"
      if o.get("limite"):
        a += ["--steps", o["limite"]]
      if o.get("semilla") is not None:
        a += ["--seed", o["semilla"]]
      cmds = [self._cmd(mod, *a, *extra, o=o)]
      if not o.get("limite"):  # el decoder completo queda como extra de su corrida, con su sha256, y se sube con ella
        cmds += self._registrar(dueño, dec.as_posix())
      return cmds
    if self.evaluacion == "latente":  # decodificación, máscaras y curva de error: --segment, --start, --decoder, --mp4 van después de `--`
      return [self._cmd("src.infer.demo_latente", "--model", self._spec(o), *extra, o=o)]
    pasos = _lista_csv(o["pasos"]) if o.get("pasos") else [None]
    if len(pasos) > 1:
      raise ToolkitError("demo usa un solo valor de --pasos")
    a = ["--model", self._spec(o)]
    if pasos[0]:
      a += ["--steps", pasos[0]]
    if o.get("temperatura") is not None:
      a += ["--temperature", o["temperatura"]]
    if o.get("dtype"):
      a += ["--dtype", o["dtype"]]
    if o.get("limite"):
      a += ["--frames", o["limite"]]
    return [self._cmd("src.infer.demo", *a, *extra, o=o)]

  def _encoder_guardado(self, run):
    g = self._guardado(run)
    if not g:
      f = rutas.ckpt() / run / "final.pt"
      if not f.exists():
        return None
      import torch
      cfg = torch.load(f, map_location="cpu", weights_only=False).get("config", {})
    else:
      cfg = json.loads(g["args"].get("config") or "{}")
    return cfg.get("encoder_run") or None

  # ---- exportar
  def _plan_exportar(self, o, extra, avisos):
    if self.modelo == "codec":
      return [self._cmd("src.infer.export_codec", *(["--fp16"] if o.get("dtype") == "fp16" else []), *extra, o=o)]
    if self.evaluacion == "latente":  # encoder y/o predictor a ONNX (sale con 1 si ORT difiere de PyTorch en más de 1e-3)
      partes = {"lewm": ("encoder", "predictor"), "vjepa2": ("encoder",), "vjepa2_ac": ("predictor",)}[self.modelo]
      return [self._cmd("src.export_latente", "--model", self._spec(o), "--parte", parte, *extra, o=o) for parte in partes]
    a = ["--model", self._spec(o)] + (["--fp16"] if o.get("dtype") == "fp16" else [])
    return [self._cmd("src.export_onnx", *a, *extra, o=o)]


def _exigir_tag(dev, o):
  """src.bench nombra el JSON <name>__<variante>.json, sin el dispositivo: en rocm (el device resuelto, pedido o no)
  se llamaría igual que el de la RTX 3080 Ti y lo pisaría."""
  if dev == "rocm" and not o.get("tag"):
    raise ToolkitError("con rocm, src.bench escribiría results/bench/<name>__<variante>.json, el mismo nombre que el de la "
                       "RTX 3080 Ti, y lo pisaría: agregá --tag <dispositivo> (p. ej. --tag radeon860m)")


def _dev_torch(d):
  """El nombre que entiende torch.device (rocm -> cuda), para ML2_DEVICE; dml y npu no son de torch: pasan tal cual (ONNX Runtime).
  Los --device de src.bench, bench_codec y bench_e2e llevan el nombre del backend (rocm), no este: con `--device cuda` esos
  scripts exigen una GPU NVIDIA y fallan en una Radeon."""
  try:
    return backends.torch_device(d)
  except ValueError:
    return d


# ---------------------------------------------------------------- acciones generales (sin modelo)

def plan_general(nombre, extra=(), cli_argv=None):
  """reporte, resultados y estadistica: scripts de CPU que escriben en results/ (quedan en la bitácora)."""
  mod = GENERALES_SCRIPT[nombre]
  cmds = [Cmd([PY, "-m", "src.configs"])] if nombre == "reporte" else []   # el reporte parte de results/configs.json derivado de results/
  cmds.append(Cmd([PY, "-m", mod, *[str(a) for a in extra]]))
  return Plan(cmds, None, nombre, None, "cpu", None,
              cli_argv or [nombre, *(["--", *extra] if extra else [])])
