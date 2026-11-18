"""Partes automáticas de las guías de Docs/modelos/: tablas de fidelidad y de receta, acciones declaradas y opciones de la CLI.

Cada guía tiene bloques entre marcadores, que este módulo regenera desde el código (así no se desactualizan):

  <!-- auto:fidelidad kind=gpt -->         tabla de fidelidad (python -m src.recipes --fidelidad <kind>)
  <!-- auto:receta kind=gpt -->            receta de entrenamiento (python -m src.recipes <kind>) y sus valores forzados
  <!-- auto:acciones modelo=gpt -->        ACCIONES y EVALUACION que declara el modelo
  <!-- auto:opciones modelo=gpt accion=entrenar -->   opciones de la CLI propias de la acción para ese modelo
  <!-- auto:opciones_comunes -->           opciones que valen para todas las acciones
  <!-- auto:modelos -->                    tabla de modelos del índice (Docs/modelos/README.md)
  ...contenido generado...
  <!-- /auto -->

  python -m src.docs_gen            reescribe los bloques de Docs/modelos/*.md
  python -m src.docs_gen --check    no escribe: sale con 1 si algún bloque está desactualizado o falta una guía o un ejemplo

--check también exige que cada modelo registrado tenga su guía (los kinds, Docs/modelos/<kind>.md; gpt2m, los baselines, real y
codec, Docs/modelos/referencias.md) y que cada acción declarada tenga su subsección (`### <acción>`) y su ejemplo
(`python -m src.cli <modelo> <acción>`). La CLI lo expone como `python -m src.cli docs [--check]`.
"""
import argparse
import dataclasses
import difflib
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DOCS = RAIZ / "Docs" / "modelos"
REFERENCIAS = "referencias"
BLOQUE = re.compile(r"<!-- auto:(?P<nombre>\w+)(?P<args>(?: \w+=[\w.\-]+)*) -->\n(?P<cuerpo>.*?)<!-- /auto -->", re.S)

DESCRIPCION = {
  "receta": "muestra la receta y la Config efectivas, con los overrides y los valores forzados marcados",
  "entrenar": "entrena de cero con la receta de la fuente (`--humo`: prueba corta de 300 pasos para elegir el lr)",
  "continuar": "retoma una corrida cortada o, con `--presupuesto`, la extiende",
  "evaluar": "mide el modelo (video: CE, rollouts, LPIPS y FVD; latentes: error, probes, sorpresa y CEM)",
  "bench": "mide la latencia (el modelo solo, o la cadena de punta a punta con `--cadena`)",
  "demo": "genera la demostración (mp4 con el video real y el predicho, o figura para los latentes)",
  "exportar": "exporta a ONNX",
  "estado": "porcentaje, paso, loss, lr, tramo y ETA de las corridas del modelo",
}
# opciones que solo tienen sentido con cierto tipo de evaluación o modelo (el Toolkit las rechaza al planificar, no al validar)
SOLO_VIDEO = {"desde_cache", "barrer_temperatura"}
SOLO_LATENTE = {"sin_plan"}
SOLO_GPT = {"destilar"}
EN_EVALUAR_SOLO_VIDEO = {"cadena"}
CON_COMPANERA = {"modo": {"cadena": True}, "ritmo": {"cadena": True}, "diagnostico": {"cadena": True},
                 "humo_candidatos": {"humo": True}}


def _toolkit():
  from src import toolkit
  return toolkit


def _celda(x):
  return str(x).replace("|", "/").replace("\n", " ")


# ---------------------------------------------------------------- generadores

def gen_fidelidad(kind):
  from src import recipes
  if kind not in recipes.fidelidad_kinds():
    raise KeyError(f"fidelidad: {kind!r} no tiene FIDELIDAD (kinds: {', '.join(recipes.fidelidad_kinds())})")
  return recipes.fidelidad_md(kind)


def _recetas(kind):
  """[(etiqueta, Recipe)] de un kind (una por tamaño si lo tiene)."""
  from src import recipes
  from src.models import SIZES, model_class
  cls = model_class(kind)
  if not hasattr(cls, "recipe"):
    return []
  sized = "n_layer" in {f.name for f in dataclasses.fields(cls.Config)}
  return [(f"{kind}_{s}" if s else kind, cls.recipe(s)) for s in (SIZES if sized else [None])]


def gen_receta(kind):
  from src import recipes
  filas = _recetas(kind)
  if not filas:
    return f"El modelo `{kind}` no declara receta."
  grupos = []   # [(etiquetas, receta)]: los tamaños con la misma receta van juntos
  for etiqueta, r in filas:
    for g in grupos:
      if g[1].hash() == r.hash():
        g[0].append(etiqueta)
        break
    else:
      grupos.append(([etiqueta], r))
  cab = ["modelo", "fuente", "optimizador", "betas", "lr", "wd", "EMA", "clip", "precisión", "batch x frames (frameskip) x acumulación",
         "presupuesto (frames)", "extras", "pasos del paper"]
  lineas = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
  for etiquetas, r in grupos:
    lineas.append("| " + " | ".join(_celda(c) for c in [
      ", ".join(f"`{e}`" for e in etiquetas), r.fuente, r.optimizer, tuple(r.betas), "; ".join(recipes._seg_text(s) for s in r.lr_segs),
      recipes._wd_text(r) + (f" ({r.wd_on})" if r.wd_on != "matrices" else ""), "sí" if r.ema_segs else "no", r.grad_clip,
      r.amp or "fp32", f"{r.batch} x {r.frames} ({r.frameskip}) x {r.accum}", r.presupuesto,
      ", ".join(f"{k}={v}" for k, v in r.extras.items()) or "-", r.paper_steps or "-"]) + " |")
  forzados = []
  for etiquetas, r in grupos:
    for campo, motivo, cat in r.forzado:
      forzados.append(f"- `{etiquetas[0]}`: {campo}: {motivo} ({cat})")
  if forzados:
    lineas += ["", "Valores forzados (campo: motivo, categoría):", ""] + forzados
  return "\n".join(lineas)


def _descripcion_acciones(modelo):
  tk = _toolkit()
  acciones = tk.catalogo().get(modelo) or ()
  return acciones


def gen_acciones(modelo):
  tk = _toolkit()
  acciones = _descripcion_acciones(modelo)
  if not acciones:
    return f"El modelo `{modelo}` todavía no declara `ACCIONES`."
  try:
    ev = tk.evaluacion_de(modelo)
  except tk.ToolkitError:
    ev = None
  lineas = []
  if modelo in tk.kinds():
    lineas.append(f"Evaluación: `{ev}`." if ev else "Evaluación: no declarada.")
    lineas.append("")
  lineas += ["| acción | qué hace |", "|---|---|"]
  for a in acciones:
    lineas.append(f"| `{modelo} {a}` | {DESCRIPCION[a]} |")
  return "\n".join(lineas)


def _aplica(modelo, accion, o):
  """¿El Toolkit acepta la opción `o` para (modelo, acción)? Usa su propia validación y las reglas que aplica al planificar."""
  tk = _toolkit()
  try:
    ev = tk.evaluacion_de(modelo)
  except tk.ToolkitError:
    ev = "video"
  if o.nombre in SOLO_VIDEO and ev != "video":
    return False
  if o.nombre in SOLO_LATENTE and ev != "latente":
    return False
  if o.nombre in SOLO_GPT and modelo != "gpt":
    return False
  if o.nombre in EN_EVALUAR_SOLO_VIDEO and accion == "evaluar" and ev != "video":
    return False
  if o.nombre == "entrenar_decoder" and ev != "latente":
    return False
  valor = {"bool": True, "int": 1, "float": 1.0, "str": "s" if o.nombre == "tamano" else "x", "lista": ["receta.lr_peak=1e-4"]}[o.tipo]
  if o.nombre == "pasos":
    valor = "1"
  if o.choices:
    valor = o.choices[0]
  try:
    tk.Toolkit(modelo)._validar(accion, {o.nombre: valor, **CON_COMPANERA.get(o.nombre, {})})
  except tk.ToolkitError:
    return False
  return True


def gen_opciones(modelo, accion):
  tk = _toolkit()
  propias = [o for o in tk.OPCIONES if o.acciones and accion in o.acciones and _aplica(modelo, accion, o)]
  if accion not in tk.catalogo().get(modelo, ()):
    return f"`{modelo}` no tiene la acción `{accion}`."
  if not propias:
    return "Sin opciones propias; valen las [opciones comunes](README.md#opciones-comunes)."
  lineas = [f"- `{o.flags[0]}`: {o.ayuda}" for o in propias]
  return "\n".join(lineas + ["", "Más las [opciones comunes](README.md#opciones-comunes)."])


def gen_opciones_comunes():
  tk = _toolkit()
  lineas = ["| opción | qué hace |", "|---|---|"]
  for o in tk.OPCIONES:
    if o.acciones is None:
      flags = ", ".join(f"`{f}`" for f in o.flags)
      lineas.append(f"| {flags} | {_celda(o.ayuda)} |")
  return "\n".join(lineas)


def archivo_de(modelo):
  tk = _toolkit()
  return DOCS / (f"{modelo}.md" if modelo in tk.kinds() else f"{REFERENCIAS}.md")


def gen_modelos():
  tk = _toolkit()
  lineas = ["| modelo | evaluación | acciones | guía |", "|---|---|---|---|"]
  for m, acc in tk.catalogo().items():
    try:
      ev = tk.evaluacion_de(m)
    except tk.ToolkitError:
      ev = "-"
    guia = archivo_de(m).name
    lineas.append(f"| `{m}` | {ev} | {', '.join(acc) if acc else '-'} | [{guia}]({guia}) |")
  return "\n".join(lineas)


GENERADORES = {
  "fidelidad": lambda a: gen_fidelidad(a["kind"]),
  "receta": lambda a: gen_receta(a["kind"]),
  "acciones": lambda a: gen_acciones(a["modelo"]),
  "opciones": lambda a: gen_opciones(a["modelo"], a["accion"]),
  "opciones_comunes": lambda a: gen_opciones_comunes(),
  "modelos": lambda a: gen_modelos(),
}


# ---------------------------------------------------------------- bloques

def _args(texto):
  return dict(kv.split("=", 1) for kv in texto.split())


def regenerar(texto, origen="", errores=None):
  """Texto con todos los bloques `auto` reescritos. Los errores (bloque desconocido, marcador sin cerrar) van a `errores`."""
  errores = errores if errores is not None else []

  def _sub(m):
    nombre, args = m.group("nombre"), _args(m.group("args"))
    if nombre not in GENERADORES:
      errores.append(f"{origen}: bloque auto:{nombre} desconocido (hay: {', '.join(GENERADORES)})")
      return m.group(0)
    try:
      cuerpo = GENERADORES[nombre](args)
    except KeyError as e:
      errores.append(f"{origen}: auto:{nombre} {m.group('args').strip()}: falta el argumento o el modelo {e}")
      return m.group(0)
    return f"<!-- auto:{nombre}{m.group('args')} -->\n{cuerpo}\n<!-- /auto -->"

  nuevo = BLOQUE.sub(_sub, texto)
  abiertos, cerrados = len(re.findall(r"<!-- auto:\w+", nuevo)), len(re.findall(r"<!-- /auto -->", nuevo))
  if abiertos != cerrados:
    errores.append(f"{origen}: {abiertos} marcadores <!-- auto:... --> y {cerrados} <!-- /auto -->")
  return nuevo


def guias():
  return sorted(DOCS.glob("*.md")) if DOCS.exists() else []


def faltantes():
  """Problemas de estructura: guías que faltan, acciones sin subsección o sin ejemplo."""
  tk = _toolkit()
  problemas = []
  if not (DOCS / "README.md").exists():
    problemas.append("falta Docs/modelos/README.md (el índice)")
  textos = {}
  for modelo, acciones in tk.catalogo().items():
    f = archivo_de(modelo)
    if not f.exists():
      if modelo != "plantilla":   # la guía de la plantilla es opcional hasta que exista
        problemas.append(f"falta la guía {f.relative_to(RAIZ).as_posix()} (modelo {modelo})")
      continue
    t = textos.setdefault(f, f.read_text(encoding="utf-8"))
    titulos = set(re.findall(r"^###\s+`?([\w ]+?)`?\s*$", t, re.M))
    for a in acciones:
      if f"python -m src.cli {modelo} {a}" not in t:
        problemas.append(f"{f.name}: falta un ejemplo `python -m src.cli {modelo} {a}`")
      if f.stem != REFERENCIAS and a not in titulos:
        problemas.append(f"{f.name}: falta la subsección `### {a}`")
  return problemas


def main(argv=None):
  p = argparse.ArgumentParser(prog="python -m src.docs_gen", description=__doc__.split("\n")[0])
  p.add_argument("--check", action="store_true", help="no escribe: sale con 1 si algo está desactualizado o falta")
  a = p.parse_args(argv)
  if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  errores, distintos = [], []
  for f in guias():
    viejo = f.read_text(encoding="utf-8")
    nuevo = regenerar(viejo, f.name, errores)
    if nuevo != viejo:
      distintos.append(f)
      if a.check:
        diff = difflib.unified_diff(viejo.splitlines(), nuevo.splitlines(), f"{f.name} (en disco)", f"{f.name} (generado)", lineterm="", n=1)
        print("\n".join(list(diff)[:40]))
      else:
        f.write_text(nuevo, encoding="utf-8", newline="\n")
        print(f"actualizado {f.relative_to(RAIZ).as_posix()}")
  errores += faltantes()
  for e in errores:
    print(f"error: {e}")
  if a.check:
    if distintos or errores:
      print(f"docs --check: {len(distintos)} guías desactualizadas, {len(errores)} errores (python -m src.docs_gen las regenera)")
      return 1
    print("docs --check: 0 diferencias")
    return 0
  if not distintos:
    print("docs: todo al día")
  return 1 if errores else 0


if __name__ == "__main__":
  sys.exit(main())
