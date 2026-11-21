"""Punto de entrada único: `python -m src.cli <modelo> <acción> [opciones] [-- args del script]`.

Es un argparse fino sobre src/toolkit.py (la misma API que usan los notebooks). Cada acción corre scripts existentes como
subprocesos, con candado de GPU, cola y bitácora (results/bitacora.jsonl).

  python -m src.cli genie entrenar --tamaño s --semilla 1
  python -m src.cli vjepa2 continuar --run vjepa2_s --presupuesto 10x
  python -m src.cli lewm evaluar --run lewm --limite 50 --device cpu
  python -m src.cli convlstm_px bench --cadena --ritmo 50
  python -m src.cli gpt entrenar --tamaño s --dry-run            # imprime los comandos, no corre nada
  python -m src.cli cola correr --dry-run                        # toda la cola de scripts/queue.toml

Comandos generales: cola {ver,correr,quitar}, listar, resultados, reporte, bitacora {ver,verificar,repetir}.
"""
import argparse
import contextlib
import io
import shlex
import sys
import tomllib
from pathlib import Path

from src import bitacora, cola, ejecutor, rutas
from src import toolkit as T
from src.toolkit import ACCIONES_TODAS, OPCIONES, Toolkit, ToolkitError

GENERALES = ("cola", "listar", "resultados", "reporte", "estadistica", "bitacora", "docs", "verificar", "backends", "evaluadores")
HOOKS_VERIFICAR = [  # (nombre, archivo que lo implementa, comando): se corren cuando el paquete que los agrega ya está
  ("docs --check", "src/docs_gen.py", ["python", "-m", "src.docs_gen", "--check"]),
  ("make_notebooks --check", "scripts/make_notebooks.py", ["python", "scripts/make_notebooks.py", "--check"]),
  ("check_docs", "scripts/check_docs.py", ["python", "scripts/check_docs.py"]),
]
QUEUE_TOML = Path("scripts/queue.toml")


# ---------------------------------------------------------------- parsers

def _añadir_opciones(p):
  for o in OPCIONES:
    kw = {"help": o.ayuda, "dest": o.nombre, "default": None}
    if o.tipo == "bool":
      p.add_argument(*o.flags, action="store_true", **{**kw, "default": False})
    elif o.tipo == "lista":
      p.add_argument(*o.flags, action="append", metavar="CLAVE=VALOR", **{**kw, "default": []})
    elif o.tipo == "float" and o.const is not None:
      p.add_argument(*o.flags, nargs="?", const=o.const, type=float, **kw)
    else:
      p.add_argument(*o.flags, type={"str": str, "int": int, "float": float}[o.tipo], choices=o.choices, **kw)


def _texto_catalogo():
  lineas = ["modelos y sus acciones:"]
  for m, acc in T.catalogo().items():
    ev = ""
    if m in T.kinds():
      try:
        ev = f"  [evaluación: {T.evaluacion_de(m)}]"
      except ToolkitError:
        ev = ""
    lineas.append(f"  {m:12s} {', '.join(acc) if acc else '(todavía no declara ACCIONES)'}{ev}")
  lineas += ["", "comandos generales:", "  cola {ver,correr,quitar}   la cola de GPU (results/queue.json y scripts/queue.toml)",
             "  listar                     modelos, acciones, corridas, backends (con lo detectado acá) y evaluadores",
             "  backends                   dispositivos registrados y cuáles hay en esta máquina",
             "  evaluadores {ver,verificar}   evaluaciones registradas y su prueba sintética",
             "  resultados                 estado (vigente / protocolo anterior / pendiente) de cada resultado",
             "  reporte                    tablas y figuras (src.report)",
             "  estadistica {semillas,piso,diferencia}   media entre semillas, piso de FVD, diferencia pareada (src.estadistica)",
             "  verificar [--cuda] [--notebooks]   pytest, docs, notebooks, check_docs y bitacora verificar",
             "  bitacora {ver,verificar,repetir}",
             "  docs [--check]             tablas y guías generadas (src.docs_gen)"]
  return "\n".join(lineas)


def parser_modelo():
  p = argparse.ArgumentParser(
    prog="python -m src.cli <modelo> <acción>", formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Una acción de un modelo. Lo que venga después de `--` pasa tal cual al script envuelto.",
    epilog=f"acciones: {', '.join(ACCIONES_TODAS)}")
  p.add_argument("modelo", choices=T.modelos())
  p.add_argument("accion", choices=ACCIONES_TODAS)
  _añadir_opciones(p)
  return p


def parser_principal():
  return argparse.ArgumentParser(
    prog="python -m src.cli", formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Punto de entrada único: python -m src.cli <modelo> <acción> [opciones] [-- args del script]",
    epilog=_texto_catalogo() + "\n\nopciones comunes (con `<modelo> <acción> --help` se ven todas): " +
    ", ".join(f for o in OPCIONES for f in o.flags[:1] if o.nombre in ("run", "tamano", "semilla", "presupuesto", "device", "limite",
                                                                     "set", "dry_run", "esperar", "encolar")))


def _partir(argv):
  """(antes, extra): los argumentos antes y después del primer `--`."""
  if "--" in argv:
    i = argv.index("--")
    return argv[:i], argv[i + 1:]
  return list(argv), []


# ---------------------------------------------------------------- acciones de modelo

def accion_modelo(argv, esperar=False, desde_cola=False, meta=None, salida=print):
  antes, extra = _partir(argv)
  ns = parser_modelo().parse_args(antes)
  opts = {o.nombre: getattr(ns, o.nombre) for o in OPCIONES}
  if esperar:
    opts["esperar"] = True
  try:
    return Toolkit(ns.modelo).accion(ns.accion, extra=extra, cli_argv=list(argv), desde_cola=desde_cola, meta=meta,
                                     salida=salida, **opts).codigo
  except ToolkitError as e:
    print(f"error: {e}", file=sys.stderr)
    return 2


def accion_general(nombre, argv):
  """reporte, resultados y estadistica: scripts de CPU que escriben en results/ (con bitácora, sin candado)."""
  p = argparse.ArgumentParser(prog=f"python -m src.cli {nombre}", description="Lo que venga después de `--` pasa al script.")
  p.add_argument("--dry-run", action="store_true")
  if nombre == "estadistica":  # todo lo demás son los argumentos de src.estadistica (semillas|piso|diferencia ...)
    ns, extra = p.parse_known_args(argv)
  else:
    antes, extra = _partir(argv)
    ns = p.parse_args(antes)
  plan = T.plan_general(nombre, extra, cli_argv=[nombre, *argv])
  return ejecutor.ejecutar(plan, dry_run=ns.dry_run).codigo


# ---------------------------------------------------------------- cola

def cargar_toml(archivo):
  """[{id, nombre, horas, comandos}] de la cola declarada (scripts/queue.toml)."""
  with open(archivo, "rb") as f:
    datos = tomllib.load(f)
  trabajos = datos.get("trabajo", [])
  for t in trabajos:
    if "comandos" not in t or "id" not in t:
      raise ToolkitError(f"{archivo}: cada [[trabajo]] necesita `id` y `comandos`")
  return trabajos


def _con_dry_run(args):
  """args con --dry-run para la CLI: antes del `--`, si lo hay; después iría al script envuelto y correría de verdad."""
  antes, extra = _partir(args)
  return [*antes, "--dry-run", *(["--", *extra] if "--" in args else [])]


def _expandir(cmd):
  """Líneas que ejecutaría un comando de la cola (su --dry-run)."""
  args = shlex.split(cmd)
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    try:
      rc = main(_con_dry_run(args), _anidado=True)
    except SystemExit as e:
      rc = e.code or 0
  return rc, buf.getvalue().rstrip().splitlines()


def cola_correr(archivo, dry_run=False, trabajos=None, desde=None, salida=print, incluir_opcionales=False):
  items = cola.pendientes()
  try:
    jobs = cargar_toml(archivo) if archivo and Path(archivo).exists() else []
  except (tomllib.TOMLDecodeError, ToolkitError) as e:
    salida(f"error: {e}")
    return 2
  if archivo and not Path(archivo).exists():
    salida(f"aviso: no existe {archivo}: solo corro results/queue.json")
  if trabajos:
    jobs = [j for j in jobs if j["id"] in set(trabajos)]   # pedido explícito: corre también los opcionales
  elif not incluir_opcionales and not dry_run:
    jobs = [j for j in jobs if not j.get("opcional")]
  if desde is not None:
    jobs = [j for j in jobs if j["id"] >= desde]
  if dry_run:
    malos = 0
    if items:
      salida(f"# results/queue.json: {len(items)} acciones encoladas")
      for i in items:
        salida(f"python -m src.cli {shlex.join(i['argv'])}")
    total_h = 0.0
    activos = [j for j in jobs if trabajos or incluir_opcionales or not j.get("opcional")]
    for j in jobs:
      marca = "" if j in activos else " [OPCIONAL, desactivado: --trabajo N o --incluir-opcionales]"
      if j in activos:
        total_h += float(j.get("horas", 0))
      salida(f"# trabajo {j['id']}: {j.get('nombre', '')} ({j.get('horas', '?')} h){marca}")
      for cmd in j["comandos"]:
        salida(f"python -m src.cli {cmd}")
        rc, lineas = _expandir(cmd)
        for ln in lineas:
          salida("    " + ln)
        if rc != 0:
          malos += 1
          salida(f"    ^ ERROR (código {rc})")
    salida(f"# {len(activos)} trabajos, {sum(len(j['comandos']) for j in activos)} comandos, ~{total_h:g} h"
           + (f"; {len(jobs) - len(activos)} opcionales desactivados" if len(jobs) > len(activos) else "")
           + (f"; {malos} con error" if malos else ""))
    return 1 if malos else 0
  for i in items:
    salida(f"== cola {i['id']}: python -m src.cli {shlex.join(i['argv'])}")
    cola.actualizar(i["id"], estado="corriendo")
    rc = main(i["argv"], _esperar=True, _desde_cola=True)
    if rc != 0:
      cola.actualizar(i["id"], estado="fallada", codigo=rc)
      salida(f"la cola se frena: {i['id']} terminó con código {rc} (queda en results/queue.json como fallada; `cola quitar {i['id']}`)")
      return rc
    cola.quitar(i["id"])
  for j in jobs:
    salida(f"== trabajo {j['id']}: {j.get('nombre', '')}")
    for cmd in j["comandos"]:
      args = shlex.split(cmd)
      salida(f"-- python -m src.cli {cmd}")
      rc = main(args, _esperar=True, _desde_cola=True)
      if rc != 0:
        salida(f"la cola se frena: el trabajo {j['id']} falló (código {rc}) en `{cmd}`. Arreglá la causa (código, test, commit) y "
               "repetí con `bitacora repetir <id>`")
        return rc
  return 0


def comando_cola(argv):
  p = argparse.ArgumentParser(prog="python -m src.cli cola")
  sub = p.add_subparsers(dest="sub", required=True)
  sub.add_parser("ver", help="GPU, items encolados y su estado")
  c = sub.add_parser("correr", help="único que ejecuta la cola, de a un trabajo; frena ante una falla")
  c.add_argument("--archivo", default=str(QUEUE_TOML), help="cola declarada (default scripts/queue.toml)")
  c.add_argument("--dry-run", action="store_true")
  c.add_argument("--trabajo", type=int, action="append", help="solo este trabajo (repetible)")
  c.add_argument("--desde", type=int, help="desde este trabajo")
  c.add_argument("--incluir-opcionales", action="store_true", help="también los trabajos marcados `opcional = true`")
  q = sub.add_parser("quitar", help="saca un item de results/queue.json")
  q.add_argument("id")
  ns = p.parse_args(argv)
  if ns.sub == "ver":
    print(cola.cola_texto())
    return 0
  if ns.sub == "quitar":
    ok = cola.quitar(ns.id)
    print(f"quitado {ns.id}" if ok else f"no hay un item {ns.id} en {rutas.cola()}")
    return 0 if ok else 1
  return cola_correr(ns.archivo, ns.dry_run, ns.trabajo, ns.desde, incluir_opcionales=ns.incluir_opcionales)


# ---------------------------------------------------------------- bitácora y otros

def comando_bitacora(argv):
  p = argparse.ArgumentParser(prog="python -m src.cli bitacora")
  sub = p.add_subparsers(dest="sub", required=True)
  v = sub.add_parser("ver")
  v.add_argument("-n", type=int, default=20)
  sub.add_parser("verificar", help="cada resultado vigente: entrada con salida 0, árbol limpio y commit en main")
  sub.add_parser("vigentes", help="marca vigente cada resultado sin estado que escribió una entrada válida")
  r = sub.add_parser("repetir")
  r.add_argument("id")
  r.add_argument("--dry-run", action="store_true")
  ns = p.parse_args(argv)
  if ns.sub == "ver":
    print(bitacora.ver(ns.n))
    return 0
  if ns.sub == "verificar":
    problemas = bitacora.verificar()
    for arch, motivo in problemas:
      print(f"{arch}: {motivo}")
    print("bitácora: todos los resultados vigentes tienen su comando" if not problemas else f"{len(problemas)} problemas")
    return 1 if problemas else 0
  if ns.sub == "vigentes":
    marcados, problemas = bitacora.marcar_vigentes()
    for arch in marcados:
      print(f"vigente: {arch}")
    for arch, motivo in problemas:
      print(f"sin marcar: {arch}: {motivo}")
    print(f"{len(marcados)} marcados vigentes, {len(problemas)} sin marcar")
    return 1 if problemas else 0
  try:
    av = bitacora.repetir(ns.id)
  except KeyError as e:
    print(f"error: {e.args[0]}", file=sys.stderr)
    return 2
  if ns.dry_run:
    print(ejecutor.argv_texto(av))
    return 0
  return main(av, _meta={"repetido_de": bitacora.buscar(ns.id)["id"]})


def comando_verificar(argv):
  """pytest (sin --cuda con la GPU oculta; con --cuda corren los needs_cuda), bitacora verificar y los ganchos de docs y notebooks."""
  p = argparse.ArgumentParser(prog="python -m src.cli verificar")
  p.add_argument("--cuda", action="store_true", help="con la GPU visible (corren los tests needs_cuda)")
  p.add_argument("--notebooks", action="store_true", help="también ejecuta los notebooks (scripts/run_notebooks.py)")
  p.add_argument("--dry-run", action="store_true")
  ns = p.parse_args(argv)
  cmds = [ejecutor.Cmd(["python", "-m", "pytest", "-q"], {} if ns.cuda else {"CUDA_VISIBLE_DEVICES": "-1"}),
          ejecutor.Cmd(["python", "-m", "src.cli", "evaluadores", "verificar"]),
          ejecutor.Cmd(["python", "-m", "src.cli", "backends"]),
          ejecutor.Cmd(["python", "-m", "src.cli", "bitacora", "verificar"])]
  avisos = []
  for nombre, archivo, cmd in HOOKS_VERIFICAR:
    if Path(archivo).exists():
      cmds.append(ejecutor.Cmd(cmd))
    else:
      avisos.append(f"{nombre}: no disponible (todavía no existe {archivo}); se suma cuando llegue")
  if ns.notebooks:
    if Path("scripts/run_notebooks.py").exists():
      cmds.append(ejecutor.Cmd(["python", "scripts/run_notebooks.py"]))
    else:
      avisos.append("run_notebooks: no disponible (todavía no existe scripts/run_notebooks.py)")
  dev = ejecutor.resolver_device() if ns.cuda else "cpu"
  plan = ejecutor.Plan(cmds, None, "verificar", None, "cpu" if dev == "cpu" else dev, None, ["verificar", *argv], avisos)
  return ejecutor.ejecutar(plan, dry_run=ns.dry_run).codigo


def comando_docs(argv):
  """Tablas y guías generadas (src/docs_gen.py): `docs [--check]`. Queda en la bitácora; si el módulo no existe, avisa."""
  p = argparse.ArgumentParser(prog="python -m src.cli docs", description="Genera (o con --check verifica) las tablas de las guías.")
  p.add_argument("--check", action="store_true")
  p.add_argument("--dry-run", action="store_true")
  ns = p.parse_args(argv)
  if not Path("src/docs_gen.py").exists():
    print("docs: no disponible (todavía no existe src/docs_gen.py)", file=sys.stderr)
    return 2
  plan = ejecutor.Plan([ejecutor.Cmd(["python", "-m", "src.docs_gen", *(["--check"] if ns.check else [])])], None, "docs", None, "cpu",
                       None, ["docs", *argv])
  return ejecutor.ejecutar(plan, dry_run=ns.dry_run).codigo


def comando_backends(argv):
  from src.infer import backends
  print("backends (dispositivos):")
  for f in backends.tabla():
    print(f"  {f['backend']:5s} {'sí ' if f['disponible'] else 'no '} {'+'.join(f['motores']):9s} {f['descripcion']}"
          + (f"  [{f['motivo']}]" if f["motivo"] else ""))
  print(f"por defecto: {backends.por_defecto()}")
  return 0


def comando_evaluadores(argv):
  from src import evaluadores
  p = argparse.ArgumentParser(prog="python -m src.cli evaluadores")
  p.add_argument("sub", nargs="?", choices=["ver", "verificar"], default="ver")
  ns = p.parse_args(argv)
  if ns.sub == "ver":
    for f in evaluadores.tabla():
      print(f"  {f['evaluador']:14s} aplica a {','.join(f['aplica_a']):16s} {f['descripcion']}")
    return 0
  malos = 0
  for n in evaluadores.nombres():
    try:
      evaluadores.verificar(n)
      print(f"  {n}: ok")
    except Exception as e:  # noqa: BLE001
      malos += 1
      print(f"  {n}: FALLA {type(e).__name__}: {e}")
  print("evaluadores: todos pasan su prueba sintética" if not malos else f"{malos} evaluadores con problemas")
  return 1 if malos else 0


def comando_listar(argv):
  print(_texto_catalogo())
  try:
    from src import store
    runs = sorted(store.load_manifest())
  except Exception:  # noqa: BLE001
    runs = []
  print("\ncorridas registradas (results/checkpoints.json):\n  " + ("\n  ".join(runs) if runs else "(ninguna)"))
  print()
  comando_backends([])
  print("\nevaluadores registrados:")
  comando_evaluadores(["ver"])
  return 0


# ---------------------------------------------------------------- main

def main(argv=None, _esperar=False, _desde_cola=False, _meta=None, _anidado=False):
  """Devuelve el código de salida (0 = bien). `--help` y los errores de argumentos salen con SystemExit como argparse."""
  argv = list(sys.argv[1:] if argv is None else argv)
  if hasattr(sys.stdout, "reconfigure") and not _anidado:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  if not argv or argv[0] in ("-h", "--help"):
    parser_principal().print_help()
    return 0
  cmd, resto = argv[0], argv[1:]
  if cmd == "cola":
    return comando_cola(resto)
  if cmd == "listar":
    return comando_listar(resto)
  if cmd == "bitacora":
    return comando_bitacora(resto)
  if cmd == "backends":
    return comando_backends(resto)
  if cmd == "evaluadores":
    return comando_evaluadores(resto)
  if cmd == "docs":
    return comando_docs(resto)
  if cmd == "verificar":
    return comando_verificar(resto)
  if cmd in ("reporte", "resultados", "estadistica"):
    return accion_general(cmd, resto)
  return accion_modelo(argv, esperar=_esperar, desde_cola=_desde_cola, meta=_meta)


if __name__ == "__main__":
  sys.exit(main())
