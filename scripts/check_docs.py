"""Chequeos de la documentación: python scripts/check_docs.py [--estricto]

  1. Links: cada link relativo de un .md apunta a un archivo (y a un ancla) que existe. Un link a un archivo que se genera
     (notebooks/modelos/*.ipynb, results/...) es solo un aviso, salvo con --estricto.
  2. Planes: no queda ningún Docs/plan-*.md ni ninguna mención a los planes borrados (plan-proyecto, plan-implementacion,
     plan-inferencia, plan-jepa) en los archivos del repo.
  3. Números sin respaldo: en las tablas de Docs/resultados.md (entre `<!-- tabla:... -->` y `<!-- /tabla -->`) toda celda
     numérica necesita, en su fila, el link a un JSON de results/ que exista y esté `vigente`; si no, la celda dice
     "pendiente". En el texto de los docs, una métrica (FVD, LPIPS, CE, p50, p99...) seguida de un número necesita lo mismo
     en la línea. Los bloques generados (`<!-- auto:... -->`) y los papers quedan afuera. `<!-- ok-numero -->` en una línea la
     exime.
  4. Ejemplos: cada comando `python -m src.cli ...` de un bloque ```bash corre con `--dry-run` (los que no lo tienen, con
     `--help`); cada `python -m <módulo>` y `python <script>` apunta a algo que existe.

Sale con 1 si hay algún problema. Corre en CPU y no escribe nada (el dry-run no toca la bitácora).
"""
import argparse
import contextlib
import importlib.util
import io
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

RAIZ = Path(__file__).resolve().parent.parent
EXCLUIR_DIRS = {".venv", "venv", "third_party", "data", "checkpoints", "node_modules", ".git", "__pycache__", ".pytest_cache"}
SIN_NUMEROS = {"Docs/world-models-conduccion-papers.md"}           # los papers citan sus propios números
SIN_EJEMPLOS = {"Docs/world-models-conduccion-papers.md"}
PLAN_RE = re.compile(r"plan-(proyecto|implementacion|inferencia|jepa)")
LINK_RE = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
AUTO_RE = re.compile(r"<!-- auto:.*?<!-- /auto -->", re.S)
TABLA_RE = re.compile(r"<!-- tabla:(\w+) -->\n(.*?)<!-- /tabla -->", re.S)
NUM_CELDA = re.compile(r"^[~≈<>+\-−]?\d[\d.,]*(?:\s?(?:ms|%|s|h|x|M|k))?(?:\s?±\s?\d[\d.,]*)?$")
METRICA = re.compile(r"\b(?:FVD|LPIPS|ce_all|ce_last|CE|p50|p99)\b[^|\n`]{0,25}?\b\d+[.,]\d+")
JSON_RE = re.compile(r"results/[\w./\-]+\.json")
BASH_RE = re.compile(r"^```bash\s*\n(.*?)^```", re.S | re.M)


def archivos():
  """Todos los archivos del repo que no son datos, entornos ni basura."""
  try:
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=RAIZ, capture_output=True, text=True, check=True)
    rutas = [RAIZ / p for p in out.stdout.splitlines() if p]
  except (OSError, subprocess.CalledProcessError):
    rutas = [p for p in RAIZ.rglob("*") if p.is_file()]
  return [p for p in rutas if p.is_file() and not (set(p.relative_to(RAIZ).parts) & EXCLUIR_DIRS)]


def docs_md(todos):
  return [p for p in todos if p.suffix == ".md" and "pre-entrega" not in p.relative_to(RAIZ).parts]


def rel(p):
  return p.relative_to(RAIZ).as_posix()


def sin_codigo(texto):
  """El texto sin los bloques ``` (para los links y las métricas)."""
  return re.sub(r"^```.*?^```", "", texto, flags=re.S | re.M)


# ---------------------------------------------------------------- links

def slug(titulo):
  t = re.sub(r"`", "", titulo.strip().lower())
  t = re.sub(r"[^\w\- ]", "", t)
  return t.replace(" ", "-")


def anclas(p, cache={}):
  if p not in cache:
    vistos, out = {}, set()
    for m in re.finditer(r"^#{1,6}\s+(.*?)\s*#*\s*$", sin_codigo(p.read_text(encoding="utf-8")), re.M):
      s = slug(m.group(1))
      n = vistos.get(s, 0)
      vistos[s] = n + 1
      out.add(s if n == 0 else f"{s}-{n}")
    cache[p] = out
  return cache[p]


def chequear_links(mds, estricto, problemas, avisos):
  for p in mds:
    texto = sin_codigo(p.read_text(encoding="utf-8"))
    for m in LINK_RE.finditer(texto):
      dest = m.group(1)
      if re.match(r"^[a-z]+:", dest):
        continue
      ruta, _, ancla = unquote(dest).partition("#")
      if not ruta and not ancla:
        continue
      objetivo = (p.parent / ruta).resolve() if ruta else p
      if not objetivo.exists():
        generado = re.search(r"(^|/)(notebooks/modelos/|results/)", rel_seguro(objetivo))
        (problemas if (estricto or not generado) else avisos).append(f"{rel(p)}: link roto a {dest}")
        continue
      if ancla and objetivo.suffix == ".md" and slug(ancla) not in anclas(objetivo):
        problemas.append(f"{rel(p)}: el ancla #{ancla} no existe en {rel_seguro(objetivo)}")


def rel_seguro(p):
  try:
    return rel(p)
  except ValueError:
    return str(p)


# ---------------------------------------------------------------- planes

def chequear_planes(todos, problemas):
  for p in (RAIZ / "Docs").glob("plan-*.md"):
    problemas.append(f"{rel(p)}: los planes no van en el repo")
  for p in todos:
    if p.suffix not in (".md", ".py", ".txt", ".toml", ".ipynb", ".html", ".yaml", ".yml", ".json", ".sh") or rel(p) == "scripts/check_docs.py":
      continue
    try:
      t = p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
      continue
    for n, linea in enumerate(t.splitlines(), 1):
      if PLAN_RE.search(linea):
        problemas.append(f"{rel(p)}:{n}: menciona un plan borrado ({PLAN_RE.search(linea).group(0)})")


# ---------------------------------------------------------------- números

def json_vigente(ruta):
  if not (RAIZ / ruta).exists():
    return False
  try:
    sys.path.insert(0, str(RAIZ))
    from src import resultados
    return resultados.estado(ruta) == "vigente"
  except Exception:  # noqa: BLE001
    return False


def respaldada(linea):
  js = JSON_RE.findall(linea)
  return bool(js) and all(json_vigente(j) for j in js)


def chequear_numeros(mds, problemas):
  for p in mds:
    r = rel(p)
    if r in SIN_NUMEROS:
      continue
    texto = p.read_text(encoding="utf-8")
    if r == "Docs/resultados.md":
      for m in TABLA_RE.finditer(texto):
        for linea in m.group(2).splitlines():
          if not linea.startswith("|") or set(linea.replace("|", "").strip()) <= set("-: "):
            continue
          celdas = [c.strip() for c in linea.strip().strip("|").split("|")]
          for c in celdas:
            if NUM_CELDA.match(c.replace("`", "")) and not respaldada(linea):
              problemas.append(f"{r}: número sin JSON vigente que lo respalde en la tabla {m.group(1)}: {c!r} (poné `pendiente`)")
    limpio = sin_codigo(AUTO_RE.sub("", texto))
    for n, linea in enumerate(limpio.splitlines(), 1):
      if "ok-numero" in linea or linea.lstrip().startswith("<!--"):
        continue
      if METRICA.search(linea) and not respaldada(linea):
        problemas.append(f"{r}: una métrica con número sin JSON vigente que la respalde: {METRICA.search(linea).group(0)!r}")


# ---------------------------------------------------------------- ejemplos

CON_DRY_RUN_GENERAL = {"reporte", "estadistica", "verificar", "docs", "resultados"}


def comandos_bash(mds):
  """[(archivo, línea de comando)] de los bloques ```bash, con las continuaciones con \\ unidas y sin comentarios."""
  out = []
  for p in mds:
    r = rel(p)
    if r in SIN_EJEMPLOS:
      continue
    for bloque in BASH_RE.finditer(p.read_text(encoding="utf-8")):
      lineas = re.sub(r"\\\s*\n\s*", " ", bloque.group(1)).splitlines()
      for ln in lineas:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
          continue
        try:
          partes = shlex.split(ln, comments=True)
        except ValueError:
          partes = ln.split()
        out.append((r, ln, partes))
  return out


def correr_cli(args):
  """(código, salida) de src.cli.main(args) en este proceso."""
  from src import cli
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    try:
      rc = cli.main(args)
    except SystemExit as e:
      rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except Exception as e:  # noqa: BLE001
      return 1, f"{type(e).__name__}: {e}"
  return (rc or 0), buf.getvalue()


def chequear_ejemplos(mds, problemas, avisos):
  os.chdir(RAIZ)
  sys.path.insert(0, str(RAIZ))
  os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
  from src import toolkit
  modelos = set(toolkit.modelos())
  vistos, n = set(), 0
  for r, ln, partes in comandos_bash(mds):
    if partes[:3] == ["python", "-m", "src.cli"]:
      args = partes[3:]
      if any(("<" in a and ">" in a) or a.startswith("[") for a in args):
        continue                                           # plantillas (<modelo> <acción>, [opciones])
      clave = " ".join(args)
      if clave in vistos:
        continue
      vistos.add(clave)
      if not args:
        continue
      if args[0] in modelos:
        probar = args if "--dry-run" in args else [*args, "--dry-run"]
      elif args[0] in ("cola",) and args[1:2] == ["correr"]:
        probar = args if "--dry-run" in args else [*args, "--dry-run"]
      elif args[0] == "bitacora" and args[1:2] == ["repetir"]:
        probar = args if "--dry-run" in args else [*args, "--dry-run"]
      elif args[0] in CON_DRY_RUN_GENERAL and args[0] != "resultados":
        probar = args if "--dry-run" in args else [*args, "--dry-run"]
      elif "--help" in args or "-h" in args:
        probar = args
      else:
        probar = [*args, "--help"]
      n += 1
      rc, salida = correr_cli(probar)
      if rc != 0:
        resto = " | ".join(salida.strip().splitlines()[-3:])
        problemas.append(f"{r}: `{ln}` falla con --dry-run (código {rc}): {resto}")
    elif partes[:2] == ["python", "-m"] and len(partes) > 2:
      mod = partes[2]
      if mod in ("pytest", "pip", "venv", "http.server", "jupyter", "ipykernel"):
        continue
      try:
        ok = importlib.util.find_spec(mod) is not None
      except (ImportError, ValueError):
        ok = False
      if not ok:
        problemas.append(f"{r}: `{ln}`: el módulo {mod} no existe")
    elif len(partes) > 1 and partes[0] == "python" and partes[1].endswith(".py"):
      if not (RAIZ / partes[1]).exists():
        problemas.append(f"{r}: `{ln}`: no existe {partes[1]}")
  return n


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--estricto", action="store_true", help="un link a un archivo generado que falta (notebooks, results) también es un error")
  ap.add_argument("--sin-ejemplos", action="store_true", help="no corre los comandos de los bloques bash (más rápido)")
  a = ap.parse_args(argv)
  if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  todos = archivos()
  mds = docs_md(todos)
  problemas, avisos = [], []
  chequear_links(mds, a.estricto, problemas, avisos)
  chequear_planes(todos, problemas)
  chequear_numeros(mds, problemas)
  n = 0 if a.sin_ejemplos else chequear_ejemplos(mds, problemas, avisos)
  for av in avisos:
    print(f"aviso: {av}")
  for pr in problemas:
    print(f"error: {pr}")
  print(f"check_docs: {len(mds)} documentos, {n} comandos de la CLI probados, {len(avisos)} avisos, {len(problemas)} problemas")
  return 1 if problemas else 0


if __name__ == "__main__":
  sys.exit(main())
