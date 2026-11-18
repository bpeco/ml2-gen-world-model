"""Bitácora de acciones: results/bitacora.jsonl, una línea por acción de la CLI (D2, K6).

Cada línea: id, fecha, argv exacto de la CLI, comandos que corrió (con su código de salida), commit de git, si el
árbol estaba limpio (cambios en src/ o scripts/), hash de la receta, device, duración, código de salida y archivos que
escribió (diff de results/ y de checkpoints/<run>/ antes y después). Las entradas falladas no se borran nunca.
A cada JSON de resultados que la acción escribió se le agrega "_bitacora": {id, comando, commit}.

  python -m src.cli bitacora ver [-n 20]
  python -m src.cli bitacora verificar          cada resultado `vigente` tiene entrada con salida 0, árbol limpio y commit en main
  python -m src.cli bitacora repetir ID [--dry-run]
"""
import json
import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

from src import resultados, rutas

CARPETAS_SUCIAS = ("src", "scripts")
# JSON de results/ que no son resultados medidos (manifiestos o cuyas claves son nombres): no se anotan
SIN_ANOTAR = {"checkpoints.json", "configs.json", "estado.json", "queue.json", "clips_eval.json", "val_split.json"}
SIN_ANOTAR_DIRS = ("cache", "logs")
SIN_RASTREAR = {"bitacora.jsonl", ".gpu.lock", "queue.json", "queue.json.lock", "queue.json.tmp"}


# ---------------------------------------------------------------- git

def _git(*args):
  r = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
  return r.returncode, r.stdout.strip()


def git_estado():
  """(commit, limpio): commit HEAD y si src/ y scripts/ no tienen cambios (ni archivos sin rastrear)."""
  rc, commit = _git("rev-parse", "HEAD")
  if rc != 0:
    return None, False
  rc, sucio = _git("status", "--porcelain", "--", *CARPETAS_SUCIAS)
  return commit, rc == 0 and not sucio


def es_ancestro_de_main(commit):
  if not commit:
    return False
  for rama in ("main", "origin/main"):
    if _git("rev-parse", "--verify", "--quiet", rama)[0] == 0 and _git("merge-base", "--is-ancestor", commit, rama)[0] == 0:
      return True
  return False


# ---------------------------------------------------------------- archivos escritos

def _recorrer(raiz):
  out = {}
  raiz = Path(raiz)
  if not raiz.exists():
    return out
  for d, _, files in os.walk(raiz):
    for f in files:
      p = Path(d) / f
      try:
        st = p.stat()
      except OSError:
        continue
      out[p.as_posix()] = (st.st_mtime_ns, st.st_size)
  return out


def instantanea(run=None):
  """{ruta: (mtime, tamaño)} de results/ y de checkpoints/<run>/."""
  snap = {k: v for k, v in _recorrer(rutas.results()).items() if Path(k).name not in SIN_RASTREAR}
  if run:
    snap.update(_recorrer(rutas.ckpt() / run))
  return snap


def archivos_escritos(antes, despues):
  return sorted(k for k, v in despues.items() if antes.get(k) != v)


def _relativa(ruta):
  p = Path(ruta)
  try:
    return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
  except ValueError:
    return p.as_posix()


def anotar_json(archivos, entrada):
  """Agrega "_bitacora" a cada JSON de resultados (dict) de `archivos`. Devuelve los anotados."""
  res = rutas.results().resolve()
  hechos = []
  for f in archivos:
    p = Path(f)
    if p.suffix != ".json" or p.name in SIN_ANOTAR or not p.exists():
      continue
    try:
      rel = p.resolve().relative_to(res)
    except ValueError:
      continue
    if rel.parts and rel.parts[0] in SIN_ANOTAR_DIRS:
      continue
    try:
      d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
      continue
    if not isinstance(d, dict):
      continue
    d["_bitacora"] = {"id": entrada["id"], "comando": entrada["argv_texto"], "commit": entrada["commit"],
                      **({"meta": entrada["meta"]} if entrada.get("meta") else {})}
    p.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    hechos.append(f)
  return hechos


# ---------------------------------------------------------------- registro

def nuevo_id():
  return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]


def registrar(entrada):
  p = rutas.bitacora()
  p.parent.mkdir(parents=True, exist_ok=True)
  with open(p, "a", encoding="utf-8") as f:
    f.write(json.dumps(entrada, ensure_ascii=False) + "\n")


def leer():
  p = rutas.bitacora()
  if not p.exists():
    return []
  out = []
  for linea in p.read_text(encoding="utf-8").splitlines():
    if linea.strip():
      try:
        out.append(json.loads(linea))
      except json.JSONDecodeError:
        pass
  return out


def buscar(id_):
  """Entrada por id (o prefijo único)."""
  hits = [e for e in leer() if e["id"] == id_] or [e for e in leer() if e["id"].startswith(id_)]
  if not hits:
    raise KeyError(f"no hay una entrada con id {id_!r} en {rutas.bitacora()}")
  if len({e["id"] for e in hits}) > 1:
    raise KeyError(f"el prefijo {id_!r} es ambiguo: {', '.join(e['id'] for e in hits)}")
  return hits[-1]


def ver(n=20):
  lineas = []
  for e in leer()[-n:]:
    estado = "encolada" if e.get("encolada") else ("rechazada" if e.get("rechazada") else f"salida {e.get('salida')}")
    sucio = "" if e.get("limpio") else " [árbol sucio]"
    lineas.append(f"{e['id']}  {e['fecha']}  {estado:10s} {e.get('duracion_s') or 0:7.1f}s  {e['argv_texto']}{sucio}")
  return "\n".join(lineas) or "(bitácora vacía)"


def repetir(id_, dry_run=False):
  """Argumentos de la CLI de la entrada, para relanzarla tal cual. Con dry_run solo los devuelve."""
  return list(buscar(id_)["argv"])


# ---------------------------------------------------------------- verificar

def verificar(estado_file=None):
  """Problemas de la bitácora respecto de los resultados `vigente` (lista de (archivo, motivo)); vacía = todo bien."""
  reg = resultados.cargar(estado_file or resultados.ARCHIVO)
  entradas = leer()
  escritores = {}
  for e in entradas:
    for f in e.get("escribio", []):
      escritores[_relativa(f)] = e  # la última que lo escribió
  problemas = []
  # un resultado que existe y no está marcado de otro modo es vigente: todos necesitan su entrada
  vigentes = {k for k, v in reg.items() if v.get("estado") == "vigente"}
  vigentes |= {k for k in resultados.existentes(rutas.results()) if k not in reg}
  for k in sorted(vigentes):
    e = escritores.get(k)
    if e is None:
      problemas.append((k, "ninguna entrada de la bitácora lo escribió"))
    elif e.get("salida") != 0:
      problemas.append((k, f"la entrada {e['id']} terminó con salida {e.get('salida')}"))
    elif not e.get("limpio") or e.get("permitir_sucio"):
      problemas.append((k, f"la entrada {e['id']} corrió con el árbol sucio"))
    elif not es_ancestro_de_main(e.get("commit")):
      problemas.append((k, f"el commit {str(e.get('commit'))[:10]} de la entrada {e['id']} no está en main"))
  return problemas
