"""Candado de GPU y cola de trabajos (una sola tarea de GPU a la vez).

  results/.gpu.lock   lo crea de forma atómica (O_EXCL) la acción que usa la GPU: pid, corrida, acción, hora.
                      Si su pid ya no existe, el candado está huérfano: se libera con un aviso.
  results/queue.json  acciones que encontraron la GPU ocupada: [{id, argv, modelo, accion, run, estado, ...}].
                      Solo `python -m src.cli cola correr` las ejecuta, de a una, y frena ante una falla.

El archivo de la cola es chico y se modifica con otro candado (queue.json.lock, también O_EXCL).
"""
import contextlib
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

from src import rutas
from src.status import pid_alive


def _ahora():
  return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- candado de GPU

def leer_candado():
  """Contenido del candado, o None si no existe (o está ilegible: se trata como tomado por un desconocido)."""
  p = rutas.candado()
  if not p.exists():
    return None
  try:
    return json.loads(p.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return {"pid": None, "run": "?", "accion": "?", "hora": "?"}


def _crear(p, info):
  p.parent.mkdir(parents=True, exist_ok=True)
  fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY)  # falla si ya existe: atómico entre procesos
  with os.fdopen(fd, "w", encoding="utf-8") as f:
    json.dump(info, f)


def tomar_candado(run, accion, pid=None, avisar=print):
  """Toma el candado de GPU. Devuelve (True, info) si lo tomó, o (False, info_del_que_lo_tiene)."""
  p = rutas.candado()
  info = {"pid": os.getpid() if pid is None else pid, "run": run, "accion": accion, "hora": _ahora()}
  for _ in range(3):
    try:
      _crear(p, info)
      return True, info
    except FileExistsError:
      actual = leer_candado()
      if actual is None:  # se liberó entre medio
        continue
      if actual.get("pid") is not None and not pid_alive(actual["pid"]):
        avisar(f"aviso: candado huérfano de {actual.get('run')} / {actual.get('accion')} (pid {actual['pid']} ya no existe, "
               f"desde {actual.get('hora')}): lo libero")
        with contextlib.suppress(FileNotFoundError):
          p.unlink()
        continue
      return False, actual
  return False, leer_candado() or {}


def liberar_candado(pid=None):
  """Libera el candado solo si es de este proceso (o del pid indicado)."""
  actual = leer_candado()
  if actual is None:
    return False
  if actual.get("pid") != (os.getpid() if pid is None else pid):
    return False
  with contextlib.suppress(FileNotFoundError):
    rutas.candado().unlink()
  return True


def esperar_candado(run, accion, intervalo=5.0, avisar=print, pid=None, maximo=None):
  """Bloquea hasta tomar el candado. Devuelve su info; con `maximo` segundos lanza TimeoutError."""
  t0, avisado = time.time(), None
  while True:
    ok, info = tomar_candado(run, accion, pid=pid, avisar=avisar)
    if ok:
      return info
    if avisado != info.get("pid"):
      avisar(f"GPU ocupada por {info.get('run')} / {info.get('accion')} (pid {info.get('pid')}): espero")
      avisado = info.get("pid")
    if maximo is not None and time.time() - t0 > maximo:
      raise TimeoutError("la GPU sigue ocupada")
    time.sleep(intervalo)


# ---------------------------------------------------------------- cola

@contextlib.contextmanager
def _candado_cola(intervalo=0.05, maximo=10.0):
  p = Path(str(rutas.cola()) + ".lock")
  p.parent.mkdir(parents=True, exist_ok=True)
  t0 = time.time()
  while True:
    try:
      fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
      os.close(fd)
      break
    except FileExistsError:
      if time.time() - t0 > maximo:  # un candado de cola viejo (proceso muerto): el trabajo es de milisegundos
        with contextlib.suppress(FileNotFoundError):
          p.unlink()
        continue
      time.sleep(intervalo)
  try:
    yield
  finally:
    with contextlib.suppress(FileNotFoundError):
      p.unlink()


def cargar_cola():
  p = rutas.cola()
  if not p.exists():
    return []
  try:
    return json.loads(p.read_text(encoding="utf-8"))
  except json.JSONDecodeError:
    return []


def _guardar_cola(items):
  p = rutas.cola()
  p.parent.mkdir(parents=True, exist_ok=True)
  tmp = Path(str(p) + ".tmp")
  tmp.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
  os.replace(tmp, p)


def encolar(argv, modelo=None, accion=None, run=None, motivo=""):
  """Agrega una acción de la CLI (sus argumentos, sin --esperar ni --encolar) al final de la cola. Devuelve el item."""
  item = {"id": uuid.uuid4().hex[:8], "argv": list(argv), "modelo": modelo, "accion": accion, "run": run,
          "estado": "pendiente", "encolada": _ahora(), "motivo": motivo}
  with _candado_cola():
    items = cargar_cola()
    items.append(item)
    _guardar_cola(items)
  return item


def pendientes():
  return [i for i in cargar_cola() if i.get("estado") == "pendiente"]


def actualizar(id_, **campos):
  with _candado_cola():
    items = cargar_cola()
    for i in items:
      if i["id"] == id_:
        i.update(campos)
    _guardar_cola(items)


def quitar(id_):
  """Saca un item de la cola. Devuelve True si existía."""
  with _candado_cola():
    items = cargar_cola()
    nuevos = [i for i in items if i["id"] != id_]
    if len(nuevos) != len(items):
      _guardar_cola(nuevos)
    return len(nuevos) != len(items)


def cola_texto():
  items = cargar_cola()
  cand = leer_candado()
  lineas = []
  if cand:
    vivo = pid_alive(cand.get("pid")) if cand.get("pid") else False
    lineas.append(f"GPU: ocupada por {cand.get('run')} / {cand.get('accion')} (pid {cand.get('pid')}, desde {cand.get('hora')})"
                  + ("" if vivo else " [huérfano]"))
  else:
    lineas.append("GPU: libre")
  if not items:
    lineas.append("cola: vacía")
  for i in items:
    lineas.append(f"  {i['id']}  {i.get('estado', '?'):9s} {i.get('encolada', '')}  python -m src.cli {' '.join(i['argv'])}"
                  + (f"  (código {i['codigo']})" if i.get("codigo") else ""))
  return "\n".join(lineas)
