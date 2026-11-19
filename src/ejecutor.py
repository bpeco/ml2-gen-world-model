"""Ejecuta un Plan (lista de comandos de subproceso) con candado de GPU, cola y bitácora.

Lo usan Toolkit (notebooks) y la CLI. Un Plan es lo que una acción de la CLI va a correr:
  - si la acción usa la GPU: rechaza con el árbol sucio (salvo permitir_sucio), toma results/.gpu.lock o, si está
    tomado, se encola en results/queue.json (o espera con esperar=True);
  - corre cada comando como subproceso (frena en el primero que falla);
  - escribe una línea en results/bitacora.jsonl y anota "_bitacora" en los JSON de resultados que escribió.
"""
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src import bitacora, cola, rutas

PY = "python"  # en lo que se imprime y registra; al lanzar se reemplaza por sys.executable


@dataclass
class Cmd:
  argv: list
  env: dict = field(default_factory=dict)

  def texto(self):
    pre = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in self.env.items())
    return (pre + " " if pre else "") + shlex.join(str(a) for a in self.argv)


@dataclass
class Plan:
  comandos: list                  # [Cmd]
  modelo: str | None = None
  accion: str = ""
  run: str | None = None
  device: str = "cpu"             # device resuelto: cuda, rocm, dml, npu o cpu
  receta_hash: str | None = None
  cli_argv: list = field(default_factory=list)   # los argumentos de `python -m src.cli` que lo producen
  avisos: list = field(default_factory=list)
  meta: dict = field(default_factory=dict)       # datos de la acción que van a la bitácora y a los JSON (p. ej. de dónde salió la T)
  error: str | None = None        # el pedido no se puede ejecutar (solo se puede imprimir con dry_run)

  @property
  def gpu(self):
    return self.device != "cpu"


@dataclass
class Resultado:
  comandos: list                  # [[argv]] lo que se ejecuta (o ejecutaría)
  codigo: int = 0
  id: str | None = None
  estado: str = "ok"              # ok | fallada | dry-run | encolada | rechazada
  cola_id: str | None = None
  salidas: list = field(default_factory=list)

  @property
  def ok(self):
    return self.codigo == 0


def argv_texto(cli_argv):
  return "python -m src.cli " + shlex.join(cli_argv)


def imprimir_plan(plan, salida=print):
  for av in plan.avisos:
    salida(f"# aviso: {av}")
  for c in plan.comandos:
    salida(c.texto())


def _entorno(c):
  env = dict(os.environ)
  env.update({k: str(v) for k, v in c.env.items()})
  env.setdefault("PYTHONUNBUFFERED", "1")
  env.setdefault("PYTHONIOENCODING", "utf-8")
  return env


def correr_comando(c, salida=None):
  """Lanza un comando y reenvía su salida línea a línea (en un notebook el subproceso no escribe en la celda solo)."""
  salida = salida or sys.stdout
  argv = [sys.executable if (a == PY and i == 0) else str(a) for i, a in enumerate(c.argv)]
  p = subprocess.Popen(argv, env=_entorno(c), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  try:
    for linea in iter(p.stdout.readline, b""):
      salida.write(linea.decode("utf-8", errors="replace"))
      salida.flush()
  except KeyboardInterrupt:
    p.terminate()
    raise
  finally:
    p.stdout.close()
  return p.wait()


def actualizar_configs(salida=print):
  """Regenera results/configs.json (una fila por modelo con resultados) con `python -m src.configs`; un fallo no frena la acción."""
  r = subprocess.run([sys.executable, "-m", "src.configs"], capture_output=True, text=True, encoding="utf-8", errors="replace")
  if r.returncode != 0:
    ult = (r.stderr.strip().splitlines() or ["?"])[-1]
    salida(f"aviso: no pude actualizar results/configs.json: {ult}")


def resolver_device(device=None):
  """cuda | rocm | dml | npu | mps | cpu. Sin device: lo que elegiría src.common.device() (ML2_DEVICE, CUDA, CPU)."""
  if device:
    return device
  forzado = os.environ.get("ML2_DEVICE")
  if forzado:
    return forzado.split(":")[0]
  from src.infer import backends
  try:
    return backends.por_defecto()   # cuda -> rocm -> mps -> cpu
  except ImportError:
    return "cpu"


def _entrada_base(plan, argv_cli, commit, limpio, permitir_sucio):
  return {"id": bitacora.nuevo_id(), "fecha": datetime.now().isoformat(timespec="seconds"), "argv": list(argv_cli),
          "argv_texto": argv_texto(argv_cli), "modelo": plan.modelo, "accion": plan.accion, "run": plan.run,
          "device": plan.device, "receta_hash": plan.receta_hash, "commit": commit, "limpio": limpio,
          "permitir_sucio": bool(permitir_sucio and not limpio)}


def _hash_de_train(plan):
  """El hash de la receta de una corrida ya entrenada (results/train_<run>.json), si la acción no lo calculó."""
  if plan.receta_hash or not plan.run:
    return plan.receta_hash
  f = rutas.results() / f"train_{plan.run}.json"
  try:
    return json.loads(f.read_text(encoding="utf-8")).get("receta_hash")
  except (OSError, json.JSONDecodeError):
    return None


def ejecutar(plan, esperar=False, encolar=False, permitir_sucio=False, dry_run=False, desde_cola=False,
             salida=print, intervalo=5.0, meta=None):
  """Ejecuta el plan. Devuelve un Resultado; nunca lanza por un comando fallado (el código queda en .codigo)."""
  argv_cli = plan.cli_argv
  comandos = [list(c.argv) for c in plan.comandos]
  if dry_run:
    imprimir_plan(plan, salida)
    return Resultado(comandos, 0, estado="dry-run")
  for av in plan.avisos:
    salida(f"aviso: {av}")
  commit, limpio = bitacora.git_estado()
  base = {**_entrada_base(plan, argv_cli, commit, limpio, permitir_sucio), **(meta or {})}
  if plan.meta:
    base["meta"] = plan.meta

  def asentar(**kw):
    e = {**base, "comandos": [{"argv": c.argv, "env": c.env} for c in plan.comandos], "duracion_s": 0.0,
         "salida": None, "escribio": [], **kw}
    bitacora.registrar(e)
    return e

  if plan.gpu and not limpio and not permitir_sucio:
    asentar(rechazada="árbol sucio: hay cambios sin commitear en src/ o scripts/ (usá --permitir-sucio, el resultado no cuenta)")
    salida("rechazada: hay cambios sin commitear en src/ o scripts/; una acción de GPU solo corre con el árbol limpio "
           "(commiteá, o usá --permitir-sucio: queda anotado y el resultado no cuenta como vigente)")
    return Resultado(comandos, 2, base["id"], "rechazada")

  tomado = False
  if plan.gpu:
    nombre = plan.run or plan.modelo or "?"
    if encolar:
      ok, info = False, {}
    elif esperar:
      cola.esperar_candado(nombre, plan.accion, intervalo=intervalo, avisar=salida)
      ok = True
    else:
      ok, info = cola.tomar_candado(nombre, plan.accion, avisar=salida)
    if not ok:
      cuerpo = [a for a in argv_cli if a not in ("--esperar", "--encolar")]
      motivo = "pedido con --encolar" if encolar else f"GPU ocupada por {info.get('run')} / {info.get('accion')}"
      item = cola.encolar(cuerpo, plan.modelo, plan.accion, plan.run, motivo=motivo)
      asentar(encolada=item["id"])
      salida(f"encolada ({item['id']}): {motivo}; se ejecuta con `python -m src.cli cola correr`")
      return Resultado(comandos, 0, base["id"], "encolada", item["id"])
    tomado = True

  antes = bitacora.instantanea(plan.run)
  t0 = time.time()
  salidas, codigo = [], 0
  try:
    for c in plan.comandos:
      salida("$ " + c.texto())
      cod = correr_comando(c)
      salidas.append(cod)
      if cod != 0:
        codigo = cod
        salida(f"falló con código {cod}: {c.texto()}")
        break
  except KeyboardInterrupt:
    codigo = 130
    salida("interrumpida")
  finally:
    dur = time.time() - t0
    if codigo == 0 and plan.accion in ("entrenar", "evaluar", "bench") and rutas.results() == Path("results"):
      actualizar_configs(salida)   # results/configs.json (lo que lee el reporte) se deriva de lo que hay en results/
    despues = bitacora.instantanea(plan.run)
    ajenos = bitacora.ajenos(t0)  # lo que escribió otra acción que terminó mientras corría esta no es de esta
    escritos = [f for f in bitacora.archivos_escritos(antes, despues) if bitacora._relativa(f) not in ajenos]
    entrada = {**base, "comandos": [{"argv": c.argv, "env": c.env, "salida": s}
                                    for c, s in zip(plan.comandos, salidas + [None] * len(plan.comandos))],
               "duracion_s": round(dur, 2), "salida": codigo, "escribio": escritos,
               "receta_hash": _hash_de_train(plan)}
    if desde_cola:
      entrada["desde_cola"] = True
    anotados = bitacora.anotar_json(escritos, entrada)
    if anotados:
      entrada["anotados"] = [bitacora._relativa(a) for a in anotados]
    bitacora.registrar(entrada)
    if tomado:
      cola.liberar_candado()
  return Resultado(comandos, codigo, base["id"], "ok" if codigo == 0 else "fallada", salidas=salidas)
