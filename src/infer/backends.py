"""Registro de backends de inferencia: dónde corre un modelo (cuda, cpu, rocm, mps, dml, npu) y cómo.

Un backend es una entrada del registro con:
  - detección: qué hay disponible en esta máquina (`detectar()`, `disponibles()`), sin cargar ningún modelo;
  - cómo se corre: con torch (`torch_device`: el nombre que entiende torch.device) y/o con ONNX Runtime
    (`ort_provider`: el nombre corto que entiende src.ort_model.make_session, y `ort_ep`: el execution provider).

Todos los `--device` del repo (src.bench, src.infer.bench_e2e, src.infer.bench_ort, src.ort_model, la CLI) salen de acá:
sumar un backend es una entrada nueva, sin tocar esos scripts (CONTRIBUTING, "Sumar un backend").

  from src.infer import backends
  backends.nombres()                  # ("cuda", "cpu", "rocm", "dml", "npu", ...) todo lo registrado
  backends.disponibles()              # los que se pueden usar en esta máquina
  backends.detectar()                 # {nombre: Estado(disponible, motivo)}
  backends.torch_device("rocm")       # "cuda": los builds ROCm usan la API torch.cuda
  backends.por_defecto()              # "cuda" / "rocm" / "mps" si hay GPU de torch, si no "cpu"

Sumar uno (p. ej. desde un test o desde un módulo propio):

  @backends.backend("mi_gpu", descripcion="...", torch_device="xpu", instalar="pip install ...")
  def _detectar_mi_gpu():
    return hay_mi_gpu(), "motivo si no hay"
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

NOMBRE_VITIS = "VitisAIExecutionProvider"


@dataclass(frozen=True)
class Estado:
  """Resultado de detectar un backend en esta máquina."""
  disponible: bool
  motivo: str = ""      # por qué no está (o qué se detectó, si está)


@dataclass(frozen=True)
class Backend:
  nombre: str
  descripcion: str
  detectar: Callable[[], tuple]            # () -> (disponible: bool, motivo: str)
  torch_device: Optional[str] = None       # str de torch.device ("cuda" para ROCm); None si torch no lo maneja
  ort_provider: Optional[str] = None       # nombre corto para src.ort_model.make_session (cpu, cuda, dml, vitisai)
  ort_ep: Optional[str] = None             # execution provider de ONNX Runtime (None: uno especial, ver ort_model)
  instalar: str = ""                       # qué instalar si falta (archivo de requirements o pista)

  @property
  def motores(self):
    return tuple(m for m, v in (("torch", self.torch_device), ("ort", self.ort_provider)) if v)


_REGISTRO: dict[str, Backend] = {}


# ---------------------------------------------------------------- registro

def registrar(b: Backend) -> Backend:
  """Agrega (o reemplaza) un backend. Falla con un mensaje claro si no se puede correr con ningún motor."""
  if not b.nombre or not b.nombre.islower() or " " in b.nombre:
    raise ValueError(f"backend {b.nombre!r}: el nombre va en minúsculas y sin espacios (es un tag de dispositivo)")
  if not callable(b.detectar):
    raise ValueError(f"backend {b.nombre}: falta la función de detección (() -> (disponible, motivo))")
  if not (b.torch_device or b.ort_provider):
    raise ValueError(f"backend {b.nombre}: declará torch_device (torch) o ort_provider (ONNX Runtime), si no no corre nada")
  _REGISTRO[b.nombre] = b
  return b


def backend(nombre, descripcion="", torch_device=None, ort_provider=None, ort_ep=None, instalar=""):
  """Decorador: la función decorada es la detección, `() -> (disponible, motivo)`."""
  def deco(fn):
    registrar(Backend(nombre, descripcion, fn, torch_device, ort_provider, ort_ep, instalar))
    return fn
  return deco


def quitar(nombre):
  """Saca un backend del registro (para los que se registran desde un test)."""
  _REGISTRO.pop(nombre, None)


def obtener(nombre) -> Backend:
  if nombre not in _REGISTRO:
    raise ValueError(f"backend {nombre!r} desconocido; registrados: {', '.join(_REGISTRO)}")
  return _REGISTRO[nombre]


def nombres(motor=None) -> tuple:
  """Nombres registrados, en orden de registro. motor="torch" u "ort": solo los que corren con ese motor."""
  return tuple(n for n, b in _REGISTRO.items() if motor is None or motor in b.motores)


def todos() -> list:
  return list(_REGISTRO.values())


# ---------------------------------------------------------------- detección

def estado(nombre) -> Estado:
  b = obtener(nombre)
  try:
    ok, motivo = b.detectar()
  except Exception as e:  # noqa: BLE001  (una detección rota no tiene que romper el listado)
    ok, motivo = False, f"falló la detección: {type(e).__name__}: {e}"
  return Estado(bool(ok), motivo)


def detectar() -> dict:
  """{nombre: Estado} de todo lo registrado."""
  return {n: estado(n) for n in _REGISTRO}


def disponibles(motor=None) -> tuple:
  """Nombres de los backends que se pueden usar en esta máquina (motor="torch"|"ort" filtra por cómo corren)."""
  return tuple(n for n in nombres(motor) if estado(n).disponible)


def por_defecto() -> str:
  """El backend de torch que se usa sin --device: el primero disponible entre cuda, rocm, mps y cpu (lo que hacía torch.cuda.is_available)."""
  for n in ("cuda", "rocm", "mps"):
    if n in _REGISTRO and estado(n).disponible:
      return n
  return "cpu"


def tabla() -> list:
  """Filas para el reporte y la CLI: una por backend, con su disponibilidad y cómo corre."""
  out = []
  for b in _REGISTRO.values():
    e = estado(b.nombre)
    out.append({"backend": b.nombre, "descripcion": b.descripcion, "disponible": e.disponible, "motivo": e.motivo,
                "motores": list(b.motores), "torch_device": b.torch_device, "ort_provider": b.ort_provider,
                "instalar": b.instalar})
  return out


# ---------------------------------------------------------------- cómo se corre

def torch_device(nombre) -> str:
  """El string de torch.device del backend. Falla con un mensaje claro si torch no lo corre (dml, npu: son de ONNX Runtime)."""
  b = obtener(nombre)
  if not b.torch_device:
    raise ValueError(f"el backend {nombre!r} no corre con torch; corre con ONNX Runtime (provider {b.ort_provider}): "
                     "usá src.infer.bench_ort o src.ort_model")
  return b.torch_device


def resolver_ort(nombre) -> Backend:
  """El backend de un nombre de backend o de provider corto (p. ej. "vitisai" -> npu)."""
  if nombre in _REGISTRO:
    b = _REGISTRO[nombre]
  else:
    b = next((x for x in _REGISTRO.values() if x.ort_provider == nombre), None)
    if b is None:
      raise ValueError(f"backend o provider {nombre!r} desconocido; registrados: {', '.join(_REGISTRO)}")
  if not b.ort_provider:
    raise ValueError(f"el backend {b.nombre!r} no corre con ONNX Runtime (solo torch, device {b.torch_device})")
  return b


def ort_providers() -> tuple:
  """Nombres cortos de provider de ONNX Runtime registrados (cpu, cuda, dml, vitisai...)."""
  return tuple(b.ort_provider for b in _REGISTRO.values() if b.ort_provider)


def exigir(nombre):
  """Falla con un mensaje claro (y qué instalar) si el backend no está disponible en esta máquina."""
  e = estado(nombre)
  if not e.disponible:
    b = obtener(nombre)
    raise SystemExit(f"backend {nombre}: no está disponible ({e.motivo})" + (f"; instalar: {b.instalar}" if b.instalar else ""))


def agregar_device(ap, motor="torch", flag="--device", default=None, **kw):
  # motor="torch" ofrece solo los de torch; motor="torch+ort" ofrece todos (los de ONNX Runtime fallan con un mensaje que dice qué usar)
  """Agrega --device a un argparse con las opciones salidas del registro (evaluadas ahora: ve los backends registrados
  hasta este momento). Con motor="torch" el default es por_defecto(); con "ort", "cpu"."""
  opciones = nombres(None if motor == "torch+ort" else motor)
  ap.add_argument(flag, choices=opciones, default=default or (por_defecto() if motor.startswith("torch") else "cpu"),
                  help="backend (src.infer.backends): " + ", ".join(opciones), **kw)


def dispositivo(nombre):
  """torch.device de un backend de torch, verificando antes que esté disponible."""
  import torch
  b = obtener(nombre)
  if not b.torch_device:
    raise SystemExit(f"--device {nombre}: este backend corre con ONNX Runtime (provider {b.ort_provider}), no con torch. Para medir "
                     "usá la cadena de ONNX Runtime: `python -m src.infer.bench_ort --enc {0} --dec {0}` (o src.ort_model)".format(nombre))
  exigir(nombre)
  dev = torch.device(torch_device(nombre))
  from src.common import aviso_mps
  aviso_mps(dev)   # MPS: recomienda PYTORCH_ENABLE_MPS_FALLBACK=1, sin setearla
  return dev


# ---------------------------------------------------------------- los backends del repo

def _torch_cuda(hip):
  import torch
  if not torch.cuda.is_available():
    return False, "torch no ve ninguna GPU (torch.cuda.is_available() es False)"
  es_hip = bool(getattr(torch.version, "hip", None))
  if es_hip != hip:
    return False, ("esta GPU es ROCm (usar --device rocm)" if es_hip else "esta GPU es NVIDIA (usar --device cuda)")
  return True, torch.cuda.get_device_name(0)


def _ort_ep(ep):
  try:
    import onnxruntime as ort
  except ImportError:
    return False, "falta onnxruntime"
  hay = ort.get_available_providers()
  return (ep in hay), (f"{ep} disponible" if ep in hay else f"onnxruntime {ort.__version__} no trae {ep} (hay {hay})")


@backend("cuda", "GPU NVIDIA con CUDA (torch; ONNX Runtime con CUDAExecutionProvider)", torch_device="cuda",
         ort_provider="cuda", ort_ep="CUDAExecutionProvider", instalar="requirements.txt (torch con CUDA)")
def _detectar_cuda():
  return _torch_cuda(hip=False)


@backend("cpu", "CPU (torch; ONNX Runtime con CPUExecutionProvider)", torch_device="cpu", ort_provider="cpu",
         ort_ep="CPUExecutionProvider")
def _detectar_cpu():
  return True, "siempre disponible"


@backend("rocm", "GPU AMD con ROCm (torch usa la API torch.cuda; ver requirements-rocm.txt)", torch_device="cuda",
         instalar="requirements-rocm.txt, y TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 en gfx1152")
def _detectar_rocm():
  return _torch_cuda(hip=True)


@backend("mps", "GPU de Apple Silicon por Metal (torch MPS). Sin CUDA graphs ni pinned memory; tiempos con perf_counter y "
         "torch.mps.synchronize(); fp16 en autocast, bf16 solo si el sistema lo soporta. Recomendado: "
         "PYTORCH_ENABLE_MPS_FALLBACK=1 para las ops sin soporte (no se setea solo)", torch_device="mps",
         instalar="torch para macOS arm64; export PYTORCH_ENABLE_MPS_FALLBACK=1")
def _detectar_mps():
  import torch
  mps = getattr(torch.backends, "mps", None)
  if mps is None or not mps.is_available():
    return False, "torch.backends.mps.is_available() es False (hace falta un Mac con Apple Silicon y torch para macOS)"
  return True, "Apple GPU (MPS)"


@backend("dml", "iGPU/GPU por DirectML (solo ONNX Runtime, DmlExecutionProvider)", ort_provider="dml",
         ort_ep="DmlExecutionProvider", instalar="onnxruntime-directml (o onnxruntime-windowsml, requirements-npu.txt)")
def _detectar_dml():
  return _ort_ep("DmlExecutionProvider")


@backend("npu", "NPU AMD XDNA 2 por ONNX Runtime con VitisAIExecutionProvider (sin probar en esta máquina; ver "
         "src/ort_model.make_session)", ort_provider="vitisai", instalar="requirements-npu.txt y ML2_VITISAI_EP_DIR")
def _detectar_npu():
  try:
    import onnxruntime as ort
  except ImportError:
    return False, "falta onnxruntime"
  if not (hasattr(ort, "register_execution_provider_library") and hasattr(ort, "get_ep_devices")):
    return False, f"onnxruntime {ort.__version__} no tiene register_execution_provider_library (usar onnxruntime-windowsml)"
  ep_dir = os.environ.get("ML2_VITISAI_EP_DIR")
  if not ep_dir:
    return False, "falta ML2_VITISAI_EP_DIR (carpeta con onnxruntime_vitisai_ep.dll)"
  if not (Path(ep_dir) / "onnxruntime_vitisai_ep.dll").exists():
    return False, f"no existe {Path(ep_dir) / 'onnxruntime_vitisai_ep.dll'}"
  return True, f"EP en {ep_dir} (la NPU se confirma al abrir la sesión)"
