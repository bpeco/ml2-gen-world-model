"""Registro de evaluaciones: qué mide cada una, a qué modelos aplica y de dónde sale (CONTRIBUTING, "Sumar una evaluación").

Hay dos clases de entrada:
  - las que existen (video: CE, rollouts, LPIPS, FVD; latente: eval_latent): apuntan a su módulo (`modulo`/`entrada`),
    sin reescribirlas; la CLI las sigue lanzando como subprocesos (src/toolkit.py);
  - las nuevas, que se registran con una función y cuatro piezas obligatorias:

      @evaluador("nitidez", aplica_a=("video",), menor_es_mejor=False,
                 sintetico=mis_datos, referencia=mi_referencia)       # ic: por defecto src.estadistica.ic_media_segmento
      def nitidez(pred, real):
        return valores_por_clip                                     # (N,) una medida por clip

    función      fn(pred, real) -> (N,) valores por clip (np.ndarray). pred y real salen del modelo y de los datos.
    sintético    sintetico(rng) -> (pred, real, seg): datos chicos en los que `pred` es mejor que la referencia trivial
                 por construcción; seg (N,) es el segmento de cada clip (el bootstrap remuestrea segmentos, D14).
    referencia   referencia(real) -> pred_trivial: la predicción más tonta (copiar el último frame, la media...).
                 Toda evaluación se lee al lado de su referencia (D12, D14).
    IC           ic(valores, seg) -> {"mean", "lo", "hi", ...}; por defecto el IC 95% por segmento de src.estadistica.

  `verificar(nombre)` corre la función sobre los datos sintéticos, calcula la referencia y el IC y comprueba que el modelo
  sintético le gana a la referencia. Lo usa el contrato (tests/test_evaluadores.py) y `verificar` de la CLI.

Qué tiene que llamar quien lista (toolkit, CLI, reporte):
  evaluadores.listar()                 todas las entradas (en orden de registro)
  evaluadores.para("latente")          las que aplican a una evaluación ("video" | "latente") o a un kind (p. ej. "gpt")
  evaluadores.para_modelo("genie")     lo mismo, tomando EVALUACION del módulo del modelo
  evaluadores.tabla()                  filas (dict) para el reporte
"""
import importlib
import re
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

Ref = Union[Callable, str, None]   # una función, o un puntero "paquete.modulo:funcion" (o un texto descriptivo)
IC_DEFECTO = "src.estadistica:ic_media_segmento"


@dataclass(frozen=True)
class Evaluador:
  nombre: str
  aplica_a: tuple                        # "video", "latente", "*" o kinds de src/models (p. ej. "gpt")
  descripcion: str = ""
  fn: Optional[Callable] = None          # evaluadores nuevos: fn(pred, real) -> (N,)
  modulo: Optional[str] = None           # evaluadores que existen: módulo con la evaluación completa
  entrada: str = "main"                  # función del módulo que la corre (main(argv))
  sintetico: Ref = None
  referencia: Ref = None
  ic: Ref = None
  menor_es_mejor: bool = True

  @property
  def propio(self):
    """True si es una función registrada con @evaluador; False si apunta a un módulo que ya existía."""
    return self.fn is not None

  def punto_de_entrada(self):
    """La función ejecutable: fn, o `entrada` del módulo (se importa recién acá)."""
    if self.fn is not None:
      return self.fn
    return getattr(importlib.import_module(self.modulo), self.entrada)


_REGISTRO: dict[str, Evaluador] = {}


# ---------------------------------------------------------------- registro

def _resolver(ref: Ref):
  """Función de un puntero "modulo:funcion"; una función pasa igual; un texto sin ':' es descriptivo (devuelve None)."""
  if callable(ref):
    return ref
  if isinstance(ref, str) and re.fullmatch(r"[\w.]+:\w+", ref):
    mod, _, fn = ref.partition(":")
    return getattr(importlib.import_module(mod), fn)
  return None


def registrar(e: Evaluador) -> Evaluador:
  """Agrega (o reemplaza) una entrada; falla con un mensaje claro si falta algo de lo obligatorio."""
  if not e.nombre or " " in e.nombre:
    raise ValueError(f"evaluador {e.nombre!r}: el nombre va sin espacios")
  if not e.aplica_a:
    raise ValueError(f"evaluador {e.nombre}: falta aplica_a (p. ej. ('video',) o ('latente',))")
  if e.fn is None and not e.modulo:
    raise ValueError(f"evaluador {e.nombre}: falta la función (con @evaluador) o el módulo que ya existe (modulo=)")
  if e.fn is not None:  # una evaluación nueva trae su prueba, su referencia y su IC: sin ellas no se lee
    faltan = [n for n, v in (("sintetico", e.sintetico), ("referencia", e.referencia)) if not callable(v)]
    if faltan:
      raise ValueError(f"evaluador {e.nombre}: falta {' y '.join(faltan)} (una función cada una): una evaluación nueva "
                       "se registra con datos sintéticos y una referencia trivial (CONTRIBUTING)")
    if e.ic is not None and _resolver(e.ic) is None:
      raise ValueError(f"evaluador {e.nombre}: ic tiene que ser una función ic(valores, seg)")
  _REGISTRO[e.nombre] = e
  return e


def evaluador(nombre, aplica_a, sintetico=None, referencia=None, ic=None, menor_es_mejor=True, descripcion=""):
  """Decorador: registra la función `fn(pred, real) -> (N,)` como evaluación."""
  def deco(fn):
    registrar(Evaluador(nombre, tuple(aplica_a), descripcion or (fn.__doc__ or "").strip().split("\n")[0], fn=fn,
                        sintetico=sintetico, referencia=referencia, ic=ic or IC_DEFECTO, menor_es_mejor=menor_es_mejor))
    return fn
  return deco


def quitar(nombre):
  """Saca una entrada del registro (para las que se registran desde un test)."""
  _REGISTRO.pop(nombre, None)


def obtener(nombre) -> Evaluador:
  if nombre not in _REGISTRO:
    raise ValueError(f"evaluador {nombre!r} desconocido; registrados: {', '.join(_REGISTRO)}")
  return _REGISTRO[nombre]


def nombres() -> tuple:
  return tuple(_REGISTRO)


def listar() -> list:
  return list(_REGISTRO.values())


def para(clave) -> list:
  """Evaluadores que aplican a una evaluación ("video", "latente") o a un kind de modelo."""
  return [e for e in _REGISTRO.values() if "*" in e.aplica_a or clave in e.aplica_a]


def para_modelo(kind) -> list:
  """Los que aplican a un modelo registrado en src/models: por su kind o por la EVALUACION de su módulo."""
  from src.models import KINDS
  if kind not in KINDS:
    raise ValueError(f"modelo {kind!r} desconocido; registrados: {', '.join(KINDS)}")
  evaluacion = getattr(importlib.import_module(KINDS[kind][0]), "EVALUACION", None)
  return [e for e in _REGISTRO.values() if "*" in e.aplica_a or kind in e.aplica_a or evaluacion in e.aplica_a]


def tabla() -> list:
  """Filas (dict) para el reporte y la CLI."""
  return [{"evaluador": e.nombre, "aplica_a": list(e.aplica_a), "descripcion": e.descripcion,
           "origen": "función" if e.propio else f"{e.modulo}.{e.entrada}",
           "referencia": e.referencia if isinstance(e.referencia, str) else (e.referencia.__name__ if e.referencia else None),
           "ic": e.ic if isinstance(e.ic, str) else (e.ic.__name__ if e.ic else None),
           "menor_es_mejor": e.menor_es_mejor} for e in _REGISTRO.values()]


# ---------------------------------------------------------------- verificación

def verificar(nombre, semilla=0) -> dict:
  """Prueba sintética de una evaluación. Devuelve {"modelo", "referencia", "ic_modelo", "ic_referencia"}.

  Las evaluaciones que apuntan a un módulo existente se verifican sin correrlas: el módulo se importa, tiene la
  `entrada` y su IC apunta a una función que existe."""
  e = obtener(nombre)
  if not e.propio:
    mod = importlib.import_module(e.modulo)
    if not callable(getattr(mod, e.entrada, None)):
      raise AssertionError(f"{e.nombre}: {e.modulo} no tiene la función {e.entrada}")
    for ref in (e.ic, e.referencia):
      _resolver(ref)
    return {"modulo": e.modulo, "entrada": e.entrada}
  rng = np.random.default_rng(semilla)
  pred, real, seg = e.sintetico(rng)
  vm = np.asarray(e.fn(pred, real), dtype=np.float64)
  vr = np.asarray(e.fn(e.referencia(real), real), dtype=np.float64)
  n = len(seg)
  for etiqueta, v in (("el modelo", vm), ("la referencia", vr)):
    if v.shape != (n,) or not np.isfinite(v).all():
      raise AssertionError(f"{e.nombre}: {etiqueta} dio {v.shape} (hacen falta {n} valores por clip, finitos)")
  ic = _resolver(e.ic or IC_DEFECTO)
  icm, icr = ic(vm, seg), ic(vr, seg)
  for etiqueta, r, v in (("modelo", icm, vm), ("referencia", icr, vr)):
    if not (r["lo"] <= v.mean() <= r["hi"]):
      raise AssertionError(f"{e.nombre}: el IC de la {etiqueta} no contiene a su media ({r})")
  mejor = vm.mean() < vr.mean() if e.menor_es_mejor else vm.mean() > vr.mean()
  if not mejor:
    raise AssertionError(f"{e.nombre}: el modelo sintético ({vm.mean():.4g}) no le gana a la referencia ({vr.mean():.4g}); "
                         f"menor_es_mejor={e.menor_es_mejor}: revisá la función o los datos sintéticos")
  return {"modelo": float(vm.mean()), "referencia": float(vr.mean()), "ic_modelo": icm, "ic_referencia": icr}


# ---------------------------------------------------------------- las evaluaciones que existen (apuntan a su módulo)

_CE = "src.evaluate"
for _e in (
  Evaluador("ce", ("video",), "Cross-entropy por token con verosimilitud del protocolo de comma (CE de 1 paso en los modelos sin ella)",
            modulo=_CE, referencia="transicion", ic=IC_DEFECTO),
  Evaluador("rollouts", ("video",), "Rollouts a 3 s sobre los 400 clips de evaluación (insumo de LPIPS y FVD)",
            modulo="src.rollouts", referencia="real (clips reales) y congelado", ic=IC_DEFECTO),
  Evaluador("lpips", ("video",), "LPIPS por horizonte contra el video real, con IC por segmento",
            modulo="src.metrics_video", referencia="congelado (repite el último frame)", ic=IC_DEFECTO),
  Evaluador("fvd", ("video",), "FVD contra el video real, con piso real-real e IC por segmento",
            modulo="src.metrics_video", referencia="piso: real contra real (src.estadistica.piso_completo)",
            ic="src.estadistica:ic_fvd_segmento"),
  Evaluador("eval_latent", ("latente",), "Latentes: error por horizonte contra copiar el último, probes, sorpresa y CEM",
            modulo="src.eval_latent", referencia="copiar el último latente y encoder sin entrenar", ic=IC_DEFECTO),
):
  registrar(_e)
