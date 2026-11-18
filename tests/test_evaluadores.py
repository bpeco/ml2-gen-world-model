"""Registro de evaluadores (src/evaluadores.py): las que existen apuntan a su módulo; una evaluación nueva se registra con una función,
datos sintéticos, una referencia trivial y un IC, y aparece en el registro sin tocar el núcleo."""
import numpy as np
import pytest

from src import evaluadores as E


def sintetico(rng):
  seg = np.repeat(np.arange(10), 4)
  real = rng.normal(size=(40, 8))
  return real + 0.1 * rng.normal(size=real.shape), real, seg


def media(real):
  return np.broadcast_to(real.mean(), real.shape)    # referencia trivial: la media global


@pytest.fixture
def mse():
  @E.evaluador("mse_juguete", aplica_a=("video",), sintetico=sintetico, referencia=media)
  def mse_juguete(pred, real):
    """Error cuadrático medio por clip."""
    return ((pred - real) ** 2).mean(-1)
  yield "mse_juguete"
  E.quitar("mse_juguete")


def test_las_que_existen_estan_registradas_y_apuntan_a_su_modulo():
  for n in ("ce", "rollouts", "lpips", "fvd", "eval_latent"):
    e = E.obtener(n)
    assert not e.propio and callable(e.punto_de_entrada())
  assert {e.nombre for e in E.para("latente")} == {"eval_latent"}
  assert {"ce", "rollouts", "lpips", "fvd"} <= {e.nombre for e in E.para("video")}
  assert E.obtener("eval_latent").modulo == "src.eval_latent"
  for n in ("ce", "fvd", "eval_latent"):
    E.verificar(n)


def test_para_modelo_usa_la_evaluacion_del_modelo():
  lewm = {e.nombre for e in E.para_modelo("lewm")}
  genie = {e.nombre for e in E.para_modelo("genie")}
  assert "eval_latent" in lewm and "fvd" not in lewm
  assert "fvd" in genie and "eval_latent" not in genie


def test_un_evaluador_de_juguete_aparece_sin_tocar_el_nucleo(mse):
  assert mse in E.nombres() and mse in {e.nombre for e in E.para("video")} and mse in {e.nombre for e in E.para_modelo("gpt")}
  assert mse not in {e.nombre for e in E.para("latente")}
  assert mse in {f["evaluador"] for f in E.tabla()}
  r = E.verificar(mse)
  assert r["modelo"] < r["referencia"] and r["ic_modelo"]["lo"] <= r["modelo"] <= r["ic_modelo"]["hi"]


def test_quitarlo_lo_saca_del_registro():
  assert "mse_juguete" not in E.nombres()


def test_exige_funcion_sintetico_y_referencia():
  with pytest.raises(ValueError, match="sintetico y referencia"):
    E.evaluador("sin_nada", aplica_a=("video",))(lambda p, r: p)
  with pytest.raises(ValueError, match="referencia"):
    E.evaluador("sin_ref", aplica_a=("video",), sintetico=sintetico)(lambda p, r: p)
  with pytest.raises(ValueError, match="aplica_a"):
    E.registrar(E.Evaluador("x", ()))
  with pytest.raises(ValueError, match="falta la función"):
    E.registrar(E.Evaluador("x", ("video",)))
  assert "sin_nada" not in E.nombres() and "sin_ref" not in E.nombres()


def test_verificar_detecta_una_evaluacion_rota():
  @E.evaluador("mala", aplica_a=("video",), sintetico=sintetico, referencia=media, menor_es_mejor=False)
  def mala(pred, real):
    return ((pred - real) ** 2).mean(-1)         # menor es mejor, pero se declaró al revés
  try:
    with pytest.raises(AssertionError, match="no le gana a la referencia"):
      E.verificar("mala")
  finally:
    E.quitar("mala")


def test_desconocido():
  with pytest.raises(ValueError, match="desconocido"):
    E.obtener("nope")
