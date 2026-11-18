"""Registro de backends de inferencia (src/infer/backends.py): detección, cómo corre cada uno y un backend de juguete que aparece
en el registro y en los --device sin tocar el núcleo."""
import argparse

import pytest
import torch

from src.infer import backends


@pytest.fixture
def juguete():
  @backends.backend("juguete", "backend de prueba", torch_device="cpu", ort_provider="cpu", ort_ep="CPUExecutionProvider")
  def _detectar():
    return True, "siempre"
  yield "juguete"
  backends.quitar("juguete")


def test_los_del_repo_estan_registrados():
  assert {"cuda", "cpu", "rocm", "mps", "dml", "npu"} <= set(backends.nombres())
  assert backends.torch_device("rocm") == "cuda" and backends.torch_device("cpu") == "cpu"
  assert "cpu" in backends.disponibles() and backends.estado("cpu").disponible
  assert set(backends.nombres("torch")) >= {"cuda", "cpu", "rocm"} and "dml" not in backends.nombres("torch")
  assert {"dml", "npu", "cpu"} <= set(backends.nombres("ort")) and "vitisai" in backends.ort_providers()


def test_detectar_da_motivo_cuando_no_hay():
  det = backends.detectar()
  assert set(det) == set(backends.nombres())
  for n, e in det.items():
    assert e.disponible or e.motivo, n


def test_torch_no_corre_los_de_onnx():
  with pytest.raises(ValueError, match="ONNX Runtime"):
    backends.torch_device("npu")
  with pytest.raises(ValueError, match="solo torch"):
    backends.resolver_ort("rocm")
  assert backends.resolver_ort("vitisai").nombre == "npu"


def test_registro_valida():
  with pytest.raises(ValueError, match="minúsculas"):
    backends.registrar(backends.Backend("Mal Nombre", "", lambda: (True, ""), torch_device="cpu"))
  with pytest.raises(ValueError, match="no corre nada"):
    backends.registrar(backends.Backend("nada", "", lambda: (True, "")))


def test_un_backend_de_juguete_aparece_sin_tocar_el_nucleo(juguete):
  assert juguete in backends.nombres() and juguete in backends.disponibles("torch")
  assert any(f["backend"] == juguete and f["disponible"] for f in backends.tabla())
  ap = argparse.ArgumentParser()
  backends.agregar_device(ap, "torch")           # lo que hacen src.bench, bench_e2e y bench_codec
  assert ap.parse_args(["--device", juguete]).device == juguete
  assert backends.dispositivo(juguete) == torch.device("cpu")
  assert backends.resolver_ort(juguete).ort_ep == "CPUExecutionProvider" and juguete in backends.nombres("ort")


def test_el_de_juguete_se_va_al_quitarlo():
  assert "juguete" not in backends.nombres()
  ap = argparse.ArgumentParser()
  backends.agregar_device(ap, "torch")
  with pytest.raises(SystemExit):
    ap.parse_args(["--device", "juguete"])


def test_no_disponible_falla_con_instrucciones():
  backends.registrar(backends.Backend("faltante", "x", lambda: (False, "no hay"), torch_device="cpu", instalar="pip install x"))
  try:
    assert "faltante" not in backends.disponibles()
    with pytest.raises(SystemExit, match="no hay.*pip install x"):
      backends.exigir("faltante")
  finally:
    backends.quitar("faltante")


def test_deteccion_rota_no_rompe_el_listado():
  def mala():
    raise RuntimeError("boom")
  backends.registrar(backends.Backend("rota", "x", mala, torch_device="cpu"))
  try:
    e = backends.estado("rota")
    assert not e.disponible and "boom" in e.motivo
  finally:
    backends.quitar("rota")
