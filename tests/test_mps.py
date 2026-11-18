"""Soporte de Apple Silicon (MPS) simulado en CPU con monkeypatch: no hay Mac. Selección de device, backend, AMP, sin graphs
ni pinned memory, tiempos con sincronización y el diagnóstico de 20 ms fuera de Windows."""
import platform

import numpy as np
import pytest
import torch

from src import bench, common
from src.infer import backends, bench_e2e
from src.windows import Windows


@pytest.fixture
def mps(monkeypatch):
  """Una máquina con MPS y sin CUDA."""
  monkeypatch.delenv("ML2_DEVICE", raising=False)
  monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
  monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)


def test_device_cuda_antes_que_mps_antes_que_cpu(mps, monkeypatch):
  assert common.device().type == "mps"
  monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
  assert common.device().type == "cuda"
  monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
  monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
  assert common.device().type == "cpu"


def test_device_explicito_gana(mps, monkeypatch):
  monkeypatch.setenv("ML2_DEVICE", "cpu")
  assert common.device().type == "cpu"


def test_backend_mps_se_detecta_y_es_el_default(mps, monkeypatch):
  assert backends.estado("mps").disponible and "mps" in backends.disponibles("torch")
  assert backends.por_defecto() == "mps" and backends.torch_device("mps") == "mps"
  assert backends.dispositivo("mps").type == "mps"
  monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
  e = backends.estado("mps")
  assert not e.disponible and "is_available" in e.motivo and backends.por_defecto() == "cpu"
  with pytest.raises(SystemExit, match="mps"):
    backends.exigir("mps")


def test_aviso_de_fallback_sin_setearlo(mps, monkeypatch, capsys):
  monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
  common.aviso_mps(torch.device("mps"))
  assert "PYTORCH_ENABLE_MPS_FALLBACK=1" in capsys.readouterr().out
  import os
  assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ
  monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
  common.aviso_mps(torch.device("mps"))
  assert capsys.readouterr().out == ""


def test_amp_en_mps(monkeypatch):
  mps_dev = torch.device("mps")
  assert common.amp_dtype(mps_dev, "fp16") == (torch.float16, "")
  assert common.amp_dtype(mps_dev, None) == (None, "") and common.amp_dtype("cpu", "fp16") == (None, "")
  assert common.amp_dtype("cuda", "bf16") == (torch.bfloat16, "")
  assert common.amp_dtype(mps_dev, "bf16") == (None, "MPS de este sistema no soporta bf16: se entrena en fp32") or \
    common.amp_dtype(mps_dev, "bf16")[0] is torch.bfloat16  # según el torch de la máquina
  monkeypatch.setattr(torch, "zeros", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bf16 no soportado")))
  dt, nota = common.amp_dtype(mps_dev, "bf16")
  assert dt is None and "bf16" in nota


def test_windows_no_pinea_fuera_de_cuda(monkeypatch):
  w = Windows(np.random.default_rng(0).integers(0, 1024, (6, 30, 128)).astype(np.int16), holdout=1)
  usados = []
  monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
  monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: usados.append(1) or self)
  w.sample(0, 2, 4)
  assert usados == [1]                        # sin device y con CUDA: como siempre
  usados.clear()
  w.device = torch.device("mps")              # device de MPS: no residente (acá) y sin pinned
  w.sample(1, 2, 4)
  assert usados == []
  assert not w.resident


def test_medicion_en_mps_sincroniza_y_no_usa_eventos(monkeypatch):
  llamadas = []
  monkeypatch.setattr(bench, "sync", lambda dev: llamadas.append(dev.type))
  monkeypatch.setattr(torch.cuda, "Event", lambda **k: pytest.fail("MPS no tiene eventos de CUDA"))
  ms = bench.timed(iter([1, 2, 3]), 3, torch.device("mps"))
  assert len(ms) == 3 and llamadas == ["mps"] * 6


def test_sin_cuda_graphs_en_mps():
  from src.infer.codec import Codec
  # el criterio de los graphs es dev.type == "cuda": bench_e2e (graphs) y Codec (self.graphs) lo usan tal cual
  import inspect
  assert 'dev.type == "cuda"' in inspect.getsource(bench_e2e.main)
  assert 'self.dev.type == "cuda"' in inspect.getsource(Codec.__init__)


def test_diagnostico_fuera_de_windows_no_rompe(monkeypatch):
  monkeypatch.setattr(platform, "system", lambda: "Darwin")
  with bench_e2e.timer_1ms() as ok:
    assert ok is False
  info = bench_e2e.timer_info(n=3)
  assert info["sistema"] == "Darwin" and info["sleep_1ms_con_timeBeginPeriod_1"]["aplica"] is False


def test_bench_acepta_mps_y_explica_dml_npu(monkeypatch):
  ap = __import__("argparse").ArgumentParser()
  backends.agregar_device(ap, "torch+ort")
  assert ap.parse_args(["--device", "mps"]).device == "mps" and ap.parse_args(["--device", "npu"]).device == "npu"
  for d in ("dml", "npu"):
    with pytest.raises(SystemExit, match="bench_ort"):
      backends.dispositivo(d)
