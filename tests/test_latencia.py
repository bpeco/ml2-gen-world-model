"""Protocolo de latencia único (D15): modelo solo, cadena cerrada y cadena a ritmo de cámara, y el diagnóstico del piso de
20 ms. Todo en CPU, con un modelo latente chico (LeWM sin entrenar) y pocos frames."""
import json
from pathlib import Path

import pytest

from src import bench
from src.infer import bench_e2e

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "data/sample_video_128x256.npy").exists(), reason="falta el video de ejemplo")


@pytest.fixture
def salida(tmp_path, monkeypatch):
  monkeypatch.setattr(bench_e2e, "OUT", tmp_path)
  monkeypatch.setattr(bench, "OUT", tmp_path)
  return tmp_path


COMUN = ["--model", "random:lewm", "--device", "cpu", "--frames", "3", "--warmup", "1", "--tag", "pytest"]


def test_protocolo_tiene_las_tres_columnas(salida):
  r = bench_e2e.main(COMUN + ["--protocolo", "--pace", "50"])
  d = json.loads((salida / "latencia_random_lewm__pytest.json").read_text())
  assert r["unidad"].startswith("paso de modelo") and d["budget_ms"] == 250
  for col in ("modelo_solo", "cadena_cerrada", "cadena_ritmo"):
    assert d[col]["p50"] > 0 and d[col]["p99"] >= d[col]["p50"] and d[col]["n"] == 3
  assert d["cadena_cerrada"]["pace_ms"] == 0 and d["cadena_ritmo"]["pace_ms"] == 50
  assert d["modelo_solo"]["graphs_usados"] is False and "graphs_error" in d["modelo_solo"]
  assert d["dtype"] == "fp32" and d["dtype_pedido"]["dtype"] == "fp16"          # en CPU se declara el cambio de dtype


def test_diagnostico_una_fila_por_variante(salida):
  d = bench_e2e.main(COMUN + ["--diagnostico", "--pace", "50"])
  nombres = [f["nombre"] for f in d["filas"]]
  assert nombres == ["cerrado"] + [v[0] for v in bench_e2e.VARIANTES]
  assert all(f["p50"] > 0 and f["p99"] > 0 for f in d["filas"])
  assert d["timer"]["sleep_1ms"]["p50_ms"] > 0 and "resolution_s" in d["timer"]["perf_counter"]
  nota = {f["nombre"]: f.get("nota") for f in d["filas"]}
  assert nota["sleep_keepalive"] and nota["espera_activa_sync_poll"] and nota["sleep"] is None   # CPU: sin efecto, declarado
  assert (salida / "diagnostico_random_lewm__online__pace50__pytest.json").exists()


def test_e2e_modo_unico_sigue_andando(salida):
  d = bench_e2e.main(COMUN + ["--mode", "imagine"])
  assert d["steady"]["p99"] > 0 and d["graphs"] is False and d["pace_ms"] == 0
  assert (salida / "e2e_random_lewm__imagine__closed__pytest.json").exists()


def test_bench_por_stream_con_p50_y_p99(salida):
  r = bench.main(["--model", "random:maskgit_s", "--steps", "1", "--device", "cpu", "--warmup", "1", "--frames", "2",
                  "--name", "pytest_stream"])
  assert r["p50_ms"] == r["median_ms"] > 0 and r["p99_ms"] >= r["p50_ms"] and r["graphs_usados"] is False
  r = bench.main(["--model", "random:lewm", "--device", "cpu", "--warmup", "1", "--frames", "2", "--name", "pytest_lewm"])
  assert r["unidad"].startswith("paso de modelo") and r["realtime"]


def test_bench_vjepa2_mide_el_encoder_sobre_un_clip(salida):
  """Trabajo 14: `vjepa2 bench` caía en el stream de tokens (V-JEPA 2 sin acciones no tiene stream)."""
  r = bench.main(["--model", "random:vjepa2", "--device", "cpu", "--warmup", "1", "--frames", "2", "--name", "pytest_vjepa2"])
  assert r["p50_ms"] > 0 and r["unidad"].startswith("clip del encoder") and r["graphs_usados"] is False


def test_timer_1ms_no_rompe_fuera_de_windows():
  with bench_e2e.timer_1ms(False) as ok:
    assert ok is False
