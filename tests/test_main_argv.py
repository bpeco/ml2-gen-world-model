import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULOS = ["src.evaluate", "src.rollouts", "src.metrics_video", "src.bench", "src.eval_latent", "src.report",
           "src.ejemplos", "src.export_onnx", "src.infer.bench_codec", "src.infer.bench_e2e", "src.infer.bench_ort",
           "src.infer.quality", "src.infer.demo", "src.infer.export_codec", "src.infer.report_e2e",
           "src.configs", "scripts.analysis", "src.resultados"]


def _mod(name):
  if name.startswith("scripts."):
    sys.path.insert(0, str(ROOT))
    if not (ROOT / "scripts/__init__.py").exists():  # carpeta sin paquete: se importa como módulo suelto
      sys.path.insert(0, str(ROOT / "scripts"))
      return importlib.import_module(name.split(".")[1])
  return importlib.import_module(name)


@pytest.mark.parametrize("name", MODULOS)
def test_main_help(name, capsys):
  m = _mod(name)
  with pytest.raises(SystemExit) as e:
    m.main(["--help"])
  assert e.value.code == 0
  assert "usage" in capsys.readouterr().out.lower()


def test_main_devuelve_resultado(tmp_path):
  """report.main y bench.main devuelven su resultado (CPU, liviano)."""
  from src import bench, report
  r = bench.main(["--model", "random:gpt_s", "--device", "cpu", "--warmup", "1", "--frames", "2", "--name",
                  "pytest_argv"])
  (ROOT / "results/bench/pytest_argv__fp32_cpu.json").unlink(missing_ok=True)
  assert r["median_ms"] > 0 and r["frames"] == 2
  m = tmp_path / "configs.json"
  m.write_text(json.dumps([]))
  assert report.main(["--manifest", str(m), "--out", str(tmp_path / "o"), "--estados", str(tmp_path / "e.json")]) == []
