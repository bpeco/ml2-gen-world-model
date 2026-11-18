"""scripts/make_notebooks.py y scripts/run_notebooks.py: los notebooks del repo están al día, sin salidas y con todas las acciones."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def cargar(nombre):
  spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / f"{nombre}.py")
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


@pytest.fixture(scope="module")
def generados():
  return cargar("make_notebooks").generar()


def src(c):
  """El texto de una celda (nbformat guarda el código como lista de líneas)."""
  return c["source"] if isinstance(c["source"], str) else "".join(c["source"])


def test_generacion_es_determinista(generados):
  assert cargar("make_notebooks").generar() == generados


def test_check_esta_al_dia():
  r = subprocess.run([sys.executable, "scripts/make_notebooks.py", "--check"], cwd=ROOT, capture_output=True, text=True,
                     encoding="utf-8", errors="replace")
  assert r.returncode == 0, r.stdout + r.stderr


def test_un_notebook_por_modelo_registrado_y_los_generales(generados):
  from src import toolkit as T
  assert {f"modelos/{m}.ipynb" for m in T.modelos()} <= set(generados)
  for g in ("resultados", "latentes", "inferencia", "colab", "demo_clase"):
    assert f"{g}.ipynb" in generados
  assert not (ROOT / "notebooks" / "entrenamiento.ipynb").exists()


def test_sin_salidas_ni_numeros_viejos(generados):
  for rel, texto in generados.items():
    nb = json.loads(texto)
    for c in nb["cells"]:
      if c["cell_type"] == "code":
        assert c["outputs"] == [] and c["execution_count"] is None, rel
    assert ".venv" not in texto and "Scripts/python" not in texto, f"{rel}: ruta fija del venv"


def test_cada_modelo_tiene_sus_acciones_en_el_orden_de_la_cli(generados):
  from src import toolkit as T
  for m in T.modelos():
    nb = json.loads(generados[f"modelos/{m}.ipynb"])
    titulos = [src(c).split("\n")[0] for c in nb["cells"] if c["cell_type"] == "markdown" and src(c).startswith("## ")]
    assert [t.split(".")[0] for t in titulos[:6]] == ["## 1", "## 2", "## 3", "## 4", "## 5", "## 6"], (m, titulos)
    assert [t.split(". ", 1)[1] for t in titulos[:6]] == ["Receta y fidelidad", "Entrenar o continuar", "Evaluar", "Latencia", "Demo", "Exportar"]
    fuente = "\n".join(src(c) for c in nb["cells"] if c["cell_type"] == "code")
    for a in T.acciones_de(m):
      if a != "estado":
        assert f'"{a}"' in fuente, f"{m}: el notebook no tiene la acción {a}"
    assert "DRY_RUN = True" in fuente                      # las acciones van con dry_run por defecto
    assert titulos[-1].startswith("## 7"), m                  # y al final la sección propia


def test_demo_de_clase_tiene_lo_pedido(generados):
  nb = json.loads(generados["demo_clase.ipynb"])
  fuente = "\n".join(src(c) for c in nb["cells"])
  for pieza in ("MODELOS", "DEVICE", "FUENTE", "nbui.demo(", "nbui.comparar(", "nbui.imaginar(", "catalogo_corridas", "entrenar"):
    assert pieza in fuente, pieza


def test_lista_de_run_notebooks():
  rn = cargar("run_notebooks")
  assert all(p.suffix == ".ipynb" for p in rn.lista())
  assert [p.name for p in rn.lista(["modelos/genie"])] == ["genie.ipynb"]


def test_ejecuta_un_notebook_de_modelo_en_cpu():
  r = subprocess.run([sys.executable, "scripts/run_notebooks.py", "--limite", "2", "--solo", "modelos/lewm.ipynb"], cwd=ROOT,
                     capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
  assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-1000:]
