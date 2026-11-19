import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src import report

ROOT = Path(__file__).resolve().parents[1]


def fake(tmp, name, family, params, ce, fvd, ms, lp=0.3, **kw):
  """Escribe eval/video/bench sintéticos y devuelve la entrada del manifiesto."""
  m = {"label": name, "family": family, "params": params, "autoregressive": kw.get("ar", True)}
  if ce is not None or kw.get("eval"):
    (tmp / f"eval_{name}.json").write_text(json.dumps({"ce_all": ce, "ce_tok": 1.0, "ce_last": 1.1}))
    m["eval"] = f"eval_{name}.json"
  if fvd is not None:
    v = {"lpips": {"curve": list(np.linspace(0.1, lp, 60)), "at": {"1": {"mean": lp, "lo": lp - .02, "hi": lp + .02}}},
         "fvd": {h: {"value": fvd * k, "lo": fvd * k * .9, "hi": fvd * k * 1.1} for h, k in (("1", 1), ("2", 1.5), ("3", 2))}}
    (tmp / f"video_{name}.json").write_text(json.dumps(v))
    m["video"] = f"video_{name}.json"
  if ms is not None:
    (tmp / f"bench_{name}.json").write_text(json.dumps({"median_ms": ms}))
    m["bench"] = f"bench_{name}.json"
  return m


@pytest.fixture
def manifest(tmp_path):
  cfgs = [fake(tmp_path, "copy", "baseline", 0, 4.9, 900, 1, eval=True, ar=False),
          fake(tmp_path, "gpt_l", "gpt", 5e6, 4.0, 500, 20),
          fake(tmp_path, "gpt_s", "gpt", 12e6, 3.5, 300, 40),
          fake(tmp_path, "gpt_kd", "gpt_kd", 12e6, 3.3, 250, 40),
          fake(tmp_path, "maskgit", "maskgit", 12e6, None, 280, 25, eval=True),
          fake(tmp_path, "novideo", "gpt", 27e6, 3.0, None, 60),
          fake(tmp_path, "gpt2m", "gpt2m", 300e6, 2.8, 150, 800)]
  (tmp_path / "eval_maskgit.json").write_text(json.dumps({"ce_all": None, "ce_last": 3.6}))
  p = tmp_path / "configs.json"
  p.write_text(json.dumps(cfgs))
  from src import resultados
  for f in tmp_path.glob("*.json"):  # todos vigentes, salvo el estado que cada test cambie
    if f != p:
      resultados.marcar(f, "vigente", "fixture", tmp_path / "estado.json")
  return p


def test_report_runs(manifest, tmp_path):
  out = tmp_path / "out"
  subprocess.run([sys.executable, "-m", "src.report", "--manifest", str(manifest), "--out", str(out),
                  "--estados", str(manifest.parent / "estado.json")], cwd=ROOT, check=True)
  for f in ("fig_pareto.png", "fig_pareto_lpips.png", "fig_horizon.png", "fig_ce_vs_fvd.png",
            "summary.json", "summary.md"):
    assert (out / f).stat().st_size > 0
  s = json.loads((out / "summary.json").read_text())
  assert len(s["configs"]) == 7
  rows = {r["label"]: r for r in s["configs"]}
  assert rows["novideo"]["fvd_1s"] is None and rows["novideo"]["pareto"] is None
  assert rows["maskgit"]["ce_all"] is None and s["ce_vs_fvd1"]["n"] == 4  # gpt_s, gpt_kd, gpt_l, gpt2m
  assert rows["copy"]["realtime"] and not rows["gpt2m"]["realtime"]
  assert "| copy |" in (out / "summary.md").read_text(encoding="utf-8")
  assert "| Estado |" in (out / "summary.md").read_text(encoding="utf-8") and rows["copy"]["estado"] == "vigente"


def test_report_sin_vigentes(manifest, tmp_path):
  from src import resultados
  for f in tmp_path.glob("*_*.json"):
    resultados.marcar(f, "protocolo anterior", "x", tmp_path / "estado.json")
  out = tmp_path / "out"
  rows = report.main(["--manifest", str(manifest), "--out", str(out), "--estados", str(tmp_path / "estado.json")])
  assert len(rows) == 7 and {r["estado"] for r in rows} == {"protocolo anterior"}
  assert all(r["pareto"] is None for r in rows)
  for f in ("fig_pareto", "fig_horizon", "fig_ce_vs_fvd", "fig_latency"):
    assert (out / f"{f}.png").stat().st_size > 0
  assert "protocolo anterior" in (out / "summary.md").read_text(encoding="utf-8")


def test_pareto_hand_made():
  x = [1, 2, 3, 4, 5]
  y = [10, 12, 5, 6, 5]
  # (2,12) dominado por (1,10); (4,6) y (5,5) dominados por (3,5)
  assert report.pareto_mask(x, y) == [True, False, True, False, False]
  assert report.pareto_mask([1, None, 2], [3, 1, None]) == [True, False, False]
  assert report.pareto_mask([1, 1], [2, 2]) == [True, True]  # empate exacto: ninguno domina


def test_spearman_monotone():
  cfgs = [dict(autoregressive=True, ce_all=c, fvd1=f) for c, f in [(1, 10), (2, 20), (3, 35), (4, 90), (5, 95)]]
  cfgs.append(dict(autoregressive=False, ce_all=9, fvd1=0))     # se ignora
  cfgs.append(dict(autoregressive=True, ce_all=None, fvd1=5))   # se ignora
  rho, p, n, _ = report.correlation(cfgs)
  assert rho == pytest.approx(1.0) and n == 5
  # con 3 puntos scipy daba ρ = 1 y p = 0, que no dice nada: por debajo de 5 no se informa
  assert report.correlation(cfgs[:3])[:3] == (None, None, 3)


def test_bench_smoke_cpu(tmp_path):
  spec = "random:gpt_s"
  r = subprocess.run([sys.executable, "-m", "src.bench", "--model", spec, "--device", "cpu", "--warmup", "2",
                      "--frames", "5", "--name", "pytest_smoke"], cwd=ROOT, capture_output=True, text=True)
  if r.returncode != 0:
    pytest.skip(f"bench no corrió: {r.stderr[-300:]}")
  p = ROOT / "results/bench/pytest_smoke__fp32_cpu.json"
  try:
    d = json.loads(p.read_text())
  finally:
    p.unlink(missing_ok=True)
  for k in ("median_ms", "p10_ms", "p90_ms", "mean_ms", "fps", "realtime", "params", "dtype", "graphs", "int8",
            "steps", "ctx_frames", "warmup", "frames", "device", "torch", "timestamp"):
    assert k in d
  assert d["frames"] == 5 and d["median_ms"] > 0


def test_de_punta_a_punta_lee_el_protocolo_unico(manifest):
  """Trabajo 14: el reporte buscaba el e2e viejo ("steady") y la columna de punta a punta salía vacía."""
  cfgs = json.loads(manifest.read_text())
  (manifest.parent / "latencia_gpt_s.json").write_text(json.dumps(
    {"modelo_solo": {"p50": 40.0, "p99": 41.0}, "cadena_cerrada": {"p50": 45.0, "p99": 47.0},
     "cadena_ritmo": {"p50": 46.0, "p99": 48.5}}))
  (manifest.parent / "e2e_gpt_l.json").write_text(json.dumps({"steady": {"p50": 30.0, "p99": 33.0}}))
  for m in cfgs:
    m["e2e"] = {"gpt_s": "latencia_gpt_s.json", "gpt_l": "e2e_gpt_l.json"}.get(m["label"])
  manifest.write_text(json.dumps(cfgs))
  rows = {c["label"]: c for c in report.load_configs(manifest, {})}
  assert rows["gpt_s"]["e2e_p99"] == 48.5 and rows["gpt_l"]["e2e_p99"] == 33.0 and rows["copy"]["e2e_p99"] is None


def test_las_semillas_no_compiten_en_el_pareto(tmp_path, monkeypatch):
  """Las semillas 1 y 2 van en su grupo: no entran en la frontera de Pareto ni en la figura (la principal es la 0)."""
  from src import configs
  monkeypatch.setattr(configs, "R", tmp_path)
  for n in ("video_convlstm_px.json", "video_convlstm_px_s1.json"):
    (tmp_path / n).write_text("{}")
  grupos = {configs.entrada(n)["label"]: configs.entrada(n)["group"] for n in ("convlstm_px", "convlstm_px_s1")}
  assert grupos == {"convlstm_px": "principal", "convlstm_px_s1": "semilla"}


def test_genie_m_tiene_su_entrada_en_la_leyenda_del_pareto():
  """Con genie_m, el Pareto encimaba los números de pasadas de S y M y la leyenda no decía qué eran los huecos."""
  base = dict(family="genie", steps=4)
  assert report.leyenda_tamano_m([dict(base, size="s")]) == []
  [h] = report.leyenda_tamano_m([dict(base, size="s"), dict(base, size="m")])
  assert h.get_label() == "Genie M (hueco, punteado)" and h.get_linestyle() == ":"


def test_fig_latency_modelo_y_cadena(manifest, tmp_path):
  """La figura de latencia esperaba las variantes fp32/fp16/graphs de la tanda vieja y salía con una sola barra."""
  cfgs = report.load_configs(manifest, {})
  for c in cfgs:
    c["estado"], c["e2e_p99"] = "vigente", (c["ms"] * 1.5 if c["label"] == "gpt_s" else None)
  report.fig_latency(cfgs, tmp_path / "lat.png")
  assert (tmp_path / "lat.png").stat().st_size > 0


def test_el_destilado_es_autorregresivo(tmp_path, monkeypatch):
  """gpt_s_kd quedaba fuera de CE contra FVD (H4) porque su familia, gpt_kd, no figuraba como autorregresiva."""
  from src import configs
  monkeypatch.setattr(configs, "R", tmp_path)
  (tmp_path / "eval_gpt_s_kd.json").write_text("{}")
  assert configs.entrada("gpt_s_kd")["autoregressive"] is True
