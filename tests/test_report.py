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
  return p


def test_report_runs(manifest, tmp_path):
  out = tmp_path / "out"
  subprocess.run([sys.executable, "-m", "src.report", "--manifest", str(manifest), "--out", str(out)],
                 cwd=ROOT, check=True)
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


def test_pareto_hand_made():
  x = [1, 2, 3, 4, 5]
  y = [10, 12, 5, 6, 5]
  # (2,12) dominado por (1,10); (4,6) y (5,5) dominados por (3,5)
  assert report.pareto_mask(x, y) == [True, False, True, False, False]
  assert report.pareto_mask([1, None, 2], [3, 1, None]) == [True, False, False]
  assert report.pareto_mask([1, 1], [2, 2]) == [True, True]  # empate exacto: ninguno domina


def test_spearman_monotone():
  cfgs = [dict(autoregressive=True, ce_all=c, fvd1=f) for c, f in [(1, 10), (2, 20), (3, 35), (4, 90)]]
  cfgs.append(dict(autoregressive=False, ce_all=9, fvd1=0))     # se ignora
  cfgs.append(dict(autoregressive=True, ce_all=None, fvd1=5))   # se ignora
  rho, p, n, _ = report.correlation(cfgs)
  assert rho == pytest.approx(1.0) and n == 4


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
