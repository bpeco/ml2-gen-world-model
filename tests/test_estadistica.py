"""Estadística de D14 (src/estadistica.py): bootstrap por segmento, piso de FVD, diferencia pareada y semillas. Sintético, CPU."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src import estadistica as E
from src.metrics_video import boot_ci, fvd_from_feats


def clips_sinteticos(n_seg=100, por_seg=4):
  return [{"clip": i * por_seg + j, "segment": i} for i in range(n_seg) for j in range(por_seg)]


def test_remuestreo_lleva_segmentos_enteros():
  seg = E.segmentos_de(clips_sinteticos(10))
  grupos = E.grupos_de(seg)
  assert len(grupos) == 10 and all(len(g) == 4 for g in grupos)
  idx = E.remuestrear(grupos, np.random.default_rng(0))
  assert len(idx) == 40
  cuentas = np.bincount(idx, minlength=40).reshape(10, 4)
  assert (cuentas == cuentas[:, :1]).all()          # los 4 clips de un segmento aparecen siempre juntos y las mismas veces


def test_cobertura_por_segmento_y_clips_subcubren():
  """Datos con efecto de segmento fuerte y media verdadera 0: el IC por segmento cubre ~95%; el de clips, mucho menos."""
  seg = E.segmentos_de(clips_sinteticos(100))
  rng = np.random.default_rng(0)
  ok_seg = ok_clip = 0
  sims = 150
  for k in range(sims):
    x = np.repeat(rng.normal(size=100), 4) + 0.3 * rng.normal(size=400)
    r = E.ic_media_segmento(x, seg, reps=200, seed=k)
    ok_seg += r["lo"] <= 0 <= r["hi"]
    lo, hi = boot_ci(lambda i: x[i].mean(), 400, 200, seed=k)
    ok_clip += lo <= 0 <= hi
  assert ok_seg / sims >= 0.88
  assert ok_clip / sims < ok_seg / sims - 0.15


def test_ic_normal_con_punto_y_percentil_sin_punto():
  seg = E.segmentos_de(clips_sinteticos(20))
  x = np.random.default_rng(1).normal(size=80) + 5.0   # positiva: el IC normal se recorta en 0 (es de FVD)
  p = E.ic_segmento(lambda i: x[i].mean(), seg, 100, punto=float(x.mean()))
  assert p["lo"] < x.mean() < p["hi"] and "normal" in p["metodo"]
  q = E.ic_segmento(lambda i: x[i].mean(), seg, 100)
  assert "percentil" in q["metodo"] and q["lo"] < q["hi"]


def feats(n, d=6, shift=0.0, seed=0):
  return np.random.default_rng(seed).normal(size=(n, d)) + shift


def test_diferencia_de_modelos_iguales_cubre_cero():
  seg = E.segmentos_de(clips_sinteticos(30))
  real, f = feats(120), feats(120, seed=1, shift=0.4)
  r = E.diferencia_pareada(real, f, f.copy(), seg, reps=40, fvd_fn=fvd_from_feats)
  assert r["diff"] == 0 and r["lo"] == 0 == r["hi"] and r["p"] == 1.0


def test_diferencia_de_modelos_equivalentes_cubre_cero_y_distintos_no():
  seg = E.segmentos_de(clips_sinteticos(50))
  real = feats(200, seed=0)
  a, b = feats(200, seed=1, shift=0.3), feats(200, seed=2, shift=0.3)    # misma calidad, distinto ruido
  r = E.diferencia_pareada(real, a, b, seg, reps=60, fvd_fn=fvd_from_feats)
  assert r["lo"] <= 0 <= r["hi"] and r["p"] > 0.05
  malo = feats(200, seed=3, shift=1.5)
  r = E.diferencia_pareada(real, a, malo, seg, reps=60, fvd_fn=fvd_from_feats)   # A mejor que B
  assert r["diff"] < 0 and r["hi"] < 0 and r["p"] < 0.05


def test_piso_usa_mitades_disjuntas_por_segmento():
  seg = E.segmentos_de(clips_sinteticos(20))
  real = np.stack([np.arange(80), *np.random.default_rng(0).normal(size=(5, 80))], 1)   # columna 0 = id del clip
  vistos = []

  def fvd_falso(a, b):
    sa, sb = set(seg[a[:, 0].astype(int)]), set(seg[b[:, 0].astype(int)])
    assert not (sa & sb) and len(sa) == len(sb) == 10     # ningún segmento en las dos mitades
    vistos.append(1)
    return 1.0

  r = E.piso_fvd(real, seg, reps=5, fvd_fn=fvd_falso)
  assert len(vistos) == 5 and r["value"] == 1.0 and r["n_a"] == r["n_b"] == 40


def test_piso_es_positivo_y_baja_con_mas_clips():
  chico = E.piso_fvd(feats(80, d=8), E.segmentos_de(clips_sinteticos(20)), reps=10, fvd_fn=fvd_from_feats)
  grande = E.piso_fvd(feats(800, d=8), E.segmentos_de(clips_sinteticos(200)), reps=10, fvd_fn=fvd_from_feats)
  assert chico["value"] > grande["value"] > 0
  con_modelo = E.piso_fvd(feats(80, d=8), E.segmentos_de(clips_sinteticos(20)), reps=3, fvd_fn=fvd_from_feats,
                          fake=feats(80, d=8, shift=2.0, seed=5))
  assert con_modelo["modelo_mitades"] > con_modelo["value"]


def test_resumen_semillas():
  r = E.resumen_semillas([1.0, 2.0, 3.0])
  assert r["mean"] == 2.0 and r["std"] == pytest.approx(1.0) and r["n"] == 3
  assert E.resumen_semillas([5.0])["std"] is None
  assert E.resumen_semillas([None, None])["n"] == 0


def escribir_run(d, run, fvd1, ce, h="h1"):
  d.joinpath(f"video_{run}.json").write_text(json.dumps({
    "clips_hash": h, "fvd": {k: {"value": fvd1 * (1 + i)} for i, k in enumerate(E.HORIZONTES)},
    "lpips": {"at": {"1": {"mean": fvd1 / 1000}}}}))
  d.joinpath(f"eval_{run}.json").write_text(json.dumps({"ce_last": ce, "ce_all": None}))


def test_agregar_semillas_y_comando(tmp_path):
  for run, f, c in (("gpt_s", 200.0, 3.0), ("gpt_s_s1", 220.0, 3.2), ("gpt_s_s2", 240.0, 3.4)):
    escribir_run(tmp_path, run, f, c)
  E.main(["semillas", "--nombre", "gpt_s", "--runs", "gpt_s,gpt_s_s1,gpt_s_s2", "--results", str(tmp_path)])
  r = json.loads((tmp_path / "semillas_gpt_s.json").read_text())
  assert r["n_corridas"] == 3 and r["video"]["fvd"]["1"]["mean"] == pytest.approx(220.0)
  assert r["video"]["fvd"]["1"]["std"] == pytest.approx(20.0)
  assert r["video"]["fvd"]["3"]["mean"] == pytest.approx(660.0)
  assert r["eval"]["ce_last"]["std"] == pytest.approx(0.2) and r["eval"]["ce_all"]["n"] == 0
  assert r["video"]["lpips"]["2"]["n"] == 0


def test_semillas_errores(tmp_path):
  escribir_run(tmp_path, "a", 1.0, 1.0)
  with pytest.raises(FileNotFoundError, match="b"):
    E.agregar_semillas("a", ["a", "b"], tmp_path)
  escribir_run(tmp_path, "b", 1.0, 1.0, h="otro")
  with pytest.raises(ValueError, match="distintos"):
    E.agregar_semillas("a", ["a", "b"], tmp_path)


def test_write_stats_trae_piso_e_ic_por_segmento(tmp_path, monkeypatch):
  import src.metrics_video as MV
  monkeypatch.setattr(MV, "RESULTS", tmp_path)
  monkeypatch.setattr(MV, "OUT", tmp_path)
  rng = np.random.default_rng(0)
  n = 24                                   # 6 segmentos de los clips reales de evaluación
  ff = {k: rng.normal(size=(n, 5)) for k in MV.WINDOWS}
  rf = {k: rng.normal(size=(n, 5)) for k in MV.WINDOWS}
  a = SimpleNamespace(name="x", boot=20, boot_fvd=8, boot_piso=3, fvd_model="i3d")
  res = MV.write_stats(a, rng.random((n, 60)), ff, rf, "h", torch.device("cpu"), 0.0)
  assert res["n_segmentos"] == 6 and "segmento" in res["fvd_ci"]
  for k in MV.WINDOWS:
    assert res["piso"][k]["value"] > 0 and "modelo_mitades" in res["piso"][k]
    assert res["fvd"][k]["lo"] <= res["fvd"][k]["value"] <= res["fvd"][k]["hi"]
  assert json.loads((tmp_path / "video_x.json").read_text())["piso"]["1"]["reps"] == 3


def test_clips_eval_b_sin_solape_con_eval(tmp_path, monkeypatch):
  from src import rollouts
  split = tmp_path / "split.json"
  split.write_text(json.dumps({"segments": [f"s{i}" for i in range(5)]}))
  val = tmp_path / "val.npy"
  np.save(val, np.zeros((5, 1200, 1), np.int16))
  monkeypatch.setattr(rollouts, "SPLIT", split)
  monkeypatch.setattr(rollouts, "VAL", str(val))
  b = rollouts.make_clips_b(tmp_path / "b.json")
  a = rollouts.make_clips(tmp_path / "a.json")
  assert len(b["clips"]) == len(a["clips"]) == 20 and b["source"] == "val_eval_b"
  for ca, cb in zip(a["clips"], b["clips"]):
    assert ca["segment"] == cb["segment"] and ca["key"] == cb["key"]
    assert ca["start"] + 80 <= cb["start"] or cb["start"] + 80 <= ca["start"]      # los 80 frames no se pisan
    assert cb["start"] + 80 <= 1200
  assert rollouts.load_clips(tmp_path / "b.json")["starts"] == rollouts.STARTS_B


def test_piso_completo_n_contra_n_y_comando(tmp_path, monkeypatch):
  seg = E.segmentos_de(clips_sinteticos(25))
  a, b = feats(100, d=6, seed=0), feats(100, d=6, seed=1)
  r = E.piso_completo(a, b, seg, reps=20, fvd_fn=fvd_from_feats)
  assert r["n_a"] == r["n_b"] == 100 and r["lo"] <= r["value"] <= r["hi"] and r["value"] > 0
  assert r["value"] == pytest.approx(fvd_from_feats(a, b))
  # comando: caché sintética en tmp
  cache = tmp_path / "cache"
  cache.mkdir()
  np.savez(cache / "fvd_fake_i3d_real_eval_b.npz", **{h: b for h in E.HORIZONTES})
  monkeypatch.setattr(E, "RESULTS", tmp_path)
  monkeypatch.setattr(E, "CACHE", cache)
  monkeypatch.setattr(E, "_cargar_feats", lambda name, n=None, results=None: ({h: a for h in E.HORIZONTES}, None, seg))
  res = E.main(["piso", "--reps", "3", "--boot", "5"])
  d = json.loads((tmp_path / "piso_fvd.json").read_text())
  assert d["b"] == "real_eval_b" and d["piso"]["1"]["n_a"] == 100 and d["mitades"]["1"]["value"] > 0 and res["piso"]
  assert d["piso"]["1"]["value"] < d["mitades"]["1"]["value"]       # a más clips por lado, piso menor


def test_report_usa_piso_completo_y_mitades(tmp_path):
  from src import report
  (tmp_path / "piso_fvd.json").write_text(json.dumps({"piso": {"1": {"value": 111.0, "lo": 100.0, "hi": 120.0}}}))
  v = {"lpips": {"curve": [0.1] * 60, "at": {"1": {"mean": .3, "lo": .2, "hi": .4}}},
       "fvd": {h: {"value": 200.0, "lo": 190.0, "hi": 210.0} for h in "123"},
       "piso": {"1": {"value": 311.0, "modelo_mitades": 323.0}}}
  (tmp_path / "video_m.json").write_text(json.dumps(v))
  (tmp_path / "c.json").write_text(json.dumps([{"label": "m", "family": "gpt", "video": "video_m.json", "autoregressive": True}]))
  cfgs = report.load_configs(tmp_path / "c.json", registro={})
  row = report.summary(cfgs)[0]
  assert row["fvd_piso_1s"] == 111.0 and row["fvd_piso_mitades_1s"] == 311.0 and row["fvd_1s_mitades"] == 323.0
  md = report.markdown([row])
  assert "111.0" in md and "311.0 (323.0)" in md
