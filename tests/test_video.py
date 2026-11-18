"""Pruebas rápidas y sintéticas de rollouts y métricas de video (sin descargas)."""
import json

import numpy as np
import scipy.linalg

from src import rollouts
from src.metrics_video import AT, WINDOWS, boot_ci, frechet, fvd_from_feats


def feats(n=300, d=16, shift=0.0, seed=0):
  return np.random.default_rng(seed).normal(size=(n, d)) + shift


def test_frechet_identico_es_cero():
  x = feats()
  assert fvd_from_feats(x, x) < 1e-6


def test_frechet_crece_con_el_desplazamiento():
  x = feats()
  v = [fvd_from_feats(x, feats(seed=1, shift=s)) for s in (0.0, 0.5, 1.0, 2.0)]
  assert v == sorted(v) and v[-1] > v[0] + 10
  # con medias distintas y covarianza igual el valor es ~ d * shift^2
  assert abs(fvd_from_feats(x, x + 2.0) - 16 * 4) < 1e-6


def frechet_sqrtm(mu1, s1, mu2, s2):
  """Versión de cd-fvd con scipy.linalg.sqrtm (referencia para contrastar)."""
  s = scipy.linalg.sqrtm(s1 @ s2)
  return float(np.real(np.square(mu1 - mu2).sum() + np.trace(s1 + s2 - 2 * s)))


def test_frechet_coincide_con_sqrtm():
  a, b = feats(), feats(seed=1, shift=0.3) * 1.2
  args = (a.mean(0), np.cov(a, rowvar=False), b.mean(0), np.cov(b, rowvar=False))
  assert abs(frechet(*args) - frechet_sqrtm(*args)) < 1e-6


def test_bootstrap_contiene_el_valor():
  r, f = feats(), feats(seed=1, shift=0.5)
  v = fvd_from_feats(r, f)
  lo, hi = boot_ci(lambda i: fvd_from_feats(r[i], f[i]), len(r), 50)
  assert lo <= v <= hi
  col = np.random.default_rng(0).normal(size=400)
  lo, hi = boot_ci(lambda i: col[i].mean(), 400, 500)
  assert lo <= col.mean() <= hi


def test_indices_de_horizonte():
  # 0.25, 0.5, 1, 2, 3 s a 20 FPS -> frame 4, 9, 19, 39, 59
  for k, f in AT.items():
    assert f == round(float(k) * 20) - 1
  # ventanas de 16 frames que terminan a 1, 2 y 3 s
  for k, (a, b) in WINDOWS.items():
    assert b - a == 16 and b == int(k) * 20


def test_nombres():
  assert rollouts.default_name("gpt_s") == "gpt_s"
  assert rollouts.default_name("maskgit_s", steps=4) == "maskgit_s_s4"
  assert rollouts.default_name("gpt_s", dtype="fp16") == "gpt_s_fp16"
  assert rollouts.default_name("gpt_s", ctx_frames=8) == "gpt_s_ctx8"


def test_clips_estructura(tmp_path, monkeypatch):
  split = tmp_path / "split.json"
  split.write_text(json.dumps({"segments": [f"seg{i}" for i in range(100)]}))
  monkeypatch.setattr(rollouts, "SPLIT", split)
  d = rollouts.load_clips(tmp_path / "clips.json")
  assert (d["ctx"], d["horizon"], d["fps"], d["starts"]) == (20, 60, 20, [100, 370, 640, 910])
  c = d["clips"]
  assert len(c) == 400
  assert [x["clip"] for x in c] == list(range(400))
  assert [x["segment"] for x in c] == [i // 4 for i in range(400)]
  assert [x["start"] for x in c[:4]] == d["starts"] and c[5]["key"] == "seg1"
  assert all(x["start"] + d["ctx"] + d["horizon"] <= 1200 for x in c)


def test_clip_tokens_formas():
  val = np.arange(3 * 1200 * 128).reshape(3, 1200, 128) % 1024
  d = {"ctx": 20, "horizon": 60, "clips": [{"segment": 2, "start": 100}, {"segment": 0, "start": 910}]}
  ctx, real = rollouts.clip_tokens(val, d)
  assert ctx.shape == (2, 20, 128) and real.shape == (2, 60, 128)
  assert (ctx[0] == val[2, 100:120]).all() and (real[1] == val[0, 930:990]).all()
