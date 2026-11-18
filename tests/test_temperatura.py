"""Barrido de temperatura (E2) y clips del holdout: puro, y de punta a punta en CPU con un modelo chico."""
import json

import numpy as np
import pytest

from src import rollouts, temperatura


def test_grilla_parte_del_default_del_modelo():
  assert temperatura.grilla(1.0) == [0.7, 0.8, 0.9, 1.0, 1.1]
  assert temperatura.grilla(2.0) == [1.4, 1.6, 1.8, 2.0, 2.2]                  # genie: default 2
  assert temperatura.grilla(2.0, [0.5, 1.0]) == [0.5, 1.0, 2.0]                # el default siempre está


def test_default_de_cada_modelo():
  from src.models import load_model
  assert rollouts.temperatura_default(load_model("random:genie_s")) == 2.0
  assert rollouts.temperatura_default(load_model("random:maskgit_s")) == 1.0
  assert rollouts.temperatura_default(load_model("random:gpt_s")) == 1.0


def test_nombre_solo_marca_la_temperatura_distinta_del_default():
  assert rollouts.default_name("genie_s", temperature=2.0, default_temperature=2.0) == "genie_s"
  assert rollouts.default_name("genie_s", temperature=1.0, default_temperature=2.0) == "genie_s_t1"
  assert rollouts.default_name("gpt_s", temperature=0.8) == "gpt_s_t0.8"


def test_elegir_menor_fvd_con_desempate_por_lpips():
  t = [{"T": 0.8, "fvd": 100.0, "lpips": 0.40}, {"T": 1.0, "fvd": 101.0, "lpips": 0.35}, {"T": 1.2, "fvd": 150.0, "lpips": 0.1}]
  assert temperatura.elegir(t, 1.0, tol=0.0)["T"] == 0.8
  assert temperatura.elegir(t, 1.0, tol=0.02)["T"] == 1.0                      # 101 está a 1% del mejor: gana por LPIPS
  assert temperatura.elegir([{"T": 0.9, "fvd": 1.0, "lpips": 0.5}, {"T": 1.1, "fvd": 1.0, "lpips": 0.5}], 1.0)["T"] in (0.9, 1.1)


def test_clips_del_holdout_son_los_ultimos_segmentos_de_train(tmp_path):
  toks = tmp_path / "t.npy"
  np.save(toks, np.zeros((7, 10, 1), np.int16))
  d = rollouts.make_clips_holdout(tmp_path / "c.json", toks, holdout=3)
  assert len(d["clips"]) == 3 * len(rollouts.STARTS) and d["source"] == "train_holdout"
  assert {c["row"] for c in d["clips"]} == {4, 5, 6} and {c["segment"] for c in d["clips"]} == {0, 1, 2}
  c, v = rollouts.load_clips_holdout(tmp_path / "c.json", toks, 3)
  assert v.shape[0] == 3 and c["clips"] == d["clips"]
  assert rollouts.CLIPS_HOLDOUT.name == "clips_holdout.json" and rollouts.CLIPS.name == "clips_eval.json"


def test_rollouts_de_val_siguen_igual():
  assert rollouts.load_clips().get("source") is None and len(rollouts.load_clips()["clips"]) == 400


def test_temperatura_de_punta_a_punta(tmp_path, monkeypatch):
  """Modelo chico, 8 clips (2 segmentos), rollout y decoder falsos (rápidos); el resto es real: I3D, LPIPS y bootstrap."""
  import src.models as M
  import src.viz as V
  from src.models import build
  model = build("maskgit", size="s")
  model.name = "pytest_mg"
  rng = np.random.default_rng(0)
  llamadas = []

  def rollout_falso(ctx, n, temperature=None, steps=None, generator=None):
    llamadas.append(temperature)
    import torch
    return torch.randint(0, 1024, (len(ctx), n, 128))
  monkeypatch.setattr(model, "rollout", rollout_falso)
  monkeypatch.setattr(M, "load_model", lambda spec, dev="cpu": model)
  monkeypatch.setattr(V, "load_decoder", lambda dev="cpu": None)
  monkeypatch.setattr(V, "decode", lambda dec, tok, bs=16: np.broadcast_to((np.asarray(tok) % 256).astype(np.uint8)[:, :, None, None],
                                                                            (len(tok), 128, 256, 3)).copy())
  r = temperatura.main(["--model", "x", "--limite", "8", "--temperaturas", "0.9", "--bs", "4", "--boot", "4",
                        "--out", str(tmp_path)])
  assert sorted(set(llamadas)) == [0.9, 1.0]
  d = json.loads((tmp_path / "temperatura_pytest_mg.json").read_text())
  assert d["clips"] == "holdout" and d["n_clips"] == 8 and d["default"] == 1.0
  assert [row["T"] for row in d["tabla"]] == [0.9, 1.0] and d["elegida"] in (0.9, 1.0)
  assert all(row["fvd"] > 0 and "lpips" in row for row in d["tabla"]) and r["elegida"] == d["elegida"]
