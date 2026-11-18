"""Recetas, schedules, continuar corridas, estado y frameskip (CPU, datos sintéticos)."""
import json
import math
import os
import time

import numpy as np
import pytest
import torch

from src import recipes as R
from src import schedules as S
from src import status as ST
from src.models import build, load_checkpoint
from src.train import main
from src.windows import Windows

CFG = {"n_layer": 2, "n_head": 2, "dim": 32, "dropout": 0.0}


def synth(n=60, t=30, seed=0):
  return np.random.default_rng(seed).integers(0, 1024, (n, t, 128)).astype(np.int16)


@pytest.fixture
def data(tmp_path, monkeypatch):
  monkeypatch.setenv("ML2_DEVICE", "cpu")
  p = tmp_path / "tok.npy"
  np.save(p, synth())
  return p


def base_args(tmp_path, data, run, extra=()):
  return ["--kind", "gpt", "--config", json.dumps(CFG), "--run", run, "--receta", "legacy_cosine", "--bs", "2", "--frames", "4",
          "--eval-every", "100", "--log-every", "1", "--data", str(data), "--ckpt-dir", str(tmp_path / "ck"),
          "--results-dir", str(tmp_path / "res"), *extra]


# ---------------------------------------------------------------- schedules

def test_schedule_shapes():
  segs = [S.linear(0.0, 1.0, pasos=4, nombre="warmup"), S.constant(1.0), S.linear(1.0, 0.1, frac=0.2, nombre="decay")]
  total = 20
  assert S.lengths(segs, total) == [(0, 4), (4, 12), (16, 4)]
  assert S.value(segs, 0, total) == 0.25 and S.value(segs, 3, total) == 1.0 and S.value(segs, 10, total) == 1.0
  assert S.value(segs, 19, total) == pytest.approx(0.1) and S.value(segs, 99, total) == pytest.approx(0.1)
  assert S.decay_start(segs, total) == 16 and S.segment_name(segs, 17, total) == "decay"
  assert S.decay_start(segs[:2], total) is None
  # el warmup escalado tiene piso de 100 pasos
  sc = [S.linear(0.0, 1.0, frac=0.01, nombre="warmup"), S.constant(1.0)]
  assert S.lengths(sc, 1000)[0][1] == 100 and S.lengths(sc, 100000)[0][1] == 1000
  c = [S.cosine(2.0, 0.1)]
  assert S.value(c, 0, 10) == pytest.approx(2.0) and S.value(c, 10, 10) == pytest.approx(0.2)


def test_recipe_roundtrip_hash_table():
  r = R.legacy_cosine(lr=2e-3)
  r2 = R.Recipe.from_dict(json.loads(json.dumps(r.to_dict())))
  assert r2 == r and r2.hash() == r.hash()
  assert r.con_pico(5e-4).hash() != r.hash() and r.con_pico(5e-4).lr(1000, 1000) == pytest.approx(5e-5)
  assert r.kind_continuacion() == "coseno" and r.decay_start(1000) == 200
  t = R.table()
  assert "gpt_s" in t and "maskgit_l" in t and "lewm" in t and "propio" in t
  for kind in ("gpt", "maskgit", "lewm"):
    from src.models import model_class
    assert model_class(kind).recipe("s").fuente


# ---------------------------------------------------------------- legacy bit a bit

def old_lr(step, lr, warmup, min_lr_frac, total):
  if step < warmup:
    return lr * (step + 1) / warmup
  prog = (step - warmup) / max(1, total - warmup)
  return lr * (min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))


def test_legacy_cosine_matches_old_code(tmp_path, data):
  total, lr, warm, mf = 6, 3e-3, 2, 0.1
  r = R.legacy_cosine(lr=lr, warmup=warm, min_lr_frac=mf)
  for t in (6, 7, 500, 100000):
    assert [r.lr(s, t) for s in range(t + 3)] == [old_lr(s, lr, warm, mf, t) for s in range(t + 3)]

  # el camino viejo de train.py (AdamW (0.9, 0.95), wd 0.1 en >= 2 dims, clip 1) a mano
  torch.manual_seed(0)
  model = build("gpt", CFG)
  w = Windows(str(data), seed=0)
  decay = [p for p in model.parameters() if p.ndim >= 2]
  other = [p for p in model.parameters() if p.ndim < 2]
  opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                          lr=lr, betas=(0.9, 0.95))
  model.train()
  for step in range(total):
    torch.manual_seed(0 * 1_000_003 + step)
    for g in opt.param_groups:
      g["lr"] = old_lr(step, lr, warm, mf, total)
    loss, _ = model.train_loss(w.sample(step, 2, 4))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
  main(base_args(tmp_path, data, "nuevo", ["--tokens", str(1032 * 6), "--warmup", "2", "--lr", "3e-3"]))
  new = load_checkpoint(tmp_path / "ck" / "nuevo" / "final.pt").state_dict()
  for k, v in model.state_dict().items():
    assert torch.equal(v, new[k]), k
  # la receta guardada y las desviaciones (--lr, --warmup, --bs y --frames difieren de la receta actual 1e-3 / 200)
  res = json.loads((tmp_path / "res" / "train_nuevo.json").read_text())
  assert {d["campo"] for d in res["desviaciones"]} == {"lr", "warmup", "batch", "frames"} and res["receta"]["name"] == "legacy_cosine"
  assert res["args"]["desviaciones"] == res["desviaciones"]
  # una corrida vieja (args de su checkpoint) se reconstruye con la misma receta
  old_args = dict(lr=lr, warmup=warm, min_lr_frac=mf, wd=0.1, clip=1.0, bs=2, accum=1, frames=4, amp=None)
  assert R.from_legacy_args(old_args).lr(3, 6) == old_lr(3, lr, warm, mf, 6)


# ---------------------------------------------------------------- continuar

WCD = [S.linear(0.0, 3e-3, pasos=2, nombre="warmup"), S.constant(3e-3), S.linear(3e-3, 3e-4, frac=0.2, nombre="decay")]


def wcd_args(tmp_path, data, run, frames, extra=()):
  return base_args(tmp_path, data, run, ["--presupuesto", str(frames), "--set", "receta.lr_segs=" + json.dumps(WCD), *extra])


def weights(path, key="model"):
  return torch.load(path, map_location="cpu", weights_only=False)[key]


def test_extend_wcd_bit_identical(tmp_path, data):
  # 8 frames por paso: 80 frames = 10 pasos (decay desde 8), 160 = 20 pasos (decay desde 16)
  main(wcd_args(tmp_path, data, "n", 80))
  ck = tmp_path / "ck"
  assert (ck / "n" / "pre_decay.pt").exists()
  pre = torch.load(ck / "n" / "pre_decay.pt", map_location="cpu", weights_only=False)
  assert pre["step"] == 8 and pre["decay_start"] == 8
  res = main(["--extend-from", "n", "--presupuesto", "160", "--ckpt-dir", str(ck), "--results-dir", str(tmp_path / "res"),
              "--data", str(data), "--eval-every", "100", "--log-every", "1"])
  assert res["run"] == "n_x16" or res["run"].startswith("n_x")
  assert res["steps"] == 20 and res["parent"] == "n" and res["parent_file"] == "pre_decay.pt"
  assert len(res["parent_sha256"]) == 64 and "continuacion_no_equivalente" not in res or not res.get("continuacion_no_equivalente")
  main(wcd_args(tmp_path, data, "directo", 160))
  a, b = weights(ck / res["run"] / "final.pt"), weights(ck / "directo" / "final.pt")
  for k in a:
    assert torch.equal(a[k], b[k]), k
  # en el inicio del decay de N: el estado de pre_decay.pt es el de entrenar la corrida grande hasta ahí
  main(wcd_args(tmp_path, data, "directo8", 160, ["--max-steps", "8"]))
  c = weights(ck / "directo8" / "last.pt")
  for k in c:
    assert torch.equal(pre["model"][k], c[k]), k
  # el manifiesto recibe receta, madre y desviaciones
  from src import store
  store.CKPT_ROOT, store.RESULTS, store.MANIFEST = ck, tmp_path / "res", tmp_path / "res" / "checkpoints.json"
  e = store.register(res["run"])
  assert e["parent"] == "n" and e["receta_hash"] == res["receta_hash"] and "desviaciones" in e


def test_extend_constant_and_cosine(tmp_path, data):
  const = [S.linear(0.0, 3e-3, pasos=2, nombre="warmup"), S.constant(3e-3)]
  main(base_args(tmp_path, data, "c", ["--presupuesto", "40", "--set", "receta.lr_segs=" + json.dumps(const)]))
  ck = tmp_path / "ck"
  assert not (ck / "c" / "pre_decay.pt").exists()
  common = ["--ckpt-dir", str(ck), "--results-dir", str(tmp_path / "res"), "--data", str(data), "--log-every", "1"]
  main(["--extend-from", "c", "--presupuesto", "80", "--run", "c2", *common])
  main(base_args(tmp_path, data, "cdir", ["--presupuesto", "80", "--set", "receta.lr_segs=" + json.dumps(const)]))
  a, b = weights(ck / "c2" / "final.pt"), weights(ck / "cdir" / "final.pt")
  assert all(torch.equal(a[k], b[k]) for k in a)
  # coseno: exige --restart y queda marcado
  main(base_args(tmp_path, data, "cos", ["--presupuesto", "40", "--warmup", "2"]))
  with pytest.raises(SystemExit, match="--restart"):
    main(["--extend-from", "cos", "--presupuesto", "80", "--run", "cos2", *common])
  res = main(["--extend-from", "cos", "--restart", "--presupuesto", "80", "--run", "cos2", *common])
  assert res["continuacion_no_equivalente"] is True and res["steps"] == 10 and res["parent"] == "cos"
  r = R.Recipe.from_dict(res["receta"])
  assert r.lr_segs[0]["fin"] == pytest.approx(0.5e-3 / 1) and r.lr_segs[1]["forma"] == "cosine"


def test_resume_still_exact(tmp_path, data):
  main(wcd_args(tmp_path, data, "full", 80))
  main(wcd_args(tmp_path, data, "split", 80, ["--max-steps", "5"]))
  main(wcd_args(tmp_path, data, "split", 80))
  a, b = weights(tmp_path / "ck" / "full" / "final.pt"), weights(tmp_path / "ck" / "split" / "final.pt")
  assert all(torch.equal(a[k], b[k]) for k in a)
  assert (tmp_path / "ck" / "split" / "pre_decay.pt").exists()


# ---------------------------------------------------------------- estado

def test_status_written_and_failed(tmp_path, data):
  main(base_args(tmp_path, data, "ok", ["--presupuesto", "40"]))
  st = json.loads((tmp_path / "ck" / "ok" / "status.json").read_text())
  assert st["estado"] == "lista" and st["paso"] == st["total"] == 5 and st["porcentaje"] == 100 and st["run"] == "ok"
  with pytest.raises(Exception):
    main(base_args(tmp_path, data, "mal", ["--presupuesto", "40", "--set", "receta.optimizer=nada"]))
  st = json.loads((tmp_path / "ck" / "mal" / "status.json").read_text())
  assert st["estado"] == "fallada" and "nada" in st["error"]


def test_status_table(tmp_path):
  ck = tmp_path / "ck"
  now = time.time()

  def put(name, **kw):
    d = ck / name
    d.mkdir(parents=True)
    ST.write_status(d, run=name, kind="gpt", paso=10, total=100, porcentaje=10.0, loss=2.5, tramo="warmup", eta_s=90,
                    **kw)
  put("viva", estado="en curso", pid=os.getpid(), hora=now)
  put("muerta", estado="en curso", pid=2 ** 22 + 12345, hora=now)
  put("vieja", estado="en curso", pid=os.getpid(), hora=now - 400)
  put("lista", estado="lista", pid=1, hora=now - 9999)
  (ck / "_descartados" / "x").mkdir(parents=True)
  (ck / "_descartados" / "x" / "status.json").write_text("{}")
  out = ST.render(str(ck))
  lines = {l.split()[0]: l for l in out.splitlines()}
  assert "colgada" in lines["muerta"] and "colgada" in lines["vieja"]
  assert "en curso" in lines["viva"] and "10.0" in lines["viva"] and "10/100" in lines["viva"]
  assert "lista" in lines["lista"] and "_descartados" not in out and "x" not in lines
  assert ST.pid_alive(os.getpid()) and not ST.pid_alive(2 ** 22 + 12345)


# ---------------------------------------------------------------- frameskip

def test_frameskip_alignment_and_default_unchanged():
  tok = synth(12, 40)
  pos = np.random.default_rng(1).normal(size=(12, 40, 6)).astype(np.float32)
  pos[:, 7] = np.nan
  w = Windows(tok, holdout=2, seed=3, poses=pos)
  # sin frameskip: idéntico al comportamiento de antes (mismos segmentos, mismas poses)
  x1, p1 = w.sample(4, 3, 10, with_poses=True)
  assert x1.shape == (3, 10, 128) and p1.shape == (3, 10, 6) and torch.equal(x1, w.sample(4, 3, 10))
  k, n = 5, 6
  x, p = w.sample(9, 4, n, with_poses=True, frameskip=k)
  assert x.shape == (4, n, 128) and p.shape == (4, n, 6 * k)
  assert torch.equal(x, w.sample(9, 4, n, frameskip=k))
  found = 0
  for i in range(4):
    # buscar la ventana: el frame j tomado es el crudo t + j*k, y su acción son las k poses crudas desde ahí
    for s in range(w.n_train):
      for t in range(40 - n * k + 1):
        if np.array_equal(tok[s, t:t + (n - 1) * k + 1:k], x[i].numpy()):
          assert np.allclose(pos[s, t:t + n * k].reshape(n, 6 * k), p[i].numpy(), equal_nan=True)
          found += 1
  assert found == 4
  mean, std = w.pose_stats(k)
  assert mean.shape == std.shape == (6 * k,) and np.allclose(mean[:6], mean[6:12])
  assert w.monitor(4, n, frameskip=k).shape == (4, n, 128)


def test_train_with_frameskip_tokens_only(tmp_path, data):
  res = main(base_args(tmp_path, data, "fs", ["--presupuesto", "40", "--frameskip", "2"]))
  assert res["steps"] == 5 and any(d["campo"] == "frameskip" for d in res["desviaciones"])
