import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src import store


@pytest.fixture
def env(tmp_path, monkeypatch):
  """Raíz de checkpoints, resultados, rollouts y manifiesto falsos; sin backends en el entorno."""
  monkeypatch.setattr(store, "CKPT_ROOT", tmp_path / "checkpoints")
  monkeypatch.setattr(store, "RESULTS", tmp_path / "results")
  monkeypatch.setattr(store, "MANIFEST", tmp_path / "results" / "checkpoints.json")
  monkeypatch.setattr(store, "ROLL_DIR", tmp_path / "results" / "rollouts")
  monkeypatch.delenv("ML2_STORE_DIR", raising=False)
  monkeypatch.delenv("ML2_HF_REPO", raising=False)
  (tmp_path / "results").mkdir()
  return tmp_path


def make_run(env, run, done=True):
  d = env / "checkpoints" / run
  d.mkdir(parents=True)
  sd = {"w": torch.randn(4, 5), "b": torch.randn(5)}
  torch.save({"kind": "gpt", "config": {"dim": 8}, "model": sd, "name": run, "step": 7, "tokens": 1234}, d / "final.pt")
  if done:
    (env / "results" / f"train_{run}.json").write_text("{}")
  return d / "final.pt"


def make_rollout(env, name):
  d = env / "results" / "rollouts"
  d.mkdir(exist_ok=True)
  np.save(d / f"{name}.npy", np.arange(100, dtype=np.int16))
  (d / f"{name}.json").write_text(json.dumps({"name": name}))


def test_register(env):
  p = make_run(env, "a")
  e = store.register("a")
  assert e["sha256"] == store.sha256(p) and e["bytes"] == p.stat().st_size
  assert e["params"] == 25 and e["kind"] == "gpt" and e["tokens"] == 1234 and e["step"] == 7
  assert e["train_json"] == "results/train_a.json"
  assert json.loads(store.MANIFEST.read_text())["a"]["config"] == {"dim": 8}


def test_register_solo_terminados(env):
  make_run(env, "a")
  make_run(env, "corriendo", done=False)
  assert store.finished_runs() == ["a"]
  with pytest.raises(FileNotFoundError):
    store.register("corriendo")


def test_roundtrip_dir(env):
  p = make_run(env, "a")
  want = store.sha256(p)
  remote = env / "remote"
  store.push(["a"], [store.DirStore(remote)])
  assert (remote / "checkpoints" / "a" / "final.pt").exists()
  assert store.load_manifest()["a"]["remotes"]["dir"] == str(remote)
  p.unlink()
  assert store.pull_run("a", [store.DirStore(remote)]) == p
  assert store.sha256(p) == want
  assert store.verify() == []


def test_pull_cli_y_skip(env, capsys):
  p = make_run(env, "a")
  remote = env / "remote"
  store.main(["push", "--all", "--dir", str(remote)])
  p.unlink()
  store.main(["pull", "--all", "--dir", str(remote)])
  assert p.exists()
  capsys.readouterr()
  store.main(["pull", "a", "--dir", str(remote)])  # ya está con el hash correcto: no baja nada
  assert "bajado" not in capsys.readouterr().out


def test_ensure_local(env, monkeypatch):
  p = make_run(env, "a")
  remote = env / "remote"
  store.push(["a"], [store.DirStore(remote)])
  p.unlink()
  monkeypatch.setenv("ML2_STORE_DIR", str(remote))
  assert store.ensure_local("a") == p and p.exists()


def test_ensure_local_errores(env):
  with pytest.raises(FileNotFoundError):
    store.ensure_local("nadie")
  p = make_run(env, "a")
  store.register("a")
  p.unlink()
  with pytest.raises(RuntimeError, match="sin backend"):
    store.ensure_local("a")


def test_corrupto(env, monkeypatch, capsys):
  p = make_run(env, "a")
  remote = env / "remote"
  store.push(["a"], [store.DirStore(remote)])
  good = p.read_bytes()
  p.write_bytes(good[:-1] + bytes([good[-1] ^ 1]))  # mismo tamaño, un bit distinto
  assert store.verify() == ["a"]
  assert "CORRUPTO" in capsys.readouterr().out
  with pytest.raises(SystemExit):
    store.main(["verify"])
  monkeypatch.setenv("ML2_STORE_DIR", str(remote))
  store.ensure_local("a")
  assert p.read_bytes() == good and store.verify() == []


def test_fetch_rechaza_hash_malo(env):
  make_run(env, "a")
  remote = env / "remote"
  store.push(["a"], [store.DirStore(remote)])
  (remote / "checkpoints" / "a" / "final.pt").write_bytes(b"basura")
  (env / "checkpoints" / "a" / "final.pt").unlink()
  with pytest.raises(RuntimeError, match="sha256 distinto"):
    store.pull_run("a", [store.DirStore(remote)])
  assert not (env / "checkpoints" / "a" / "final.pt").exists()


def test_register_conserva_remotos_solo_si_igual(env):
  p = make_run(env, "a")
  store.push(["a"], [store.DirStore(env / "remote")])
  store.register("a")
  assert "remotes" in store.load_manifest()["a"]
  p.write_bytes(p.read_bytes() + b"x")
  store.register("a")
  assert "remotes" not in store.load_manifest()["a"]


def test_list(env):
  make_run(env, "a")
  make_run(env, "b")
  store.push(["a", "b"], [store.DirStore(env / "remote")])
  (env / "checkpoints" / "b" / "final.pt").unlink()
  out = store.list_runs()
  la, lb = [l for l in out.splitlines() if l.startswith(("a ", "b "))]
  assert "gpt" in la and "yes" in la and store.load_manifest()["a"]["sha256"][:10] in la
  assert "no" in lb.split() and "dir=" in lb
  assert out.splitlines()[0].split()[0] == "run"


def test_rollouts(env):
  make_rollout(env, "r1")
  make_rollout(env, "r2")
  make_run(env, "a")
  remote = env / "remote"
  store.push(["a"], [store.DirStore(remote)], rollouts=True)
  m = store.load_manifest()
  assert set(m[store.ROLL]) == {"r1", "r2"} and len(m[store.ROLL]["r1"]["npy_sha256"]) == 64
  assert (remote / "rollouts" / "r1.npy").exists() and (remote / "rollouts" / "r1.json").exists()
  want = (env / "results" / "rollouts" / "r1.npy").read_bytes()
  for f in (env / "results" / "rollouts").iterdir():
    f.unlink()
  store.pull_rollouts([store.DirStore(remote)])
  assert (env / "results" / "rollouts" / "r1.npy").read_bytes() == want
  assert store.verify() == []
  assert "rollout r2" in store.list_runs()


def test_push_sin_backend(env):
  make_run(env, "a")
  with pytest.raises(SystemExit):
    store.main(["push", "a"])
  assert not (env / "remote").exists()


def test_hf_mock(env, monkeypatch):
  """El backend hf usa los paths esperados (sin red)."""
  calls = {"up": [], "create": [], "dl": []}
  blobs = {}

  class FakeApi:
    def whoami(self):
      return {"name": "x"}

    def create_repo(self, repo, **kw):
      calls["create"].append((repo, kw))

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type):
      calls["up"].append((path_in_repo, repo_id, repo_type))
      blobs[path_in_repo] = Path(path_or_fileobj).read_bytes()

  def fake_download(repo_id, filename, repo_type=None, local_dir=None, **kw):
    calls["dl"].append((repo_id, filename, repo_type))
    out = Path(local_dir) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blobs[filename])
    return str(out)

  import huggingface_hub
  monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
  monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
  p = make_run(env, "a")
  want = p.read_bytes()
  store.main(["push", "a", "--hf", "user/repo", "--create"])
  assert calls["create"] == [("user/repo", {"repo_type": "model", "private": True, "exist_ok": True})]
  assert calls["up"] == [("checkpoints/a/final.pt", "user/repo", "model")]
  assert store.load_manifest()["a"]["remotes"]["hf"] == "user/repo"
  p.unlink()
  monkeypatch.setenv("ML2_STORE_DIR", str(env / "vacio"))  # el dir no tiene nada: cae al hf anotado
  store.ensure_local("a")
  assert calls["dl"] == [("user/repo", "checkpoints/a/final.pt", "model")]
  assert p.read_bytes() == want


def test_hf_sin_sesion(env, monkeypatch):
  class NoLogin:
    def whoami(self):
      raise OSError("401")

  import huggingface_hub
  monkeypatch.setattr(huggingface_hub, "HfApi", NoLogin)
  make_run(env, "a")
  with pytest.raises(SystemExit, match="hf auth login"):
    store.main(["push", "a", "--hf", "user/repo"])
