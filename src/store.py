"""Almacén de modelos entrenados (y rollouts) para compartirlos sin reentrenar.

El manifiesto results/checkpoints.json va en git: dice qué checkpoints existen, su sha256 y dónde están.
Los archivos pesados viven en un backend, elegido con flags o variables de entorno:
  dir  ML2_STORE_DIR / --dir RUTA     carpeta compartida (p. ej. Google Drive: /content/drive/MyDrive/ml2/store)
  hf   ML2_HF_REPO  / --hf REPO_ID    repo de modelos en Hugging Face (hf auth login o HF_TOKEN)

  python -m src.store register --all
  python -m src.store push gpt_s_kd --dir /content/drive/MyDrive/ml2/store [--rollouts]
  python -m src.store push --all --hf usuario/ml2-wm [--create]
  python -m src.store pull --all [--rollouts]     # baja checkpoints/<run>/final.pt y verifica el sha256
  python -m src.store list | verify

Nada se sube sin un `push` explícito. Layout remoto: checkpoints/<run>/final.pt, rollouts/<name>.npy|.json.
"""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

# módulo-nivel para poder redirigirlos en los tests
CKPT_ROOT = Path("checkpoints")
RESULTS = Path("results")
MANIFEST = RESULTS / "checkpoints.json"
ROLL_DIR = RESULTS / "rollouts"
ROLL = "_rollouts"


def sha256(path, chunk=1 << 22):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    while b := f.read(chunk):
      h.update(b)
  return h.hexdigest()


def load_manifest():
  return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def save_manifest(m):
  MANIFEST.parent.mkdir(parents=True, exist_ok=True)
  MANIFEST.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")


def final_path(run):
  return CKPT_ROOT / run / "final.pt"


def finished_runs():
  """Runs terminados: final.pt y results/train_<run>.json existen (los demás aún entrenan)."""
  if not CKPT_ROOT.exists():
    return []
  return sorted(d.name for d in CKPT_ROOT.iterdir()
                if (d / "final.pt").exists() and (RESULTS / f"train_{d.name}.json").exists())


# ---------- backends ----------

class DirStore:
  def __init__(self, path):
    self.root = Path(path)
    self.key = "dir"
    self.ref = str(self.root)

  def put(self, local, rel):
    dst = self.root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(local, tmp)
    os.replace(tmp, dst)

  def get(self, rel, dest):
    src = self.root / rel
    if not src.exists():
      raise FileNotFoundError(f"{src} no existe")
    shutil.copyfile(src, dest)


class HfStore:
  def __init__(self, repo, create=False):
    from huggingface_hub import HfApi
    self.repo, self.key, self.ref, self.create, self.ready = repo, "hf", repo, create, False
    self.api = HfApi()

  def _login(self):
    """Comprueba la sesión al primer uso (así un remoto hf anotado no estorba si el dir basta)."""
    if self.ready:
      return
    try:
      self.api.whoami()
    except Exception as e:
      raise RuntimeError(f"Hugging Face: no hay sesión ({type(e).__name__}). Corré `hf auth login` o definí HF_TOKEN.")
    if self.create:
      self.api.create_repo(self.repo, repo_type="model", private=True, exist_ok=True)
    self.ready = True

  def put(self, local, rel):
    self._login()
    self.api.upload_file(path_or_fileobj=str(local), path_in_repo=rel, repo_id=self.repo, repo_type="model")

  def get(self, rel, dest):
    from huggingface_hub import hf_hub_download
    self._login()
    with tempfile.TemporaryDirectory(dir=Path(dest).parent) as tmp:
      got = hf_hub_download(self.repo, rel, repo_type="model", local_dir=tmp)
      shutil.copyfile(got, dest)


def env_stores(dir=None, hf=None, create=False):
  """Backends pedidos por flags o variables de entorno (dir primero)."""
  dir = dir or os.environ.get("ML2_STORE_DIR")
  hf = hf or os.environ.get("ML2_HF_REPO")
  return ([DirStore(dir)] if dir else []) + ([HfStore(hf, create)] if hf else [])


def recorded_stores(entry):
  """Backends anotados en el manifiesto (el dir solo si existe en esta máquina)."""
  r, out = entry.get("remotes", {}), []
  if r.get("dir") and Path(r["dir"]).exists():
    out.append(DirStore(r["dir"]))
  if r.get("hf"):
    out.append(HfStore(r["hf"]))
  return out


def fetch(stores, rel, dest, want_sha):
  """Baja rel a dest desde el primer backend que lo tenga; solo lo deja si el sha256 coincide."""
  dest = Path(dest)
  dest.parent.mkdir(parents=True, exist_ok=True)
  tmp = dest.with_name(dest.name + ".part")
  errs = []
  for s in stores:
    try:
      s.get(rel, tmp)
    except Exception as e:
      errs.append(f"{s.key}:{s.ref}: {type(e).__name__}: {e}")
      continue
    if sha256(tmp) == want_sha:
      os.replace(tmp, dest)
      return s
    tmp.unlink()
    errs.append(f"{s.key}:{s.ref}: sha256 distinto al del manifiesto")
  raise RuntimeError(f"no pude traer {rel}: " + ("; ".join(errs) if errs else "sin backend configurado "
                     "(--dir/--hf o ML2_STORE_DIR/ML2_HF_REPO)"))


def remember(entry, stores):
  for s in stores:
    entry.setdefault("remotes", {})[s.key] = s.ref


# ---------- register ----------

def register(run):
  """Añade/actualiza la entrada del manifiesto a partir de checkpoints/<run>/final.pt."""
  import torch
  p = final_path(run)
  if not p.exists() or not (RESULTS / f"train_{run}.json").exists():
    raise FileNotFoundError(f"{run}: no está terminado (falta final.pt o results/train_{run}.json)")
  ck = torch.load(p, map_location="cpu", weights_only=False)
  m = load_manifest()
  digest = sha256(p)
  old = m.get(run, {})
  # params de inferencia según train.py si están (los latentes con EMA guardan además el encoder target)
  tj = json.loads((RESULTS / f"train_{run}.json").read_text() or "{}")
  params = tj.get("params") or sum(v.numel() for v in ck["model"].values())
  entry = {"file": "final.pt", "sha256": digest, "bytes": p.stat().st_size, "kind": ck["kind"], "config": ck["config"],
           "tokens": ck["tokens"], "step": ck["step"], "params": params,
           "train_json": f"results/train_{run}.json"}
  if old.get("sha256") == digest and old.get("remotes"):
    entry["remotes"] = old["remotes"]  # si el archivo cambió, los remotos viejos ya no sirven
  m[run] = entry
  save_manifest(m)
  return entry


def register_rollouts():
  m = load_manifest()
  r = m.setdefault(ROLL, {})
  for npy in sorted(ROLL_DIR.glob("*.npy")):
    js = npy.with_suffix(".json")
    if not js.exists():
      continue
    new = {"npy_sha256": sha256(npy), "json_sha256": sha256(js), "bytes": npy.stat().st_size}
    if all(r.get(npy.stem, {}).get(k) == v for k, v in new.items()):
      continue
    r[npy.stem] = new
  save_manifest(m)
  return sorted(r)


# ---------- push / pull ----------

def push(runs, stores, rollouts=False):
  if not stores:
    raise SystemExit("push: falta backend (--dir/--hf o ML2_STORE_DIR/ML2_HF_REPO)")
  for run in runs:
    entry = register(run)
    for s in stores:
      s.put(final_path(run), f"checkpoints/{run}/final.pt")
      print(f"{run}: subido a {s.key}:{s.ref}")
    m = load_manifest()
    remember(m[run], stores)
    save_manifest(m)
  if rollouts:
    names = register_rollouts()
    for name in names:
      for ext in ("npy", "json"):
        for s in stores:
          s.put(ROLL_DIR / f"{name}.{ext}", f"rollouts/{name}.{ext}")
      print(f"rollout {name}: subido")
    m = load_manifest()
    for name in names:
      remember(m[ROLL][name], stores)
    save_manifest(m)


def pull_run(run, stores=None):
  """Trae final.pt de run si falta o está corrupto. Devuelve la ruta local."""
  m = load_manifest()
  if run not in m:
    raise KeyError(f"{run} no está en {MANIFEST}")
  e, p = m[run], final_path(run)
  if p.exists() and p.stat().st_size == e["bytes"] and sha256(p) == e["sha256"]:
    return p
  s = fetch((stores or []) + recorded_stores(e), f"checkpoints/{run}/{e['file']}", p, e["sha256"])
  print(f"{run}: bajado de {s.key}:{s.ref}")
  return p


def pull_rollouts(stores=None):
  m = load_manifest()
  for name, e in sorted(m.get(ROLL, {}).items()):
    for ext in ("npy", "json"):
      p, want = ROLL_DIR / f"{name}.{ext}", e[f"{ext}_sha256"]
      if p.exists() and sha256(p) == want:
        continue
      fetch((stores or []) + recorded_stores(e), f"rollouts/{name}.{ext}", p, want)
      print(f"rollout {name}.{ext}: bajado")


def ensure_local(run):
  """checkpoints/<run>/final.pt, bajándolo del almacén si falta o no coincide con el manifiesto."""
  p, m = final_path(run), load_manifest()
  if run not in m:
    if p.exists():
      return p
    raise FileNotFoundError(f"{p} no existe y {run} no está en {MANIFEST}")
  try:
    return pull_run(run, env_stores())
  except Exception as e:
    raise RuntimeError(f"{run}: falta o está corrupto {p} y no pude bajarlo. {e}") from e


# ---------- list / verify ----------

def local_state(run, e):
  p = final_path(run)
  if not p.exists():
    return "no"
  return "yes" if p.stat().st_size == e["bytes"] else "BAD"


def list_runs():
  m = load_manifest()
  rows = [("run", "kind", "params", "tokens", "sha256", "local", "remotes")]
  for run, e in sorted(m.items()):
    if run == ROLL:
      continue
    rem = ",".join(f"{k}={v}" for k, v in e.get("remotes", {}).items()) or "-"
    rows.append((run, e["kind"], f"{e['params'] / 1e6:.1f}M", f"{e['tokens'] / 1e6:.0f}M", e["sha256"][:10],
                 local_state(run, e), rem))
  w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
  lines = ["  ".join(c.ljust(w[i]) for i, c in enumerate(r)).rstrip() for r in rows]
  for name, e in sorted(m.get(ROLL, {}).items()):
    lines.append(f"rollout {name} ({e['bytes'] / 1e6:.1f} MB) {e['npy_sha256'][:10]}")
  return "\n".join(lines)


def verify():
  """Compara los archivos locales con el manifiesto. Devuelve la lista de problemas (corruptos)."""
  m, bad = load_manifest(), []
  checks = [(f"{r}", final_path(r), e["sha256"]) for r, e in m.items() if r != ROLL]
  for name, e in m.get(ROLL, {}).items():
    checks += [(f"rollout {name}.{x}", ROLL_DIR / f"{name}.{x}", e[f"{x}_sha256"]) for x in ("npy", "json")]
  for label, p, want in checks:
    if not p.exists():
      print(f"{label}: falta (no está en esta máquina)")
    elif sha256(p) != want:
      print(f"{label}: CORRUPTO (sha256 distinto)")
      bad.append(label)
    else:
      print(f"{label}: ok")
  return bad


# ---------- CLI ----------

def main(argv=None):
  p = argparse.ArgumentParser(prog="python -m src.store", description=__doc__.split("\n")[0])
  sub = p.add_subparsers(dest="cmd", required=True)
  for name in ("register", "push", "pull"):
    s = sub.add_parser(name)
    s.add_argument("runs", nargs="*")
    s.add_argument("--all", action="store_true")
    if name != "register":
      s.add_argument("--rollouts", action="store_true", help="también results/rollouts/*.npy|.json")
      s.add_argument("--dir", help="carpeta (default ML2_STORE_DIR)")
      s.add_argument("--hf", help="repo de Hugging Face (default ML2_HF_REPO)")
    if name == "push":
      s.add_argument("--create", action="store_true", help="crea el repo HF (privado) si no existe")
  sub.add_parser("list")
  sub.add_parser("verify")
  a = p.parse_args(argv)
  try:
    run_cmd(a)
  except RuntimeError as e:
    raise SystemExit(str(e))


def run_cmd(a):
  if a.cmd == "list":
    print(list_runs())
  elif a.cmd == "verify":
    raise SystemExit(1 if verify() else 0)
  else:
    if a.cmd == "pull":
      runs = [r for r in load_manifest() if r != ROLL] if a.all else a.runs
    else:
      runs = finished_runs() if a.all else a.runs
    if not runs and not (a.cmd != "register" and a.rollouts):
      raise SystemExit(f"{a.cmd}: indicá runs o --all")
    if a.cmd == "register":
      for r in runs:
        e = register(r)
        print(f"{r}: {e['kind']} {e['params'] / 1e6:.1f}M {e['sha256'][:10]}")
    elif a.cmd == "push":
      push(runs, env_stores(a.dir, a.hf, a.create), a.rollouts)
    else:
      stores = env_stores(a.dir, a.hf)
      for r in runs:
        pull_run(r, stores)
      if a.rollouts:
        pull_rollouts(stores)


if __name__ == "__main__":
  main()
