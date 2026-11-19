"""Almacén de modelos entrenados (y rollouts) para compartirlos sin reentrenar.

El manifiesto results/checkpoints.json va en git: dice qué checkpoints existen, su sha256 y dónde están.
Los archivos pesados viven en un backend, elegido con flags o variables de entorno:
  hf   --hf REPO_ID / ML2_HF_REPO / HF_REPO   repo público de modelos en Hugging Face (HF_REPO: el del equipo)
  dir  --dir RUTA / ML2_STORE_DIR            carpeta compartida (p. ej. Google Drive: /content/drive/MyDrive/ml2/store)

Bajar no pide login si el repo es público. load_model("<run>") baja el final.pt que falte (ensure_local) y la demo el
decoder (ensure_file), por partes en paralelo, y los deja solo si el sha256 coincide con el manifiesto. Un archivo local
distinto del manifiesto no se pisa: queda al lado como <nombre>.local-<sha>. Subir pide `hf auth login` con permiso de
escritura; la CLI sube cada corrida al terminar de entrenarla (`push --si-hay-sesion`: sin sesión avisa y sigue).

  python -m src.store register --all
  python -m src.store register lewm --extra checkpoints/lewm/decoder.pt   # otro archivo de la corrida (un decoder)
  python -m src.store push gpt_s_kd [--hf usuario/repo] [--create --publico] [--rollouts]
  python -m src.store push --all                  # todo lo registrado en el manifiesto (nunca los humos)
  python -m src.store pull --all [--rollouts]     # baja final.pt y los extras de cada corrida y verifica el sha256
  python -m src.store list | verify

Layout remoto: checkpoints/<run>/final.pt, checkpoints/<extra> (p. ej. lewm/decoder.pt), rollouts/<name>.npy|.json.
"""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

# módulo-nivel para poder redirigirlos en los tests
CKPT_ROOT = Path("checkpoints")
RESULTS = Path("results")
MANIFEST = RESULTS / "checkpoints.json"
ROLL_DIR = RESULTS / "rollouts"
ROLL = "_rollouts"
HF_REPO = "KevinSoulier/ml2-gen-world-model"  # repo público del equipo en Hugging Face; lo pisan ML2_HF_REPO y --hf


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


def bajar_por_partes(url, size, dest, headers=None, partes=16, min_parte=8 << 20, timeout=60):
  """Baja url (de `size` bytes) a dest con varias conexiones HTTP Range en paralelo.

  En la máquina de desarrollo una sola conexión al CDN de Hugging Face va a ~0,4 MB/s y hf_hub_download se cuelga, pero
  la velocidad crece con las conexiones. Cada tramo escribe en su lugar del archivo; uno incompleto es un error.
  """
  n = max(1, min(partes, -(-size // min_parte)))
  with open(dest, "wb") as f:
    f.truncate(size)

  def tramo(i):
    ini, fin = i * size // n, (i + 1) * size // n - 1
    if fin < ini:
      return
    req = urllib.request.Request(url, headers={**(headers or {}), "Range": f"bytes={ini}-{fin}"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "r+b") as f:
      if n > 1 and r.status != 206:
        raise RuntimeError(f"el servidor no respetó el Range (HTTP {r.status})")
      f.seek(ini)
      pos = ini
      while b := r.read(1 << 20):
        f.write(b)
        pos += len(b)
    if pos != fin + 1:
      raise RuntimeError(f"tramo {i}: llegaron {pos - ini} de {fin - ini + 1} bytes")

  with ThreadPoolExecutor(n) as ex:
    list(ex.map(tramo, range(n)))


def bajar_hf(repo, rel, dest):
  """Un archivo de un repo de modelos de HF, sin login si es público. Con sesión, el token va solo a huggingface.co
  (no a la URL firmada del CDN)."""
  from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url
  meta = get_hf_file_metadata(hf_hub_url(repo, rel, repo_type="model"))
  token = get_token()
  host = urlparse(meta.location).hostname or ""
  hdr = {"Authorization": f"Bearer {token}"} if token and host.endswith("huggingface.co") else {}
  bajar_por_partes(meta.location, meta.size, dest, hdr)


class HfStore:
  def __init__(self, repo, create=False, publico=False):
    from huggingface_hub import HfApi
    self.repo, self.key, self.ref, self.create, self.publico, self.ready = repo, "hf", repo, create, publico, False
    self.api = HfApi()

  def _login(self):
    """Comprueba la sesión al primer uso de put (bajar de un repo público no la necesita)."""
    if self.ready:
      return
    try:
      self.api.whoami()
    except Exception as e:
      raise RuntimeError(f"Hugging Face: no hay sesión ({type(e).__name__}). Corré `hf auth login` o definí HF_TOKEN.")
    if self.create:
      self.api.create_repo(self.repo, repo_type="model", private=not self.publico, exist_ok=True)
    self.ready = True

  def put(self, local, rel):
    self._login()
    self.api.upload_file(path_or_fileobj=str(local), path_in_repo=rel, repo_id=self.repo, repo_type="model")

  def get(self, rel, dest):
    try:
      bajar_hf(self.repo, rel, dest)
      return
    except Exception as e:  # noqa: BLE001 (el sha256 del manifiesto valida lo que llegue por cualquiera de los dos caminos)
      print(f"hf:{self.repo}: la bajada por partes de {rel} falló ({type(e).__name__}: {e}); pruebo con hf_hub_download")
    from huggingface_hub import hf_hub_download
    with tempfile.TemporaryDirectory(dir=Path(dest).parent) as tmp:
      got = hf_hub_download(self.repo, rel, repo_type="model", local_dir=tmp)
      shutil.copyfile(got, dest)


def env_stores(dir=None, hf=None, create=False, publico=False):
  """Backends pedidos por flags, variables de entorno o el repo del equipo (HF_REPO); el dir primero."""
  dir = dir or os.environ.get("ML2_STORE_DIR")
  hf = hf or os.environ.get("ML2_HF_REPO") or HF_REPO
  return ([DirStore(dir)] if dir else []) + ([HfStore(hf, create, publico)] if hf else [])


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
      apartar(dest, want_sha)
      os.replace(tmp, dest)
      return s
    tmp.unlink()
    errs.append(f"{s.key}:{s.ref}: sha256 distinto al del manifiesto")
  raise RuntimeError(f"no pude traer {rel}: " + ("; ".join(errs) if errs else "sin backend configurado "
                     "(--dir/--hf o ML2_STORE_DIR/ML2_HF_REPO)"))


def apartar(path, want_sha):
  """Un archivo local distinto del manifiesto no se pisa: queda al lado como <nombre>.local-<sha8>."""
  path = Path(path)
  if not path.exists():
    return None
  got = sha256(path)
  if got == want_sha:
    return None
  aparte = path.with_name(f"{path.name}.local-{got[:8]}")
  os.replace(path, aparte)
  print(f"{path}: no coincide con el manifiesto; la copia local queda en {aparte}")
  return aparte


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
  for k in ("receta", "receta_hash", "desviaciones", "parent", "parent_sha256", "parent_file",
            "continuacion_no_equivalente"):  # de dónde sale el checkpoint: receta, desviaciones y corrida madre
    if k in tj:
      entry[k] = tj[k]
  if old.get("sha256") == digest and old.get("remotes"):
    entry["remotes"] = old["remotes"]  # si el archivo cambió, los remotos viejos ya no sirven
  if old.get("extras"):
    entry["extras"] = old["extras"]  # decoders: cada uno con su sha256 y sus remotos
  m[run] = entry
  save_manifest(m)
  return entry


def _rel_ckpt(path):
  """Ruta relativa a checkpoints/ (posix), o ValueError si el archivo no está adentro."""
  try:
    return Path(path).resolve().relative_to(CKPT_ROOT.resolve()).as_posix()
  except ValueError:
    raise ValueError(f"{path} no está dentro de {CKPT_ROOT}") from None


def extra_path(rel):
  return CKPT_ROOT / rel


def register_extra(run, path):
  """Anota otro archivo de la corrida (p. ej. su decoder) en su entrada del manifiesto, con su sha256."""
  m = load_manifest()
  if run not in m or run == ROLL:
    raise KeyError(f"{run} no está en {MANIFEST}: registrá la corrida antes que sus extras")
  rel = _rel_ckpt(path)
  p = extra_path(rel)
  if not p.exists():
    raise FileNotFoundError(f"{p} no existe")
  digest = sha256(p)
  ex = m[run].setdefault("extras", {})
  old = ex.get(rel, {})
  ex[rel] = {"sha256": digest, "bytes": p.stat().st_size}
  if old.get("sha256") == digest and old.get("remotes"):
    ex[rel]["remotes"] = old["remotes"]
  save_manifest(m)
  return ex[rel]


def _extra_ok(rel, x):
  p = extra_path(rel)
  return p.exists() and p.stat().st_size == x["bytes"] and sha256(p) == x["sha256"]


def pull_extra(rel, x, stores=None):
  p = extra_path(rel)
  if _extra_ok(rel, x):
    return p
  s = fetch((stores or []) + recorded_stores(x), f"checkpoints/{rel}", p, x["sha256"])
  print(f"{rel}: bajado de {s.key}:{s.ref}")
  return p


def ensure_file(path):
  """Un archivo de checkpoints/ (p. ej. un decoder): si está anotado en el manifiesto y falta o no coincide, lo baja.

  Devuelve la ruta si quedó disponible, o None si no está ni se pudo bajar (avisa por qué).
  """
  p = Path(path)
  try:
    rel = _rel_ckpt(p)
  except ValueError:
    return p if p.exists() else None
  for run, e in load_manifest().items():
    x = e.get("extras", {}).get(rel) if run != ROLL else None
    if x is not None:
      try:
        return pull_extra(rel, x, env_stores())
      except RuntimeError as err:
        print(f"{rel}: no pude bajarlo ({err})")
        return p if p.exists() else None
  return p if p.exists() else None


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
      for rel in entry.get("extras", {}):
        s.put(extra_path(rel), f"checkpoints/{rel}")
        print(f"{rel}: subido a {s.key}:{s.ref}")
    m = load_manifest()
    remember(m[run], stores)
    for x in m[run].get("extras", {}).values():
      remember(x, stores)
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


def pull_run(run, stores=None, extras=False):
  """Trae final.pt de run si falta o no coincide (y con extras=True, sus decoders). Devuelve la ruta local."""
  m = load_manifest()
  if run not in m:
    raise KeyError(f"{run} no está en {MANIFEST}")
  e, p = m[run], final_path(run)
  if extras:
    for rel, x in e.get("extras", {}).items():
      pull_extra(rel, x, stores)
  if p.exists() and p.stat().st_size == e["bytes"] and sha256(p) == e["sha256"]:
    return p
  s = fetch((stores or []) + recorded_stores(e), f"checkpoints/{run}/{e['file']}", p, e["sha256"])
  print(f"{run}: bajado de {s.key}:{s.ref}")
  return p


def push_si_se_puede(runs, stores, rollouts=False):
  """El push del final de un entrenamiento: nunca corta la cola. Sin almacén o sin sesión de HF avisa y no sube; si la
  subida falla (sin permiso de escritura, sin red) avisa y deja todo local, para reintentar con `push <run>`."""
  if not stores:
    print(f"no subo {', '.join(runs)}: no hay almacén configurado (HF_REPO en src/store.py, ML2_HF_REPO o --hf)")
    return False
  if any(s.key == "hf" for s in stores):
    from huggingface_hub import get_token
    if get_token() is None:
      print(f"no subo {', '.join(runs)}: no hay sesión de Hugging Face (`hf auth login` con permiso de escritura)")
      return False
  try:
    push(runs, stores, rollouts)
    return True
  except Exception as e:  # noqa: BLE001
    print(f"no pude subir {', '.join(runs)} ({type(e).__name__}: {e}); queda local: `python -m src.store push "
          f"{' '.join(runs)}` lo reintenta")
    return False


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
    for rel, x in sorted(e.get("extras", {}).items()):
      xr = ",".join(f"{k}={v}" for k, v in x.get("remotes", {}).items()) or "-"
      xp = extra_path(rel)
      xl = "no" if not xp.exists() else ("yes" if xp.stat().st_size == x["bytes"] else "BAD")
      rows.append((f"  {rel}", "extra", "", "", x["sha256"][:10], xl, xr))
  w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
  lines = ["  ".join(c.ljust(w[i]) for i, c in enumerate(r)).rstrip() for r in rows]
  for name, e in sorted(m.get(ROLL, {}).items()):
    lines.append(f"rollout {name} ({e['bytes'] / 1e6:.1f} MB) {e['npy_sha256'][:10]}")
  return "\n".join(lines)


def verify():
  """Compara los archivos locales con el manifiesto. Devuelve la lista de problemas (corruptos)."""
  m, bad = load_manifest(), []
  checks = [(f"{r}", final_path(r), e["sha256"]) for r, e in m.items() if r != ROLL]
  checks += [(rel, extra_path(rel), x["sha256"]) for r, e in m.items() if r != ROLL
             for rel, x in e.get("extras", {}).items()]
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
    if name == "register":
      s.add_argument("--extra", action="append", default=[], help="otro archivo de la corrida (p. ej. su decoder)")
    else:
      s.add_argument("--rollouts", action="store_true", help="también results/rollouts/*.npy|.json")
      s.add_argument("--dir", help="carpeta (default ML2_STORE_DIR)")
      s.add_argument("--hf", help="repo de Hugging Face (default ML2_HF_REPO o el del equipo, HF_REPO)")
    if name == "push":
      s.add_argument("--create", action="store_true", help="crea el repo HF si no existe (privado salvo --publico)")
      s.add_argument("--publico", action="store_true", help="con --create: el repo nuevo es público")
      s.add_argument("--si-hay-sesion", action="store_true",
                     help="lo usa la CLI al terminar un entrenamiento: sin almacén, sin sesión o si falla, avisa y "
                          "sale con 0")
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
    if a.cmd in ("pull", "push"):  # push --all: lo registrado (los humos no se registran), no todo lo del disco
      runs = [r for r in load_manifest() if r != ROLL] if a.all else a.runs
    else:
      runs = finished_runs() if a.all else a.runs
    if not runs and not (a.cmd != "register" and a.rollouts):
      raise SystemExit(f"{a.cmd}: indicá runs o --all")
    if a.cmd == "register":
      if a.extra and len(runs) != 1:
        raise SystemExit("register --extra: indicá una sola corrida")
      for r in runs:
        e = register(r)
        print(f"{r}: {e['kind']} {e['params'] / 1e6:.1f}M {e['sha256'][:10]}")
      for x in a.extra:
        print(f"{runs[0]}: extra {_rel_ckpt(x)} {register_extra(runs[0], x)['sha256'][:10]}")
    elif a.cmd == "push":
      stores = env_stores(a.dir, a.hf, a.create, a.publico)
      if a.si_hay_sesion:
        push_si_se_puede(runs, stores, a.rollouts)
      else:
        push(runs, stores, a.rollouts)
    else:
      stores = env_stores(a.dir, a.hf)
      for r in runs:
        pull_run(r, stores, extras=True)
      if a.rollouts:
        pull_rollouts(stores)


if __name__ == "__main__":
  main()
