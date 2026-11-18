"""Descarga y extracción de commaVQ a arrays de numpy.

Salida:
  data/train_tokens.npy  (N, 1200, 128) int16, desde un shard webdataset
  data/train_poses.npy   (N, 1200, 6) float32, poses por frame del mismo shard (acciones de lewm)
  data/train_keys.json   nombre de cada segmento de train, en el mismo orden
  data/val_tokens.npy    (100, 1200, 128) int16, val oficial viejo (val.zip, sin poses)

Solo las poses, del shard ya bajado (sin descargar): python -m src.data --poses-only
  data/raw/              decoder del VQ-VAE y pesos de gpt2m (commaai/commavq-gpt2m)
  results/val_split.json nombres de los segmentos de val (fijos para el benchmark)
"""
import argparse
import io
import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

RAW = Path("data/raw")
VAL_REVISION = "610fb2854cdb1b2e59e137ae340b4da5aa23cc8d"  # última revisión con val.zip


def load_npy(b: bytes) -> np.ndarray:
  return np.load(io.BytesIO(b), allow_pickle=False)


def extract_shard(tar_path: Path, with_poses: bool = False):
  """keys, tokens (N, 1200, 128) int16 y, con with_poses, poses (N, 1200, 6) float32 en el mismo orden.

  Las poses (ego-movimiento por frame: traslación y rotación) se usan como acciones en los world models
  latentes (src/models/lewm.py). Un segmento sin pose.npy queda en NaN.
  """
  keys, tokens, poses = [], [], {}
  with tarfile.open(tar_path, "r:gz") as tar:
    for m in tar:
      if not m.isfile():
        continue
      if m.name.endswith("token.npy"):
        t = load_npy(tar.extractfile(m).read())
        if t.shape != (1200, 8, 16):
          print("salteo", m.name, t.shape)
          continue
        keys.append(m.name[: -len(".token.npy")])
        tokens.append(t.reshape(1200, 128).astype(np.int16))
      elif with_poses and m.name.endswith("pose.npy"):
        p = load_npy(tar.extractfile(m).read())
        if p.shape == (1200, 6):
          poses[m.name[: -len(".pose.npy")]] = p.astype(np.float32)
  if not with_poses:
    return keys, np.stack(tokens)
  missing = [k for k in keys if k not in poses]
  if missing:
    print(f"{len(missing)} segmentos sin pose.npy (quedan en NaN), p. ej. {missing[:3]}")
  nan = np.full((1200, 6), np.nan, np.float32)
  return keys, np.stack(tokens), np.stack([poses.get(k, nan) for k in keys])


def save_poses(keys, poses, tokens):
  """data/train_poses.npy y data/train_keys.json, verificando el orden contra data/train_tokens.npy."""
  ref = Path("data/train_tokens.npy")
  if ref.exists():
    old = np.load(ref, mmap_mode="r")
    assert old.shape == tokens.shape and np.array_equal(old, tokens), \
      "los tokens re-extraídos no coinciden con data/train_tokens.npy: el orden de las poses no sería el mismo"
  np.save("data/train_poses.npy", poses)
  Path("data/train_keys.json").write_text(json.dumps(keys, indent=0))
  print(f"poses {poses.shape}, frames con NaN: {np.isnan(poses).any(-1).mean():.2%}")


def extract_val(zip_path: Path):
  keys, tokens = [], []
  with zipfile.ZipFile(zip_path) as z:
    for name in sorted(z.namelist()):
      if name.endswith(".npy"):
        t = load_npy(z.read(name))
        keys.append(name[:-4])
        tokens.append(t.reshape(1200, 128).astype(np.int16))
  return keys, np.stack(tokens)


def route_of(key: str) -> str:
  # "<hash de ruta>_<n segmento>", con o sin prefijo de carpeta
  return Path(key).name.rsplit("_", 1)[0]


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--shard", default="data-0000.tar.gz")
  ap.add_argument("--poses-only", action="store_true",
                  help="solo extrae las poses del shard ya bajado en data/raw (no descarga nada)")
  args = ap.parse_args()

  if args.poses_only:
    keys, tokens, poses = extract_shard(RAW / args.shard, with_poses=True)
    save_poses(keys, poses, tokens)
    raise SystemExit(0)

  shard = Path(hf_hub_download("commaai/commavq", args.shard, repo_type="dataset", local_dir=RAW))
  valzip = Path(hf_hub_download("commaai/commavq", "val.zip", repo_type="dataset", revision=VAL_REVISION, local_dir=RAW))
  for f in ("decoder_pytorch_model.bin", "pytorch_model.bin"):  # decoder VQ-VAE y gpt2m de comma
    hf_hub_download("commaai/commavq-gpt2m", f, local_dir=RAW)

  Path("results").mkdir(exist_ok=True)
  train_keys, train, train_poses = extract_shard(shard, with_poses=True)
  val_keys, val = extract_val(valzip)
  np.save("data/train_tokens.npy", train)
  save_poses(train_keys, train_poses, train)
  np.save("data/val_tokens.npy", val)

  overlap = sorted({route_of(k) for k in train_keys} & {route_of(k) for k in val_keys})
  json.dump({"source": f"commaai/commavq@{VAL_REVISION}/val.zip", "segments": val_keys}, open("results/val_split.json", "w"), indent=1)
  print(f"train {train.shape} ({len(set(map(route_of, train_keys)))} rutas), val {val.shape}")
  print(f"rutas compartidas train/val: {len(overlap)}")
  print("ejemplo de keys:", train_keys[:3], val_keys[:3])
