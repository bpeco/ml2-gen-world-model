"""Descarga y extracción de commaVQ a arrays de numpy.

Salida:
  data/train_tokens.npy  (N, 1200, 128) int16, desde un shard webdataset
  data/val_tokens.npy    (100, 1200, 128) int16, val oficial viejo (val.zip)
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


def extract_shard(tar_path: Path):
  keys, tokens = [], []
  with tarfile.open(tar_path, "r:gz") as tar:
    for m in tar:
      if m.isfile() and m.name.endswith("token.npy"):
        t = load_npy(tar.extractfile(m).read())
        if t.shape != (1200, 8, 16):
          print("salteo", m.name, t.shape)
          continue
        keys.append(m.name[: -len(".token.npy")])
        tokens.append(t.reshape(1200, 128).astype(np.int16))
  return keys, np.stack(tokens)


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
  args = ap.parse_args()

  shard = Path(hf_hub_download("commaai/commavq", args.shard, repo_type="dataset", local_dir=RAW))
  valzip = Path(hf_hub_download("commaai/commavq", "val.zip", repo_type="dataset", revision=VAL_REVISION, local_dir=RAW))
  for f in ("decoder_pytorch_model.bin", "pytorch_model.bin"):  # decoder VQ-VAE y gpt2m de comma
    hf_hub_download("commaai/commavq-gpt2m", f, local_dir=RAW)

  train_keys, train = extract_shard(shard)
  val_keys, val = extract_val(valzip)
  np.save("data/train_tokens.npy", train)
  np.save("data/val_tokens.npy", val)

  overlap = sorted({route_of(k) for k in train_keys} & {route_of(k) for k in val_keys})
  json.dump({"source": f"commaai/commavq@{VAL_REVISION}/val.zip", "segments": val_keys}, open("results/val_split.json", "w"), indent=1)
  print(f"train {train.shape} ({len(set(map(route_of, train_keys)))} rutas), val {val.shape}")
  print(f"rutas compartidas train/val: {len(overlap)}")
  print("ejemplo de keys:", train_keys[:3], val_keys[:3])
