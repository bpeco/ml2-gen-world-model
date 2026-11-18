"""Carpetas que usan la CLI, la cola y la bitácora. Los tests las cambian a un directorio temporal."""
from pathlib import Path

RESULTS = Path("results")
CKPT = Path("checkpoints")


def results() -> Path:
  return Path(RESULTS)


def ckpt() -> Path:
  return Path(CKPT)


def bitacora() -> Path:
  return results() / "bitacora.jsonl"


def candado() -> Path:
  return results() / ".gpu.lock"


def cola() -> Path:
  return results() / "queue.json"
