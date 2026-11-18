"""Descarga de archivos de Hugging Face con pedidos por rangos en paralelo y chequeo de sha256.

En algunas redes una sola conexión al CDN de HF va a ~0,4 MB/s (y hf_hub_download se cuelga), pero el
ancho de banda escala con la cantidad de conexiones. El sha256 sale del encabezado X-Linked-ETag
(archivos LFS) y se verifica al terminar.

  python -m src.hf_fetch commaai/commavq-gpt2m encoder_pytorch_model.bin data/raw
"""
import hashlib
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

UA = {"User-Agent": "ml2-gen-world-model"}


def _head(url):
  """(tamaño, sha256 o None) del archivo, sin seguir la redirección al CDN."""
  class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
      return None
  op = urllib.request.build_opener(NoRedirect)
  req = urllib.request.Request(url, method="HEAD", headers=UA)
  try:
    r = op.open(req, timeout=30)
    h = r.headers
  except urllib.error.HTTPError as e:  # 302 al CDN: los metadatos LFS vienen en este encabezado
    h = e.headers
  size = h.get("X-Linked-Size") or h.get("Content-Length")
  etag = (h.get("X-Linked-ETag") or "").strip('"')
  return int(size), (etag if len(etag) == 64 else None)


def sha256(path, chunk=1 << 22):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    while b := f.read(chunk):
      h.update(b)
  return h.hexdigest()


def fetch(repo: str, filename: str, dest_dir="data/raw", revision="main", workers: int = 16,
          part_mb: int = 8, repo_type: str = "model") -> Path:
  """Baja repo/filename a dest_dir/filename si no está (o si el sha256 no coincide). Devuelve la ruta."""
  base = "https://huggingface.co" + ("/datasets" if repo_type == "dataset" else "")
  url = f"{base}/{repo}/resolve/{revision}/{filename}"
  dest = Path(dest_dir) / filename
  size, digest = _head(url)
  if dest.exists() and dest.stat().st_size == size and (digest is None or sha256(dest) == digest):
    return dest
  dest.parent.mkdir(parents=True, exist_ok=True)
  tmp = dest.with_suffix(dest.suffix + ".part")
  with open(tmp, "wb") as f:
    f.truncate(size)
  step = part_mb << 20
  ranges = [(a, min(a + step, size) - 1) for a in range(0, size, step)]

  def get(rg):
    a, b = rg
    for _ in range(5):
      try:
        req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={a}-{b}"})
        data = urllib.request.urlopen(req, timeout=60).read()
        if len(data) == b - a + 1:
          with open(tmp, "r+b") as f:
            f.seek(a)
            f.write(data)
          return
      except OSError:
        pass
    raise RuntimeError(f"no se pudo bajar {filename} bytes {a}-{b}")

  with ThreadPoolExecutor(workers) as ex:
    list(ex.map(get, ranges))
  if digest is not None and sha256(tmp) != digest:
    tmp.unlink()
    raise RuntimeError(f"sha256 de {filename} no coincide")
  tmp.replace(dest)
  return dest


if __name__ == "__main__":
  repo, fn, *rest = sys.argv[1:]
  print(fetch(repo, fn, *(rest or ["data/raw"])))
