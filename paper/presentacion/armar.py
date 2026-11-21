"""Arma las presentaciones HTML autocontenidas a partir de las fuentes de esta carpeta.

  python paper/presentacion/armar.py      # -> paper/arquitecturas.html y paper/presentacion.html

Inserta comun.js (dibujos SVG compartidos), frames.json (frames de ejemplo; ver exportar_frames.py) y, en la
presentación, las figuras de paper/figs como data URIs (las de curvas en PNG; la de frames, en JPEG para que pese menos).
"""
import base64
import io
import json
from pathlib import Path

from PIL import Image

AQUI = Path(__file__).resolve().parent
PAPER = AQUI.parent
FIGS = {"drift": "drift.png", "pasos": "pasos.png", "latencia": "calidad_latencia.png",
        "escala": "datos_escala.png", "frames": "frames_clip000.png"}


def uri(nombre):
  p = PAPER / "figs" / nombre
  if nombre.startswith("frames_"):
    buf = io.BytesIO()
    Image.open(p).convert("RGB").save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
  return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def armar(fuente, salida, figs=None):
  html = (AQUI / fuente).read_text(encoding="utf-8")
  html = html.replace("/*__COMUN__*/", (AQUI / "comun.js").read_text(encoding="utf-8"))
  html = html.replace("__DATA__", (AQUI / "frames.json").read_text(encoding="utf-8"))
  if figs is not None:
    html = html.replace("__FIGS__", json.dumps(figs))
  (PAPER / salida).write_text(html, encoding="utf-8")
  print(f"{salida}: {len(html) / 1e6:.2f} MB")


if __name__ == "__main__":
  armar("arquitecturas.src.html", "arquitecturas.html")
  armar("presentacion.src.html", "presentacion.html", {k: uri(v) for k, v in FIGS.items()})
