"""Notas del orador y archivo único de la presentación.

  python presentacion/armar.py
    -> presentacion/notas.js           las notas de cada slide, desde guion.md (en la presentación, la tecla N)
    -> presentacion/presentacion.html  todo en un solo archivo para compartir: datos, video, notas e imágenes embebidas.
                                       La hoja de arquitecturas (arquitecturas.html) va al lado, porque se enlaza.

Correr después de generar_datos.py y generar_video.py. index.html es la fuente: se edita ese, no presentacion.html.
La idea del archivo único es de Tomás (paper/presentacion/armar.py, en la rama paper).
"""
import base64
import html
import json
import re
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
CABEZA = re.compile(r"^## (\d+) · (.+?) \(([^,]+), (\d+) s\)\s*$", re.M)


def notas():
  """guion.md -> notas.js: una entrada por slide principal, con quién la dice, los segundos y el texto en HTML."""
  t = (AQUI / "guion.md").read_text(encoding="utf-8").split("\n---\n")[1]   # entre la tabla y el respaldo
  heads = list(CABEZA.finditer(t))
  out = []
  for k, m in enumerate(heads):
    cuerpo = t[m.end():heads[k + 1].start() if k + 1 < len(heads) else len(t)]
    ps = []
    for par in (p.strip() for p in cuerpo.strip().split("\n\n") if p.strip()):
      if par.startswith("**") and par.endswith("**"):
        ps.append(f"<p><b>{html.escape(par.strip('*'))}</b></p>")
      elif par.startswith("*") and par.endswith("*"):
        ps.append(f"<p><i>{html.escape(par.strip('*'))}</i></p>")
      else:
        ps.append(f"<p>{html.escape(par)}</p>")
    n, titulo, quien, seg = m.groups()
    out.append({"slide": int(n), "titulo": titulo, "quien": quien, "seg": int(seg), "html": "".join(ps)})
  assert [x["slide"] for x in out] == list(range(len(out))), "guion.md: faltan slides o están desordenadas"
  (AQUI / "notas.js").write_text("// Generado por presentacion/armar.py desde guion.md. No editar a mano.\nwindow.NOTAS = "
                                 + json.dumps(out, ensure_ascii=False) + ";\n", encoding="utf-8")
  total = sum(x["seg"] for x in out)
  print(f"notas.js: {len(out)} slides, {total // 60}:{total % 60:02d}")


def uri(p):
  mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}[p.suffix.lower()]
  return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def unico():
  h = (AQUI / "index.html").read_text(encoding="utf-8")
  imgs = {p.name: uri(p) for p in sorted((AQUI / "img").glob("*.jpg"))}
  for js in ("datos.js", "video.js", "notas.js"):
    tag = f'<script src="{js}"></script>'
    assert h.count(tag) == 1, tag
    pre = f"<script>window.IMG = {json.dumps(imgs)};</script>\n" if js == "datos.js" else ""
    h = h.replace(tag, pre + "<script>\n" + (AQUI / js).read_text(encoding="utf-8") + "</script>")
  for n, u in imgs.items():
    h = h.replace(f'src="img/{n}"', f'src="{u}"')
  for f in set(re.findall(r'src="\.\./paper/figs/([^"]+)"', h)):
    h = h.replace(f'src="../paper/figs/{f}"', f'src="{uri(RAIZ / "paper" / "figs" / f)}"')
  assert 'src="img/' not in h and "../paper/figs" not in h, "quedó una imagen sin embeber"
  (AQUI / "presentacion.html").write_text(h, encoding="utf-8")
  print(f"presentacion.html: {len(h.encode()) / 1e6:.1f} MB")


if __name__ == "__main__":
  notas()
  unico()
