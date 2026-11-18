"""Ejemplos visuales de los rollouts: decodifica clips de los 400 fijos y arma, para cada uno, una grilla
(modelo x horizonte) y un GIF con todos los modelos lado a lado. También se usa desde
notebooks/resultados.ipynb.

Los clips por defecto no se eligen por cómo le va a un modelo: se ordenan por cuánto cambia la escena
real en 3 s (LPIPS del frame congelado a 3 s) y se toman los percentiles 50, 75 y 90 (escena típica, con
más movimiento y con mucho movimiento). Necesita results/rollouts/*.npy, results/cache/lpips_frozen.npy,
data/ y third_party/commavq.

  python -m src.ejemplos [--clips 12 57 ...] [--modelos real gpt2m maskgit_s_s4 ...] [--out results/ejemplos]
"""
import argparse
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.common import CTX_FRAMES, FPS, device
from src.rollouts import VAL, clip_tokens, load_clips

R = Path("results")
ROLL = R / "rollouts"
# rollout -> (etiqueta en results/summary.json, nombre corto)
ETIQUETAS = {
  "real": (None, "Real"),
  "frozen": ("Frame congelado", "Frame congelado"),
  "unigram": ("Unigrama", "Unigrama"),
  "copy": ("Transición por celda", "Transición (baseline)"),  # rollouts/copy.npy: nombre viejo
  "gpt2m": ("gpt2m (comma)", "gpt2m (comma)"),
  "gpt_s": ("GPT-S", "GPT-S"),
  "gpt_s_kd": ("GPT-S destilado", "GPT-S destilado"),
  "gpt_m": ("GPT-M", "GPT-M"),
  "gpt_m_kd": ("GPT-M destilado", "GPT-M destilado"),
  **{f"maskgit_{z}_s{s}": (f"MaskGIT-{z.upper()} {s} pasos", f"MaskGIT-{z.upper()}, {s} paso{'s' * (s > 1)}")
     for z in "sm" for s in (1, 2, 4, 8)},
}
MODELOS = ["real", "gpt2m", "gpt_s", "maskgit_s_s4", "copy"]
HORIZONTES = [10, 20, 40, 60]  # frames generados: 0,5 / 1 / 2 / 3 s
PERCENTILES = [50, 75, 90]
_decoder = None


def disponibles():
  """Nombres de los rollouts guardados en results/rollouts/."""
  return sorted(p.stem for p in ROLL.glob("*.npy"))


def etiqueta(nombre):
  """(nombre corto, 'parámetros · ms por frame') de un rollout, con los datos de results/summary.json."""
  label, corto = ETIQUETAS.get(nombre, (None, nombre))
  try:
    s = {c["label"]: c for c in json.loads((R / "summary.json").read_text(encoding="utf-8"))["configs"]}
  except FileNotFoundError:
    s = {}
  c = s.get(label) or {}
  extra = []
  if c.get("params_M"):
    extra.append(f"{c['params_M']:.0f}M")
  if c.get("ms_frame") is not None:
    ms = c["ms_frame"]
    extra.append((f"{ms:.0f}" if ms >= 10 else f"{ms:.1f}".replace(".", ",")) + " ms/frame")
  return corto, " · ".join(extra)


def cambio_escena():
  """LPIPS entre el último frame de contexto y el frame real a 3 s, por clip (400,)."""
  return np.load(R / "cache/lpips_frozen.npy")[:, -1]


def elegir_clips(percentiles=PERCENTILES):
  """Clips en esos percentiles de cambio de la escena real a 3 s."""
  cambio = cambio_escena()
  orden = np.argsort(cambio)
  return [int(orden[int(round(p / 100 * (len(orden) - 1)))]) for p in percentiles]


def decoder():
  global _decoder
  if _decoder is None:
    from src.viz import load_decoder
    _decoder = load_decoder(device())
  return _decoder


def decodificar(tokens):
  """tokens (..., 128) en [0, 1024) -> imágenes uint8 (..., 128, 256, 3)."""
  from src.viz import decode
  t = np.asarray(tokens, dtype=np.int64)
  assert t.min() >= 0 and t.max() < 1024, "hay tokens fuera del vocabulario del decoder"
  img = decode(decoder(), t.reshape(-1, 128))
  return img.reshape(*t.shape[:-1], *img.shape[1:])


def contexto(clip):
  """Tokens del contexto real (20, 128) y de los 3 s reales (60, 128) de un clip."""
  clips = load_clips()
  val = np.load(VAL, mmap_mode="r")
  ctx, real = clip_tokens(val, {**clips, "clips": [clips["clips"][clip]]})
  return ctx[0], real[0]


def clip(clip, modelos=MODELOS, generados=None):
  """Imágenes de un clip: contexto (20, H, W, 3) y {modelo: (60, H, W, 3)}.

  generados: {nombre: tokens (60, 128)} extra, por ejemplo un rollout recién hecho en el notebook."""
  ctx, real = contexto(clip)
  toks = {}
  for m in modelos:
    toks[m] = real if m == "real" else np.load(ROLL / f"{m}.npy", mmap_mode="r")[clip]
  toks.update(generados or {})
  img = decodificar(np.concatenate([ctx] + list(toks.values())))
  return img[:len(ctx)], dict(zip(toks, img[len(ctx):].reshape(len(toks), -1, *img.shape[1:])))


def grilla(ctx_img, imgs, titulo="", path=None):
  """Filas = modelos, columnas = último frame de contexto (solo en la primera fila) + horizontes."""
  n, m = len(imgs), len(HORIZONTES) + 1
  fig, axs = plt.subplots(n, m, figsize=(2.56 * m, 1.28 * n + 0.6), facecolor="white", squeeze=False,
                          gridspec_kw={"wspace": 0.03, "hspace": 0.12})
  for i, (nombre, img) in enumerate(imgs.items()):
    corto, extra = etiqueta(nombre)
    for j in range(m):
      ax = axs[i, j]
      ax.set_xticks([]), ax.set_yticks([])
      for sp in ax.spines.values():
        sp.set_visible(False)
      if j == 0:
        if i == 0:
          ax.imshow(ctx_img[-1])
        ax.text(-0.04, 0.5, corto + (f"\n{extra}" if extra else ""), transform=ax.transAxes, ha="right",
                va="center", fontsize=8.5, color="#1f1f1f")
      elif HORIZONTES[j - 1] <= len(img):
        ax.imshow(img[HORIZONTES[j - 1] - 1])
      if i == 0:
        t = "contexto (t = 0)" if j == 0 else f"+{HORIZONTES[j - 1] / FPS:g} s".replace(".", ",")
        ax.set_title(t, fontsize=9, color="#1f1f1f")
  if titulo:
    # arriba de los rótulos de columna, sin importar cuántas filas haya (bbox_inches="tight" lo incluye)
    fig.suptitle(titulo, fontsize=10, color="#1f1f1f", y=1.0, va="bottom", linespacing=1.5)
  if path:
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor="white")
  return fig


def _fuente(size):
  for f in ("arial.ttf", "DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc"):
    try:
      return ImageFont.truetype(f, size)
    except OSError:
      pass
  return ImageFont.load_default()


def gif(ctx_imgs, imgs, path=None, cols=4, fps=10):
  """GIF con todos los modelos lado a lado: el contexto real y después lo generado, a velocidad real
  (fps=10 toma 1 de cada 2 frames). Escribe en path o, sin path, devuelve los bytes."""
  h, w = ctx_imgs.shape[1:3]
  top, lab = 26, 34
  cols = min(cols, len(imgs))
  rows = -(-len(imgs) // cols)
  f_t, f_l, f_s = _fuente(15), _fuente(13), _fuente(11)
  labels = [etiqueta(n) for n in imgs]
  n_gen = min(len(v) for v in imgs.values())
  frames = []
  for k in range(0, len(ctx_imgs) + n_gen, max(1, FPS // fps)):
    gen = k >= len(ctx_imgs)
    canvas = Image.new("RGB", (cols * w, top + rows * (h + lab)), "white")
    d = ImageDraw.Draw(canvas)
    t = (k - len(ctx_imgs) + 1) / FPS
    estado = f"generado  +{t:.1f} s" if gen else f"contexto real  {t:.1f} s"
    d.text((8, 5), estado.replace(".", ","), fill="#c0392b" if gen else "#1f1f1f", font=f_t)
    for i, (img, (corto, extra)) in enumerate(zip(imgs.values(), labels)):
      x, y = (i % cols) * w, top + (i // cols) * (h + lab)
      canvas.paste(Image.fromarray(img[k - len(ctx_imgs)] if gen else ctx_imgs[k]), (x, y + lab))
      d.text((x + 4, y + 2), corto, fill="#1f1f1f", font=f_l)
      if extra:
        d.text((x + 4, y + 18), extra, fill="#5f5f5f", font=f_s)
    frames.append(canvas)
  frames += [frames[-1]] * fps  # pausa de 1 s al final
  pal = [f.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
  out = path or io.BytesIO()
  pal[0].save(out, format="GIF", save_all=True, append_images=pal[1:], duration=1000 // fps, loop=0,
              optimize=True)
  return None if path else out.getvalue()


def titulo_clip(c):
  x = load_clips()["clips"][c]
  cambio = cambio_escena()
  p = int(round(100 * (cambio < cambio[c]).mean()))
  return (f"Clip {c} (segmento {x['segment']}, frame {x['start']}): {CTX_FRAMES / FPS:g} s de contexto real y "
          f"3 s generados a temperatura 1\nCambio de la escena real a 3 s: percentil {p} de los 400 clips"), p


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--clips", type=int, nargs="*")
  ap.add_argument("--modelos", nargs="*", default=MODELOS)
  ap.add_argument("--out", default="results/ejemplos")
  ap.add_argument("--gif-fps", type=int, default=10)
  a = ap.parse_args()
  plt.switch_backend("Agg")
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  info = []
  for c in a.clips or elegir_clips():
    ctx_img, imgs = clip(c, a.modelos)
    titulo, p = titulo_clip(c)
    plt.close(grilla(ctx_img, imgs, titulo, out / f"clip{c:03d}.png"))
    gif(ctx_img, imgs, out / f"clip{c:03d}.gif", fps=a.gif_fps)
    x = load_clips()["clips"][c]
    info.append({"clip": c, "segment": x["segment"], "start": x["start"], "percentil_cambio": p,
                 "png": f"clip{c:03d}.png", "gif": f"clip{c:03d}.gif"})
    print(f"clip {c}: percentil {p} -> {out / f'clip{c:03d}'}.png, .gif", flush=True)
  (out / "ejemplos.json").write_text(json.dumps({"modelos": a.modelos, "clips": info}, indent=1, ensure_ascii=False),
                                    encoding="utf-8")


if __name__ == "__main__":
  main()
