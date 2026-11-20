"""Figura de frames generados para el paper: filas = modelos, columnas = último frame de contexto y +0,5/1/2/3 s.

  python paper/hacer_frames.py [--clips 0 3]       # -> paper/figs/frames_clip<NNN>.pdf/.png

Lee results/rollouts/<nombre>_ej.npy (40 clips, u 8 en gpt2m; ver el final de paper/hacer_figuras.py) y decodifica
los tokens con el decoder del VQ-VAE (src.ejemplos). El ConvLSTM ya genera píxeles y no se decodifica.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common import FPS
from src.ejemplos import contexto, decodificar

ROLL = Path("results/rollouts")
OUT = Path("paper/figs")
HORIZONTES = [10, 20, 40, 60]   # frames generados: 0,5 / 1 / 2 / 3 s
FILAS = [("real", "Real"), ("gpt2m_ej", "gpt2m (307M)"), ("gpt_s_kd_ej", "GPT-S destilado"),
         ("maskgit_s_abs_p8_ej", "MaskGIT-S, 8 pasos"), ("genie_s_p4_ej", "Genie-S, 4 pasos"),
         ("convlstm_px_ej", "ConvLSTM (píxeles)"), ("congelado_ej", "Frame congelado")]


def imagenes(c):
  ctx, real = contexto(c)
  toks, pix = {}, {}
  for n, _ in FILAS:
    if n == "real":
      toks[n] = real
      continue
    p = ROLL / f"{n}.npy"
    if not p.exists():
      print("falta", p)
      continue
    a = np.load(p, mmap_mode="r")
    if c >= len(a):
      print(f"{n}: no tiene el clip {c}")
      continue
    (pix if a.ndim == 5 else toks)[n] = np.asarray(a[c])
  img = decodificar(np.concatenate([ctx] + list(toks.values())))
  out = dict(zip(toks, img[len(ctx):].reshape(len(toks), -1, *img.shape[1:])))
  out.update(pix)
  return img[len(ctx) - 1], out


def figura(c):
  ult_ctx, imgs = imagenes(c)
  filas = [(n, et) for n, et in FILAS if n in imgs]
  m = len(HORIZONTES) + 1
  fig, axs = plt.subplots(len(filas), m, figsize=(7.0, 0.72 * len(filas) + 0.25), squeeze=False,
                          gridspec_kw={"wspace": 0.03, "hspace": 0.06})
  for i, (n, et) in enumerate(filas):
    for j in range(m):
      ax = axs[i, j]
      ax.set_xticks([]), ax.set_yticks([])
      for sp in ax.spines.values():
        sp.set_visible(False)
      if j == 0:
        if i == 0:
          ax.imshow(ult_ctx)
        ax.text(-0.05, 0.5, et, transform=ax.transAxes, ha="right", va="center", fontsize=7)
      else:
        ax.imshow(np.clip(imgs[n][HORIZONTES[j - 1] - 1], 0, 255).astype(np.uint8))
      if i == 0:
        t = "contexto (t = 0)" if j == 0 else f"+{HORIZONTES[j - 1] / FPS:g} s".replace(".", ",")
        ax.set_title(t, fontsize=7.5)
  OUT.mkdir(parents=True, exist_ok=True)
  for ext in ("pdf", "png"):
    fig.savefig(OUT / f"frames_clip{c:03d}.{ext}", dpi=260, bbox_inches="tight", pad_inches=0.02)
  plt.close(fig)
  print("figura", OUT / f"frames_clip{c:03d}.pdf")


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--clips", type=int, nargs="*", default=[0, 3])
  for c in ap.parse_args().clips:
    figura(c)
