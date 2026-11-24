"""Figuras de movimiento a partir de results/flow/<name>.json (src.metrics_flow) y la latencia del paper.

  python experiments/motion/figuras.py                     figuras de la Fase 1 en experiments/motion/figs/
  python experiments/motion/figuras.py --names a b --tag X  otras corridas (mismo set --clips motion)

  fig_<tag>_tok.png    tasa de cambio de tokens por frame de rollout, rectas y curvas, con el real y el piso del codec
  fig_<tag>_flow.png   |flow| medio por frame, ídem
  fig_<tag>_giro.png   flow horizontal en curvas alineado con el sentido del giro real (lo que tiene que seguir al yaw)
  fig_<tag>_vs_ms.png  giro sostenido (flow horizontal alineado a 2-3 s, relativo al real) contra ms/frame, línea de 50 ms
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FLOW = Path("results/flow")
FIGS = Path("experiments/motion/figs")
# nombre de corrida -> (etiqueta, label de latencia en results/summary.json)
FASE1 = {"gpt2m_motion": ("GPT2M (maestro)", "gpt2m"), "gpt_s_kd_motion": ("GPT-S kd", "gpt_s_kd"),
         "gpt_s_d050_motion": ("GPT-S d050", "gpt_s_kd"),          # misma arquitectura que kd: misma latencia
         "genie_s_p4_motion": ("Genie-S p4", "genie_s_p4"), "mg_p1_motion": ("MaskGIT p1", "maskgit_s_abs_p1"),
         "mg_p2_motion": ("MaskGIT p2", "maskgit_s_abs_p2"), "mg_p4_motion": ("MaskGIT p4", "maskgit_s_abs_p4"),
         "mg_p8_motion": ("MaskGIT p8", "maskgit_s_abs_p8")}
# Okabe-Ito (seguro para daltonismo), orden fijo
COLORES = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#999999"]
SUAVE = 5


def suavizar(y, k=SUAVE):
  y = np.asarray(y, float)
  pad = np.pad(y, (k // 2, k - 1 - k // 2), mode="edge")
  return np.convolve(pad, np.ones(k) / k, mode="valid")


CACHE = Path("results/cache")


def giro_alineado(name):
  """(n_curvas, 60) flow horizontal con signo, alineado con el sentido del giro real de cada curva (+ = gira como el
  real). En el promedio con signo crudo las curvas a izquierda y derecha se cancelan; el |flow| mezcla giro y parpadeo."""
  clips = json.loads(Path("results/clips_motion.json").read_text())["clips"]
  cur = np.array([c["tipo"] == "curva" for c in clips])
  h = json.loads((FLOW / "piso_codec.json").read_text())["clips_hash"]
  real = np.load(CACHE / f"flow_real_{h}.npz")["rot"][cur]
  s = np.sign(real.sum(1, keepdims=True))
  mod = real if name == "real" else np.load(CACHE / f"flow_{name}_{h}.npz")["rot"][cur]
  return mod * s


def latencias():
  s = json.loads(Path("results/summary.json").read_text())
  return {c["label"]: c["ms_frame"] for c in s["configs"]}


def por_horizonte(runs, campo, archivo, titulo, ylabel, grupos=("recta", "curva"), piso=None):
  fig, axs = plt.subplots(1, len(grupos), figsize=(5.2 * len(grupos), 3.8), sharey=True, squeeze=False)
  t = np.arange(1, 61) / 20
  for ax, g in zip(axs[0], grupos):
    real = None
    for i, (name, (lab, _)) in enumerate(runs.items()):
      r = json.loads((FLOW / f"{name}.json").read_text())["resumen"][g][campo]
      real = r["curva_real"]
      ax.plot(t, suavizar(r["curva_modelo"]), color=COLORES[i % len(COLORES)], lw=1.6, label=lab)
    ax.plot(t, suavizar(real), color="black", lw=2.4, label="real")
    if piso is not None:
      ax.axhline(piso, color="#666666", ls="--", lw=1.2, label="piso del codec")
    ax.set_title(f"{g}s (n=100)")
    ax.set_xlabel("horizonte del rollout (s)")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
  axs[0][0].set_ylabel(ylabel)
  axs[0][-1].legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
  fig.suptitle(titulo, fontsize=11)
  fig.tight_layout()
  fig.savefig(archivo, dpi=150, bbox_inches="tight")
  plt.close(fig)


def giro(runs, archivo):
  fig, ax = plt.subplots(figsize=(6.6, 3.9))
  t = np.arange(1, 61) / 20
  for i, (name, (lab, _)) in enumerate(runs.items()):
    ax.plot(t, suavizar(giro_alineado(name).mean(0)), color=COLORES[i % len(COLORES)], lw=1.6, label=lab)
  ax.plot(t, suavizar(giro_alineado("real").mean(0)), color="black", lw=2.4, label="real")
  ax.axhline(0, color="#666666", lw=0.8)
  ax.set_xlabel("horizonte del rollout (s)")
  ax.set_ylabel("flow horizontal en el sentido del giro real (px)")
  ax.set_title("Curvas (n=100): ¿el modelo gira para el mismo lado que el auto?", fontsize=10)
  ax.grid(alpha=0.25)
  ax.spines[["top", "right"]].set_visible(False)
  ax.legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
  fig.tight_layout()
  fig.savefig(archivo, dpi=150, bbox_inches="tight")
  plt.close(fig)


def giro_sostenido(name):
  """giro alineado medio a 2-3 s, relativo al real (1 = gira tanto como el real, 0 = no gira o gira al azar)."""
  return float(giro_alineado(name)[:, 40:].mean() / giro_alineado("real")[:, 40:].mean())


def vs_ms(runs, archivo):
  lat = latencias()
  fig, ax = plt.subplots(figsize=(7.6, 4.2))
  for i, (name, (lab, lab_lat)) in enumerate(runs.items()):
    ms = lat.get(lab_lat)
    if ms is None:
      continue
    r = giro_sostenido(name)
    ax.scatter(ms, r, s=60, color=COLORES[i % len(COLORES)], edgecolor="white", linewidth=1.5, zorder=3,
               label=f"{lab} ({ms:.0f} ms)")
  ax.axvline(50, color="#D55E00", ls="--", lw=1.2)
  ax.text(52, 1.02, "50 ms (tiempo real)", color="#D55E00", fontsize=8, va="top")
  ax.axhline(1, color="black", lw=0.8, alpha=0.5)
  ax.set_xscale("log")
  ax.set_ylim(-0.15, 1.1)
  ax.set_xlabel("ms por frame (B=1, GPU del paper, escala log)")
  ax.set_ylabel("giro sostenido: flow horiz. alineado, 2-3 s / real")
  ax.set_title("Giro sostenido contra latencia", fontsize=11)
  ax.legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
  ax.grid(alpha=0.25)
  ax.spines[["top", "right"]].set_visible(False)
  fig.tight_layout()
  fig.savefig(archivo, dpi=150, bbox_inches="tight")
  plt.close(fig)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--names", nargs="*")
  ap.add_argument("--tag", default="fase1")
  a = ap.parse_args()
  runs = FASE1 if not a.names else {n: FASE1.get(n, (n, None)) for n in a.names}
  FIGS.mkdir(parents=True, exist_ok=True)
  piso = json.loads((FLOW / "piso_codec.json").read_text())["resumen"]
  por_horizonte(runs, "tok", FIGS / f"fig_{a.tag}_tok.png", "Tasa de cambio de tokens entre frames consecutivos",
                "fracción de tokens que cambian", piso=piso["tok"]["media"])
  por_horizonte(runs, "mag", FIGS / f"fig_{a.tag}_flow.png", "|flow| medio entre frames consecutivos (RAFT)",
                "|flow| (px)", piso=piso["mag"]["media"])
  giro(runs, FIGS / f"fig_{a.tag}_giro.png")
  vs_ms(runs, FIGS / f"fig_{a.tag}_vs_ms.png")
  print("figuras en", FIGS)


if __name__ == "__main__":
  main()

