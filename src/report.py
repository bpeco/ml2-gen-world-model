"""Figuras y tabla resumen a partir de results/configs.json (lo escribe el lead al final).

  python -m src.report [--manifest results/configs.json] [--out results]
Cada entrada: label, family, params, eval, video, bench, autoregressive, horizon (opcional).
Los JSON que falten se saltan.

Cada configuración lleva el estado de sus resultados (results/estado.json, src/resultados.py): las tablas lo
muestran en la columna "Estado" y las figuras grafican solo las vigentes. Sin ninguna vigente, cada figura es un
cartel que lo dice.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from src import resultados

FPS = 20
RT_MS = 1000 / FPS
# Paleta categórica validada (dataviz, --pairs all, 3 tonos): GPT azul, MaskGIT naranja.
# gpt2m y los baselines van en neutros. La forma del marcador es el segundo canal de identidad y los
# destilados (y los extras) llevan marcador hueco.
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
SURF = "#fcfcfb"
# Los latentes (lewm) van en su propia figura (fig_pareto_latente*.png), con el violeta de la ablación y las
# referencias de tokens en gris. No aparecen en las figuras de las otras familias.
# Genie es la dinámica MaskGIT fiel: lleva el naranja de MaskGIT. El ConvLSTM de píxeles lleva el violeta (el tercer
# tono validado); el MaskGIT propio queda como ablación opcional.
COLORS = {"gpt": "#2a78d6", "gpt_kd": "#2a78d6", "maskgit": "#eb6834", "genie": "#eb6834", "convlstm": "#6250d6",
          "gpt2m": INK, "baseline": MUTED, "ablación": "#6250d6", "contexto": "#eb6834", "lewm": "#6250d6"}
MARKERS = {"gpt": "o", "gpt_kd": "o", "maskgit": "^", "genie": "^", "convlstm": "s", "gpt2m": "D", "baseline": "X",
           "ablación": "v", "contexto": "^", "lewm": "h"}
FAMILY_NAMES = {"baseline": "baselines de conteo", "gpt": "GPT de cero",
                "gpt_kd": "GPT-S destilado (hueco)", "maskgit": "MaskGIT: S lleno, M hueco (número = pasos)",
                "genie": "Genie S (número = pasadas)", "convlstm": "ConvLSTM de píxeles", "gpt2m": "gpt2m de comma (307M)",
                "ablación": "ablación", "contexto": "contexto",
                "lewm": "LeWorldModel (un vector por frame)"}
LATENT_FAMS = ("lewm",)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def color(fam):
  return COLORS.get(fam, MUTED)


def _load(p, base):
  """Lee un JSON; las rutas relativas se buscan desde el cwd y desde la carpeta del manifiesto."""
  if not p:
    return None
  p = Path(p)
  for q in (p, base / p):
    if q.exists():
      return json.loads(q.read_text())
  return None


def _get(d, *keys):
  for k in keys:
    try:
      d = d[k]
    except (KeyError, IndexError, TypeError):
      return None
  return d


def estado_config(m, registro, base=Path(".")):
  """Estado de una configuración: el peor entre los archivos que tiene (pendiente < protocolo anterior < vigente)."""
  def ruta(p):  # igual que _load: desde el cwd o desde la carpeta del manifiesto
    return Path(p) if Path(p).exists() else base / p
  es = [resultados.estado(ruta(m[k]), registro) for k in ("eval", "video", "bench", "e2e") if m.get(k)]
  for e in ("pendiente", "protocolo anterior"):   # un resultado que existe y no está marcado es vigente
    if e in es:
      return e
  return "vigente" if es else resultados.SIN_ESTADO


def load_configs(manifest, registro=None):
  """Une el manifiesto con sus eval/video/bench en una lista de dicts planos.
  registro: dict de results/estado.json (por defecto el del repo)."""
  manifest = Path(manifest)
  registro = resultados.cargar() if registro is None else registro
  cfgs = []
  pf = _load("piso_fvd.json", manifest.parent)  # piso n contra n (E1)
  for m in json.loads(manifest.read_text(encoding="utf-8")):
    ev, vd, bn, e2e = (_load(m.get(k), manifest.parent) for k in ("eval", "video", "bench", "e2e"))
    c = dict(label=m["label"], estado=estado_config(m, registro, manifest.parent), short=m.get("short", m["label"]), family=m.get("family", "otro"),
             group=m.get("group", "principal"), size=m.get("size"), steps=m.get("steps"), kd=m.get("kd", False),
             params=m.get("params") or _get(bn, "params"),
             autoregressive=bool(m.get("autoregressive", False)), horizon=bool(m.get("horizon", True)),
             ce_all=_get(ev, "ce_all"), ce_last=_get(ev, "ce_last"),
             lpips_curve=_get(vd, "lpips", "curve"),
             lpips1=_get(vd, "lpips", "at", "1", "mean"),
             lpips1_ci=(_get(vd, "lpips", "at", "1", "lo"), _get(vd, "lpips", "at", "1", "hi")),
             fvd={h: _get(vd, "fvd", h) for h in ("1", "2", "3")},
             piso1=_get(vd, "piso", "1", "value"), piso_n1=_get(pf, "piso", "1", "value"),
             piso_n1_ci=(_get(pf, "piso", "1", "lo"), _get(pf, "piso", "1", "hi")), fvd1_mitades=_get(vd, "piso", "1", "modelo_mitades"),
             ms=_get(bn, "median_ms"),
             e2e_p99=_get(e2e, "cadena_ritmo", "p99") or _get(e2e, "steady", "p99"))   # protocolo único (D15), o el e2e viejo
    c["fvd1"] = _get(c["fvd"]["1"], "value")
    cfgs.append(c)
  return cfgs


def pareto_mask(x, y):
  """True para los puntos no dominados (menor es mejor en x e y). None/NaN: nunca óptimos."""
  n = len(x)
  ok = [x[i] is not None and y[i] is not None and np.isfinite(x[i]) and np.isfinite(y[i]) for i in range(n)]
  res = []
  for i in range(n):
    dom = ok[i] and any(ok[j] and x[j] <= x[i] and y[j] <= y[i] and (x[j] < x[i] or y[j] < y[i]) for j in range(n))
    res.append(ok[i] and not dom)
  return res


def _err(c, v, lo, hi):
  return None if lo is None or hi is None else [[max(v - lo, 0)], [max(hi - v, 0)]]


def _mark(ax, x, y, c, yerr=None, ms=7, hollow=False, z=3):
  col = color(c["family"])
  ax.errorbar(x, y, yerr=yerr, fmt=MARKERS.get(c["family"], "o"), ms=ms, color=col, ecolor=col, elinewidth=0.8,
              capsize=2, mfc=SURF if hollow else col, mec=col, mew=1.4, zorder=z)


def _txt(ax, x, y, t, dx=6, dy=4, ha="left", size=8, col=INK2):
  ax.annotate(t, (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha, va="center", fontsize=size, color=col)


def _legend(ax, fams, loc="upper right", extra=(), **kw):
  from matplotlib.lines import Line2D
  h = [Line2D([], [], ls="", marker=MARKERS[f], ms=7, color=color(f), mfc=SURF if f == "gpt_kd" else color(f),
              mew=1.4, label=FAMILY_NAMES[f]) for f in fams]
  ax.legend(handles=h + list(extra), loc=loc, fontsize=8, handletextpad=0.4, borderaxespad=0.3, **kw)


def _repel(ax, items, points=()):
  """Etiquetas directas sin solaparse entre sí ni con los marcadores (`points`, pares x, y en datos). Cada
  etiqueta prueba, en orden, su lado preferido y el opuesto, primero a la altura del punto y después corrida
  ±1, ±2, ... renglones; queda en la primera posición libre. items: dicts con x, y, t, dx, ha, size, col.
  Llamar con los límites de los ejes ya fijos (usa la transformación de datos a puntos)."""
  k = 72 / ax.figure.dpi
  to_pt = lambda x, y: tuple(v * k for v in ax.transData.transform((x, y)))
  boxes = [(px - 5, px + 5, py - 5, py + 5) for px, py in (to_pt(x, y) for x, y in points)]
  (ax0, ay0), (ax1, ay1) = (tuple(v * k for v in p) for p in ax.transAxes.transform([(0, 0), (1, 1)]))
  free = lambda b: (b[0] >= ax0 and b[1] <= ax1 and b[2] >= ay0 and b[3] <= ay1
                    and all(b[1] <= a[0] or b[0] >= a[1] or b[3] <= a[2] or b[2] >= a[3] for a in boxes))
  for it in sorted(items, key=lambda i: -to_pt(i["x"], i["y"])[1]):  # de arriba hacia abajo
    px, py = to_pt(it["x"], it["y"])
    w, hgt = len(it["t"]) * it["size"] * 0.55, it["size"] * 1.3
    sides = [(it["dx"], it["ha"]), (-it["dx"], "left" if it["ha"] == "right" else "right")]
    cands = [(dx, ha, dy) for dy in [0] + [s * n * hgt for n in range(1, 8) for s in (1, -1)] for dx, ha in sides]
    for dx, ha, dy in cands:
      x0 = px + dx - (w if ha == "right" else 0)
      b = (x0, x0 + w, py + dy - hgt / 2, py + dy + hgt / 2)
      if free(b):
        break
    boxes.append(b)
    guide = dict(arrowstyle="-", color=it["col"], lw=0.5, shrinkA=0, shrinkB=4) if dy else None  # guía si se corrió
    ax.annotate(it["t"], (it["x"], it["y"]), textcoords="offset points", xytext=(dx, dy), ha=ha, va="center",
                fontsize=it["size"], color=it["col"], arrowprops=guide)

def fig_pareto(cfgs, key, ylabel, path, ci):
  """Figura central: calidad vs. ms/frame. Pareto sobre el grupo principal (mismo presupuesto de tokens)."""
  from matplotlib.lines import Line2D
  pts = [c for c in cfgs if c["ms"] and c[key] is not None and c["group"] in ("principal", "extra")
         and c["family"] not in ("contexto",) + LATENT_FAMS]
  if not pts:
    return
  main = [c for c in pts if c["group"] == "principal"]
  fig, ax = plt.subplots(figsize=(8, 5.2))
  ax.set_xscale("log")
  ax.axvspan(RT_MS, 5000, color=GRID, alpha=0.45, lw=0, zorder=0)
  ax.axvline(RT_MS, ls="--", c=INK2, lw=0.9, zorder=1)
  ax.text(RT_MS * 1.08, 0.98, f"más lento que tiempo real\n({FPS} FPS = {RT_MS:.0f} ms por frame)",
          transform=ax.get_xaxis_transform(), va="top", fontsize=8, color=INK2)
  opt = pareto_mask([c["ms"] for c in main], [c[key] for c in main])
  front = sorted((c["ms"], c[key]) for c, o in zip(main, opt) if o)
  if front:
    ax.step([p[0] for p in front] + [2000], [p[1] for p in front] + [front[-1][1]], where="post", c=MUTED, lw=1,
            zorder=1)
  CON_PASADAS = ("maskgit", "genie")   # una línea por familia y tamaño, de 1 a N pasadas, rotuladas con su número
  for fam in CON_PASADAS:
    for size in ("s", "m"):
      mg = sorted([c for c in main if c["family"] == fam and c["size"] == size and c["steps"]], key=lambda c: c["steps"])
      if not mg:
        continue
      ax.plot([c["ms"] for c in mg], [c[key] for c in mg], c=color(fam), lw=1, alpha=0.6,
              ls="-" if size == "s" else ":", zorder=2)
      for c in mg:
        _mark(ax, c["ms"], c[key], c, _err(c, c[key], *ci(c)), ms=6 if size == "s" else 5, hollow=size == "m")
        _txt(ax, c["ms"], c[key], str(c["steps"]), dx=0 if size == "s" else 8, dy=-12 if size == "s" else 0,
             ha="center" if size == "s" else "left", size=7, col=INK)
      last = mg[-1]
      _txt(ax, last["ms"], last[key], f"{'MaskGIT' if fam == 'maskgit' else 'Genie'}-{size.upper()}",
           dx=10 if size == "s" else 12, dy=8 if size == "s" else -16)
  for c in pts:
    if c["family"] not in CON_PASADAS:
      _mark(ax, c["ms"], c[key], c, _err(c, c[key], *ci(c)), hollow=c.get("kd") or c["group"] == "extra")
  lab = {c["short"]: c for c in pts if c["family"] not in CON_PASADAS}
  pos = {"Transición por celda": (9, 0, "left"), "gpt2m": (-9, 9, "right"), "gpt_s_kd": (-9, 9, "right"),
         "convlstm_px": (9, 0, "left"), "gpt_s": (-9, -7, "right"), "gpt_m": (9, 7, "left")}
  for name, (dx, dy, ha) in pos.items():
    if name in lab:
      c = lab[name]
      _txt(ax, c["ms"], c[key], name, dx=dx, dy=dy, ha=ha)
  fams = [f for f in ("gpt", "gpt_kd", "genie", "maskgit", "convlstm", "gpt2m", "baseline") if any(c["family"] == f for c in pts)]
  _legend(ax, fams, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False,
          extra=[Line2D([], [], c=MUTED, lw=1, label="frontera de Pareto (780.000 frames)")])
  ax.set_xlim(0.3, 2000)
  ax.set_ylim(0, None)
  ax.set_xlabel("latencia por frame generado (ms, mediana, batch 1, RTX 3080 Ti; escala log)")
  ax.set_ylabel(ylabel)
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def fig_pareto_latente(cfgs, key, ylabel, path, ci):
  """Los world models latentes (lewm) sobre las referencias de tokens (gris, rotuladas)."""
  from matplotlib.lines import Line2D
  lat = [c for c in cfgs if c["family"] in LATENT_FAMS and c["ms"] and c[key] is not None]
  if not lat:
    return
  refs = [c for c in cfgs if c["group"] == "principal" and c["family"] not in LATENT_FAMS and c["ms"]
          and c[key] is not None]
  main = refs + [c for c in lat if c["group"] == "principal"]
  fig, ax = plt.subplots(figsize=(8, 6))
  ax.set_xscale("log")
  ax.axvspan(RT_MS, 5000, color=GRID, alpha=0.45, lw=0, zorder=0)
  ax.axvline(RT_MS, ls="--", c=INK2, lw=0.9, zorder=1)
  opt = pareto_mask([c["ms"] for c in main], [c[key] for c in main])
  front = sorted((c["ms"], c[key]) for c, o in zip(main, opt) if o)
  if front:
    ax.step([p[0] for p in front] + [2000], [p[1] for p in front] + [front[-1][1]], where="post", c=MUTED, lw=1, zorder=1)
  ref_lab = {"Transición": "Transición", "GPT-S": "GPT-S", "gpt2m (comma)": "gpt2m"}
  labels = []
  for c in refs:
    ax.plot(c["ms"], c[key], ls="", marker=MARKERS.get(c["family"], "o"), ms=5, color=MUTED, mfc=MUTED, zorder=2)
    t = ref_lab.get(c["short"]) if not c.get("kd") else None
    if c["family"] == "maskgit" and c["size"] == "s" and c["steps"] in (1, 8):
      t = f"MaskGIT-S {c['steps']}"
    if t:
      labels.append(dict(x=c["ms"], y=c[key], t=t, dx=-7, ha="right", size=7, col=MUTED))
  for c in lat:  # etiqueta directa a la derecha de cada punto (los controles comparten latencia con su principal)
    _mark(ax, c["ms"], c[key], c, _err(c, c[key], *ci(c)), ms=8 if c["group"] == "principal" else 6,
          hollow=c["group"] != "principal", z=4)
    labels.append(dict(x=c["ms"], y=c[key], t=c["short"], dx=9, ha="left", size=7.5 if c["group"] == "principal" else 7,
                       col=INK if c["group"] == "principal" else INK2))
  ax.set_xlim(0.2, 2000)  # margen a la izquierda para las etiquetas de LeWM
  ax.autoscale_view()
  ax.set_ylim(0, ax.get_ylim()[1])
  _repel(ax, labels, points=[(c["ms"], c[key]) for c in refs + lat])
  fams = [f for f in LATENT_FAMS if any(c["family"] == f for c in lat)]
  _legend(ax, fams, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False,
          extra=[Line2D([], [], ls="", marker="o", ms=5, color=MUTED, label="modelos de tokens (100M tokens)"),
                 Line2D([], [], c=MUTED, lw=1, label="frontera de Pareto (principales)"),
                 Line2D([], [], ls="", marker="o", ms=6, color=INK2, mfc=SURF, label="hueco: control, ablación o rollout latente")])
  ax.set_xlabel("latencia por frame generado (ms, mediana, batch 1, RTX 3080 Ti; escala log)")
  ax.set_ylabel(ylabel)
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


HORIZON = [("gpt2m", "-"), ("gpt_s_kd", "-"), ("genie_s_p4", "-"), ("genie_s_p25", "--"), ("convlstm_px", "-"),
           ("Frame congelado", "--"), ("Transición por celda", "-")]


def fig_horizon(cfgs, path):
  by = {c["label"]: c for c in cfgs}
  sel = [(by[l], ls) for l, ls in HORIZON if l in by and by[l]["lpips_curve"]]
  if not sel:  # otro conjunto de configuraciones: las marcadas con horizon, hasta 9
    sel = [(c, "-") for c in cfgs if c["horizon"] and c["lpips_curve"]][:9]
  if not sel:
    return
  fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
  for k, (c, ls) in enumerate(sel):
    col = color(c["family"])
    cur = np.asarray(c["lpips_curve"], float)
    a.plot((np.arange(len(cur)) + 1) / FPS, cur, color=col, lw=1.6, ls=ls, label=c["label"])
    hs = [h for h in ("1", "2", "3") if c["fvd"][h]]
    if hs:
      d = [c["fvd"][h] for h in hs]
      val = np.array([x["value"] for x in d])
      lo, hi = np.array([x["lo"] for x in d]), np.array([x["hi"] for x in d])
      x = np.array([int(h) for h in hs]) + (k - len(sel) / 2) * 0.035
      b.errorbar(x, val, yerr=[val - lo, hi - val], fmt=MARKERS.get(c["family"], "o"), ms=5, ls=ls, lw=1.1,
                 capsize=2, color=col, mfc=SURF if ls == "--" else col, mew=1.2)
  a.set_xlabel("segundos de rollout")
  a.set_ylabel("LPIPS frente al futuro real (menor es mejor)")
  a.set_title("LPIPS por horizonte (400 clips)", fontsize=10, color=INK)
  a.set_xlim(0, 3.05)
  b.set_xticks([1, 2, 3])
  b.set_xlabel("ventana de 16 frames que termina a los ... s")
  b.set_ylabel("FVD (I3D, menor es mejor)")
  b.set_title("FVD a 1, 2 y 3 s (IC 95 %)", fontsize=10, color=INK)
  b.set_ylim(0, None)
  fig.legend(*a.get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.06))
  fig.tight_layout(rect=(0, 0.06, 1, 1))
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def correlation(cfgs):
  """Spearman entre CE (ce_all) y FVD a 1 s de las configuraciones autoregresivas. -> (rho, p, n, puntos)."""
  pts = [c for c in cfgs if c["autoregressive"] and c["ce_all"] is not None and c["fvd1"] is not None]
  if len(pts) < 3:
    return None, None, len(pts), pts
  r = spearmanr([c["ce_all"] for c in pts], [c["fvd1"] for c in pts])
  return float(r.statistic), float(r.pvalue), len(pts), pts


def _p(p):
  return "p < 0,001" if p < 1e-3 else f"p = {p:.3f}".replace(".", ",")


def fig_ce_vs_fvd(cfgs, path):
  rho, p, n, pts = correlation(cfgs)
  if not pts:
    return
  fig, ax = plt.subplots(figsize=(7, 4.8))
  for c in pts:
    _mark(ax, c["ce_all"], c["fvd1"], c, _err(c, c["fvd1"], _get(c["fvd"]["1"], "lo"), _get(c["fvd"]["1"], "hi")),
          hollow=c.get("kd") or c["group"] != "principal")
  pos = {"gpt2m (comma)": (9, 0, "left"),
         "GPT-M KD": (-9, -3, "right"), "GPT-S KD": (-9, 9, "right"), "GPT-S": (9, -10, "left"), "GPT-M": (9, 6, "left"),
         "Transición": (-9, 0, "right"), "GPT-S, posición absoluta": (9, 8, "left")}
  for c in pts:
    name = c["short"] if c["family"] != "ablación" else "GPT-S, posición absoluta"
    dx, dy, ha = pos.get(name, (9, 0, "left"))
    _txt(ax, c["ce_all"], c["fvd1"], name, dx=dx, dy=dy, ha=ha)
  if rho is not None:
    ax.text(0.03, 0.97, f"Spearman ρ = {rho:.2f}".replace(".", ",") + f"\n{_p(p)}, n = {n}", transform=ax.transAxes,
            va="top", fontsize=9, color=INK)
  ax.set_xlabel("CE en val (nats por token, protocolo de comma)")
  ax.set_ylabel("FVD a 1 s (menor es mejor)")
  ax.set_ylim(0, None)
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


LAT = [("gpt2m (307M)", "gpt2m", ["fp32", "fp16", "fp16_graphs"]),
       ("GPT-L (88M, sin entrenar)", "random_gpt_l", ["fp32", "fp16", "fp16_graphs"]),
       ("GPT-M (26M)", "gpt_m", ["fp32", "fp16", "fp16_graphs"]), ("GPT-S (11M)", "gpt_s", ["fp32", "fp16", "fp16_graphs"]),
       ("MaskGIT-M, 8 pasos", "maskgit_m", ["fp32_s8", "fp16_s8", "fp16_graphs_s8"]),
       ("MaskGIT-S, 8 pasos", "maskgit_s", ["fp32_s8", "fp16_s8", "fp16_graphs_s8"]),
       ("MaskGIT-S, 2 pasos", "maskgit_s", ["fp32_s2", None, "fp16_graphs_s2"])]


def fig_latency(bench_dir, path, registro=None):
  """H3: ms/frame en fp32, fp16 y fp16 + CUDA graphs (rampa de un tono: más oscuro = más optimizado)."""
  shades = ["#a9c9ef", "#5d9ce3", "#1d5fae"]
  names = ["fp32", "fp16", "fp16 + CUDA graphs"]
  rows = []
  for lab, n, vs in LAT:
    vals = [_get(_load(Path(bench_dir) / f"{n}__{v}.json", Path(".")), "median_ms")
            if v and resultados.vigente(Path(bench_dir) / f"{n}__{v}.json", registro) else None for v in vs]
    if any(vals):
      rows.append((lab, vals))
  if not rows:
    return
  fig, ax = plt.subplots(figsize=(8, 4.8))
  h = 0.26
  for i, (lab, vals) in enumerate(rows):
    for j, v in enumerate(vals):
      if v:
        y = i + (j - 1) * h
        ax.barh(y, v, height=h * 0.88, color=shades[j], label=names[j] if i == 0 else None)
        ax.text(v * 1.08, y, f"{v:.0f}" if v >= 10 else f"{v:.1f}", va="center", fontsize=7, color=INK2)
  ax.set_yticks(range(len(rows)), [r[0] for r in rows])
  ax.invert_yaxis()
  ax.set_xscale("log")
  ax.set_xlim(0.5, 6000)
  ax.axvline(RT_MS, ls="--", c=INK2, lw=0.9)
  ax.text(RT_MS * 1.06, -0.62, f"{RT_MS:.0f} ms (20 FPS)", fontsize=8, color=INK2)
  ax.set_xlabel("ms por frame generado (mediana, batch 1, RTX 3080 Ti; escala log)")
  ax.grid(axis="y", visible=False)
  ax.legend(loc="lower right", fontsize=8)
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def fig_ce_posicion(path_json, path, registro=None):
  d = _load(path_json, Path(".")) if resultados.vigente(path_json, registro) else None
  if not d:
    return
  fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
  f = np.arange(1, 21)
  for k, lab, col, ls in [("gpt_s_abs", "GPT-S, posición absoluta", color("gpt"), "--"),
                          ("gpt_s", "GPT-S, frame + celda", color("gpt"), "-"),
                          ("gpt2m", "gpt2m (comma)", INK, "-")]:
    if k in d["por_frame"]:
      a.plot(f[1:], d["por_frame"][k][1:], color=col, ls=ls, lw=1.6, marker="o", ms=3, label=lab)
  a.set_xticks([2, 5, 10, 15, 20])
  a.set_xlabel("frame de la ventana de 20 (el rollout genera en el 20)")
  a.set_ylabel("CE del frame (nats por token)")
  a.set_title("GPT: el embedding absoluto falla al final de la ventana", fontsize=10, color=INK)
  a.legend(fontsize=8)
  for k, lab, col, ls in [("maskgit_s_nopc", "frame + celda, sin p_clean", color("maskgit"), "--"),
                          ("maskgit_s_abs", "posición absoluta", MUTED, "-"),
                          ("maskgit_s", "frame + celda, p_clean = 0,5", color("maskgit"), "-")]:
    if k in d["por_contexto"]:
      b.plot(f[1:] - 1, d["por_contexto"][k][1:], color=col, ls=ls, lw=1.6, marker="^", ms=3, label=lab)
  b.set_xticks([1, 4, 8, 12, 16, 19])
  b.set_xlabel("frames de contexto limpios")
  b.set_ylabel("CE del frame siguiente, 1 paso")
  b.set_title("MaskGIT-S: sin contexto limpio al entrenar, colapsa", fontsize=10, color=INK)
  b.legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def _r(v, nd=2):
  return None if v is None else round(float(v), nd)


def summary(cfgs):
  main = [c["group"] == "principal" and c["estado"] == "vigente" for c in cfgs]  # solo vigentes compiten
  opt = pareto_mask([c["ms"] if m else None for c, m in zip(cfgs, main)], [c["fvd1"] for c in cfgs])
  rows = []
  for c, o, m in zip(cfgs, opt, main):
    f1, f3 = c["fvd"]["1"], c["fvd"]["3"]
    rows.append({
      "label": c["label"], "family": c["family"], "params_M": _r(c["params"] / 1e6 if c["params"] else None, 1),
      "ce_all": _r(c["ce_all"], 3), "ce_last": _r(c["ce_last"], 3), "lpips_1s": _r(c["lpips1"], 3),
      "fvd_1s": _r(c["fvd1"], 1), "fvd_1s_ci": [_r(f1.get("lo"), 1), _r(f1.get("hi"), 1)] if f1 else None,
      "fvd_3s": _r(_get(f3, "value"), 1),
      "fvd_piso_1s": _r(c.get("piso_n1"), 1), "fvd_piso_mitades_1s": _r(c.get("piso1"), 1), "fvd_1s_mitades": _r(c.get("fvd1_mitades"), 1), "ms_frame": _r(c["ms"], 2), "e2e_p99_ms": _r(c.get("e2e_p99"), 1),
      "fps": _r(1000 / c["ms"], 1) if c["ms"] else None,
      "realtime": None if c["ms"] is None else bool(c["ms"] < RT_MS),
      "group": c["group"], "estado": c["estado"],
      "pareto": (bool(o) if c["ms"] is not None and c["fvd1"] is not None else None) if m else None})
  return rows


def markdown(rows):
  f = lambda v: "–" if v is None else str(v)
  piso = lambda r: "–" if r.get("fvd_piso_mitades_1s") is None else f"{r['fvd_piso_mitades_1s']}" + (
    f" ({r['fvd_1s_mitades']})" if r.get("fvd_1s_mitades") is not None else "")
  yn = lambda v: "–" if v is None else ("sí" if v else "no")
  out = ["| Configuración | Familia | Params (M) | CE | CE último frame | LPIPS@1s | FVD@1s [IC 95% por segmento] | FVD@3s "
         "| Piso FVD@1s (400 contra 400) | Piso en mitades (modelo en mitades) | ms/frame | FPS | Tiempo real (<50 ms) | Pareto | De punta a punta p99 (ms) | Estado |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|:-:|---:|---|"]
  out = ["Pareto: no dominada en (ms/frame, FVD a 1 s) entre las configuraciones principales (780.000 frames); las semillas 1 y 2, las "
         "ablaciones y los extras no compiten. Latencia de los transformers con fp16 + CUDA graphs. De punta a punta: "
         "cámara → encoder → modelo → decoder → frame en la GPU, a ritmo de cámara (un frame cada 50 ms), p99 de 3000 "
         "frames (results/infer/latencia_*.json). Estado: solo las filas vigentes entran en la frontera de Pareto y en las figuras "
         "(results/estado.json). IC del FVD y de LPIPS: bootstrap por segmento (los 4 clips de un segmento no son independientes). "
         "Piso (400 contra 400): FVD de los clips reales de evaluación contra otros 400 reales de los mismos segmentos "
         "(results/piso_fvd.json); es el comparable con la columna de FVD. Piso en mitades: real contra real con dos mitades "
         "disjuntas por segmento (n/2 por lado), y entre paréntesis el FVD del modelo con esas mismas mitades.", ""] + out
  for r in rows:
    ci = r["fvd_1s_ci"]
    fvd = f(r["fvd_1s"]) + (f" [{f(ci[0])}, {f(ci[1])}]" if ci and r["fvd_1s"] is not None else "")
    out.append(f"| {r['label']} | {r['family']} | {f(r['params_M'])} | {f(r['ce_all'])} | {f(r['ce_last'])} | "
               f"{f(r['lpips_1s'])} | {fvd} | {f(r['fvd_3s'])} | {f(r.get('fvd_piso_1s'))} | {piso(r)} | {f(r['ms_frame'])} | {f(r['fps'])} | "
               f"{yn(r['realtime'])} | {yn(r['pareto'])} | {f(r.get('e2e_p99_ms'))} | {r['estado']} |")
  return "\n".join(out) + "\n"


def cartel(path, texto="Sin resultados vigentes:" + chr(10) + "todos son de un protocolo anterior o están pendientes" + chr(10) + "(ver results/estado.json)"):
  """Figura vacía con un texto, para cuando no hay nada vigente que graficar."""
  fig, ax = plt.subplots(figsize=(6, 2.4))
  ax.axis("off")
  ax.text(0.5, 0.5, texto, ha="center", va="center", fontsize=10, color=INK2, transform=ax.transAxes)
  fig.savefig(path, dpi=100, bbox_inches="tight")
  plt.close(fig)


FIGURAS = ("fig_pareto", "fig_pareto_lpips", "fig_pareto_latente", "fig_pareto_latente_lpips", "fig_horizon",
           "fig_ce_vs_fvd", "fig_latency", "fig_ce_posicion")


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--manifest", default="results/configs.json")
  ap.add_argument("--out", default="results")
  ap.add_argument("--estados", default=str(resultados.ARCHIVO), help="results/estado.json")
  a = ap.parse_args(argv)
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  registro = resultados.cargar(a.estados)
  cfgs = load_configs(a.manifest, registro)
  vig = [c for c in cfgs if c["estado"] == "vigente"]
  if not vig:
    for f in FIGURAS:
      cartel(out / f"{f}.png")
  else:
    fig_pareto(vig, "fvd1", "FVD a 1 s (menor es mejor)", out / "fig_pareto.png",
               lambda c: (_get(c["fvd"]["1"], "lo"), _get(c["fvd"]["1"], "hi")))
    fig_pareto(vig, "lpips1", "LPIPS a 1 s (menor es mejor)", out / "fig_pareto_lpips.png", lambda c: c["lpips1_ci"])
    fig_pareto_latente(vig, "fvd1", "FVD a 1 s (menor es mejor)", out / "fig_pareto_latente.png",
                       lambda c: (_get(c["fvd"]["1"], "lo"), _get(c["fvd"]["1"], "hi")))
    fig_pareto_latente(vig, "lpips1", "LPIPS a 1 s (menor es mejor)", out / "fig_pareto_latente_lpips.png",
                       lambda c: c["lpips1_ci"])
    fig_horizon(vig, out / "fig_horizon.png")
    fig_ce_vs_fvd(vig, out / "fig_ce_vs_fvd.png")
    fig_latency(Path(a.manifest).parent / "bench", out / "fig_latency.png", registro)
    fig_ce_posicion(Path(a.manifest).parent / "ce_por_posicion.json", out / "fig_ce_posicion.png", registro)
  rows = summary(cfgs)
  rho, p, n, _ = correlation(vig)
  (out / "summary.json").write_text(json.dumps({"configs": rows, "ce_vs_fvd1": {"spearman_rho": rho, "p": p, "n": n}},
                                               indent=1, ensure_ascii=False), encoding="utf-8")
  md = markdown(rows)
  if rho is not None:
    md += f"\nCE vs. FVD@1s (autoregresivas, vigentes): Spearman ρ = {rho:.2f}, {_p(p)}, n = {n}.\n"
  (out / "summary.md").write_text(md, encoding="utf-8")
  print(f"{len(rows)} configuraciones ({len(vig)} vigentes) -> {out}")
  return rows


if __name__ == "__main__":
  main()
