"""Figuras y tablas del paper, leídas de los JSON vigentes de results/ (no mide nada).

  python paper/hacer_figuras.py        # -> paper/figs/*.pdf y paper/tablas/*.tex

Figuras:
  calidad_latencia  (a) FVD a 3 s contra ms/frame, misma GPU (RTX 3080 Ti); (b) cadena completa
  drift             FVD a 1, 2 y 3 s con IC 95 % por segmento (cuánto se degrada el video con el horizonte)
  pasos             FVD contra pasos de decodificación paralela (Genie S/M, MaskGIT S) a 1 y 3 s
  datos_escala      (a) frames vistos frente al dataset y a gpt2m; (b) CE contra tokens del GPT-S y extrapolación
Tablas: principal, latentes, semillas, cadena.
La figura de frames generados sale de src.ejemplos (ver el final de este archivo).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("results")
OUT_F, OUT_T = Path("paper/figs"), Path("paper/tablas")
OUT_F.mkdir(parents=True, exist_ok=True)
OUT_T.mkdir(parents=True, exist_ok=True)

# --- datos del dataset y del presupuesto (README de commaVQ y Docs/resumen.md) ---
FPS, TOK_FRAME = 20, 128
FRAMES_PRESUPUESTO = 780_000                # frames vistos por cada modelo (100M de tokens)
FRAMES_SHARD0 = 2_500 * 60 * FPS            # el único shard bajado: 2.500 segmentos de 1 min
FRAMES_COMMAVQ = 100_000 * 60 * FPS         # dataset público completo: 100.000 minutos
FRAMES_GPT2M = 3_000_000 * 60 * FPS         # con lo que comma entrenó gpt2m: 3.000.000 de minutos

# --- estilo: un color por familia (paleta categórica validada) y un marcador por familia ---
TINTA, TINTA2, GRILLA = "#2b2b29", "#6b6a64", "#e4e3de"
COLOR = {"gpt2m": "#3d3d3a", "gpt": "#2a78d6", "maskgit": "#eb6834", "genie": "#1baf7a", "genie_m": "#0f7d56",
         "convlstm": "#6250d6", "congelado": "#eda100", "transicion": "#e87ba4"}
MARCA = {"gpt2m": "*", "gpt": "o", "maskgit": "s", "genie": "^", "genie_m": "v", "convlstm": "D",
         "congelado": "P", "transicion": "X"}
plt.rcParams.update({
  "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7,
  "ytick.labelsize": 7, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA, "xtick.color": TINTA2,
  "ytick.color": TINTA2, "text.color": TINTA, "axes.grid": True, "grid.color": GRILLA, "grid.linewidth": 0.6,
  "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6, "lines.markersize": 5,
  "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL1, COL2 = 3.35, 7.0   # ancho de una columna y de la página (pulgadas)


def leer(p):
  p = Path(p)
  return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def video(n):
  return leer(R / f"video_{n}.json")


def fvd(n, h):
  v = video(n)
  return None if v is None else v["fvd"][str(h)]


def bench_ms(nombre):
  """p50 de results/bench/<nombre>__*.json (fp16 + graphs si existe)."""
  cands = sorted((R / "bench").glob(f"{nombre}__*.json"))
  cands = [c for c in cands if "graphs" in c.name] or cands
  if not cands:
    return None
  b = leer(cands[0])
  for k in ("p50", "p50_ms", "ms_p50"):
    if k in b:
      return b[k]
  return b.get("latency", {}).get("p50") or b.get("ms", {}).get("p50")


def guardar(fig, nombre):
  fig.savefig(OUT_F / f"{nombre}.pdf")
  fig.savefig(OUT_F / f"{nombre}.png", dpi=320)
  plt.close(fig)
  print("figura", OUT_F / f"{nombre}.pdf")


SUM = {c["label"]: c for c in leer(R / "summary.json")["configs"]}


# =================================================================================================
# 1. calidad contra latencia
# =================================================================================================
CADENAS = ("gpt_s_kd", "genie_s_p4", "genie_m_p4", "convlstm_px", "transicion")


def cadenas():
  """Las mediciones canónicas de la cadena en la 3080 Ti (no los pasos intermedios de la optimización del GPT)."""
  return [R / "infer" / f"latencia_{n}__rtx3080ti.json" for n in CADENAS if (R / "infer" / f"latencia_{n}__rtx3080ti.json").exists()]


def fig_calidad_latencia():
  fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.5), gridspec_kw={"width_ratios": [1.35, 1]})
  # (a) misma GPU: RTX 3080 Ti (latencias canónicas, con el decode optimizado del GPT)
  series = [("gpt2m", "gpt2m (307M)", [("gpt2m", "gpt2m", None)]),
            ("gpt", "GPT-S destilado", [("gpt_s_kd", "gpt_s_kd", None)]),
            ("maskgit", "MaskGIT-S", [(f"maskgit_s_abs_p{s}", f"maskgit_s_abs_p{s}", s) for s in (1, 2, 4, 8, 12)]),
            ("genie", "Genie-S", [(f"genie_s_p{s}", f"genie_s_p{s}", s) for s in (1, 2, 4, 8, 12)])]
  for fam, etiqueta, pts in series:
    xs, ys, ss = [], [], []
    for nb, nv, s in pts:
      ms, f = bench_ms(nb), fvd(nv, 3)
      if ms is None or f is None:
        continue
      xs.append(ms); ys.append(f["value"]); ss.append(s)
    if not xs:
      continue
    a.plot(xs, ys, marker=MARCA[fam], color=COLOR[fam], label=etiqueta, lw=1.2 if len(xs) > 1 else 0,
           ms=8 if fam == "gpt2m" else 5, mec="white", mew=0.6, zorder=3)
    for x, y, s in zip(xs, ys, ss):
      if s in (1, 4, 12):
        a.annotate(f"{s}", (x, y), textcoords="offset points", xytext=(3, -11 if fam == "genie" else 4), fontsize=6.5, color=TINTA2)
  a.axvline(50, color=TINTA2, ls="--", lw=0.9)
  a.text(53, a.get_ylim()[0] + 15, "50 ms = 20 FPS", fontsize=6.5, color=TINTA2, va="bottom")
  a.set_xscale("log")
  a.set_xlabel("ms por frame, modelo solo (p50, RTX 3080 Ti, escala log)")
  a.set_ylabel("FVD a 3 s (menor es mejor)")
  a.set_title("(a) Calidad contra latencia en la misma GPU", loc="left")
  a.legend(loc="upper left", bbox_to_anchor=(0.12, 1.0))
  # (b) cadena completa a ritmo de cámara, 3080 Ti
  filas = []
  for f in cadenas():
    d = leer(f)
    st = d["cadena_cerrada"].get("stages_p50_ms") or {}
    filas.append((d["name"], st.get("encoder", 0), st.get("model", 0), st.get("decoder", 0), d["cadena_ritmo"]["p99"]))
  filas.sort(key=lambda r: r[4])
  bonito = {"gpt_s_kd": "GPT-S destilado", "convlstm_px": "ConvLSTM", "genie_s_p4": "Genie-S, 4 pasos",
            "genie_m_p4": "Genie-M, 4 pasos", "transicion": "Transición por celda"}
  nombres = [bonito.get(r[0], r[0]) for r in filas]
  y = np.arange(len(filas))
  izq = np.zeros(len(filas))
  for i, (et, col) in enumerate([("encoder", "#9ec5f4"), ("modelo", "#2a78d6"), ("decoder", "#184f95")]):
    v = np.array([r[1 + i] for r in filas])
    b.barh(y, v, left=izq, color=col, label=et, height=0.62, edgecolor="white", linewidth=1)
    izq += v
  for yi, r in zip(y, filas):
    b.text(max(izq[yi], 1) + 3, yi, f"p99 {r[4]:.0f} ms", va="center", fontsize=6.5, color=TINTA)
  b.axvline(50, color=TINTA2, ls="--", lw=0.9)
  b.set_yticks(y, nombres)
  b.set_xlabel("ms por frame (p50 por etapa, RTX 3080 Ti)")
  b.set_title("(b) Cadena completa a ritmo de cámara", loc="left")
  b.legend(loc="center right", fontsize=6.5)
  b.grid(axis="y", visible=False)
  b.set_xlim(0, max(izq.max() * 1.45, 60))
  fig.tight_layout()
  guardar(fig, "calidad_latencia")


# =================================================================================================
# 2. drift: FVD contra horizonte
# =================================================================================================
DRIFT = [("gpt2m", "gpt2m", "gpt2m (comma)"), ("gpt_s_kd", "gpt", "GPT-S destilado"),
         ("maskgit_s_abs_p8", "maskgit", "MaskGIT-S, 8 pasos"), ("genie_s_p4", "genie", "Genie-S, 4 pasos"),
         ("genie_s_p1", "genie", "Genie-S, 1 paso"), ("transicion", "transicion", "Transición por celda"),
         ("congelado", "congelado", "Frame congelado"), ("convlstm_px", "convlstm", "ConvLSTM (píxeles)")]


def fig_drift():
  fig, ax = plt.subplots(figsize=(COL1, 2.6))
  hs = [1, 2, 3]
  for n, fam, et in DRIFT:
    if video(n) is None:
      continue
    v = [fvd(n, h) for h in hs]
    y = [x["value"] for x in v]
    ls = ":" if n == "genie_s_p1" else "-"
    ax.fill_between(hs, [x["lo"] for x in v], [x["hi"] for x in v], color=COLOR[fam], alpha=0.12, lw=0)
    ax.plot(hs, y, marker=MARCA[fam], color=COLOR[fam], ls=ls, label=et, mec="white", mew=0.6)
  ax.set_yscale("log")
  ax.set_xticks(hs, ["1 s", "2 s", "3 s"])
  ax.set_xlabel("horizonte de generación (después de 1 s de contexto)")
  ax.set_ylabel("FVD (escala log, menor es mejor)")
  ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
  guardar(fig, "drift")


# =================================================================================================
# 3. pasos de decodificación
# =================================================================================================
def fig_pasos():
  fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.2), sharey=False)
  fams = [("genie", "Genie-S (15M)", "genie_s_p{}", (1, 2, 4, 8, 12, 25)),
          ("genie_m", "Genie-M (35M)", "genie_m_p{}", (1, 2, 4, 8, 12, 25)),
          ("maskgit", "MaskGIT-S (12M, backbone del GPT)", "maskgit_s_abs_p{}", (1, 2, 4, 8, 12))]
  for ax, h in zip(axs, (1, 3)):
    for fam, et, pat, pasos in fams:
      ps = [s for s in pasos if video(pat.format(s))]
      vs = [fvd(pat.format(s), h) for s in ps]
      ax.fill_between(ps, [v["lo"] for v in vs], [v["hi"] for v in vs], color=COLOR[fam], alpha=0.12, lw=0)
      ax.plot(ps, [v["value"] for v in vs], marker=MARCA[fam], color=COLOR[fam], label=et, mec="white", mew=0.6)
    g = fvd("gpt_s_kd", h)
    ax.axhline(g["value"], color=COLOR["gpt"], ls="--", lw=1)
    ax.text(25, g["value"], " GPT-S destilado\n (128 pasadas)", color=TINTA2, fontsize=6.5, va="center")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 12, 25], ["1", "2", "4", "8", "12", "25"])
    ax.set_xlabel("pasos de decodificación paralela por frame")
    ax.set_title(f"FVD a {h} s", loc="left")
  axs[0].set_ylabel("FVD (menor es mejor)")
  axs[1].legend(loc="upper right")
  fig.tight_layout()
  guardar(fig, "pasos")


# =================================================================================================
# 4. datos y escala
# =================================================================================================
def puntos_escala():
  """[(tokens, CE en val, run)] de los GPT-S de cero; el de 1x entra cuando exista."""
  out = []
  for run in ("gpt_s_d025", "gpt_s_d050", "gpt_s"):
    t, e = leer(R / f"train_{run}.json"), leer(R / f"eval_{run}.json")
    if t and e and run in SUM and SUM[run].get("ce_all"):
      out.append((t["tokens"], SUM[run]["ce_all"], run))
  return out


def ajuste(pts, e_fijo):
  """L(D) = E + B·D^-β. Con 3+ puntos ajusta los tres; con 2 fija E (supuesto declarado)."""
  D = np.array([p[0] for p in pts], float)
  L = np.array([p[1] for p in pts], float)
  if len(pts) >= 3:
    from scipy.optimize import curve_fit
    f = lambda d, E, B, b: E + B * (d / 1e8) ** (-b)
    (E, B, b), _ = curve_fit(f, D, L, p0=(2.0, 1.0, 0.5), maxfev=20000)
    return E, B, b, "E, B y β ajustados"
  E = e_fijo
  b = np.log((L[0] - E) / (L[1] - E)) / np.log(D[1] / D[0])
  B = (L[1] - E) * (D[1] / 1e8) ** b
  return E, B, b, f"E fijado en la CE de gpt2m ({E:.2f})"


def fig_datos_escala():
  fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.4), gridspec_kw={"width_ratios": [1, 1.25]})
  # (a) frames
  et = ["Presupuesto por\nmodelo (este trabajo)", "Shard 0\n(lo que bajamos)", "commaVQ\ncompleto", "Datos de gpt2m\n(comma)"]
  v = [FRAMES_PRESUPUESTO, FRAMES_SHARD0, FRAMES_COMMAVQ, FRAMES_GPT2M]
  cols = [COLOR["gpt"], "#9ec5f4", "#b9b8b0", COLOR["gpt2m"]]
  y = np.arange(len(v))[::-1]
  a.barh(y, v, color=cols, height=0.6, edgecolor="white")
  for yi, x in zip(y, v):
    a.text(x * 1.3, yi, f"{x / 1e6:,.2f} M".replace(",", "X").replace(".", ",").replace("X", ".") if x < 1e7
           else f"{x / 1e6:,.0f} M".replace(",", "."), va="center", fontsize=6.5)
    pass
  a.set_xscale("log")
  a.set_yticks(y, et)
  a.set_xlim(1e5, 1e13)
  a.set_xlabel("frames (escala log)")
  a.set_title("(a) Frames de entrenamiento", loc="left")
  a.text(0.98, 0.50, f"presupuesto = {100 * FRAMES_PRESUPUESTO / FRAMES_COMMAVQ:.2f} % de commaVQ\n"
         f"y {100 * FRAMES_PRESUPUESTO / FRAMES_GPT2M:.3f} % de los datos de gpt2m".replace(".", ","),
         transform=a.transAxes, ha="right", va="center", fontsize=6.5, color=TINTA2)
  a.grid(axis="y", visible=False)
  # (b) escala
  pts = puntos_escala()
  ce_gpt2m = SUM["gpt2m"]["ce_all"]
  N = 12.4e6
  if len(pts) >= 2:
    E, B, beta, como = ajuste(pts, ce_gpt2m)
    d = np.logspace(np.log10(2e7), np.log10(1.2e9), 200)
    lpred = E + B * (d / 1e8) ** (-beta)
    b.plot(d, lpred, color=COLOR["gpt"], ls="--", lw=1.2, label=f"ajuste $E+B\\,D^{{-\\beta}}$, β={beta:.2f}\n({como})")
    for k, mult in ((1, "1×"), (2.48, "Chinchilla\n(20 tokens/param)"), (10, "10×")):
      dd = 1.008e8 * k
      ll = E + B * (dd / 1e8) ** (-beta)
      b.plot([dd], [ll], marker="o", mfc="white", mec=COLOR["gpt"], ms=5)
      b.annotate(f"{mult}: {ll:.2f}".replace(".", ","), (dd, ll), textcoords="offset points", xytext=(4, 6),
                 fontsize=6.5, color=TINTA2)
  for t, ce, run in pts:
    b.plot([t], [ce], marker="o", color=COLOR["gpt"], ms=6, mec="white", mew=0.6, zorder=4)
    b.annotate(run.replace("gpt_s_d0", "").replace("gpt_s", "100") + " %" if run != "gpt_s" else "100 %",
               (t, ce), textcoords="offset points", xytext=(-5, -4), ha="right", va="top", fontsize=6.5,
               color=TINTA)  # abajo a la izquierda del punto: la curva baja hacia la derecha y no lo pisa
  kd = SUM["gpt_s_kd"]["ce_all"]
  b.plot([1.008e8], [kd], marker="o", color=COLOR["maskgit"], ms=6, mec="white", mew=0.6, ls="",
         label=f"GPT-S destilado (1×): {kd:.2f}".replace(".", ","))
  b.axhline(ce_gpt2m, color=COLOR["gpt2m"], lw=1)
  b.text(2.1e7, ce_gpt2m + 0.04, "gpt2m (307M, ~1.500 tokens/param): " + f"{ce_gpt2m:.2f}".replace(".", ","),
         fontsize=6.5, va="bottom",
         color=TINTA2)
  b.set_xscale("log")
  b.set_xlabel("tokens de entrenamiento D (escala log)")
  b.set_ylabel("cross-entropy en val (nats/token)")
  b.set_title("(b) GPT-S (12,4M) de cero: CE contra datos", loc="left")
  b.legend(loc="upper right", fontsize=6.3)
  fig.tight_layout()
  guardar(fig, "datos_escala")
  return pts


# =================================================================================================
# tablas
# =================================================================================================
def num(x, d=1):
  return "--" if x is None else f"{x:.{d}f}".replace(".", "{,}")


def tabla_principal():
  filas = [("gpt2m", "gpt2m (comma)", "gpt2m", None),
           ("gpt_s_kd", "GPT-S destilado", "gpt_s_kd", None),
           ("gpt_s_d050", "GPT-S de cero, 50\\,\\%", "gpt_s_d050", None),
           ("gpt_s_d025", "GPT-S de cero, 25\\,\\%", "gpt_s_d025", None)]
  if "gpt_s" in SUM:
    filas.insert(2, ("gpt_s", "GPT-S de cero, 100\\,\\%", "gpt_s", None))
  filas += [(f"maskgit_s_abs_p{s}", f"MaskGIT-S, {s} paso{'s' * (s > 1)}", f"maskgit_s_abs_p{s}",
             None) for s in (1, 2, 4, 8, 12)]
  filas += [(f"genie_s_p{s}", f"Genie-S, {s} paso{'s' * (s > 1)}", f"genie_s_p{s}", None)
            for s in (1, 2, 4, 8, 12, 25)]
  filas += [(f"genie_m_p{s}", f"Genie-M, {s} pasos", f"genie_m_p{s}", None) for s in (4, 8)]
  filas += [("convlstm_px", "ConvLSTM (píxeles)", "convlstm_px", None),
            ("Transición por celda", "Transición por celda", "transicion", None),
            ("Frame congelado", "Frame congelado", "congelado", None),
            ("Unigrama", "Unigrama", "unigram", None)]
  L = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
       r"Modelo & Params & CE & LPIPS & \multicolumn{3}{c}{FVD $\downarrow$} & Drift & ms/frame \\",
       r"\cmidrule(lr){5-7}",
       r" & (M) & $\downarrow$ & 1\,s & 1\,s [IC 95\,\%] & 2\,s & 3\,s & 3\,s/1\,s & 3080\,Ti \\", r"\midrule"]
  for lab, nombre, nv, nb in filas:
    c = SUM.get(lab, {})
    v = video(nv)
    if v is None:
      continue
    f1, f2, f3 = (v["fvd"][h] for h in ("1", "2", "3"))
    ce = c.get("ce_all")
    ms38 = c.get("ms_frame")
    L.append(f"{nombre} & {num(c.get('params_M'))} & {num(ce, 2)} & {num(c.get('lpips_1s'), 2)} & "
             f"{num(f1['value'], 0)} [{num(f1['lo'], 0)}, {num(f1['hi'], 0)}] & {num(f2['value'], 0)} & "
             f"{num(f3['value'], 0)} & {num(f3['value'] / f1['value'], 1)} & {num(ms38)} \\\\")
  L += [r"\bottomrule", r"\end{tabular}"]
  (OUT_T / "principal.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
  print("tabla", OUT_T / "principal.tex")


def tabla_latentes():
  L = [r"\begin{tabular}{lrrrr}", r"\toprule",
       r"Encoder & \multicolumn{2}{c}{$R^2$ velocidad} & \multicolumn{2}{c}{$R^2$ giro} \\",
       r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
       r" & entrenado & sin entrenar & entrenado & sin entrenar \\", r"\midrule"]
  pix = None
  for n, et in (("lewm", "LeWM"), ("lewm_lrx3", "LeWM, lr $\\times3$"), ("vjepa2_s", "V-JEPA~2-S"),
                ("vjepa2_ac", "V-JEPA~2-AC")):
    d = leer(R / f"latent_{n}.json")
    if not d:
      continue
    o = d["probes"]["obs"]
    r = lambda s, t: o[s][t]["r2"]["valor"]
    pix = pix or o["pixeles"]
    L.append(f"{et} & {num(r('modelo', 'velocidad'), 3)} & {num(r('sin_entrenar', 'velocidad'), 3)} & "
             f"{num(r('modelo', 'giro'), 3)} & {num(r('sin_entrenar', 'giro'), 3)} \\\\")
  if pix:
    L.append(r"\midrule")
    L.append(f"Píxeles crudos & \\multicolumn{{2}}{{c}}{{{num(pix['velocidad']['r2']['valor'], 3)}}} & "
             f"\\multicolumn{{2}}{{c}}{{{num(pix['giro']['r2']['valor'], 3)}}} \\\\")
  L += [r"\bottomrule", r"\end{tabular}"]
  (OUT_T / "latentes.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
  print("tabla", OUT_T / "latentes.tex")


def tabla_semillas():
  L = [r"\begin{tabular}{lrrr}", r"\toprule", r"Modelo (3 semillas) & FVD 1\,s & FVD 2\,s & FVD 3\,s \\", r"\midrule"]
  for n, et, runs in (("genie_s_p8", "Genie-S, 8 pasos", ("genie_s_p8", "genie_s_s1_p8", "genie_s_s2_p8")),
                      ("convlstm_px", "ConvLSTM", ("convlstm_px", "convlstm_px_s1", "convlstm_px_s2"))):
    c = []
    for h in ("1", "2", "3"):
      xs = np.array([video(r)["fvd"][h]["value"] for r in runs])
      c.append(f"{num(xs.mean(), 0)} $\\pm$ {num(xs.std(ddof=1), 0)}")
    L.append(f"{et} & " + " & ".join(c) + r" \\")
  L += [r"\bottomrule", r"\end{tabular}"]
  (OUT_T / "semillas.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
  print("tabla", OUT_T / "semillas.tex")


NOMBRE_CADENA = {"convlstm_px": "ConvLSTM (píxeles)", "genie_m_p4": "Genie-M, 4 pasos", "genie_s_p4": "Genie-S, 4 pasos",
                 "gpt_s_kd": "GPT-S destilado", "transicion": "Transición por celda"}


def tabla_cadena():
  L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
       r"Configuración & encoder & modelo & decoder & p99 a ritmo & tarde \\",
       r" & \multicolumn{3}{c}{p50 (ms)} & (ms) & (\%) \\", r"\midrule"]
  for f in cadenas():
    d = leer(f)
    st = d["cadena_cerrada"].get("stages_p50_ms") or {}
    cr = d["cadena_ritmo"]
    nombre = NOMBRE_CADENA.get(d["name"], d["name"].replace("_", chr(92) + "_"))
    L.append(f"{nombre} & {num(st.get('encoder'))} & {num(st.get('model'))} & "
             f"{num(st.get('decoder'))} & {num(cr['p99'])} & {num(cr.get('late_pct'))} \\\\")
  L += [r"\bottomrule", r"\end{tabular}"]
  (OUT_T / "cadena.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
  print("tabla", OUT_T / "cadena.tex")


def fig_online():
  """Uso en línea (results/online.json, de `python -m src.infer.online_eval`): cada periodo el modelo ve los frames
  reales que llegaron y genera `antic` s hacia adelante. (a) línea de tiempo de un clip; (b) con cuánta anticipación
  está listo el frame a +antic (IQR entre clips); (c) fracción del tiempo con una predicción vigente en pantalla."""
  d = leer(R / "online.json")
  if d is None:
    print("falta results/online.json: correr python -m src.infer.online_eval")
    return
  cfg, ms = d["config"], d["modelos"]
  col = {"gpt_s_kd": COLOR["gpt"], "maskgit_s8": "#b84c1f", "maskgit_s4": COLOR["maskgit"], "genie_s4": COLOR["genie"]}
  c_de = lambda m: col.get(m["id"] + (str(m["pasos"]) if m["pasos"] else ""), TINTA2)
  fig = plt.figure(figsize=(COL2, 2.15))
  gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.05, 0.95], wspace=0.22)
  ax = fig.add_subplot(gs[0])
  dur, antic = cfg["dur_s"], cfg["antic_s"]
  for r, m in enumerate(ms):
    y = len(ms) - 1 - r
    c0 = m["clips"][0]
    for p in c0["linea"]:   # cómputo de cada plan (barra clara) y el frame real desde el que parte (tick)
      ax.broken_barh([(p["ini"] / 1000, p["ms"] / 1000)], (y + 0.08, 0.3), facecolors=c_de(m), alpha=0.35,
                     edgecolor="white", linewidth=0.6)
    ed = c0["edad"]   # tramos en pantalla, cortados donde entra un plan nuevo (la edad vuelve a bajar)
    tramos, ini = [], None
    for k, e in enumerate(ed + [None]):
      corte = e is None or (k > 0 and ed[k - 1] is not None and e < ed[k - 1])
      if ini is not None and corte:
        tramos.append((ini, k / 20 - ini)); ini = None
      if e is not None and ini is None:
        ini = k / 20
    ax.broken_barh(tramos, (y - 0.38, 0.3), facecolors=c_de(m), edgecolor="white", linewidth=0.6)
    if not tramos:
      ax.text(0.1, y - 0.23, "sin predicción vigente: cada plan llega tarde", va="center", fontsize=6.5, color=TINTA2)
  ax.set_yticks(range(len(ms)))
  ax.set_yticklabels([m["nombre"] for m in ms][::-1])
  ax.set_xlim(0, dur)
  ax.set_ylim(-0.6, len(ms) - 0.4)
  ax.set_xlabel("tiempo real (s)")
  ax.set_axisbelow(True)
  ax.grid(axis="y", visible=False)
  ax.tick_params(axis="y", length=0)
  ax.set_title(f"(a) Clip {c0['clip']}: cálculo y uso de cada plan", loc="left")
  # (b) margen del frame +antic: cuánto antes de su instante está listo (negativo: tarde)
  for j, (k, tit, esc, xl) in enumerate([("margen_medio", f"(b) Anticipación a +{antic:g} s",
                                           1e-3, "s (negativo: tarde)"),
                                          ("cobertura", "(c) Cobertura", 100, "% del tiempo")]):
    a = fig.add_subplot(gs[1 + j])
    for r, m in enumerate(ms):
      y = len(ms) - 1 - r
      vals = np.array([c[k] for c in m["clips"]]) * esc
      med, q1, q3 = np.median(vals), *np.percentile(vals, [25, 75])
      a.barh(y, med, height=0.55, color=c_de(m), edgecolor="white", linewidth=0.6)
      a.plot([q1, q3], [y, y], color=TINTA, lw=1)
      a.text((q3 if med >= 0 else q1) + (0.12 if k == "margen_medio" else 3) * (1 if med >= 0 else -1), y,
             f"{med:+.2f}".replace(".", ",") if k == "margen_medio" else f"{med:.0f}", fontsize=6.5, color=TINTA,
             ha="left" if med >= 0 else "right", va="center")
    a.axvline(0, color=TINTA2, lw=0.8)
    if k == "margen_medio":
      a.axvline(antic, color=TINTA2, lw=0.8, ls=":")
      mn = min(np.median([c["margen_medio"] for c in m["clips"]]) / 1000 for m in ms)
      a.set_xlim(min(-1.0, mn - 1.2), antic + 0.7)
      a.set_xticks([t for t in (-4, -3, -2, -1, 0, 1, 2) if t >= a.get_xlim()[0]])
    else:
      a.set_xlim(0, 112)
      a.set_xticks([0, 50, 100])
    a.set_yticks(range(len(ms)))
    a.set_yticklabels([])
    a.tick_params(axis="y", length=0)
    a.set_ylim(-0.6, len(ms) - 0.4)
    a.grid(axis="y", visible=False)
    a.set_xlabel(xl)
    a.set_axisbelow(True)
    a.set_title(tit, loc="left")
  guardar(fig, "online")


if __name__ == "__main__":
  fig_online()
  fig_calidad_latencia()
  fig_drift()
  fig_pasos()
  pts = fig_datos_escala()
  tabla_principal()
  tabla_latentes()
  tabla_semillas()
  tabla_cadena()
  print("puntos de escala:", pts)

# Figura de frames generados (necesita la GPU y los pesos; tarda unos minutos):
#   python -m src.rollouts --limit 40 --bs 40 --model <spec> --name <run>_ej [--steps S] --temperature T --dtype fp16
#   python -m src.ejemplos --clips ... --modelos real gpt2m_ej gpt_s_kd_ej maskgit_s_abs_p8_ej genie_s_p4_ej ...
