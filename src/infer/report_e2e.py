"""Tablas y figura de la inferencia de punta a punta: results/infer/summary_e2e.md y fig_pareto_e2e[_<disp>].png.

  python -m src.infer.report_e2e

Lee results/infer/codec_<disp>.json, e2e_*.json y online_*.json, y la calidad de los rollouts de la tabla principal
(results/video_<nombre>.json: FVD a 1 s). Un bloque por dispositivo.

Colores: MaskGIT naranja y los latentes (LeWM) violeta, con la forma del marcador y el rótulo
como segundo canal (validado con el validador de dataviz, --pairs all: ΔE CVD 9,2, normal 27,6). GPT-S queda fuera
de la figura (~170 ms por frame, fuera de escala) y está en la tabla.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.report import GRID, INK, INK2, MUTED, SURF, _repel, pareto_mask

R = Path("results")
INF = R / "infer"
RT_MS = 50.0
COL = {"maskgit": "#eb6834", "lewm": "#6250d6", "baseline": MUTED, "gpt": MUTED, "codec": MUTED}
MARK = {"maskgit": "^", "lewm": "h", "baseline": "X", "gpt": "o", "codec": "|"}
# "copy" es el nombre viejo de la tabla de transición: los JSON de e2e que ya están en disco siguen llamándose así
ORDER = ["codec", "transicion", "copy", "maskgit_s_s1", "maskgit_s_s2", "maskgit_s_s4", "maskgit_s_s8",
         "maskgit_s_s12", "maskgit_m_s1", "maskgit_m_s2", "maskgit_m_s4", "maskgit_m_s8", "maskgit_m_s12", "lewm",
         "gpt_s"]
LABEL = {"codec": "solo codec (encoder + decoder)", "copy": "Transición por celda", "transicion": "Transición por celda",
         "lewm": "LeWM", "gpt_s": "GPT-S"}


def label(name):
  m = re.match(r"maskgit_([sm])_s(\d+)", name)
  return f"MaskGIT-{m[1].upper()} {m[2]} paso{'s' if m[2] != '1' else ''}" if m else LABEL.get(name, name)


def family(name):
  for f in ("maskgit", "lewm", "gpt"):
    if name.startswith(f):
      return f
  return "codec" if name == "codec" else "baseline"


def jload(p):
  return json.loads(p.read_text()) if p.exists() else None


def fvd1(name):
  v = jload(R / f"video_{name}.json")
  if not v:
    return None, None
  f = v["fvd"]["1"]
  return f["value"], (f.get("lo"), f.get("hi"))


def collect():
  """{disp: {"codec": json, "rows": {nombre: {modo: json}}}}"""
  out = {}
  for p in sorted(INF.glob("codec_*.json")):
    d = jload(p)
    out.setdefault(d["tag"], {"codec": None, "rows": {}})["codec"] = d
  for p in sorted(INF.glob("e2e_*.json")):
    d = jload(p)
    key = d["mode"] + (f"_k{d['k']}" if d["mode"] == "online" and d["k"] > 1 else "") \
      + ("_pace" if d["pace_ms"] else "")
    out.setdefault(d["tag"], {"codec": None, "rows": {}})["rows"].setdefault(d["name"], {})[key] = d
  return out


def f(v, nd=1):
  return "–" if v is None else f"{v:.{nd}f}"


def gpu_name(block):
  for d in [block["codec"]] + [m for r in block["rows"].values() for m in r.values()]:
    if d and d.get("device", {}).get("gpu"):
      return d["device"]["gpu"]
  return "CPU"


def table(tag, block):
  lines = [f"### {gpu_name(block)} (`{tag}`)", ""]
  c = block["codec"]
  if c:
    lines += ["Codec solo, batch 1 (ms, mediana; codec = frame en el host → encoder → decoder → host):", "",
              "| Variante | Encoder | Decoder | Codec |", "|---|---:|---:|---:|"]
    for k, v in c["latency"].items():
      if "error" in v:
        lines.append(f"| {k} | falló: {v['error']} | | |")
        continue
      lines.append(f"| {k} | {f(v['enc']['p50'], 2)} | {f(v['dec']['p50'], 2)} | {f(v['codec']['p50'], 2)} |")
    fi = c.get("fidelity")
    if fi:
      lines += ["", f"Fidelidad (fp16 + graphs contra fp32, {fi['n_frames']} frames del video de cámara de comma): "
                f"tokens iguales {fi['tokens_fp16_vs_fp32'] * 100:.2f} % (fp32 contra los de comma: "
                f"{fi['tokens_fp32_vs_comma'] * 100:.2f} %), PSNR del decoder {fi['decoder_psnr_fp16_vs_fp32_db']:.1f} dB, "
                f"LPIPS de la reconstrucción contra el frame de entrada {fi['lpips_recon_fp32_vs_input']:.3f} (fp32) y "
                f"{fi['lpips_recon_fp16_vs_input']:.3f} (fp16), entre las dos {fi['lpips_recon_fp16_vs_fp32']:.4f}."]
    lines.append("")
  lines += ["Cadena completa, batch 1, 3000 frames en bucle cerrado salvo indicación (ms por frame; tarde = más de 50 ms):",
            "", "| Configuración | FVD@1s | En línea p50 | En línea p99 | Imaginación p50 | Imaginación p99 "
            "| A 50 ms p99 | Tarde (%) | Etapas en línea (enc / modelo / dec) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
  names = [n for n in ORDER if n in block["rows"]] + sorted(set(block["rows"]) - set(ORDER))
  for n in names:
    r = block["rows"][n]
    on, im = r.get("online"), r.get("imagine")
    pc = r.get("imagine_pace") or r.get("online_pace")
    v, _ = fvd1(n)
    st = (on or {}).get("stages_p50_ms")
    late = max([x["steady"]["late_pct"] for x in (on, im, pc) if x] or [0])
    lines.append(f"| {label(n)} | {f(v)} | {f(on and on['steady']['p50'])} | {f(on and on['steady']['p99'])} | "
                 f"{f(im and im['steady']['p50'])} | {f(im and im['steady']['p99'])} | "
                 f"{f(pc and pc['steady']['p99'])} | {f(late, 2)} | "
                 + (f"{st['encoder']:.1f} / {st['model']:.1f} / {st['decoder']:.1f}" if st else "–") + " |")
  ks = [(n, k) for n in names for k in ("online_k5", "online_k20") if k in block["rows"][n]]
  if ks:
    lines += ["", "Predicción a k frames en línea (k pasos del modelo y un decode; p50 / p99 en ms):", "",
              "| Configuración | k | p50 | p99 |", "|---|---:|---:|---:|"]
    for n, k in ks:
      d = block["rows"][n][k]
      lines.append(f"| {label(n)} | {d['k']} | {f(d['steady']['p50'])} | {f(d['steady']['p99'])} |")
  first = [(n, r[m]["first_second"]) for n in names for m, r_ in [("online", block["rows"][n])] for r in [r_]
           if m in r and r[m].get("first_second")]
  if first:
    worst = max(first, key=lambda x: x[1]["max"])
    lines += ["", f"Primer segundo (contexto incompleto, sin graphs, y la captura de los graphs al llenarse): el peor "
              f"frame es de {worst[1]['max']:.0f} ms ({label(worst[0])}). No entra en las mediciones de arriba."]
  return lines + [""]


def online_table():
  rows = []
  for p in sorted(INF.glob("online_*.json")):
    d = jload(p)
    if d["frames"] != 400 or d["k"] != 1:
      continue
    rows.append(d)
  if not rows:
    return []
  fr = rows[0]["frozen_metrics"]
  lines = ["### Calidad a 1 paso en línea", "",
           f"El modelo ve los frames reales de val ({rows[0]['segments']} segmentos × {rows[0]['frames']} frames) y en "
           "cada uno predice el siguiente. Referencia: repetir el último frame (frame congelado), acierto "
           f"{fr['token_acc'] * 100:.1f} % y LPIPS {fr['lpips']:.3f}.", "",
           "| Configuración | Temperatura | Acierto de tokens (%) | LPIPS [IC 95%] |", "|---|---:|---:|---:|"]
  key = lambda d: (ORDER.index(d["name"]) if d["name"] in ORDER else 99, -d["temperature"])
  for d in sorted(rows, key=key):
    m = d["model_metrics"]
    lines.append(f"| {label(d['name'])} | {d['temperature']:g} | {m['token_acc'] * 100:.1f} | {m['lpips']:.3f} "
                 f"[{m['lpips_ci'][0]:.3f}, {m['lpips_ci'][1]:.3f}] |")
  chk = [jload(p) for p in sorted(INF.glob("imagine_check_*.json"))]
  for c in [c for c in chk if c]:
    n = c["name"]
    v0, ci0 = fvd1(n)
    v1, ci1 = fvd1(n + "_stream")
    lines += ["", f"Camino nuevo contra la tabla ({label(n)}, 400 clips): tokens iguales "
              f"{c.get('tokens_equal', float('nan')) * 100:.1f} %, primer frame {c.get('first_frame_equal', float('nan')) * 100:.1f} %"
              + (f"; FVD@1s {v1:.1f} [{ci1[0]:.1f}, {ci1[1]:.1f}] contra {v0:.1f} [{ci0[0]:.1f}, {ci0[1]:.1f}]" if v0 and v1 else "")
              + "."]
  return lines + [""]


def figure(tag, block, path):
  from matplotlib.lines import Line2D
  pts = []
  for n, r in block["rows"].items():
    im = r.get("imagine")
    v, ci = fvd1(n)
    if im and v is not None and family(n) != "gpt":
      pts.append(dict(name=n, fam=family(n), x=im["steady"]["p99"], y=v, ci=ci))
  if not pts:
    return
  fig, ax = plt.subplots(figsize=(8, 5.2))
  xmax = max(60, max(p["x"] for p in pts) * 1.1)
  ax.axvspan(RT_MS, xmax, color=GRID, alpha=0.45, lw=0, zorder=0)
  ax.axvline(RT_MS, ls="--", c=INK2, lw=0.9, zorder=1)
  ax.text(RT_MS + 0.8, 0.98, "más lento que\ntiempo real\n(50 ms)", transform=ax.get_xaxis_transform(), va="top",
          ha="left", fontsize=7.5, color=INK2)
  c = block["codec"]
  if c and "fp16_graphs" in c["latency"]:
    dec = c["latency"]["fp16_graphs"]["dec"]["p99"]
    ax.axvline(dec, ls=":", c=MUTED, lw=1, zorder=1)
    ax.text(dec + 0.4, 0.02, f"piso: solo el decoder\n({dec:.1f} ms)", transform=ax.get_xaxis_transform(),
            fontsize=7.5, color=MUTED, va="bottom")
  opt = pareto_mask([p["x"] for p in pts], [p["y"] for p in pts])
  front = sorted((p["x"], p["y"]) for p, o in zip(pts, opt) if o)
  if front:
    ax.step([q[0] for q in front] + [xmax], [q[1] for q in front] + [front[-1][1]], where="post", c=MUTED, lw=1,
            zorder=1)
  for size in ("s", "m"):
    mg = sorted([p for p in pts if p["name"].startswith(f"maskgit_{size}_")], key=lambda p: p["x"])
    if mg:
      ax.plot([p["x"] for p in mg], [p["y"] for p in mg], c=COL["maskgit"], lw=1, alpha=0.6,
              ls="-" if size == "s" else ":", zorder=2)
  items = []
  for p in pts:
    hollow = p["name"].startswith("maskgit_m")
    col = COL[p["fam"]]
    lo, hi = p["ci"] if p["ci"] else (None, None)
    if lo is not None:
      ax.errorbar(p["x"], p["y"], yerr=[[p["y"] - lo], [hi - p["y"]]], fmt="none", ecolor=col, elinewidth=0.8,
                  alpha=0.6, zorder=2)
    ax.plot(p["x"], p["y"], MARK[p["fam"]], ms=7, mfc=SURF if hollow else col, mec=SURF if not hollow else col,
            mew=1.2 if hollow else 0.8, zorder=3)
    m = re.match(r"maskgit_([sm])_s(\d+)", p["name"])
    t = (f"{m[2]}" if m else label(p["name"]))
    items.append(dict(x=p["x"], y=p["y"], t=t, col=INK if m else INK2, size=7 if m else 8, dx=7, ha="left"))
  ax.set_xlim(0, xmax)
  ax.set_ylim(0, max(p["ci"][1] if p["ci"] and p["ci"][1] else p["y"] for p in pts) * 1.08)
  _repel(ax, items, points=[(p["x"], p["y"]) for p in pts])
  leg = [Line2D([], [], marker=MARK[k], ls="", c=COL[k], mec=SURF, ms=7, label=lab) for k, lab in
         (("maskgit", "MaskGIT-S y -M (hueco); n.º = pasos"), ("lewm", "LeWM"),
          ("baseline", "Transición por celda")) if any(p["fam"] == k for p in pts)]
  leg.append(Line2D([], [], c=MUTED, lw=1, label="frontera de Pareto"))
  ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(0.4, 1.0), fontsize=8)
  ax.set_xlabel(f"latencia de punta a punta por frame, modo imaginación (ms, p99, batch 1, {gpu_name(block)})")
  ax.set_ylabel("FVD a 1 s (menor es mejor)")
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def main():
  data = collect()
  md = ["# Inferencia de punta a punta", "",
        "Video en la CPU → encoder VQ → world model → decoder VQ → video en la CPU, todo residente en el dispositivo "
        "(fp16 + CUDA graphs), batch 1. Ver `Docs/plan-inferencia.md`. Lo genera `python -m src.infer.report_e2e`.", "",
        "- **En línea:** llega un frame de cámara, se codifica, entra al estado del modelo y se predice y decodifica "
        "el siguiente.", "- **Imaginación:** el modelo genera a partir de su propia salida; el encoder solo corre "
        "al principio. Es el modo de los rollouts de la tabla principal (FVD).", ""]
  for tag in sorted(data, key=lambda t: (t != "rtx3080ti", t)):
    md += table(tag, data[tag])
    name = "fig_pareto_e2e.png" if tag == "rtx3080ti" else f"fig_pareto_e2e_{tag}.png"
    figure(tag, data[tag], R / name)
    if (R / name).exists():
      md += [f"![Pareto de punta a punta]({'../' + name})", ""]
  md += online_table()
  (INF / "summary_e2e.md").write_text("\n".join(md), encoding="utf-8")
  print(f"{len(data)} dispositivo(s) -> {INF / 'summary_e2e.md'}")


if __name__ == "__main__":
  main()
