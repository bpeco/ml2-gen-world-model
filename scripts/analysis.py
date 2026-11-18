"""Pruebas de las hipótesis H1–H4 del plan con los resultados de results/.

Todas las comparaciones son pareadas (mismos segmentos de val o mismos clips), con IC 95 % por bootstrap
sobre segmentos o clips y, cuando aplica, test de Wilcoxon de rangos con signo.

  H1  destilado vs. de cero, mismo tamaño y tokens: CE por segmento, LPIPS a 1 s por clip, FVD a 1 s.
  H2  MaskGIT con S <= 8 pasos: ¿baja de 50 ms/frame? ¿LPIPS a 1 s cerca del GPT del mismo tamaño?
  H3  fp16 / CUDA graphs / int8: latencia y cambio de calidad (CE, LPIPS, FVD).
  H4  correlación de rangos entre CE y FVD a 1 s (configuraciones autoregresivas).

Uso: python scripts/analysis.py  ->  results/hypotheses.json y results/hypotheses.md
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

R = Path("results")
C = R / "cache"
RT = 50.0


def load(p):
  p = Path(p)
  return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def paired(a, b, reps=5000, seed=0):
  """Diferencia media b − a con IC 95 % bootstrap y p de Wilcoxon (a y b pareados)."""
  a, b = np.asarray(a, float), np.asarray(b, float)
  d = b - a
  rng = np.random.default_rng(seed)
  boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)])
  p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
  return {"diff": float(d.mean()), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
          "p_wilcoxon": p, "n": int(len(d)), "frac_mejor": float((d < 0).mean())}


def ce_seg(name, key="ce_all"):
  d = load(R / f"eval_{name}.json")
  return None if d is None or "per_segment" not in d else d["per_segment"].get(key)


def lpips_clip(name, t=19):
  p = C / f"lpips_{name}.npy"
  return np.load(p)[:, t] if p.exists() else None


def fvd(name, k="1"):
  d = load(R / f"video_{name}.json")
  return None if d is None else d["fvd"][k]


def lpips(name, k="1"):
  d = load(R / f"video_{name}.json")
  return None if d is None else d["lpips"]["at"][k]


def bench(name):
  d = load(R / "bench" / f"{name}.json")
  return None if d is None else d["median_ms"]


def h1():
  out = {}
  for size in ("s", "m"):
    sc, kd = f"gpt_{size}", f"gpt_{size}_kd"
    r = {}
    for key in ("ce_all", "ce_last"):
      a, b = ce_seg(sc, key), ce_seg(kd, key)
      if a and b:
        r[key] = paired(a, b)
    a, b = lpips_clip(sc), lpips_clip(kd)
    if a is not None and b is not None:
      r["lpips_1s"] = paired(a, b)
    r["fvd_1s"] = {"scratch": fvd(sc), "kd": fvd(kd)}
    tr = {k: load(R / f"train_{k}.json") for k in (sc, kd)}
    r["monitor_ce"] = {k: (v or {}).get("final_monitor_ce") for k, v in tr.items()}
    out[size] = r
  return out


def h2():
  out = {}
  for size in ("s", "m"):
    gpt_ms = bench(f"gpt_{size}__fp16_graphs")
    g_lp = lpips_clip(f"gpt_{size}")
    rows = []
    for s in (1, 2, 4, 8):
      name = f"maskgit_{size}_s{s}"
      ms = bench(f"maskgit_{size}__fp16_graphs_s{s}")
      row = {"steps": s, "ms_frame": ms, "realtime": ms is not None and ms < RT, "lpips_1s": lpips(name),
             "fvd_1s": fvd(name), "fvd_3s": fvd(name, "3")}
      m_lp = lpips_clip(name)
      if g_lp is not None and m_lp is not None:
        row["lpips_1s_vs_gpt"] = paired(g_lp, m_lp)  # > 0: MaskGIT peor
      rows.append(row)
    out[size] = {"gpt_ms_frame": gpt_ms, "gpt_lpips_1s": lpips(f"gpt_{size}"), "gpt_fvd_1s": fvd(f"gpt_{size}"),
                 "maskgit": rows}
  return out


def h3():
  out = {"ce": {}, "video": {}, "latencia": {}}
  for n in ("gpt2m", "gpt_s", "maskgit_s"):
    a, b = load(R / f"eval_{n}.json"), load(R / f"eval_{n}_fp16.json")
    if a and b:
      key = "ce_all" if a.get("ce_all") is not None else "ce_last"
      out["ce"][n] = {"metrica": key, "fp32": a[key], "fp16": b[key], "diff": b[key] - a[key]}
  for fp16, fp32 in (("gpt_s", "gpt_s_fp32"), ("maskgit_s_s8", "maskgit_s_s8_fp32")):
    a, b = lpips_clip(fp32), lpips_clip(fp16)
    if a is not None and b is not None:
      out["video"][fp16] = {"lpips_1s_fp16_menos_fp32": paired(a, b), "fvd_1s": {"fp32": fvd(fp32), "fp16": fvd(fp16)}}
  variants = {"gpt2m": ["fp32", "fp16", "fp16_graphs"], "gpt_s": ["fp32", "fp16", "fp16_graphs"],
              "gpt_m": ["fp32", "fp16", "fp16_graphs"], "random_gpt_l": ["fp32", "fp16", "fp16_graphs"],
              "maskgit_s": ["fp32_s8", "fp16_s8", "fp16_graphs_s8"], "maskgit_m": ["fp32_s8", "fp16_s8", "fp16_graphs_s8"]}
  for n, vs in variants.items():
    out["latencia"][n] = {v: bench(f"{n}__{v}") for v in vs}
  # nombre de variante de src/bench.py: dtype + _int8 + _s<pasos> + _cpu
  out["int8_cpu"] = {n: {"fp32": bench(f"{n}__fp32{st}_cpu"), "int8": bench(f"{n}__fp32_int8{st}_cpu")}
                     for n, st in (("gpt_s", ""), ("maskgit_s", "_s4"))}
  return out


def h4():
  cfgs = load(R / "configs.json") or []
  pts = []
  for c in cfgs:
    if not c.get("autoregressive") or not c.get("eval") or not c.get("video"):
      continue
    e, v = load(c["eval"]), load(c["video"])
    if e and v and e.get("ce_all") is not None:
      pts.append((c["label"], e["ce_all"], v["fvd"]["1"]["value"], v["lpips"]["at"]["1"]["mean"]))
  if len(pts) < 3:
    return {"n": len(pts)}
  ce, fv, lp = (np.array([p[i] for p in pts]) for i in (1, 2, 3))
  r1, r2 = spearmanr(ce, fv), spearmanr(ce, lp)
  return {"n": len(pts), "puntos": pts, "spearman_ce_fvd1s": {"rho": float(r1.statistic), "p": float(r1.pvalue)},
          "spearman_ce_lpips1s": {"rho": float(r2.statistic), "p": float(r2.pvalue)}}


def fmt(x, nd=3):
  return "–" if x is None else f"{x:.{nd}f}"


def md(res):
  L = ["# Hipótesis", ""]
  L += ["## H1: destilado vs. de cero (mismo tamaño, datos, semilla y 100M tokens)", "",
        "| Tamaño | Métrica | De cero | Destilado − de cero [IC 95 %] | p (Wilcoxon) |", "|---|---|---:|---:|---:|"]
  for size, r in res["H1"].items():
    for key, lab in (("ce_all", "CE val"), ("ce_last", "CE último frame"), ("lpips_1s", "LPIPS 1 s")):
      if key in r:
        x = r[key]
        base = np.mean(ce_seg(f"gpt_{size}", key)) if key != "lpips_1s" else lpips(f"gpt_{size}")["mean"]
        L.append(f"| {size.upper()} | {lab} | {base:.3f} | {x['diff']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}] | {x['p_wilcoxon']:.2g} |")
    f = r.get("fvd_1s", {})
    if f.get("scratch") and f.get("kd"):
      L.append(f"| {size.upper()} | FVD 1 s | {f['scratch']['value']:.0f} [{f['scratch']['lo']:.0f}, {f['scratch']['hi']:.0f}] "
               f"| destilado {f['kd']['value']:.0f} [{f['kd']['lo']:.0f}, {f['kd']['hi']:.0f}] | |")
  L += ["", "## H2: MaskGIT con pocos pasos", ""]
  for size, r in res["H2"].items():
    L += [f"Tamaño {size.upper()}: GPT {fmt(r['gpt_ms_frame'], 1)} ms/frame, LPIPS 1 s "
          f"{fmt((r['gpt_lpips_1s'] or {}).get('mean'))}, FVD 1 s {fmt((r['gpt_fvd_1s'] or {}).get('value'), 0)}.", "",
          "| Pasos | ms/frame | < 50 ms | LPIPS 1 s | MaskGIT − GPT [IC 95 %] | FVD 1 s | FVD 3 s |", "|---:|---:|:-:|---:|---:|---:|---:|"]
    for row in r["maskgit"]:
      d = row.get("lpips_1s_vs_gpt")
      L.append(f"| {row['steps']} | {fmt(row['ms_frame'], 1)} | {'sí' if row['realtime'] else 'no'} | "
               f"{fmt((row['lpips_1s'] or {}).get('mean'))} | "
               + (f"{d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]" if d else "–")
               + f" | {fmt((row['fvd_1s'] or {}).get('value'), 0)} | {fmt((row['fvd_3s'] or {}).get('value'), 0)} |")
    L.append("")
  L += ["## H3: precisión y CUDA graphs", "", "| Modelo | Métrica | fp32 | fp16 | Diferencia |", "|---|---|---:|---:|---:|"]
  for n, x in res["H3"]["ce"].items():
    L.append(f"| {n} | {x['metrica']} | {x['fp32']:.5f} | {x['fp16']:.5f} | {x['diff']:+.2e} |")
  for n, x in res["H3"]["video"].items():
    d = x["lpips_1s_fp16_menos_fp32"]
    L.append(f"| {n} | LPIPS 1 s (fp16 − fp32) | | | {d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] |")
    L.append(f"| {n} | FVD 1 s | {x['fvd_1s']['fp32']['value']:.0f} | {x['fvd_1s']['fp16']['value']:.0f} | |")
  L += ["", "| Modelo | " + " | ".join(["fp32", "fp16", "fp16 + graphs"]) + " | Aceleración total |", "|---|---:|---:|---:|---:|"]
  for n, v in res["H3"]["latencia"].items():
    ms = list(v.values())
    acc = f"{ms[0] / min(m for m in ms if m):.1f}×" if ms[0] and any(ms) else "–"
    L.append(f"| {n} | " + " | ".join(fmt(m, 1) for m in ms) + (" | –" if len(ms) == 2 else "") + f" | {acc} |")
  L += ["", "int8 dinámico en CPU (ms/frame):", ""]
  for n, v in res["H3"]["int8_cpu"].items():
    L.append(f"- {n}: fp32 {fmt(v['fp32'], 0)} · int8 {fmt(v['int8'], 0)}")
  h = res["H4"]
  L += ["", "## H4: CE vs. FVD", ""]
  if "spearman_ce_fvd1s" in h:
    pf = lambda p: "p < 0,001" if p < 1e-3 else f"p = {p:.3f}"
    L.append(f"Spearman entre CE y FVD a 1 s: ρ = {h['spearman_ce_fvd1s']['rho']:.2f} ({pf(h['spearman_ce_fvd1s']['p'])}, "
             f"n = {h['n']}). Entre CE y LPIPS a 1 s: ρ = {h['spearman_ce_lpips1s']['rho']:.2f} ({pf(h['spearman_ce_lpips1s']['p'])}).")
  return "\n".join(L) + "\n"


def main(argv=None):
  argparse.ArgumentParser(description="Pruebas de H1-H4 con los resultados de results/").parse_args(argv)
  res = {"H1": h1(), "H2": h2(), "H3": h3(), "H4": h4()}
  (R / "hypotheses.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
  (R / "hypotheses.md").write_text(md(res), encoding="utf-8")
  sys.stdout.reconfigure(encoding="utf-8")  # la consola de Windows es cp1252 y no tiene "−"
  print(md(res))
  return res


if __name__ == "__main__":
  main()
