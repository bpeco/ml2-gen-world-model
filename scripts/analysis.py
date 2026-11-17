"""Pruebas de las hipótesis H1–H7 del plan con los resultados de results/.

Todas las comparaciones son pareadas (mismos segmentos de val o mismos clips), con IC 95 % por bootstrap
sobre segmentos o clips y, cuando aplica, test de Wilcoxon de rangos con signo.

  H1  destilado vs. de cero, mismo tamaño y tokens: CE por segmento, LPIPS a 1 s por clip, FVD a 1 s.
  H2  MaskGIT con S <= 8 pasos: ¿baja de 50 ms/frame? ¿LPIPS a 1 s cerca del GPT del mismo tamaño?
  H3  fp16 / CUDA graphs / int8: latencia y cambio de calidad (CE, LPIPS, FVD).
  H4  correlación de rangos entre CE y FVD a 1 s (configuraciones autoregresivas).
  H5  latente vs. tokens: jepa_s y lewm_s contra su control CE (misma arquitectura), y rollout latente vs. token.
  H6  acciones: error latente con pose real vs. acción nula, y planificación con CEM (results/latent_*.json).
  H7  2×2 granularidad (celda / vector) × antiaplastamiento (EMA / SIGReg), y barrido de λ de LeWM.

Uso: python scripts/analysis.py  ->  results/hypotheses.json y results/hypotheses.md
"""
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
              "maskgit_s": ["fp32_s8", "fp16_s8", "fp16_graphs_s8"], "maskgit_m": ["fp32_s8", "fp16_s8", "fp16_graphs_s8"],
              "convlstm": ["fp32", "fp16"]}
  for n, vs in variants.items():
    out["latencia"][n] = {v: bench(f"{n}__{v}") for v in vs}
  # nombre de variante de src/bench.py: dtype + _int8 + _s<pasos> + _cpu
  out["int8_cpu"] = {n: {"fp32": bench(f"{n}__fp32{st}_cpu"), "int8": bench(f"{n}__fp32_int8{st}_cpu")}
                     for n, st in (("gpt_s", ""), ("convlstm", ""), ("maskgit_s", "_s4"))}
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


LATENT = {"jepa_s": "JEPA-S (V-JEPA: celda + EMA)", "jepa_s_sigreg": "JEPA-S con SIGReg (celda + SIGReg)",
          "jepa_s_codebook": "JEPA-S, target codebook", "jepa_s_ce": "JEPA-S, control CE",
          "lewm_s": "LeWM (vector + SIGReg)", "lewm_s_ema": "LeWM con EMA (vector + EMA)", "lewm_s_ce": "LeWM, control CE",
          "lewm_s_l003": "LeWM, λ = 0,03", "lewm_s_l03": "LeWM, λ = 0,3"}


def last_log(run, key):
  """Último valor de una métrica en checkpoints/<run>/log.jsonl (p. ej. z_std al final del entrenamiento)."""
  p = Path("checkpoints") / run / "log.jsonl"
  if not p.exists():
    return None
  vals = [json.loads(l).get(key) for l in p.read_text().splitlines() if l.strip()]
  vals = [v for v in vals if v is not None]
  return vals[-1] if vals else None


def quality(name, video=None, bench_name=None):
  """Calidad de una corrida: CE (val), LPIPS y FVD de su rollout, latencia y z_std final."""
  video = video or name
  e, tr = load(R / f"eval_{name}.json") or {}, load(R / f"train_{name}.json") or {}
  return {"ce_all": e.get("ce_all"), "ce_last": e.get("ce_last"), "lpips_1s": lpips(video), "fvd_1s": fvd(video),
          "fvd_3s": fvd(video, "3"), "ms_frame": bench(bench_name or f"{name}__fp16_graphs"), "params": tr.get("params"),
          "monitor_ce": tr.get("final_monitor_ce"), "z_std": last_log(name, "z_std")}


def h5():
  """Latente vs. tokens con la misma arquitectura: cada familia contra su control CE, y contra MaskGIT-S 1 paso."""
  out = {}
  for fam, ctrl in (("jepa_s", "jepa_s_ce"), ("lewm_s", "lewm_s_ce")):
    r = {"modelo": quality(fam), "control_ce": quality(ctrl, bench_name=f"{fam}__fp16_graphs"),
         "maskgit_s_1paso": quality("maskgit_s", "maskgit_s_s1", "maskgit_s__fp16_graphs_s1")}
    for key in ("ce_all", "ce_last"):
      a, b = ce_seg(ctrl, key), ce_seg(fam, key)
      if a and b:
        r[f"{key}_vs_control"] = paired(a, b)  # > 0: el latente es peor
    a, b = lpips_clip(ctrl), lpips_clip(fam)
    if a is not None and b is not None:
      r["lpips_1s_vs_control"] = paired(a, b)
    a = lpips_clip("maskgit_s_s1")
    if a is not None and b is not None:
      r["lpips_1s_vs_maskgit_1paso"] = paired(a, b)
    lat = lpips_clip(f"{fam}_latent")
    if lat is not None and b is not None:
      r["rollout_latente"] = {"lpips_1s": lpips(f"{fam}_latent"), "fvd_1s": fvd(f"{fam}_latent"),
                              "fvd_3s": fvd(f"{fam}_latent", "3"), "lpips_1s_vs_token": paired(b, lat),
                              "ms_frame": bench(f"{fam}_latent__fp16_graphs")}
    out[fam] = r
  return out


def h6():
  """Acciones: error latente con la pose real vs. con la acción nula; planificación con CEM vs. referencias."""
  out = {}
  for run in LATENT:
    d = load(R / f"latent_{run}.json")
    if not d or "acciones" not in d:
      continue
    a = d["acciones"]
    out[run] = {"error_pose_real": a["error_pose_real"], "error_accion_nula": a["error_accion_nula"],
                "nula_menos_real": paired(a["per_window_real"], a["per_window_nula"]), "plan": d.get("plan")}
  return out


def probe_r(probes, feat):
  p = (probes or {}).get(feat)
  if not p:
    return None
  return {**{d: p["lineal"][d]["pearson"] for d in ("velocidad", "giro")},
          **{f"{d}_mlp": p["mlp"][d]["pearson"] for d in ("velocidad", "giro")}}


def latent_summary(run):
  d = load(R / f"latent_{run}.json") or {}
  s = d.get("sorpresa", {})
  return {"probe_pred": probe_r(d.get("probes"), "pred"), "probe_obs": probe_r(d.get("probes"), "obs"),
          "sorpresa_tele": (s.get("teletransporte") or {}).get("ratio"),
          "sorpresa_tele_p": (s.get("teletransporte") or {}).get("p"),
          "sorpresa_salto": (s.get("salto_2s") or {}).get("ratio"),
          "sorpresa_salto_p": (s.get("salto_2s") or {}).get("p"), "rectitud": d.get("rectitud"),
          "plan_cem": (d.get("plan") or {}).get("cem")}


def h7():
  """El 2×2: granularidad del latente (celda / vector) x antiaplastamiento (EMA / SIGReg), y el barrido de λ."""
  cells = {"celda|EMA": "jepa_s", "celda|SIGReg": "jepa_s_sigreg", "vector|EMA": "lewm_s_ema", "vector|SIGReg": "lewm_s"}
  out = {"2x2": {k: {"run": r, **quality(r), **latent_summary(r)} for k, r in cells.items()},
         "lambda": {lam: {"run": r, **quality(r, bench_name="lewm_s__fp16_graphs"), **latent_summary(r)}
                    for lam, r in (("0.03", "lewm_s_l003"), ("0.09", "lewm_s"), ("0.3", "lewm_s_l03"))},
         "extras": {r: {**quality(r), **latent_summary(r)} for r in ("jepa_s_codebook", "jepa_s_ce", "lewm_s_ce")}}
  refs = {r: (load(R / f"latent_{r}.json") or {}).get("probes") for r in ("gpt_s", "maskgit_s")}
  out["probe_referencias"] = {r: probe_r(p, "pred") for r, p in refs.items() if p}
  return out


def fmt(x, nd=3):
  return "–" if x is None else f"{x:.{nd}f}"


def _v(d, k="value", nd=0):
  return fmt((d or {}).get(k) if isinstance(d, dict) else d, nd)


def _diff(x):
  return "–" if not x else f"{x['diff']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}], p = {x['p_wilcoxon']:.2g}"


def md_latent(res):
  L = ["", "## H5: latente vs. tokens (misma arquitectura, solo cambia la pérdida)", "",
       "| Modelo | CE val | CE último frame | LPIPS 1 s | FVD 1 s | FVD 3 s | ms/frame | Params (M) |",
       "|---|---:|---:|---:|---:|---:|---:|---:|"]
  for fam, r in res["H5"].items():
    for key, lab in (("modelo", LATENT[fam]), ("control_ce", LATENT[f"{fam}_ce"]), ("maskgit_s_1paso", "MaskGIT-S 1 paso")):
      q = r[key]
      if key == "maskgit_s_1paso" and fam != "jepa_s":
        continue
      L.append(f"| {lab} | {fmt(q['ce_all'])} | {fmt(q['ce_last'])} | {_v(q['lpips_1s'], 'mean', 3)} | {_v(q['fvd_1s'])} | "
               f"{_v(q['fvd_3s'])} | {fmt(q['ms_frame'], 1)} | {fmt((q['params'] or 0) / 1e6, 1)} |")
  L += ["", "Diferencias pareadas (latente − control CE; > 0 = el latente es peor):", ""]
  for fam, r in res["H5"].items():
    L.append(f"- {LATENT[fam]}: CE val {_diff(r.get('ce_all_vs_control'))}; CE último frame "
             f"{_diff(r.get('ce_last_vs_control'))}; LPIPS 1 s {_diff(r.get('lpips_1s_vs_control'))}.")
    if r.get("lpips_1s_vs_maskgit_1paso"):
      L.append(f"  - contra MaskGIT-S 1 paso: LPIPS 1 s {_diff(r['lpips_1s_vs_maskgit_1paso'])}.")
    rl = r.get("rollout_latente")
    if rl:
      L.append(f"  - rollout latente (ẑ al contexto, sin re-codificar): LPIPS 1 s {_v(rl['lpips_1s'], 'mean', 3)}, "
               f"FVD 1 s {_v(rl['fvd_1s'])}, FVD 3 s {_v(rl['fvd_3s'])}, {fmt(rl['ms_frame'], 1)} ms/frame; "
               f"latente − token: {_diff(rl['lpips_1s_vs_token'])}.")
  L += ["", "## H6: acciones (holdout de train separado por ruta, poses reales)", "",
        "| Modelo | Error latente con pose real | con acción nula | nula − real [IC 95 %] | CEM: velocidad (m/s) | CEM: giro (rad/s) | ms por plan |",
        "|---|---:|---:|---:|---:|---:|---:|"]
  ref = None
  for run, r in res["H6"].items():
    pl = r.get("plan") or {}
    ref = ref or pl
    d = r["nula_menos_real"]
    L.append(f"| {LATENT[run]} | {r['error_pose_real']:.4f} | {r['error_accion_nula']:.4f} | "
             f"{d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] | {fmt((pl.get('cem') or {}).get('velocidad'), 2)} | "
             f"{fmt((pl.get('cem') or {}).get('giro'), 4)} | {fmt(pl.get('ms_por_plan'), 0)} |")
  if ref:
    L.append(f"| Referencia: repetir la última pose | | | | {fmt(ref['ultima_pose']['velocidad'], 2)} | "
             f"{fmt(ref['ultima_pose']['giro'], 4)} | |")
    L.append(f"| Referencia: pose media de train | | | | {fmt(ref['pose_media']['velocidad'], 2)} | "
             f"{fmt(ref['pose_media']['giro'], 4)} | |")
  h = res["H7"]
  L += ["", "## H7: granularidad del latente × antiaplastamiento", "",
        "| Latente | Antiaplastamiento | Corrida | CE último frame | LPIPS 1 s | FVD 1 s | z_std final | Probe ẑ: r velocidad | r giro | Sorpresa teletransporte | Sorpresa salto 2 s | Rectitud |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
  for k, r in h["2x2"].items():
    g, a = k.split("|")
    pp = r["probe_pred"] or {}
    L.append(f"| {g} | {a} | {r['run']} | {fmt(r['ce_last'])} | {_v(r['lpips_1s'], 'mean', 3)} | {_v(r['fvd_1s'])} | "
             f"{fmt(r['z_std'], 2)} | {fmt(pp.get('velocidad'), 2)} | {fmt(pp.get('giro'), 2)} | "
             f"{fmt(r['sorpresa_tele'], 2)}× | {fmt(r['sorpresa_salto'], 2)}× | {fmt(r['rectitud'], 2)} |")
  for run, pr in h["probe_referencias"].items():
    L.append(f"| tokens | — | {run} | | | | | {fmt(pr.get('velocidad'), 2)} | {fmt(pr.get('giro'), 2)} | | | |")
  L += ["", "Barrido de λ (LeWM, 100M tokens):", "", "| λ | CE último frame | LPIPS 1 s | FVD 1 s | z_std final | Probe ẑ: r velocidad | Sorpresa teletransporte |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
  for lam, r in h["lambda"].items():
    L.append(f"| {lam} | {fmt(r['ce_last'])} | {_v(r['lpips_1s'], 'mean', 3)} | {_v(r['fvd_1s'])} | {fmt(r['z_std'], 2)} | "
             f"{fmt((r['probe_pred'] or {}).get('velocidad'), 2)} | {fmt(r['sorpresa_tele'], 2)}× |")
  return L


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
  if "H5" in res:
    L += md_latent(res)
  return "\n".join(L) + "\n"


def main():
  res = {"H1": h1(), "H2": h2(), "H3": h3(), "H4": h4(), "H5": h5(), "H6": h6(), "H7": h7()}
  (R / "hypotheses.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
  (R / "hypotheses.md").write_text(md(res), encoding="utf-8")
  sys.stdout.reconfigure(encoding="utf-8")  # la consola de Windows es cp1252 y no tiene "−"
  print(md(res))


if __name__ == "__main__":
  main()
