"""Datos de las slides, leídos de los JSON vigentes de results/ (no mide nada).

  python presentacion/generar_datos.py        # -> presentacion/datos.js

Cada número que muestran los gráficos y los textos con data-d sale de acá. Se toman los mismos archivos que
paper/hacer_figuras.py, así que las slides y las tablas del informe coinciden. Si cambian los resultados o el informe,
se corre de nuevo y se recarga index.html.
"""
import json
from pathlib import Path

R = Path("results")
OUT = Path("presentacion/datos.js")

# Datos del dataset y del presupuesto (README de commaVQ, Docs/resumen.md y paper/hacer_figuras.py)
FPS = 20
FRAMES_PRESUPUESTO = 780_000
FRAMES_SHARD0 = 2_500 * 60 * FPS
FRAMES_COMMAVQ = 100_000 * 60 * FPS
FRAMES_GPT2M = 3_000_000 * 60 * FPS
CODES = 128


def leer(p):
  p = Path(p)
  return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


SUM = {c["label"]: c for c in leer(R / "summary.json")["configs"]}


def video(n):
  return leer(R / f"video_{n}.json")


def fvd(n):
  """FVD a 1, 2 y 3 s con su IC 95 % por bootstrap por segmento."""
  v = video(n)
  if v is None:
    return None
  return {h: {"v": v["fvd"][h]["value"], "lo": v["fvd"][h]["lo"], "hi": v["fvd"][h]["hi"]}
          for h in ("1", "2", "3")}


def bench_ms(nombre):
  """p50 de results/bench/<nombre>__*.json (fp16 + CUDA graphs si existe), igual que paper/hacer_figuras.py."""
  cands = sorted((R / "bench").glob(f"{nombre}__*.json"))
  cands = [c for c in cands if "graphs" in c.name] or cands
  if not cands:
    return None
  b = leer(cands[0])
  return b["p50_ms"]


def fila(clave, nombre, run_video, bench_4060=None, familia=""):
  c = SUM.get(clave, {})
  f = fvd(run_video)
  if f is None:
    return None
  return {"clave": run_video, "nombre": nombre, "familia": familia, "params": c.get("params_M"),
          "ce": c.get("ce_all"), "lpips1": c.get("lpips_1s"), "fvd": f,
          "drift": f["3"]["v"] / f["1"]["v"], "ms3080": c.get("ms_frame"),
          "ms4060": bench_ms(bench_4060) if bench_4060 else None}


def tabla():
  """Las filas de la tabla principal del informe (paper/tablas/principal.tex)."""
  filas = [fila("gpt2m", "gpt2m (comma)", "gpt2m", "gpt2m_4060", "gpt2m"),
           fila("gpt_s_kd", "GPT-S destilado", "gpt_s_kd", "gpt_s_kd_4060", "gpt"),
           fila("gpt_s_d050", "GPT-S de cero, 50 %", "gpt_s_d050", None, "gpt"),
           fila("gpt_s_d025", "GPT-S de cero, 25 %", "gpt_s_d025", None, "gpt")]
  filas += [fila(f"maskgit_s_abs_p{s}", f"MaskGIT-S, {s} paso{'s' * (s > 1)}", f"maskgit_s_abs_p{s}",
                 f"maskgit_s_abs_p{s}", "maskgit") for s in (1, 2, 4, 8, 12)]
  filas += [fila(f"genie_s_p{s}", f"Genie-S, {s} paso{'s' * (s > 1)}", f"genie_s_p{s}", f"genie_s_4060_p{s}", "genie")
            for s in (1, 2, 4, 8, 12, 25)]
  filas += [fila(f"genie_m_p{s}", f"Genie-M, {s} pasos", f"genie_m_p{s}", None, "genie_m") for s in (4, 8)]
  filas += [fila("convlstm_px", "ConvLSTM (píxeles)", "convlstm_px", None, "convlstm"),
            fila("Transición por celda", "Transición por celda", "transicion", None, "base"),
            fila("Frame congelado", "Frame congelado", "congelado", None, "base"),
            fila("Unigrama", "Unigrama", "unigram", None, "base")]
  return [f for f in filas if f]


def pareto():
  """FVD a 3 s contra ms por frame (modelo solo, p50), todos en la misma RTX 4060 Laptop (figura del informe)."""
  def pt(bench, run, pasos=None):
    ms, f = bench_ms(bench), fvd(run)
    return None if ms is None or f is None else {"ms": ms, "fvd3": f["3"]["v"], "pasos": pasos}
  return {"gpt2m": [pt("gpt2m_4060", "gpt2m")],
          "gpt": [pt("gpt_s_kd_4060", "gpt_s_kd")],
          "maskgit": [pt(f"maskgit_s_abs_p{s}", f"maskgit_s_abs_p{s}", s) for s in (1, 2, 4, 8, 12)],
          "genie": [pt(f"genie_s_4060_p{s}", f"genie_s_p{s}", s) for s in (1, 2, 4, 8, 12)]}


def pasos():
  """FVD a 1 y 3 s contra pasos de decodificación paralela (figura pasos del informe)."""
  def serie(base, ss):
    out = []
    for s in ss:
      f = fvd(f"{base}_p{s}")
      if f:
        out.append({"pasos": s, "1": f["1"], "3": f["3"]})
    return out
  return {"genie": serie("genie_s", (1, 2, 4, 8, 12, 25)), "genie_m": serie("genie_m", (1, 2, 4, 8, 12, 25)),
          "maskgit": serie("maskgit_s_abs", (1, 2, 4, 8, 12)), "gpt": fvd("gpt_s_kd")}


def horizonte():
  """FVD a 1, 2 y 3 s (figura drift del informe)."""
  runs = [("gpt2m", "gpt2m (comma)", "gpt2m"), ("gpt_s_kd", "GPT-S destilado", "gpt"),
          ("maskgit_s_abs_p8", "MaskGIT-S, 8 pasos", "maskgit"), ("genie_s_p4", "Genie-S, 4 pasos", "genie"),
          ("genie_s_p1", "Genie-S, 1 paso", "genie"), ("congelado", "Frame congelado", "base")]
  return [{"clave": n, "nombre": et, "familia": fam, "fvd": fvd(n)} for n, et, fam in runs if fvd(n)]


def cadena():
  """Cadena completa en la 3080 Ti: p50 por etapa con la cadena cerrada y p99 a ritmo de cámara (tabla cadena)."""
  out = {}
  for f in sorted((R / "infer").glob("latencia_*__rtx3080ti.json")):
    d = leer(f)
    st = d["cadena_cerrada"].get("stages_p50_ms") or {}
    out[d["name"]] = {"encoder": round(st.get("encoder", 0), 2), "modelo": round(st.get("model", 0), 2),
                      "decoder": round(st.get("decoder", 0), 2), "p50": round(d["cadena_cerrada"]["p50"], 2),
                      "p99_ritmo": round(d["cadena_ritmo"]["p99"], 2), "tarde": d["cadena_ritmo"].get("late_pct")}
  return out


def latentes():
  """R² de los probes lineales de la pose (tabla latentes del informe)."""
  out, pix = [], None
  for n, et in (("lewm", "LeWM"), ("lewm_lrx3", "LeWM, lr x3"), ("vjepa2_s", "V-JEPA 2-S"), ("vjepa2_ac", "V-JEPA 2-AC")):
    d = leer(R / f"latent_{n}.json")
    if not d:
      continue
    o = d["probes"]["obs"]
    r = lambda s, t: round(o[s][t]["r2"]["valor"], 3)
    pix = pix or o["pixeles"]
    out.append({"nombre": et, "params": round(d["params"] / 1e6, 1) if isinstance(d.get("params"), (int, float)) else None,
                "vel": r("modelo", "velocidad"), "vel_azar": r("sin_entrenar", "velocidad"),
                "giro": r("modelo", "giro"), "giro_azar": r("sin_entrenar", "giro")})
  return {"modelos": out, "pixeles": {"vel": round(pix["velocidad"]["r2"]["valor"], 3),
                                       "giro": round(pix["giro"]["r2"]["valor"], 3)}}


def diferencias():
  """Diferencias pareadas de FVD (bootstrap por segmento, 200 remuestreos): a - b, negativo = a es mejor."""
  out = {}
  for f in sorted(R.glob("diferencia_*.json")):
    d = leer(f)
    out[f"{d['a']}__{d['b']}"] = {h: {"a": round(x["fvd_a"], 3), "b": round(x["fvd_b"], 3), "diff": round(x["diff"], 3),
                                      "lo": round(x["lo"], 3), "hi": round(x["hi"], 3), "p": round(x["p"], 3)}
                                  for h, x in d["fvd"].items()}
  return out


def semillas():
  import statistics as st
  out = {}
  for n, runs in (("genie_s_p8", ("genie_s_p8", "genie_s_s1_p8", "genie_s_s2_p8")),
                  ("convlstm_px", ("convlstm_px", "convlstm_px_s1", "convlstm_px_s2"))):
    out[n] = {h: {"media": round(st.mean(video(r)["fvd"][h]["value"] for r in runs), 1),
                  "desvio": round(st.stdev(video(r)["fvd"][h]["value"] for r in runs), 1)} for h in ("1", "2", "3")}
  return out


def piso():
  """FVD real contra real: 400 contra 400 clips (eval contra eval_b) y dos mitades disjuntas por segmento."""
  d = leer(R / "piso_fvd.json")
  return {"real_400": round(d["piso"]["1"]["value"], 1), "mitades": round(d["mitades"]["1"]["value"], 1),
          "gpt2m_mitades": round(d["mitades"]["1"]["modelo_mitades"], 1) if "modelo_mitades" in d["mitades"]["1"]
          else round(video("gpt2m")["piso"]["1"]["modelo_mitades"], 1)}


def temperaturas():
  out = {}
  for n in ("genie_s", "gpt_s_kd", "maskgit_s_abs"):
    d = leer(R / f"temperatura_{n}.json")
    out[n] = {"elegida": d["elegida"], "default": d["default"],
              "tabla": {str(t["T"]): round(t["fvd"], 1) for t in d["tabla"]}}
  return out


def escala():
  """Puntos del GPT-S de cero y el ajuste L(D) = E + B·D^-β con E fijo en la CE de gpt2m (informe, sección 5.4)."""
  import math
  pts = []
  for run in ("gpt_s_d025", "gpt_s_d050", "gpt_s"):
    t, e = leer(R / f"train_{run}.json"), leer(R / f"eval_{run}.json")
    if t and e and run in SUM and SUM[run].get("ce_all"):
      pts.append({"run": run, "tokens": t["tokens"], "ce": SUM[run]["ce_all"]})
  E = SUM["gpt2m"]["ce_all"]
  (D0, L0), (D1, L1) = [(p["tokens"], p["ce"]) for p in pts[:2]]
  b = math.log((L0 - E) / (L1 - E)) / math.log(D1 / D0)
  B = (L1 - E) * (D1 / 1e8) ** b
  tok_kd = leer(R / "train_gpt_s_kd.json")["tokens"]
  params_s = SUM["gpt_s_kd"]["params_M"] * 1e6
  pred = lambda D: round(E + B * (D / 1e8) ** (-b), 2)
  return {"puntos": pts, "E": E, "beta": round(b, 2), "tokens_presupuesto": tok_kd,
          "pred_1x": pred(tok_kd), "pred_chinchilla": pred(20 * params_s), "pred_10x": pred(10 * tok_kd),
          "tokens_por_param": round(tok_kd / params_s, 1),
          "tokens_por_param_genie_m": round(tok_kd / (SUM["genie_m"]["params_M"] * 1e6), 1),
          "tokens_por_param_gpt2m": round(FRAMES_GPT2M * (CODES + 1) / (SUM["gpt2m"]["params_M"] * 1e6))}


def datos():
  return {"presupuesto": FRAMES_PRESUPUESTO, "shard0": FRAMES_SHARD0, "commavq": FRAMES_COMMAVQ, "gpt2m": FRAMES_GPT2M,
          "pct_shard": round(100 * FRAMES_PRESUPUESTO / FRAMES_SHARD0, 1),
          "pct_commavq": round(100 * FRAMES_PRESUPUESTO / FRAMES_COMMAVQ, 2),
          "pct_gpt2m": round(100 * FRAMES_PRESUPUESTO / FRAMES_GPT2M, 3),
          "veces_gpt2m": round(FRAMES_GPT2M / FRAMES_PRESUPUESTO, -2)}


def carrera():
  """ms por frame del modelo solo (p50) para la carrera del video: 3080 Ti; MaskGIT-S solo se midió en la 4060."""
  return {"gpt2m": SUM["gpt2m"]["ms_frame"], "gpt_s_kd": SUM["gpt_s_kd"]["ms_frame"],
          "genie_s_p4": SUM["genie_s_p4"]["ms_frame"], "convlstm_px": SUM["convlstm_px"]["ms_frame"],
          "maskgit_s_abs_p8": bench_ms("maskgit_s_abs_p8"), "camara": 1000 / FPS}


def main():
  t = tabla()
  porclave = {f["clave"]: f for f in t}
  d = {"tabla": t, "pareto": pareto(), "pasos": pasos(), "horizonte": horizonte(), "cadena": cadena(),
       "latentes": latentes(), "diferencias": diferencias(), "semillas": semillas(), "piso": piso(),
       "temperaturas": temperaturas(), "escala": escala(), "datos": datos(), "carrera": carrera(),
       "ce_fvd": {k: round(v, 3) if isinstance(v, float) else v
                  for k, v in leer(R / "summary.json")["ce_vs_fvd1"].items()},
       "m": porclave}
  # cocientes que se dicen en las slides (todos salen de los valores de arriba)
  m = porclave
  d["cocientes"] = {
    "gpt2m_vs_camara": round(m["gpt2m"]["ms3080"] / 50, 1),
    "maskgit4_vs_gpt_4060": round(m["gpt_s_kd"]["ms4060"] / m["maskgit_s_abs_p4"]["ms4060"], 1),
    "params_gpt2m_vs_chicos": [round(m["gpt2m"]["params"] / m["genie_s_p4"]["params"], 1),
                               round(m["gpt2m"]["params"] / m["gpt_s_kd"]["params"], 1)],
    "fvd3_chicos_vs_gpt2m": [round(m["maskgit_s_abs_p12"]["fvd"]["3"]["v"] / m["gpt2m"]["fvd"]["3"]["v"], 1),
                             round(m["gpt_s_kd"]["fvd"]["3"]["v"] / m["gpt2m"]["fvd"]["3"]["v"], 1)],
    "genie_m_vs_s_params": round(m["genie_m_p4"]["params"] / m["genie_s_p4"]["params"], 1),
    # slide del problema: FPS de gpt2m y frames que graba la cámara (uno cada 50 ms, desde t = 0) mientras genera uno
    "gpt2m_fps": 1000 / m["gpt2m"]["ms3080"],
    "frames_camara_gpt2m": int(m["gpt2m"]["ms3080"] // 50) + 1,
    # slide 4: dónde está cada uno a los 50 ms, con los ms de la 4060
    "gpt_tokens_en_50ms": int(50 / m["gpt_s_kd"]["ms4060"] * 128),
    "mg_frames_en_50ms": int(50 // m["maskgit_s_abs_p4"]["ms4060"]),
    "mg_frames_en_un_gpt": int(m["gpt_s_kd"]["ms4060"] // m["maskgit_s_abs_p4"]["ms4060"]),
  }
  OUT.write_text("// Generado por presentacion/generar_datos.py a partir de results/*.json. No editar a mano.\n"
                 "window.DATOS = " + json.dumps(d, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
  print("escribí", OUT, f"({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
  main()
