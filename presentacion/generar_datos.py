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


def fila(clave, nombre, run_video, familia=""):
  """Una fila de la tabla. La latencia es la de la RTX 3080 Ti (summary.json): las de la 4060 son del código de antes
  de la atención explícita y no se volvieron a medir."""
  c = SUM.get(clave, {})
  f = fvd(run_video)
  if f is None:
    return None
  return {"clave": run_video, "nombre": nombre, "familia": familia, "params": c.get("params_M"),
          "ce": c.get("ce_all"), "lpips1": c.get("lpips_1s"), "fvd": f,
          "drift": f["3"]["v"] / f["1"]["v"], "ms3080": c.get("ms_frame")}


def tabla():
  """Las filas de la tabla principal del informe (paper/tablas/principal.tex)."""
  filas = [fila("gpt2m", "gpt2m (comma)", "gpt2m", "gpt2m"),
           fila("gpt_s_kd", "GPT-S destilado", "gpt_s_kd", "gpt"),
           fila("gpt_s_d050", "GPT-S de cero, 50 %", "gpt_s_d050", "gpt"),
           fila("gpt_s_d025", "GPT-S de cero, 25 %", "gpt_s_d025", "gpt")]
  filas += [fila(f"maskgit_s_abs_p{s}", f"MaskGIT-S, {s} paso{'s' * (s > 1)}", f"maskgit_s_abs_p{s}", "maskgit")
            for s in (1, 2, 4, 8, 12)]
  filas += [fila(f"genie_s_p{s}", f"Genie-S, {s} paso{'s' * (s > 1)}", f"genie_s_p{s}", "genie") for s in (1, 2, 4, 8, 12, 25)]
  filas += [fila(f"genie_m_p{s}", f"Genie-M, {s} pasos", f"genie_m_p{s}", "genie_m") for s in (4, 8)]
  filas += [fila("convlstm_px", "ConvLSTM (píxeles)", "convlstm_px", "convlstm"),
            fila("Transición por celda", "Transición por celda", "transicion", "base"),
            fila("Frame congelado", "Frame congelado", "congelado", "base"),
            fila("Unigrama", "Unigrama", "unigram", "base")]
  return [f for f in filas if f]


def pareto():
  """FVD a 3 s contra ms por frame (modelo solo, p50), todos en la misma RTX 3080 Ti (summary.json)."""
  def pt(clave, pasos=None):
    ms, f = SUM.get(clave, {}).get("ms_frame"), fvd(clave)
    return None if ms is None or f is None else {"ms": ms, "fvd3": f["3"]["v"], "pasos": pasos}
  return {"gpt2m": [pt("gpt2m")],
          "gpt": [pt("gpt_s_kd")],
          "maskgit": [pt(f"maskgit_s_abs_p{s}", s) for s in (1, 2, 4, 8, 12)],
          "genie": [pt(f"genie_s_p{s}", s) for s in (1, 2, 4, 8, 12)]}


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
  """Cadena completa en la 3080 Ti: p50 por etapa con la cadena cerrada y p99 a ritmo de cámara (tabla cadena).
  Sin las mediciones de antes y después de la optimización (OPT_SUFIJOS): esas van en optimizacion()."""
  out = {}
  for f in sorted((R / "infer").glob("latencia_*__rtx3080ti.json")):
    d = leer(f)
    if d["name"].endswith(OPT_SUFIJOS):
      continue
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
  """ms por frame del modelo solo (p50) para la carrera del video, todos en la RTX 3080 Ti."""
  return {"gpt2m": SUM["gpt2m"]["ms_frame"], "gpt_s_kd": SUM["gpt_s_kd"]["ms_frame"],
          "genie_s_p4": SUM["genie_s_p4"]["ms_frame"], "convlstm_px": SUM["convlstm_px"]["ms_frame"],
          "maskgit_s_abs_p8": SUM["maskgit_s_abs_p8"]["ms_frame"], "camara": 1000 / FPS}


# ---- optimización de la inferencia (Docs/optimizar-decode-gpt.md): cada paso medido con su commit, en la 3080 Ti
OPT_SUFIJOS = ("_eager", "_base_hoy", "_base_hoy_r2", "_framegraph", "_framegraph_r2", "_attn", "_attn_r2")
FRAME_BYTES = 128 * 256 * 3          # un frame de cámara uint8: lo único que cruza el PCIe en cada sentido
CACHE_LEN = 2584                     # caché KV estática: 20 frames x 129 tokens, alineada a 8
GPT2M_CFG = {"n_layer": 24, "n_head": 16, "dim": 1024}   # commaVQ utils/gpt.py, GPTConfig


def perfil(nombre, variante="graphs"):
  d = leer(R / "perfil" / f"{nombre}.json")
  return (d["variantes"].get(variante), d) if d else (None, None)


NOMBRES_ATT = (("direct_copy", "copia del bias"), ("gemv", "p·v"), ("SoftMax", "softmax"), ("gemm", "q·kᵀ"))
ORDEN_ATT = ("copia del bias", "q·kᵀ", "softmax", "p·v")    # el orden en que corren (Docs/atencion-decode.md)


def resumen_perfil(antes, despues, n_layer):
  """Tiempo de GPU por frame en la atención y en el resto de la red, antes y después de cached_attention.
  Antes, la atención es el kernel de SDPA (prefill y decode). Después, los kernels de cached_attention (los que se
  llaman n_layer x 127 veces por frame, solo en el decode) más el SDPA del prefill."""
  att = next(k for k in antes["top"] if k["categoria"] == "atencion")
  nuevos = [k for k in despues["top"] if round(k["llamadas_por_paso"]) == n_layer * (CODES - 1)]
  prefill = despues["por_categoria"].get("atencion", {}).get("us_por_paso", 0)
  # una llamada del decode (el promedio del perfil mezcla las del prefill, más largas; el prefill sigue por SDPA después)
  grids = sorted(att.get("grids", {}).items(), key=lambda kv: -kv[1])
  pasos = antes.get("pasos_perfilados") or 1
  pre = next((k for k in despues["top"] if k["categoria"] == "atencion"), None)
  us_dec = ((att["us_por_paso"] - sum(n for _, n in grids[1:]) / pasos * pre["us_por_llamada"]) / (grids[0][1] / pasos)
            if grids and pre else att["us_por_llamada"])
  att_d = (sum(k["us_por_paso"] for k in nuevos) + prefill) / 1000
  return {
    "antes": {"ms_gpu": round(antes["ms_gpu_por_paso"], 1), "kernels": round(antes["kernels_por_paso"]),
              "att_ms": round(att["us_por_paso"] / 1000, 1), "att_pct": round(100 * att["fraccion"]),
              "att_us": round(att["us_por_llamada"], 1), "att_us_decode": round(us_dec, 1),
              "att_llamadas": round(att["llamadas_por_paso"]),
              "att_bloques": att.get("bloques"), "resto_ms": round(antes["ms_gpu_por_paso"] - att["us_por_paso"] / 1000, 1)},
    "despues": {"ms_gpu": round(despues["ms_gpu_por_paso"], 1), "kernels": round(despues["kernels_por_paso"]),
                "att_ms": round(att_d, 1), "att_pct": round(100 * att_d / despues["ms_gpu_por_paso"]),
                "att_us": round(sum(k["us_por_llamada"] for k in nuevos), 1), "att_kernels": len(nuevos),
                "att_bloques": [k.get("bloques") for k in nuevos], "resto_ms": round(despues["ms_gpu_por_paso"] - att_d, 1),
                "att_lista": lista_att(nuevos)},
  }


def lista_att(nuevos):
  """Los kernels de cached_attention en el orden en que corren: nombre, µs por llamada y bloques (del perfil)."""
  por = {next(n for clave, n in NOMBRES_ATT if clave in k["kernel"]): k for k in nuevos}
  return [{"n": n, "us": round(por[n]["us_por_llamada"], 2), "bloques": por[n].get("bloques")} for n in ORDEN_ATT if n in por]


def optimizacion():
  """Los cuatro pasos (p50 del modelo solo en la RTX 3080 Ti) y los perfiles por kernel de antes y después.
  Pasos: eager (la CPU lanza cada kernel), un CUDA graph por token (código de main antes de la optimización), el frame
  entero en un graph (GPTStream) y la atención explícita contra la caché (cached_attention, lo vigente)."""
  def p50(n):
    b = leer(n)
    return round(b["p50_ms"], 1) if b else None
  B = R / "bench"
  pasos = {m: [p50(B / f"{m}_eager__fp16.json"), p50(B / f"{m}_base_hoy__fp16_graphs.json"),
               p50(B / f"{m}_framegraph__fp16_graphs.json"), p50(B / f"{m}__fp16_graphs.json")]
           for m in ("gpt_s_kd", "gpt2m")}
  cfg = leer(R / "checkpoints.json")["gpt_s_kd"]["config"]
  kv_mb = lambda c: 2 * c["n_layer"] * CACHE_LEN * c["dim"] * 2 / 1e6          # k y v, fp16
  # perfiles: antes de cambiar la atención (el frame en un graph, con el grid de cada kernel) y después
  antes, dA = perfil("gpt_s_kd_perfil")
  despues, _ = perfil("gpt_s_kd_perfil_attn")
  gpt = resumen_perfil(antes, despues, cfg["n_layer"])
  g2a, _ = perfil("gpt2m_perfil")
  g2d, _ = perfil("gpt2m_perfil_attn")
  gpt2m = resumen_perfil(g2a, g2d, GPT2M_CFG["n_layer"]) if g2a and g2d else None
  cad = {k: leer(R / "infer" / f"latencia_gpt_s_kd{s}__rtx3080ti.json") for k, s in (("antes", "_base_hoy"), ("despues", ""))}
  return {
    "pasos": pasos, "gpu": "RTX 3080 Ti", "sm": dA["sm"], "perfil_gpt2m": gpt2m,
    "maskgit_p4": [p50(B / "maskgit_s_abs_p4_base_hoy__fp16_graphs_s4.json"), p50(B / "maskgit_s_abs_p4__fp16_graphs_s4.json")],
    "genie_p4": SUM["genie_s_p4"]["ms_frame"],
    "cadena": {k: {"p50": round(d["cadena_cerrada"]["p50"], 1), "p99_ritmo": round(d["cadena_ritmo"]["p99"], 1),
                   "tarde": round(d["cadena_ritmo"]["late_pct"], 1)} for k, d in cad.items() if d},
    "kb_frame": round(FRAME_BYTES / 1e3), "mb_pesos": round(SUM["gpt_s_kd"]["params_M"] * 2, 1), "mb_kv": round(kv_mb(cfg), 1),
    "mb_pesos_gpt2m": round(SUM["gpt2m"]["params_M"] * 2), "mb_kv_gpt2m": round(kv_mb(GPT2M_CFG)),
    "cache_len": CACHE_LEN, "capas": cfg["n_layer"], "cabezas": cfg["n_head"], "dim_cabeza": cfg["dim"] // cfg["n_head"],
    # solo la atención (cached_attention, antes de frame_noise), para separar su efecto del ruido
    "attn": {m: p50(B / f"{m}_attn__fp16_graphs.json") for m in ("gpt_s_kd", "gpt2m")},
    # MB de k y v que lee cada llamada del decode (una capa entera de la caché, fp16)
    "mb_kv_capa": round(2 * cfg["n_head"] * CACHE_LEN * (cfg["dim"] // cfg["n_head"]) * 2 / 1e6, 2),
    "mb_kv_capa_gpt2m": round(2 * GPT2M_CFG["n_head"] * CACHE_LEN * (GPT2M_CFG["dim"] // GPT2M_CFG["n_head"]) * 2 / 1e6, 2),
    "gbs_placa": 912,                  # RTX 3080 Ti: 384 bits a 19 Gbps (especificación de NVIDIA)
    "antes": gpt["antes"], "despues": gpt["despues"],
    "fvd3": [fvd("gpt_s_kd")["3"]["v"], fvd("gpt_s_kd_stream")["3"]["v"]] if fvd("gpt_s_kd_stream") else None,
  }


def main():
  t = tabla()
  porclave = {f["clave"]: f for f in t}
  d = {"tabla": t, "pareto": pareto(), "pasos": pasos(), "horizonte": horizonte(), "cadena": cadena(),
       "latentes": latentes(), "diferencias": diferencias(), "semillas": semillas(), "piso": piso(),
       "temperaturas": temperaturas(), "escala": escala(), "datos": datos(), "carrera": carrera(),
       "opt": optimizacion(),
       "ce_fvd": {k: round(v, 3) if isinstance(v, float) else v
                  for k, v in leer(R / "summary.json")["ce_vs_fvd1"].items()},
       "m": porclave}
  # cocientes que se dicen en las slides (todos salen de los valores de arriba)
  m = porclave
  d["cocientes"] = {
    "gpt2m_vs_camara": round(m["gpt2m"]["ms3080"] / 50, 1),
    "maskgit4_vs_gpt": round(m["gpt_s_kd"]["ms3080"] / m["maskgit_s_abs_p4"]["ms3080"], 1),
    "params_gpt2m_vs_chicos": [round(m["gpt2m"]["params"] / m["genie_s_p4"]["params"], 1),
                               round(m["gpt2m"]["params"] / m["gpt_s_kd"]["params"], 1)],
    "fvd3_chicos_vs_gpt2m": [round(m["maskgit_s_abs_p12"]["fvd"]["3"]["v"] / m["gpt2m"]["fvd"]["3"]["v"], 1),
                             round(m["gpt_s_kd"]["fvd"]["3"]["v"] / m["gpt2m"]["fvd"]["3"]["v"], 1)],
    "genie_m_vs_s_params": round(m["genie_m_p4"]["params"] / m["genie_s_p4"]["params"], 1),
    # slide del problema: FPS de gpt2m y frames que graba la cámara (uno cada 50 ms, desde t = 0) mientras genera uno
    "gpt2m_fps": 1000 / m["gpt2m"]["ms3080"],
    "frames_camara_gpt2m": int(m["gpt2m"]["ms3080"] // 50) + 1,
    # slide 4: qué completa cada uno, con los ms de la 3080 Ti
    "gpt_frames_en_50ms": int(50 // m["gpt_s_kd"]["ms3080"]),
    "mg_frames_en_50ms": int(50 // m["maskgit_s_abs_p4"]["ms3080"]),
    "mg_frames_en_un_gpt": int(m["gpt_s_kd"]["ms3080"] // m["maskgit_s_abs_p4"]["ms3080"]),
    # slide de la optimización: cuántas veces más rápido quedó cada uno, del primer paso al último y del segundo al último
    "opt_gpt_total": round(d["opt"]["pasos"]["gpt_s_kd"][0] / d["opt"]["pasos"]["gpt_s_kd"][3], 1),
    "opt_gpt_att": round(d["opt"]["pasos"]["gpt_s_kd"][2] / d["opt"]["pasos"]["gpt_s_kd"][3], 1),
    "opt_gpt2m_att": round(d["opt"]["pasos"]["gpt2m"][2] / d["opt"]["pasos"]["gpt2m"][3], 1),
  }
  OUT.write_text("// Generado por presentacion/generar_datos.py a partir de results/*.json. No editar a mano.\n"
                 "window.DATOS = " + json.dumps(d, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
  print("escribí", OUT, f"({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
  main()
