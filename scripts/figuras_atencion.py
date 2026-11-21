"""Figuras de Docs/atencion-decode.md, leídas de los JSON de results/ (no mide nada).

  python scripts/figuras_atencion.py        # -> Docs/img/atencion/*.svg

Los números salen de los perfiles por kernel (results/perfil/, `bench --perfil`), de los bench del modelo solo
(results/bench/) y de la cadena completa (results/infer/). Los tamaños (caché, cabezas, dimensiones) salen de la
configuración del GPT-S en results/checkpoints.json y de src.common. Lo único escrito a mano son las especificaciones de
la placa (VRAM y ancho de banda, de NVIDIA). Cada SVG tiene su propio fondo claro, así se lee igual en el tema claro y en
el oscuro de GitHub.
"""
import json
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src.common import CTX_FRAMES, TPF  # noqa: E402

R = RAIZ / "results"
OUT = RAIZ / "Docs" / "img" / "atencion"
PASOS = TPF - 2                   # pasos de decode por frame: 128 tokens, el primero sale del prefill
VRAM_GB, ANCHO_GBS = 12, 912      # RTX 3080 Ti: 12 GB GDDR6X, 384 bits a 19 Gbps (especificación de NVIDIA)

# paleta: naranja = atención, azul = el resto de la red / el modelo, gris = otros
FONDO, BORDE, TXT, SUAVE, TENUE = "#fcfcfb", "#e1e0d9", "#1f1e1c", "#5f5e5a", "#8a8984"
OCIOSO, ATN, ATN_CLARO, RED, RED_CLARO, GRIS, GRIS_CLARO = (
  "#ecebe6", "#eb6834", "#fbe3d8", "#2a78d6", "#dbe8f8", "#b4b2a9", "#e6e5df")
FUENTE = "Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', Consolas, 'SFMono-Regular', monospace"
W = 720


# ---------------------------------------------------------------- datos

def leer(p):
  return json.loads((R / p).read_text(encoding="utf-8"))


def p50(nombre):
  return leer(f"bench/{nombre}.json")["p50_ms"]


def perfil(nombre):
  d = leer(f"perfil/{nombre}.json")
  return d, d["variantes"]["graphs"]


def kernels_atencion(antes, despues, n_layer):
  """El kernel de SDPA de antes y los cuatro de cached_attention de ahora (los que se llaman n_layer x PASOS veces por frame)."""
  sdpa = next(k for k in antes["top"] if k["categoria"] == "atencion")
  grids = sorted(sdpa["grids"].items(), key=lambda kv: -kv[1])
  decode = [int(x) for x in grids[0][0].split("x")]
  prefill = [int(x) for x in grids[1][0].split("x")]
  nombres = (("direct_copy", "copia del bias"), ("gemv", "p·v"), ("SoftMax", "softmax"), ("gemm", "q·kᵀ"))
  nuevos = {}
  for k in despues["top"]:
    if round(k["llamadas_por_paso"]) != n_layer * PASOS:
      continue
    nombre = next(n for clave, n in nombres if clave in k["kernel"])
    nuevos[nombre] = {"us": k["us_por_llamada"], "bloques": k["bloques"], "us_frame": k["us_por_paso"]}
  orden = ["copia del bias", "q·kᵀ", "softmax", "p·v"]
  assert set(nuevos) == set(orden), nuevos.keys()
  prefill_despues = despues["por_categoria"].get("atencion", {}).get("us_por_paso", 0)
  return {
    "sdpa_us": sdpa["us_por_llamada"], "sdpa_llamadas": round(sdpa["llamadas_por_paso"]),
    "sdpa_ms": sdpa["us_por_paso"] / 1000, "sdpa_bloques": math.prod(decode), "sdpa_grid": decode,
    "sdpa_grid_prefill": prefill,
    "nuevos": [(n, nuevos[n]) for n in orden], "nuevos_us": sum(nuevos[n]["us"] for n in orden),
    "att_despues_ms": (sum(nuevos[n]["us_frame"] for n in orden) + prefill_despues) / 1000,
    "gpu_antes": antes["ms_gpu_por_paso"], "gpu_despues": despues["ms_gpu_por_paso"],
    "kernels_antes": round(antes["kernels_por_paso"]), "kernels_despues": round(despues["kernels_por_paso"]),
  }


def kernels_mascara(antes, n_att):
  """Los kernels con que SDPA convertía la máscara bool en cada llamada: un where y dos fill por llamada."""
  where = next(k for k in antes["top"] if "TensorCompare" in k["kernel"] and round(k["llamadas_por_paso"]) == n_att)
  fill = next(k for k in antes["top"] if "FillFunctor" in k["kernel"] and round(k["llamadas_por_paso"]) == 2 * n_att)
  return {"where_us": where["us_por_llamada"], "where_bloques": where["bloques"], "fill_us": fill["us_por_llamada"],
          "kernels": round(where["llamadas_por_paso"] + fill["llamadas_por_paso"]),
          "ms": (where["us_por_paso"] + fill["us_por_paso"]) / 1000}


def datos():
  cfg = leer("checkpoints.json")["gpt_s_kd"]["config"]
  from src.models.backbone import mult8
  from src.models.gpt import GPT
  cache = mult8(CTX_FRAMES * TPF)
  hd = cfg["dim"] // cfg["n_head"]
  dA, antes = perfil("gpt_s_kd_perfil")
  _, despues = perfil("gpt_s_kd_perfil_attn")
  _, g2a = perfil("gpt2m_perfil")
  _, g2d = perfil("gpt2m_perfil_attn")
  gpt = kernels_atencion(antes, despues, cfg["n_layer"])
  gpt2m = kernels_atencion(g2a, g2d, 24)
  prefill_q = GPT.ctx_frames * TPF + 1
  lat = {s: leer(f"infer/latencia_gpt_s_kd{s}__rtx3080ti.json") for s in ("_base_hoy", "")}
  return {
    "sm": dA["sm"], "capas": cfg["n_layer"], "cabezas": cfg["n_head"], "hd": hd, "cache": cache,
    "ctx": GPT.ctx_frames, "prefill_q": prefill_q,
    "kv_bytes": 2 * cfg["n_head"] * cache * hd * 2,
    "s_bytes": {"decode": cfg["n_head"] * cache * 2, "maskgit": cfg["n_head"] * TPF * cache * 2,
                "prefill": cfg["n_head"] * prefill_q * cache * 2},
    "gpt": gpt, "gpt2m": gpt2m, "mascara": kernels_mascara(antes, gpt["sdpa_llamadas"]),
    "pasos": {m: [p50(f"{m}_eager__fp16"), p50(f"{m}_base_hoy__fp16_graphs"), p50(f"{m}_framegraph__fp16_graphs"),
                  p50(f"{m}__fp16_graphs")] for m in ("gpt_s_kd", "gpt2m")},
    "ruido": {"128": [p50("gpt_s_kd_attn__fp16_graphs"), p50("gpt_s_kd_attn_r2__fp16_graphs")],
              "1": p50("gpt_s_kd__fp16_graphs")},
    "cadena": {k: {"p50": v["cadena_ritmo"]["p50"], "etapas": v["cadena_ritmo"]["stages_p50_ms"],
                   "p99": v["cadena_ritmo"]["p99"], "tarde": v["cadena_ritmo"]["late_pct"],
                   "p50_cerrada": v["cadena_cerrada"]["p50"]}
               for k, v in (("antes", lat["_base_hoy"]), ("despues", lat[""]))},
  }


# ---------------------------------------------------------------- SVG

def f1(v):
  return f"{v:.1f}".replace(".", ",")


def miles(n):
  """Separador de miles a la española: solo desde 5 cifras (2584, pero 12.921)."""
  return str(n) if n < 10_000 else f"{n:,}".replace(",", ".")


def t(x, y, s, size=13, fill=TXT, anchor="start", peso=400, mono=False):
  fam = f' font-family="{MONO}"' if mono else ""
  return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
          f'font-weight="{peso}"{fam}>{escape(s)}</text>')


def r(x, y, w, h, fill, stroke=None, rx=3, extra="", grosor=1):
  st = f' stroke="{stroke}" stroke-width="{grosor}"' if stroke else ""
  return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{st} {extra}/>'


def ln(x1, y1, x2, y2, color=SUAVE, ancho=1, flecha=False, guiones=None):
  m = ' marker-end="url(#f)"' if flecha else ""
  d = f' stroke-dasharray="{guiones}"' if guiones else ""
  return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{ancho}"'
          f' stroke-linecap="round"{m}{d}/>')


def svg(h, cuerpo, titulo, desc, extra_defs=""):
  return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" height="{h}" role="img" '
          f'font-family="{FUENTE}">\n<title>{escape(titulo)}</title>\n<desc>{escape(desc)}</desc>\n'
          f'<defs><marker id="f" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
          f'orient="auto-start-reverse"><path d="M1 1L9 5L1 9" fill="none" stroke="{SUAVE}" stroke-width="1.5" '
          f'stroke-linecap="round" stroke-linejoin="round"/></marker>'
          f'<pattern id="rayas" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
          f'<line x1="0" y1="0" x2="0" y2="5" stroke="{TENUE}" stroke-width="1.5"/></pattern>{extra_defs}</defs>\n'
          f'{r(0.5, 0.5, W - 1, h - 1, FONDO, BORDE, rx=10)}\n' + "\n".join(cuerpo) + "\n</svg>\n")


def ancho_texto(s, size=12):
  return len(s) * size * 0.56


def eje(x0, x1, y, vmax, ticks, unidad):
  out = [ln(x0, y, x1, y, BORDE)]
  for v in ticks:
    x = x0 + (x1 - x0) * v / vmax
    out.append(ln(x, y, x, y + 4, TENUE))
    out.append(t(x, y + 17, f"{v:g}".replace(".", ",") + (f" {unidad}" if v == ticks[-1] else ""), 11, TENUE,
                 "middle"))
  return out


def leyenda(x, y, items):
  out = []
  for color, texto, borde in items:
    out.append(r(x, y - 10, 12, 12, color if not borde else FONDO, color if borde else None, rx=2))
    out.append(t(x + 18, y, texto, 12, SUAVE))
    x += 18 + ancho_texto(texto) + 22
  return out


def grilla_sm(x0, y0, encendidos, cw=28, ch=16, gx=4, gy=5, color=ATN, n=80, cols=10):
  out = []
  for i in range(n):
    x, y = x0 + (i % cols) * (cw + gx), y0 + (i // cols) * (ch + gy)
    out.append(r(x, y, cw, ch, color if i in encendidos else OCIOSO, rx=3))
  return out


# ---------------------------------------------------------------- figuras

def fig_kernel(d):
  c = [r(20, 20, 230, 120, "#ffffff", BORDE, rx=8),
       t(36, 46, "CPU", 14, peso=600), t(36, 66, "Python y PyTorch", 12, SUAVE),
       t(36, 92, "torch.softmax(s, -1)", 12, mono=True),
       t(36, 118, "elige el kernel y su grilla", 12, SUAVE),
       ln(250, 80, 296, 80, flecha=True), t(273, 70, "lanza", 12, SUAVE, "middle"),
       r(20, 180, 230, 130, "#ffffff", ATN, rx=8),
       t(36, 206, "1 bloque", 14, peso=600), t(36, 226, "hasta 1024 hilos", 12, SUAVE),
       t(36, 244, "mismo código; cada hilo", 12, SUAVE), t(36, 262, "con otra parte de los datos", 12, SUAVE)]
  c += [ln(40 + 12 * i, 280, 40 + 12 * i, 296, ATN, 1.5) for i in range(16)]
  c += [r(304, 20, 396, 300, "#ffffff", BORDE, rx=8),
        t(320, 46, "GPU: RTX 3080 Ti", 14, peso=600),
        t(320, 66, f"{d['sm']} SM; cada bloque corre entero en uno", 12, SUAVE)]
  enc = {3, 17, 25, 46, 58, 60}
  c += grilla_sm(320, 80, enc, cw=33, ch=18, gx=3.5, gy=6)
  c += [ln(250, 245, 318, 218, TENUE, guiones="3 3"),
        r(320, 280, 364, 28, GRIS_CLARO, rx=4),
        t(332, 299, f"VRAM: {VRAM_GB} GB a {ANCHO_GBS} GB/s; cada kernel lee y escribe acá", 12, TXT)]
  return svg(334, c, "Qué es un kernel",
             "La CPU lanza un kernel; la GPU lo corre como una grilla de bloques de hilos; cada bloque ocupa un SM "
             "y los datos entran y salen por la VRAM.")


def fichas(x, y, items, alto=26, gap=8, xmax=700, flechas=True):
  """Fichas de texto en fila, con flechas entre ellas; cortan de renglón al llegar a xmax."""
  out, x0 = [], x
  for i, (texto, color) in enumerate(items):
    w = ancho_texto(texto) + 20
    if x + w > xmax:
      x, y = x0, y + alto + 14
    out.append(r(x, y, w, alto, "#ffffff", color, rx=5, grosor=1.5))
    out.append(t(x + w / 2, y + alto / 2 + 4, texto, 12, TXT, "middle"))
    x += w
    if flechas and i < len(items) - 1:
      out.append(t(x + gap, y + alto / 2 + 4, "→", 12, TENUE, "middle"))
      x += 2 * gap
    else:
      x += gap
  return out, y + alto


def fig_capa(d):
  g, m = d["gpt"], d["mascara"]
  capa = [("layer norm", GRIS), ("qkv", RED), ("escribe k", GRIS), ("escribe v", GRIS), ("atención", ATN),
          ("proyección", RED), ("suma residual", GRIS), ("layer norm", GRIS), ("fc", RED), ("gelu", GRIS),
          ("proyección", RED), ("suma residual", GRIS)]
  c = leyenda(24, 34, [(RED, "matmul de una capa lineal", True), (ATN, "atención", True), (GRIS, "otros", True)])
  f, y = fichas(24, 52, capa)
  c += f
  y += 34
  c.append(t(24, y, "El grupo de la atención, en un paso de decode de una capa (bl. = bloques):", 13, peso=600))
  y += 16
  antes = [("fill", GRIS), ("fill", GRIS), ("where (máscara)", GRIS),
           (f"fmha_cutlassF · {g['sdpa_bloques']} bl.", ATN)]
  ahora = [(f"{n.replace(' del bias', '')} · {k['bloques']} bl.", ATN) for n, k in g["nuevos"]]
  tot_antes = 2 * m["fill_us"] + m["where_us"] + g["sdpa_us"]
  for etiqueta, items, total in (("antes", antes, f"≈{tot_antes:.0f} µs"), ("ahora", ahora, f"{f1(g['nuevos_us'])} µs")):
    c.append(t(24, y + 17, etiqueta, 12, SUAVE))
    f, _ = fichas(84, y, items, flechas=False, xmax=640)
    c += f
    c.append(t(696, y + 17, total, 12, TXT, "end"))
    y += 40
  return svg(y + 10, c, "Una capa del GPT en kernels",
             "En un paso de decode, cada capa es una cadena de kernels; el grupo de la atención pasó de un kernel de "
             f"{g['sdpa_bloques']} bloques precedido por la conversión de la máscara a cuatro kernels cortos.")


def fig_cache(d):
  S, x0, x1 = d["cache"], 30, 690
  e = (x1 - x0) / S
  ctx_pos = d["ctx"] * TPF                    # posiciones del contexto (sin el BOS del frame nuevo)
  q_pos = d["prefill_q"] + 63                 # un paso de decode a mitad del frame
  c = []
  qx, qy = 360, 64
  centros = [x0 + (k * TPF + TPF / 2) * e for k in range(d["ctx"])]
  escrito_fin = x0 + (q_pos + 1) * e
  for cx in centros:
    c.append(ln(qx, qy, cx, 108, ATN, 1).replace("/>", ' opacity="0.4"/>'))
  c.append(ln(qx, qy, (x0 + ctx_pos * e + escrito_fin) / 2, 108, ATN, 1.2))
  c.append(ln(qx, qy, (escrito_fin + x1) / 2, 108, TENUE, 1, guiones="3 3"))
  c += [r(qx - 95, 24, 190, 40, ATN_CLARO, ATN, rx=6), t(qx, 49, "query del token t", 14, peso=600, anchor="middle"),
        t(qx + 110, 49, f"lee las {miles(S)} posiciones", 12, SUAVE)]
  for k in range(d["ctx"]):
    c.append(r(x0 + k * TPF * e, 110, TPF * e - 1.5, 40, RED_CLARO, RED, rx=2))
  c.append(r(x0 + ctx_pos * e, 110, escrito_fin - (x0 + ctx_pos * e), 40, ATN_CLARO, ATN, rx=2))
  c.append(r(escrito_fin + 0.5, 110, x1 - escrito_fin - 0.5, 40, "url(#rayas)", TENUE, rx=2))
  fin_ctx = x0 + ctx_pos * e - 1.5
  c += [f'<path d="M{x0} 158 V164 H{fin_ctx:.1f} V158" fill="none" stroke="{SUAVE}" stroke-width="0.8"/>',
        t((x0 + fin_ctx) / 2, 182, f"{d['ctx']} frames de contexto: {miles(ctx_pos)} posiciones, del prefill", 12,
          SUAVE, "middle"),
        ln(x0 + ctx_pos * e + 8, 154, x0 + ctx_pos * e + 8, 196, TENUE, guiones="3 3"),
        t(x1, 212, "frame nuevo: el decode escribe 1 posición por paso", 12, SUAVE, "end")]
  c += leyenda(30, 246, [(RED_CLARO, "escrita en el prefill", False), (ATN_CLARO, "escrita en este frame", False)])
  c += [r(372, 236, 12, 12, "url(#rayas)", TENUE, rx=2), t(390, 246, "vacía: −∞ en la máscara, pero se lee igual", 12,
                                                            SUAVE)]
  return svg(266, c, "Un paso de decode lee toda la caché KV",
             f"La query del token nuevo se compara con las {S} posiciones de la caché de una capa: el contexto, el "
             "frame nuevo a medio escribir y las posiciones vacías que la máscara anula.")


def barras_apiladas(x0, x1, y, filas, vmax, titulo, ticks):
  c = [t(24, y, titulo, 13, peso=600)]
  y += 14
  for etiqueta, a, b in filas:
    wa, wb = (x1 - x0) * a / vmax, (x1 - x0) * b / vmax
    c += [t(24, y + 16, etiqueta, 12, SUAVE), r(x0, y, wa, 22, ATN, rx=0),
          r(x0 + wa + 2, y, wb - 2, 22, RED, extra=""),
          t(x0 + wa + wb + 8, y + 16, f"{f1(a + b)} ms", 12),
          t(x0 + wa / 2, y + 16, f"{100 * a / (a + b):.0f} %", 12, "#ffffff", "middle", 600) if wa > 40 else ""]
    y += 32
  c += eje(x0, x1, y, vmax, ticks, "ms")
  return c, y + 30


def fig_perfil(d):
  c = leyenda(24, 30, [(ATN, "atención (con el prefill)", False), (RED, "resto de la red", False)])
  y = 60
  for nombre, k, vmax, ticks in (("GPT-S destilado", d["gpt"], 160, [0, 40, 80, 120, 160]),
                                 ("gpt2m", d["gpt2m"], 750, [0, 250, 500, 750])):
    filas = [("antes", k["sdpa_ms"], k["gpu_antes"] - k["sdpa_ms"]),
             ("después", k["att_despues_ms"], k["gpu_despues"] - k["att_despues_ms"])]
    f, y = barras_apiladas(100, 620, y, filas, vmax, nombre, ticks)
    c += f
  return svg(y - 4, c, "Tiempo de GPU por frame",
             "Tiempo de GPU por frame según el perfil, antes y después, en el GPT-S y gpt2m: la atención era la mayor "
             "parte y bajó a una fracción.")


def fig_sm(d):
  g = d["gpt"]
  qk = g["nuevos"][1][1]["bloques"]
  c = [t(30, 30, "Antes: fmha_cutlassF", 14, peso=600),
       t(30, 48, f"grilla {' × '.join(map(str, g['sdpa_grid']))} (x, y, z): un bloque por cabeza", 12, SUAVE),
       t(384, 30, "Ahora: q·kᵀ de cached_attention", 14, peso=600),
       t(384, 48, f"{g['nuevos'][1][1]['bloques']} bloques en los {d['sm']} SM", 12, SUAVE)]
  c += grilla_sm(30, 60, {2, 15, 27, 44, 58, 71})
  c += grilla_sm(384, 60, set(range(d["sm"])))
  c += [t(30, 246, f"{d['sm'] - g['sdpa_bloques']} SM sin bloques; cada bloque recorre la caché en serie", 12, SUAVE),
        t(384, 246, f"{qk // d['sm']} o {qk // d['sm'] + 1} bloques por SM: la GPU entera", 12, SUAVE)]
  # línea de tiempo de una llamada, en la misma escala
  x0, x1, vmax, y = 110, 640, 170, 280
  c.append(t(24, y, "Una llamada, por capa y por token (µs):", 13, peso=600))
  y += 14
  e = (x1 - x0) / vmax
  c += [t(24, y + 15, "antes", 12, SUAVE), r(x0, y, g["sdpa_us"] * e, 20, ATN),
        t(x0 + g["sdpa_us"] * e + 8, y + 15, f"{f1(g['sdpa_us'])} µs", 12)]
  y += 30
  c.append(t(24, y + 15, "ahora", 12, SUAVE))
  acc = 0
  for i, (n, k) in enumerate(g["nuevos"]):
    c.append(r(x0 + acc * e, y, k["us"] * e - 1, 20, ATN, rx=0, extra=f'opacity="{1 if i % 2 == 0 else 0.6}"'))
    acc += k["us"]
  c.append(t(x0 + acc * e + 8, y + 15, f"{f1(acc)} µs", 12))
  y += 30
  c += eje(x0, x1, y, vmax, [0, 40, 80, 120, 160], "µs")
  # ampliación del después
  y += 40
  c.append(t(24, y, "Ahora, ampliado: los cuatro kernels", 13, peso=600))
  y += 14
  vz, acc = 16, 0
  ez = (x1 - x0) / vz
  for i, (n, k) in enumerate(g["nuevos"]):
    xa = x0 + acc * ez
    c.append(r(xa, y, k["us"] * ez - 2, 20, ATN, rx=0, extra=f'opacity="{1 if i % 2 == 0 else 0.6}"'))
    c.append(t(xa + k["us"] * ez / 2, y + 36, n, 12, TXT, "middle"))
    c.append(t(xa + k["us"] * ez / 2, y + 52, f"{f1(k['us'])} µs · {k['bloques']} bl.", 11, SUAVE, "middle"))
    acc += k["us"]
  y += 62
  c += eje(x0, x1, y, vz, [0, 4, 8, 12, 16], "µs")
  return svg(y + 34, c, "Seis bloques para 80 SM",
             f"Antes, el kernel de atención usaba {g['sdpa_bloques']} de los {d['sm']} SM y tardaba unos "
             f"{g['sdpa_us']:.0f} µs; ahora el q·kᵀ reparte {qk} bloques entre los {d['sm']} SM, p·v usa "
             f"{dict(g['nuevos'])['p·v']['bloques']} bloques y el softmax {dict(g['nuevos'])['softmax']['bloques']}; los "
             f"cuatro kernels suman {f1(g['nuevos_us'])} µs.")


def fig_algoritmo(d):
  g, S = d["gpt"], d["cache"]
  vueltas = math.ceil(S / 64)
  c = [t(24, 32, "Antes: un bloque por cabeza", 14, peso=600),
       t(24, 50, "la cabeza 1; las otras hacen lo mismo en otros SM", 12, SUAVE)]
  x0, x1 = 24, 340
  wv = (x1 - x0) / vueltas
  for i in range(vueltas):
    c.append(r(x0 + i * wv, 64, wv - 1.2, 18, ATN if i < 12 else OCIOSO, rx=1))
  c += [t(x0, 98, "vuelta 1", 11, TENUE), t(x1, 98, f"vuelta {vueltas}", 11, TENUE, "end"),
        t((x0 + x1) / 2, 98, f"{miles(S)} claves, de a 64", 11, TENUE, "middle"),
        ln(x0, 112, x0 + 12 * wv, 112, ATN, 1.5, flecha=True),
        t(24, 136, "en cada vuelta: 64 puntajes; actualiza el máximo,", 12, SUAVE),
        t(24, 154, "la suma y la salida (softmax en línea)", 12, SUAVE),
        t(24, 180, f"{vueltas} vueltas en serie: ≈{g['sdpa_us']:.0f} µs", 12, TXT),
        t(24, 198, "no guarda los puntajes en la VRAM", 12, TXT)]
  # después
  X0, X1 = 380, 696
  c += [t(X0, 32, "Ahora: tres pasos, cada uno en paralelo", 14, peso=600),
        t(X0, 50, "cuBLAS y PyTorch eligen la grilla", 12, SUAVE)]
  n = dict(g["nuevos"])
  wseg = (X1 - X0) / 21
  for i in range(21):
    c.append(r(X0 + i * wseg, 64, wseg - 1.2, 18, ATN, rx=1))
  s_kb = d["s_bytes"]["decode"] / 1000
  c += [t(X0, 98, f"1. q·kᵀ: todos los puntajes a la vez ({n['q·kᵀ']['bloques']} bloques)", 12, TXT),
        ln(X0 + 10, 106, X0 + 10, 124, flecha=True),
        t(X0 + 22, 120, f"s: {d['cabezas']} × {miles(S)} puntajes en la VRAM, {s_kb:.0f} KB", 12, SUAVE),
        t(X0, 144, f"2. softmax de cada fila ({n['softmax']['bloques']} bloques)", 12, TXT),
        ln(X0 + 10, 152, X0 + 10, 170, flecha=True),
        t(X0 + 22, 166, "p: los mismos puntajes, ya como pesos", 12, SUAVE),
        t(X0, 190, f"3. p·v: la suma de v con esos pesos ({n['p·v']['bloques']} bloques)", 12, TXT),
        t(X0, 214, f"con la copia del bias: {f1(g['nuevos_us'])} µs", 12, TXT)]
  return svg(232, c, "Dos formas de calcular la misma atención",
             "El kernel de antes recorre las claves en serie con un bloque por cabeza; cached_attention calcula todos "
             "los puntajes a la vez, los guarda y después hace el softmax y la suma con v.")


def fig_bmm(d):
  H, S, hd = d["cabezas"], d["cache"], d["hd"]
  c = [t(24, 30, f"bmm y baddbmm: {H} multiplicaciones independientes, una por cabeza (acá, el q·kᵀ)", 13, peso=600)]
  y0, dx, dy = 70, 7, 7

  def pila(x, y, w, h, color, borde):
    out = []
    for i in reversed(range(H)):
      out.append(r(x + i * dx, y + i * dy, w, h, color, borde, rx=2))
    return out

  c += pila(30, y0 + 16, 46, 8, ATN_CLARO, ATN)
  c += [t(30, y0 + 86, "q", 13, peso=600), t(30, y0 + 102, f"({H}, 1, {hd})", 12, SUAVE, mono=True)]
  c.append(t(112, y0 + 30, "@", 16, SUAVE, "middle"))
  c += pila(140, y0, 250, 46, RED_CLARO, RED)
  c += [t(140, y0 + 86, "kᵀ", 13, peso=600), t(140, y0 + 102, f"({H}, {hd}, {miles(S)})", 12, SUAVE, mono=True)]
  c.append(t(448, y0 + 30, "=", 16, SUAVE, "middle"))
  c += pila(476, y0 + 16, 190, 8, ATN_CLARO, ATN)
  c += [t(476, y0 + 86, "s, los puntajes", 13, peso=600),
        t(476, y0 + 102, f"({H}, 1, {miles(S)})", 12, SUAVE, mono=True)]
  c += [t(24, 210, f"Cada hoja de la pila es una cabeza: (1 × {hd}) @ ({hd} × {miles(S)}) = (1 × {miles(S)}). "
                   "Ninguna depende de otra.", 12, SUAVE),
        t(24, 232, "baddbmm suma la máscara al resultado, bias + alpha · (q @ kᵀ), y lanza dos kernels: una copia y "
                   "la multiplicación.", 12, SUAVE),
        t(24, 254, f"bmm(p, v) es la otra: ({H}, 1, {miles(S)}) @ ({H}, {miles(S)}, {hd}) = ({H}, 1, {hd}).", 12,
          SUAVE)]
  return svg(274, c, "Qué es bmm",
             f"Una multiplicación de matrices por lotes: las {H} cabezas son {H} productos independientes en una sola "
             "llamada.")


def fig_cached(d):
  n = dict(d["gpt"]["nuevos"])
  xs, w, y = [24, 200, 376, 552], 144, 112
  c = []
  for x, texto in ((xs[0], "fila de máscara"), (xs[1], "q y k"), (xs[3], "v")):
    c += [r(x, 24, w, 36, "#ffffff", BORDE, rx=6), t(x + w / 2, 47, texto, 13, anchor="middle"),
          ln(x + w / 2, 60, x + w / 2, y - 4, flecha=True)]
  cajas = [("copia del bias", "baddbmm, 1.º", n["copia del bias"]), ("q·kᵀ (GEMM)", "baddbmm, 2.º", n["q·kᵀ"]),
           ("softmax", "softmax", n["softmax"]), ("p·v (gemv)", "bmm", n["p·v"])]
  for i, (titulo, linea, k) in enumerate(cajas):
    x = xs[i]
    color = ATN if titulo.startswith("q·kᵀ") else BORDE
    fondo = ATN_CLARO if titulo.startswith("q·kᵀ") else "#ffffff"
    c += [r(x, y, w, 96, fondo, color, rx=8), t(x + w / 2, y + 24, titulo, 13, peso=600, anchor="middle"),
          t(x + w / 2, y + 44, linea, 12, SUAVE, "middle", mono=True),
          t(x + w / 2, y + 66, f"{k['bloques']} bloques", 12, SUAVE, "middle"),
          t(x + w / 2, y + 84, f"{f1(k['us'])} µs", 12, SUAVE, "middle")]
    if i < 3:
      c.append(ln(x + w + 2, y + 48, xs[i + 1] - 4, y + 48, flecha=True))
  c += [ln(xs[3] + w / 2, y + 96, xs[3] + w / 2, y + 128, flecha=True),
        r(xs[3], y + 132, w, 36, "#ffffff", BORDE, rx=6), t(xs[3] + w / 2, y + 155, "y → proyección", 13,
                                                            anchor="middle"),
        t(24, y + 146, f"por capa y por token: {f1(d['gpt']['nuevos_us'])} µs en total", 12, TXT),
        t(24, y + 164, f"antes: 1 kernel de {d['gpt']['sdpa_bloques']} bloques, ≈{d['gpt']['sdpa_us']:.0f} µs", 12,
          SUAVE)]
  return svg(y + 186, c, "cached_attention en cuatro kernels",
             "La fila de la máscara se copia como bias, q·kᵀ se suma en una multiplicación de muchos bloques, el "
             "softmax normaliza cada fila y el producto con v da la salida.")


def fig_ruido(d):
  lento, rapido = max(d["ruido"]["128"]), d["ruido"]["1"]
  extra = lento - rapido
  x0, x1, vmax = 110, 640, 46
  e = (x1 - x0) / vmax
  c, y = [], 30
  for titulo, cpu, rotulo in (("Antes: 128 llamadas a exponential_ por frame", extra, "128 lanzamientos"),
                              ("Ahora: una llamada a frame_noise", 0.08, "1 lanzamiento")):
    c.append(t(24, y, titulo, 13, peso=600))
    y += 12
    c += [t(24, y + 16, "CPU", 12, SUAVE), r(x0, y, x1 - x0, 22, GRIS_CLARO, rx=3),
          r(x0, y, max(cpu * e, 3), 22, TENUE, rx=0),
          t(x0 + max(cpu * e, 3) + 8, y + 16,
            f"{rotulo}" + (f": ≈{f1(cpu)} ms (diferencia entre los dos benches)" if cpu > 1 else ""), 12, SUAVE)]
    y += 30
    c += [t(24, y + 16, "GPU", 12, SUAVE), r(x0, y, x1 - x0, 22, GRIS_CLARO, rx=3),
          r(x0 + cpu * e, y, rapido * e, 22, RED, rx=3),
          t(x0 + cpu * e + 10, y + 16, f"graph del frame, ≈{f1(rapido)} ms", 12, "#ffffff", peso=600)]
    y += 34
    c.append(t(x1 + 50, y - 18, f"{f1(lento if cpu > 1 else rapido)} ms", 12, TXT, "end"))
    y += 18
  c += eje(x0, x1, y - 8, vmax, [0, 10, 20, 30, 40], "ms")
  return svg(y + 28, c, "El ruido del muestreo, antes y ahora",
             "Con 128 llamadas, la CPU lanzaba el ruido de cada token antes del graph del frame y la GPU esperaba; con "
             "una sola llamada, el graph arranca enseguida.")


def fig_escalera(d):
  pasos = ["eager", "1 graph por token", "el frame en 1 graph", "atención y ruido"]
  c, y = [], 30
  for nombre, vals, vmax, ticks in (("GPT-S destilado (ms por frame, modelo solo)", d["pasos"]["gpt_s_kd"], 440,
                                     [0, 100, 200, 300, 400]),
                                    ("gpt2m", d["pasos"]["gpt2m"], 1650, [0, 400, 800, 1200, 1600])):
    c.append(t(24, y, nombre, 13, peso=600))
    y += 12
    x0, x1 = 170, 630
    for i, v in enumerate(vals):
      wv = (x1 - x0) * v / vmax
      c += [t(24, y + 15, pasos[i], 12, SUAVE), r(x0, y, wv, 20, RED if i == 3 else GRIS),
            t(x0 + wv + 8, y + 15, f"{f1(v)} ms", 12)]
      y += 28
    c += eje(x0, x1, y, vmax, ticks, "ms")
    y += 52
  # cadena
  c.append(t(24, y, "GPT-S, cámara → encoder → modelo → decoder, a ritmo de cámara (mediana de cada etapa)", 13,
             peso=600))
  y += 8
  c += leyenda(24, y + 16, [(RED, "modelo", False), (GRIS, "encoder y decoder", False)])
  y += 30
  x0, x1, vmax = 170, 470, 180
  e = (x1 - x0) / vmax
  for etiqueta, k in (("antes", "antes"), ("ahora", "despues")):
    ca = d["cadena"][k]
    et = ca["etapas"]
    a, b, cc = et["encoder"] * e, et["model"] * e, et["decoder"] * e
    c += [t(24, y + 15, etiqueta, 12, SUAVE), r(x0, y, a, 20, GRIS, rx=0), r(x0 + a + 2, y, b - 2, 20, RED, rx=0),
          r(x0 + a + b + 2, y, cc - 2, 20, GRIS),
          t(max(x0 + a + b + cc + 8, x0 + 50 * e + 8), y + 15,
            f"{f1(ca['p50'])} ms · p99 {f1(ca['p99'])} · {ca['tarde']:.0f} % tarde", 12)]
    y += 28
  xl = x0 + 50 * e
  c += [ln(xl, y - 64, xl, y + 2, SUAVE, 1.5, guiones="4 3"), t(xl + 4, y - 68, "50 ms: 20 FPS", 11, SUAVE)]
  c += eje(x0, x1, y + 4, vmax, [0, 50, 100, 150], "ms")
  return svg(y + 40, c, "Los cuatro pasos de la optimización",
             "Latencia por frame del GPT-S y de gpt2m en cada paso, y la cadena completa del GPT-S contra el "
             "presupuesto de 50 ms.")


FIGURAS = {"kernel": fig_kernel, "capa": fig_capa, "cache": fig_cache, "perfil": fig_perfil, "sm": fig_sm,
           "algoritmo": fig_algoritmo, "bmm": fig_bmm, "cached-attention": fig_cached, "ruido": fig_ruido,
           "escalera": fig_escalera}


def main():
  d = datos()
  OUT.mkdir(parents=True, exist_ok=True)
  for nombre, fig in FIGURAS.items():
    (OUT / f"{nombre}.svg").write_text(fig(d), encoding="utf-8")
    print(f"Docs/img/atencion/{nombre}.svg")
  resumen = {
    "kv_mb": d["kv_bytes"] / 1e6, "s_kb_decode": d["s_bytes"]["decode"] / 1e3,
    "s_mb_maskgit": d["s_bytes"]["maskgit"] / 1e6, "s_mb_prefill": d["s_bytes"]["prefill"] / 1e6,
    "gbs_antes": d["kv_bytes"] / d["gpt"]["sdpa_us"] / 1e3, "gbs_despues": d["kv_bytes"] / d["gpt"]["nuevos_us"] / 1e3,
    "mascara": d["mascara"], "kernels_despues": d["gpt"]["kernels_despues"], "gpu_despues": d["gpt"]["gpu_despues"],
    "resto_antes": d["gpt"]["gpu_antes"] - d["gpt"]["sdpa_ms"],
    "resto_despues": d["gpt"]["gpu_despues"] - d["gpt"]["att_despues_ms"],
    "gbs_k": d["kv_bytes"] / 2 / dict(d["gpt"]["nuevos"])["q·kᵀ"]["us"] / 1e3,
    "gbs_v": d["kv_bytes"] / 2 / dict(d["gpt"]["nuevos"])["p·v"]["us"] / 1e3,
    "us_copia_softmax": dict(d["gpt"]["nuevos"])["copia del bias"]["us"] + dict(d["gpt"]["nuevos"])["softmax"]["us"],
    "cadena": d["cadena"], "att_despues_ms": d["gpt"]["att_despues_ms"], "gpt2m": {k: d["gpt2m"][k] for k in (
      "sdpa_ms", "att_despues_ms", "gpu_antes", "gpu_despues", "nuevos_us", "sdpa_us")},
  }
  print(json.dumps(resumen, ensure_ascii=False, indent=1))


if __name__ == "__main__":
  main()
