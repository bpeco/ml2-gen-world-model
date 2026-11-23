"""Banco de pruebas interactivo: elegir un clip de val, uno o dos modelos, generar 3 s en vivo y medir la corrida.

  python -m src.infer.probador                 # abre http://localhost:8765
  python -m src.infer.probador --port 9000 --no-abrir

Qué hace cada corrida (los mismos clips y el mismo protocolo que el paper, sobre un solo clip):
- Clip: uno de los 400 de results/clips_eval.json (20 frames reales de contexto y 60 de continuación).
- Generación: el stream de src/infer/stream.py, batch 1, fp16 y CUDA graphs donde andan (el protocolo de
  src/bench.py). La latencia es la del paso del modelo (step() con torch.cuda.synchronize), sin el decoder ni el
  envío del frame, que se miden aparte. Antes de medir se calientan los graphs con predicciones descartadas y se
  vuelve a sembrar el generador.
- Comparación: los dos modelos se intercalan en la GPU y siempre avanza el que lleva menos tiempo de modelo
  acumulado. La página reproduce cada uno con su propio reloj medido, al lado de la cámara real a 20 FPS.
- Métricas de la corrida: LPIPS (alex) frame a frame contra los tokens reales decodificados con el mismo decoder,
  a 1, 2 y 3 s, la deriva LPIPS 3 s / 1 s, la CE del último frame (teacher forcing, como ce_last de src.evaluate)
  sobre los 60 frames del clip, y la latencia (media, p50, p99, FPS, fracción de frames bajo 50 ms).
- FVD: compara distribuciones de video, con un clip no se puede calcular; la página muestra el del paper
  (results/summary.json, 400 clips) al lado.

Servidor HTTP de la stdlib, un solo trabajo de GPU a la vez. Los frames viajan como JPEG en Server-Sent Events.
"""
import argparse
import base64
import io
import json
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image

from src.common import CTX_FRAMES
from src.rollouts import VAL, clip_tokens, load_clips

R = Path("results")
HTML = Path(__file__).with_name("probador.html")
PRESUPUESTO_MS = 50.0
HORIZONTES = {"1": 19, "2": 39, "3": 59}   # segundos -> índice del frame generado

# id -> temperatura de las evaluaciones del paper (results/eval_*.json: elegida en el holdout, Genie 1 y no el 2 de su
# paper), etiqueta, familia, spec de load_model, pasos posibles y por defecto, etiqueta en summary.json
CATALOGO = [
  dict(id="gpt2m", temp=1.0, nombre="gpt2m (comma)", familia="GPT", spec="gpt2m", params="307M", resumen="gpt2m",
       nota="El GPT de referencia de comma: 24 capas, token a token. Los pesos (~1,2 GB) se bajan la primera vez."),
  dict(id="gpt_s_kd", temp=0.9, nombre="GPT-S destilado", familia="GPT", spec="gpt_s_kd", params="12,4M", resumen="gpt_s_kd",
       nota="6 capas, destilado de gpt2m. Genera los 128 tokens de a uno."),
  dict(id="maskgit_s", temp=1.0, nombre="MaskGIT-S", familia="MaskGIT", spec="maskgit_s_abs", params="12,4M",
       pasos=[1, 2, 4, 8, 12], pasos_def=8, resumen="maskgit_s_abs_p{s}",
       nota="El mismo backbone que GPT-S, con decodificación paralela en S pasos."),
  dict(id="maskgit_s_shard0", temp=1.0, nombre="MaskGIT-S, shard completo", familia="MaskGIT", spec="maskgit_s_abs_shard0",
       params="12,4M", pasos=[1, 2, 4, 8, 12], pasos_def=8, resumen="maskgit_s_abs_shard0_p{s}",
       nota="La misma red y receta que MaskGIT-S, entrenada con todo el shard 0 (2,94M frames, 3,8 veces el presupuesto común). Mejora el LPIPS pero empeora el FVD: más cerca del real frame a frame, menos realista como video."),
  dict(id="v2_peso3", temp=1.0, nombre="v2 · peso3", familia="MaskGIT v2 (movimiento)", spec="maskgit_s_v2_peso3", params="12,4M",
       pasos=[4, 8], pasos_def=8, resumen="maskgit_s_v2_peso3_p{s}",
       nota="Desde cero. La CE pesa 3 veces más en los tokens que cambian respecto del frame anterior. Se mueve más (ratio 0,47 con 8 pasos), pero el FVD a 3 s empeora mucho (683)."),
  dict(id="v2_pself", temp=1.0, nombre="v2 · pself", familia="MaskGIT v2 (movimiento)", spec="maskgit_s_v2_pself", params="12,4M",
       pasos=[4, 8], pasos_def=8, resumen="maskgit_s_v2_pself_p{s}",
       nota="Desde cero, scheduled sampling fuerte: la mitad de los frames de entrada son reconstrucciones del propio modelo. Es el que más se mueve (ratio 0,93), pero el video es poco realista (FVD 3 s 618)."),
  dict(id="v2_motion", temp=1.0, nombre="v2 · motion", familia="MaskGIT v2 (movimiento)", spec="maskgit_s_v2_motion", params="12,4M",
       pasos=[4, 8], pasos_def=8, resumen="maskgit_s_v2_motion_p{s}",
       nota="Desde cero. Cada token visible recibe un embedding de \"igual o distinto al frame anterior\". Más quieto que el base y sin mejora de FVD."),
  dict(id="v2_motioncb", temp=1.0, nombre="v2 · motioncb", familia="MaskGIT v2 (movimiento)", spec="maskgit_s_v2_motioncb", params="12,4M",
       pasos=[4, 8], pasos_def=8, resumen="maskgit_s_v2_motioncb_p{s}",
       nota="Desde cero. Suma la diferencia de vectores del codebook de commaVQ con el frame anterior. Es el punto de partida de todos los ajustes finos (ft_*). FVD a 3 s parecido al base (430 con 8 pasos)."),
  dict(id="v2_flow_s1", temp=1.0, nombre="v2 · flow_s1", familia="MaskGIT v2 (movimiento)", spec="maskgit_s_v2_flow_s1", params="13,0M",
       pasos=[4, 8], pasos_def=8, resumen="maskgit_s_v2_flow_s1_p{s}",
       nota="motioncb + transformer de flujo (las últimas 4 transiciones de cada celda), entrenado desde cero, semilla 1. Tardó en arrancar a aprender y quedó a medio entrenar: el error se acumula (FVD 3 s ~1870)."),
  dict(id="v2_ft_cb", temp=1.0, nombre="v2 · ft_cb", familia="MaskGIT v2 (movimiento)", spec="maskgit_s_v2_ft_cb", params="12,4M",
       pasos=[4, 8], pasos_def=8, resumen="maskgit_s_v2_ft_cb_p{s}",
       nota="Control: motioncb + 1000 pasos más de ajuste fino, sin flujo. FVD 3 s 434 con 8 pasos."),
  dict(id="v2_ft_flow", temp=1.0, nombre="v2 · ft_flow", familia="MaskGIT v2 (movimiento)", spec="maskgit_s_v2_ft_flow", params="13,0M",
       pasos=[4, 8], pasos_def=8, resumen="maskgit_s_v2_ft_flow_p{s}",
       nota="Ajuste fino de motioncb con el transformer de flujo. Mantiene el movimiento, pero extrapola sus propios errores: bien a 1 s, FVD 3 s 1045 con 8 pasos."),
  dict(id="v2_ft_l_cb", temp=1.0, nombre="v2 · ft_l_cb", familia="MaskGIT v2 (movimiento)", spec="maskgit_s_v2_ft_l_cb", params="12,4M",
       pasos=[4, 8], pasos_def=8, resumen="maskgit_s_v2_ft_l_cb_p{s}",
       nota="Control con flow_limpio (el movimiento pasado se calcula sobre frames completos, como al generar). Es el mejor con 4 pasos (FVD 3 s 415), pero es una sola semilla."),
  dict(id="v2_ft_l_flow", temp=1.0, nombre="v2 · ft_l_flow", familia="MaskGIT v2 (movimiento)", spec="maskgit_s_v2_ft_l_flow", params="13,0M",
       pasos=[4, 8], pasos_def=8, resumen="maskgit_s_v2_ft_l_flow_p{s}",
       nota="Flujo + flow_limpio, sin scheduled sampling. Mejor que ft_flow, pero sigue acumulando error (FVD 3 s 657 con 8 pasos)."),
  dict(id="v2_ft_l_cb_ss", temp=1.0, nombre="v2 · ft_l_cb_ss", familia="MaskGIT v2 (movimiento)", spec="maskgit_s_v2_ft_l_cb_ss", params="12,4M",
       pasos=[4, 8], pasos_def=8, resumen="maskgit_s_v2_ft_l_cb_ss_p{s}",
       nota="Control con flow_limpio + scheduled sampling (25 %). Más estable, pero más quieto y peor a 1 s."),
  dict(id="v2_ft_l_flow_ss", temp=1.0, nombre="v2 · ft_l_flow_ss", familia="MaskGIT v2 (movimiento)", spec="maskgit_s_v2_ft_l_flow_ss", params="13,0M",
       pasos=[4, 8], pasos_def=8, resumen="maskgit_s_v2_ft_l_flow_ss_p{s}",
       nota="Flujo con las dos correcciones (flow_limpio + scheduled sampling). Ya no empeora, pero tampoco mejora: empata con su control (FVD 3 s 473 contra 471 con 8 pasos)."),
  dict(id="genie_s", temp=1.0, nombre="Genie-S", familia="Genie", spec="genie_s", params="15,0M",
       pasos=[1, 2, 4, 8, 12, 25], pasos_def=4, resumen="genie_s_p{s}",
       nota="ST-transformer de Genie con decodificación MaskGIT. Temperatura 1, la elegida en el holdout (su paper usa 2)."),
  dict(id="genie_m", temp=1.0, nombre="Genie-M", familia="Genie", spec="genie_m", params="34,8M",
       pasos=[1, 2, 4, 8, 12, 25], pasos_def=4, resumen="genie_m_p{s}",
       nota="Genie de 8 capas. Si no está en checkpoints/ se baja del almacén la primera vez."),
  dict(id="convlstm_px", temp=1.0, nombre="ConvLSTM", familia="Píxeles", spec="convlstm_px", params="8,0M",
       resumen="convlstm_px",
       nota="Shi 2015 sobre píxeles: pronostica 10 frames de una vez, así que la latencia llega en bloques."),
  dict(id="congelado", temp=0.0, nombre="Frame congelado", familia="Baseline", spec="congelado", params="—",
       resumen="Frame congelado", nota="Repite el último frame de contexto."),
  dict(id="transicion", temp=1.0, nombre="Transición por celda", familia="Baseline", spec="transicion", params="—",
       resumen="Transición por celda", nota="Tabla de transición de primer orden por celda."),
  dict(id="unigram", temp=1.0, nombre="Unigrama", familia="Baseline", spec="unigram", params="—",
       resumen="Unigrama", nota="Muestrea cada token de la frecuencia global."),
]
POR_ID = {m["id"]: m for m in CATALOGO}

_gpu = threading.Lock()     # un trabajo de GPU a la vez
_modelos = {}               # spec -> modelo cargado (fp16, en el device)
_codec = _lpips = None
_clip_cache = {}


def dev():
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync():
  if torch.cuda.is_available():
    torch.cuda.synchronize()


# ---------------------------------------------------------------- datos de referencia

def resumen_paper():
  try:
    return {c["label"]: c for c in json.loads((R / "summary.json").read_text(encoding="utf-8"))["configs"]}
  except FileNotFoundError:
    return {}


def resumen_v2(m, pasos):
  """Modelos de la rama v2 (no están en summary.json): las métricas salen de results/video_* y results/eval_*."""
  try:
    v = json.loads((R / f"video_{m['resumen'].format(s=pasos)}.json").read_text(encoding="utf-8"))
  except (FileNotFoundError, KeyError):
    return {}
  c = {"fvd_1s": v["fvd"]["1"]["value"], "fvd_3s": v["fvd"]["3"]["value"], "lpips_1s": v["lpips"]["curve"][19]}
  try:
    c["ce_last"] = json.loads((R / f"eval_{m['spec']}.json").read_text(encoding="utf-8"))["ce_last"]
  except (FileNotFoundError, KeyError):
    pass
  return c


def paper(m, pasos):
  c = resumen_paper().get(m["resumen"].format(s=pasos)) or resumen_v2(m, pasos)
  f1, f3 = c.get("fvd_1s"), c.get("fvd_3s")
  return {"fvd_1s": f1, "fvd_3s": f3, "deriva": (f3 / f1) if f1 and f3 else None, "lpips_1s": c.get("lpips_1s"),
          "ce_last": c.get("ce_last"), "ms_frame": c.get("ms_frame")}


def disponible(m):
  """'local' si los pesos están en disco; 'descarga' si se bajan al cargarlo."""
  if m["spec"] == "gpt2m":
    return "local" if Path("data/raw/pytorch_model.bin").exists() else "descarga"
  if m["familia"] == "Baseline":
    return "local"
  return "local" if (Path("checkpoints") / m["spec"] / "final.pt").exists() else "descarga"


def catalogo():
  out = []
  for m in CATALOGO:
    pasos = m.get("pasos")
    out.append({**{k: m[k] for k in ("id", "nombre", "familia", "params", "nota", "temp")}, "pasos": pasos,
                "pasos_def": m.get("pasos_def"), "disponible": disponible(m),
                "paper": {str(s): paper(m, s) for s in pasos} if pasos else {"": paper(m, None)}})
  gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
  return {"modelos": out, "n_clips": len(load_clips()["clips"]), "gpu": gpu, "presupuesto_ms": PRESUPUESTO_MS}


# ---------------------------------------------------------------- GPU: codec, LPIPS, modelos

def codec():
  global _codec
  if _codec is None:
    from src.infer.codec import Codec, ensure_weights
    ensure_weights()
    _codec = Codec(dev(), "fp16", graphs=torch.cuda.is_available())
  return _codec


def lpips_net():
  global _lpips
  if _lpips is None:
    import lpips
    _lpips = lpips.LPIPS(net="alex", verbose=False).to(dev()).eval()
  return _lpips


def decodificar(tok):
  """tokens (N, 128) -> uint8 (N, 128, 256, 3) en CPU."""
  c, t = codec(), torch.as_tensor(np.asarray(tok, dtype=np.int64), device=dev())
  return np.stack([c.decode(t[i:i + 1]).cpu().numpy()[0] for i in range(len(t))])


def jpeg(img, q=88):
  buf = io.BytesIO()
  Image.fromarray(img).save(buf, "JPEG", quality=q)
  return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def cargar(m, avisar):
  spec = m["spec"]
  if spec in _modelos:
    return _modelos[spec]
  if disponible(m) == "descarga":
    avisar(f"Bajando los pesos de {m['nombre']} (una sola vez)…")
    if spec == "gpt2m":
      from src.hf_fetch import fetch
      fetch("commaai/commavq-gpt2m", "pytorch_model.bin", Path("data/raw"))
  avisar(f"Cargando {m['nombre']}…")
  from src.models import load_model
  if len(_modelos) >= 3:  # 8 GB de VRAM: no acumular
    _modelos.clear()
    torch.cuda.empty_cache()
  modelo = load_model(spec, "cpu").eval().to(dev(), torch.float16)
  _modelos[spec] = modelo
  return modelo


def clip(i):
  """Tokens e imágenes del clip i: ctx (20, 128), real (60, 128) y sus frames decodificados."""
  if i not in _clip_cache:
    clips = load_clips()
    c = clips["clips"][i]
    val = np.load(VAL, mmap_mode="r")
    ctx, real = clip_tokens(val, {**clips, "clips": [c]})
    ctx, real = ctx[0].astype(np.int64), real[0].astype(np.int64)
    img = decodificar(np.concatenate([ctx, real]))
    _clip_cache.clear()
    _clip_cache[i] = dict(ctx=ctx, real=real, img_ctx=img[:len(ctx)], img_real=img[len(ctx):],
                          segmento=int(c["segment"]), inicio=int(c["start"]))
  return _clip_cache[i]


def movimiento(i):
  """Percentil del clip en cambio de escena real a 3 s (LPIPS del frame congelado), si está la caché."""
  p = R / "cache/lpips_frozen.npy"
  if not p.exists():
    return None
  v = np.load(p)[:, -1]
  return int(round(100 * (v < v[i]).mean()))


@torch.no_grad()
def lpips_curva(pred, real):
  """pred, real uint8 (N, H, W, 3) -> LPIPS (N,)."""
  f = lambda a: torch.as_tensor(a, device=dev()).permute(0, 3, 1, 2).float() / 127.5 - 1
  out = []
  for k in range(0, len(pred), 20):
    out.append(lpips_net()(f(pred[k:k + 20]), f(real[k:k + 20])).flatten().float().cpu().numpy())
  return np.concatenate(out)


@torch.no_grad()
def ce_clip(modelo, ctx, real):
  """CE del último frame (ce_last de src.evaluate) en los 60 frames del clip: cada uno dado los 19 reales anteriores."""
  if getattr(modelo, "output", "tokens") == "pixels":
    return None
  todo = torch.as_tensor(np.concatenate([ctx, real]), device=dev())
  n = len(ctx)
  ventanas = torch.stack([todo[t - CTX_FRAMES + 1:t + 1] for t in range(n, n + len(real))])  # (60, 20, 128)
  tot = 0.0
  for k in range(0, len(ventanas), 6):
    v = ventanas[k:k + 6]
    lg = modelo.last_frame_logits(v).float()
    tot += torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), v[:, -1].reshape(-1), reduction="sum").item()
  return tot / (len(ventanas) * 128)


def preparar(modelo, ctx, pasos, semilla, temp):
  """Stream con el contexto real observado y los graphs calentados; el generador vuelve a la semilla."""
  from src.infer.stream import make_stream
  t = torch.as_tensor(ctx, device=dev())
  for graphs in ([True, False] if torch.cuda.is_available() else [False]):
    try:
      s = make_stream(modelo, temperature=temp, steps=pasos, seed=semilla, graphs=graphs)
      for f in t:
        s.observe(f[None])
      k = getattr(getattr(modelo, "cfg", None), "n_out", 1) if hasattr(s, "observe_pixels") else 1
      for _ in range(3):
        s.predict(k)
      sync()
      s.gen.manual_seed(semilla)
      return s, graphs
    except RuntimeError:
      if not graphs:
        raise
      if hasattr(modelo, "use_cuda_graphs"):
        modelo.use_cuda_graphs(False)
  raise RuntimeError("no se pudo preparar el stream")


# ---------------------------------------------------------------- corrida

def correr(pedido, enviar):
  """pedido: {clip, semilla, modelos: [{id, pasos, temp}], escenario: "rollout" | "online", dur, antic, periodo}."""
  if pedido.get("escenario") == "online":
    return correr_online(pedido, enviar)
  i, semilla = int(pedido["clip"]), int(pedido.get("semilla", 0))
  sel = [(POR_ID[m["id"]], (int(m["pasos"]) if m.get("pasos") else None),
          float(m["temp"]) if m.get("temp") is not None else POR_ID[m["id"]]["temp"]) for m in pedido["modelos"]]
  with _gpu:
    enviar("estado", {"texto": "Decodificando el clip…"})
    c = clip(i)
    corridas = []
    for m, pasos, temp in sel:
      modelo = cargar(m, lambda t: enviar("estado", {"texto": t}))
      enviar("estado", {"texto": f"Preparando {m['nombre']} (contexto y CUDA graphs)…"})
      s, graphs = preparar(modelo, c["ctx"], pasos, semilla, temp)
      corridas.append(dict(m=m, pasos=pasos, temp=temp, modelo=modelo, s=s, graphs=graphs, ms=[], dec_ms=[], frames=[],
                           pix=hasattr(s, "observe_pixels")))
    enviar("listo", {"modelos": [{"id": r["m"]["id"], "nombre": r["m"]["nombre"], "pasos": r["pasos"],
                                  "temp": r["temp"], "graphs": r["graphs"]} for r in corridas]})

    n = len(c["real"])
    while any(len(r["ms"]) < n for r in corridas):
      k = min((r for r in corridas if len(r["ms"]) < n), key=lambda r: sum(r["ms"]))  # el más atrasado en su reloj
      t0 = time.perf_counter()
      out = k["s"].step()
      sync()
      ms = (time.perf_counter() - t0) * 1000
      t1 = time.perf_counter()
      img = out.cpu().numpy()[0] if k["pix"] else codec().decode(out).cpu().numpy()[0]
      k["dec_ms"].append((time.perf_counter() - t1) * 1000)
      k["ms"].append(ms)
      k["frames"].append(img)
      enviar("frame", {"m": corridas.index(k), "i": len(k["ms"]) - 1, "ms": ms, "img": jpeg(img)})

    enviar("estado", {"texto": "Calculando métricas (LPIPS y CE)…"})
    congelado = lpips_curva(np.repeat(c["img_ctx"][-1:], n, 0), c["img_real"])
    res = []
    for r in corridas:
      ms = np.asarray(r["ms"])
      lp = lpips_curva(np.stack(r["frames"]), c["img_real"])
      lp_h = {h: float(lp[j]) for h, j in HORIZONTES.items()}
      res.append({
        "latencia": {"media": float(ms.mean()), "p50": float(np.median(ms)), "p99": float(np.percentile(ms, 99)),
                     "max": float(ms.max()), "fps": 1000 / float(ms.mean()), "bajo_presupuesto": float((ms < PRESUPUESTO_MS).mean()),
                     "total_s": float(ms.sum() / 1000), "decoder_ms": float(np.median(r["dec_ms"])), "por_frame": ms.tolist()},
        "lpips": {"curva": lp.tolist(), **lp_h, "deriva": lp_h["3"] / lp_h["1"] if lp_h["1"] > 0 else None},
        "ce_last": ce_clip(r["modelo"], c["ctx"], c["real"]),
        "paper": paper(r["m"], r["pasos"]),
      })
    enviar("metricas", {"modelos": res, "congelado": {"curva": congelado.tolist(),
                                                      **{h: float(congelado[j]) for h, j in HORIZONTES.items()}}})


# ---------------------------------------------------------------- escenario en línea

def _guardar(s):
  """Estado del stream para imaginar un plan y volver atrás (el ConvLSTM guarda ventana y cola)."""
  if hasattr(s, "observe_pixels") and hasattr(s, "win"):
    return (s.win, s.queue, s.qi, s.n_seen)
  return (s._save(), s.n_seen)


def _restaurar(s, snap):
  if hasattr(s, "observe_pixels") and hasattr(s, "win"):
    s.win, s.queue, s.qi, s.n_seen = snap
  else:
    s._load(snap[0])
    s.n_seen = snap[1]


def tramo_online(i, dur_s, antic_s):
  """Tokens reales del segmento del clip i: 20 de contexto + (dur + antic) s, corriendo el arranque si no entra."""
  clips = load_clips()
  c = clips["clips"][i]
  val = np.load(VAL, mmap_mode="r")
  n = CTX_FRAMES + int(round((dur_s + antic_s) * 20))
  ini = max(0, min(int(c["start"]), val.shape[1] - n))
  return np.asarray(val[c["segment"], ini:ini + n]).astype(np.int64), int(c["segment"]), ini


def simular_online(r, toks, dur_f, antic_f, periodo_ms, enviar):
  """Simulación de uso real con un modelo solo en la GPU.

  El reloj t (ms) arranca al terminar el contexto; el frame real k (1..) llega en t = 50 k. El modelo, cuando está
  libre (y en el próximo tick del período, si hay período), observa todos los frames reales que llegaron y
  imagina antic_f frames: un plan. El plan cuesta la suma de los pasos medidos y queda vigente cuando termina.
  En cada instante se muestra el frame del plan vigente más nuevo que cubre ese instante.
  """
  s, ctx_n = r["s"], CTX_FRAMES
  vistos, libre, planes = 0, 0.0, []
  fin_ms = dur_f * 50
  while True:
    ini = libre if periodo_ms <= 0 else np.ceil(libre / periodo_ms - 1e-9) * periodo_ms
    if planes and ini < (planes[-1]["n"] + 1) * 50:  # sin un frame real nuevo, replanear no aporta nada
      ini = (planes[-1]["n"] + 1) * 50 if periodo_ms <= 0 else np.ceil((planes[-1]["n"] + 1) * 50 / periodo_ms - 1e-9) * periodo_ms
    if ini >= fin_ms:
      break
    n = int(ini // 50)                         # frames reales que ya llegaron
    for k in range(vistos, n):                 # entran al estado (el encoder no se cuenta: los tokens son de val)
      s.observe(torch.as_tensor(toks[ctx_n + k], device=dev())[None])
    vistos = n
    snap = _guardar(s)
    salida, ms = [], 0.0
    for _ in range(antic_f):
      t0 = time.perf_counter()
      out = s.step()
      sync()
      ms += (time.perf_counter() - t0) * 1000
      salida.append(out)
    _restaurar(s, snap)
    planes.append(dict(ini=float(ini), listo=float(ini + ms), n=n, ms=ms, salida=salida))
    if len(planes) > 1:  # memoria: del plan anterior solo se ven los frames anteriores a que este esté listo (y el último)
      q, desde = planes[-2], int(np.ceil((ini + ms) / 50))
      for j in range(max(0, desde - q["n"] - 1), antic_f - 1):
        q["salida"][j] = None
    libre = ini + ms
    if len(planes) % 3 == 1:
      enviar("estado", {"texto": f"{r['m']['nombre']}: plan {len(planes)} desde el frame real {n} "
                                 f"({ms:.0f} ms de cómputo para {antic_f / 20:g} s)"})
  # qué se ve en cada frame t = 1..dur_f: el plan más nuevo ya listo que cubre t
  vista = []
  for t in range(1, dur_f + 1):
    elegido = None
    for p_i, p in enumerate(planes):
      if p["listo"] <= t * 50 and p["n"] < t <= p["n"] + antic_f:
        elegido = p_i
    vista.append(elegido)
  return planes, vista


def correr_online(pedido, enviar):
  i, semilla = int(pedido["clip"]), int(pedido.get("semilla", 0))
  dur_s, antic_s = float(pedido.get("dur", 10)), float(pedido.get("antic", 2))
  periodo_ms = float(pedido.get("periodo", 1)) * 1000
  dur_f, antic_f = int(round(dur_s * 20)), int(round(antic_s * 20))
  sel = [(POR_ID[m["id"]], (int(m["pasos"]) if m.get("pasos") else None),
          float(m["temp"]) if m.get("temp") is not None else POR_ID[m["id"]]["temp"]) for m in pedido["modelos"]]
  with _gpu:
    enviar("estado", {"texto": "Decodificando el tramo de video…"})
    toks, segmento, ini = tramo_online(i, dur_s, antic_s)
    real_img = decodificar(toks[CTX_FRAMES - 1:])       # [0] = último frame de contexto, [t] = frame real t
    salida = {"segmento": segmento, "inicio": ini, "dur_f": dur_f, "antic_f": antic_f, "periodo_ms": periodo_ms,
              "ctx": [jpeg(x) for x in decodificar(toks[:CTX_FRAMES])],
              "real": [jpeg(x) for x in real_img[1:dur_f + 1]], "modelos": []}
    for m, pasos, temp in sel:
      modelo = cargar(m, lambda t: enviar("estado", {"texto": t}))
      enviar("estado", {"texto": f"Preparando {m['nombre']} (contexto y CUDA graphs)…"})
      s, graphs = preparar(modelo, toks[:CTX_FRAMES], pasos, semilla, temp)
      r = dict(m=m, s=s, pix=hasattr(s, "observe_pixels"))
      planes, vista = simular_online(r, toks, dur_f, antic_f, periodo_ms, enviar)
      enviar("estado", {"texto": f"{m['nombre']}: decodificando lo que se ve y midiendo LPIPS…"})
      a_img = lambda out: out.cpu().numpy()[0] if r["pix"] else codec().decode(out).cpu().numpy()[0]
      imgs, cache = {}, {}

      def frame(p_i, j):
        k = f"{p_i}_{j}"
        if k not in cache:
          cache[k] = a_img(planes[p_i]["salida"][j])
          imgs[k] = jpeg(cache[k])
        return k

      vista_k, edad, pred, real_v, ref = [], [], [], [], []
      for t, p_i in enumerate(vista, 1):
        if p_i is None:
          vista_k.append(None); edad.append(None)
          continue
        p = planes[p_i]
        k = frame(p_i, t - p["n"] - 1)
        vista_k.append(k); edad.append((t - p["n"]) * 50)
        pred.append(cache[k]); real_v.append(real_img[t]); ref.append(real_img[p["n"]])
      lp_v = lpips_curva(np.stack(pred), np.stack(real_v)).tolist() if pred else []
      lp_r = lpips_curva(np.stack(ref), np.stack(real_v)).tolist() if ref else []
      it_v, it_r = iter(lp_v), iter(lp_r)
      lp_t = [None if v is None else next(it_v) for v in vista_k]
      ref_t = [None if v is None else next(it_r) for v in vista_k]
      # calidad a la anticipación pedida: el último frame de cada plan contra el real de ese instante
      ult = [(p_i, p["n"] + antic_f) for p_i, p in enumerate(planes)]
      lp_ant = lpips_curva(np.stack([cache.get(f"{p_i}_{antic_f - 1}") if f"{p_i}_{antic_f - 1}" in cache
                                     else a_img(planes[p_i]["salida"][-1]) for p_i, _ in ult]),
                           np.stack([real_img[t] for _, t in ult])).tolist()
      for p_i, _ in ult:
        frame(p_i, antic_f - 1)                    # para la miniatura "lo que espera en +K"
      cub = [e for e in edad if e is not None]
      ms_planes = [p["ms"] for p in planes]
      lp_ok = [v for v in lp_t if v is not None]
      ref_ok = [v for v in ref_t if v is not None]
      salida["modelos"].append({
        "graphs": graphs, "imgs": imgs, "vista": vista_k, "edad": edad, "lpips": lp_t, "ref": ref_t,
        "planes": [{"ini": p["ini"], "listo": p["listo"], "n": p["n"], "ms": p["ms"], "lpips_ant": lp_ant[j],
                    "ultimo": f"{j}_{antic_f - 1}", "margen": (p["n"] + antic_f) * 50 - p["listo"]} for j, p in enumerate(planes)],
        "resumen": {"planes": len(planes), "ms_plan": float(np.mean(ms_planes)), "ms_paso": float(np.sum(ms_planes) / (len(planes) * antic_f)),
                    "sostiene": bool(periodo_ms <= 0 or np.mean(ms_planes) <= periodo_ms),
                    "cobertura": len(cub) / dur_f, "edad_media": float(np.mean(cub)) if cub else None,
                    "edad_max": float(np.max(cub)) if cub else None,
                    "lpips_ahora": float(np.mean(lp_ok)) if lp_ok else None,
                    "ref_ahora": float(np.mean(ref_ok)) if ref_ok else None,
                    "lpips_ant": float(np.mean(lp_ant)),
                    "primera_s": planes[0]["listo"] / 1000 if planes else None,
                    "margen_medio": float(np.mean([(p["n"] + antic_f) * 50 - p["listo"] for p in planes]))},
        "paper": paper(m, pasos),
      })
    enviar("online", salida)


def datos_clip(i):
  with _gpu:
    c = clip(i)
  return {"clip": i, "segmento": c["segmento"], "inicio": c["inicio"], "movimiento": movimiento(i),
          "ctx": [jpeg(x) for x in c["img_ctx"]], "real": [jpeg(x) for x in c["img_real"]]}


# ---------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
  def log_message(self, *a):
    pass

  def _json(self, obj, code=200):
    b = json.dumps(obj, ensure_ascii=False).encode()
    self.send_response(code)
    self.send_header("Content-Type", "application/json; charset=utf-8")
    self.send_header("Content-Length", str(len(b)))
    self.end_headers()
    self.wfile.write(b)

  def do_GET(self):
    u = urlparse(self.path)
    q = {k: v[0] for k, v in parse_qs(u.query).items()}
    try:
      if u.path in ("/", "/index.html"):
        b = HTML.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
      elif u.path == "/api/catalogo":
        self._json(catalogo())
      elif u.path == "/api/clip":
        self._json(datos_clip(int(q["i"])))
      elif u.path == "/api/correr":
        self._sse(json.loads(q["pedido"]))
      else:
        self._json({"error": "no existe"}, 404)
    except (BrokenPipeError, ConnectionResetError):
      pass
    except Exception as e:
      traceback.print_exc()
      try:
        self._json({"error": f"{type(e).__name__}: {e}"}, 500)
      except Exception:
        pass

  def _sse(self, pedido):
    self.send_response(200)
    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
    self.send_header("Cache-Control", "no-cache")
    self.end_headers()

    def enviar(ev, data):
      self.wfile.write(f"event: {ev}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode())
      self.wfile.flush()

    try:
      correr(pedido, enviar)
      enviar("fin", {})
    except (BrokenPipeError, ConnectionResetError):
      pass
    except Exception as e:
      traceback.print_exc()
      enviar("error", {"texto": f"{type(e).__name__}: {e}"})


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--port", type=int, default=8765)
  ap.add_argument("--no-abrir", action="store_true")
  a = ap.parse_args(argv)
  srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
  url = f"http://localhost:{a.port}"
  print(f"Banco de pruebas en {url}  (Ctrl+C para cortar)", flush=True)
  if not a.no_abrir:
    webbrowser.open(url)
  try:
    srv.serve_forever()
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  main()
