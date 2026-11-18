"""Latencia de punta a punta por frame: video en la CPU -> encoder -> world model -> decoder -> video en la CPU.

  python -m src.infer.bench_e2e --model maskgit_s --steps 4                  # modo en línea, bucle cerrado
  python -m src.infer.bench_e2e --model maskgit_s --mode imagine --pace 50   # imaginación, un frame cada 50 ms
  python -m src.infer.bench_e2e --model none                                 # solo el codec (piso de la cadena)
  python -m src.infer.bench_e2e --model random:maskgit_m --steps 2           # sin checkpoint (la latencia no depende de los pesos)
  python -m src.infer.bench_e2e --model lewm --protocolo                     # las tres columnas de D15 en un JSON
  python -m src.infer.bench_e2e --model maskgit_s --diagnostico --pace 50    # el piso de 20 ms a ritmo de cámara

ConvLSTM (convlstm_px, predice píxeles): la cámara entra directo y sale el frame predicho, sin encoder ni decoder
de comma (las etapas "encoder" y "decoder" quedan en ~0; el "modelo" incluye la copia a la salida).

Latentes (lewm, vjepa2_ac): ven píxeles y no emiten tokens ni imagen, así que la cadena no usa el codec de comma: la
unidad de medida es un PASO DE MODELO (lewm: 4 Hz con LatentStream, 5 frames de cámara; vjepa2_ac: 4 Hz con
VJEPAStream), no un frame de cámara. En cada paso entra el frame de cámara que abre el bloque (los otros 4 solo
acumulan su pose y quedan fuera de la medición) y sale el embedding predicho (B, D) al host. Etapas: "encoder" = subida
del frame, "modelo" = codificar + predecir (o solo el paso, en imagine), "decoder" = copia del embedding al host. Con
--pace P el ritmo es P ms por frame de cámara, o sea un paso de modelo cada P x 5 ms (4 Hz con P = 50). Presupuesto 250 ms.

Modos:
- online: llega un frame de cámara; se codifica, entra al estado del modelo, se predice el frame t+k (k pasos del
  modelo sobre una copia del estado) y se decodifica. Es la cadena completa.
- imagine: el encoder corre solo para los ctx_frames frames iniciales; después cada frame es un paso del modelo
  (el frame generado entra al estado) y un decode. Es lo que mide la tabla de calidad (rollouts de 3 s).

Medición (batch 1):
- Tiempo de reloj (perf_counter) desde que el frame está en la memoria del host (pinned) hasta que el frame
  predicho está de vuelta en el host, sincronizando el device. Desglose por etapa con CUDA events.
- Se separan el primer segundo (los primeros ctx_frames frames: contexto incompleto, sin graphs, y la captura de
  los graphs cuando se llena) y un calentamiento; después se miden --frames frames (default 3000: con menos,
  el p99 es casi el máximo de la muestra).
- --pace 0: bucle cerrado (cada frame apenas termina el anterior). --pace 50: un frame cada 50 ms, como una
  cámara a 20 FPS; con la GPU ociosa entre frames pueden bajar los relojes y subir la latencia.
- Reporta p50, p90, p99, máximo y % de frames tarde (latencia > --budget ms, default 50).
- fp16 con CUDA graphs donde andan; si fallan se mide en eager y el JSON lo declara (graphs, graphs_usados, graphs_error).
Escribe results/infer/e2e_<nombre>__<modo>[_k<k>]__<pace>__<dispositivo>.json

--protocolo (D15): un solo JSON con una columna por modo, results/infer/latencia_<nombre>__<dispositivo>.json
- modelo_solo: el paso del stream sin codec (src.bench.medir_modelo; es lo que escribe results/bench);
- cadena_cerrada: la cadena en bucle cerrado (--pace 0);
- cadena_ritmo: la cadena a ritmo de cámara (--pace, default 50).
Cada columna trae p50, p99 y el % de frames tarde.

--diagnostico: el piso de ~20 ms a ritmo de cámara. Mide la misma cadena con variantes (una fila del JSON cada una):
la referencia en bucle cerrado; la espera entre frames con time.sleep contra espera activa; la resolución del timer de
Windows (sleep con timeBeginPeriod(1)); un kernel de mantenimiento chico en la GPU cada ~1 ms mientras se espera, para
que los relojes no bajen; y la sincronización (evento, torch.cuda.synchronize, sondeo con event.query). Escribe
results/infer/diagnostico_<nombre>__<modo>__<pace>__<dispositivo>.json, con la resolución del timer medida. En CPU corre
todo y produce todas las filas (las variantes de GPU quedan sin efecto, y el JSON lo anota).
"""
import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.bench import es_latente, medir_modelo
from src.common import device_info
from src.infer import backends
from src.infer.bench_codec import device_tag
from src.infer.codec import Codec
from src.infer.stream import make_stream
from src.infer.video import sample_video

OUT = Path("results/infer")
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def pct(ms, q):
  return float(np.percentile(ms, q)) if len(ms) else None


def summarize(ms, budget):
  ms = np.asarray(ms)
  return {"p50": pct(ms, 50), "p90": pct(ms, 90), "p99": pct(ms, 99), "max": float(ms.max()),
          "mean": float(ms.mean()), "late_pct": float((ms > budget).mean() * 100), "n": int(len(ms))}


class Chain:
  """Codec + stream del modelo (o solo codec con model=None), con un paso por frame y su desglose.

  Para los latentes (lewm, vjepa2_ac) no hay codec: un paso es un paso de modelo (ver el docstring del módulo)."""

  def __init__(self, model, codec, dev, mode="online", k=1, sync="event", **stream_kw):
    self.codec, self.dev, self.mode, self.k, self.sync = codec, dev, mode, k, sync
    self.stream = make_stream(model, **stream_kw) if model is not None else None
    self.cuda = dev.type == "cuda"
    self.latent = model is not None and es_latente(model)
    self.k_cam = (getattr(self.stream, "k", None) or getattr(self.stream, "cada", 1)) if self.latent else 1
    self.out_host = None if self.latent else torch.empty(1, 128, 256, 3, dtype=torch.uint8, pin_memory=self.cuda)
    self.pix_out = getattr(model, "output", "tokens") == "pixels"  # el modelo ya predice píxeles: sin decoder de comma
    self._imagine = None
    self._last = None
    self._ka = torch.zeros(16, device=dev) if self.cuda else None

  def _ev(self):
    if not self.cuda:
      return None
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e

  def keepalive(self):
    """Kernel de mantenimiento: un add chico y asíncrono para que la GPU no quede ociosa entre frames."""
    if self._ka is not None:
      self._ka.add_(1)

  def _esperar_gpu(self, e):
    if self.sync == "device":
      torch.cuda.synchronize()
    elif self.sync == "poll":
      while not e.query():
        pass
    else:
      e.synchronize()

  def step(self, frame_host):
    """Un frame de entrada (1, 128, 256, 3) uint8 en el host -> frame predicho (latentes: embedding) en el host.
    Devuelve los eventos (inicio, tras encoder, tras modelo, fin) o None en CPU."""
    e0 = self._ev()
    pixels = self.stream is not None and hasattr(self.stream, "observe_pixels")
    encode = self.mode == "online" or self.stream is None or self._imagine is None
    if encode:
      x = frame_host.to(self.dev, non_blocking=True)
      if not pixels:
        tok = self.codec.encode(x)
    e1 = self._ev()  # el paso del modelo incluye observe (los que ven píxeles: codificar el frame)
    if encode and self.stream is not None:
      self.stream.observe_pixels(x) if pixels else self.stream.observe(tok)
    if self.stream is None:
      pred = tok                                  # solo codec: se reconstruye el frame de entrada
    elif self.mode == "online" or self._imagine is None:  # imagine: durante el contexto inicial, como en línea
      pred = self.stream.predict() if self.latent else self.stream.predict(self.k)
    else:
      pred = next(self._imagine)
    e2 = self._ev()
    if self.latent:
      if self.out_host is None:
        self.out_host = torch.empty(pred.shape, dtype=pred.dtype, pin_memory=self.cuda)
      self.out_host.copy_(pred, non_blocking=True)
    else:
      self.out_host.copy_(pred if self.pix_out else self.codec.decode(pred), non_blocking=True)
    e3 = self._ev()
    if self.cuda:
      self._esperar_gpu(e3)
    elif self.dev.type == "mps":
      torch.mps.synchronize()   # sin eventos en MPS: el tiempo es perf_counter con sincronización al final del frame
    self._last = x if encode else None
    return (e0, e1, e2, e3) if self.cuda else None

  def post(self):
    """Latentes en línea: los frames de cámara que no abren bloque solo acumulan pose (fuera de la medición)."""
    if self.latent and self._last is not None:
      for _ in range(self.k_cam - 1):
        self.stream.observe_pixels(self._last)

  def _pasos(self):
    while True:
      yield self.stream.step()

  def start_imagining(self):
    """Modo imagine: a partir de acá el modelo genera y el encoder deja de correr."""
    if self.mode == "imagine" and self.stream is not None:
      self._imagine = self._pasos() if self.latent else self.stream.imagine(10 ** 9)


class timer_1ms:
  """Windows: pide al sistema una resolución de timer de 1 ms (timeBeginPeriod) mientras dura el bloque. En otros
  sistemas, o si falla, no hace nada y el valor del `as` es False."""

  def __init__(self, activo=True):
    self.activo = activo

  def __enter__(self):
    self.ok = False
    if self.activo and platform.system() == "Windows":
      try:
        import ctypes
        self.ok = ctypes.windll.winmm.timeBeginPeriod(1) == 0
      except (OSError, AttributeError):
        pass
    return self.ok

  def __exit__(self, *exc):
    if self.ok:
      import ctypes
      ctypes.windll.winmm.timeEndPeriod(1)


def timer_info(n=200):
  """Resolución del reloj y del sleep del sistema: perf_counter y cuánto dura de verdad un time.sleep(1 ms)."""
  def sleeps():
    v = []
    for _ in range(n):
      t = time.perf_counter()
      time.sleep(0.001)
      v.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": pct(v, 50), "p99_ms": pct(v, 99), "max_ms": float(max(v)), "n": n}
  ci = time.get_clock_info("perf_counter")
  info = {"sistema": platform.system(), "perf_counter": {"resolution_s": ci.resolution, "monotonic": ci.monotonic},
          "sleep_1ms": sleeps()}
  with timer_1ms() as ok:
    info["sleep_1ms_con_timeBeginPeriod_1"] = {**sleeps(), "aplica": ok}
  return info


def esperar(hasta, espera="sleep", ka=None):
  """Espera hasta perf_counter() >= hasta.
  sleep: time.sleep del resto (con la resolución del timer del sistema); spin: espera activa; ka: si hay, se lanza
  un kernel chico cada ~1 ms mientras se espera (el sleep se parte en trozos de 1 ms)."""
  if espera == "sleep" and ka is None:
    d = hasta - time.perf_counter()
    if d > 0:
      time.sleep(d)
    return
  while True:
    now = time.perf_counter()
    if now >= hasta:
      return
    if ka is not None:
      ka()
    if espera == "sleep":
      time.sleep(min(0.001, hasta - now))


def run(chain, frames, n_ctx, n_warm, n, pace_ms, budget, espera="sleep", keepalive=False):
  host = torch.from_numpy(frames)
  if chain.cuda:
    host = host.pin_memory()
  total = n_ctx + n_warm + n
  wall, stages = [], []
  ka = chain.keepalive if keepalive else None
  pace_ms = pace_ms * chain.k_cam                  # latentes: un paso de modelo cada k_cam frames de cámara
  next_t = time.perf_counter()
  for i in range(total):
    if i == n_ctx:
      chain.start_imagining()
    if pace_ms:
      esperar(next_t, espera, ka)
      next_t = max(next_t, time.perf_counter()) + pace_ms / 1000
    x = host[i % len(host)][None]
    t = time.perf_counter()
    ev = chain.step(x)
    wall.append((time.perf_counter() - t) * 1000)
    chain.post()
    if ev is not None and i >= n_ctx + n_warm:
      stages.append((ev[0].elapsed_time(ev[1]), ev[1].elapsed_time(ev[2]), ev[2].elapsed_time(ev[3])))
  res = {"first_second": summarize(wall[:n_ctx], budget) if n_ctx else None,
         "steady": summarize(wall[n_ctx + n_warm:], budget)}
  if stages:
    s = np.asarray(stages)
    res["stages_p50_ms"] = {"encoder": float(np.median(s[:, 0])), "model": float(np.median(s[:, 1])),
                            "decoder": float(np.median(s[:, 2]))}
  return res


def cargar_modelo(a, dev):
  if a.model == "none":
    return None, 0
  from src.models import apply_opts, load_model
  model = apply_opts(load_model(a.model, "cpu"), a.opt).eval().to(dev, DTYPES[a.dtype])
  return model, model.num_params() if hasattr(model, "num_params") else sum(p.numel() for p in model.parameters())


def medir_cadena(model, dev, a, frames, pace, graphs, espera="sleep", keepalive=False, sync="event"):
  """Una medición de la cadena. Con graphs, si fallan se repite en eager. -> (res, graphs_usados, graphs_error, n_ctx)."""
  err = None
  need_codec = model is None or not (es_latente(model) or getattr(model, "output", "tokens") == "pixels")
  for g in ([graphs, False] if graphs else [False]):
    codec = Codec(dev, a.codec_dtype, graphs=g) if need_codec else None
    kw = dict(temperature=a.temperature, steps=a.steps, seed=0, graphs=g)
    chain = Chain(model, codec, dev, a.mode, a.k, sync=sync, **kw)
    n_ctx = (getattr(chain.stream, "ctx_frames", None) or chain.stream.ctx) if chain.stream is not None else 0
    try:
      res = run(chain, frames, n_ctx, a.warmup, a.frames, pace, a.budget, espera, keepalive)
      return res, g, err, n_ctx
    except RuntimeError as e:  # p. ej. ROCm sin soporte de graphs: se mide sin graphs y se anota
      if not g:
        raise
      err = str(e).splitlines()[0]
      print(f"   los CUDA graphs fallaron ({err}); se mide sin graphs", flush=True)
      if model is not None and hasattr(model, "use_cuda_graphs"):
        model.use_cuda_graphs(False)


def columna(res):
  """Una columna del protocolo: p50, p99 y % tarde del régimen, más el desglose por etapa."""
  st = res["steady"]
  return {"p50": st["p50"], "p99": st["p99"], "max": st["max"], "late_pct": st["late_pct"], "n": st["n"],
          "stages_p50_ms": res.get("stages_p50_ms")}


VARIANTES = [  # (nombre, espera, keepalive, sync, timer de 1 ms)
  ("sleep", "sleep", False, "event", False),
  ("sleep_timer_1ms", "sleep", False, "event", True),
  ("espera_activa", "spin", False, "event", False),
  ("sleep_keepalive", "sleep", True, "event", False),
  ("espera_activa_keepalive", "spin", True, "event", False),
  ("sleep_sync_device", "sleep", False, "device", False),
  ("espera_activa_sync_poll", "spin", False, "poll", False),
]


def diagnostico(model, dev, a, frames, graphs):
  """Filas del piso de 20 ms: bucle cerrado como referencia y cada variante de espera, kernel de mantenimiento y sincronización."""
  filas = []

  def fila(nombre, pace, t1=False, **kw):
    t = time.perf_counter()
    with timer_1ms(t1) as aplicado:
      res, g, err, _ = medir_cadena(model, dev, a, frames, pace, graphs, **kw)
    f = {"nombre": nombre, "pace_ms": pace, **kw, **columna(res), "graphs_usados": g, "graphs_error": err,
         "seconds": time.perf_counter() - t}
    if t1:
      f["timer_1ms_aplicado"] = bool(aplicado)
    if dev.type != "cuda" and (kw.get("keepalive") or kw.get("sync", "event") != "event"):
      f["nota"] = "sin efecto fuera de CUDA: no hay eventos ni kernel de mantenimiento que probar"
    if t1 and platform.system() != "Windows":
      f["nota_timer"] = "timeBeginPeriod es de Windows: en este sistema no se aplica (se saltea)"
    filas.append(f)
    print(f"   {nombre:26s} p50 {f['p50']:.2f}  p99 {f['p99']:.2f}  tarde {f['late_pct']:.2f}%", flush=True)

  fila("cerrado", 0)
  for nombre, espera, ka, sync, t1 in VARIANTES:
    fila(nombre, a.pace or 50, t1=t1, espera=espera, keepalive=ka, sync=sync)
  return filas


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True, help="spec de load_model, o none (solo codec)")
  ap.add_argument("--name")
  ap.add_argument("--mode", choices=["online", "imagine"], default="online")
  ap.add_argument("--k", type=int, default=1, help="online: predecir el frame t+k")
  ap.add_argument("--steps", type=int)
  ap.add_argument("--temperature", type=float, default=1.0)
  ap.add_argument("--dtype", default="fp16", choices=DTYPES)
  ap.add_argument("--codec-dtype", default="fp16", choices=DTYPES)
  ap.add_argument("--no-graphs", action="store_true")
  ap.add_argument("--frames", type=int, help="frames medidos (default 3000; 600 con --diagnostico)")
  ap.add_argument("--warmup", type=int, default=50)
  ap.add_argument("--pace", type=float, default=0, help="ms entre frames de cámara (0 = bucle cerrado)")
  ap.add_argument("--budget", type=float, help="ms (default 50; 250 en los latentes, un paso de modelo)")
  ap.add_argument("--protocolo", action="store_true", help="las tres columnas de D15 en un JSON (pace por defecto 50)")
  ap.add_argument("--diagnostico", action="store_true", help="variantes del piso de 20 ms a ritmo de cámara")
  ap.add_argument("--opt", action="append", help="campo=valor de la Config del modelo")
  backends.agregar_device(ap, "torch+ort")  # cuda, cpu, rocm...: lo que registra src/infer/backends.py
  ap.add_argument("--tag")
  a = ap.parse_args(argv)
  dev = backends.dispositivo(a.device)
  graphs = not a.no_graphs and dev.type == "cuda"
  tag = a.tag or device_tag(dev)
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = True
  pedido = {"dtype": a.dtype, "codec_dtype": a.codec_dtype}
  if dev.type == "cpu":  # fp16 en CPU no es un camino real (y es lentísimo): se mide en fp32 y el JSON lo declara
    a.dtype = a.codec_dtype = "fp32"

  model, params = cargar_modelo(a, dev)
  latent = model is not None and es_latente(model)
  if a.frames is None:
    a.frames = 600 if a.diagnostico else 3000
  if a.budget is None:
    a.budget = 250.0 if latent else 50.0
  base = a.name or ("codec" if model is None else model.name)
  name = base + (f"_s{a.steps}" if a.steps else "")
  frames = sample_video()
  OUT.mkdir(parents=True, exist_ok=True)
  comun = {"name": name, "model": a.model, "mode": a.mode, "k": a.k, "steps": a.steps, "temperature": a.temperature,
           "dtype": a.dtype, "codec_dtype": a.codec_dtype, "dtype_pedido": pedido, "budget_ms": a.budget, "opts": a.opt, "params": params,
           "unidad": "paso de modelo (250 ms de video)" if latent else "frame", "warmup": a.warmup, "frames": a.frames,
           "device": device_info(dev), "tag": tag, "timestamp": datetime.now().isoformat(timespec="seconds")}
  mode = a.mode + (f"_k{a.k}" if a.mode == "online" and a.k > 1 else "")

  if a.diagnostico:
    pace = a.pace or 50
    print(f"== diagnóstico {name} {a.mode} pace={pace:g} en {tag}", flush=True)
    filas = diagnostico(model, dev, a, frames, graphs)
    out = {**comun, "pace_ms": pace, "timer": timer_info(), "filas": filas,
           "lectura": "comparar p50/p99 de las filas con ritmo contra 'cerrado': la diferencia es el piso a ritmo de cámara"}
    path = OUT / f"diagnostico_{name}__{mode}__pace{pace:g}__{tag}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"-> {path}", flush=True)
    return out

  if a.protocolo:
    from src.bench import load_context, percentiles
    pace = a.pace or 50
    print(f"== protocolo {name} {a.mode} en {tag}", flush=True)
    solo = None
    if model is not None:
      ms, info = medir_modelo(model, load_context().to(dev), a.warmup + a.frames, a.warmup, dev, graphs, a.steps)
      pc = percentiles(ms)
      solo = {"p50": pc["p50"], "p99": pc["p99"], "max": pc["max"],
              "late_pct": float((np.asarray(ms) > a.budget).mean() * 100), "n": pc["n"], **info}
    cols = {}
    for col, p in (("cadena_cerrada", 0), ("cadena_ritmo", pace)):
      res, g, err, _ = medir_cadena(model, dev, a, frames, p, graphs)
      cols[col] = {**columna(res), "pace_ms": p, "graphs_usados": g, "graphs_error": err}
      print(f"   {col:15s} p50 {cols[col]['p50']:.2f}  p99 {cols[col]['p99']:.2f}", flush=True)
    out = {**comun, "graphs": graphs, "modelo_solo": solo, **cols}
    path = OUT / f"latencia_{name}__{tag}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"-> {path}", flush=True)
    return out

  print(f"== {name} {a.mode}{f' k={a.k}' if a.k > 1 else ''} pace={a.pace or 'cerrado'} en {tag}", flush=True)
  res, graphs_used, graphs_error, n_ctx = medir_cadena(model, dev, a, frames, a.pace, graphs)
  st = res["steady"]
  pace = f"pace{a.pace:g}" if a.pace else "closed"
  out = {**comun, "graphs": graphs_used, "graphs_pedidos": graphs, "graphs_error": graphs_error,
         "pace_ms": a.pace, "ctx_frames": n_ctx,
         **res, "fps_p50": 1000 / st["p50"], "realtime_p99": st["p99"] < a.budget,
         "peak_mem_mb": torch.cuda.max_memory_allocated(dev) / 2**20 if dev.type == "cuda" else None}
  path = OUT / f"e2e_{name}__{mode}__{pace}__{tag}.json"
  path.write_text(json.dumps(out, indent=1))
  sp = res.get("stages_p50_ms") or {}
  print(f"   p50 {st['p50']:.2f} ms  p99 {st['p99']:.2f}  max {st['max']:.1f}  tarde {st['late_pct']:.2f}%  "
        f"({', '.join(f'{k} {v:.2f}' for k, v in sp.items())}) -> {path}", flush=True)
  return out


if __name__ == "__main__":
  main()
