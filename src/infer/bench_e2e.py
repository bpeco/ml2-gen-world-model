"""Latencia de punta a punta por frame: video en la CPU -> encoder -> world model -> decoder -> video en la CPU.

  python -m src.infer.bench_e2e --model maskgit_s --steps 4                  # modo en línea, bucle cerrado
  python -m src.infer.bench_e2e --model maskgit_s --mode imagine --pace 50   # imaginación, un frame cada 50 ms
  python -m src.infer.bench_e2e --model lewm --mode online --k 5             # en línea, predice el frame t+5
  python -m src.infer.bench_e2e --model none                                 # solo el codec (piso de la cadena)
  python -m src.infer.bench_e2e --model random:maskgit_m --steps 2           # sin checkpoint (la latencia no depende de los pesos)
  python -m src.infer.bench_e2e --model lewm --opt rollout=latent            # LeWM: la cámara va directo al modelo

ConvLSTM (convlstm_px, predice píxeles): la cámara entra directo y sale el frame predicho, sin encoder ni decoder
de comma (las etapas "encoder" y "decoder" quedan en ~0; el "modelo" incluye la copia a la salida).

LeWM (ve píxeles): en línea, el frame de la cámara entra al modelo sin el encoder de comma (la
etapa "encoder" queda en ~0). Para imaginar conviene rollout=latent: con rollout=token cada frame generado se
decodifica y se vuelve a codificar desde píxeles.

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
Escribe results/infer/e2e_<nombre>__<modo>[_k<k>]__<pace>__<dispositivo>.json
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.common import device_info
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
  """Codec + stream del modelo (o solo codec con model=None), con un paso por frame y su desglose."""

  def __init__(self, model, codec: Codec, dev, mode="online", k=1, **stream_kw):
    self.codec, self.dev, self.mode, self.k = codec, dev, mode, k
    self.stream = make_stream(model, **stream_kw) if model is not None else None
    self.cuda = dev.type == "cuda"
    self.out_host = torch.empty(1, 128, 256, 3, dtype=torch.uint8, pin_memory=self.cuda)
    self.pix_out = getattr(model, "output", "tokens") == "pixels"  # el modelo ya predice píxeles: sin decoder de comma
    self._imagine = None
    self.last_tok = None

  def _ev(self):
    if not self.cuda:
      return None
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e

  def step(self, frame_host):
    """Un frame de entrada (1, 128, 256, 3) uint8 en el host -> frame predicho en self.out_host.
    Devuelve los eventos (inicio, tras encoder, tras modelo, fin) o None en CPU."""
    e0 = self._ev()
    encode = self.mode == "online" or self.stream is None or self._imagine is None
    pixels = self.stream is not None and hasattr(self.stream, "observe_pixels")
    if encode:
      x = frame_host.to(self.dev, non_blocking=True)
      if not pixels:  # LeWM no necesita el encoder de comma: la imagen va directo al modelo
        tok = self.codec.encode(x)
    e1 = self._ev()  # el paso del modelo incluye observe (LeWM: codificar el latente)
    if encode and self.stream is not None:
      self.stream.observe_pixels(x) if pixels else self.stream.observe(tok)
    if self.stream is None:
      pred = tok                                  # solo codec: se reconstruye el frame de entrada
    elif self.mode == "online" or self._imagine is None:  # imagine: durante el contexto inicial, como en línea
      pred = self.stream.predict(self.k)
    else:
      pred = next(self._imagine)
    e2 = self._ev()
    self.out_host.copy_(pred if self.pix_out else self.codec.decode(pred), non_blocking=True)
    e3 = self._ev()
    if self.cuda:
      e3.synchronize()
    return (e0, e1, e2, e3) if self.cuda else None

  def start_imagining(self):
    """Modo imagine: a partir de acá el modelo genera y el encoder deja de correr."""
    if self.mode == "imagine" and self.stream is not None:
      self._imagine = self.stream.imagine(10 ** 9)


def run(chain, frames, n_ctx, n_warm, n, pace_ms, budget):
  host = torch.from_numpy(frames)
  if chain.cuda:
    host = host.pin_memory()
  total = n_ctx + n_warm + n
  wall, stages = [], []
  next_t = time.perf_counter()
  for i in range(total):
    if i == n_ctx:
      chain.start_imagining()
    if pace_ms:
      now = time.perf_counter()
      if next_t > now:
        time.sleep(next_t - now)
      next_t = max(next_t, time.perf_counter()) + pace_ms / 1000
    x = host[i % len(host)][None]
    t = time.perf_counter()
    ev = chain.step(x)
    wall.append((time.perf_counter() - t) * 1000)
    if ev is not None and i >= n_ctx + n_warm:
      stages.append((ev[0].elapsed_time(ev[1]), ev[1].elapsed_time(ev[2]), ev[2].elapsed_time(ev[3])))
  res = {"first_second": summarize(wall[:n_ctx], budget) if n_ctx else None,
         "steady": summarize(wall[n_ctx + n_warm:], budget)}
  if stages:
    s = np.asarray(stages)
    res["stages_p50_ms"] = {"encoder": float(np.median(s[:, 0])), "model": float(np.median(s[:, 1])),
                            "decoder": float(np.median(s[:, 2]))}
  return res


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
  ap.add_argument("--frames", type=int, default=3000)
  ap.add_argument("--warmup", type=int, default=50)
  ap.add_argument("--pace", type=float, default=0, help="ms entre frames (0 = bucle cerrado)")
  ap.add_argument("--budget", type=float, default=50)
  ap.add_argument("--opt", action="append", help="campo=valor de la Config del modelo (p. ej. rollout=latent)")
  ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  ap.add_argument("--tag")
  a = ap.parse_args(argv)
  dev = torch.device(a.device)
  graphs = not a.no_graphs and dev.type == "cuda"
  tag = a.tag or device_tag(dev)
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = True

  model, params = None, 0
  if a.model != "none":
    from src.models import apply_opts, load_model
    model = apply_opts(load_model(a.model, "cpu"), a.opt).eval().to(dev, DTYPES[a.dtype])
    params = model.num_params() if hasattr(model, "num_params") else sum(p.numel() for p in model.parameters())
  base = a.name or ("codec" if model is None else model.name)
  name = base + (f"_s{a.steps}" if a.steps else "")
  frames = sample_video()
  print(f"== {name} {a.mode}{f' k={a.k}' if a.k > 1 else ''} pace={a.pace or 'cerrado'} en {tag}", flush=True)
  graphs_error = None
  for g in ([graphs, False] if graphs else [False]):
    codec = Codec(dev, a.codec_dtype, graphs=g)
    kw = dict(temperature=a.temperature, steps=a.steps, seed=0, graphs=g)
    chain = Chain(model, codec, dev, a.mode, a.k, **kw)
    n_ctx = chain.stream.ctx_frames if chain.stream is not None else 0
    try:
      res = run(chain, frames, n_ctx, a.warmup, a.frames, a.pace, a.budget)
      graphs = g
      break
    except RuntimeError as e:  # p. ej. ROCm sin soporte de graphs: se mide sin graphs y se anota
      if not g:
        raise
      graphs_error = str(e).splitlines()[0]
      print(f"   los CUDA graphs fallaron ({graphs_error}); se mide sin graphs", flush=True)
      if model is not None and hasattr(model, "use_cuda_graphs"):
        model.use_cuda_graphs(False)
  st = res["steady"]
  mode = a.mode + (f"_k{a.k}" if a.mode == "online" and a.k > 1 else "")
  pace = f"pace{a.pace:g}" if a.pace else "closed"
  out = {"name": name, "model": a.model, "mode": a.mode, "k": a.k, "steps": a.steps, "temperature": a.temperature,
         "dtype": a.dtype, "codec_dtype": a.codec_dtype, "graphs": graphs, "graphs_error": graphs_error,
         "pace_ms": a.pace, "budget_ms": a.budget,
         "opts": a.opt, "params": params, "ctx_frames": n_ctx, "warmup": a.warmup,
         **res, "fps_p50": 1000 / st["p50"], "realtime_p99": st["p99"] < a.budget,
         "peak_mem_mb": torch.cuda.max_memory_allocated(dev) / 2**20 if dev.type == "cuda" else None,
         "device": device_info(dev), "tag": tag, "timestamp": datetime.now().isoformat(timespec="seconds")}
  OUT.mkdir(parents=True, exist_ok=True)
  path = OUT / f"e2e_{name}__{mode}__{pace}__{tag}.json"
  path.write_text(json.dumps(out, indent=1))
  sp = res.get("stages_p50_ms") or {}
  print(f"   p50 {st['p50']:.2f} ms  p99 {st['p99']:.2f}  max {st['max']:.1f}  tarde {st['late_pct']:.2f}%  "
        f"({', '.join(f'{k} {v:.2f}' for k, v in sp.items())}) -> {path}", flush=True)
  return out


if __name__ == "__main__":
  main()
