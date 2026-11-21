"""Latencia por paso, batch 1 (protocolo único de D15): columna "modelo solo", sin codec ni decoder.

  python -m src.bench --model random:gpt_s --dtype fp16 --graphs
  python -m src.bench --decoder --dtype fp16
Escribe results/bench/<name>__<variante>.json, o <name>__<variante>__<tag>.json con --tag <dispositivo> (otra máquina:
sin el tag, el nombre es el mismo y el resultado pisa el de la RTX 3080 Ti).

Protocolo (el mismo que usa src/infer/bench_e2e.py, que agrega las columnas de la cadena):
- Unidad: un paso del stream (src/infer/stream.py) con batch 1 y el contexto lleno: step() para los tokens, el
  paso del predictor para los latentes (lewm a 4 Hz con LatentStream, vjepa2_ac con VJEPAStream: el paso de modelo
  es de 250 ms de video, no un frame de cámara), observe + predict para convlstm_px.
- fp16 con CUDA graphs donde andan. Si los graphs fallan se mide en eager y el JSON lo declara: "graphs" es lo
  pedido, "graphs_usados" lo que se midió y "graphs_error" el motivo del fallback.
- Se reporta p50 y p99 (además de p10, p90, media y máximo). "median_ms" sigue siendo el p50, que lee report.py.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.common import CTX_FRAMES, device_info, sync
from src.infer import backends

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
CLIPS = Path("results/clips_eval.json")
VAL = Path("data/val_tokens.npy")
OUT = Path("results/bench")


def load_context(k=None):
  """Contexto real (1, 20, 128): el clip 0 de clips_eval.json, o val segmento 0 frames 100..119."""
  seg, start = 0, 100
  if CLIPS.exists():
    c = json.loads(CLIPS.read_text())
    clip = c["clips"][0]
    seg, start = clip["segment"], clip["start"]
    n = c.get("ctx", CTX_FRAMES)
  else:
    n = CTX_FRAMES
  val = np.load(VAL, mmap_mode="r")
  ctx = torch.as_tensor(np.array(val[seg, start:start + n]), dtype=torch.long)[None]
  return ctx[:, -k:] if k else ctx


def quantize_int8(model, dev):
  """CPU: cuantización dinámica de nn.Linear. CUDA: torchao int8 weight-only si está instalado."""
  if dev.type == "cpu":
    from torch.ao.quantization import quantize_dynamic
    return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8), "torch.ao quantize_dynamic"
  try:
    from torchao.quantization import quantize_, Int8WeightOnlyConfig
  except ImportError:
    try:
      from torchao.quantization import quantize_, int8_weight_only as Int8WeightOnlyConfig
    except ImportError:
      raise SystemExit("--int8 en CUDA necesita torchao (no instalado). Usar --device cpu.")
  quantize_(model, Int8WeightOnlyConfig())
  return model, "torchao int8 weight-only"


def percentiles(ms):
  """Resumen de latencias en ms: p10, p50, p90, p99, máximo, media y n."""
  ms = np.asarray(ms, dtype=np.float64)
  return {"p10": float(np.percentile(ms, 10)), "p50": float(np.percentile(ms, 50)), "p90": float(np.percentile(ms, 90)),
          "p99": float(np.percentile(ms, 99)), "max": float(ms.max()), "mean": float(ms.mean()), "n": int(len(ms))}


def es_latente(model):
  """lewm, vjepa2 y vjepa2_ac: predicen embeddings, se miden por paso de modelo."""
  return type(model).__name__ in ("LeWM", "VJEPA2", "VJEPA2AC")


def es_encoder_solo(model):
  """V-JEPA 2 sin acciones no predice en línea: su bench es el target encoder sobre un clip (Docs/modelos/vjepa2.md)."""
  return type(model).__name__ == "VJEPA2"


def unidad(model):
  if es_encoder_solo(model):
    return f"clip del encoder ({model.cfg.frames} frames a 4 Hz, uno por paso de modelo de 250 ms)"
  return "paso de modelo (250 ms de video)" if es_latente(model) else "frame"


def preparar_stream(model, ctx, dev, graphs, steps=None, ctx_frames=None):
  """(stream, paso) con el contexto lleno; paso() es la unidad del protocolo y devuelve su salida."""
  from src.infer.stream import make_stream
  from src.pixels import render
  kw = dict(graphs=graphs)
  if es_encoder_solo(model):                             # el clip se arma una vez en el device; se mide el encoder
    cam = render(ctx)[0]
    img = cam[torch.arange(model.cfg.frames, device=cam.device) % len(cam)][None]
    clip = model.clip(img)
    return None, lambda: model.features(clip)
  if getattr(model, "output", "tokens") == "pixels":    # convlstm_px: entra un frame de cámara y sale el predicho
    cam = render(ctx)[0]
    stream, i = make_stream(model, **kw), [0]
    for f in cam[:stream.ctx_frames]:
      stream.observe_pixels(f[None])

    def paso():
      stream.observe_pixels(cam[i[0] % len(cam)][None])
      i[0] += 1
      return stream.predict()
    return stream, paso
  if es_latente(model):                                  # un paso de modelo = frameskip frames de cámara
    cam = render(ctx)[0]
    stream = make_stream(model, **kw)
    j = 0
    while not stream.full:                               # llena el estado con el video de cámara
      stream.observe_pixels(cam[j % len(cam)][None])
      j += 1
    return stream, stream.step
  stream = make_stream(model, temperature=1.0, steps=steps, seed=0, **kw)
  if ctx_frames:
    stream.ctx_frames = min(stream.ctx_frames, ctx_frames)
  for i in range(ctx.shape[1]):
    stream.observe(ctx[:, i])
  return stream, stream.step


def medir_modelo(model, ctx, n, warmup, dev, graphs, steps=None, ctx_frames=None):
  """Protocolo de latencia del modelo solo: n pasos de stream (los primeros `warmup` se descartan), batch 1.

  Con graphs pedidos, si fallan (RuntimeError) se repite en eager. Devuelve (ms, info) con info = {"graphs",
  "graphs_usados", "graphs_error"}."""
  err = None
  for g in ([graphs, False] if graphs else [False]):
    try:
      with torch.no_grad():
        _, paso = preparar_stream(model, ctx, dev, g, steps, ctx_frames)
        ms = timed((paso() for _ in range(n)), n, dev)[warmup:]
      return ms, {"graphs": graphs, "graphs_usados": g, "graphs_error": err}
    except RuntimeError as e:
      if not g:
        raise
      err = str(e).splitlines()[0]
      print(f"   los CUDA graphs fallaron ({err}); se mide en eager", flush=True)
      if hasattr(model, "use_cuda_graphs"):
        model.use_cuda_graphs(False)


def timed(step_iter, n, dev):
  """Tiempo en ms de cada next() de un iterador, con CUDA events o perf_counter."""
  ms = []
  for _ in range(n):
    if dev.type == "cuda":
      a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
      a.record()
      next(step_iter)
      b.record()
      b.synchronize()
      ms.append(a.elapsed_time(b))
    else:
      sync(dev)   # MPS: asíncrono, sin eventos: perf_counter con sincronización a cada lado
      t = time.perf_counter()
      next(step_iter)
      sync(dev)
      ms.append((time.perf_counter() - t) * 1000)
  return ms


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", help="spec de load_model (gpt2m, transicion, random:gpt_s, ruta.pt)")
  ap.add_argument("--name")
  ap.add_argument("--dtype", default="fp32", choices=DTYPES)
  ap.add_argument("--graphs", action="store_true")
  ap.add_argument("--int8", action="store_true")
  ap.add_argument("--steps", type=int)
  ap.add_argument("--ctx-frames", type=int)
  ap.add_argument("--warmup", type=int, default=20)
  ap.add_argument("--frames", type=int, default=100)
  backends.agregar_device(ap, "torch+ort")  # cuda, cpu, rocm...: lo que registra src/infer/backends.py
  ap.add_argument("--tag", help="nombre del dispositivo (radeon860m...): se agrega al nombre del JSON para no pisar el de otra máquina")
  ap.add_argument("--decoder", action="store_true", help="medir el decoder del VQ-VAE (1 frame por llamada)")
  ap.add_argument("--opt", action="append", help="campo=valor de la Config del modelo (p. ej. rollout=latent); repetible")
  a = ap.parse_args(argv)
  if not a.decoder and not a.model:
    ap.error("falta --model (o --decoder)")
  dev = backends.dispositivo(a.device)
  # defaults explícitos de PyTorch: el fp32 depende de TF32
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = True
  dtype = DTYPES[a.dtype]
  n = a.warmup + a.frames
  if dev.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
  method = None

  if a.decoder:
    from src.viz import load_decoder
    # el one-hot del cuantizador es fp32 fijo: en fp16/bf16 se usa autocast en vez de model.to(dtype)
    model = load_decoder("cpu").to(dev)
    name, ctx_frames = a.name or "decoder", None
    toks = load_context()[:, :1].to(dev)[0]  # (1, 128)

    def it():
      with torch.no_grad(), torch.autocast(dev.type, dtype, enabled=dtype != torch.float32):
        for _ in range(n):
          yield model(toks)
  else:
    from src.models import apply_opts, load_model
    model = apply_opts(load_model(a.model, "cpu"), a.opt).eval()
    if a.int8:
      if a.dtype != "fp32":
        raise SystemExit("--int8 se combina con --dtype fp32")
      model, method = quantize_int8(model.to(dev), dev)
    model = model.to(dev, dtype) if not a.int8 else model
    if a.graphs:
      if not hasattr(model, "use_cuda_graphs") and getattr(model, "output", "tokens") != "pixels":
        raise SystemExit(f"{a.model} no soporta --graphs (sin use_cuda_graphs)")
      if dev.type != "cuda":
        raise SystemExit("--graphs necesita CUDA")
      if hasattr(model, "use_cuda_graphs"):
        model.use_cuda_graphs(True)
    name, ctx_frames = a.name or model.name, a.ctx_frames
    ctx = load_context(a.ctx_frames).to(dev)
    ms, info = medir_modelo(model, ctx, n, a.warmup, dev, a.graphs, a.steps, a.ctx_frames)

  if a.decoder:
    ms, info = timed(it(), n, dev)[a.warmup:], {"graphs": a.graphs, "graphs_usados": False, "graphs_error": None}
  pc = percentiles(ms)
  med = pc["p50"]
  variant = a.dtype + ("_graphs" if info["graphs_usados"] else "") + ("_int8" if a.int8 else "") \
    + (f"_s{a.steps}" if a.steps else "") + (f"_ctx{a.ctx_frames}" if a.ctx_frames else "") \
    + ("_cpu" if dev.type == "cpu" else "")
  res = {
    "name": name, "model": "decoder" if a.decoder else a.model, "variant": variant, "decoder": a.decoder,
    "median_ms": med, "p50_ms": med, "p99_ms": pc["p99"], "p10_ms": pc["p10"], "p90_ms": pc["p90"], "max_ms": pc["max"],
    "mean_ms": pc["mean"], "fps": 1000 / med, "realtime": med < (250 if es_latente(model) else 50), "unidad": unidad(model),
    "params": sum(p.numel() for p in model.parameters()),
    "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None,
    "dtype": a.dtype, "graphs": a.graphs, "graphs_usados": info["graphs_usados"], "graphs_error": info["graphs_error"], "int8": a.int8, "int8_method": method, "steps": a.steps,
    "ctx_frames": ctx_frames, "warmup": a.warmup, "frames": a.frames,
    "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    "device": device_info(dev), "tag": a.tag, "torch": torch.__version__, "timestamp": datetime.now().isoformat(timespec="seconds"),
  }
  OUT.mkdir(parents=True, exist_ok=True)
  path = OUT / f"{name}__{variant}{f'__{a.tag}' if a.tag else ''}.json"
  path.write_text(json.dumps(res, indent=1))
  print(f"{name} {variant}: p50 {med:.2f} ms (p99 {res['p99_ms']:.2f}, p10 {res['p10_ms']:.2f}) "
        f"{res['fps']:.1f} FPS, tiempo real: {'si' if res['realtime'] else 'no'} -> {path}")
  return res


if __name__ == "__main__":
  main()
