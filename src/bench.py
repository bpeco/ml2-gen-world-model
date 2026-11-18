"""Latencia por frame, batch 1 (protocolo 2.4 del plan): mediana de next_frame sin el decoder.

  python -m src.bench --model random:gpt_s --dtype fp16 --graphs
  python -m src.bench --decoder --dtype fp16
Escribe results/bench/<name>__<variante>.json
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.common import CTX_FRAMES, device_info

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
      t = time.perf_counter()
      next(step_iter)
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
  ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
  ap.add_argument("--decoder", action="store_true", help="medir el decoder del VQ-VAE (1 frame por llamada)")
  ap.add_argument("--opt", action="append", help="campo=valor de la Config del modelo (p. ej. rollout=latent); repetible")
  a = ap.parse_args(argv)
  if not a.decoder and not a.model:
    ap.error("falta --model (o --decoder)")
  dev = torch.device(a.device)
  # defaults explícitos de PyTorch: el fp32 depende de TF32
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = True
  dtype = DTYPES[a.dtype]
  n = a.warmup + a.frames
  if dev.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
  gen = torch.Generator(dev).manual_seed(0)
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
      if not hasattr(model, "use_cuda_graphs"):
        raise SystemExit(f"{a.model} no soporta --graphs (sin use_cuda_graphs)")
      if dev.type != "cuda":
        raise SystemExit("--graphs necesita CUDA")
      model.use_cuda_graphs(True)
    name, ctx_frames = a.name or model.name, a.ctx_frames
    ctx = load_context(a.ctx_frames).to(dev)

    def it():
      yield from model.generate(ctx, n, temperature=1.0, steps=a.steps, generator=gen)

  ms = timed(it(), n, dev)[a.warmup:]
  med = float(np.median(ms))
  variant = a.dtype + ("_graphs" if a.graphs else "") + ("_int8" if a.int8 else "") \
    + (f"_s{a.steps}" if a.steps else "") + (f"_ctx{a.ctx_frames}" if a.ctx_frames else "") \
    + ("_cpu" if dev.type == "cpu" else "")
  res = {
    "name": name, "model": "decoder" if a.decoder else a.model, "variant": variant, "decoder": a.decoder,
    "median_ms": med, "p10_ms": float(np.percentile(ms, 10)), "p90_ms": float(np.percentile(ms, 90)),
    "mean_ms": float(np.mean(ms)), "fps": 1000 / med, "realtime": med < 50,
    "params": sum(p.numel() for p in model.parameters()),
    "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None,
    "dtype": a.dtype, "graphs": a.graphs, "int8": a.int8, "int8_method": method, "steps": a.steps,
    "ctx_frames": ctx_frames, "warmup": a.warmup, "frames": a.frames,
    "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    "device": device_info(dev), "torch": torch.__version__, "timestamp": datetime.now().isoformat(timespec="seconds"),
  }
  OUT.mkdir(parents=True, exist_ok=True)
  path = OUT / f"{name}__{variant}.json"
  path.write_text(json.dumps(res, indent=1))
  print(f"{name} {variant}: mediana {med:.2f} ms (p10 {res['p10_ms']:.2f}, p90 {res['p90_ms']:.2f}) "
        f"{res['fps']:.1f} FPS, tiempo real: {'si' if res['realtime'] else 'no'} -> {path}")
  return res


if __name__ == "__main__":
  main()
