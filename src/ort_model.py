"""WorldModel sobre ONNX Runtime: una pasada de ventana fija de un MaskGIT exportado (extra D8).

  python -m src.ort_model --onnx checkpoints/onnx/<name>_k5.onnx --provider cpu|cuda|dml|vitisai|npu [--n 100]
Mide ms por pasada (mediana tras warm-up) y escribe results/bench/ort_<name>_k<K>__<provider>.json.

Sin caché KV: cada pasada recalcula el contexto, que es lo que midió el plan (2.5). La ventana es
[K-1 frames de contexto, frame nuevo con BOS + MASK x128]; las S pasadas siguen el mismo esquema
coseno / confianza que src/models/maskgit.py. El muestreo va en torch sobre CPU.

También mide el paso de decodificación de Genie que escribe `src.export_onnx --model random:genie_s` (entradas cur, k0, v0...,
sin "tokens") y su prefill (`--parte prefill`, entrada ctx): ms por llamada y acuerdo con ORT en CPU sobre las mismas
entradas (la NPU calcula en bf16).

NPU (provider "vitisai"): SIN PROBAR (esta máquina no tiene NPU). Sigue la receta del plan 2.5:
ML2_VITISAI_EP_DIR = carpeta (sin espacios, copia de ExecutionProvider del paquete
Microsoft.WinML.AMD.NPU.EP.2) con onnxruntime_vitisai_ep.dll; se registra con
ort.register_execution_provider_library y el dispositivo NPU se elige con add_provider_for_devices.
"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from src.bench import load_context
from src.common import BOS, MASK, TPF, device_info, with_bos
from src.infer import backends
from src.models.base import WorldModel
from src.models.maskgit import CODES, decode_frame

VITIS = backends.NOMBRE_VITIS
OUT = Path("results/bench")


def make_session(path, provider="cpu", so=None):
  """InferenceSession con el backend pedido: un nombre de src/infer/backends.py (cpu, cuda, dml, npu...) o su provider
  corto (vitisai); error claro si no está disponible.
  so: SessionOptions propias (p. ej. con enable_profiling para ver qué nodos caen en la CPU)."""
  so = so or ort.SessionOptions()
  b = backends.resolver_ort(provider)
  provider = b.ort_provider
  if provider == "vitisai":  # SIN PROBAR (sin NPU en esta máquina)
    ep_dir = os.environ.get("ML2_VITISAI_EP_DIR")
    if not ep_dir:
      raise RuntimeError("vitisai: definir ML2_VITISAI_EP_DIR (carpeta con onnxruntime_vitisai_ep.dll, sin espacios)")
    dll = Path(ep_dir) / "onnxruntime_vitisai_ep.dll"
    if not dll.exists():
      raise RuntimeError(f"vitisai: no existe {dll}")
    for f in ("register_execution_provider_library", "get_ep_devices"):
      if not hasattr(ort, f):
        raise RuntimeError(f"vitisai: este onnxruntime {ort.__version__} no tiene ort.{f} (usar onnxruntime-windowsml)")
    if not hasattr(so, "add_provider_for_devices"):
      raise RuntimeError("vitisai: este onnxruntime no tiene SessionOptions.add_provider_for_devices")
    try:
      ort.register_execution_provider_library(VITIS, str(dll))
    except Exception as e:  # ya registrado en esta sesión
      if "already" not in str(e).lower():
        raise
    npus = [d for d in ort.get_ep_devices()
            if d.ep_name == VITIS and d.device.type == ort.OrtHardwareDeviceType.NPU]
    if not npus:
      raise RuntimeError("vitisai: el EP no expone ningún dispositivo NPU (driver 32.0.203.280+?)")
    so.add_provider_for_devices(npus, {})
    return ort.InferenceSession(str(path), sess_options=so)
  if not b.ort_ep:
    raise ValueError(f"el backend {b.nombre!r} no declara ort_ep (el execution provider de ONNX Runtime)")
  if b.ort_ep not in ort.get_available_providers():
    raise RuntimeError(f"{b.ort_ep} no está disponible; hay {ort.get_available_providers()}"
                       + (f"; instalar: {b.instalar}" if b.instalar else ""))
  eps = [b.ort_ep] + (["CPUExecutionProvider"] if b.ort_ep != "CPUExecutionProvider" else [])
  return ort.InferenceSession(str(path), sess_options=so, providers=eps)


class OrtMaskGIT(WorldModel):
  """MaskGIT de ventana fija sobre ONNX Runtime. ctx_frames = K - 1; solo batch 1."""
  name = "ort_maskgit"
  has_likelihood = False

  def __init__(self, path, provider="cpu", choice_temp=1.0):
    super().__init__()
    self.sess = make_session(path, provider)
    self.provider = provider
    self.choice_temp = choice_temp
    L = self.sess.get_inputs()[0].shape[1]
    assert isinstance(L, int) and L % TPF == 0, f"entrada ONNX de forma fija (1, K*{TPF}), no {L}"
    self.K = L // TPF
    self.ctx_frames = self.K - 1
    self.name = Path(path).stem

  def window_logits(self, seq):
    """(1, K*129) long -> (1, K*129, V) float32."""
    return torch.from_numpy(self.sess.run(None, {"tokens": seq.numpy()})[0])

  @torch.no_grad()
  def next_frame(self, ctx, temperature=1.0, steps=None, generator=None):
    S = steps or 8
    if ctx.shape[0] != 1 or ctx.shape[1] < self.ctx_frames:
      raise ValueError(f"OrtMaskGIT: batch 1 y al menos {self.ctx_frames} frames de contexto, llegó {tuple(ctx.shape)}")
    ctx = ctx[:, -self.ctx_frames:].cpu()
    seq = torch.cat([with_bos(ctx), torch.full((1, TPF), MASK, dtype=torch.long)], 1)
    seq[:, -TPF] = BOS

    def logits_fn(cur):
      seq[:, -CODES:] = cur
      return self.window_logits(seq)[:, -CODES:]  # la posición p predice el token en p

    cur = torch.full((1, CODES), MASK, dtype=torch.long)
    return decode_frame(logits_fn, S, temperature, self.choice_temp, cur, generator).clone()


def bench_ort(sess, K, n=100, warmup=20):
  """ms por pasada de la ventana de K frames: lista de n tiempos tras warm-up."""
  ctx = load_context(K - 1)
  seq = torch.cat([with_bos(ctx), torch.full((1, TPF), MASK, dtype=torch.long)], 1)
  seq[:, -TPF] = BOS
  feed = {"tokens": seq.numpy()}
  ms = []
  for i in range(warmup + n):
    t = time.perf_counter()
    sess.run(None, feed)
    ms.append((time.perf_counter() - t) * 1000)
  return ms[warmup:]


def feed_paso(sess, seed=0):
  """Entradas de un paso de Genie (cur todo MASK y un (k, v) aleatorio por capa) o de su prefill (ctx de tokens aleatorios).
  La latencia no depende de los valores."""
  rng = np.random.default_rng(seed)
  feed = {}
  for i in sess.get_inputs():
    if i.type == "tensor(int64)":
      feed[i.name] = np.full(i.shape, MASK, np.int64) if i.name == "cur" else rng.integers(0, 1024, i.shape).astype(np.int64)
    else:
      feed[i.name] = rng.standard_normal(i.shape).astype(np.float16 if "float16" in i.type else np.float32)
  return feed


def bench_paso(sess, feed, n=100, warmup=20):
  """ms por llamada (n tiempos tras el warm-up) y las salidas de la última corrida."""
  ms = []
  for _ in range(warmup + n):
    t = time.perf_counter()
    out = sess.run(None, feed)
    ms.append((time.perf_counter() - t) * 1000)
  return ms[warmup:], out


def main_paso(a, cpu):
  """Mide un paso de Genie (o su prefill) exportado en el provider pedido y lo compara con ORT en CPU (cpu: la sesión ya creada)."""
  path = Path(a.onnx)
  sess = cpu if backends.resolver_ort(a.provider).ort_provider == "CPUExecutionProvider" else make_session(path, a.provider)
  feed = feed_paso(sess)
  ms, out = bench_paso(sess, feed, a.n, a.warmup)
  ref = [r.astype(np.float32) for r in cpu.run(None, feed)]
  out = [o.astype(np.float32) for o in out]
  med = float(np.median(ms))
  name = f"ort_{path.stem}"
  prefill = "cur" not in feed
  ctx = feed["ctx"].shape[1] if prefill else next(v.shape[2] for k, v in feed.items() if k.startswith("k"))
  vs = {"max_abs_diff": max(float(np.abs(o - r).max()) for o, r in zip(out, ref)),
        "rel_l2": float(np.sqrt(sum(((o - r) ** 2).sum() for o, r in zip(out, ref)) / sum((r ** 2).sum() for r in ref))),
        # el argmax solo dice algo sobre logits (un paso); las salidas del prefill son (k, v)
        "acuerdo_argmax": None if prefill else float((out[0].argmax(-1) == ref[0].argmax(-1)).mean())}
  res = {"name": name, "model": str(path), "variant": a.provider, "decoder": False, "unit": "prefill" if prefill else "paso",
         "median_ms": med, "p10_ms": float(np.percentile(ms, 10)), "p90_ms": float(np.percentile(ms, 90)),
         "mean_ms": float(np.mean(ms)), "fps": 1000 / med, "warmup": a.warmup, "frames": a.n, "provider": a.provider,
         "contexto": [ctx], "vs_cpu": vs,
         "ort": ort.__version__, "ort_providers": sess.get_providers(), "device": device_info("cpu"),
         "timestamp": datetime.now().isoformat(timespec="seconds")}
  OUT.mkdir(parents=True, exist_ok=True)
  dest = OUT / f"{name}__{a.provider}.json"
  dest.write_text(json.dumps(res, indent=1))
  print(f"{name} {a.provider}: {med:.2f} ms por {res['unit']} (p10 {res['p10_ms']:.2f}, p90 {res['p90_ms']:.2f}); "
        f"vs ORT CPU: diff máx {vs['max_abs_diff']:.2e}, error relativo {vs['rel_l2']:.1e}"
        + ("" if prefill else f", argmax {vs['acuerdo_argmax'] * 100:.1f} %") + f" -> {dest}")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--onnx", required=True)
  ap.add_argument("--provider", default="cpu", choices=[*backends.nombres("ort"), *backends.ort_providers()],
                  help="backend de ONNX Runtime (src/infer/backends.py)")
  ap.add_argument("--n", type=int, default=100)
  ap.add_argument("--warmup", type=int, default=20)
  a = ap.parse_args()
  cpu = make_session(a.onnx, "cpu")
  if "tokens" not in {i.name for i in cpu.get_inputs()}:   # un paso de Genie, no una ventana de MaskGIT
    return main_paso(a, cpu)
  m = OrtMaskGIT(a.onnx, a.provider)
  ms = bench_ort(m.sess, m.K, a.n, a.warmup)
  med = float(np.median(ms))
  name = f"ort_{m.name}"
  res = {
    "name": name, "model": str(a.onnx), "variant": a.provider, "decoder": False, "unit": "pass",
    "median_ms": med, "p10_ms": float(np.percentile(ms, 10)), "p90_ms": float(np.percentile(ms, 90)),
    "mean_ms": float(np.mean(ms)), "fps": 1000 / med, "realtime": med < 50, "ctx_frames": m.ctx_frames,
    "window_frames": m.K, "tokens": m.K * TPF, "warmup": a.warmup, "frames": a.n, "provider": a.provider,
    "ort": ort.__version__, "ort_providers": m.sess.get_providers(), "device": device_info("cpu"),
    "timestamp": datetime.now().isoformat(timespec="seconds"),
  }
  OUT.mkdir(parents=True, exist_ok=True)
  path = OUT / f"{name}__{a.provider}.json"
  path.write_text(json.dumps(res, indent=1))
  print(f"{name} {a.provider}: {med:.2f} ms por pasada (p10 {res['p10_ms']:.2f}, p90 {res['p90_ms']:.2f}) -> {path}")


if __name__ == "__main__":
  main()
