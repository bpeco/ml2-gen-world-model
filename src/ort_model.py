"""WorldModel sobre ONNX Runtime: una pasada de ventana fija de un MaskGIT exportado (extra D8).

  python -m src.ort_model --onnx checkpoints/onnx/<name>_k5.onnx --provider cpu|cuda|dml|vitisai [--n 100]
Mide ms por pasada (mediana tras warm-up) y escribe results/bench/ort_<name>_k<K>__<provider>.json.

Sin caché KV: cada pasada recalcula el contexto, que es lo que midió el plan (2.5). La ventana es
[K-1 frames de contexto, frame nuevo con BOS + MASK x128]; las S pasadas siguen el mismo esquema
coseno / confianza que src/models/maskgit.py. El muestreo va en torch sobre CPU.

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
from src.models.base import WorldModel, sample_logits
from src.models.maskgit import CODES, n_masked

PROVIDERS = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider", "dml": "DmlExecutionProvider"}
VITIS = "VitisAIExecutionProvider"
OUT = Path("results/bench")


def make_session(path, provider="cpu", so=None):
  """InferenceSession con el provider pedido (cpu|cuda|dml|vitisai); error claro si no está disponible.
  so: SessionOptions propias (p. ej. con enable_profiling para ver qué nodos caen en la CPU)."""
  so = so or ort.SessionOptions()
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
  if provider not in PROVIDERS:
    raise ValueError(f"provider no soportado: {provider} (cpu|cuda|dml|vitisai)")
  if PROVIDERS[provider] not in ort.get_available_providers():
    raise RuntimeError(f"{PROVIDERS[provider]} no está disponible; hay {ort.get_available_providers()}")
  eps = [PROVIDERS[provider]] + (["CPUExecutionProvider"] if provider != "cpu" else [])
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
    fixed = torch.zeros(1, CODES, dtype=torch.bool)
    n_prev = CODES
    for s in range(1, S + 1):
      lg = self.window_logits(seq)[:, -CODES:]  # la posición p predice el token en p
      samp = sample_logits(lg, temperature, generator)
      lp = torch.log_softmax(lg / temperature if temperature > 0 else lg, -1).gather(-1, samp[..., None])[..., 0]
      if temperature > 0 and self.choice_temp > 0 and s < S:
        gum = -torch.log(torch.empty_like(lp).exponential_(1, generator=generator))
        lp = lp + self.choice_temp * (1 - s / S) * gum
      n_s = n_masked(s, S)
      k = n_prev - n_s
      lp = lp.masked_fill(fixed, float("-inf"))
      if k > 0:
        fixed.scatter_(1, lp.topk(k, dim=-1).indices, True)
      new = seq[:, -CODES:]
      seq[:, -CODES:] = torch.where(fixed & (new == MASK), samp, new)
      n_prev = n_s
    return seq[:, -CODES:].clone()


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


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--onnx", required=True)
  ap.add_argument("--provider", default="cpu", choices=[*PROVIDERS, "vitisai"])
  ap.add_argument("--n", type=int, default=100)
  ap.add_argument("--warmup", type=int, default=20)
  a = ap.parse_args()
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
