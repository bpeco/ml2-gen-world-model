"""Cadena de punta a punta con ONNX Runtime, eligiendo el dispositivo de cada etapa (laptop: NPU e iGPU; plan, fase 7).

  python -m src.infer.export_codec                                   # una vez: exporta a checkpoints/onnx/
  python -m src.infer.bench_ort --enc cpu --dec cpu                  # solo codec
  python -m src.infer.bench_ort --enc vitisai --dec dml --model convlstm --model-provider vitisai --profile
  python -m src.infer.bench_ort --enc vitisai --dec vitisai --fp16

Etapas por frame (modo en línea, batch 1): imagen uint8 -> encoder (ORT) -> argmin al codebook (CPU, numpy) ->
[paso de ConvLSTM (ORT) y muestreo en la CPU] -> lookup del codebook (CPU) -> decoder (ORT) -> uint8.
Entre etapas cruzan 128 KB (latentes) o 96 KB (imagen): no hace falta IO binding.

Providers: cpu | dml (DirectML, iGPU) | cuda | vitisai (NPU XDNA 2; ver src/ort_model.make_session y la sección 2.5
del plan del proyecto: SIN PROBAR en esta máquina). --profile activa el profiler de ORT y cuenta cuántos nodos y
cuánto tiempo cae en cada provider: lo que la NPU no soporta vuelve a la CPU.
Para medir el encoder y el decoder a la vez en la misma NPU o iGPU (comparten el ancho de banda de memoria), correr
con las dos etapas en el mismo provider; para medirlos solos, --only enc|dec.
Escribe results/infer/ort_<tag>.json (tag por defecto: enc-<p>_model-<p>_dec-<p>).
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from src.infer.video import sample_video

ONNX = Path("checkpoints/onnx")
OUT = Path("results/infer")


def session(path, provider, profile):
  import onnxruntime as ort
  from src.ort_model import make_session
  so = ort.SessionOptions()
  if profile:
    so.enable_profiling = True
    so.profile_file_prefix = str(OUT / f"ort_profile_{path.stem}_{provider}")
  return make_session(path, provider, so)


def placement(prof_path):
  """{provider: {"nodes": n, "ms": tiempo total}} a partir del JSON del profiler de ORT."""
  ev = json.loads(Path(prof_path).read_text())
  nodes, ms = Counter(), defaultdict(float)
  seen = set()
  for e in ev:
    if e.get("cat") != "Node" or not e.get("name", "").endswith("_kernel_time"):
      continue
    prov = e.get("args", {}).get("provider", "?")
    name = e["name"][:-len("_kernel_time")]
    if name not in seen:
      nodes[prov] += 1
      seen.add(name)
    ms[prov] += e.get("dur", 0) / 1000
  return {p: {"nodes": nodes[p], "ms": round(ms[p], 2)} for p in nodes}


class OrtChain:
  def __init__(self, a):
    sfx = "_fp16" if a.fp16 else ""
    self.cb = np.load(ONNX / "codec_codebook.npy").astype(np.float32)          # (1024, 256)
    self.cb2 = (self.cb ** 2).sum(1)
    self.enc = session(ONNX / f"codec_encoder_body{sfx}.onnx", a.enc, a.profile) if a.only != "dec" else None
    self.dec = session(ONNX / f"codec_decoder_body{sfx}.onnx", a.dec, a.profile) if a.only != "enc" else None
    self.model = None
    if a.model:
      self.model = session(ONNX / f"{a.model}_step{sfx}.onnx", a.model_provider, a.profile)
      self.emb = np.load(ONNX / f"{a.model}_tok_emb.npy").astype(np.float32)   # (1024, emb)
      self.start = np.load(ONNX / f"{a.model}_start.npy").astype(np.float32)   # (emb, 8, 16)
      self.state_names = [i.name for i in self.model.get_inputs()[1:]]
      shp = self.model.get_inputs()[1].shape
      self.state = [np.zeros(shp, np.float32) for _ in self.state_names]
      self.lg = None
    self.rng = np.random.default_rng(0)
    self.temperature = a.temperature

  def encode(self, img):            # (128, 256, 3) uint8 -> (128,) tokens
    x = img.transpose(2, 0, 1)[None].astype(np.float32)
    h = self.enc.run(None, {"img": x})[0][0]                                      # (128, 256)
    d = (h ** 2).sum(1, keepdims=True) + self.cb2 - 2 * h @ self.cb.T
    return d.argmin(1)

  def _step(self, x):
    out = self.model.run(None, {"x": x[None], **dict(zip(self.state_names, self.state))})
    self.lg, self.state = out[0][0], out[1:]

  def observe(self, tok):
    if self.lg is None:
      self._step(self.start)
    self._step(self.emb[tok].reshape(8, 16, -1).transpose(2, 0, 1))

  def predict(self):                # muestreo como sample_logits: argmax(p / q), q ~ Exp(1)
    lg = self.lg.astype(np.float64)
    if self.temperature == 0:
      return lg.argmax(-1)
    z = lg / self.temperature
    p = np.exp(z - z.max(-1, keepdims=True))
    return (p / self.rng.exponential(size=p.shape)).argmax(-1)

  def decode(self, tok):            # (128,) -> (128, 256, 3) uint8
    y = self.dec.run(None, {"emb": self.cb[tok][None]})[0][0]
    return np.clip(y, 0, 255).astype(np.uint8).transpose(1, 2, 0)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--enc", default="cpu")
  ap.add_argument("--dec", default="cpu")
  ap.add_argument("--model", help="corrida de ConvLSTM exportada (p. ej. convlstm); sin --model, solo codec")
  ap.add_argument("--model-provider", default="cpu")
  ap.add_argument("--only", choices=["enc", "dec"], help="medir una sola etapa del codec")
  ap.add_argument("--fp16", action="store_true", help="usar los ONNX exportados con --fp16")
  ap.add_argument("--temperature", type=float, default=1.0)
  ap.add_argument("--frames", type=int, default=500)
  ap.add_argument("--warmup", type=int, default=20)
  ap.add_argument("--profile", action="store_true")
  ap.add_argument("--tag")
  a = ap.parse_args()
  OUT.mkdir(parents=True, exist_ok=True)
  c = OrtChain(a)
  frames = sample_video()
  times = defaultdict(list)
  tok = np.zeros(128, np.int64)
  for i in range(a.warmup + a.frames):
    img = frames[i % len(frames)]
    t0 = time.perf_counter()
    if c.enc is not None:
      tok = c.encode(img)
    t1 = time.perf_counter()
    if c.model is not None:
      c.observe(tok)
      tok = c.predict()
    t2 = time.perf_counter()
    if c.dec is not None:
      c.decode(tok)
    t3 = time.perf_counter()
    if i >= a.warmup:
      for k, v in (("encoder", t1 - t0), ("model", t2 - t1), ("decoder", t3 - t2), ("total", t3 - t0)):
        times[k].append(v * 1000)
  stat = lambda ms: {"p50": float(np.median(ms)), "p99": float(np.percentile(ms, 99)), "mean": float(np.mean(ms))}
  tag = a.tag or (f"enc-{a.enc if c.enc else 'no'}_model-{a.model_provider if a.model else 'no'}"
                  f"_dec-{a.dec if c.dec else 'no'}" + ("_fp16" if a.fp16 else ""))
  import onnxruntime as ort
  res = {"tag": tag, "providers": {"enc": a.enc, "model": a.model_provider if a.model else None, "dec": a.dec},
         "model": a.model, "fp16": a.fp16, "only": a.only, "frames": a.frames,
         "latency_ms": {k: stat(v) for k, v in times.items()}, "onnxruntime": ort.__version__,
         "timestamp": datetime.now().isoformat(timespec="seconds")}
  from src.common import device_info
  res["device"] = device_info("cpu")
  if a.profile:
    res["placement"] = {}
    for name, s in (("encoder", c.enc), ("model", c.model), ("decoder", c.dec)):
      if s is not None:
        res["placement"][name] = placement(s.end_profiling())
  path = OUT / f"ort_{tag}.json"
  path.write_text(json.dumps(res, indent=1))
  print(" ".join(f"{k} {v['p50']:.2f}" for k, v in res["latency_ms"].items()), "ms (p50) ->", path)
  if a.profile:
    print("nodos y tiempo por provider:", json.dumps(res["placement"]))


if __name__ == "__main__":
  main()
