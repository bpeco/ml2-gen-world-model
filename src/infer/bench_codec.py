"""Fidelidad y latencia del codec de comma (encoder + decoder del VQ-VAE) en un dispositivo, batch 1.

  python -m src.infer.bench_codec                 # latencia + fidelidad -> results/infer/codec_<tag>.json
  python -m src.infer.bench_codec --no-check      # solo latencia (más rápido)
  python -m src.infer.bench_codec --device cpu

Latencia: tiempo de reloj (perf_counter, sincronizando el device) por llamada, después de un calentamiento.
- enc: uint8 en el device -> tokens. dec: tokens -> uint8 en el device.
- codec: frame uint8 en memoria del host (pinned) -> subida -> encoder -> decoder -> bajada al host.
  Es la cadena sin world model: el piso de la latencia de punta a punta en modo en línea.
Variantes: fp32 y fp16 sin graphs, y fp16 con CUDA graphs (solo CUDA/ROCm). En fp32 se apaga TF32 para la
fidelidad (referencia exacta) pero no para la latencia (default de PyTorch: TF32 en cuDNN).

Fidelidad (--check, sobre los primeros --n-check frames del video de cámara de ejemplo de commaVQ):
- tokens iguales a los que calculó comma (examples/tokens.npy) y a fp32 sin TF32 en el mismo device;
- decoder: PSNR de fp16 contra fp32 con los tokens de comma;
- LPIPS (alex) de la reconstrucción contra el frame de entrada (el techo del modo en línea) y del camino
  fp16 + graphs contra el fp32.
"""
import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.common import device_info
from src.infer.codec import Codec
from src.infer.video import sample_tokens, sample_video

OUT = Path("results/infer")


def device_tag(dev) -> str:
  """Nombre corto del dispositivo para los archivos: rtx3080ti, t4, radeon860m, cpu."""
  dev = torch.device(dev)
  if dev.type != "cuda":
    return dev.type
  name = torch.cuda.get_device_name(dev).lower()
  for pat, tag in ((r"tesla t4|\bt4\b", "t4"), (r"a100", "a100"), (r"l4\b", "l4")):
    if re.search(pat, name):
      return tag
  name = re.sub(r"nvidia|geforce|amd|radeon\(tm\)|graphics|\(r\)|\(tm\)", "", name)
  return re.sub(r"[^a-z0-9]", "", name) or "cuda"


def sync(dev):
  if dev.type == "cuda":
    torch.cuda.synchronize(dev)


def wall_ms(fn, n, warmup, dev):
  """Tiempos en ms de n llamadas de fn() después de warmup llamadas, sincronizando antes y después."""
  for _ in range(warmup):
    fn()
  sync(dev)
  ms = []
  for _ in range(n):
    t = time.perf_counter()
    fn()
    sync(dev)
    ms.append((time.perf_counter() - t) * 1000)
  return ms


def stats(ms):
  ms = np.asarray(ms)
  return {"p50": float(np.median(ms)), "p90": float(np.percentile(ms, 90)), "p99": float(np.percentile(ms, 99)),
          "mean": float(ms.mean()), "n": int(len(ms))}


def latency(dev, frames, n, warmup):
  """{variante: {enc, dec, codec}} con los primeros frames del video (cíclicos)."""
  variants = [("fp32", False), ("fp16", False)] + ([("fp16", True)] if dev.type == "cuda" else [])
  host = torch.from_numpy(frames[:64])
  if dev.type == "cuda":
    host = host.pin_memory()
  out_host = torch.empty_like(host[:1]).pin_memory() if dev.type == "cuda" else torch.empty_like(host[:1])
  res = {}
  for dt, g in variants:
    key = dt + ("_graphs" if g else "")
    c = Codec(dev, dt, graphs=g)
    img = host[:1].to(dev)
    try:
      tok = c.encode(img).clone()
    except RuntimeError as e:  # p. ej. ROCm sin soporte de graphs: la captura falla
      if not g:
        raise
      res[key] = {"error": str(e).splitlines()[0]}
      print(key, "falló:", res[key]["error"], flush=True)
      continue
    i = [0]

    def enc():
      c.encode(img)

    def dec():
      c.decode(tok)

    def codec():
      i[0] = (i[0] + 1) % len(host)
      x = host[i[0]:i[0] + 1].to(dev, non_blocking=True)
      out_host.copy_(c.decode(c.encode(x)), non_blocking=True)

    try:
      res[key] = {k: stats(wall_ms(f, n, warmup, dev)) for k, f in (("enc", enc), ("dec", dec), ("codec", codec))}
    except RuntimeError as e:  # p. ej. ROCm sin soporte de graphs
      if not g:
        raise
      res[key] = {"error": str(e).splitlines()[0]}
      print(key, "falló:", res[key]["error"], flush=True)
      continue
    res[key]["params"] = c.num_params()
    if dev.type == "cuda":
      res[key]["peak_mem_mb"] = torch.cuda.max_memory_allocated(dev) / 2**20
      torch.cuda.reset_peak_memory_stats(dev)
    print(key, {k: round(v["p50"], 2) for k, v in res[key].items() if isinstance(v, dict) and "p50" in v}, flush=True)
    del c
  return res


def psnr(a, b):
  mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
  return float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


@torch.no_grad()
def fidelity(dev, frames, ref_tok, n):
  import lpips
  x = torch.from_numpy(frames[:n])
  ref = ref_tok[:n]

  def run(c, fn, inp):
    return torch.cat([fn(inp[i:i + 1].to(dev)).clone().cpu() for i in range(len(inp))])

  tf32 = torch.backends.cudnn.allow_tf32
  torch.backends.cudnn.allow_tf32 = False  # referencia fp32 exacta
  c32 = Codec(dev, "fp32", graphs=False)
  t32 = run(c32, c32.encode, x)
  r32 = run(c32, c32.decode, t32)                       # reconstrucción fp32
  d32 = run(c32, c32.decode, torch.from_numpy(ref))     # decoder fp32 con los tokens de comma
  del c32
  torch.backends.cudnn.allow_tf32 = tf32
  try:
    c16 = Codec(dev, "fp16", graphs=dev.type == "cuda")
    t16 = run(c16, c16.encode, x)
  except RuntimeError:  # sin soporte de graphs (algunos ROCm)
    c16 = Codec(dev, "fp16", graphs=False)
    t16 = run(c16, c16.encode, x)
  r16 = run(c16, c16.decode, t16)
  d16 = run(c16, c16.decode, torch.from_numpy(ref))
  del c16

  lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()

  def lpips_mean(a, b, bs=50):
    to_t = lambda u: u.permute(0, 3, 1, 2).float().div(127.5).sub(1).to(dev)
    return float(torch.cat([lp(to_t(a[i:i + bs]), to_t(b[i:i + bs])).flatten().cpu()
                            for i in range(0, len(a), bs)]).mean())

  res = {
    "n_frames": n,
    "tokens_fp32_vs_comma": float((t32.numpy() == ref).mean()),
    "tokens_fp16_vs_comma": float((t16.numpy() == ref).mean()),
    "tokens_fp16_vs_fp32": float((t16 == t32).float().mean()),
    "decoder_psnr_fp16_vs_fp32_db": psnr(d16.numpy(), d32.numpy()),
    "decoder_max_abs_diff": int((d16.int() - d32.int()).abs().max()),
    "lpips_recon_fp32_vs_input": lpips_mean(r32, x),
    "lpips_recon_fp16_vs_input": lpips_mean(r16, x),
    "lpips_recon_fp16_vs_fp32": lpips_mean(r16, r32),
    "lpips_decoder_fp16_vs_fp32": lpips_mean(d16, d32),
    "psnr_recon_fp32_vs_input_db": psnr(r32.numpy(), x.numpy()),
  }
  for k, v in res.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}", flush=True)
  return res


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  ap.add_argument("--frames", type=int, default=500, help="llamadas medidas por etapa")
  ap.add_argument("--warmup", type=int, default=50)
  ap.add_argument("--no-check", action="store_true")
  ap.add_argument("--n-check", type=int, default=200)
  ap.add_argument("--tag", help="nombre del dispositivo en el archivo (default: el de la GPU)")
  a = ap.parse_args()
  dev = torch.device(a.device)
  tag = a.tag or device_tag(dev)
  if dev.type == "cpu":
    a.frames, a.warmup = min(a.frames, 30), min(a.warmup, 3)
  frames = sample_video()
  res = {"device": device_info(dev), "tag": tag, "torch": torch.__version__,
         "timestamp": datetime.now().isoformat(timespec="seconds"),
         "protocol": "batch 1, reloj de pared con sincronización; codec = host pinned -> enc -> dec -> host"}
  print(f"== latencia en {tag}", flush=True)
  res["latency"] = latency(dev, frames, a.frames, a.warmup)
  if not a.no_check:
    print("== fidelidad", flush=True)
    res["fidelity"] = fidelity(dev, frames, sample_tokens(), a.n_check)
  OUT.mkdir(parents=True, exist_ok=True)
  path = OUT / f"codec_{tag}.json"
  path.write_text(json.dumps(res, indent=1))
  print("->", path)


if __name__ == "__main__":
  main()
