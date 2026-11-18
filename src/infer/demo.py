"""Demo: mp4 con el video real y el predicho lado a lado, con la cadena de punta a punta en el device.

  python -m src.infer.demo --model maskgit_s --steps 4                          # video de cámara de comma, imaginación
  python -m src.infer.demo --model maskgit_s --mode online --k 10               # en línea, predice 0,5 s adelante
  python -m src.infer.demo --model maskgit_s --steps 4 --video mi_video.mp4     # video propio (fuera de distribución)
  python -m src.infer.demo --model maskgit_s --segment 3                        # un segmento de val (tokens decodificados)

Los modelos latentes (lewm, vjepa2, vjepa2_ac) no emiten tokens ni pasan por el codec: esta demo los manda a
src/infer/demo_latente.py (decoder de visualización, máscaras multibloque y curva de error latente), con los mismos argumentos
de esa demo.

Modos:
- imagine: el modelo ve --ctx frames reales (1 s), imagina --horizon frames (default 3 s) y vuelve a anclarse en
  el video real; se repite. Los frames de contexto se ven iguales a los dos lados (borde gris); los imaginados,
  con borde naranja.
- online: en cada frame t el modelo ve el frame real y predice el t+k; se muestra al lado del real t+k.

Entrada: el video de cámara de ejemplo de commaVQ (default, mismo recorte que el dataset), un mp4 propio (se
recorta como la cámara de comma con --comma-crop, o se escala) o un segmento de val (--segment: los tokens reales
decodificados, sin pasar por el encoder). Escribe results/demo/<nombre>_<modo>.mp4 (y un .json con la latencia).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.infer.codec import Codec
from src.infer.stream import make_stream
from src.infer.video import read_video, sample_video, write_mp4

OUT = Path("results/demo")
DTYPES = {"fp32": torch.float32, "fp16": torch.float16}
GRAY, ORANGE = (138, 137, 133), (235, 104, 52)


def border(img, color, w=3):
  out = img.copy()
  out[:w], out[-w:], out[:, :w], out[:, -w:] = color, color, color, color
  return out


def label(img, text):
  """Rótulo blanco sobre un recuadro oscuro en la esquina superior izquierda."""
  import cv2
  out = img.copy()
  font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1
  (w, h), base = cv2.getTextSize(text, font, scale, th)
  cv2.rectangle(out, (3, 3), (9 + w, 9 + h + base), (11, 11, 11), -1)
  cv2.putText(out, text, (6, 6 + h), font, scale, (240, 239, 236), th, cv2.LINE_AA)
  return out


def es_latente(model) -> bool:
  """lewm, vjepa2 y vjepa2_ac: modelos de embeddings, con su propia demo."""
  from src.models.latent import LatentWM
  from src.models.vjepa.action import VJEPA2AC
  from src.models.vjepa.pretrain import VJEPA2
  return isinstance(model, (LatentWM, VJEPA2, VJEPA2AC))


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True, help="un modelo de video (gpt, maskgit, genie, convlstm_px); los latentes van a demo_latente")
  ap.add_argument("--steps", type=int)
  ap.add_argument("--temperature", type=float, default=1.0)
  ap.add_argument("--mode", choices=["imagine", "online"], default="imagine")
  ap.add_argument("--k", type=int, default=10, help="online: frames adelante (10 = 0,5 s)")
  ap.add_argument("--ctx", type=int, default=20)
  ap.add_argument("--horizon", type=int, default=60)
  ap.add_argument("--video", help="mp4 propio")
  ap.add_argument("--comma-crop", action="store_true", help="recortar el video propio como la cámara de comma")
  ap.add_argument("--segment", type=int, help="segmento de val (tokens reales decodificados)")
  ap.add_argument("--start", type=int, default=0)
  ap.add_argument("--frames", type=int, default=400)
  ap.add_argument("--dtype", default="fp16", choices=DTYPES)
  ap.add_argument("--scale", type=int, default=2)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--out")
  ap.add_argument("--opt", action="append", help="campo=valor de la Config del modelo (p. ej. rollout=latent)")
  a, extra = ap.parse_known_args(argv)  # las opciones de la demo latente se las deja a demo_latente
  dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  from src.models import apply_opts, load_model
  model = load_model(a.model, "cpu")
  if es_latente(model):
    from src.infer import demo_latente
    return demo_latente.main(sys.argv[1:] if argv is None else argv)
  if extra:
    ap.error(f"argumentos no reconocidos: {' '.join(extra)}")
  model = apply_opts(model, a.opt).eval().to(dev, DTYPES[a.dtype])
  codec = Codec(dev, a.dtype, graphs=dev.type == "cuda")
  stream = make_stream(model, temperature=a.temperature, steps=a.steps, seed=a.seed)

  tokens = None
  if a.segment is not None:
    val = np.load("data/val_tokens.npy", mmap_mode="r")
    tokens = torch.as_tensor(np.array(val[a.segment, a.start:a.start + a.frames]).astype(np.int64), device=dev)
    real = np.stack([codec.decode(t[None]).cpu().numpy()[0] for t in tokens])
    src = f"val segmento {a.segment}"
  elif a.video:
    real = read_video(a.video, limit=a.start + a.frames, comma_crop=a.comma_crop)[a.start:]
    src = Path(a.video).name
  else:
    real = sample_video()[a.start:a.start + a.frames]
    src = "video de cámara de commaVQ"
  n = len(real)

  pixels = hasattr(stream, "observe_pixels")  # ConvLSTM de píxeles: la cámara entra directo, sin el encoder de comma

  def observe(t):
    if pixels and tokens is None:
      stream.observe_pixels(torch.from_numpy(real[t:t + 1]).to(dev))
    elif tokens is not None:
      stream.observe(tokens[t:t + 1])
    else:
      stream.observe(codec.encode(torch.from_numpy(real[t:t + 1]).to(dev)).clone())

  left, right, ms = [], [], []
  name = model.name + (f"_s{a.steps}" if a.steps else "")
  if a.mode == "imagine":
    cycle = a.ctx + a.horizon
    for t in range(n):
      ph = t % cycle
      t0 = time.perf_counter()
      if ph == 0:
        stream.reset()
      if ph < a.ctx:  # contexto: el modelo ve el frame real
        observe(t)
        pred, col = real[t], GRAY
      else:
        tok = stream.step()
        pred = codec.decode(tok).cpu().numpy()[0]
        col = ORANGE
        if dev.type == "cuda":
          torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
      left.append(label(border(real[t], GRAY), "real"))
      right.append(label(border(pred, col), f"{name}: " + ("contexto" if ph < a.ctx else f"imagina +{ph - a.ctx + 1}")))
  else:
    preds = {}
    for t in range(n):
      t0 = time.perf_counter()
      observe(t)
      if stream.full and t + a.k < n:
        preds[t + a.k] = codec.decode(stream.predict(a.k)).cpu().numpy()[0]
        if dev.type == "cuda":
          torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
    for t in range(n):
      left.append(label(border(real[t], GRAY), "real"))
      p = preds.get(t)
      right.append(label(border(p, ORANGE), f"{name}: predicho desde t-{a.k}") if p is not None
                   else label(border(real[t], GRAY), "contexto"))
  frames = np.concatenate([np.stack(left), np.stack(right)], 2)
  out = Path(a.out) if a.out else OUT / f"{name}_{a.mode}{f'_k{a.k}' if a.mode == 'online' else ''}.mp4"
  write_mp4(frames, out, scale=a.scale)
  info = {"model": a.model, "steps": a.steps, "mode": a.mode, "k": a.k, "source": src, "frames": n,
          "ms_per_frame_p50": float(np.median(ms)) if ms else None,
          "ms_per_frame_p99": float(np.percentile(ms, 99)) if ms else None,
          "note": "latencia por frame generado en la demo (incluye bajar el frame a la CPU), sin la escritura del mp4"}
  out.with_suffix(".json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
  print(f"{out}  ({n} frames, {info['ms_per_frame_p50'] or 0:.1f} ms por frame generado, mediana)")
  return info


if __name__ == "__main__":
  main()
