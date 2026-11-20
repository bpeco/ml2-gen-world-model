"""Imágenes y tokens de las slides del pipeline (1), de cómo se genera un frame (3) y del video lado a lado (5).

Paso 1, rollouts de los 4 clips del segmento 0 de val, con la temperatura y la semilla de la evaluación:
  python -m src.rollouts --model real --limit 4 --bs 4 --name real_ej
  python -m src.rollouts --model gpt2m --limit 4 --bs 4 --temperature 1.0 --name gpt2m_ej
  python -m src.rollouts --model gpt_s_kd --limit 4 --bs 4 --temperature 0.9 --name gpt_s_kd_ej
  python -m src.rollouts --model genie_s --steps 4 --limit 4 --bs 4 --temperature 1.0 --name genie_s_p4_ej
  python -m src.rollouts --model maskgit_s_abs --steps 8 --limit 4 --bs 4 --temperature 1.0 --name maskgit_s_abs_p8_ej
  python -m src.rollouts --model convlstm_px --limit 4 --bs 4 --name convlstm_px_ej
Paso 2, desde la raíz del repo (necesita data/, third_party/commavq y los checkpoints):
  python presentacion/generar_video.py --clip 0      # -> presentacion/img/video_*.jpg y presentacion/video.js

Cada sprite tiene 80 frames apilados en vertical: los 20 de contexto real (1 s) y los 60 que siguen (3 s): los reales
en video_real.jpg y los generados en los demás. Un modelo sin rollout queda afuera y la slide lo marca como pendiente.
También guarda los tokens para las grillas animadas: el último frame de contexto, el primer frame que genera el GPT-S
(token a token) y el primer frame que generan Genie-S y MaskGIT-S con 4 pasos, con el paso en que quedó fija cada celda.
MaskGIT-S solo si su checkpoint está en checkpoints/maskgit_s_abs/final.pt (no está en Hugging Face).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ejemplos import contexto, decodificar  # noqa: E402

MODELOS = ["real_ej", "gpt2m_ej", "gpt_s_kd_ej", "maskgit_s_abs_p8_ej", "genie_s_p4_ej", "convlstm_px_ej"]


def sprite(imgs, path, quality=80):
  """Frames (N, 128, 256, 3) apilados en vertical: el HTML los recorre con translateY."""
  Image.fromarray(np.concatenate(list(imgs), 0)).save(path, quality=quality, optimize=True)


def orden(run, ctx, pasos=4, temperatura=1.0, semilla=0):
  """Primer frame que genera `run` (genie_s o maskgit_s_abs) desde ctx (20, 128) y el paso (1..pasos) en que se fijó
  cada celda.

  Envuelve la función de logits que recibe decode_frame: en cada llamada, las celdas que ya no son MASK son las que
  fijaron los pasos anteriores. No cambia la decodificación: es la misma llamada que hace el modelo al generar.
  """
  from src.common import device
  from src.models import load_model
  import src.models.genie as GE
  import src.models.maskgit as MG
  G = GE if run.startswith("genie") else MG
  dev = device()
  m = load_model(run, dev).eval()
  estados = []
  original = G.decode_frame

  def espia(logits_fn, *a, **k):
    def fn(cur):
      estados.append(cur.detach().cpu().clone())
      return logits_fn(cur)
    return original(fn, *a, **k)

  G.decode_frame = espia
  try:
    g = torch.Generator(device=dev).manual_seed(semilla)
    x = torch.as_tensor(ctx[None], dtype=torch.long, device=dev)
    with torch.no_grad():
      if G is GE:
        out = m.decode(x, temperature=temperatura, steps=pasos, generator=g)
      else:
        out = m.next_frame(x, temperature=temperatura, steps=pasos, generator=g)
  finally:
    G.decode_frame = original
  out = out.cpu()[0]
  paso = torch.full((128,), pasos, dtype=torch.long)
  for s, cur in enumerate(estados[1:], start=1):   # estados[s] = celdas ya fijas antes del paso s + 1
    nuevas = (cur[0] != G.MASK) & (paso == pasos)
    paso[nuevas] = torch.minimum(paso[nuevas], torch.tensor(s))
  # las celdas fijadas en el paso s aparecen en estados[s] por primera vez; las que nunca aparecen, en el último paso
  return out.tolist(), paso.tolist()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--clip", type=int, default=0)
  ap.add_argument("--rollouts", default="results/rollouts")
  ap.add_argument("--out", default=str(ROOT / "presentacion"))
  a = ap.parse_args()
  out = Path(a.out)
  (out / "img").mkdir(parents=True, exist_ok=True)

  ctx, real = contexto(a.clip)                   # (20, 128) y (60, 128) tokens reales
  ctx_img = decodificar(ctx)
  hechos = []
  for n in MODELOS:
    if n == "real_ej":
      gen = decodificar(real)
    else:
      p = Path(a.rollouts) / f"{n}.npy"
      if not p.exists():
        print("falta", p, "(la slide lo marca como pendiente)")
        continue
      x = np.load(p, mmap_mode="r")[a.clip]
      gen = np.asarray(x) if x.ndim == 4 else decodificar(np.asarray(x))   # convlstm_px ya trae píxeles
    nombre = n.removesuffix("_ej")
    sprite(np.concatenate([ctx_img, gen]), out / "img" / f"video_{nombre}.jpg")
    hechos.append(nombre)
    print("sprite", nombre)

  gpt = np.load(Path(a.rollouts) / "gpt_s_kd_ej.npy", mmap_mode="r")[a.clip][0]
  genie_tok, genie_paso = orden("genie_s", ctx)
  video = {"clip": a.clip, "modelos": hechos, "ctx_frames": int(len(ctx)), "gen_frames": int(len(real)),
           "tok_ctx": ctx[-1].tolist(), "tok_gpt": np.asarray(gpt).tolist(),
           "tok_genie": genie_tok, "paso_genie": genie_paso, "pasos_genie": 4, "tok_mg": None, "paso_mg": None}
  if Path("checkpoints/maskgit_s_abs/final.pt").exists():
    video["tok_mg"], video["paso_mg"] = orden("maskgit_s_abs", ctx)
  else:
    print("falta checkpoints/maskgit_s_abs/final.pt: la slide 4 usa el orden de Genie-S y lo dice")
  (out / "video.js").write_text("// Generado por presentacion/generar_video.py. No editar a mano.\nwindow.VIDEO = "
                                + json.dumps(video, separators=(",", ":")) + ";\n", encoding="utf-8")
  cuenta = [genie_paso.count(s) for s in range(1, 5)]
  print("video.js: celdas fijadas por paso en Genie-S:", cuenta)


if __name__ == "__main__":
  main()
