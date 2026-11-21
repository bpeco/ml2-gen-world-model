"""Exporta los frames de ejemplo (un clip fijo) como JPEG base64 y los tokens del último frame de contexto.

  python paper/presentacion/exportar_frames.py [clip=3] [salida=paper/presentacion/frames.json]

Necesita results/rollouts/*_ej.npy (no están en git); el JSON resultante sí, así armar.py no depende de los rollouts.
"""
import base64, io, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, ".")
from src.ejemplos import contexto, decodificar

C = int(sys.argv[1]) if len(sys.argv) > 1 else 3
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "paper/presentacion/frames.json")
ROLL = Path("results/rollouts")
MOD = ["gpt2m_ej", "gpt_s_kd_ej", "maskgit_s_abs_p8_ej", "genie_s_p4_ej", "convlstm_px_ej", "congelado_ej"]
H = [20, 60]  # +1 s, +3 s


def b64(a):
  im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
  buf = io.BytesIO(); im.save(buf, "JPEG", quality=86)
  return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


ctx, real = contexto(C)
toks, pix = {"real": real}, {}
for n in MOD:
  a = np.load(ROLL / f"{n}.npy", mmap_mode="r")
  (pix if a.ndim == 5 else toks)[n] = np.asarray(a[C])
img = decodificar(np.concatenate([ctx] + list(toks.values())))
L = len(ctx)
out = dict(zip(toks, img[L:].reshape(len(toks), -1, *img.shape[1:])))
out.update(pix)
res = {"ctx": [b64(img[i]) for i in (L - 20, L - 10, L - 1)],
       "tokens": np.asarray(ctx[-1]).reshape(8, 16).tolist(),
       "frames": {n: [b64(v[h - 1]) for h in H] for n, v in out.items()}}
OUT.write_text(json.dumps(res))
print("ok", {k: len(v) for k, v in res["frames"].items()}, np.asarray(ctx).shape)
