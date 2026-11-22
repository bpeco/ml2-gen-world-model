"""v2: ¿dónde está el movimiento en la representación de commaVQ? Experimento sintético con movimiento conocido.

  python -m src.analisis_movimiento [--n-base 96] [--n-pares 3000] [--device cuda]

Escribe results/v2/latente_movimiento.json.

Base: frames reales de val decodificados a píxeles (128x256). A cada uno se le aplica un movimiento conocido con
grid_sample (desplazamiento dx, dy en píxeles; zoom s alrededor del centro, que imita al auto avanzando) y se vuelve a
pasar por el encoder de comma. Se compara siempre encode(T(img)) contra encode(img), los dos re-codificados.

1. Barrido: fracción de los 128 tokens que cambia con ruido de píxel solo (parpadeo sin movimiento), con cada
   desplazamiento y con cada zoom. La referencia es el ~60 % de cambio entre frames reales consecutivos.
2. Distancia en el codebook entre el token viejo y el nuevo, en las celdas que cambian: ¿los cambios reales se parecen
   más a los del ruido (vecinos cercanos) o a los del movimiento?
3. Sonda: pares con (dx, dy, zoom) al azar y ruido independiente en cada frame. Una red convolucional chica sobre la
   grilla de 8x16 celdas (2 capas 3x3 de 128 canales, promedio espacial, lineal) predice los tres parámetros desde tres
   representaciones del par, separando train/test por frame base (el test son escenas que la sonda nunca vio):
     bits      : qué celdas cambiaron de token (1 canal; lo que le da motion_in al MaskGIT)
     codebook  : vectores del codebook de las dos celdas [a, b - a] (512 canales; lo que ve el transformer)
     continuo  : latente del encoder antes de cuantizar, [a, b - a] (512 canales)
   R2 en test por parámetro. Si el continuo predice mucho mejor que el codebook, cuantizar destruye el movimiento;
   si bits ya alcanza, el embedding de motion_in tiene casi toda la información.
   Una sonda lineal no sirve: cómo cambia el latente con dx depende del contenido de cada celda (bordes, gradientes).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.infer.codec import H, W, ensure_weights, load_decoder, load_encoder
from src.rollouts import VAL

OUT = Path("results/v2/latente_movimiento.json")


class Captura(torch.nn.Module):
  """Envuelve al cuantizador del encoder y guarda el latente continuo h (B, 128, 256)."""

  def __init__(self, q):
    super().__init__()
    self.q, self.h = q, None

  def forward(self, h):
    self.h = h.float()
    return self.q(h)


def transformar(img, dx=0.0, dy=0.0, zoom=1.0, ruido=0.0, gen=None):
  """img float (B, 3, H, W) en [0, 255]. Desplaza (dx, dy) píxeles y hace zoom alrededor del centro, con bilineal y
  bordes replicados; suma ruido gaussiano de sigma `ruido` y redondea a uint8 como la cámara. Escalares o (B,)."""
  B = img.shape[0]
  t = lambda v: torch.as_tensor(v, dtype=torch.float32, device=img.device).expand(B)
  dx, dy, s = t(dx), t(dy), t(zoom)
  th = torch.zeros(B, 2, 3, device=img.device)
  th[:, 0, 0], th[:, 1, 1] = 1 / s, 1 / s
  th[:, 0, 2], th[:, 1, 2] = -2 * dx / W, -2 * dy / H   # el contenido se mueve +dx a la derecha, +dy abajo
  grid = F.affine_grid(th, img.shape, align_corners=False)
  y = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)
  if ruido > 0:
    y = y + ruido * torch.randn(y.shape, device=y.device, generator=gen)
  return y.round().clamp(0, 255)


@torch.no_grad()
def codificar(enc, cap, img, bs=64):
  """img float (B, 3, H, W) -> tokens (B, 128) y latente continuo (B, 128, 256)."""
  toks, hs = [], []
  for i in range(0, len(img), bs):
    toks.append(enc(img[i:i + bs]).view(-1, 128))
    hs.append(cap.h)
  return torch.cat(toks), torch.cat(hs)


def sonda_r2(Xtr, ytr, Xte, yte, dev, epocas=60, bs=128, seed=0):
  """X (N, C, 8, 16) en CPU, y (N, 3). Red conv chica; normaliza X por canal e y por columna con el train.
  Devuelve el R2 en test por columna y el de train (para ver sobreajuste)."""
  torch.manual_seed(seed)
  mu = Xtr.mean((0, 2, 3), keepdim=True)
  sd = Xtr.std((0, 2, 3), keepdim=True) + 1e-6
  ym, ys = ytr.mean(0), ytr.std(0)
  C = Xtr.shape[1]
  net = torch.nn.Sequential(torch.nn.Conv2d(C, 128, 3, padding=1), torch.nn.GELU(),
                            torch.nn.Conv2d(128, 128, 3, padding=1), torch.nn.GELU(),
                            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(128, 3)).to(dev)
  opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epocas)
  norm = lambda X: ((X.to(dev) - mu.to(dev)) / sd.to(dev))
  for _ in range(epocas):
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(perm), bs):
      j = perm[i:i + bs]
      loss = F.mse_loss(net(norm(Xtr[j])), ((ytr[j] - ym) / ys).to(dev))
      opt.zero_grad()
      loss.backward()
      opt.step()
    sched.step()

  @torch.no_grad()
  def r2(X, y):
    pred = torch.cat([net(norm(X[i:i + 512])).cpu() for i in range(0, len(X), 512)]) * ys + ym
    return [round(float(v), 3) for v in 1 - ((pred - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)]

  return r2(Xte, yte), r2(Xtr, ytr)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--n-base", type=int, default=96)
  ap.add_argument("--n-pares", type=int, default=3000)
  ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  ap.add_argument("--seed", type=int, default=0)
  a = ap.parse_args(argv)
  t0 = time.time()
  dev = torch.device(a.device)
  gen = torch.Generator(device=dev).manual_seed(a.seed)
  rng = np.random.default_rng(a.seed)
  ensure_weights()
  enc, dec = load_encoder(dev), load_decoder(dev)
  cb = enc.quantize.cb                      # (1024, 256) codebook en fp32
  cap = Captura(enc.quantize)
  enc.quantize = cap

  # frames base: uno por segmento, a mitad del segmento, más los 20 siguientes para la referencia real
  val = np.load(VAL, mmap_mode="r")
  segs = rng.choice(len(val), a.n_base, replace=False)
  tok0 = torch.as_tensor(np.stack([val[s, 600] for s in segs]).astype(np.int64), device=dev)
  with torch.no_grad():
    base = dec(tok0).clamp(0, 255).round()                       # (N, 3, 128, 256)
  t_base, h_base = codificar(enc, cap, base)
  res = {"n_base": a.n_base, "roundtrip_cambio": float((t_base != tok0).float().mean())}

  # 1 y 2. barrido
  def medir(img):
    t, _ = codificar(enc, cap, img)
    ch = t != t_base
    d = (cb[t] - cb[t_base]).norm(dim=-1)[ch]
    return {"cambio": round(float(ch.float().mean()), 4),
            "dist_codebook_mediana": round(float(d.median()), 3) if len(d) else None}

  barrido = {"ruido": {}, "dx": {}, "dy": {}, "zoom": {}}
  for sg in (1, 2, 4, 8):
    barrido["ruido"][sg] = medir(transformar(base, ruido=sg, gen=gen))
  for v in (0.5, 1, 2, 4, 8, 16):
    barrido["dx"][v] = medir(transformar(base, dx=v))
  for v in (0.5, 1, 2, 4, 8):
    barrido["dy"][v] = medir(transformar(base, dy=v))
  for v in (1.005, 1.01, 1.02, 1.05, 1.1):
    barrido["zoom"][v] = medir(transformar(base, zoom=v))
  res["barrido"] = barrido

  # referencia real: frames consecutivos 600 -> 601 de los mismos segmentos (tokens originales, sin re-codificar)
  tok1 = torch.as_tensor(np.stack([val[s, 601] for s in segs]).astype(np.int64), device=dev)
  ch = tok1 != tok0
  res["real_consecutivo"] = {"cambio": round(float(ch.float().mean()), 4),
                             "dist_codebook_mediana": round(float((cb[tok1] - cb[tok0]).norm(dim=-1)[ch].median()), 3)}
  # distancia típica entre dos códigos cualesquiera, como escala
  i, j = torch.randint(0, 1024, (2, 20000), device=dev, generator=gen)
  res["dist_codebook_azar_mediana"] = round(float((cb[i] - cb[j]).norm(dim=-1).median()), 3)

  # 3. sonda lineal
  P = a.n_pares
  bidx = torch.as_tensor(rng.integers(0, a.n_base, P), device=dev)
  dx = torch.as_tensor(rng.uniform(-8, 8, P), dtype=torch.float32, device=dev)
  dy = torch.as_tensor(rng.uniform(-4, 4, P), dtype=torch.float32, device=dev)
  lz = torch.as_tensor(rng.uniform(0, math.log(1.08), P), dtype=torch.float32, device=dev)
  grilla = lambda z: z.view(z.shape[0], 8, 16, -1).permute(0, 3, 1, 2).half().cpu()  # (B, C, 8, 16)
  par = lambda za, zb: torch.cat([grilla(za), grilla(zb - za)], 1)
  feats = {"bits": [], "codebook": [], "continuo": []}
  for k in range(0, P, 100):
    sl = slice(k, k + 100)
    img = base[bidx[sl]]
    a_img = transformar(img, ruido=2, gen=gen)
    b_img = transformar(img, dx=dx[sl], dy=dy[sl], zoom=lz[sl].exp(), ruido=2, gen=gen)
    ta, ha = codificar(enc, cap, a_img)
    tb, hb = codificar(enc, cap, b_img)
    feats["bits"].append(grilla((ta != tb).float()[..., None]))
    feats["codebook"].append(par(cb[ta], cb[tb]))
    feats["continuo"].append(par(ha, hb))
  y = torch.stack([dx, dy, lz], 1).cpu()
  test_base = set(rng.choice(a.n_base, a.n_base * 3 // 10, replace=False).tolist())
  te = torch.as_tensor([int(b) in test_base for b in bidx.tolist()])
  sonda = {}
  for nombre, fs in feats.items():
    X = torch.cat(fs).float()
    r2, r2_tr = sonda_r2(X[~te], y[~te], X[te], y[te], dev)
    sonda[nombre] = {"r2_dx": r2[0], "r2_dy": r2[1], "r2_zoom": r2[2], "r2_train": r2_tr, "canales": X.shape[1]}
    print(nombre, sonda[nombre], flush=True)
  res["sonda"] = sonda
  res["sonda_config"] = {"n_pares": P, "dx": [-8, 8], "dy": [-4, 4], "zoom": [1, 1.08], "ruido_sigma": 2,
                         "test": "30 % de los frames base", "n_test": int(te.sum())}
  res["segundos"] = round(time.time() - t0, 1)
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
  print(json.dumps(res, indent=1))


if __name__ == "__main__":
  main()
