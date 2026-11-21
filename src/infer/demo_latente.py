"""Demo de los world models latentes: `lewm`, `vjepa2` y `vjepa2_ac` (los que no emiten tokens ni pasan por el codec).

  python -m src.infer.demo_latente --model lewm --segment 3                    # real | decoder(z) | decoder(rollout)
  python -m src.infer.demo_latente --model vjepa2_ac --segment 3 --mp4         # lo mismo y un mp4
  python -m src.infer.demo_latente --model vjepa2 --segment 3                  # máscaras multibloque sobre un clip
  python -m src.infer.demo_latente --model random:lewm --decoder otro.pt

Lee un segmento del val con poses (data/valp_*.npy, el de eval_latent), lo pasa a píxeles con el decoder de comma y:
- lewm / vjepa2_ac: codifica la ventana (16 pasos a 4 Hz), imagina `--pasos` pasos desde 3 frames reales con las acciones
  reales (rollout latente) y los dibuja con el decoder de visualización: fila de frames reales, de decoder(z real) y de
  decoder(ẑ imaginado); abajo, la curva de error latente por horizonte contra "copiar el último latente" (la misma que
  mide eval_latent). El decoder es el de visualización, entrenado aparte con el modelo congelado
  (src/train_decoder.py para lewm, src/models/vjepa/train_decoder.py para vjepa2 y vjepa2_ac): no es parte del modelo, y
  sin él la figura trae solo los reales y la curva.
- vjepa2: las dos máscaras multibloque (8 bloques cortos de escala 0,15 y 2 largos de 0,7) que usa el preentrenamiento,
  sobre cuatro instantes del clip, y los mismos frames decodificados desde las representaciones del encoder.
Escribe results/demo/<nombre>_latente.png (o _mascaras.png) y un .json con las curvas; con --mp4, un mp4 a 4 Hz.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src import eval_latent as EL
from src.common import device
from src.infer.video import write_mp4
from src.models import load_model
from src.models.lewm import LeWM
from src.models.vjepa.pretrain import VJEPA2

OUT = Path("results/demo")
CKPT = Path("checkpoints")
ROJO = np.array([235, 60, 52], np.float32)


def buscar_decoder(model, spec, explicito=None):
  """Ruta del decoder de visualización de este modelo, o None si todavía no se entrenó."""
  if explicito:
    return Path(explicito)
  from src.train_decoder import run_name
  run = run_name(spec)
  if isinstance(model, LeWM):
    cand = [CKPT / run / "decoder.pt"]
  else:  # vjepa2: <run>_viz; vjepa2_ac: el del encoder vjepa2 en el que se apoya, o el propio
    enc = getattr(model.cfg, "encoder_run", "") or ""
    cand = [CKPT / f"{run_name(enc)}_viz" / "decoder.pt"] * bool(enc) + [CKPT / f"{run}_viz" / "decoder.pt"]
  from src import store  # si está anotado en el manifiesto y falta acá, lo baja del almacén (Hugging Face)
  return next((c for c in cand if store.ensure_file(c) is not None), None)


def cargar_decoder(model, ruta, dev):
  """-> f(z) -> uint8 (..., H, W, 3), o None. z: (..., D) en lewm; (..., N, D) en vjepa2 y vjepa2_ac."""
  if ruta is None:
    return None
  if isinstance(model, LeWM):
    from src.models.lewm_decoder import load_decoder
    dec = load_decoder(ruta, dev)
    return lambda z: dec.decode(z.to(dev))
  from src.models.vjepa.patch import IMAGENET_MEDIA, IMAGENET_STD
  from src.models.vjepa.train_decoder import cargar
  dec = cargar(ruta, dev)
  mean = torch.tensor(IMAGENET_MEDIA, device=dev).view(3, 1, 1)
  std = torch.tensor(IMAGENET_STD, device=dev).view(3, 1, 1)

  @torch.no_grad()
  def f(z):
    x = dec(z.to(dev)).float() * std + mean
    return (x.clamp(0, 1) * 255).round().to(torch.uint8).movedim(-3, -1)
  return f


def ventana(a, dev):
  """(píxeles (1, 16, H, W, 3) uint8, poses (1, 16, 30)) del segmento pedido del val con poses."""
  tok, pos, keys = EL.cargar(EL.VAL)
  seg = a.segment % len(keys)
  t0 = min(a.start, tok.shape[1] - EL.SPAN - 1)
  tk, ps = EL.tomar(tok, pos, [(seg, t0, 0)])
  return EL.render(torch.from_numpy(tk).to(dev)), torch.from_numpy(ps).to(dev), f"{keys[seg]} desde el frame {t0}"


def curvas(model, img, poses, H):
  r = model.lat_horizonte(img, poses, EL.CTX, H)
  return {k: v[0].float().cpu().numpy() for k, v in r.items()}


def a_np(x):
  return x.detach().cpu().numpy()


def dibujar(path, titulo, filas, curva=None, nota=None):
  """filas: [(rótulo, uint8 (n, H, W, 3), [subtítulo por imagen])]; curva: {segundos, mse, copia} -> png."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  n = max(len(f[1]) for f in filas)
  fig = plt.figure(figsize=(1.9 * n + 1.2, 1.15 * len(filas) + (2.6 if curva else 0.4)), constrained_layout=True)
  gs = fig.add_gridspec(len(filas) + bool(curva), n, height_ratios=[1] * len(filas) + [2.2] * bool(curva))
  for i, (rot, imgs, subs) in enumerate(filas):
    for j in range(n):
      ax = fig.add_subplot(gs[i, j])
      ax.axis("off")
      if j < len(imgs):
        ax.imshow(imgs[j])
        if subs:
          ax.set_title(subs[j], fontsize=7)
      if j == 0:
        ax.text(-0.04, 0.5, rot, transform=ax.transAxes, ha="right", va="center", fontsize=8)
  if curva:
    ax = fig.add_subplot(gs[len(filas), :])
    ax.plot(curva["segundos"], curva["mse"], "o-", label="modelo (rollout con acciones reales)")
    ax.plot(curva["segundos"], curva["copia"], "s--", label="copiar el último latente")
    ax.set_xlabel("horizonte (s)")
    ax.set_ylabel("error latente (MSE)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
  fig.suptitle(titulo + (f"\n{nota}" if nota else ""), fontsize=9)
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path, dpi=110)
  plt.close(fig)
  return Path(path)


def demo_prediccion(model, a, dev, name):
  """lewm y vjepa2_ac: real, decodificación del latente real y decodificación del rollout imaginado."""
  img, poses, origen = ventana(a, dev)
  H, c = a.pasos, EL.CTX
  dec = cargar_decoder(model, buscar_decoder(model, a.model, a.decoder), dev)
  with torch.no_grad():
    if isinstance(model, LeWM):
      z = model.encode(img).float()
      ac = model.action_embed(poses, 1, z.shape[1], dev)
      zh = model.rollout_latent(z[:, :c], ac[:, :c - 1], ac[:, c - 1:c - 1 + H]).float()
    else:
      z = model._codificar_ventana(img).float()
      s, ac = model.estados_acciones(poses, 1, z.shape[1], dev)
      zh = model.rollout(z, s, ac, c, H)
  real = a_np(img[0, c:c + H])
  subs = [f"+{(h + 1) * 0.25:g} s" for h in range(len(real))]
  filas = [("real", real, subs)]
  info = {"model": a.model, "origen": origen, "pasos": H, "decoder": None if dec is None else str(buscar_decoder(model, a.model, a.decoder))}
  if dec is not None:
    rec = a_np(dec(z[0, c:c + H]))
    imag = a_np(dec(zh[0]))
    filas += [("decoder(z real)", rec, None), ("decoder(rollout)", imag, None)]
  err = curvas(model, img, poses, H)
  seg = [(h + 1) * EL.SKIP / 20 for h in range(len(real))]
  info.update(segundos=seg, error_mse=err["mse"].tolist(), error_copiar_ultimo_mse=err["copia_mse"].tolist(),
              error_l1=err["l1"].tolist(), error_copiar_ultimo_l1=err["copia_l1"].tolist())
  nota = None if dec else "sin decoder de visualización: entrenalo con el comando de la guía del modelo (aquí solo los reales y la curva)"
  png = dibujar(a.out or OUT / f"{name}_latente.png", f"{name}: {origen}", filas,
                {"segundos": seg, "mse": err["mse"], "copia": err["copia_mse"]}, nota)
  info["png"] = str(png)
  if a.mp4 and dec is not None:
    cols = [real, rec, imag]
    video = np.stack([np.concatenate([c[k] for c in cols], 1) for k in range(len(real))])  # lado a lado, un frame por paso
    info["mp4"] = str(write_mp4(video, png.with_suffix(".mp4"), fps=4, scale=a.scale))
  return info


def mascaras_sobre_clip(model, img, dev):
  """Para cada conjunto de máscara: la imagen del instante t con los tokens tapados (no visibles) teñidos de rojo."""
  c = model.cfg
  me, _ = model.masks(1, dev)
  T, Hh, Ww = model.grid
  clip_ini = img.shape[1] - c.frames
  out = []
  for s, m in enumerate(me):
    vis = torch.zeros(T * Hh * Ww, dtype=torch.bool, device=dev)
    vis[m[0]] = True
    vis = a_np(vis.view(T, Hh, Ww))
    filas = []
    for t in np.unique(np.linspace(0, T - 1, 4).round().astype(int)):
      fr = a_np(img[0, clip_ini + t * c.tubelet]).astype(np.float32)
      tapado = np.kron(~vis[t], np.ones((c.patch, c.patch), bool))
      tapado = tapado[:fr.shape[0], :fr.shape[1]]
      fr = np.where(tapado[..., None], 0.35 * fr + 0.65 * ROJO, fr)
      filas.append(fr.astype(np.uint8))
    out.append((1 - float(vis.mean()), np.stack(filas), [int(t) for t in np.unique(np.linspace(0, T - 1, 4).round().astype(int))]))
  return out


def demo_mascaras(model, a, dev, name):
  img, _, origen = ventana(a, dev)
  conj = mascaras_sobre_clip(model, img, dev)
  nombres = ["A: 8 bloques cortos (escala 0,15)", "B: 2 bloques largos (escala 0,7)"]
  filas = [(f"{nombres[i] if i < 2 else i}\ntapado {100 * f:.0f} %", im, [f"tubelet {t}" for t in ts]) for i, (f, im, ts) in enumerate(conj)]
  c = model.cfg
  dec = cargar_decoder(model, buscar_decoder(model, a.model, a.decoder), dev)
  info = {"model": a.model, "origen": origen, "tapado": [f for f, _, _ in conj], "decoder": None}
  if dec is not None:
    with torch.no_grad():
      x = model.clip(img[:, -c.frames:])
      idx = [conj[0][2][k] * c.tubelet for k in range(len(conj[0][2]))]
      from src.models.vjepa.patch import clip_a_frames
      from src.models.vjepa.pretrain import layer_norm_canal
      z = layer_norm_canal(model.target_encoder(clip_a_frames(x[:, :, idx], c.tubelet)))
    filas += [("real", a_np(img[0, img.shape[1] - c.frames + np.array(idx)]), None), ("decoder(encoder)", a_np(dec(z)), None)]
    info["decoder"] = str(buscar_decoder(model, a.model, a.decoder))
  png = dibujar(a.out or OUT / f"{name}_mascaras.png", f"{name}: máscaras multibloque sobre un clip de {c.frames} frames ({origen})",
                filas, None, None if dec else "sin decoder de visualización: solo las máscaras")
  info["png"] = str(png)
  return info


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--model", required=True, help="corrida, ruta .pt o random:<kind> (lewm, vjepa2, vjepa2_ac)")
  p.add_argument("--segment", type=int, default=0, help="segmento del val con poses")
  p.add_argument("--start", type=int, default=300, help="frame crudo donde empieza la ventana")
  p.add_argument("--pasos", type=int, default=8, help="pasos imaginados (cada uno 0,25 s)")
  p.add_argument("--decoder", help="decoder.pt del decoder de visualización (por defecto el de la corrida)")
  p.add_argument("--mp4", action="store_true", help="además del png, un mp4 a 4 Hz (lewm y vjepa2_ac con decoder)")
  p.add_argument("--scale", type=int, default=2)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--out", help="ruta del png (por defecto results/demo/<nombre>_latente.png)")
  a, ignorados = p.parse_known_args(argv)  # opciones de la demo de video (--steps, --frames, --dtype...): no aplican acá
  if ignorados:
    print(f"ignoro (la demo latente no las usa): {' '.join(ignorados)}", flush=True)
  torch.manual_seed(a.seed)
  dev = device()
  model = load_model(a.model, dev).eval()
  if type(model) not in EL.KINDS:
    raise SystemExit(f"{a.model}: la demo latente es de lewm, vjepa2 y vjepa2_ac, no de {type(model).__name__}")
  _, tpos, _ = EL.cargar(EL.TRAIN)
  EL.poner_estadisticas_pose(model, tpos, len(tpos) - EL.HOLDOUT)
  name = model.name
  a.pasos = min(a.pasos, EL.PASOS - EL.CTX)
  info = demo_mascaras(model, a, dev, name) if isinstance(model, VJEPA2) else demo_prediccion(model, a, dev, name)
  Path(info["png"]).with_suffix(".json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
  print(info["png"])
  return info


if __name__ == "__main__":
  main()
