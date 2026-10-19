"""Rollouts: desde 19 frames reales de val, el modelo genera N frames nuevos (sampling, temp 1).

Mide coherencia vs horizonte contra el futuro real:
  - token match: % de tokens iguales al frame real en ese instante
  - PSNR entre frame decodificado generado y real
Referencia trivial: "copiar el último frame" (el mundo no se mueve).

Uso:
  python -m src.generate --model checkpoints/gpt_mini.pt --name gpt_mini
  python -m src.generate --model gpt2m --name gpt2m
  python -m src.generate --plot gpt_mini gpt2m
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common import BLOCK, BOS, CTX_FRAMES, TPF, device, with_bos
from src.evaluate import load_model

SEGMENTS = list(range(0, 100, 12))  # 9 segmentos de val, espaciados
START = 600                         # primer frame a predecir (mitad del minuto)
N_NEW = 40                          # 2 s a 20 FPS
FPS = 20


@torch.no_grad()
def rollout_mini(model, ctx, n_new):
  """ctx (B, 19, 128). Sin KV cache: recalcula la ventana entera por token (alcanza para un modelo chico)."""
  B, dev = ctx.shape[0], ctx.device
  out = []
  for _ in range(n_new):
    cur = torch.full((B, 1), BOS, dtype=torch.long, device=dev)
    for _ in range(TPF - 1):
      x = torch.cat([with_bos(ctx), cur], 1)
      logits = model(x)[:, -1, :BOS]
      nxt = torch.multinomial(torch.softmax(logits.float(), -1), 1)
      cur = torch.cat([cur, nxt], 1)
    out.append(cur[:, 1:])
    ctx = torch.cat([ctx[:, 1:], cur[:, None, 1:]], 1)
  return torch.stack(out, 1)


@torch.no_grad()
def rollout_gpt2m(model, ctx, n_new):
  """Usa el generate() con KV cache de comma, de a una secuencia. Caches en fp32 para MPS."""
  from utils.gpt import KVCache
  c, dev = model.config, ctx.device
  L = BLOCK + 8 - BLOCK % 8
  for b in model.transformer.h:
    b.attn.kv_cache = KVCache(1, L, c.n_head, c.head_dim, dtype=torch.float32).to(dev)
  model.causal_mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev)).view(1, 1, L, L)
  model.max_batch_size, model.max_seq_length = 1, L
  outs = []
  for seq in ctx:
    frames = []
    for _ in range(n_new):
      new = model.generate(with_bos(seq).view(-1), TPF)[1:].long()  # descarta el BOS generado
      new[new == BOS] = seq[-1][new == BOS]  # muy raro: BOS en medio del frame -> copia el slot previo
      frames.append(new)
      seq = torch.cat([seq[1:], new[None]], 0)
    outs.append(torch.stack(frames))
  return torch.stack(outs)


def psnr(a, b):
  mse = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean(axis=(-1, -2, -3))
  return 10 * np.log10(255 ** 2 / np.maximum(mse, 1e-6))


def horizon_metrics(gen, real, decoder):
  from src.viz import decode
  match = (gen == real).mean(-1).mean(0)  # (N_NEW,)
  B, N = gen.shape[:2]
  img_g = decode(decoder, gen.reshape(-1, 128)).reshape(B, N, 128, 256, 3)
  img_r = decode(decoder, real.reshape(-1, 128)).reshape(B, N, 128, 256, 3)
  return match, psnr(img_g, img_r).mean(0), img_g, img_r


def plot(names):
  from src.viz import decode, load_decoder
  dec = load_decoder()
  runs = {n: np.load(f"results/rollout_{n}.npz") for n in names}
  first = runs[names[0]]
  real, last = first["real"], first["ctx_last"]
  copy = np.repeat(last[:, None], N_NEW, 1)
  t = np.arange(1, N_NEW + 1) / FPS
  res = {"seconds": t.tolist()}
  fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
  for label, gen in [("copiar último frame", copy)] + [(n, r["gen"]) for n, r in runs.items()]:
    m, p, _, _ = horizon_metrics(gen, real, dec)
    res[label] = {"token_match": m.tolist(), "psnr": p.tolist()}
    style = dict(ls="--", color="gray") if label.startswith("copiar") else {}
    ax[0].plot(t, 100 * m, label=label, **style)
    ax[1].plot(t, p, label=label, **style)
  ax[0].set(xlabel="segundos hacia adelante", ylabel="% tokens = real", title="token match vs horizonte")
  ax[1].set(xlabel="segundos hacia adelante", ylabel="PSNR (dB)", title="PSNR vs horizonte")
  ax[0].legend(fontsize=8)
  for a in ax:
    a.grid(alpha=0.3)
  fig.tight_layout()
  fig.savefig("results/fig_horizon.png", dpi=200, bbox_inches="tight")
  json.dump(res, open("results/horizon.json", "w"), indent=1)

  # tira: real vs generado en +0.25 / +0.5 / +1 / +2 s, para un segmento (filas = horizonte)
  k, idx = 0, [4, 9, 19, 39]
  cols = [decode(dec, real[k, idx])] + [decode(dec, r["gen"][k, idx]) for r in runs.values()]
  labels = ["real"] + [f"generado: {n}" for n in runs]
  fig, ax = plt.subplots(len(idx), len(cols), figsize=(3.2 * len(cols), 1.75 * len(idx)), squeeze=False)
  for j, (col, label) in enumerate(zip(cols, labels)):
    ax[0, j].set_title(label, fontsize=17)
    for i, img in enumerate(col):
      ax[i, j].imshow(img)
      ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
      if j == 0:
        ax[i, j].set_ylabel(f"+{(idx[i] + 1) / FPS:g} s", fontsize=17)
  fig.tight_layout(h_pad=0.4, w_pad=0.4)
  fig.savefig("results/fig_rollout_strip.png", dpi=200, bbox_inches="tight", transparent=True)
  print(json.dumps({k: (v if k == "seconds" else {m: [round(x, 3) for x in v[m][::10]] for m in v}) for k, v in res.items() if k != "seconds"}, indent=1))


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--model")
  ap.add_argument("--name")
  ap.add_argument("--plot", nargs="+")
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()
  if args.plot:
    plot(args.plot)
  else:
    torch.manual_seed(args.seed)
    dev = device()
    model = load_model(args.model, dev)
    val = np.load("data/val_tokens.npy").astype(np.int64)
    ctx = torch.from_numpy(val[SEGMENTS, START - (CTX_FRAMES - 1):START]).to(dev)
    t0 = time.time()
    gen = (rollout_gpt2m if args.model == "gpt2m" else rollout_mini)(model, ctx, N_NEW)
    print(f"{args.name}: {time.time() - t0:.0f}s")
    np.savez(f"results/rollout_{args.name}.npz", gen=gen.cpu().numpy(), real=val[SEGMENTS, START:START + N_NEW],
             ctx_last=val[SEGMENTS, START - 1])
