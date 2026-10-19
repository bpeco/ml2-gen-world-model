"""Decoder del VQ-VAE de comma (tokens -> imagen) y figuras de datos para las slides."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "third_party/commavq")
from utils.vqvae import CompressorConfig, Decoder  # noqa: E402

DECODER = "data/raw/decoder_pytorch_model.bin"


def load_decoder(dev="cpu"):
  d = Decoder(CompressorConfig())
  d.load_state_dict(torch.load(DECODER, map_location="cpu", weights_only=True))
  return d.eval().to(dev)


@torch.no_grad()
def decode(decoder, tokens, bs=16):
  """tokens (N, 128) -> imágenes uint8 (N, 128, 256, 3)."""
  dev = next(decoder.parameters()).device
  t = torch.as_tensor(np.asarray(tokens, dtype=np.int64)).to(dev)
  out = torch.cat([decoder(t[i:i + bs]).cpu() for i in range(0, len(t), bs)])
  return out.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).numpy()


def fig_frame_tokens(seg, decoder, path, frame=600):
  img = decode(decoder, seg[frame:frame + 1])[0]
  fig, ax = plt.subplots(1, 2, figsize=(10, 2.6), gridspec_kw={"wspace": 0.08})
  ax[0].imshow(img)
  ax[0].set_title("frame decodificado (128x256 px)", fontsize=10)
  ax[1].imshow(seg[frame].reshape(8, 16), cmap="viridis", vmin=0, vmax=1023)
  for (r, c), v in np.ndenumerate(seg[frame].reshape(8, 16)):
    ax[1].text(c, r, v, ha="center", va="center", fontsize=5.5, color="w")
  ax[1].set_title("lo que ve el modelo: 8x16 = 128 tokens (0-1023)", fontsize=10)
  for a in ax:
    a.axis("off")
  fig.savefig(path, dpi=200, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
  val = np.load("data/val_tokens.npy")
  dec = load_decoder()
  fig_frame_tokens(val[0], dec, "results/fig_frame_tokens.png")
  # tira de 4 frames a lo largo de un segmento
  frames = decode(dec, val[0][[0, 300, 600, 900]])
  plt.imsave("results/fig_segment_strip.png", np.concatenate(frames, 1))
  print("ok")
