import torch

BOS = 1024
VOCAB = 1025          # 1024 códigos del VQ-VAE + BOS
TPF = 129             # tokens por frame (BOS + 8x16)
CTX_FRAMES = 20       # 1 s a 20 FPS, igual que gpt2m
BLOCK = CTX_FRAMES * TPF


def device():
  return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def with_bos(frames):
  """(..., F, 128) tokens -> (..., F*129) con BOS al inicio de cada frame."""
  bos = torch.full((*frames.shape[:-1], 1), BOS, dtype=frames.dtype, device=frames.device)
  return torch.cat([bos, frames], -1).flatten(-2)
