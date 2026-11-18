import numpy as np
import torch

from src import pixels


def test_render_igual_al_codec():
  from src.infer.codec import Codec
  tok = torch.from_numpy(np.load("data/val_tokens.npy", mmap_mode="r")[0, :3].astype(np.int64))
  img = pixels.render(tok[None], chunk=2)
  assert img.shape == (1, 3, 128, 256, 3) and img.dtype == torch.uint8
  ref = Codec("cpu", "fp32", graphs=False, encoder=False).decode(tok)
  # el chunk cambia el tamaño de batch y con él el orden de las sumas fp32: a lo sumo 1-2 niveles de gris
  assert (img[0].int() - ref.int()).abs().max() <= 2


def test_cache_memmap(tmp_path):
  tok = np.load("data/val_tokens.npy", mmap_mode="r")[:1, :2]
  fr = pixels.frames(tok, "t", 0, 1, 2, cache_dir=tmp_path, dev="cpu")
  assert fr.shape == (1, 128, 256, 3)
  assert len(list(tmp_path.glob("pixels_t_*.npy"))) == 1
  ref = pixels.render(torch.as_tensor(np.asarray(tok[0, :2], np.int64)))[1].numpy()
  assert np.abs(fr.astype(int) - ref.astype(int)).max() <= 2
