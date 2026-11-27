"""Condicionamiento por pose del GPT (fase 3 de motion, src/models/gpt.py, Config.pose_dim).

  - init en cero: con los mismos pesos, el modelo con pose_dim = 6 da exactamente los logits del modelo sin poses
    (con poses o sin ellas), así que un ajuste fino arranca desde el modelo base
  - causalidad: la pose del frame i (ego-movimiento i -> i+1) solo cambia los logits desde el bloque del frame i+1
  - generación: el decode con caché KV (y CUDA graphs si hay GPU) da los mismos tokens que el argmax de teacher forcing
    sobre la secuencia generada con las mismas poses
  python -m pytest tests/test_pose_gpt.py -q
"""
import pytest
import torch

from src.common import TPF, with_bos
from src.models.gpt import GPT

CFG = dict(n_layer=2, n_head=2, dim=32, dropout=0.0)


def par(seed=0):
  torch.manual_seed(seed)
  base = GPT(GPT.Config(**CFG)).eval()
  pm = GPT(GPT.Config(**CFG, pose_dim=6)).eval()
  falta, sobra = pm.load_state_dict(base.state_dict(), strict=False)
  assert not sobra and {k.split(".")[0] for k in falta} == {"pose_proj", "pose_mean", "pose_std"}
  pm.set_pose_stats(torch.zeros(6), torch.ones(6) * 2)
  return base, pm


def datos(B=2, F=4, seed=1):
  g = torch.Generator().manual_seed(seed)
  return torch.randint(0, 1024, (B, F, 128), generator=g), torch.randn(B, F, 6, generator=g)


def test_init_cero_igual_al_base():
  base, pm = par()
  fr, po = datos()
  x = with_bos(fr)[:, :-1]
  with torch.no_grad():
    ref = base.logits(x)
    assert torch.equal(pm.logits(x), ref)
    assert torch.equal(pm.logits(x, po), ref)


def test_causalidad():
  _, pm = par()
  torch.nn.init.normal_(pm.pose_proj.weight, 0, 0.5)
  fr, po = datos(F=4)
  x = with_bos(fr)[:, :-1]
  i = 1
  po2 = po.clone()
  po2[:, i] += 3.0
  with torch.no_grad():
    a, b = pm.logits(x, po), pm.logits(x, po2)
  corte = (i + 1) * TPF                      # primera posición del bloque del frame i+1 (su BOS)
  assert torch.equal(a[:, :corte], b[:, :corte])
  assert not torch.allclose(a[:, corte:], b[:, corte:])
  # NaN = sin pose: igual que no pasar poses en ese frame
  po3 = po.clone()
  po3[:, :] = float("nan")
  with torch.no_grad():
    assert torch.allclose(pm.logits(x, po3), pm.logits(x), atol=1e-6)


@pytest.mark.parametrize("dev,graphs", [("cpu", False)] + ([("cuda", True)] if torch.cuda.is_available() else []))
def test_generacion_igual_a_teacher_forcing(dev, graphs):
  _, pm = par()
  torch.nn.init.normal_(pm.pose_proj.weight, 0, 0.5)
  pm = pm.to(dev)
  pm.use_cuda_graphs(graphs)
  fr, po = datos(F=5)
  fr, po = fr.to(dev), po.to(dev)
  ctx, n = fr[:, :3], 2                      # 3 frames de contexto, 2 generados; poses de los 5 frames
  gen = pm.rollout(ctx, n, temperature=0, poses=po)
  seq = torch.cat([ctx, gen], 1)
  x = with_bos(seq)[:, :-1]
  with torch.no_grad():
    lg = pm.logits(x, po).float()
  for k in range(n):
    f = ctx.shape[1] + k
    arg = lg[:, f * TPF:f * TPF + 128, :1024].argmax(-1)
    assert (arg == gen[:, k]).float().mean() > 0.99, (k, (arg == gen[:, k]).float().mean())
  # y la pose importa: otras poses -> otros frames
  gen2 = pm.rollout(ctx, n, temperature=0, poses=po + 5)
  assert not torch.equal(gen, gen2)
