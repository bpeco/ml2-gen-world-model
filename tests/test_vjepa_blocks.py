"""Bloques de V-JEPA 2 (CPU, tamaños chicos)."""
import pytest
import torch

from src.models.vjepa import FIDELIDAD, MultiSeqMasks, VJEPAEncoder, VJEPAPredictor, apply_masks, normalizar
from src.models.vjepa.masks import CFG_MASCARAS, MaskGenerator
from src.models.vjepa.patch import PatchEmbed3D, clip_a_frames
from src.models.vjepa.rope import coords_desde_ids, dim_rope, rope_3d
from src.models.vjepa.vit import PRESETS, RoPEAttention

GRID = (8, 8, 16)


def _gen(seed=0):
  g = torch.Generator()
  g.manual_seed(seed)
  return g


def test_patch_y_normalizacion():
  x = torch.randint(0, 256, (2, 4, 32, 48, 3), dtype=torch.uint8)
  n = normalizar(x)
  assert n.shape == (2, 3, 4, 32, 48)
  assert abs(n[0, 0, 0, 0, 0].item() - (x[0, 0, 0, 0, 0].item() / 255 - 0.485) / 0.229) < 1e-5
  pe = PatchEmbed3D(16, 2, 3, 24)
  assert pe(n).shape == (2, 2 * 2 * 3, 24)
  assert clip_a_frames(n).shape == (8, 3, 2, 32, 48)


def test_formas_y_presets():
  assert PRESETS["tiny"] == dict(embed_dim=192, depth=12, num_heads=3)
  assert PRESETS["small"] == dict(embed_dim=384, depth=12, num_heads=6)
  assert VJEPAEncoder.preset("tiny").embed_dim == 192
  assert all(set(f) == {"campo", "paper", "aquí", "categoría", "fuente"} for f in FIDELIDAD)
  enc = VJEPAEncoder(embed_dim=48, depth=2, num_heads=3)
  x = torch.randn(2, 3, 4, 32, 64)  # rejilla (2, 2, 4) = 16 tokens
  assert enc(x).shape == (2, 16, 48)
  assert enc.encode_frames(x).shape == (2, 4 * 8, 48)  # 4 frames x 2x4 parches
  pred = VJEPAPredictor(embed_dim=48, grid_hw=(2, 4), predictor_dim=24, depth=2, num_heads=2)
  mc, mt = torch.tensor([[0, 1, 2, 3, 8, 9]] * 2), torch.tensor([[4, 5, 6]] * 2)
  assert pred(enc(x, mc), mc, mt).shape == (2, 3, 48)
  assert pred.mask_token.abs().sum() == 0


@pytest.mark.parametrize("hd", [32, 64])
def test_rope_invariante_a_desplazamientos(hd):
  # con la corrección (repeat_interleave) el producto q·k solo depende de la posición relativa
  torch.manual_seed(0)
  N = 40
  q, k = torch.randn(1, 2, N, hd), torch.randn(1, 2, N, hd)
  ids = torch.randperm(8 * 8 * 16)[:N][None]
  t, h, w = coords_desde_ids(ids, 8, 16)
  base = lambda c, o: (lambda qk: qk[0] @ qk[1].transpose(-2, -1))(rope_3d(q, k, c, o))
  for eje in range(3):
    c = [t, h, w]
    c2 = list(c)
    c2[eje] = c[eje] + 3.0
    assert torch.allclose(base(c, False), base(c2, False), atol=1e-4)
  assert 3 * dim_rope(hd) <= hd
  # la versión oficial (frecuencias duplicadas) no lo es: documentado en rope.py
  assert not torch.allclose(base([t, h, w], True), base([t + 3, h, w], True), atol=1e-4)


def test_atencion_invariante_con_rope_corregido():
  torch.manual_seed(0)
  att = RoPEAttention(48, 3, oficial=False)
  x = torch.randn(2, 10, 48)
  ids = torch.randint(0, 16, (2, 10))
  s1 = att.scores(x, ids, 2, 4)
  s2 = att.scores(x, ids + 8, 2, 4)  # +8 = un paso en t (H*W = 8)
  assert torch.allclose(s1, s2, atol=1e-4)


def test_rope_posiciones_desde_ids():
  t, h, w = coords_desde_ids(torch.tensor([[0, 1, 15, 16, 127, 128]]), 8, 16)
  assert t.tolist() == [[0, 0, 0, 0, 0, 1]] and h.tolist() == [[0, 0, 0, 1, 7, 0]] and w.tolist() == [[0, 1, 15, 0, 15, 0]]


def test_mascaras_estadistica_y_disjuntas():
  m = MultiSeqMasks(GRID)
  g = _gen(1)
  fracs = {0: [], 1: []}
  for _ in range(150):
    enc, pred = m(16, g)
    for s in range(2):
      e, p = enc[s], pred[s]
      assert e.shape[0] == p.shape[0] == 16 and e.dtype == torch.long
      assert e.shape[1] + p.shape[1] <= 1024
      for i in range(16):  # context y target disjuntos y en rango
        assert not (set(e[i].tolist()) & set(p[i].tolist()))
      assert e.max() < 1024 and p.min() >= 0
      assert (e[:, 1:] > e[:, :-1]).all() and (p[:, 1:] > p[:, :-1]).all()  # ascendentes, sin repetidos
      fracs[s].append(p.shape[1] / 1024)
  fa, fb = sum(fracs[0]) / 150, sum(fracs[1]) / 150
  print(f"fracción enmascarada (tras recorte al mínimo): A={fa:.3f}  B={fb:.3f}  media={(fa + fb) / 2:.3f}")
  assert 0.3 < fa < 0.95 and 0.5 < fb <= 1.0


def test_mascaras_bloques_respetan_escala_y_aspecto():
  # un solo bloque: la máscara es un rectángulo h x w que cubre todo el tiempo
  for seed in range(30):
    gen = MaskGenerator(GRID, num_blocks=1, spatial_scale=(0.15, 0.15), aspect_ratio=(0.75, 1.5))
    _, pred = gen.banco(1, 6, _gen(seed))[0]
    p = pred[0]
    f = torch.zeros(1024, dtype=torch.bool)
    f[p] = True
    f = f.view(8, 8, 16)
    assert (f == f[0:1]).all()  # todo el tiempo (escala temporal 1)
    ys, xs = f[0].nonzero(as_tuple=True)
    hh, ww = int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)
    # el recorte al mínimo del lote solo quita tokens del final: la caja puede perder filas, nunca crecer
    assert hh * ww >= 0.5 * f[0].sum() and f[0].sum() <= int(128 * 0.15) + 1
    n = int(128 * 0.15)
    assert 0.75 - 0.2 <= (hh / ww) <= 1.5 + 0.7 and hh <= round((n * 1.5) ** 0.5) and ww <= round((n / 0.75) ** 0.5)


def test_mascaras_sin_recorte_tamano_exacto():
  # con batch 1 no hay recorte: el bloque es exactamente h x w x T
  for seed in range(20):
    gen = MaskGenerator(GRID, num_blocks=1, spatial_scale=(0.15, 0.15), aspect_ratio=(0.75, 1.5))
    _, pred = gen(1, _gen(seed))
    f = torch.zeros(1024, dtype=torch.bool)
    f[pred[0]] = True
    f = f.view(8, 8, 16)
    ys, xs = f[0].nonzero(as_tuple=True)
    hh, ww = int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)
    assert hh * ww == int(f[0].sum()) and f.sum() == 8 * hh * ww
    assert 0.75 - 0.25 <= hh / ww <= 1.5 + 0.25  # el redondeo de h y w mueve algo la razón


def test_mascaras_determinista_cpu_sin_numpy(monkeypatch):
  def sin_numpy(*a, **k):
    raise AssertionError("numpy en el camino del sampler")
  monkeypatch.setattr(torch.Tensor, "numpy", sin_numpy)
  m = MultiSeqMasks(GRID, device="cpu")
  a = m(8, _gen(7))
  b = m(8, _gen(7))
  c = m(8, _gen(8))
  for xs, ys in zip(a, b):
    for x, y in zip(xs, ys):
      assert x.device.type == "cpu" and torch.equal(x, y)
  assert not torch.equal(a[1][0], c[1][0])
  # banco: n pasos con una lectura; mismo resultado que n llamadas encadenadas no se exige (otro flujo), pero sí formas válidas
  banco = m.banco(5, 4, _gen(3))
  assert len(banco) == 5 and all(e[0].shape[0] == 4 for e, _ in banco)


def test_encoder_con_mascaras_igual_a_tokens_reunidos():
  torch.manual_seed(0)
  enc = VJEPAEncoder(embed_dim=48, depth=2, num_heads=3).eval()
  x = torch.randn(2, 3, 4, 32, 64)
  m = torch.stack([torch.randperm(16)[:7].sort().values for _ in range(2)])
  with torch.no_grad():
    z = enc(x, m)
    # referencia: bloques a mano sobre los tokens reunidos, con los ids originales como posición; rejilla (2, 2, 4)
    t = torch.gather(enc.patch_embed(x), 1, m[..., None].expand(-1, -1, 48))
    for b in enc.blocks:
      t = b(t, m, 2, 4)
    ref = enc.norm(t)
    assert torch.allclose(z, ref, atol=1e-5)
    # máscara completa = sin máscara
    todo = torch.arange(16)[None].expand(2, -1)
    assert torch.allclose(enc(x, todo), enc(x), atol=1e-5)
    # lista de dos conjuntos: concatenados en el batch, cada uno igual a su forward suelto
    m2 = torch.stack([torch.randperm(16)[:7].sort().values for _ in range(2)])
    zz = enc(x, [m, m2])
    assert zz.shape == (4, 7, 48)
    assert torch.allclose(zz[:2], z, atol=1e-5) and torch.allclose(zz[2:], enc(x, m2), atol=1e-5)
    assert apply_masks(torch.randn(2, 16, 5), [m, m2]).shape == (4, 7, 5)


def test_predictor_solo_ve_su_secuencia():
  torch.manual_seed(0)
  enc = VJEPAEncoder(embed_dim=48, depth=2, num_heads=3).eval()
  pred = VJEPAPredictor(embed_dim=48, grid_hw=(2, 4), predictor_dim=24, depth=2, num_heads=2).eval()
  for p in pred.parameters():  # el mask token en 0 y el init chico dejan salidas casi planas: se vuelve sensible
    p.data.add_(0.05 * torch.randn_like(p))
  x = torch.randn(2, 3, 4, 32, 64)
  mc = [torch.tensor([[0, 1, 2, 3, 8, 9, 10]] * 2), torch.tensor([[4, 5, 6, 7, 12, 13, 14]] * 2)]
  mt = [torch.tensor([[5, 6, 7, 12]] * 2), torch.tensor([[0, 1, 2, 15]] * 2)]
  with torch.no_grad():
    z = enc(x, mc)
    out = pred(z, mc, mt)
    assert out.shape == (4, 4, 48)
    # cada conjunto igual a su forward suelto
    for s in range(2):
      suelto = pred(z[2 * s:2 * s + 2], mc[s], mt[s])
      assert torch.allclose(out[2 * s:2 * s + 2], suelto, atol=1e-5)
    # perturbar el contexto de otra secuencia no cambia esta salida
    z2 = z.clone()
    z2[2:] += torch.randn_like(z2[2:])
    assert torch.allclose(pred(z2, mc, mt)[:2], out[:2], atol=1e-5)
    # y perturbar un token de contexto de la propia secuencia sí la cambia
    z3 = z.clone()
    z3[0, 0] += 5.0
    assert not torch.allclose(pred(z3, mc, mt)[0], out[0], atol=1e-4)
    # el orden de los tokens objetivo solo permuta la salida (atiende solo a contexto + máscaras, por posición)
    perm = torch.tensor([3, 0, 2, 1])
    o2 = pred(z[:2], mc[0], mt[0][:, perm])
    assert torch.allclose(o2, pred(z[:2], mc[0], mt[0])[:, perm], atol=1e-5)
