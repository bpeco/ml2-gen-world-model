"""Muestreador de máscaras multi-secuencia multi-bloque 3D de V-JEPA 2, vectorizado en torch (CPU o GPU).

Fuente: src/masks/multiseq_multiblock3d.py (clases MaskCollator y _MaskGenerator) y la receta
configs/train/vitl16/pretrain-256px-16f.yaml L41-67:
  conjunto A: 8 bloques, escala espacial 0.15; conjunto B: 2 bloques, escala 0.7; aspecto [0.75, 1.5];
  escala temporal 1.0 (el bloque cubre todo el tiempo). Cada muestra lleva los dos conjuntos.

Qué se conserva del oficial (L145-269):
  - L145-171: el tamaño del bloque (t, h, w) se sortea UNA vez por llamada y es el mismo para todo el lote y todos los
    bloques del conjunto: t = max(1, int(T*escala_t)); n = int(H*W*escala_s); h = round(sqrt(n*ar)), w = round(sqrt(n/ar)),
    recortados a la rejilla (round de Python = mitad a par = torch.round).
  - L173-188: la posición (top, left, start) de cada bloque se sortea por muestra y por bloque, uniforme.
  - L214-219: la máscara de predicción es la unión de los bloques; el contexto es el complemento.
  - L211-229: si el contexto queda vacío se resortea esa muestra.
  - L231-235: el nº de contexto y de predichos se recorta al mínimo del lote con cm[:min], sobre los índices en orden
    ascendente: se pierden los de mayor índice (final del clip, abajo a la derecha). Es un sesgo del oficial; se conserva.
  - L263-269: devuelve (enc, pred) como tensores (B, K) de índices planos (t-mayor, luego h, luego w).
No se implementan max_context_frames_ratio, max_keep, full_complement, pred_full_complement, inv_block (la receta no los usa).

Diferencias de implementación (misma estadística, no el mismo flujo de números aleatorios):
  - Todo son operaciones de torch en `device` con un `torch.Generator` de ese dispositivo; sin numpy ni bucles por
    muestra. El oficial siembra el generador del tamaño con un contador (L197-199); aquí se pasa `generator`
    (o, si es None, se siembra uno con el contador del propio objeto).
  - El resorteo por contexto vacío se hace sorteando CANDIDATOS intentos por muestra y quedándose con el primero válido
    (si ninguno lo fuera, que es prácticamente imposible, se resortea el banco entero).
  - Las formas de salida dependen del mínimo de contexto/predichos del lote, así que algún escalar tiene que llegar a
    la CPU para fijar las formas. Para no hacerlo en cada paso, `banco(n_pasos, ...)` sortea n_pasos lotes de una vez y
    hace UNA sola lectura (una lista de enteros) para todo el banco. `__call__` es `banco(1)`. Memoria transitoria
    del sorteo: ~ n_pasos * CANDIDATOS * B * num_blocks * N bytes (N = T*H*W tokens).
"""
import torch

CANDIDATOS = 8

# Receta de preentrenamiento (yaml L41-67).
CFG_MASCARAS = (
  dict(num_blocks=8, spatial_scale=(0.15, 0.15), temporal_scale=(1.0, 1.0), aspect_ratio=(0.75, 1.5)),
  dict(num_blocks=2, spatial_scale=(0.7, 0.7), temporal_scale=(1.0, 1.0), aspect_ratio=(0.75, 1.5)),
)


def _interp(r, rango):
  lo, hi = rango
  return lo + r * (hi - lo)


class MaskGenerator:
  """Un conjunto de máscaras (equivale a un _MaskGenerator). grid = (T, H, W) en tokens (T ya dividido por el tubelet)."""

  def __init__(self, grid, num_blocks, spatial_scale, temporal_scale=(1.0, 1.0), aspect_ratio=(0.75, 1.5),
               device="cpu"):
    self.duration, self.height, self.width = grid
    self.npred = num_blocks
    self.spatial_scale, self.temporal_scale, self.aspect_ratio = spatial_scale, temporal_scale, aspect_ratio
    self.device = torch.device(device)
    self._step = -1

  def _generador(self, generator):
    if generator is not None:
      return generator
    self._step += 1  # como el contador de L138-143 (L197-199)
    g = torch.Generator(device=self.device)
    g.manual_seed(self._step)
    return g

  def _sortear(self, n, B, g):
    """n pasos x B muestras -> (pred (n, B, N) bool = tokens a predecir, ok (n, B) bool = hubo candidato con contexto)."""
    T, H, W, dev = self.duration, self.height, self.width, self.device
    f64 = torch.float64
    # tamaño por paso (L145-171)
    r = torch.rand(n, 3, generator=g, device=dev, dtype=f64)
    t = torch.floor(T * _interp(r[:, 0], self.temporal_scale)).clamp(min=1)
    num = torch.floor(H * W * _interp(r[:, 1], self.spatial_scale))
    ar = _interp(r[:, 2], self.aspect_ratio)
    h = torch.round(torch.sqrt(num * ar)).clamp(max=H)
    w = torch.round(torch.sqrt(num / ar)).clamp(max=W)
    # posiciones por paso, candidato, muestra y bloque (L173-180)
    u = torch.rand(n, CANDIDATOS, B, self.npred, 3, generator=g, device=dev, dtype=f64)
    v = lambda x: x.view(n, 1, 1, 1)
    top = torch.floor(u[..., 0] * v(H - h + 1))
    left = torch.floor(u[..., 1] * v(W - w + 1))
    start = torch.floor(u[..., 2] * v(T - t + 1))
    tt = torch.arange(T, device=dev, dtype=f64).view(T, 1, 1)
    yy = torch.arange(H, device=dev, dtype=f64).view(1, H, 1)
    xx = torch.arange(W, device=dev, dtype=f64).view(1, 1, W)
    e = lambda x: x[..., None, None, None]  # (n, C, B, K) -> (n, C, B, K, 1, 1, 1)
    dentro = ((tt >= e(start)) & (tt < e(start + v(t))) & (yy >= e(top)) & (yy < e(top + v(h)))
              & (xx >= e(left)) & (xx < e(left + v(w))))  # (n, C, B, K, T, H, W)
    pred = dentro.any(3).flatten(-3)  # unión de bloques: (n, C, B, N)
    valido = ~pred.all(-1)  # contexto no vacío (L224): (n, C, B)
    primero = valido.int().argmax(1)  # primer candidato válido: (n, B)
    idx = primero[:, None, :, None].expand(-1, 1, -1, pred.shape[-1])
    return pred.gather(1, idx).squeeze(1), valido.any(1)

  def banco(self, n_pasos, batch_size, generator=None):
    """n_pasos lotes -> lista de (enc (B, Kc), pred (B, Kp)) long en `device`; una sola lectura a CPU en total."""
    g = self._generador(generator)
    while True:
      pred, ok = self._sortear(n_pasos, batch_size, g)
      n_pred = pred.sum(-1)  # (n, B)
      N = pred.shape[-1]
      lectura = torch.stack([(N - n_pred).amin(1), n_pred.amin(1), ok.all(1).long()], 1).tolist()  # única lectura
      if all(m[2] for m in lectura):
        break
    # índices ascendentes de cada clase (el orden estable pone primero los 1) y recorte al mínimo (L231-235)
    ord_enc = torch.sort((~pred).int(), dim=-1, descending=True, stable=True).indices
    ord_pred = torch.sort(pred.int(), dim=-1, descending=True, stable=True).indices
    return [(ord_enc[i, :, :me], ord_pred[i, :, :mp]) for i, (me, mp, _) in enumerate(lectura)]

  def __call__(self, batch_size, generator=None):
    return self.banco(1, batch_size, generator)[0]


class MultiSeqMasks:
  """Los conjuntos de la receta. __call__(B) -> (masks_enc [lista de (B, Kc_s)], masks_pred [lista de (B, Kp_s)])."""

  def __init__(self, grid, cfgs=CFG_MASCARAS, device="cpu"):
    self.gens = [MaskGenerator(grid, device=device, **c) for c in cfgs]

  def banco(self, n_pasos, batch_size, generator=None):
    """n_pasos x (masks_enc, masks_pred); una lectura a CPU por conjunto en todo el banco."""
    por_conjunto = [g.banco(n_pasos, batch_size, generator) for g in self.gens]
    return [([c[i][0] for c in por_conjunto], [c[i][1] for c in por_conjunto]) for i in range(n_pasos)]

  def __call__(self, batch_size, generator=None):
    return self.banco(1, batch_size, generator)[0]


def multiseq_multiblock3d(batch_size, grid=(8, 8, 16), cfgs=CFG_MASCARAS, device="cpu", generator=None):
  """Atajo: un lote de máscaras de la receta. grid = (T/tubelet, H/parche, W/parche) = (8, 8, 16) para clips 16x128x256."""
  return MultiSeqMasks(grid, cfgs, device)(batch_size, generator)
