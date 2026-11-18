"""Bloques de V-JEPA 2 (arXiv 2506.09985; código oficial github.com/facebookresearch/vjepa2).

Solo componentes: el bucle de entrenamiento, el target EMA, la loss y el predictor condicionado a acciones viven fuera.

Convenciones: clip normalizado (B, 3, T, H, W) con T=16, H=128, W=256 -> rejilla (T/2, H/16, W/16) = (8, 8, 16) = 1024 tokens,
ids planos t-mayor (id = t*H*W + h*W + w). Todo son operaciones de torch sin numpy, con tablas cacheadas por dispositivo.

  patch.normalizar(frames_uint8)           (B, T, H, W, 3) uint8 -> (B, 3, T, H, W) con media/std de ImageNet
  patch.PatchEmbed3D, frame_a_tubelet, clip_a_frames
  rope.rope_3d(q, k, coords, oficial=True) coords_desde_ids(ids, H, W); rotar(x, pos, oficial)
  masks.MultiSeqMasks(grid=(8, 8, 16), device)  -> __call__(B, generator) = (masks_enc, masks_pred), listas de tensores
        (B, K) long en `device` (conjunto A, conjunto B); .banco(n_pasos, B, generator) = n_pasos lotes con una sola
        lectura a CPU (las formas dependen del mínimo del lote; ver masks.py). masks.multiseq_multiblock3d(B, ...) atajo.
  vit.VJEPAEncoder.preset("tiny"|"small");  z = enc(x, masks)        -> (S*B, Kc, D), S conjuntos concatenados en batch
        enc(x) -> (B, N, D) (sin máscara; target); enc.encode_frames(x) -> ruta de un frame duplicado (AC)
  vit.VJEPAPredictor(embed_dim, grid_hw=(8, 16));  p = pred(z, masks_enc, masks_pred) -> (S*B, Kp, D)
        (listas de S tensores o tensores ya concatenados). El target de la loss es enc(x) recortado con las mismas
        masks_pred (apply_masks(h, masks_pred)) tras layer_norm sin parámetros; en esto el llamador sigue train.py L430-447.
  vit.FIDELIDAD, vit.PRESETS, vit.PRED_PRESET

Modelos completos (registrados en src/models/__init__.py como kinds `vjepa2` y `vjepa2_ac`):
  pretrain.VJEPA2 (+ aug.py: recorte/flip en GPU)      train_loss(tokens|píxeles), after_step = EMA 0.99925, recipe()
  action.VJEPA2AC (+ ac.py, plan.py)                   encoder congelado (target_encoder de una corrida vjepa2), AC predictor,
                                                       train_loss(poses=...), planear() = CEM 800/10/10
  decoder.py / train_decoder.py                        decoder de visualización: python -m src.models.vjepa.train_decoder --run <vjepa2>
  src/infer/stream.py VJEPAStream                      cámara 20 Hz -> predictor cada 5 frames (4 Hz)
Cada módulo lista sus desviaciones en FIDELIDAD_PRETRAIN / FIDELIDAD_AC (categoría fiel|datos|presupuesto|propio).

Nota del oficial: app/vjepa llama al encoder y al predictor una vez por conjunto (utils/wrappers.py L25-28, L40-43); aquí,
equivalente, se pueden pasar los conjuntos en lista y salen concatenados en el batch (vision_transformer.py L166-193).
"""
from src.models.vjepa.masks import MultiSeqMasks, multiseq_multiblock3d  # noqa: F401
from src.models.vjepa.patch import normalizar  # noqa: F401
from src.models.vjepa.vit import FIDELIDAD, PRESETS, VJEPAEncoder, VJEPAPredictor, apply_masks  # noqa: F401
