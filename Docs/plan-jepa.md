# E5: world models latentes, JEPA (V-JEPA) y LeWorldModel

Suma a la comparación de [`resultados.md`](resultados.md) dos familias que **predicen la representación del
próximo frame** en vez de clasificar sus tokens. Cada una sigue la receta de su paper.
**Estado: ejecutado** (17/11/2026). Resultados en la sección 7 de [`resultados.md`](resultados.md) y en `results/hypotheses.md` (H5 a H7).

**En una línea por familia:**
- **JEPA-S:** 2,9 ms por frame y FVD a 1 s de 460. Su control CE, con la misma arquitectura, da 265.
- **LeWM:** 1,0–1,4 ms por frame y planifica 42 veces más barato, pero su video queda al nivel del unigrama (FVD 1016).
- **Acciones:** los dos casi no usan la pose, y el CEM no la recupera.
- **Versiones chicas (E5b):** JEPA-XXS da 1,2 ms y FVD 546; LeWM-XS, 0,8 ms con la misma calidad que LeWM. Ninguna entra en la frontera de Pareto.
- **Lo que sigue:** medir la cadena entera, video → encoder → modelo → decoder → video ([`plan-inferencia.md`](plan-inferencia.md)).

- **`jepa`, estilo V-JEPA / V-JEPA 2-AC** (secciones 10 y 14 de [`world-models-conduccion-papers.md`](world-models-conduccion-papers.md)):
  - Latente **espacial**: 128 celdas × 384 por frame.
  - Encoder target **EMA** con stop-grad.
  - Pérdida L1.
  - La acción entra como un **token** más del frame.
- **`lewm`, estilo LeWorldModel** (Maes, Le Lidec, Scieur, LeCun y Balestriero, arXiv 2603.19312; código en `github.com/lucas-maes/le-wm`, MIT):
  - **Un vector por frame** (CLS de 192).
  - **SIGReg** en vez de EMA, sin stop-grad.
  - Pérdida MSE.
  - La acción entra con **AdaLN-zero**.
  - Planificación con CEM.

Las dos se diferencian en dos ejes: cuán detallado es el latente y cómo evitan el colapso. Dos corridas cruzadas completan un 2×2.

## Arquitecturas

| | `jepa` ([`src/models/jepa.py`](../src/models/jepa.py)) | `lewm` ([`src/models/lewm.py`](../src/models/lewm.py)) |
|---|---|---|
| Entrada | Tokens → vector del codebook del VQ-VAE (congelado, 1024 × 256) → `Linear` | Igual: los 8 × 16 tokens hacen de patches |
| Encoder | `Backbone` de 2 capas, D 384, bidireccional dentro del frame, LN sin parámetros → 128 × 384 | ViT-Tiny: 12 capas, 3 cabezas, D 192 → CLS → projector MLP + BatchNorm → 192 |
| Target | Copia EMA del encoder (0,996 → 1, coseno), stop-grad | El mismo encoder, sin stop-grad |
| Predictor | `Backbone` de 4 capas, causal entre frames (máscara de MaskGIT), 129 lugares por frame: acción + 128 celdas | Transformer de 6 capas, 16 cabezas, D 320, dropout 0,1, causal en el tiempo, AdaLN-zero con la acción, `pred_proj` MLP + BN |
| Pérdida | `L1(ẑ, sg(target))` | `MSE(ẑ, z) + 0,09 · SIGReg(z)` (1024 proyecciones, 17 nodos) |
| Lectura a tokens | `LN + Linear 384 → 1024` por celda | Decoder de cross-attention: 128 queries (una por celda) sobre el vector |
| Params de inferencia | 11,6M (como GPT-S y MaskGIT-S) | 21,7M (≈ 18M de JEPA + 3M de decoder) |

**Común a las dos** (en [`src/models/latent.py`](../src/models/latent.py)):
- **Son `FrameModel`:** las celdas salen independientes dado el pasado, como en ConvLSTM. Tienen CE completa con el protocolo de comma.
- **Frame vacío aprendido:** abre la ventana, así el frame 0 se predice sin contexto.
- **Lectura sin influencia:** se entrena con la CE sobre `ẑ.detach()` y sobre el target. Así mide el "techo" de información del latente, y no le pasa gradiente al JEPA. Un test lo verifica.
- **Acciones = poses del shard**, extraídas con `python -m src.data --poses-only`. Son 6 velocidades por frame: traslación en m/s y giros en rad/s.
  - **Dropout de acción** con p = 0,25: la pose se cambia por una acción nula aprendida.
  - En val (sin poses) corren con la acción nula, y la comparación con el resto es justa.
  - En el holdout separado por ruta corren con las poses reales.
- **Rollout:** `token` re-codifica el frame muestreado; `latent` mete ẑ al contexto. Hay caché de latentes y CUDA graphs.

## Adaptaciones respecto de los papers

| Qué | Por qué |
|---|---|
| Tokens en vez de píxeles (LeWM) | Decodificar cuesta 16 ms/frame, y pre-decodificar los 3M frames de train ocuparía ≈ 295 GB. El grid de tokens es el de patches de 16 px |
| SIGReg multiplicado por un N de referencia fijo (128, el batch del paper) | El original multiplica por el N de cada llamada. Acá N varía (B·T vectores en LeWM, 2048 celdas muestreadas en JEPA), y así λ = 0,09 significa lo mismo |
| Predictor de LeWM con D 320 | La sección 3.1 del paper da ~10M; con AdaLN y D 384 (el ViT-S del apéndice) serían 16M |
| Contexto de 19 frames (LeWM usa 3) | Igual que el resto de la tabla. Con un vector por frame es barato |
| Dropout de acción y acción nula | `val.zip` no tiene poses y el resto de los modelos no ve acciones |
| CEM con 10 iteraciones, H = 10 frames, historia 3, arrancando en la última pose | Con 30 iteraciones, JEPA tardaba más de 1 min por plan. La última pose es la referencia natural al manejar; LeWM arranca en N(0, 1) |
| lr: JEPA 1e-3 (wd 0,1, como GPT-S); LeWM 3e-4 (wd 1e-3 del paper) | Elegidos en corridas de humo de 400 pasos por menor error de predicción sin colapso. Con 100M tokens (2.423 pasos), el 5e-5 del paper no alcanza |
| Barrido de λ a 100M tokens (el plan decía 20M) | LeWM entrena 100M tokens en ~6 min, así que el barrido es comparable directo con `lewm_s` |

## Corridas (100M tokens, semilla 0, mismos batches que E2/E3)

| Corrida | Latente | Antiaplastamiento | Para qué |
|---|---|---|---|
| `jepa_s` | Por celda | EMA | V-JEPA, principal |
| `jepa_s_ce` | Por celda | — (CE entrena todo) | Control de H5 |
| `jepa_s_sigreg` | Por celda | SIGReg | 2×2 (H7) |
| `jepa_s_codebook` | Por celda | Target = codebook congelado | No puede colapsar |
| `lewm_s` | Vector | SIGReg | LeWM, principal |
| `lewm_s_ce` | Vector | — (CE entrena todo) | Control de H5 |
| `lewm_s_ema` | Vector | EMA | 2×2 (H7) |
| `lewm_s_l003`, `lewm_s_l03` | Vector | SIGReg con λ 0,03 y 0,3 | Sensibilidad al único hiperparámetro |
| `jepa_xs`, `jepa_xxs` | Por celda | EMA | E5b: 4 capas con D 256 (3,7M) y 3 capas con D 192 (1,7M) |
| `lewm_xs` | Vector | SIGReg | E5b: ViT de 6 capas y predictor de 4 (10,9M) |

## Evaluación

- **Tabla común** (`bash scripts/eval_all.sh quality` y `bench-latent`): CE en val, rollouts sobre los 400 clips a temperatura 1, LPIPS y FVD, y ms/frame en fp16 con CUDA graphs.
  - Los principales entran en la tabla y en el Pareto de [`results/summary.md`](../results/summary.md).
  - Los controles y ablaciones van en el grupo "latente".
  - Las figuras propias son `fig_pareto_latente*.png`: ningún cuarto color pasa el validador de paleta junto a los de GPT, MaskGIT y ConvLSTM.
- **Evaluaciones estilo LeWM** (`bash scripts/eval_all.sh latent`, [`src/eval_latent.py`](../src/eval_latent.py)), sobre los 46 segmentos del holdout cuya ruta no aparece en train:
  - **Probes lineal y MLP de pose** (velocidad y giro) desde ẑ y desde z. Referencias: GPT-S y MaskGIT-S.
  - **Sorpresa** ante un corte a otro segmento ("teletransporte") y ante un salto de 2 s.
  - **Rectitud** de la trayectoria latente.
  - **Error latente** con pose real vs. acción nula.
  - **Planificación con CEM** contra repetir la última pose y contra la pose media.
- **Recortes para ahorrar GPU** (toda la evaluación tardó ~1,5 h):
  - El barrido de λ tiene CE y evaluaciones latentes, pero no rollouts de video (~25 min).
  - Los controles `jepa_s_ce` y `jepa_s_codebook` no tienen CEM, que no aporta a H6 (~35 min). Las versiones chicas de JEPA tampoco.

## Guardado y reutilización

Cada corrida queda en `checkpoints/<corrida>/final.pt`, con el codebook, la normalización de poses y el encoder target como buffers. Se anota en `results/checkpoints.json` (sha256, config, params) con `python -m src.store register`, que `scripts/train_all.sh` ya hace solo.
- **Cargar:** `load_model("jepa_s")` o `load_model("lewm_s")`, por nombre, en `evaluate`, `rollouts`, `bench` y `eval_latent`.
- **Copia afuera** (si hace falta): `python -m src.store push --all --rollouts --dir <carpeta>` (o `--hf <repo>`).

## Validación

31 tests nuevos en [`tests/test_latent.py`](../tests/test_latent.py):
- **Sin fugas del futuro:** las 7 variantes, con tokens y con acciones.
- **SIGReg:** ≈ 0 con muestras gaussianas, alto con embeddings colapsados.
- **Stop-grad y EMA:**
  - El gradiente del encoder con EMA es el de la pérdida latente sola.
  - EMA con momento 0 deja target = online.
- **Inicialización:** con AdaLN-zero, la acción no cambia la salida.
- **Generación:**
  - La generación con caché da los mismos tokens que la genérica.
  - CUDA graphs da lo mismo que eager.
- **Planificación:** CEM recupera el mínimo.
- **Reanudación:** se reproduce bit a bit, incluidos el dropout de acción y las proyecciones de SIGReg.
- **Guardado:** ida y vuelta `register` → `load_model("<corrida>")`.
- **Poses:** alineadas con los tokens.
