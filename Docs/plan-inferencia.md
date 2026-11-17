# Próximas pruebas: inferencia de punta a punta en GPU y NPU

**Estado:** plan, sin implementar (17/11/2026). Primeras mediciones en la sección 10 de [`resultados.md`](resultados.md).

## Por qué

Lo que importa en uso es la cadena completa, no el world model solo: **video de entrada → encoder VQ → world model → decoder VQ → video de salida**. El objetivo es que corra **en tiempo real (50 ms por frame, 20 FPS)** con la mejor calidad posible, y con todo residente en la GPU o la NPU, sin pasar por la CPU entre etapas.

Hasta ahora eso no se medía:
- El bench mide solo el world model, desde tokens.
- El encoder de commaVQ nunca se usó: partimos de los tokens del dataset.
- El decoder se midió aparte: 15 ms en fp32.

**No hay que reentrenar nada.** Los checkpoints sirven tal cual, y `src/models/` no cambia.

## Lo que ya se midió (RTX 3080 Ti, batch 1)

| Etapa | fp32, pasando por la CPU entre etapas | fp16 + CUDA graph, todo en la GPU |
|---|---:|---:|
| Encoder commaVQ (29,6M) | 7,4 ms | 3,4 ms |
| Decoder commaVQ (42,8M) | 14,2 ms | 5,8 ms (PSNR 61 dB contra fp32) |
| Subir y bajar un frame (uint8, 96 KB) | | 0,07 y 0,05 ms |
| **Cadena con MaskGIT-S, 1 paso** | **38,5 ms** | **14,0 ms** (= 3,4 + 5,2 + 5,8) |

- **fp16 en el codec:** el cuantizador del VQ-VAE mezcla tipos de dato, así que en fp16 tiene que quedar en fp32.
- **`channels_last` no ayuda al encoder:** 3,44 contra 3,37 ms.
- **Estimado de punta a punta** (encoder 3,4 + modelo + decoder 5,8):

| Modelo | ms por frame | FPS |
|---|---:|---:|
| LeWM-XS | ~10,0 | ~100 |
| ConvLSTM | ~10,3 | ~97 |
| JEPA-XXS | ~10,4 | ~96 |
| JEPA-S | ~12,1 | ~83 |
| MaskGIT-S, 1 paso | 14,0 (medido) | 71 |
| MaskGIT-S, 4 pasos | ~19,3 | ~52 |
| GPT-S | ~165 | 6 |

**Lo que dicen estas mediciones:**
1. **Residente en la GPU es 2,75 veces más rápido**, sin reentrenar.
2. **Con los modelos rápidos, el encoder y el decoder son el 85–90 % de la cadena.** Achicar más el world model casi no cambia el total.
3. **En la 3080 Ti sobran ~40 ms para un world model más grande o con más pasos.** El desafío pasa a ser la mejor calidad por ms de punta a punta.
4. **En otros dispositivos puede no alcanzar:**
   - La T4 (la GPU de referencia, D1) puede ser 4–5 veces más lenta en convoluciones: el codec solo rondaría 35–45 ms.
   - En la iGPU de la laptop, el decoder solo podría pasar los 50 ms.
   - Por eso, lo primero es medir el codec en cada dispositivo.

## Fases

### 1. Codec y medición en cada dispositivo (punto de decisión)

- **Pesos:** bajar `encoder_pytorch_model.bin` y `encoder.onnx` de `commaai/commavq-gpt2m` a `data/raw/`, con chequeo de sha256.
- **`src/infer/codec.py`:**
  - Encoder y decoder en fp16, con el cuantizador en fp32, cada uno en un CUDA graph.
  - Entrada y salida: uint8 128×256×3 en la GPU.
  - El preprocesado (rango, orden de canales) se fija contra `encoder.onnx`, porque `Encoder.forward` no lo define.
- **Validación:**
  - El encoder en torch contra `encoder.onnx` en ONNX Runtime con CPU: ~100 % de tokens iguales.
  - fp16 contra fp32: se decide por el efecto medido en LPIPS.
  - Decoder: PSNR contra fp32.
- **`python -m src.infer.bench_codec`**, más una celda en `notebooks/colab.ipynb` para la T4 e instrucciones para la laptop (iGPU con ROCm).
- **Decisión:** si el codec pasa los ~35 ms en la T4 o en la laptop, la palanca principal es optimizarlo: TensorRT o un decoder más liviano. Eso rompe la regla de "codec fijo" del plan del proyecto, así que se decide en grupo.

### 2. Estado de inferencia en la GPU (`src/infer/stream.py`)

API por frame: `observe(frame)`, `predict()` e `imagine(n)`. El estado vive en la GPU.

**Tres adaptadores:**
- **Genérico** (MaskGIT, GPT, otros): un buffer circular de tokens en la GPU más `next_frame`. El prefill de MaskGIT también va dentro del graph.
- **ConvLSTM:** un paso con (h, c) explícitos.
  - Antes, medir si el estado deriva en un stream largo (1200 frames): se entrenó con ventanas de 20.
  - Si deriva, reiniciarlo cada tanto.
- **LatentWM** (JEPA y LeWM): agrega el latente del frame nuevo a la caché, con pose opcional.

**Pasos puros, con el estado de entrada y de salida explícitos:** el calentamiento de los CUDA graphs ejecuta la función varias veces. Eso muta el estado y consume números aleatorios. El muestreo va fuera del graph, o el ruido entra como dato.

**Validación:**
- En CPU con temperatura 0, igualdad exacta contra `generate`.
- En GPU, ≥ 99,9 % de tokens iguales.
- Tests en `tests/test_infer.py`.

### 3. Protocolo de latencia (`src/infer/bench_e2e.py`)

- **Modos:**
  - **En línea:** encoder + modelo + decoder.
  - **Imaginación:** modelo + decoder; el encoder solo al principio.
  - **Predicción a k frames:** k pasos del modelo y un solo decode.
- **Cómo se mide:**
  - Al menos 3000 frames, porque el p99 importa para el tiempo real.
  - En bucle cerrado y también a ritmo de 50 ms: con la GPU ociosa bajan los relojes.
  - Tiempo de reloj desde la subida hasta la bajada del frame. Reporta p50, p99 y % de frames que llegan tarde.
  - El primer segundo va aparte: con menos de 19 frames de contexto todavía no hay graphs.

### 4. Calidad en cada modo

- **En línea:** una métrica a 1 paso, acierto de tokens y LPIPS contra el frame real siguiente, con el baseline de copia (a 20 FPS es muy fuerte).
- **Imaginación:** se reusan FVD y LPIPS de los videos ya generados. Con 1 o 2 modelos se verifica que el camino nuevo dé lo mismo.

### 5. La mejor calidad debajo de 50 ms, por dispositivo

- **Barrido:** pasos de MaskGIT (1, 2, 4, 8, 12), temperatura, tamaño S/M y predicción a k frames.
- **Salida:**
  - Pareto FVD contra p99 de punta a punta, por dispositivo.
  - Columna nueva en `results/summary.md`.
  - Figura `fig_pareto_e2e.png`.

### 6. Demo (`src/infer/demo.py`)

Un mp4 con el video real y el predicho lado a lado. La entrada es un segmento de val, o un video propio llevado a 128×256, que queda fuera de la distribución de entrenamiento.

### 7. Laptop: iGPU y NPU (Ryzen AI 7 350)

- **iGPU:** PyTorch + ROCm. DirectML ya dio 77 ms contra 11. Hay que ver si los CUDA graphs funcionan en ROCm.
- **NPU:** ONNX Runtime con el Vitis AI EP.
  - Entre etapas cruzan solo 1 KB de tokens o 96 KB por frame, así que no hace falta IO binding.
  - El encoder se exporta sin el cuantizador.
  - El decoder se exporta desde los embeddings.
  - ConvLSTM, como un paso con (h, c).
- **Orden de medición:**
  1. Decoder y encoder solos, en cada dispositivo.
  2. Qué operaciones caen en la CPU.
  3. Los dos a la vez, porque comparten el ancho de banda de memoria.

## Decisiones

El plan tuvo una revisión independiente (Claude Fable 5.2), y de ahí salieron estas decisiones:

- **Medir el codec en cada dispositivo antes de construir el resto.** Puede cambiar qué conviene optimizar.
- **No fusionar las tres etapas en un solo CUDA graph:** la cadena ya da la suma de sus partes.
- **Sin adaptador propio para GPT:** 6 FPS.

## Esfuerzo

- ~13 h de trabajo y ~2–3 h de GPU.
- Las corridas en Colab (T4) y en la laptop son de ~10–20 min cada una.
- La fase 1 puede cambiar el resto.
