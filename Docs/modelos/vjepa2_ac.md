# vjepa2_ac: V-JEPA 2-AC

## 1. Qué es y su fuente

El predictor con acciones de V-JEPA 2 (V-JEPA 2-AC, arXiv 2506.09985 §3.1; `app/vjepa_droid` del repo oficial) sobre un **encoder V-JEPA 2 congelado**: cada frame se codifica por separado, los tokens se intercalan (acción, estado, parches) con proyecciones afines separadas y atención causal por bloques, y la loss es L1 con teacher forcing (T = 15) más L1 de un rollout de 2 pasos. Planifica con CEM (800 muestras, las 10 mejores, 10 iteraciones, energía L1).

- **Requiere una corrida [`vjepa2`](vjepa2.md)** ya entrenada: se pasa con `--encoder vjepa2_s`. Su `target_encoder` se copia al checkpoint, que después no necesita a la corrida madre.
- **Estado y acción:** el estado es la pose del frame (6 velocidades normalizadas); la acción, la diferencia de estados entre frames consecutivos.
- **Es latente:** evaluación de error latente, probes, sorpresa y recuperación de acciones con CEM.
- **Forzado por el presupuesto:** predictor de 12 capas y ancho 384 (el del paper tiene 24 y 1024) y batch efectivo 32 con acumulación 2.
- **En la cola:** después de `vjepa2` (necesita su encoder); el humo y el entrenamiento son trabajos separados en `scripts/queue.toml`.

## 2. Fidelidad

<!-- auto:fidelidad kind=vjepa2_ac -->
| decisión | fuente / paper | aquí | categoría | fuente |
|---|---|---|---|---|
| Predictor | 24 capas, 16 cabezas, dim 1024, GELU (~300M), frame-causal, RoPE | 12 capas, 12 cabezas, dim 384 (GELU, MLP 4x) | presupuesto | V-JEPA 2 (arXiv 2506.09985) §3.1; ac_predictor.py; configs/train/vitg16/droid-256px-8f.yaml L51-60 |
| Entrada por frame | [acción, estado, parches]; acción y estado con Linear separados; RoPE 3D en parches y temporal en acción y estado | igual | fiel | ac_predictor.py L53-56, L145-153; modules.py L184-206 |
| Atención | block-causal por frame | igual | fiel | ac_predictor.py L109-117 |
| Snap de posiciones espaciales | h *= grid/H, w *= grid/W con grid = alto de la rejilla; factor 1 en su rejilla cuadrada | se conserva el código: con 8x16 las posiciones w se escalan por 8/16 | datos | modules.py L179-181 |
| Codificación de un frame | frame duplicado en un tubelet de 2; encoder congelado | igual; el encoder es el target_encoder de la corrida vjepa2 elegida (small, no ViT-g) | fiel | train.py L408-412 |
| Encoder | ViT-g (V-JEPA 2 de 1B), preentrenado con 22M videos | el de nuestra corrida vjepa2 (small/tiny), preentrenada con los clips de comma | presupuesto | V-JEPA 2 (arXiv 2506.09985) §3.1 |
| Normalización | layer_norm sobre el canal en targets y salidas del predictor | igual | fiel | train.py L413-414, L421-423 |
| Loss | L1 media, teacher forcing (T=15) + rollout de 2 pasos, 1:1 | igual (auto_steps=2) | fiel | train.py L439-449; paper Eq. 2-4 |
| Frames por muestra | clips de 4 s a 4 fps (16 frames según el paper; el yaml dice 8) | 16 frames = 1 de cada 5 de los 20 FPS, T = 15 pasos de teacher forcing | fiel | V-JEPA 2 (arXiv 2506.09985) §3.1; configs/train/vitg16/droid-256px-8f.yaml L7-19 |
| Estado y acción | pose del efector (7D) y su cambio entre frames | estado = pose del frame (6 velocidades normalizadas), acción = diferencia de estados entre frames tomados | datos | app/vjepa_droid/train.py L390-394; datos de comma |
| Aumentos | random-resized-crop de escala [0.9, 1] en DROID | ninguno (el recorte mueve la geometría que las velocidades de pose suponen fija) | propio | configs/train/vitg16/droid-256px-8f.yaml |
| Optimizador / LR | AdamW (0.9, 0.999), wd 0.04, sin clip; warmup 7.5e-5 → 4.25e-4 en 4500 de 94.5k pasos, constante, decay lineal a 0 en los últimos 4500 | igual con las mismas fracciones (4500/94500) | fiel | V-JEPA 2 (arXiv 2506.09985) §3.1; configs/train/vitg16/droid-256px-8f.yaml L61-70 |
| Batch | 256 | 32 efectivo = 16 x accum 2: con el presupuesto fijo, el batch del paper daría ~190 pasos; lr a confirmar en el humo | presupuesto | V-JEPA 2 (arXiv 2506.09985) §3.1 |
| Presupuesto | 94.5k pasos | el presupuesto común de 780k frames ≈ 1.5k pasos de 32 clips de 16 frames | presupuesto | V-JEPA 2 (arXiv 2506.09985) §3.1 |
| Planificador CEM | 800 muestras, 10 iteraciones, top-10, horizonte 1, N(0, 1), energía L1 | igual (plan.py), en el espacio normalizado de acciones | fiel | V-JEPA 2 (arXiv 2506.09985) §3.2; mpc_utils.py L17-140 |
| Decoder de visualización | ViT-L feedforward por frame, MSE, 150k pasos, batch 1024 | ViT propio chico, mismo objetivo (MSE), batch y pasos menores (train_decoder.py) | presupuesto | V-JEPA 2 (arXiv 2506.09985) Fig. 15, App. |
<!-- /auto -->

## 3. Receta de entrenamiento

<!-- auto:receta kind=vjepa2_ac -->
| modelo | fuente | optimizador | betas | lr | wd | EMA | clip | precisión | batch x frames (frameskip) x acumulación | presupuesto (frames) | extras | pasos del paper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `vjepa2_ac` | V-JEPA 2 (arXiv 2506.09985) §3.1; configs/train/vitg16/droid-256px-8f.yaml | adamw | (0.9, 0.999) | warmup: 7.5e-05->0.000425 (0.047619); constante: 0.000425 (resto); decay: 0.000425->0 (0.047619) | 0.04 | no | None | bf16 | 16 x 16 (5) x 2 | 780000 | - | 94500 |

Valores forzados (campo: motivo, categoría):

- `vjepa2_ac`: batch: 32 efectivo (16 x accum 2) en vez de 256: con el presupuesto común de 780k frames da ~1.5k pasos (con 256 serían ~190); lr a confirmar en el humo (presupuesto)
- `vjepa2_ac`: presupuesto: ~1.5k pasos en vez de 94.5k; las fracciones de warmup y decay se conservan (presupuesto)
- `vjepa2_ac`: predictor: 12 x 384 en vez de 24 x 1024 (presupuesto)
- `vjepa2_ac`: frames/frameskip: 16 frames a 4 fps = 1 de cada 5 de los 20 FPS (datos)
<!-- /auto -->

## 4. Acciones

<!-- auto:acciones modelo=vjepa2_ac -->
Evaluación: `latente`.

| acción | qué hace |
|---|---|
| `vjepa2_ac receta` | muestra la receta y la Config efectivas, con los overrides y los valores forzados marcados |
| `vjepa2_ac entrenar` | entrena de cero con la receta de la fuente (`--humo`: prueba corta de 300 pasos para elegir el lr) |
| `vjepa2_ac continuar` | retoma una corrida cortada o, con `--presupuesto`, la extiende |
| `vjepa2_ac evaluar` | mide el modelo (video: CE, rollouts, LPIPS y FVD; latentes: error, probes, sorpresa y CEM) |
| `vjepa2_ac bench` | mide la latencia (el modelo solo, o la cadena de punta a punta con `--cadena`) |
| `vjepa2_ac demo` | genera la demostración (mp4 con el video real y el predicho, o figura para los latentes) |
| `vjepa2_ac exportar` | exporta a ONNX |
| `vjepa2_ac estado` | porcentaje, paso, loss, lr, tramo y ETA de las corridas del modelo |
<!-- /auto -->

### receta

<!-- auto:opciones modelo=vjepa2_ac accion=receta -->
- `--set`: receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli vjepa2_ac receta
```

### entrenar

Sin `--encoder` la CLI se niega (sí lo acepta con `--humo`, pero con un encoder aleatorio: sirve para elegir lr y batch, no para evaluar).

<!-- auto:opciones modelo=vjepa2_ac accion=entrenar -->
- `--set`: receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)
- `--humo`: tres entrenamientos de 300 pasos (lr de la receta, por raíz y lineal) y elige
- `--humo-candidatos`: con --humo: candidatos a probar, p. ej. fuente,raiz,x3 (default: fuente,raiz,lineal,x3,x10)
- `--encoder`: corrida vjepa2 de la que sale el encoder congelado
- `--forzar`: entrenar: seguir aunque la corrida ya tenga last.pt o final.pt (train.py la reanuda desde last.pt)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli vjepa2_ac entrenar --encoder vjepa2_s --humo
python -m src.cli vjepa2_ac entrenar --encoder vjepa2_s
python -m src.cli vjepa2_ac entrenar --encoder vjepa2_s --variante prueba --limite 20 --device cpu
```

### continuar

Igual que en `vjepa2`: warmup, constante y decaimiento, con `pre_decay.pt`, así que la extensión es exacta. El encoder ya está dentro del checkpoint.

<!-- auto:opciones modelo=vjepa2_ac accion=continuar -->
- `--set`: receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)
- `--restart`: continuar con re-warmup y coseno nuevo (no equivalente)
- `--encoder`: corrida vjepa2 de la que sale el encoder congelado

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli vjepa2_ac continuar
python -m src.cli vjepa2_ac continuar --presupuesto 10x
```

### evaluar

Evaluación latente con el plan: error de rollout por horizonte contra copiar el último latente, probes, sorpresa y recuperación de acciones con CEM. Los segundos por plan quedan en el JSON: el CEM es caro.

<!-- auto:opciones modelo=vjepa2_ac accion=evaluar -->
- `--dtype`: precisión de la inferencia
- `--temperatura`: temperatura de muestreo
- `--ctx-frames`: frames de contexto en la inferencia (default 20 = 1 s)
- `--no-plan`: evaluación latente sin el plan con CEM

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli vjepa2_ac evaluar
python -m src.cli vjepa2_ac evaluar --no-plan
python -m src.cli vjepa2_ac evaluar --limite 50 --device cpu
```

### bench

Latencia de un paso del predictor.

<!-- auto:opciones modelo=vjepa2_ac accion=bench -->
- `--dtype`: precisión de la inferencia
- `--temperatura`: temperatura de muestreo
- `--ctx-frames`: frames de contexto en la inferencia (default 20 = 1 s)
- `--int8`: cuantización int8 dinámica en CPU
- `--aleatorio`: modelo sin entrenar con la arquitectura de la corrida (latencia)
- `--cadena`: bench: latencia de la cadena de punta a punta; evaluar: calidad en línea y con la cadena (src.infer.quality)
- `--diagnostico`: con --cadena: el piso de ~20 ms a ritmo de cámara (bench_e2e --diagnostico)
- `--modo`: con --cadena: online o imagine (default: los dos)
- `--ritmo`: con --cadena: milisegundos entre frames (0 = bucle cerrado)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli vjepa2_ac bench
python -m src.cli vjepa2_ac bench --aleatorio --device cpu --limite 20
```

### demo

Curva de error y latentes; con `--entrenar-decoder` entrena el decoder de visualización sobre el encoder indicado (si no se lo pasa, lo busca en el checkpoint).

<!-- auto:opciones modelo=vjepa2_ac accion=demo -->
- `--dtype`: precisión de la inferencia
- `--temperatura`: temperatura de muestreo
- `--aleatorio`: modelo sin entrenar con la arquitectura de la corrida (latencia)
- `--encoder`: corrida vjepa2 de la que sale el encoder congelado
- `--entrenar-decoder`: entrena el decoder de píxeles (encoder congelado) en vez del demo

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli vjepa2_ac demo
python -m src.cli vjepa2_ac demo --entrenar-decoder --encoder vjepa2_s
```

### exportar

ONNX del predictor (su encoder es el de `vjepa2` y se exporta con ese modelo).

<!-- auto:opciones modelo=vjepa2_ac accion=exportar -->
- `--dtype`: precisión de la inferencia

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli vjepa2_ac exportar
```

### estado

<!-- auto:opciones modelo=vjepa2_ac accion=estado -->
- `--vigilar`: estado: repite cada N segundos (default 10)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli vjepa2_ac estado
```

## 5. Qué archivos escribe

| Acción | Archivos |
|---|---|
| `entrenar`, `continuar` | `checkpoints/<run>/{last.pt, pre_decay.pt, final.pt, log.jsonl, status.json}`, `results/train_<run>.json`, entrada de `results/checkpoints.json`. |
| `evaluar` | `results/latent_<run>.json`. |
| `bench` | `results/bench/<run>__<dtype>.json`. |
| `demo` | `results/demo/<run>_latente.png`; con `--entrenar-decoder`, `checkpoints/<encoder>_viz/decoder.pt`. |
| `exportar` | `checkpoints/onnx/<run>_predictor.onnx` y `results/export/<run>_predictor.json`. |

## 6. Problemas frecuentes

- **"necesita un encoder congelado":** pasá `--encoder <corrida vjepa2>` (o `--set config.encoder_run=...`). Esa corrida tiene que existir en `checkpoints/` o estar registrada.
- **El encoder no encuentra la corrida madre:** entrená antes `vjepa2 entrenar` o copiá su carpeta a `checkpoints/<run>/` y registrala con `python -m src.store register <run>`.
- **El CEM tarda:** `evaluar --no-plan` lo omite; `--limite` achica el número de planes.
- **El modelo casi no usa la acción:** es un resultado a medir, no un bug; la evaluación incluye la recuperación de acciones con CEM contra referencias.
- **"la corrida ya existe", árbol sucio, GPU ocupada:** ver [`../uso.md`](../uso.md).

## 7. Notebook

[`notebooks/modelos/vjepa2_ac.ipynb`](../../notebooks/modelos/vjepa2_ac.ipynb): el error de rollout por horizonte y la recuperación de acciones.
