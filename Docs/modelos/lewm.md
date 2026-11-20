# lewm: LeWorldModel desde píxeles

## 1. Qué es y su fuente

LeWorldModel (Maes, Le Lidec, Scieur, LeCun y Balestriero, arXiv 2603.19312; código oficial `github.com/lucas-maes/le-wm`): un world model latente que aprende de punta a punta desde píxeles, con un embedding por frame (CLS de un ViT-Tiny más un projector con BatchNorm), un predictor autoregresivo AdaLN-zero que recibe las poses como acción, y la regularización SIGReg que evita el colapso, sin stop-grad ni EMA. Es una reescritura fiel del repo oficial; lo que cambia por los datos o el cómputo está en la tabla de fidelidad.

- **Reemplaza** al LeWM sobre tokens (y a `lewm_px`), que se borraron.
- **Entrada:** frames de 128x256 renderizados al vuelo con el decoder de comma congelado, historia 3 y frameskip 5 (4 Hz), con las 5 poses apiladas como acción.
- **Es latente:** no emite tokens, no tiene CE ni FVD. Se evalúa con `eval_latent`: error latente contra "copiar el último latente" por horizonte, probes contra un encoder sin entrenar y contra píxeles, sorpresa y recuperación de acciones con CEM.
- **Decoder de visualización:** se entrena aparte (`demo --entrenar-decoder`), con el modelo congelado y MSE; solo sirve para figuras.
- **En la cola:** ver `scripts/queue.toml`. La corrida 10x va al final de todo (etapa D).

## 2. Fidelidad

<!-- auto:fidelidad kind=lewm -->
| decisión | fuente / paper | aquí | categoría | fuente |
|---|---|---|---|---|
| Encoder | ViT-tiny: 12 capas, 3 cabezas, 192; CLS de la última capa; sin preentrenar | igual (HF ViT: pre-LN, eps 1e-12, LayerNorm final, init N(0, 0.02) truncada) | fiel | LeWM (arXiv 2603.19312) §3; github.com/lucas-maes/le-wm config/train/model/lewm.yaml L3-9 |
| Imagen y parche | 224 x 224, parche 14 (grilla 16 x 16) | RGB 128 x 256, parche 16 (grilla 8 x 16): lo que da la cámara de comma | datos | github.com/lucas-maes/le-wm lewm.yaml L3-9 (img_size 224) |
| Normalización de la imagen | media y desvío de ImageNet | igual | fiel | github.com/lucas-maes/le-wm utils.py get_img_preprocessor |
| Origen de los frames | video RGB | el dataset solo tiene tokens VQ: los frames se renderizan al vuelo con el decoder de comma congelado (fp16, sin gradiente, en el device); el modelo ve las reconstrucciones del VQ-VAE | datos | commaVQ; src/pixels.py |
| Projector y pred_proj | Linear(192 → 2048), BatchNorm1d, GELU, Linear(2048 → 192); otro igual después del predictor | igual | fiel | github.com/lucas-maes/le-wm lewm.yaml L29-45; module.py MLP |
| Predictor | ARPredictor: 6 capas, 16 cabezas x dim_head 64 (inner 1024), MLP 2048, dropout 0.1, emb_dropout 0, dim 192, QKV sin bias, AdaLN-zero (6 vectores con gate), posición aprendida, causal, LayerNorm final | igual | fiel | github.com/lucas-maes/le-wm lewm.yaml L11-22; module.py L68, L88-111, L244-285 |
| Embedder de la acción | Conv1d(k=1) a smoothed_dim 10 (el default de module.py) + MLP 4x con SiLU; input_dim = frameskip x action_dim | igual con input_dim = 5 x 6 = 30 | fiel | github.com/lucas-maes/le-wm module.py L189-214; data/pusht.yaml L3 |
| Acción | acciones de PushT (2D), z-score por columna; NaN → 0 | las 6 velocidades de traslación y giro de la pose de comma por frame crudo, z-score con las estadísticas de train (buffers del checkpoint); NaN → 0 | datos | github.com/lucas-maes/le-wm train.py L24, utils.py |
| Historia y predicciones | history_size 3, num_preds 1 (4 frames por muestra) | igual | fiel | github.com/lucas-maes/le-wm lewm.yaml L15, L36; train.py L32-36 |
| Frameskip | 5 | 5: de 20 fps a 4 Hz (un paso del modelo = 250 ms) | fiel | github.com/lucas-maes/le-wm data/pusht.yaml L3 |
| Loss | MSE(pred, objetivo) + λ·SIGReg, λ 0.09, sin stop-grad ni EMA | igual | fiel | github.com/lucas-maes/le-wm train.py L39-41; lewm.yaml L38-40 |
| SIGReg | (T, B, D) por paso de tiempo a través del batch; 1024 proyecciones nuevas por forward; 17 nodos en [0, 3]; trapecio con ventana exp(-t²/2); por B; promedio sobre tiempo y proyecciones | igual; en float32 y con las proyecciones generadas en el device | fiel | github.com/lucas-maes/le-wm module.py L10-36 (la precisión: propio; el oficial corre bajo autocast bf16) |
| Optimizador | AdamW lr 5e-5, wd 1e-3, betas por defecto | igual, wd en todos los parámetros | fiel | github.com/lucas-maes/le-wm lewm.yaml L31-34 |
| Variante lewm_lrx3 | lr 5e-5 | la corrida fiel (lewm) usa el lr del paper y con ~1.500 pasos apenas superó a copiar el último latente. El humo de 300 pasos (fuente, x3, x10, x30: monitor 0,261 / 0,222 / 0,250 / 0,266, sin colapso; results/humo_lewm.json) eligió x3, así que se entrena además lewm_lrx3 con --set receta.lr_peak=1.5e-4 (misma forma: warmup 1 % y coseno a 0). Las dos van a la tabla | presupuesto | github.com/lucas-maes/le-wm lewm.yaml L31-34 |
| Schedule | LinearWarmupCosineAnnealingLR por época: warmup 1 % desde 0, coseno a 0 | igual pero por paso: warmup lineal sobre el 1 % de los pasos (piso 1) y coseno a 0 | presupuesto | github.com/lucas-maes/le-wm train.py L87-94; con 1 500 pasos no hay épocas que contar |
| Precisión y clip | bf16, clip 1.0 | igual (bf16 solo en CUDA) | fiel | github.com/lucas-maes/le-wm lewm.yaml L17-29 |
| Batch | 128 muestras de 4 frames | 128 x 4, sin acumulación | fiel | github.com/lucas-maes/le-wm lewm.yaml L17-29; si la GPU de 12 GB no alcanza, accum rompe la BatchNorm y SIGReg: ajustar bs |
| Presupuesto | 100 épocas en la config, 10 en el paper (PushT) | 780k frames renderizados vistos (~1 500 pasos de 128 x 4) | presupuesto | LeWM (arXiv 2603.19312) App. E; github.com/lucas-maes/le-wm lewm.yaml |
| Datos de las muestras | trayectorias de PushT, split 90/10 por episodio | ventanas de 20 frames crudos (4 a frameskip 5) de los segmentos de comma; holdout aparte | datos | src/windows.py |
| CEM | 300 muestras, 30 élites, horizonte 5, 10 iteraciones (non-PushT), media 0 y varianza inicial 1 en el espacio normalizado | igual; la acción de cada paso son las 5 poses apiladas (30 dims) | fiel | github.com/lucas-maes/le-wm config/eval/solver/cem.yaml, config/eval/pusht.yaml; LeWM (arXiv 2603.19312) App. D |
| Costo del CEM | MSE sumado del embedding final predicho contra el de la meta | igual: rollout latente de `horizonte` pasos con la historia de 3 | fiel | github.com/lucas-maes/le-wm jepa.py criterion |
| Elites del CEM | no leído del código del solver (stable-worldmodel) | media y desvío de los élites, desvío con piso 1e-4 | propio | CEM estándar |
| Evaluación latente | el planificador (CEM, arriba); lo demás no está en las fuentes leídas | src/eval_latent.py: probes de velocidad y giro contra un encoder sin entrenar y contra píxeles; error latente por horizonte contra copiar el último latente; sorpresa; recuperación de acciones con CEM contra repetir la última acción. Los detalles propios van en el bloque `propio` del JSON | propio | src/eval_latent.py |
| Cosas propias que se quitaron | no existen | sin dropout de acción ni acción nula, sin frame start aprendido, sin rollout de tokens, sin lectura a tokens ni CE | fiel | github.com/lucas-maes/le-wm jepa.py |
| Streaming | no existe (evaluación offline en PushT) | LatentStream: cámara → ViT → predictor en la GPU, un paso del modelo cada 5 frames de cámara, sin codec | propio | src/infer/stream.py |
| Decoder de visualización | transformer liviano: CLS → cross-attention con una query aprendida por parche → parches; solo para visualizar; código no publicado | src/models/lewm_decoder.py, entrenado aparte sobre el modelo congelado (profundidad, ancho, MSE y optimizador: propio) | propio | LeWM (arXiv 2603.19312) App. D |
| Tamaño xs | no existe | reducción propia (dim 96, encoder de 6 capas, predictor de 4 capas y 8 cabezas x 32): opcional, el oficial es base | propio | Config.sized('xs') |
<!-- /auto -->

## 3. Receta de entrenamiento

<!-- auto:receta kind=lewm -->
| modelo | fuente | optimizador | betas | lr | wd | EMA | clip | precisión | batch x frames (frameskip) x acumulación | presupuesto (frames) | extras | pasos del paper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lewm` | LeWM (arXiv 2603.19312); github.com/lucas-maes/le-wm config/train/lewm.yaml, train.py | adamw | (0.9, 0.999) | warmup: 0->5e-05 (0.01); cosine: coseno 5e-05->0 (resto) | 0.001 (todos) | no | 1.0 | bf16 | 128 x 4 (5) x 1 | 780000 | - | - |

Valores forzados (campo: motivo, categoría):

- `lewm`: lr_segs: el oficial planifica por época (warmup en pasos); acá todo por paso: warmup lineal sobre el 1 % de los pasos (piso 1, sin el piso de 100) y coseno a 0 (presupuesto)
- `lewm`: presupuesto: 780k frames renderizados vistos (~1 500 pasos); el oficial son 10-100 épocas de PushT (presupuesto)
- `lewm`: batch: 128 muestras de 4 frames sin acumulación (BatchNorm y SIGReg dependen del batch); si la GPU no alcanza, bajar bs en vez de acumular (presupuesto)
- `lewm`: frames: 4 por muestra (history 3 + num_preds 1): la unidad de presupuesto son los frames renderizados (datos)
<!-- /auto -->

BatchNorm y SIGReg dependen del batch: por eso el batch es 128 sin acumulación. Si la GPU no alcanza, se baja `receta.batch` (queda como desvío), no se acumula.

## 4. Acciones

<!-- auto:acciones modelo=lewm -->
Evaluación: `latente`.

| acción | qué hace |
|---|---|
| `lewm receta` | muestra la receta y la Config efectivas, con los overrides y los valores forzados marcados |
| `lewm entrenar` | entrena de cero con la receta de la fuente (`--humo`: prueba corta de 300 pasos para elegir el lr) |
| `lewm continuar` | retoma una corrida cortada o, con `--presupuesto`, la extiende |
| `lewm evaluar` | mide el modelo (video: CE, rollouts, LPIPS y FVD; latentes: error, probes, sorpresa y CEM) |
| `lewm bench` | mide la latencia (el modelo solo, o la cadena de punta a punta con `--cadena`) |
| `lewm demo` | genera la demostración (mp4 con el video real y el predicho, o figura para los latentes) |
| `lewm exportar` | exporta a ONNX |
| `lewm estado` | porcentaje, paso, loss, lr, tramo y ETA de las corridas del modelo |
<!-- /auto -->

### receta

<!-- auto:opciones modelo=lewm accion=receta -->
- `--set`: receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli lewm receta
```

### entrenar

<!-- auto:opciones modelo=lewm accion=entrenar -->
- `--set`: receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)
- `--humo`: tres entrenamientos de 300 pasos (lr de la receta, por raíz y lineal) y elige
- `--humo-candidatos`: con --humo: candidatos a probar, p. ej. fuente,raiz,x3 (default: fuente,raiz,lineal,x3,x10)
- `--forzar`: entrenar: seguir aunque la corrida ya tenga last.pt o final.pt (train.py la reanuda desde last.pt)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli lewm entrenar --humo
python -m src.cli lewm entrenar
python -m src.cli lewm entrenar --presupuesto 10x --variante x10      # corrida 10x, de cero (etapa D)
python -m src.cli lewm entrenar --variante prueba --limite 20 --device cpu
```

### continuar

Retoma una corrida cortada. Con `--presupuesto` la receta (warmup más coseno) no se puede extender de forma equivalente: la CLI usa `--restart` (re-warmup corto y coseno nuevo) y la marca como no equivalente. Para la tabla, entrená de cero con el presupuesto nuevo.

<!-- auto:opciones modelo=lewm accion=continuar -->
- `--set`: receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)
- `--restart`: continuar con re-warmup y coseno nuevo (no equivalente)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli lewm continuar
python -m src.cli lewm continuar --presupuesto 2x --restart
```

### evaluar

Evaluación latente: error de predicción contra copiar el último latente por horizonte, probes (velocidad y giro) contra encoder sin entrenar y contra píxeles, sorpresa por segmento y recuperación de acciones con CEM. `--no-plan` salta el CEM, que es lo más caro.

<!-- auto:opciones modelo=lewm accion=evaluar -->
- `--dtype`: precisión de la inferencia
- `--temperatura`: temperatura de muestreo
- `--ctx-frames`: frames de contexto en la inferencia (default 20 = 1 s)
- `--no-plan`: evaluación latente sin el plan con CEM

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli lewm evaluar
python -m src.cli lewm evaluar --no-plan
python -m src.cli lewm evaluar --limite 50 --device cpu
```

### bench

Latencia de un paso del modelo (encoder más predictor), con batch 1.

<!-- auto:opciones modelo=lewm accion=bench -->
- `--dtype`: precisión de la inferencia
- `--temperatura`: temperatura de muestreo
- `--ctx-frames`: frames de contexto en la inferencia (default 20 = 1 s)
- `--int8`: cuantización int8 dinámica en CPU
- `--aleatorio`: modelo sin entrenar con la arquitectura de la corrida (latencia)
- `--cadena`: bench: latencia de la cadena de punta a punta; evaluar: calidad en línea y con la cadena (src.infer.quality)
- `--diagnostico`: con --cadena: el piso de ~20 ms a ritmo de cámara (bench_e2e --diagnostico)
- `--modo`: con --cadena: online o imagine (default: los dos)
- `--ritmo`: con --cadena: milisegundos entre frames (0 = bucle cerrado)
- `--tag`: nombre del dispositivo (radeon860m...) en los JSON, para no pisar los de otra máquina; con rocm es obligatorio en los benchs que escriben results/bench/

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli lewm bench
python -m src.cli lewm bench --aleatorio --device cpu --limite 20
```

### demo

Sin opciones, dibuja la curva de error y los latentes de un segmento. Con `--entrenar-decoder` entrena el decoder de píxeles (encoder congelado, MSE) para poder ver lo que el modelo "imagina". Lo que viene después de `--` pasa a `src.infer.demo_latente` (por ejemplo `--segment`, `--start`, `--mp4`).

<!-- auto:opciones modelo=lewm accion=demo -->
- `--dtype`: precisión de la inferencia
- `--temperatura`: temperatura de muestreo
- `--aleatorio`: modelo sin entrenar con la arquitectura de la corrida (latencia)
- `--entrenar-decoder`: entrena el decoder de píxeles (encoder congelado) en vez del demo

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli lewm demo
python -m src.cli lewm demo --entrenar-decoder
python -m src.cli lewm demo --entrenar-decoder --limite 50 --device cpu
```

### exportar

ONNX del encoder y del predictor (un paso), con la comparación de ONNX Runtime contra PyTorch (sale con error si difieren en más de 1e-3).

<!-- auto:opciones modelo=lewm accion=exportar -->
- `--dtype`: precisión de la inferencia
- `--aleatorio`: modelo sin entrenar con la arquitectura de la corrida (latencia)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli lewm exportar
```

### estado

<!-- auto:opciones modelo=lewm accion=estado -->
- `--vigilar`: estado: repite cada N segundos (default 10)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli lewm estado
```

## 5. Qué archivos escribe

| Acción | Archivos |
|---|---|
| `entrenar`, `continuar` | `checkpoints/<run>/{last.pt, final.pt, log.jsonl, status.json}`, `results/train_<run>.json`, entrada de `results/checkpoints.json`. |
| `evaluar` | `results/latent_<run>.json` (con las referencias al lado). |
| `bench` | `results/bench/<run>__<dtype>.json`. |
| `demo` | `results/demo/<run>_latente.png` y su `.json`; con `--entrenar-decoder`, `checkpoints/<run>/decoder.pt` y `results/train_decoder_<run>.json`. |
| `exportar` | `checkpoints/onnx/<run>_{encoder,predictor}.onnx` y `results/export/<run>_<parte>.json`. |

## 6. Problemas frecuentes

- **La loss baja pero el modelo colapsó:** SIGReg existe para evitarlo. El humo descarta los candidatos cuya dispersión de embeddings (`z_std` en `log.jsonl`) cae muy por debajo de la mediana. Mirá `results/humo_lewm.json`.
- **Memoria o BatchNorm:** no uses acumulación (rompe BatchNorm y SIGReg); bajá `--set receta.batch=64` y declará el desvío.
- **El CEM tarda mucho:** es el paso caro de `evaluar`; probá antes con `--no-plan` o con `--limite`.
- **`demo` sin decoder:** la figura de latentes anda sin decoder; para ver píxeles hay que correr antes `demo --entrenar-decoder`.
- **"la corrida ya existe", GPU ocupada:** ver [`../uso.md`](../uso.md).
- **`--temperatura` o `--pasos`:** no aplican, el modelo no genera tokens.

## 7. Notebook

[`notebooks/modelos/lewm.ipynb`](../../notebooks/modelos/lewm.ipynb): SIGReg y desvío del latente al entrenar, la sorpresa y el plan con CEM.
