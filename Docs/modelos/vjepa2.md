# vjepa2: preentrenamiento de V-JEPA 2

## 1. Qué es y su fuente

El preentrenamiento autosupervisado de V-JEPA 2 (arXiv 2506.09985; código `github.com/facebookresearch/vjepa2`): un encoder ViT con tubelets 2x16x16 y 3D-RoPE, un predictor que ve solo los tokens visibles y adivina las representaciones de los enmascarados, con máscaras multibloque (8 bloques cortos de escala 0,15 y 2 largos de 0,7), loss L1 contra un target EMA con stop-grad (momento 0,99925 constante) y la receta warmup, constante, enfriamiento lineal. Reemplaza al "JEPA" anterior sobre tokens, que nunca fue V-JEPA y se borró.

- **Datos:** clips de 16 frames a 4 fps (un frame de cada 5 de los 20 FPS), renderizados con el decoder de comma congelado y recortados en la GPU.
- **Tamaños:** `--tamaño t` (ViT-Tiny) o `s` (ViT-Small, el de la cola). El del paper es ViT-L: no entra en el presupuesto.
- **Es latente:** no hay CE ni FVD. Se evalúa con error latente contra copiar el último latente, probes contra un encoder sin entrenar y contra píxeles.
- **Lo usa [`vjepa2_ac`](vjepa2_ac.md):** su encoder (`target_encoder`) queda congelado allá.
- **En la cola:** `vjepa2_s` (`scripts/queue.toml`). La corrida 10x (`continuar --presupuesto 10x`) va al final de todo.

## 2. Fidelidad

<!-- auto:fidelidad kind=vjepa2 -->
| decisión | fuente / paper | aquí | categoría | fuente |
|---|---|---|---|---|
| Tamaño del encoder | ViT-L: 1024, 24 capas, 16 cabezas (300M) | tiny 192/12/3 o small 384/12/6 | presupuesto | paper Tab. 12; vision_transformer.py L334-359 (presets tiny/small del código oficial) |
| Entrada | 16 frames 256x256 a 4 fps -> 8x16x16 = 2048 tokens | 16 frames 128x256 (render del decoder de comma) -> 8x8x16 = 1024 tokens | datos | yaml L18-25 |
| Predictor | 12 capas, 384, 12 cabezas, MLP 1536, mask tokens en 0 | igual | fiel | yaml L77-85; paper Tab. 12; predictor.py L64-66 |
| Mask tokens | num_mask_tokens=2 por defecto; con una sola longitud de clip solo se usa el índice 0 | 1 mask token compartido por los dos conjuntos | fiel | predictor.py L42, L202-203; utils/wrappers.py L42 (mask_index = índice del grupo de fpc, no del conjunto) |
| Posiciones RoPE del predictor | decodifica ids con rejilla cuadrada (grid_size x grid_size) porque sus bloques se llaman sin H/W (predictor.py L240, modules.py L296-329) | se pasa la rejilla real (H=8, W=16), como ya hace su encoder | datos | modules.py L316-329; vision_transformer.py L178-181 |
| 3D-RoPE | expansión sin/cos con repeat (frecuencias duplicadas, bug conocido, compatible con los checkpoints) | oficial=True por defecto (igual); oficial=False aplica la corrección | fiel | modules.py L39-46 |
| Regularización del modelo | sin dropout, sin drop-path, atención SDPA | igual (no implementados) | fiel | yaml (no define drop_rate/drop_path_rate); vision_transformer.py L35-37 |
| Máscaras | A: 8 bloques escala 0.15; B: 2 bloques escala 0.7; aspecto [0.75, 1.5]; tiempo completo | igual sobre la rejilla 8x16 (bloques de menos tokens por la rejilla más chica) | fiel | yaml L41-67; multiseq_multiblock3d.py L145-269 |
| Normalización de entrada | media/std de ImageNet sobre [0, 1] | igual | fiel | app/vjepa/transforms.py L21, L48, L139-152 |
| Optimizador | AdamW β=(0.9, 0.999), eps 1e-8, sin grad-clip | igual | fiel | app/vjepa/utils.py L224-225, L243; train.py (sin clip); configs/train/vitl16/pretrain-256px-16f.yaml L92 |
| Weight decay | 0.04 constante, sin wd en bias ni en parámetros 1-D | igual (wd_on=matrices: solo ndim >= 2) | fiel | app/vjepa/utils.py L205-216 |
| LR | 1e-4 → 5.25e-4 en 12k pasos de 252k, constante, enfriamiento lineal a 1e-6 en los últimos 12k | las mismas fracciones (12k/252k = 0.0476) del total de pasos de la corrida; piso de 100 pasos de warmup | fiel | V-JEPA 2 (arXiv 2506.09985) Tab. 9; configs/train/vitl16/pretrain-256px-16f.yaml L91-97 |
| Batch | 3072 clips | 32 clips por paso, sin acumulación: con el presupuesto fijo, el batch del paper daría ~16 pasos; lr a confirmar en el humo | presupuesto | V-JEPA 2 (arXiv 2506.09985) Tab. 9 |
| Pasos / presupuesto | 252k pasos | el presupuesto común de 780k frames ≈ 1.5k pasos de 32 clips de 16 frames; con tan pocos pasos el EMA de 0.99925 (constante de tiempo ~1.3k pasos) deja al target cerca de la inicialización al principio | presupuesto | V-JEPA 2 (arXiv 2506.09985) Tab. 9 |
| Precisión | bf16 | bf16 (autocast del entrenador, solo CUDA) | fiel | configs/train/vitl16/pretrain-256px-16f.yaml L69 |
| EMA del target | 0.99925 constante | igual, actualizado con _foreach sobre los parámetros en el device (after_step) | fiel | configs/train/vitl16/pretrain-256px-16f.yaml L87-89; train.py L299-301 |
| Clip de entrada | 16 frames a 4 fps, recorte 256x256 | 16 frames = 1 de cada 5 de los 20 FPS (4 fps), frames de 128x256 renderizados del token con el decoder de comma | datos | configs/train/vitl16/pretrain-256px-16f.yaml L18-25 |
| Aumentos | random-resized-crop escala [0.3, 1], aspecto [0.75, 1.35] y flip horizontal | igual, en la GPU; el aspecto se mide respecto del aspecto 1:2 del frame (ver aug.py) | datos | configs/train/vitl16/pretrain-256px-16f.yaml L29-38; transforms.py L470-543 |
| Checkpointing de activaciones | use_activation_checkpointing: true | igual (cfg.grad_ckpt) | fiel | configs/train/vitl16/pretrain-256px-16f.yaml L82 |
| Encoder final | se evalúa el target_encoder (EMA) | igual: V-JEPA 2-AC carga target_encoder | fiel | configs/eval (checkpoint_key: target_encoder) |
| Datos de preentrenamiento | 22M videos + imágenes de internet | clips de conducción de comma (solo se compara la arquitectura, no los datos) | datos | V-JEPA 2 (arXiv 2506.09985) §2 |
<!-- /auto -->

## 3. Receta de entrenamiento

<!-- auto:receta kind=vjepa2 -->
| modelo | fuente | optimizador | betas | lr | wd | EMA | clip | precisión | batch x frames (frameskip) x acumulación | presupuesto (frames) | extras | pasos del paper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `vjepa2` | V-JEPA 2 (arXiv 2506.09985) Tab. 9; configs/train/vitl16/pretrain-256px-16f.yaml | adamw | (0.9, 0.999) | warmup: 0.0001->0.000525 (0.047619); constante: 0.000525 (resto); decay: 0.000525->1e-06 (0.047619) | 0.04 | no | None | bf16 | 32 x 16 (5) x 1 | 780000 | - | 252000 |

Valores forzados (campo: motivo, categoría):

- `vjepa2`: batch: 32 clips por paso en vez de 3072: con el presupuesto común de 780k frames da ~1.5k pasos (con 3072 serían ~16); lr a confirmar en el humo (presupuesto)
- `vjepa2`: presupuesto: ~1.5k pasos en vez de 252k; el warmup y el enfriamiento conservan sus fracciones (presupuesto)
- `vjepa2`: frames/frameskip: 16 frames a 4 fps = 1 de cada 5 de los 20 FPS (datos)
- `vjepa2`: encoder: ViT small (384, 12 capas) en vez de ViT-L (presupuesto)
<!-- /auto -->

Con el presupuesto común el entrenamiento tiene unos 1.500 pasos, no 252.000: el warmup y el enfriamiento conservan sus fracciones del total. El batch (32 clips) es una decisión de presupuesto; el humo elige el lr.

## 4. Acciones

<!-- auto:acciones modelo=vjepa2 -->
Evaluación: `latente`.

| acción | qué hace |
|---|---|
| `vjepa2 receta` | muestra la receta y la Config efectivas, con los overrides y los valores forzados marcados |
| `vjepa2 entrenar` | entrena de cero con la receta de la fuente (`--humo`: prueba corta de 300 pasos para elegir el lr) |
| `vjepa2 continuar` | retoma una corrida cortada o, con `--presupuesto`, la extiende |
| `vjepa2 evaluar` | mide el modelo (video: CE, rollouts, LPIPS y FVD; latentes: error, probes, sorpresa y CEM) |
| `vjepa2 bench` | mide la latencia (el modelo solo, o la cadena de punta a punta con `--cadena`) |
| `vjepa2 demo` | genera la demostración (mp4 con el video real y el predicho, o figura para los latentes) |
| `vjepa2 exportar` | exporta a ONNX |
| `vjepa2 estado` | porcentaje, paso, loss, lr, tramo y ETA de las corridas del modelo |
<!-- /auto -->

### receta

<!-- auto:opciones modelo=vjepa2 accion=receta -->
- `--set`: receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli vjepa2 receta --tamaño s
```

### entrenar

<!-- auto:opciones modelo=vjepa2 accion=entrenar -->
- `--set`: receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)
- `--humo`: tres entrenamientos de 300 pasos (lr de la receta, por raíz y lineal) y elige
- `--humo-candidatos`: con --humo: candidatos a probar, p. ej. fuente,raiz,x3 (default: fuente,raiz,lineal,x3,x10)
- `--forzar`: entrenar: seguir aunque la corrida ya tenga last.pt o final.pt (train.py la reanuda desde last.pt)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli vjepa2 entrenar --humo
python -m src.cli vjepa2 entrenar                       # corrida vjepa2_s
python -m src.cli vjepa2 entrenar --tamaño t --variante prueba --limite 20 --device cpu
```

### continuar

Retoma una corrida cortada o, con `--presupuesto`, la extiende. La receta es warmup, constante y decaimiento, así que la extensión es exacta: el entrenamiento guarda `pre_decay.pt` al entrar al enfriamiento, y `continuar --presupuesto` parte de ahí, sigue constante y decae al final. Da los mismos pesos que una corrida larga con el mismo warmup absoluto. La extensión es otra corrida (`vjepa2_s_x1`) con `parent` y receta.

<!-- auto:opciones modelo=vjepa2 accion=continuar -->
- `--set`: receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)
- `--restart`: continuar con re-warmup y coseno nuevo (no equivalente)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli vjepa2 continuar --run vjepa2_s
python -m src.cli vjepa2 continuar --run vjepa2_s --presupuesto 10x
```

### evaluar

Evaluación latente con las referencias al lado (copiar el último latente, encoder sin entrenar, píxeles). El plan con CEM es para el modelo con acciones; con `--no-plan` se lo omite.

<!-- auto:opciones modelo=vjepa2 accion=evaluar -->
- `--dtype`: precisión de la inferencia
- `--temperatura`: temperatura de muestreo
- `--ctx-frames`: frames de contexto en la inferencia (default 20 = 1 s)
- `--no-plan`: evaluación latente sin el plan con CEM

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli vjepa2 evaluar
python -m src.cli vjepa2 evaluar --limite 50 --device cpu
```

### bench

Latencia del encoder (un clip) con batch 1.

<!-- auto:opciones modelo=vjepa2 accion=bench -->
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
python -m src.cli vjepa2 bench
python -m src.cli vjepa2 bench --aleatorio --device cpu --limite 20
```

### demo

Muestra las máscaras multibloque sobre un clip y, si hay decoder, la decodificación. `--entrenar-decoder` entrena el decoder de píxeles (encoder congelado, MSE), solo para figuras.

<!-- auto:opciones modelo=vjepa2 accion=demo -->
- `--dtype`: precisión de la inferencia
- `--temperatura`: temperatura de muestreo
- `--aleatorio`: modelo sin entrenar con la arquitectura de la corrida (latencia)
- `--entrenar-decoder`: entrena el decoder de píxeles (encoder congelado) en vez del demo

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli vjepa2 demo
python -m src.cli vjepa2 demo --entrenar-decoder
```

### exportar

ONNX del encoder, comparado contra PyTorch.

<!-- auto:opciones modelo=vjepa2 accion=exportar -->
- `--dtype`: precisión de la inferencia

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli vjepa2 exportar
```

### estado

<!-- auto:opciones modelo=vjepa2 accion=estado -->
- `--vigilar`: estado: repite cada N segundos (default 10)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli vjepa2 estado
```

## 5. Qué archivos escribe

| Acción | Archivos |
|---|---|
| `entrenar`, `continuar` | `checkpoints/<run>/{last.pt, pre_decay.pt, final.pt, log.jsonl, status.json}`, `results/train_<run>.json`, entrada de `results/checkpoints.json`. |
| `evaluar` | `results/latent_<run>.json`. |
| `bench` | `results/bench/<run>__<dtype>.json`. |
| `demo` | `results/demo/<run>_mascaras.png`; con `--entrenar-decoder`, `checkpoints/<run>_viz/decoder.pt`. |
| `exportar` | `checkpoints/onnx/<run>_encoder.onnx` y `results/export/<run>_encoder.json`. |

## 6. Problemas frecuentes

- **Colapso:** la loss L1 puede bajar porque el target colapsó. El humo mira la dispersión del target y descarta los candidatos colapsados; mirá `target_std` en `log.jsonl`.
- **El EMA con pocos pasos:** con ~1.500 pasos el momento 0,99925 hace que el target se mueva muy poco; es una consecuencia declarada del presupuesto, no un bug.
- **`continuar --presupuesto` falla:** necesita `pre_decay.pt` (o `last.pt`) de la corrida; las corridas cortadas antes del enfriamiento no lo tienen.
- **Memoria:** la activación se recomputa (`grad_ckpt`); si igual pagina, bajá `--set receta.batch=16` y declará el desvío.
- **Banco de máscaras:** las máscaras se sortean por bancos; no es un problema, pero explica una sincronización por banco en el perfil.
- **"la corrida ya existe", árbol sucio, GPU ocupada:** ver [`../uso.md`](../uso.md).

## 7. Notebook

[`notebooks/modelos/vjepa2.ipynb`](../../notebooks/modelos/vjepa2.ipynb): las máscaras multibloque sobre un clip y los probes contra un encoder sin entrenar.
