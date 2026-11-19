# convlstm_px: ConvLSTM de Shi sobre píxeles

## 1. Qué es y su fuente

El ConvLSTM de Shi et al. 2015 (arXiv 1506.04214): un encoder-forecaster de ConvLSTM con peepholes, kernel 5x5 y tres capas 128/64/64, que **ve píxeles RGB y predice píxeles** (no tokens, no hay cabeza de cross-entropy). Reemplaza al ConvLSTM sobre tokens que tenía el repo, que no era el de Shi: ese se borró entero.

- **Entrada y salida:** 10 frames de entrada y 10 de salida a 128x256, en parches de 4x4 (32x64x48 canales). Para llegar a 3 s se realimenta la predicción (ventana deslizante); eso está declarado en la fidelidad.
- **Datos:** el dataset de comma es de tokens, así que los frames se renderizan al vuelo con el decoder de comma congelado, en la GPU.
- **Inferencia:** no usa el codec de comma para predecir; es determinista, sin temperatura ni CE. Entra en la tabla de video con LPIPS y FVD.
- **Evaluación:** `video`, sobre píxeles.
- **En la cola:** `convlstm_px` y las semillas 1 y 2 (`scripts/queue.toml`).

## 2. Fidelidad

<!-- auto:fidelidad kind=convlstm_px -->
| decisión | fuente / paper | aquí | categoría | fuente |
|---|---|---|---|---|
| Celda | ConvLSTM con peepholes: W_ci∘C_{t-1}, W_cf∘C_{t-1}, W_co∘C_t (Hadamard); transiciones entrada-estado y estado-estado convolucionales | igual; W_x*X y W_h*H se calculan con una sola conv sobre concat([X, H]) (matemáticamente idéntico a dos convs, con un solo sesgo) | fiel | Shi et al. 2015 (arXiv 1506.04214) §3.1, Eq. 3 |
| Peepholes | el paper no dice su forma | un vector por canal (broadcast en el espacio). Deducido: con parches 16x16x16, capas 128/64/64 y kernels 5x5 esta forma reproduce exactamente los 7 585 296 parámetros de la Tab. 1 (test_param_count_paper) | fiel | Shi et al. 2015 (arXiv 1506.04214) Tab. 1; deducción propia del conteo |
| Estructura | encoder y forecaster apilados; el forecaster arranca del último estado del encoder; salidas de todas las capas concatenadas -> conv 1x1 | igual; el forecaster conserva el orden de capas del encoder (el conteo de parámetros de la Tab. 1 solo cuadra así) | fiel | Shi et al. 2015 (arXiv 1506.04214) §3.1, Fig. 3 |
| Entrada del forecaster | Fig. 3: la red de pronóstico no recibe frames | la capa 1 no tiene entrada (sin pesos de entrada, como exige el conteo de la Tab. 1); no se realimenta la predicción dentro de los 10 pasos | fiel | Shi et al. 2015 (arXiv 1506.04214) Fig. 3, Tab. 1 |
| Capas y canales | 3 capas 128/64/64, kernels 5x5 (mejor de la Tab. 1: 3670,85 vs 3733,56 con 2x128 y 3887,94 con 1x256) | 128/64/64, kernel 5x5, padding 'same' (el paper no dice el borde) | fiel | Shi et al. 2015 (arXiv 1506.04214) §4.1, Tab. 1 (padding: propio) |
| Parches de entrada | parches 4x4: 64x64x1 -> 16x16x16 | parches 4x4: 128x256x3 -> 32x64x48 (pixel_unshuffle; la salida usa pixel_shuffle) | fiel | Shi et al. 2015 (arXiv 1506.04214) §4.1 |
| Imágenes | Moving-MNIST 64x64 en escala de grises, dígitos que rebotan | video de conducción RGB 128x256 (reconstrucción del VQ-VAE de comma, renderizada con su decoder congelado) | datos | commaVQ; los tokens son lo único que hay en el dataset |
| Frames de entrada y salida | 10 -> 10 | 10 -> 10 a 20 FPS (ventana de 20 frames) | fiel | Shi et al. 2015 (arXiv 1506.04214) §4.1 (frame rate: datos, comma graba a 20 FPS) |
| Horizonte de 3 s (60 frames) | no existe: el forecaster pronostica 10 frames | se pronostican 10, y esos 10 pasan a ser la ventana de entrada del siguiente bloque de 10 (ventana deslizante, estado del encoder desde cero cada vez, como se entrenó). La predicción se realimenta como probabilidades en [0, 1] sin cuantizar | propio | el paper no hace rollouts largos |
| Estado en línea (stream) | el paper no habla de inferencia en línea | el stream guarda la ventana de los últimos 10 frames y re-codifica los 10 por cada predicción (igual que en entrenamiento); no arrastra el estado del encoder frame a frame | propio | fidelidad a lo entrenado |
| Loss | entropía cruzada binaria por píxel (suma sobre la secuencia), valores en [0, 1] | sigmoide + BCE con logits, promedio sobre píxeles, canales y frames (la suma solo cambia la escala, que RMSProp normaliza; el clip sí la ve) | fiel | Shi et al. 2015 (arXiv 1506.04214) §4.1 (media en vez de suma: propio) |
| Objetivos | píxeles binarios/intensidades en [0, 1] | RGB uint8 / 255 (objetivos continuos en [0, 1]) | datos | Shi et al. 2015 (arXiv 1506.04214) §4.1 |
| Inicialización | no publicada | Conv2d por defecto de PyTorch (Kaiming uniforme), sesgos por defecto, peepholes en 0 | propio | el paper no la publica |
| Precisión | no publicada | fp32 (sin AMP) | propio | el paper no la publica |
| Optimizador | RMSProp, lr 1e-3, decay 0,9 | RMSprop(lr=1e-3, alpha=0.9), lr constante, sin momentum ni weight decay | fiel | Shi et al. 2015 (arXiv 1506.04214) §4.1 (eps 1e-8 de PyTorch: propio) |
| Batch, clip, épocas, paciencia | no publicados | batch 4 x 20 frames, clip de norma 1.0, pasos por presupuesto | propio | elegidos: el batch por memoria (12 GB) y los pasos que da; el clip porque BPTT sobre 20 pasos con RMSProp puede explotar |
| Early stopping | parar con el monitor de validación | se entrena el presupuesto completo (780k frames) y se evalúa el monitor (BCE en las ventanas del holdout) cada eval_every; train.py guarda el último estado, no el mejor | presupuesto | Shi et al. 2015 (arXiv 1506.04214) §4.1; src/train.py |
| Presupuesto | no publicado (paper: 10 000 secuencias, hasta converger) | 780k frames vistos (frames renderizados) | presupuesto | presupuesto común de la comparación |
<!-- /auto -->

## 3. Receta de entrenamiento

<!-- auto:receta kind=convlstm_px -->
| modelo | fuente | optimizador | betas | lr | wd | EMA | clip | precisión | batch x frames (frameskip) x acumulación | presupuesto (frames) | extras | pasos del paper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `convlstm_px` | Shi et al. 2015 (arXiv 1506.04214) §4.1 | rmsprop | (0.9, 0.95) | constante: 0.001 (resto) | 0 (todos) | no | 1.0 | fp32 | 4 x 20 (1) x 1 | 780000 | - | - |

Valores forzados (campo: motivo, categoría):

- `convlstm_px`: batch 4, clip 1.0, eps 1e-8: el paper no los publica; batch por la memoria de 12 GB (propio)
- `convlstm_px`: batch_fuente 4: Shi 2015 no publica su batch: sin batch de referencia, el lr de la fuente no se escala (propio)
- `convlstm_px`: early stopping: se entrena el presupuesto completo; train.py guarda el último estado (presupuesto)
- `convlstm_px`: RGB 128x256, 20 FPS, parches 4x4 -> 32x64x48: el dato de comma obliga (datos)
- `convlstm_px`: fp32: el paper no publica la precisión (propio)
<!-- /auto -->

El batch no está en el paper: 4 por la memoria de 12 GB. El lr es constante, así que continuar un entrenamiento es exacto: sigue desde `last.pt`.

## 4. Acciones

<!-- auto:acciones modelo=convlstm_px -->
Evaluación: `video`.

| acción | qué hace |
|---|---|
| `convlstm_px receta` | muestra la receta y la Config efectivas, con los overrides y los valores forzados marcados |
| `convlstm_px entrenar` | entrena de cero con la receta de la fuente (`--humo`: prueba corta de 300 pasos para elegir el lr) |
| `convlstm_px continuar` | retoma una corrida cortada o, con `--presupuesto`, la extiende |
| `convlstm_px evaluar` | mide el modelo (video: CE, rollouts, LPIPS y FVD; latentes: error, probes, sorpresa y CEM) |
| `convlstm_px bench` | mide la latencia (el modelo solo, o la cadena de punta a punta con `--cadena`) |
| `convlstm_px demo` | genera la demostración (mp4 con el video real y el predicho, o figura para los latentes) |
| `convlstm_px estado` | porcentaje, paso, loss, lr, tramo y ETA de las corridas del modelo |
<!-- /auto -->

### receta

<!-- auto:opciones modelo=convlstm_px accion=receta -->
- `--set`: receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli convlstm_px receta
```

### entrenar

<!-- auto:opciones modelo=convlstm_px accion=entrenar -->
- `--set`: receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)
- `--humo`: tres entrenamientos de 300 pasos (lr de la receta, por raíz y lineal) y elige
- `--humo-candidatos`: con --humo: candidatos a probar, p. ej. fuente,raiz,x3 (default: fuente,raiz,lineal,x3,x10)
- `--forzar`: entrenar: seguir aunque la corrida ya tenga last.pt o final.pt (train.py la reanuda desde last.pt)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli convlstm_px entrenar --humo
python -m src.cli convlstm_px entrenar
python -m src.cli convlstm_px entrenar --semilla 1       # corrida convlstm_px_s1
python -m src.cli convlstm_px entrenar --variante prueba --limite 20 --device cpu
```

### continuar

Con `--presupuesto` la extiende: como el lr es constante, entrenar N y extender a 2N da lo mismo que entrenar 2N de una vez. La extensión es una corrida nueva (`<run>_x<k>`).

<!-- auto:opciones modelo=convlstm_px accion=continuar -->
- `--set`: receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)
- `--restart`: continuar con re-warmup y coseno nuevo (no equivalente)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli convlstm_px continuar
python -m src.cli convlstm_px continuar --presupuesto 10x
```

### evaluar

No hay CE: mide LPIPS y FVD de los rollouts de 400 clips (1 s de contexto y 3 s generados), con intervalos por segmento y piso de FVD. Corre en fp32 (el modelo no usa AMP).

<!-- auto:opciones modelo=convlstm_px accion=evaluar -->
- `--dtype`: precisión de la inferencia
- `--temperatura`: temperatura de muestreo
- `--ctx-frames`: frames de contexto en la inferencia (default 20 = 1 s)
- `--barrer-temperatura`: barrido de temperatura en el holdout
- `--cadena`: bench: latencia de la cadena de punta a punta; evaluar: calidad en línea y con la cadena (src.infer.quality)
- `--desde-cache`: evaluar: no rehace los rollouts que ya existen; recalcula las métricas desde results/cache (nombres de los resultados viejos, T = 1)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli convlstm_px evaluar
python -m src.cli convlstm_px evaluar --semilla 1
python -m src.cli convlstm_px evaluar --cadena
python -m src.cli convlstm_px evaluar --limite 5 --device cpu
```

### bench

El modelo solo (ms por bloque de 10 frames, fp32), o la cadena completa con `--cadena`; `--ritmo 50` la mide a 20 FPS.

<!-- auto:opciones modelo=convlstm_px accion=bench -->
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
python -m src.cli convlstm_px bench
python -m src.cli convlstm_px bench --cadena
python -m src.cli convlstm_px bench --cadena --ritmo 50
python -m src.cli convlstm_px bench --aleatorio --device cpu --limite 20
```

### demo

<!-- auto:opciones modelo=convlstm_px accion=demo -->
- `--dtype`: precisión de la inferencia
- `--temperatura`: temperatura de muestreo
- `--aleatorio`: modelo sin entrenar con la arquitectura de la corrida (latencia)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli convlstm_px demo
```

### estado

<!-- auto:opciones modelo=convlstm_px accion=estado -->
- `--vigilar`: estado: repite cada N segundos (default 10)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli convlstm_px estado
```

## 5. Qué archivos escribe

| Acción | Archivos |
|---|---|
| `entrenar`, `continuar` | `checkpoints/<run>/{last.pt, final.pt, log.jsonl, status.json}`, `results/train_<run>.json`, entrada de `results/checkpoints.json`. |
| `evaluar` | `results/rollouts/<run>.npy` y `.json`, `results/video_<run>.json` (LPIPS, FVD, piso). No escribe `eval_<run>.json`: no hay CE. |
| `bench` | `results/bench/<run>__fp32.json`; con `--cadena`, `results/infer/latencia_<run>__<dispositivo>.json`. |
| `demo` | `results/demo/*.mp4`. |

## 6. Problemas frecuentes

- **Memoria:** BPTT sobre 20 pasos con 3 capas ConvLSTM es lo más pesado en memoria del repo. Con el batch de 4 entra en 12 GB; no lances otro trabajo de GPU en paralelo.
- **La loss explota:** RMSProp con BPTT puede inestabilizarse; la receta recorta el gradiente a norma 1,0 (decisión propia declarada). Si pasa igual, mirá `log.jsonl` y reportalo en vez de cambiar el lr a mano: el lr se cambia con el humo y un commit.
- **El FVD a 3 s depende del realimentado:** la predicción se vuelve a entrada cada 10 frames; es la forma declarada de llegar a 3 s. No es comparable con un horizonte de un solo bloque.
- **"la corrida ya existe", árbol sucio, GPU ocupada:** ver [`../uso.md`](../uso.md).
- **No hay `--temperatura` ni `--pasos`:** el modelo es determinista y no decodifica por pasos; la CLI los rechaza.

## 7. Notebook

[`notebooks/modelos/convlstm_px.ipynb`](../../notebooks/modelos/convlstm_px.ipynb): la nitidez por horizonte y la cadena sin codec.
