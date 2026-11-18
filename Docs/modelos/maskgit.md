# maskgit: MaskGIT propio y sus ablaciones

## 1. Qué es y su fuente

Una **ablación propia**, no una réplica de MaskGIT: predice los 128 tokens de un frame en paralelo con S pasadas, con el formato de secuencia del GPT (BOS más 128 códigos por frame) y atención por bloques (causal entre frames, bidireccional dentro de cada frame). Solo la decodificación sigue a MaskGIT (arXiv 2202.04200): schedule coseno de máscara y elección de los tokens más confiables con ruido. El entrenamiento (posición factorizada o absoluta, `p_clean`, tasa de máscara por frame) es del repo y está declarado como `propio`.

- **El MaskGIT fiel de video es [`genie`](genie.md)** (D7b). Este modelo queda implementado y entrenable, fuera de la cola, para las ablaciones de los trabajos opcionales de `scripts/queue.toml`: posición absoluta, máscara de Genie, sin `p_clean`.
- **Fuente de la decodificación:** MaskGIT, `libml/parallel_decode.py` y `libml/mask_schedule.py`.
- **Evaluación:** `video`.

## 2. Fidelidad

<!-- auto:fidelidad kind=maskgit -->
| decisión | fuente / paper | aquí | categoría | fuente |
|---|---|---|---|---|
| Calendario de máscara en inferencia | coseno: quedan floor(128·cos(π/2·s/S)) enmascarados | igual | fiel | MaskGIT (arXiv 2202.04200; github.com/google-research/maskgit) §3.2; libml/mask_schedule.py |
| Decodificación por confianza | se muestrea cada celda, se fijan las más confiables con ruido Gumbel sobre log p que baja a 0 | igual (choice_temp 1.0 por defecto, el de MaskGIT es 4.5) | fiel | MaskGIT (arXiv 2202.04200; github.com/google-research/maskgit) libml/parallel_decode.py L29-53; choice_temp: propio |
| Pasos de decodificación | 8 en el paper | 8 por defecto (default_steps) | fiel | MaskGIT (arXiv 2202.04200; github.com/google-research/maskgit) §3.2 |
| Loss | cross-entropy solo en los tokens enmascarados | igual, sin label smoothing (MaskGIT usa 0.1) | fiel | MaskGIT (arXiv 2202.04200; github.com/google-research/maskgit) Eq. 1; config L41-42 |
| Máscara en entrenamiento | una tasa coseno por muestra de imagen (r ~ U[0,1), cos(π/2·r)) | una tasa por frame (coseno o U[0.5,1]), independiente entre frames | propio | MaskGIT genera una imagen; el video por frames no está en el paper |
| p_clean | no existe | probabilidad de dejar un frame limpio al entrenar, para que el contexto se parezca al de inferencia | propio | decisión del repo |
| Atención | bidireccional sobre la imagen | bloques: causal entre frames, bidireccional dentro del frame | propio | extensión a video (la de Genie es el ST-transformer de genie.py) |
| Posición | aprendida | pos='abs' (por defecto) o 'frame' (frame + celda, el de las corridas propias) | propio | MaskGIT (arXiv 2202.04200; github.com/google-research/maskgit) config L55-59; variante factorizada: repo |
| Arquitectura y tamaño | 24 capas, 768, MLP 3072 (bidireccional, sobre 256 tokens de imagen) | S = 6 x 384, M = 8 x 512, L = 12 x 768 (backbone propio compartido con el GPT) | presupuesto | MaskGIT (arXiv 2202.04200; github.com/google-research/maskgit) config L55-59 |
| Receta de entrenamiento | Adam (0.9, 0.96), lr 1e-4 constante tras warmup 5k, wd 4.5e-2, batch 256, dropout 0.1 (código de entrenamiento no publicado) | receta 'maskgit_propia': AdamW (0.9, 0.95), warmup + coseno, wd 0.1, clip 1, fp16, sin dropout | propio | receta heredada del repo; MaskGIT no publica el código de entrenamiento |
| Datos | ImageNet 256x256, tokens VQGAN 16x16 | tokens de comma, 128 por frame | datos | commaVQ |
<!-- /auto -->

## 3. Receta de entrenamiento

La receta es propia porque MaskGIT no publica su código de entrenamiento: queda marcada como tal.

<!-- auto:receta kind=maskgit -->
| modelo | fuente | optimizador | betas | lr | wd | EMA | clip | precisión | batch x frames (frameskip) x acumulación | presupuesto (frames) | extras | pasos del paper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `maskgit_s`, `maskgit_m`, `maskgit_l` | propia (ablación del repo; MaskGIT no publica el entrenamiento) | adamw | (0.9, 0.95) | warmup: 0->0.001 (200p); cosine: coseno 0.001->0.0001 (resto) | 0.1 | no | 1.0 | fp16 | 16 x 20 (1) x 1 | 780000 | - | - |

Valores forzados (campo: motivo, categoría):

- `maskgit_s`: toda la receta: ablación propia: no hay receta publicada que seguir (propio)
<!-- /auto -->

## 4. Acciones

<!-- auto:acciones modelo=maskgit -->
Evaluación: `video`.

| acción | qué hace |
|---|---|
| `maskgit receta` | muestra la receta y la Config efectivas, con los overrides y los valores forzados marcados |
| `maskgit entrenar` | entrena de cero con la receta de la fuente (`--humo`: prueba corta de 300 pasos para elegir el lr) |
| `maskgit continuar` | retoma una corrida cortada o, con `--presupuesto`, la extiende |
| `maskgit evaluar` | mide el modelo (video: CE, rollouts, LPIPS y FVD; latentes: error, probes, sorpresa y CEM) |
| `maskgit bench` | mide la latencia (el modelo solo, o la cadena de punta a punta con `--cadena`) |
| `maskgit demo` | genera la demostración (mp4 con el video real y el predicho, o figura para los latentes) |
| `maskgit exportar` | exporta a ONNX |
| `maskgit estado` | porcentaje, paso, loss, lr, tramo y ETA de las corridas del modelo |
<!-- /auto -->

### receta

<!-- auto:opciones modelo=maskgit accion=receta -->
- `--set`: receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli maskgit receta --tamaño s
```

### entrenar

Las variantes de la ablación se arman con `--set config.<campo>=<valor>` y una `--variante` que las nombra. Cada `--set` queda como desvío en la bitácora. Los campos de la Config son `pos` (`abs` o `frame`), `p_clean`, `mask_schedule` (`cosine` o `uniform`), `mask_min`, `mask_max`, `default_steps` y `choice_temp`.

<!-- auto:opciones modelo=maskgit accion=entrenar -->
- `--set`: receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)
- `--humo`: tres entrenamientos de 300 pasos (lr de la receta, por raíz y lineal) y elige
- `--humo-candidatos`: con --humo: candidatos a probar, p. ej. fuente,raiz,x3 (default: fuente,raiz,lineal,x3,x10)
- `--forzar`: entrenar: seguir aunque la corrida ya tenga last.pt o final.pt (train.py la reanuda desde last.pt)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
# ablación propia: posición factorizada y p_clean 0,5
python -m src.cli maskgit entrenar --tamaño s --set config.pos=frame --set config.p_clean=0.5
# posición absoluta
python -m src.cli maskgit entrenar --tamaño s --variante abs
# máscara de Genie en el modelo propio
python -m src.cli maskgit entrenar --tamaño s --variante genie --set config.pos=frame --set config.p_clean=0.5 --set config.mask_schedule=uniform
# sin p_clean
python -m src.cli maskgit entrenar --tamaño s --variante nopc --set config.pos=frame
```

### continuar

<!-- auto:opciones modelo=maskgit accion=continuar -->
- `--set`: receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)
- `--restart`: continuar con re-warmup y coseno nuevo (no equivalente)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli maskgit continuar --tamaño s
python -m src.cli maskgit continuar --tamaño s --variante abs --presupuesto 2x
```

### evaluar

Igual que en `genie`: CE (con 1 paso, el frame entero enmascarado), rollouts, LPIPS, FVD y piso; `--pasos` barre el número de pasadas. Otra variante o configuración es otra corrida y otra fila de la tabla de resultados.

<!-- auto:opciones modelo=maskgit accion=evaluar -->
- `--dtype`: precisión de la inferencia
- `--temperatura`: temperatura de muestreo
- `--pasos`: pasos de decodificación, uno o varios separados por coma (maskgit, genie)
- `--ctx-frames`: frames de contexto en la inferencia (default 20 = 1 s)
- `--barrer-temperatura`: barrido de temperatura en el holdout
- `--cadena`: bench: latencia de la cadena de punta a punta; evaluar: calidad en línea y con la cadena (src.infer.quality)
- `--desde-cache`: evaluar: no rehace los rollouts que ya existen; recalcula las métricas desde results/cache (nombres de los resultados viejos, T = 1)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli maskgit evaluar --tamaño s --pasos 1,2,4,8,12
python -m src.cli maskgit evaluar --tamaño s --variante abs --pasos 8
python -m src.cli maskgit evaluar --tamaño s --pasos 4 --limite 5 --device cpu
```

### bench

<!-- auto:opciones modelo=maskgit accion=bench -->
- `--dtype`: precisión de la inferencia
- `--temperatura`: temperatura de muestreo
- `--pasos`: pasos de decodificación, uno o varios separados por coma (maskgit, genie)
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
python -m src.cli maskgit bench --tamaño s --pasos 1,2,4,8
python -m src.cli maskgit bench --tamaño s --pasos 4 --cadena
python -m src.cli maskgit bench --tamaño s --pasos 4 --aleatorio --device cpu --limite 20
```

### demo

<!-- auto:opciones modelo=maskgit accion=demo -->
- `--dtype`: precisión de la inferencia
- `--temperatura`: temperatura de muestreo
- `--pasos`: pasos de decodificación, uno o varios separados por coma (maskgit, genie)
- `--aleatorio`: modelo sin entrenar con la arquitectura de la corrida (latencia)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli maskgit demo --tamaño s --pasos 4
```

### exportar

Exporta una pasada de forma fija a ONNX (`--dtype fp16` para el modelo en media precisión) y verifica el resultado contra PyTorch. El muestreo con S pasos lo hace quien llama, repitiendo la pasada.

<!-- auto:opciones modelo=maskgit accion=exportar -->
- `--dtype`: precisión de la inferencia

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli maskgit exportar --tamaño s
python -m src.cli maskgit exportar --tamaño s --dtype fp16
```

### estado

<!-- auto:opciones modelo=maskgit accion=estado -->
- `--vigilar`: estado: repite cada N segundos (default 10)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli maskgit estado
```

## 5. Qué archivos escribe

| Acción | Archivos |
|---|---|
| `entrenar`, `continuar` | `checkpoints/<run>/{last.pt, final.pt, log.jsonl, status.json}`, `results/train_<run>.json`, entrada de `results/checkpoints.json`. |
| `evaluar` | `results/eval_<run>.json`, `results/rollouts/<run>_p<pasos>.npy`, `results/video_<run>_p<pasos>.json`. |
| `bench` | `results/bench/<run>_p<pasos>__<dtype>[_graphs].json`; con `--cadena`, `results/infer/latencia_<run>__<dispositivo>.json`. |
| `exportar` | `checkpoints/onnx/<run>_k<K>.onnx`. |
| `demo` | `results/demo/*.mp4`. |

## 6. Problemas frecuentes

- **El modelo "fiel" es `genie`,** no este. No los mezcles en una misma fila de resultados: son modelos distintos.
- **Los rollouts se rompen:** sin `p_clean` el contexto del entrenamiento (todo enmascarado) no se parece al de inferencia. Es parte de la ablación; la corrida `nopc` existe para mostrarlo.
- **"la corrida ya existe":** cambiá `--variante` (cada variante de la ablación lleva la suya) o usá `continuar`.
- **`--set config.pos=frame` no aplica a una corrida ya entrenada:** la Config sale del checkpoint; entrená una variante nueva.
- Ver también los problemas frecuentes de [`genie.md`](genie.md): árbol sucio, GPU ocupada, memoria en 12 GB.

## 7. Notebook

[`notebooks/modelos/maskgit.ipynb`](../../notebooks/modelos/maskgit.ipynb): el barrido de pasos y de `choice_temp`, y cómo se destapa un frame.
