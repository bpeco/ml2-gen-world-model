# genie: dinámica de Genie con decodificación MaskGIT

## 1. Qué es y su fuente

La dinámica de Genie (Bruce et al. 2024, arXiv 2402.15391) sin el modelo de acciones latentes: es el "MaskGIT de video" de la comparación, la decisión del usuario del 17/11 (D7b). Un ST-transformer (atención espacial sobre los 128 tokens de cada frame, atención temporal causal sobre la misma celda y un FFW por bloque), entrenado con una máscara Bernoulli de tasa ~ U[0,5; 1] sobre los frames 2 a T (el primero sin máscara) y con la cross-entropy solo en los enmascarados. Genera un frame por vez con pasos de decodificación MaskGIT (25 por defecto, temperatura 2).

- **Fuente:** Genie, secciones 2.3 y 3, tablas 9, 10 y 17, apéndice F.1; la decodificación sigue a MaskGIT (arXiv 2202.04200).
- **Es el modelo fiel:** el `maskgit` propio (con `p_clean`, posición factorizada) es una ablación y vive aparte, en [`maskgit.md`](maskgit.md).
- **Pasos de decodificación:** `--pasos 1,2,4,8,12,25` mide la calidad y la latencia contra el número de pasadas por frame.
- **En la cola:** `genie_s`, sus pasos de decodificación y las semillas 1 y 2 evaluadas a 8 pasos (`scripts/queue.toml`). `genie_m` es el trabajo opcional 19 (se corrió: entrenar, evaluar a 1 a 25 pasos y latencia); sus números están en [`../resultados.md`](../resultados.md#1-modelos-de-video-modelo--performance).
- **Evaluación:** `video`.

## 2. Fidelidad

<!-- auto:fidelidad kind=genie -->
| decisión | fuente / paper | aquí | categoría | fuente |
|---|---|---|---|---|
| Bloque | atención espacial sobre los H×W tokens de cada instante, atención temporal causal sobre los T tokens de la misma posición, un FFW por bloque (solo después de la temporal) | igual | fiel | Genie (arXiv 2402.15391) §2 'ST-transformer', Fig. 4 |
| Máscara causal en el tiempo | dinámica causal en el tiempo, estilo MaskGIT | igual | fiel | Genie (arXiv 2402.15391) §2.3 |
| QK-norm | se usa en la dinámica (no dice qué norma) | LayerNorm sobre head_dim en q y k | fiel | Genie (arXiv 2402.15391) §3; tipo de norma: propio (como ViT-22B) |
| Posición de las normas | no publicada | pre-LN antes de cada subcapa, residuales, LayerNorm final | propio | Genie no lo publica; es la elección estándar (Jafar, no oficial, también) |
| Posición | no publicada | embedding aprendido espacial (128 celdas) + temporal (frames), sumados al token | propio | Genie no lo publica |
| Token de máscara | los tokens enmascarados se reemplazan por un token de máscara | vector aprendido | fiel | Genie (arXiv 2402.15391) §2.3 (aprendido: propio, igual que Jafar) |
| Acciones | embeddings aditivos de acciones latentes | ninguna (action-free) | datos | Genie (arXiv 2402.15391) §2.3; nuestros world models no ven acciones |
| Frames por secuencia (T) | 16 | 16 | fiel | Genie (arXiv 2402.15391) §3 |
| Frame rate | 10 fps | 20 fps | datos | Genie (arXiv 2402.15391) §3; la tarea es video comma a 20 fps |
| Tokens por frame / vocabulario | tokenizador propio (1024 códigos, 16×16) | 128 tokens (8×16), 1024 códigos de comma | datos | commaVQ |
| Máscara en entrenamiento | Bernoulli por token con tasa ~ U[0.5, 1], frame 1 sin máscara | igual, sobre los frames 2..T (leemos 'z_{2:T-1}' como 2..T: si no, el último frame nunca se predeciría) | fiel | Genie (arXiv 2402.15391) §2.3 |
| Granularidad de la tasa | no dice si es por secuencia o por frame | una tasa por secuencia (muestra) | propio | Genie no lo dice |
| Loss | cross-entropy sobre los tokens enmascarados | igual, sin label smoothing ni dropout | fiel | Genie (arXiv 2402.15391) §2.3; MaskGIT (arXiv 2202.04200) Eq. 1 (regularización: propio, no publicada para Genie) |
| Pasos de decodificación | 25 pasos MaskGIT por frame | 25 por defecto (parámetro) | fiel | Genie (arXiv 2402.15391) §3, Tab. 17 |
| Temperatura de muestreo | 2 (CoinRun: 1.0) | 2.0 por defecto (parámetro) | fiel | Genie (arXiv 2402.15391) §3, Tab. 17 |
| Calendario de máscara en inferencia | coseno: quedan floor(128·cos(π/2·s/S)) enmascarados | igual | fiel | MaskGIT (arXiv 2202.04200) §3.2 |
| Choice temperature | Genie no la publica; MaskGIT: 4.5·(1 − (t+1)/T), ruido Gumbel sobre log p | 4.5 anillada linealmente a 0 (parámetro); la confianza es log p del token muestreado con softmax a temperatura 1, como MaskGIT | propio | MaskGIT (arXiv 2202.04200) (código oficial); combinación con la temperatura 2 de Genie: propio |
| Contexto al generar | los frames previos más el nuevo todo enmascarado | los últimos T−1 = 15 frames reales/generados + el frame nuevo todo MASK | fiel | Genie (arXiv 2402.15391) §2.3 (T = 16); recorte de la ventana: propio |
| Tamaño | el más chico es 41M: 18 capas, 8 cabezas, d_model 512, k/q 64 | S = 6 × 384 (6 cabezas), M = 8 × 512 (8 cabezas), MLP 4× | presupuesto | Genie (arXiv 2402.15391) Tab. 10; mismos tamaños que el GPT y el MaskGIT propios |
| Inicialización | no publicada | N(0, 0.02) en Linear y Embedding, bias 0 | propio | convención GPT-2/MaskGIT |
| Precisión | bf16 | bf16 (autocast; solo en CUDA) | fiel | Genie (arXiv 2402.15391) §3 |
| Optimizador | AdamW, β = (0.9, 0.9) | igual, eps 1e-8 | fiel | Genie (arXiv 2402.15391) Tab. 9 |
| Weight decay | 1e-4 | 1e-4 en todo lo de >= 2 dimensiones (no biases ni normas) | fiel | Genie (arXiv 2402.15391) Tab. 9; a qué parámetros se aplica: propio |
| Learning rate: valores | max 3e-5, min 3e-6 | max 3e-4, min 3e-5 (x10): lo eligió el humo de 300 pasos entre fuente, raíz, x3, x10 y x30 (CE de monitoreo 6,58 / 6,84 / 5,58 / 4,45 / 4,98; results/humo_genie.json) | presupuesto | Genie (arXiv 2402.15391) Tab. 9 |
| Warmup | 5k pasos de 200k (fracción 0.025) | fracción 0.025 de la corrida, piso 100 pasos | presupuesto | Genie (arXiv 2402.15391) Tab. 9, Tab. 10; nuestra corrida tiene muchos menos pasos |
| Forma tras el warmup | no publicada para la dinámica (el tokenizador usa cosine decay) | coseno del máximo al 10 % (3e-4 a 3e-5) | propio | Genie (arXiv 2402.15391) App. F.1; Jafar (no oficial) también usa coseno |
| Batch | 256 | 32 por paso = 16 x accum 2 (~1.520 pasos con 780k frames); el lr lo eligió el humo | presupuesto | Genie (arXiv 2402.15391) Tab. 10; micro-batch por la memoria de 12 GB |
| Grad clip | no publicado | 1.0 | propio | Genie no lo publica |
| Pasos | 200k | los que da el presupuesto de 780k frames (~1.520 pasos de 32 x 16 frames) | presupuesto | Genie (arXiv 2402.15391) Tab. 10 |
<!-- /auto -->

## 3. Receta de entrenamiento

<!-- auto:receta kind=genie -->
| modelo | fuente | optimizador | betas | lr | wd | EMA | clip | precisión | batch x frames (frameskip) x acumulación | presupuesto (frames) | extras | pasos del paper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `genie_s`, `genie_m`, `genie_l` | Genie (arXiv 2402.15391) Tab. 9, 10, App. F.1; schedule y clip: propio | adamw | (0.9, 0.9) | warmup: 0->0.0003 (0.025); cosine: coseno 0.0003->3e-05 (resto) | 0.0001 | no | 1.0 | bf16 | 16 x 16 (1) x 2 | 780000 | - | 200000 |

Valores forzados (campo: motivo, categoría):

- `genie_s`: batch: 32 muestras por paso (16 x accum 2) en vez de 256: con el presupuesto fijo de 780k frames da ~1.520 pasos (presupuesto)
- `genie_s`: lr: 3e-4 -> 3e-5 (x10 del 3e-5 -> 3e-6 de Genie): con ~1.520 pasos, el humo de 300 pasos eligió x10 entre fuente, raíz, x3, x10 y x30 (results/humo_genie.json) (presupuesto)
- `genie_s`: warmup: 0.025 de la corrida (5k de 200k) con piso de 100 pasos: el presupuesto da pocos pasos (presupuesto)
- `genie_s`: lr_segs: coseno hasta el 10 % del máximo tras el warmup: Genie no publica la forma para la dinámica (propio)
- `genie_s`: grad_clip: 1.0: Genie no lo publica (propio)
<!-- /auto -->

El batch de la fuente es 256: acá son 16 por micro-batch con acumulación 2 (32 efectivos, unos 1.500 pasos con el presupuesto común). Como cambió el batch, el humo elige el lr (el de la fuente, escalado por raíz o lineal, x3 y x10); la elección entra a la receta con un commit.

## 4. Acciones

<!-- auto:acciones modelo=genie -->
Evaluación: `video`.

| acción | qué hace |
|---|---|
| `genie receta` | muestra la receta y la Config efectivas, con los overrides y los valores forzados marcados |
| `genie entrenar` | entrena de cero con la receta de la fuente (`--humo`: prueba corta de 300 pasos para elegir el lr) |
| `genie continuar` | retoma una corrida cortada o, con `--presupuesto`, la extiende |
| `genie evaluar` | mide el modelo (video: CE, rollouts, LPIPS y FVD; latentes: error, probes, sorpresa y CEM) |
| `genie bench` | mide la latencia (el modelo solo, o la cadena de punta a punta con `--cadena`) |
| `genie demo` | genera la demostración (mp4 con el video real y el predicho, o figura para los latentes) |
| `genie exportar` | exporta a ONNX |
| `genie estado` | porcentaje, paso, loss, lr, tramo y ETA de las corridas del modelo |
<!-- /auto -->

### receta

<!-- auto:opciones modelo=genie accion=receta -->
- `--set`: receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli genie receta --tamaño s
python -m src.cli genie receta --tamaño s --set receta.lr_peak=5e-5 --set config.default_steps=8
```

### entrenar

Entrena de cero con la receta de Genie y registra la corrida. Los nombres son `genie_<tamaño>[_s<semilla>]`.

<!-- auto:opciones modelo=genie accion=entrenar -->
- `--set`: receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)
- `--humo`: tres entrenamientos de 300 pasos (lr de la receta, por raíz y lineal) y elige
- `--humo-candidatos`: con --humo: candidatos a probar, p. ej. fuente,raiz,x3 (default: fuente,raiz,lineal,x3,x10)
- `--forzar`: entrenar: seguir aunque la corrida ya tenga last.pt o final.pt (train.py la reanuda desde last.pt)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli genie entrenar --tamaño s --humo        # elige el lr con 300 pasos por candidato
python -m src.cli genie entrenar --tamaño s
python -m src.cli genie entrenar --tamaño s --semilla 1    # corrida genie_s_s1
python -m src.cli genie entrenar --tamaño s --variante prueba --limite 20 --device cpu
```

### continuar

Retoma una corrida cortada, o con `--presupuesto` la extiende. La receta de Genie decae en coseno sin meseta, así que extender no equivale a entrenar más desde el principio: la CLI usa `--restart` (re-warmup y coseno nuevo), lo anota como no equivalente y la deja fuera de la tabla principal. Para la tabla se reentrena de cero con el presupuesto nuevo.

<!-- auto:opciones modelo=genie accion=continuar -->
- `--set`: receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)
- `--restart`: continuar con re-warmup y coseno nuevo (no equivalente)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli genie continuar --tamaño s
python -m src.cli genie continuar --tamaño s --presupuesto 2x --restart
```

### evaluar

CE, rollouts de 400 clips (1 s de contexto, 3 s generados), LPIPS, FVD con intervalos por segmento y el piso de FVD. `--pasos` evalúa varios números de pasos de decodificación en la misma llamada; los resultados llevan el sufijo `_p<pasos>` (`genie_s_p8`). La temperatura de la familia sale de `results/temperatura_genie_s.json` (el barrido en el holdout).

<!-- auto:opciones modelo=genie accion=evaluar -->
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
python -m src.cli genie evaluar --tamaño s --barrer-temperatura
python -m src.cli genie evaluar --tamaño s --pasos 1,2,4,8,12,25
python -m src.cli genie evaluar --tamaño s --semilla 1 --pasos 8
python -m src.cli genie evaluar --tamaño s --pasos 4 --cadena
python -m src.cli genie evaluar --tamaño s --pasos 4 --limite 5 --device cpu
```

### bench

Latencia por frame con batch 1 para cada número de pasos (el frame completo va en un CUDA graph), o con `--cadena` la cadena de punta a punta; `--ritmo 50` la mide a 20 FPS de cámara y `--diagnostico` explica el piso de unos 20 ms.

<!-- auto:opciones modelo=genie accion=bench -->
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
- `--tag`: nombre del dispositivo (radeon860m...) en los JSON, para no pisar los de otra máquina; con rocm es obligatorio en los benchs que escriben results/bench/

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli genie bench --tamaño s --pasos 1,2,4,8,12,25
python -m src.cli genie bench --tamaño s --pasos 4 --cadena
python -m src.cli genie bench --tamaño s --pasos 4 --cadena --ritmo 50
python -m src.cli genie bench --tamaño s --pasos 4 --cadena --ritmo 50 --diagnostico
python -m src.cli genie bench --tamaño s --pasos 4 --aleatorio --device cpu --limite 20
```

### demo

mp4 con el video real y el predicho lado a lado.

<!-- auto:opciones modelo=genie accion=demo -->
- `--dtype`: precisión de la inferencia
- `--temperatura`: temperatura de muestreo
- `--pasos`: pasos de decodificación, uno o varios separados por coma (maskgit, genie)
- `--aleatorio`: modelo sin entrenar con la arquitectura de la corrida (latencia)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli genie demo --tamaño s --pasos 4
```

### exportar

Exporta a ONNX UN paso de decodificación con el contexto de forma fija (los 15 frames de la config): entra el frame nuevo con MASK y el (k, v) temporal de cada capa, sale (1, 128, 1024). Es lo que se repite S veces por frame; el prefill del contexto y el muestreo quedan en PyTorch. Verifica el resultado contra PyTorch (ORT en CPU) y escribe `checkpoints/onnx/<nombre>_paso_k16.onnx`. Para medirlo en la NPU o la iGPU: `python -m src.ort_model --onnx <archivo> --provider npu|dml|cpu` (ver [uso](../uso.md)). Con `-- --parte prefill` (los argumentos tras `--` pasan a `src.export_onnx`) exporta en cambio el prefill del contexto: entra `ctx` (1, 15, 128) y salen el (k, v) de cada capa, que es lo que recibe el paso; se mide igual.

<!-- auto:opciones modelo=genie accion=exportar -->
- `--dtype`: precisión de la inferencia
- `--aleatorio`: modelo sin entrenar con la arquitectura de la corrida (latencia)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli genie exportar --tamaño s --aleatorio
```

### estado

<!-- auto:opciones modelo=genie accion=estado -->
- `--vigilar`: estado: repite cada N segundos (default 10)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli genie estado
```

## 5. Qué archivos escribe

| Acción | Archivos |
|---|---|
| `entrenar`, `continuar` | `checkpoints/<run>/{last.pt, final.pt, log.jsonl, status.json}`, `results/train_<run>.json`, entrada de `results/checkpoints.json`. |
| `entrenar --humo` | `checkpoints/genie_humo_*/` y `results/humo_genie.json` con el lr elegido. |
| `evaluar` | `results/eval_<run>.json`; por cada valor de `--pasos`: `results/rollouts/<run>_p<pasos>.npy` y `results/video_<run>_p<pasos>.json`. Con `--barrer-temperatura`: `results/temperatura_<run>.json`. |
| `bench` | `results/bench/<run>_p<pasos>__<dtype>_graphs.json`; con `--cadena`, `results/infer/latencia_<run>__<dispositivo>.json` (cada nombre lleva `_s<pasos>`). |
| `demo` | `results/demo/*.mp4` y su `.json`. |

## 6. Problemas frecuentes

- **"la corrida ya existe":** usá `continuar`, otra `--semilla` o `--variante`.
- **GPU ocupada:** ver [`../uso.md`](../uso.md#bitácora) y la sección de la cola; una acción de GPU se encola si la GPU está tomada.
- **`--pasos` sin efecto:** solo `genie` y `maskgit` decodifican por pasos; en el resto la CLI lo rechaza.
- **Más pasos, ¿mejor?:** no necesariamente en todas las métricas; la tabla de [`../resultados.md`](../resultados.md) tiene una fila por número de pasos y las conclusiones están a evaluar.
- **Latencia irregular:** mediciones con la GPU ocupada o con otro bench en curso. Corré el bench con la GPU libre.
- **Memoria en 12 GB:** el micro-batch es 16 con acumulación 2; si pagina a la RAM bajá a `--set receta.batch=8 --set receta.accum=4` (mismo batch efectivo).

## 7. Notebook

[`notebooks/modelos/genie.ipynb`](../../notebooks/modelos/genie.ipynb): el barrido de pasos y de `choice_temp`, y cómo se destapa un frame paso a paso.
