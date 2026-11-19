# referencias: gpt2m, baselines, real y codec

Las entradas que no se entrenan con la CLI, pero que van en la tabla de resultados como referencia o como piso. Tienen las mismas acciones de evaluación y de latencia que el resto de los modelos; no tienen `entrenar`.

| Entrada | Qué es |
|---|---|
| `gpt2m` | El GPT de comma (307M, `commaai/commavq-gpt2m`): el maestro de la destilación y la referencia de calidad. Se usa tal cual, con una caché de claves y valores propia. |
| `unigram` | Cota inferior: la frecuencia de cada código en train, sin mirar el contexto. |
| `transicion` | Baseline de primer orden: la tabla de transición p(token / token de la misma celda en el frame anterior), ajustada en train. Se llamó `copy`. |
| `congelado` | Mezcla "copiar o unigrama" con temperatura 0: repite el último frame. Se llamaba `copy_mix`. |
| `real` | Los clips reales decodificados con el mismo decoder: el **piso de FVD** (400 clips contra 400). Ningún modelo puede estar mejor que esto. |
| `codec` | El VQ-VAE de comma (encoder y decoder), que es la mayor parte de la latencia de la cadena de punta a punta. |

## 1. Fuente y fidelidad

**gpt2m** (código y pesos de comma, sin cambios salvo la caché):

<!-- auto:fidelidad kind=gpt2m -->
| decisión | fuente / paper | aquí | categoría | fuente |
|---|---|---|---|---|
| Modelo | commaai/commavq-gpt2m: 24 capas, 16 cabezas, 1024, MLP 4096, vocab 1025, bloque 2580 | el código y los pesos de comma, sin cambios (third_party/commavq/utils/gpt.py) | fiel | commaVQ utils/gpt.py L26-123; config.json |
| Pesos | pytorch_model.bin de Hugging Face | data/raw/pytorch_model.bin, bajado por src.data | fiel | HF commaai/commavq-gpt2m |
| Muestreo y KV caché | generación autoregresiva con KV caché | la misma con caché estática propia (y CUDA graphs opcional); mismas distribuciones | propio | src/models/gpt2m.py: adaptación a la interfaz WorldModel |
| Receta de entrenamiento | 'a similar recipe to IRIS' (no publicada) | no se entrena aquí; es la referencia y el maestro de la destilación | fiel | commaVQ README L354 |
<!-- /auto -->

**Baselines** (`unigram`, `transicion`, `congelado`; todos propios):

<!-- auto:fidelidad kind=baselines -->
| decisión | fuente / paper | aquí | categoría | fuente |
|---|---|---|---|---|
| unigrama | no existe en la literatura de comma | p(y) = frecuencia del código en train, add-one | propio | baseline del repo (cota inferior) |
| transicion | no existe en la literatura de comma | p(y / x), x = código de la misma celda en el frame anterior; tabla 1024x1024 de conteos suavizada hacia el unigrama | propio | baseline del repo (primer orden por celda); se llamó 'copy' |
| congelado | no existe en la literatura de comma | mezcla p_copy·1[misma celda] + (1 - p_copy)·unigrama; con temperatura 0 repite el último frame | propio | baseline del repo (frame congelado, copy_mix) |
| Ajuste | - | conteos sobre train; alpha del suavizado por máxima verosimilitud en los últimos segmentos de train, nunca en val | propio | baseline del repo |
<!-- /auto -->

`real` y `codec` no tienen receta: `real` es el dato de comma pasado por su decoder, y `codec` es el VQ-VAE de comma sin cambios (el encoder y el decoder se miden en fp32 y fp16 con el cuantizador en fp32).

## 2. Receta

Ninguna de estas entradas se entrena con la CLI, así que no tienen receta. `gpt2m` viene preentrenado; los baselines se ajustan por conteo sobre train la primera vez que se cargan.

## 3. Acciones

<!-- auto:acciones modelo=gpt2m -->
| acción | qué hace |
|---|---|
| `gpt2m evaluar` | mide el modelo (video: CE, rollouts, LPIPS y FVD; latentes: error, probes, sorpresa y CEM) |
| `gpt2m bench` | mide la latencia (el modelo solo, o la cadena de punta a punta con `--cadena`) |
<!-- /auto -->

<!-- auto:acciones modelo=transicion -->
| acción | qué hace |
|---|---|
| `transicion evaluar` | mide el modelo (video: CE, rollouts, LPIPS y FVD; latentes: error, probes, sorpresa y CEM) |
| `transicion bench` | mide la latencia (el modelo solo, o la cadena de punta a punta con `--cadena`) |
<!-- /auto -->

<!-- auto:acciones modelo=real -->
| acción | qué hace |
|---|---|
| `real evaluar` | mide el modelo (video: CE, rollouts, LPIPS y FVD; latentes: error, probes, sorpresa y CEM) |
<!-- /auto -->

<!-- auto:acciones modelo=codec -->
| acción | qué hace |
|---|---|
| `codec bench` | mide la latencia (el modelo solo, o la cadena de punta a punta con `--cadena`) |
| `codec exportar` | exporta a ONNX |
<!-- /auto -->

### gpt2m evaluar

CE con el protocolo de comma (59 ventanas de 20 frames por segmento; `ce_all` incluye los BOS y es comparable con el 2,02 que reporta comma), rollouts, LPIPS, FVD y piso. Usa lotes de 16 clips.

<!-- auto:opciones modelo=gpt2m accion=evaluar -->
- `--dtype`: precisión de la inferencia
- `--temperatura`: temperatura de muestreo
- `--ctx-frames`: frames de contexto en la inferencia (default 20 = 1 s)
- `--barrer-temperatura`: barrido de temperatura en el holdout
- `--cadena`: bench: latencia de la cadena de punta a punta; evaluar: calidad en línea y con la cadena (src.infer.quality)
- `--desde-cache`: evaluar: no rehace los rollouts que ya existen; recalcula las métricas desde results/cache (nombres de los resultados viejos, T = 1)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli gpt2m evaluar
python -m src.cli gpt2m evaluar --limite 5 --device cpu
```

### gpt2m bench

Latencia por frame con batch 1 (fp16 con CUDA graphs en CUDA).

<!-- auto:opciones modelo=gpt2m accion=bench -->
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
python -m src.cli gpt2m bench
python -m src.cli gpt2m bench --cadena
```

### unigram evaluar

<!-- auto:opciones modelo=unigram accion=evaluar -->
- `--dtype`: precisión de la inferencia
- `--temperatura`: temperatura de muestreo
- `--ctx-frames`: frames de contexto en la inferencia (default 20 = 1 s)
- `--barrer-temperatura`: barrido de temperatura en el holdout
- `--cadena`: bench: latencia de la cadena de punta a punta; evaluar: calidad en línea y con la cadena (src.infer.quality)
- `--desde-cache`: evaluar: no rehace los rollouts que ya existen; recalcula las métricas desde results/cache (nombres de los resultados viejos, T = 1)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli unigram evaluar
```

### unigram bench

<!-- auto:opciones modelo=unigram accion=bench -->
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
python -m src.cli unigram bench
```

### transicion evaluar

`--cadena` mide además la calidad en línea y con la cadena de punta a punta.

<!-- auto:opciones modelo=transicion accion=evaluar -->
- `--dtype`: precisión de la inferencia
- `--temperatura`: temperatura de muestreo
- `--ctx-frames`: frames de contexto en la inferencia (default 20 = 1 s)
- `--barrer-temperatura`: barrido de temperatura en el holdout
- `--cadena`: bench: latencia de la cadena de punta a punta; evaluar: calidad en línea y con la cadena (src.infer.quality)
- `--desde-cache`: evaluar: no rehace los rollouts que ya existen; recalcula las métricas desde results/cache (nombres de los resultados viejos, T = 1)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli transicion evaluar
python -m src.cli transicion evaluar --cadena
```

### transicion bench

<!-- auto:opciones modelo=transicion accion=bench -->
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
python -m src.cli transicion bench
python -m src.cli transicion bench --cadena
```

### congelado evaluar

Corre siempre con temperatura 0 (repite el último frame).

<!-- auto:opciones modelo=congelado accion=evaluar -->
- `--dtype`: precisión de la inferencia
- `--temperatura`: temperatura de muestreo
- `--ctx-frames`: frames de contexto en la inferencia (default 20 = 1 s)
- `--barrer-temperatura`: barrido de temperatura en el holdout
- `--cadena`: bench: latencia de la cadena de punta a punta; evaluar: calidad en línea y con la cadena (src.infer.quality)
- `--desde-cache`: evaluar: no rehace los rollouts que ya existen; recalcula las métricas desde results/cache (nombres de los resultados viejos, T = 1)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli congelado evaluar
```

### congelado bench

<!-- auto:opciones modelo=congelado accion=bench -->
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
python -m src.cli congelado bench
```

### real evaluar

El piso de FVD: los clips reales contra sí mismos. Para que sea 400 contra 400 se usa un segundo set de los mismos segmentos de val con otros arranques (`eval_b`) y se compara con `estadistica piso`.

<!-- auto:opciones modelo=real accion=evaluar -->
- `--dtype`: precisión de la inferencia
- `--temperatura`: temperatura de muestreo
- `--ctx-frames`: frames de contexto en la inferencia (default 20 = 1 s)
- `--barrer-temperatura`: barrido de temperatura en el holdout
- `--cadena`: bench: latencia de la cadena de punta a punta; evaluar: calidad en línea y con la cadena (src.infer.quality)
- `--desde-cache`: evaluar: no rehace los rollouts que ya existen; recalcula las métricas desde results/cache (nombres de los resultados viejos, T = 1)
- `--clips`: set de clips de los rollouts (solo `real evaluar`: eval, holdout o eval_b)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli real evaluar
python -m src.cli real evaluar --clips eval_b
python -m src.cli estadistica piso --b real_eval_b
```

### codec bench

Latencia y fidelidad del VQ-VAE de comma en el dispositivo (encoder y decoder; fp32 y fp16 en CUDA).

<!-- auto:opciones modelo=codec accion=bench -->
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
- `--ort`: codec bench: mide el codec exportado con ONNX Runtime (src.infer.bench_ort); con dml y npu es lo único que hay

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli codec bench
python -m src.cli codec bench --device cpu --limite 20
python -m src.cli codec bench --device rocm --tag radeon860m
python -m src.cli codec bench --device npu --limite 100
```

Con `--device dml` o `--device npu` (o `--ort`, para la CPU de ONNX Runtime) mide el codec exportado con `src.infer.bench_ort`: hay que correr antes `codec exportar`. Lo que se quiera pasar al script va después de `--`: `-- --profile` cuenta los nodos por provider, `-- --only enc` mide una etapa y `-- --dec dml` pone el decoder en otro provider.

### codec exportar

ONNX del encoder y del decoder, para ONNX Runtime (CPU, DirectML o NPU).

<!-- auto:opciones modelo=codec accion=exportar -->
- `--dtype`: precisión de la inferencia

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli codec exportar
python -m src.cli codec exportar --dtype fp16
```

## 4. Qué archivos escribe

| Entrada y acción | Archivos |
|---|---|
| `<modelo> evaluar` | `results/eval_<modelo>.json` (CE), `results/rollouts/<modelo>.npy` y `.json`, `results/video_<modelo>.json`. |
| `real evaluar` | `results/rollouts/real.npy`, `results/video_real.json`; con `--clips eval_b`, `results/rollouts/real_eval_b.npy` y `results/video_real_eval_b.json`. |
| `estadistica piso` | `results/piso_fvd.json`. |
| `<modelo> bench` | `results/bench/<modelo>__<dtype>[_graphs].json`; con `--cadena`, `results/infer/latencia_<modelo>__<dispositivo>.json`. |
| `codec bench` | `results/infer/codec_<tag>.json` y `results/bench/decoder__<dtype>[__<tag>].json`; con `--device dml`, `--device npu` o `--ort`, `results/infer/ort_<tag>.json`. |
| `codec exportar` | `checkpoints/onnx/` (encoder y decoder). |

## 5. Problemas frecuentes

- **Faltan los pesos de `gpt2m`:** `python -m src.data` baja `data/raw/pytorch_model.bin`. Si Hugging Face se cuelga, `HF_HUB_DISABLE_XET=1`.
- **El piso de FVD depende de la cantidad de clips:** el FVD baja al crecer la muestra. El piso comparable es 400 contra 400 (`real evaluar --clips eval_b` y `estadistica piso`); un piso con mitades es de muestra chica y no se compara con los modelos.
- **Los nombres viejos:** `copy` y `copy_mix` son los nombres anteriores de `transicion` y `congelado`; `load_model("copy")` anda con un aviso.
- **`gpt2m evaluar` es lento:** son 307M de parámetros; la evaluación se hace en lotes de 16 clips y en fp16 en CUDA.
- **"la corrida ya existe", árbol sucio, GPU ocupada:** ver [`../uso.md`](../uso.md).

## 6. Notebook

[`notebooks/modelos/codec.ipynb`](../../notebooks/modelos/codec.ipynb): fidelidad fp32 contra fp16 y latencia por dispositivo del codec. Las demás referencias se ven en `notebooks/resultados.ipynb`, que muestra el piso de FVD y los baselines.
