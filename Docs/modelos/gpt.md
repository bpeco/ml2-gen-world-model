# gpt: GPT alumno autoregresivo

## 1. Qué es y su fuente

El GPT de commaVQ (`commaai/commavq-gpt2m`, 307M) a menor escala: el mismo formato de secuencia (128 tokens VQ por frame más un BOS, bloques de 20 frames, vocabulario de 1025), posición absoluta aprendida y atención causal, con tamaños S (6 x 384), M (8 x 512) y L (12 x 768). Genera un frame con 128 pasadas, un token por vez, con caché de claves y valores y CUDA graphs.

- **Fuente de la arquitectura:** commaVQ, `utils/gpt.py` y `config.json` de `commavq-gpt2m`.
- **Fuente de la receta:** IRIS (arXiv 2209.00588), que commaVQ dice seguir ("a similar recipe to IRIS"). Comma no publica sus hiperparámetros.
- **Destilación (`--destilar`):** la loss suma la KL contra `gpt2m` (el maestro). Es un agregado propio, declarado como tal.
- **En la cola:** el único GPT entrenado es `gpt_s_kd` (S destilado). `gpt_s` y `gpt_m` de cero están implementados, fuera de la cola, como opcionales (trabajos opcionales de `scripts/queue.toml`).
- **Evaluación:** `video` (CE con el protocolo de comma, rollouts de 1 s de contexto y 3 s generados, LPIPS y FVD).

## 2. Fidelidad

Qué sigue a la fuente y qué no. Las categorías son `fiel`, `datos`, `presupuesto` y `propio`.

<!-- auto:fidelidad kind=gpt -->
| decisión | fuente / paper | aquí | categoría | fuente |
|---|---|---|---|---|
| Arquitectura | pre-LN, GELU (tanh), MLP 4x, bias en Linear, ln_f, cabeza sin bias | igual | fiel | commaVQ (utils/gpt.py, commaai/commavq-gpt2m) L63-123 |
| Posición | absoluta aprendida (wpe) sobre 2580 posiciones | igual (pos='abs') | fiel | commaVQ (utils/gpt.py, commaai/commavq-gpt2m) L117; IRIS (arXiv 2209.00588; github.com/eloialonso/iris) world_model.py L38 |
| Weight tying | sin tying (lm_head aparte) | sin tying | fiel | commaVQ (utils/gpt.py, commaai/commavq-gpt2m) L123 |
| Formato de secuencia | 128 tokens + BOS (1024) por frame, bloque de 20 frames, vocab 1025 | igual | fiel | commaVQ (utils/gpt.py, commaai/commavq-gpt2m); nanogpt/prepare.py L16, L29-31 |
| Tamaño | 24 capas, 16 cabezas, 1024 (comma); 10 capas, 256 (IRIS) | S = 6 x 384 (6 cabezas), M = 8 x 512 (8 cabezas), L = 12 x 768 | presupuesto | commaVQ (utils/gpt.py, commaai/commavq-gpt2m) config.json; IRIS (arXiv 2209.00588; github.com/eloialonso/iris) world_model/default.yaml |
| Posición factorizada (ablación) | no existe en comma ni IRIS | pos='frame': embedding de frame + de celda; solo para correr las corridas viejas | propio | decisión del repo (corridas previas a la revisión de fidelidad) |
| Dropout | IRIS: 0.1 en embedding, residual y atención; comma: 0 (dropout_p=0.0) | 0.1 en los tres (default de Config; el de IRIS, que es la receta que comma dice seguir) | fiel | IRIS (arXiv 2209.00588; github.com/eloialonso/iris) world_model/default.yaml L8-10; commaVQ (utils/gpt.py, commaai/commavq-gpt2m) L97 |
| Inicialización | N(0, 0.02) en Linear y Embedding, bias 0, LayerNorm 1/0 | igual, sin la escala 1/sqrt(2L) de GPT-2 en las proyecciones residuales (res_scale=False) | fiel | IRIS (arXiv 2209.00588; github.com/eloialonso/iris) utils.py L52-59 |
| Optimizador | AdamW, betas (0.9, 0.999) | igual, eps 1e-8 | fiel | IRIS (arXiv 2209.00588; github.com/eloialonso/iris) trainer.py L91, Tab. 5 |
| Learning rate | 1e-4 constante, sin warmup ni scheduler | igual | fiel | IRIS (arXiv 2209.00588; github.com/eloialonso/iris) config/trainer.yaml L55 |
| Weight decay | 0.01 solo en el weight de nn.Linear y nn.Conv1d (no bias, LayerNorm ni Embedding, incluida la posición) | igual (wd_on='lineales'; incluye la cabeza) | fiel | IRIS (arXiv 2209.00588; github.com/eloialonso/iris) trainer.yaml L66, utils.py L14-36 |
| Grad clip | 10.0 | 10.0 | fiel | IRIS (arXiv 2209.00588; github.com/eloialonso/iris) trainer.yaml L65 |
| Batch | 64 secuencias de 20 pasos | 16 por paso, sin acumulación (~2.440 pasos con 780k frames); lr a confirmar en el humo | presupuesto | IRIS (arXiv 2209.00588; github.com/eloialonso/iris) trainer.yaml, world_model/default.yaml L2-3 |
| Precisión | fp32 (sin autocast) | fp32 | fiel | IRIS (arXiv 2209.00588; github.com/eloialonso/iris) trainer.py |
| Pasos / presupuesto | no publicado para comma | 780k frames (100M tokens) | presupuesto | presupuesto de cómputo del repo |
| Datos | segmentos de 1200 frames a 20 fps | igual (ventanas de 20 frames) | datos | commaVQ (utils/gpt.py, commaai/commavq-gpt2m) nanogpt/prepare.py |
| Destilación (--kd) | no existe en comma ni IRIS | loss = alpha CE + (1-alpha) tau^2 KL(maestro gpt2m // alumno), alpha 0.5, tau 1, misma receta que el alumno | propio | Hinton et al. 2015; extensión del repo |
<!-- /auto -->

## 3. Receta de entrenamiento

<!-- auto:receta kind=gpt -->
| modelo | fuente | optimizador | betas | lr | wd | EMA | clip | precisión | batch x frames (frameskip) x acumulación | presupuesto (frames) | extras | pasos del paper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `gpt_s`, `gpt_m`, `gpt_l` | IRIS (arXiv 2209.00588; github.com/eloialonso/iris); commaVQ: 'a similar recipe to IRIS' | adamw | (0.9, 0.999) | constante: 0.0001 (resto) | 0.01 (lineales) | no | 10.0 | fp32 | 16 x 20 (1) x 1 | 780000 | dropout=0.1 | - |

Valores forzados (campo: motivo, categoría):

- `gpt_s`: batch: 16 por paso en vez de 64 (IRIS): con el presupuesto fijo de 780k frames da ~2.440 pasos; lr a confirmar en el humo (presupuesto)
- `gpt_s`: presupuesto: 780k frames (100M tokens) en vez de lo que haya usado comma (no publicado) (presupuesto)
- `gpt_s`: tamaño: 6x384 / 8x512 en vez de 24x1024 (presupuesto)
<!-- /auto -->

El lr de la receta es el de la fuente; con el batch chico del presupuesto común, el humo (`entrenar --humo`) elige entre el lr de la fuente, el escalado por raíz o lineal, y x3 y x10. La elección entra a la receta con un commit, no con un `--set` a mano. Con `--destilar` la receta es la misma más la pérdida de destilación.

## 4. Acciones

<!-- auto:acciones modelo=gpt -->
Evaluación: `video`.

| acción | qué hace |
|---|---|
| `gpt receta` | muestra la receta y la Config efectivas, con los overrides y los valores forzados marcados |
| `gpt entrenar` | entrena de cero con la receta de la fuente (`--humo`: prueba corta de 300 pasos para elegir el lr) |
| `gpt continuar` | retoma una corrida cortada o, con `--presupuesto`, la extiende |
| `gpt evaluar` | mide el modelo (video: CE, rollouts, LPIPS y FVD; latentes: error, probes, sorpresa y CEM) |
| `gpt bench` | mide la latencia (el modelo solo, o la cadena de punta a punta con `--cadena`) |
| `gpt demo` | genera la demostración (mp4 con el video real y el predicho, o figura para los latentes) |
| `gpt estado` | porcentaje, paso, loss, lr, tramo y ETA de las corridas del modelo |
<!-- /auto -->

### receta

Muestra la receta y la Config efectivas, con los overrides y los forzados marcados. No usa la GPU.

<!-- auto:opciones modelo=gpt accion=receta -->
- `--set`: receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli gpt receta --tamaño s
python -m src.cli gpt receta --tamaño m --set receta.lr_peak=3e-4
```

### entrenar

Entrena de cero con la receta de la fuente y registra la corrida en `results/checkpoints.json` al terminar. Con `--destilar` carga el maestro `gpt2m` (`data/raw/pytorch_model.bin`) y entrena con KD; la corrida destilada lleva el sufijo `_kd` si se la pide con `--variante kd` (así se llama en la cola).

<!-- auto:opciones modelo=gpt accion=entrenar -->
- `--set`: receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)
- `--humo`: tres entrenamientos de 300 pasos (lr de la receta, por raíz y lineal) y elige
- `--humo-candidatos`: con --humo: candidatos a probar, p. ej. fuente,raiz,x3 (default: fuente,raiz,lineal,x3,x10)
- `--destilar`: destilación del maestro gpt2m (--kd)
- `--forzar`: entrenar: seguir aunque la corrida ya tenga last.pt o final.pt (train.py la reanuda desde last.pt)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
# el humo (entrenamientos de 300 pasos con distintos lr) corre sin el maestro: elige el lr del alumno
python -m src.cli gpt entrenar --tamaño s --humo
# GPT-S destilado con el presupuesto común (nombre de la corrida: gpt_s_kd)
python -m src.cli gpt entrenar --tamaño s --destilar --variante kd
# GPT-S de cero (opcional)
python -m src.cli gpt entrenar --tamaño s
# probar el circuito sin GPU: 20 pasos en CPU
python -m src.cli gpt entrenar --tamaño s --variante prueba --limite 20 --device cpu
```

### continuar

Retoma una corrida cortada (sigue desde `last.pt` con los mismos argumentos) o, con `--presupuesto`, la extiende en una corrida nueva `<run>_x<k>`. La receta de GPT es constante, así que la extensión es exacta.

<!-- auto:opciones modelo=gpt accion=continuar -->
- `--set`: receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)
- `--restart`: continuar con re-warmup y coseno nuevo (no equivalente)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli gpt continuar --tamaño s --variante kd
python -m src.cli gpt continuar --tamaño s --variante kd --presupuesto 2x
```

### evaluar

Mide el modelo con la evaluación de video: CE (`ce_all`, `ce_last`), rollouts de los 400 clips fijos, LPIPS y FVD con intervalos por segmento, y el piso de FVD. La temperatura sale del barrido en el holdout de la corrida de referencia de la familia (`results/temperatura_gpt_s_kd.json`); si no existe usa el default del modelo y lo anota. `--barrer-temperatura` hace ese barrido (con clips del holdout, nunca de val).

<!-- auto:opciones modelo=gpt accion=evaluar -->
- `--dtype`: precisión de la inferencia
- `--temperatura`: temperatura de muestreo
- `--ctx-frames`: frames de contexto en la inferencia (default 20 = 1 s)
- `--barrer-temperatura`: barrido de temperatura en el holdout
- `--cadena`: bench: latencia de la cadena de punta a punta; evaluar: calidad en línea y con la cadena (src.infer.quality)
- `--desde-cache`: evaluar: no rehace los rollouts que ya existen; recalcula las métricas desde results/cache (nombres de los resultados viejos, T = 1)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli gpt evaluar --tamaño s --variante kd --barrer-temperatura
python -m src.cli gpt evaluar --tamaño s --variante kd
python -m src.cli gpt evaluar --tamaño s --variante kd --cadena          # calidad en línea y con la cadena de punta a punta
python -m src.cli gpt evaluar --tamaño s --variante kd --limite 5 --device cpu
```

### bench

Latencia con batch 1: ms por frame (p50, p99) sin el decoder, o con `--cadena` la cadena completa video, encoder, modelo, decoder, video. Corre con la GPU libre.

<!-- auto:opciones modelo=gpt accion=bench -->
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
python -m src.cli gpt bench --tamaño s --variante kd
python -m src.cli gpt bench --tamaño s --variante kd --cadena
python -m src.cli gpt bench --tamaño s --variante kd --cadena --ritmo 50
python -m src.cli gpt bench --tamaño s --aleatorio --device cpu --limite 20    # arquitectura sin entrenar, para medir latencia
```

### demo

Genera un mp4 con el video real y el predicho lado a lado, con el codec de comma.

<!-- auto:opciones modelo=gpt accion=demo -->
- `--dtype`: precisión de la inferencia
- `--temperatura`: temperatura de muestreo
- `--aleatorio`: modelo sin entrenar con la arquitectura de la corrida (latencia)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli gpt demo --tamaño s --variante kd
python -m src.cli gpt demo --tamaño s --variante kd --limite 200
```

### estado

Una fila por corrida del modelo: en curso, en cola, lista, fallada o colgada (3 minutos sin actualizar), con porcentaje, paso, loss, lr, tramo y ETA.

<!-- auto:opciones modelo=gpt accion=estado -->
- `--vigilar`: estado: repite cada N segundos (default 10)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli gpt estado
python -m src.cli gpt estado --vigilar 30
```

## 5. Qué archivos escribe

| Acción | Archivos |
|---|---|
| `entrenar`, `continuar` | `checkpoints/<run>/{last.pt, final.pt, log.jsonl, status.json}`, `results/train_<run>.json` y la entrada de `results/checkpoints.json`. Una extensión es otra corrida (`<run>_x<k>`) con `parent` y la receta. |
| `entrenar --humo` | `checkpoints/gpt_humo_{fuente,raiz,lineal,x3,x10}/`, `results/humo_gpt.json` con el lr elegido. |
| `evaluar` | `results/eval_<run>.json` (CE), `results/rollouts/<run>.npy` y `.json`, `results/video_<run>.json` (LPIPS, FVD, piso). Con `--barrer-temperatura`: `results/temperatura_<run>.json`. Con `--cadena`: `results/infer/online_<run>.json`. |
| `bench` | `results/bench/<run>__<dtype>[_graphs].json`; con `--cadena`, `results/infer/latencia_<run>__<dispositivo>.json`. |
| `demo` | `results/demo/<run>_<modo>.mp4` y un `.json` con la latencia. |

Cada acción deja además una línea en `results/bitacora.jsonl` y anota el comando y el commit dentro de los JSON de resultados.

## 6. Problemas frecuentes

- **"la corrida ya existe":** `entrenar` no pisa ni reanuda una corrida con `last.pt` o `final.pt`. Retomala con `continuar`, o usá otra `--variante` o `--semilla`.
- **"rechazada: hay cambios sin commitear":** una acción de GPU exige el árbol limpio en `src/` y `scripts/`. Commiteá, o usá `--permitir-sucio` (el resultado no cuenta).
- **La acción quedó en la cola:** la GPU estaba tomada (`results/.gpu.lock`). Mirá `python -m src.cli cola ver`, esperá con `--esperar` o corré la cola con `python -m src.cli cola correr`.
- **`--destilar` no encuentra el maestro:** hace falta `data/raw/pytorch_model.bin` (`python -m src.data`). Si la descarga de Hugging Face se cuelga, `HF_HUB_DISABLE_XET=1`.
- **Se pasa de memoria o va diez veces más lento:** en una GPU de 12 GB en Windows, por encima de ~11 GB el driver pagina a la RAM sin avisar. No lances dos trabajos pesados a la vez; bajá el batch con `--set receta.batch=8`.
- **`evaluar` pide la temperatura:** hasta que exista `results/temperatura_gpt_s_kd.json` usa el default; corré antes `evaluar --barrer-temperatura`.
- **`--pasos` no aplica:** el GPT genera token a token; los pasos son de `genie` y `maskgit`.

## 7. Notebook

[`notebooks/modelos/gpt.ipynb`](../../notebooks/modelos/gpt.ipynb): las mismas acciones con barra de progreso y curvas; en particular, de cero contra destilado y la CE por posición del frame.
