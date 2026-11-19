# plantilla: el modelo mínimo para copiar

## 1. Qué es y su fuente

No reproduce ningún paper: es un predictor de tokens mínimo pero real (`src/models/_plantilla.py`, ~0,5M de parámetros, sin atención) que muestra todo lo que un modelo tiene que declarar para entrar al repo: `Config`, `recipe()`, `FIDELIDAD`, `ACCIONES`, `EVALUACION` y `CONTRATO`. Cada celda mira la misma celda de los últimos frames y un resumen global del frame anterior. Entrena en CPU en segundos.

Se copia para sumar un modelo propio; los pasos están en [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md#sumar-un-modelo). **No entra en la cola ni en la tabla de resultados.**

## 2. Fidelidad

Todo es `propio` o `datos`: no hay fuente. Un modelo real cita su paper en cada fila.

<!-- auto:fidelidad kind=plantilla -->
| decisión | fuente / paper | aquí | categoría | fuente |
|---|---|---|---|---|
| Arquitectura | no hay fuente: modelo de ejemplo | embedding por token + una proyección por retardo + resumen global del frame anterior + MLP de 2 capas | propio | propio: plantilla del repo (src/models/_plantilla.py); no reproduce ningún paper |
| Entrada y salida | no hay fuente | 128 tokens VQ por frame, logits (128, 1024) del frame siguiente | datos | commaVQ: 128 tokens por frame, vocabulario de 1024 |
| Contexto | no hay fuente | los últimos `lags` = 2 frames | propio | propio: plantilla del repo (src/models/_plantilla.py); no reproduce ningún paper |
| Loss | no hay fuente | cross-entropy sobre los frames 1..F-1 (el frame 0 no tiene contexto) | propio | propio: plantilla del repo (src/models/_plantilla.py); no reproduce ningún paper |
| Optimizador y lr | no hay fuente | AdamW (0,9, 0,95), wd 0,01, lr 3e-3 con warmup y coseno | propio | propio: plantilla del repo (src/models/_plantilla.py); no reproduce ningún paper |
| Batch y presupuesto | no hay fuente | 16 x 8 frames por paso; 780k frames como todos | presupuesto | presupuesto común del repo: 780k frames vistos (100M tokens) |
<!-- /auto -->

## 3. Receta de entrenamiento

<!-- auto:receta kind=plantilla -->
| modelo | fuente | optimizador | betas | lr | wd | EMA | clip | precisión | batch x frames (frameskip) x acumulación | presupuesto (frames) | extras | pasos del paper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `plantilla` | propio: plantilla del repo (src/models/_plantilla.py); no reproduce ningún paper | adamw | (0.9, 0.95) | warmup: 0->0.003 (0.05); cosine: coseno 0.003->0.0003 (resto) | 0.01 | no | 1.0 | fp32 | 16 x 8 (1) x 1 | 780000 | - | - |

Valores forzados (campo: motivo, categoría):

- `plantilla`: receta entera: no hay paper: valores razonables para un modelo chico (propio)
- `plantilla`: batch 16 x 8 frames: cabe en cualquier máquina; con 780k frames da ~6.000 pasos (presupuesto)
<!-- /auto -->

## 4. Acciones

<!-- auto:acciones modelo=plantilla -->
Evaluación: `video`.

| acción | qué hace |
|---|---|
| `plantilla receta` | muestra la receta y la Config efectivas, con los overrides y los valores forzados marcados |
| `plantilla entrenar` | entrena de cero con la receta de la fuente (`--humo`: prueba corta de 300 pasos para elegir el lr) |
| `plantilla continuar` | retoma una corrida cortada o, con `--presupuesto`, la extiende |
| `plantilla evaluar` | mide el modelo (video: CE, rollouts, LPIPS y FVD; latentes: error, probes, sorpresa y CEM) |
| `plantilla bench` | mide la latencia (el modelo solo, o la cadena de punta a punta con `--cadena`) |
| `plantilla demo` | genera la demostración (mp4 con el video real y el predicho, o figura para los latentes) |
| `plantilla estado` | porcentaje, paso, loss, lr, tramo y ETA de las corridas del modelo |
<!-- /auto -->

Funcionan igual que en [`gpt`](gpt.md): la CLI, la cola, la evaluación y el reporte toman el modelo del registro (`KINDS` en `src/models/__init__.py`).

### receta

<!-- auto:opciones modelo=plantilla accion=receta -->
- `--set`: receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli plantilla receta
```

### entrenar

<!-- auto:opciones modelo=plantilla accion=entrenar -->
- `--set`: receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)
- `--humo`: tres entrenamientos de 300 pasos (lr de la receta, por raíz y lineal) y elige
- `--humo-candidatos`: con --humo: candidatos a probar, p. ej. fuente,raiz,x3 (default: fuente,raiz,lineal,x3,x10)
- `--forzar`: entrenar: seguir aunque la corrida ya tenga last.pt o final.pt (train.py la reanuda desde last.pt)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli plantilla entrenar --variante prueba --limite 20 --device cpu
```

### continuar

<!-- auto:opciones modelo=plantilla accion=continuar -->
- `--set`: receta.campo=valor o config.campo=valor (se valida antes de lanzar; queda como desvío)
- `--restart`: continuar con re-warmup y coseno nuevo (no equivalente)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli plantilla continuar --variante prueba
```

### evaluar

<!-- auto:opciones modelo=plantilla accion=evaluar -->
- `--dtype`: precisión de la inferencia
- `--temperatura`: temperatura de muestreo
- `--ctx-frames`: frames de contexto en la inferencia (default 20 = 1 s)
- `--barrer-temperatura`: barrido de temperatura en el holdout
- `--cadena`: bench: latencia de la cadena de punta a punta; evaluar: calidad en línea y con la cadena (src.infer.quality)
- `--desde-cache`: evaluar: no rehace los rollouts que ya existen; recalcula las métricas desde results/cache (nombres de los resultados viejos, T = 1)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli plantilla evaluar --variante prueba --limite 5 --device cpu
```

### bench

<!-- auto:opciones modelo=plantilla accion=bench -->
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
python -m src.cli plantilla bench --aleatorio --device cpu --limite 20
```

### demo

<!-- auto:opciones modelo=plantilla accion=demo -->
- `--dtype`: precisión de la inferencia
- `--temperatura`: temperatura de muestreo
- `--aleatorio`: modelo sin entrenar con la arquitectura de la corrida (latencia)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli plantilla demo --variante prueba --limite 40 --device cpu
```

### estado

<!-- auto:opciones modelo=plantilla accion=estado -->
- `--vigilar`: estado: repite cada N segundos (default 10)

Más las [opciones comunes](README.md#opciones-comunes).
<!-- /auto -->

```bash
python -m src.cli plantilla estado
```

## 5. Qué archivos escribe

Los mismos que `gpt`: `checkpoints/<run>/`, `results/train_<run>.json`, `results/eval_<run>.json`, `results/rollouts/`, `results/video_<run>.json`, `results/bench/` y `results/demo/`. Las corridas de prueba (`--limite`) no se registran en `results/checkpoints.json`.

## 6. Problemas frecuentes

- **El test de contrato falla:** `python -m pytest tests/test_contrato.py` dice qué falta (receta, fidelidad con fuente, causalidad, sobreajuste de un batch, reanudación, stream igual a `generate`). Un modelo sin receta o sin fidelidad falla con mensaje.
- **Registrado pero no aparece en la CLI:** falta `ACCIONES` y `EVALUACION` en su módulo.
- **No lo dejes en `results/`:** las corridas de prueba con la plantilla no son resultados; borrá lo que escribieron antes de commitear.

## 7. Notebook

[`notebooks/modelos/plantilla.ipynb`](../../notebooks/modelos/plantilla.ipynb), si el generador de notebooks lo incluye; si no, el de [`gpt`](../../notebooks/modelos/gpt.ipynb) tiene las mismas secciones.
