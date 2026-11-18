# ml2-gen-world-model

World models de manejo sobre [commaVQ](https://github.com/commaai/commavq). Trabajo práctico de Aprendizaje Automático II (MIA, UdeSA).

Pregunta del proyecto: ¿cuánto se puede achicar y acelerar un world model de video de manejo sin perder calidad?

- Plan: [`Docs/plan-proyecto.md`](Docs/plan-proyecto.md). Cómo se implementó y validó: [`Docs/plan-implementacion.md`](Docs/plan-implementacion.md).
- Resultados y conclusiones: [`Docs/resultados.md`](Docs/resultados.md). Tabla de todas las configuraciones: [`results/summary.md`](results/summary.md).
- World models latentes, JEPA y LeWorldModel (E5): [`Docs/plan-jepa.md`](Docs/plan-jepa.md).
- Próximas pruebas, inferencia de punta a punta (video → encoder → modelo → decoder → video) en GPU y NPU: [`Docs/plan-inferencia.md`](Docs/plan-inferencia.md).

## Setup

### Windows con GPU NVIDIA

En PowerShell, desde la raíz del repo. El torch de PyPI para Windows es solo CPU: hay que instalarlo desde el índice de PyTorch.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-metrics.txt
git clone --depth 1 https://github.com/commaai/commavq.git third_party/commavq   # código del VQ-VAE y del GPT de comma
```

En el resto del README, `.venv/bin/python` es `.venv\Scripts\python`.

### Mac

Conviene un Python arm64 nativo: con uno x86_64 (Rosetta) solo se consigue torch 2.2.2.

```bash
uv venv --python cpython-3.12-macos-aarch64-none .venv
uv pip install --python .venv/bin/python -r requirements.txt -r requirements-metrics.txt
git clone --depth 1 https://github.com/commaai/commavq.git third_party/commavq
```

### Colab

[`notebooks/colab.ipynb`](notebooks/colab.ipynb) clona el repo, monta Drive, instala, baja los datos y corre E0, el bench en la T4 y, opcionalmente, entrenamiento y evaluación.

### Laptop con Radeon (ROCm) y NPU

`requirements-rocm.txt` instala el torch de AMD para gfx1152 (antes que `requirements.txt`); ver los comentarios del archivo y la sección 2.5 del plan. `device()` elige `cuda` igual: los builds ROCm usan la API `torch.cuda`. `requirements-npu.txt` es para el extra de ONNX Runtime en la NPU (`src/export_onnx.py`, `src/ort_model.py`).

Para medir la inferencia de punta a punta en la laptop (iGPU con ROCm y NPU con ONNX Runtime), ver la sección "Cómo correrlo" de [`Docs/plan-inferencia.md`](Docs/plan-inferencia.md).

`device()` ([`src/common.py`](src/common.py)) elige CUDA (NVIDIA o ROCm), MPS o CPU. `ML2_DEVICE=cpu` lo fuerza. Cada `results/*.json` anota el hardware usado.

## Datos

```bash
.venv/bin/python -m src.data   # shard 0 (train), val oficial, decoder y pesos del GPT de comma -> data/
.venv/bin/python -m src.viz    # frames de val decodificados -> results/
```

- Train: `data-0000.tar.gz`, 2.500 segmentos de 1 min (1200 x 8 x 16 tokens).
- Val: los 100 segmentos de `val.zip` de una revisión anterior del dataset. La lista queda en `results/val_split.json`. No comparten rutas con el shard de train.
- Si la descarga de Hugging Face se cuelga, probar con `HF_HUB_DISABLE_XET=1`.

## Código

Todo modelo implementa la interfaz `WorldModel` ([`src/models/base.py`](src/models/base.py)): `logits` (CE con el protocolo de comma), `last_frame_logits`, `next_frame`, `generate` y `rollout`. La evaluación, los rollouts y el bench no saben qué modelo miden. `src.models.load_model(spec)` acepta `gpt2m`, los baselines (`unigram`, `copy`, `copy_mix`), `random:<kind>_<size>` (sin entrenar, para latencia), un nombre de corrida (`gpt_s_kd`) o una ruta `.pt`.

| Archivo | Qué hace |
|---|---|
| `src/models/baselines.py` | Unigrama, copia (tabla de transición por celda) y `copy_mix` (a temperatura 0, frame congelado). |
| `src/models/gpt2m.py` | GPT de comma (307M) como `WorldModel`: referencia y maestro. Generación por lotes con KV cache y CUDA graphs. |
| `src/models/backbone.py` | Transformer propio: KV cache estática, máscara causal o causal por bloques, embedding de posición absoluto o factorizado (frame + celda). |
| `src/models/gpt.py` | GPT alumno autoregresivo: 128 pasadas por frame, con KV cache y CUDA graphs. |
| `src/models/maskgit.py` | MaskGIT: causal entre frames, bidireccional dentro del frame, S pasadas por frame con schedule coseno. |
| `src/models/convlstm.py` | ConvLSTM sobre la grilla 8×16: una pasada recurrente por frame. |
| `src/train.py`, `src/windows.py` | Entrenamiento con CE o CE + destilación, AMP, checkpoints cada 15 min y resume. |
| `src/evaluate.py` | CE con el protocolo de comma (`ce_all`, `ce_tok`) y `ce_last` (último frame, comparable con MaskGIT). |
| `src/rollouts.py` | 400 clips fijos (1 s de contexto + 3 s generados) → `results/rollouts/<nombre>.npy`. |
| `src/metrics_video.py` | Decodifica y mide LPIPS por horizonte y FVD (I3D) a 1, 2 y 3 s, con intervalos por bootstrap. |
| `src/bench.py` | Latencia con batch 1: ms por frame (mediana), sin el decoder. |
| `src/report.py`, `scripts/make_configs.py` | Figuras (Pareto, horizonte, CE vs. FVD) y tabla resumen. |
| `src/ejemplos.py` | Muestras visuales: decodifica rollouts y arma grillas (modelo × horizonte) y GIF lado a lado → `results/ejemplos/`. |
| `src/store.py` | Almacén de modelos entrenados para no reentrenar (ver abajo). |
| `src/export_onnx.py`, `src/ort_model.py` | Extra: una pasada de forma fija en ONNX Runtime (CPU, CUDA, DirectML o NPU). |
| `src/infer/codec.py` | Encoder y decoder del VQ-VAE de comma en el device: fp16 con el cuantizador en fp32 y un CUDA graph por etapa; entrada y salida uint8. |
| `src/infer/stream.py` | Estado de inferencia por frame (`observe`, `predict(k)`, `imagine`, `step`) para cada familia: buffer de tokens, MaskGIT con el frame entero en un graph, ConvLSTM con (h, c), y caché de latentes para JEPA y LeWM. |
| `src/infer/bench_codec.py`, `src/infer/bench_e2e.py` | Fidelidad y latencia del codec; latencia de la cadena completa (p50, p99, frames tarde), en línea e imaginación, en bucle cerrado o a 50 ms. |
| `src/infer/quality.py`, `src/infer/report_e2e.py` | Calidad a 1 paso en línea y verificación del camino nuevo contra los rollouts; tabla y figura de punta a punta. |
| `src/infer/demo.py` | Video real y predicho lado a lado en mp4. |
| `src/infer/export_codec.py`, `src/infer/bench_ort.py` | Codec y paso de ConvLSTM en ONNX, y la cadena en ONNX Runtime eligiendo el dispositivo de cada etapa (NPU, iGPU o CPU). |

## Experimentos

```bash
bash scripts/train_all.sh            # E1, E2, E2b, E3, E5 y ablaciones: 100M tokens cada uno -> checkpoints/
bash scripts/eval_all.sh quality     # CE + rollouts + LPIPS/FVD de todas las configuraciones -> results/
bash scripts/eval_all.sh latent      # E5: probes, sorpresa, acciones y CEM de JEPA y LeWM
bash scripts/eval_all.sh bench       # latencias (con la GPU libre) -> results/bench/
bash scripts/eval_all.sh bench-latent  # latencias de JEPA y LeWM (con la GPU libre)
bash scripts/e2e_all.sh all          # inferencia de punta a punta: codec, cadena completa y a ritmo de cámara
bash scripts/e2e_all.sh quality      # calidad a 1 paso en línea y verificación contra los rollouts
.venv/bin/python -m src.infer.report_e2e   # results/infer/summary_e2e.md y results/fig_pareto_e2e.png
.venv/bin/python scripts/make_configs.py && .venv/bin/python -m src.report   # figuras y results/summary.md
```

Los dos scripts saltan lo que ya existe, así que se pueden cortar y relanzar. Comandos sueltos:

```bash
.venv/bin/python -m src.evaluate --model gpt2m                      # E0: ce_all = 2,026
.venv/bin/python -m src.train --kind gpt --size s --run gpt_s_kd --config '{"pos": "frame"}' --kd
.venv/bin/python -m src.rollouts --model maskgit_s --steps 4 --dtype fp16 --graphs
.venv/bin/python -m src.metrics_video --name maskgit_s_s4
.venv/bin/python -m src.bench --model random:maskgit_s --steps 4 --dtype fp16 --graphs
.venv/bin/python -m src.ejemplos                                     # muestras de 3 clips -> results/ejemplos/
.venv/bin/python -m src.infer.bench_e2e --model maskgit_s --steps 4    # cadena completa, en línea, 3000 frames
.venv/bin/python -m src.infer.demo --model maskgit_s --steps 4         # video real y predicho -> results/demo/
```

`src/evaluate.py` replica el protocolo de `notebooks/eval.ipynb` de commaVQ (commit `a1adab98`): 59 ventanas de 20 frames (2.580 tokens) por segmento, con BOS antes de cada frame. `ce_all` incluye los targets BOS y es comparable con el 2,02 que reporta comma; gpt2m da 2,026 sobre los 100 segmentos.

## Notebooks

| Notebook | Para qué |
|---|---|
| [`notebooks/resultados.ipynb`](notebooks/resultados.ipynb) | Tabla resumen, hipótesis, figuras, muestras de cualquier clip y modelo, curvas de LPIPS y generación en vivo con un modelo entrenado. No entrena nada. |
| [`notebooks/entrenamiento.ipynb`](notebooks/entrenamiento.ipynb) | Entrena una configuración (GPT, MaskGIT o ConvLSTM; desde cero o destilada), muestra la curva de aprendizaje y la evalúa: CE, rollouts, muestras y latencia. |
| [`notebooks/colab.ipynb`](notebooks/colab.ipynb) | Preparación en Colab, E0 y latencias en la T4. |

Funcionan localmente y en Colab. Localmente necesitan un kernel en el `.venv`: `.venv/bin/python -m pip install ipykernel` (VS Code) o `jupyter`.

## Modelos entrenados: almacén compartido

Los checkpoints quedan en `checkpoints/<corrida>/` (`final.pt` es el modelo; `last.pt`, `log.jsonl` y `train.out` son del entrenamiento) y no van a git: GitHub rechaza archivos de más de 100 MB y cada versión quedaría para siempre en el historial. Para no reentrenar, [`src/store.py`](src/store.py) los sube y baja de un repo de Hugging Face o de una carpeta compartida (por ejemplo, Google Drive), y `results/checkpoints.json` (en git) lista cada modelo con su configuración, tokens y sha256.

```bash
.venv/bin/python -m src.store list                                   # qué modelos hay y dónde
.venv/bin/python -m src.store pull --all --hf <usuario>/<repo>       # o --dir /content/drive/MyDrive/ml2/store
.venv/bin/python -m src.store push --all --hf <usuario>/<repo> [--create] [--rollouts]
```

`load_model("gpt_s_kd")` baja el checkpoint solo si no está local (con `ML2_HF_REPO` o `ML2_STORE_DIR` configurados, o el remoto anotado en el manifiesto) y verifica el sha256. Con `--rollouts` también se comparten los rollouts, y `python -m src.metrics_video --name X --from-cache` recalcula las métricas sin decodificar.

## Tests

```bash
.venv/bin/python -m pytest tests -q
```

## Slides

`pre-entrega/index.html`: presentación de avance. Se navega con las flechas.

`pre-entrega/generar_imagenes.py` regenera las imágenes de las animaciones (`pre-entrega/img/anim_*.jpg`, `race_real.jpg`) e imprime los tokens y probabilidades que usa el HTML. Necesita `data/` y `third_party/commavq`.
