# Plan de implementación y validación

Plan operativo para ejecutar [`plan-proyecto.md`](plan-proyecto.md) completo en una sola máquina:
RTX 3080 Ti (12 GB, Ampere), Ryzen 9 5900X, 32 GB de RAM, Windows 11, torch 2.11 + CUDA 12.8.
**Estado: ejecutado.** Resultados y conclusiones: [`resultados.md`](resultados.md).

La GPU de referencia del plan (D1) es la T4 de Colab. Acá todas las latencias se midieron en la 3080 Ti y quedan
anotadas en cada `results/*.json` (`device_info`). El notebook de Colab permite repetir `bench.py` en una T4.

## Contrato común

- `src/common.py`: `device()` elige CUDA (NVIDIA o ROCm), MPS o CPU; `ML2_DEVICE` lo fuerza. `device_info()` va a cada JSON.
- `src/models/base.py`: `WorldModel` con `logits`, `last_frame_logits`, `next_frame`, `generate`, `rollout` y `train_loss`.
  `FrameModel` para los modelos que predicen las 128 celdas independientes (baselines, ConvLSTM).
  `lm_loss` con la loss de destilación `α·CE + (1 − α)·τ²·KL(maestro ‖ alumno)`.
- `src/models/__init__.py`: `load_model("gpt2m" | "unigram" | "copy" | "random:gpt_s" | "<corrida>" | ruta.pt)` y tamaños S, M, L.

Métrica extra para comparar familias: `ce_last`, la CE del último frame de cada ventana del protocolo de comma.
Para un GPT es con teacher forcing; para MaskGIT es la "CE con 1 paso" (frame entero enmascarado).

## Implementación

El código lo escribieron agentes en paralelo, cada uno con sus propios archivos, contra el contrato común. Las validaciones de abajo las corrieron los agentes y después las repetí y amplié yo.

| Paso | Archivos | Validación |
|---|---|---|
| 0 | `common.py`, `models/base.py`, `models/__init__.py` | Tests de interfaz: muestreo, KD con maestro = alumno da KL 0, cambio de layout ida y vuelta |
| 1a | `models/baselines.py`, `models/gpt2m.py`, `evaluate.py` genérico | Unigrama 6,8455 (slides: 6,85); copia 3,4797 (slides: 3,48); gpt2m 2,012030 en 2 segmentos (guardado: 2,012030) y 2,0257 en los 100 |
| 1b | `models/backbone.py`, `models/gpt.py`, `models/cudagraph.py` | KV cache igual al forward completo; sin fugas del futuro (diferencia exacta 0); CUDA graphs dan los mismos tokens que eager en 18 combinaciones de tamaño, dtype y batch |
| 1c | `rollouts.py`, `metrics_video.py` | Real contra real: LPIPS 0 y FVD 0; real < frame congelado < unigrama; el frame congelado coincide 100 % con el último frame de contexto |
| 1d | `windows.py`, `train.py`, `models/convlstm.py` | Sobreajuste de un batch; resume reproduce los mismos pesos; KD con α = 1 es igual a CE |
| 2a | `models/maskgit.py` | Máscara por bloques sin fugas; exactamente S pasadas por frame; 1 paso = argmax de `last_frame_logits` |
| 2b | `bench.py`, `report.py`, `scripts/analysis.py` | Eventos CUDA, 20 de warm-up y mediana de 100; Pareto y Spearman con casos hechos a mano |
| 2c | `export_onnx.py`, `ort_model.py`, `notebooks/colab.ipynb`, `requirements-*.txt` | ONNX = PyTorch en CPU (100 % de argmax iguales); el notebook valida y usa solo flags que existen |
| 3 | `store.py` | push → borrar → pull con sha256 verificado; `load_model("<corrida>")` baja el checkpoint si falta |

88 tests en `tests/` (`python -m pytest tests -q`).

## Cambios durante la ejecución

| Qué | Por qué |
|---|---|
| Baseline `copy` = tabla de transición por celda | Es lo que reproduce el 3,48 de las slides; la mezcla literal da 4,89 y queda como `copy_mix` (frame congelado a temperatura 0) |
| Embedding de posición factorizado (`pos="frame"`) | Con el absoluto, la CE del GPT sube de 2,96 a 3,52 en el último frame de la ventana, que es donde genera |
| MaskGIT con `p_clean = 0,5` | Sin contexto limpio al entrenar, colapsa con más de ~12 frames de contexto (FVD 3295) |
| Intervalo de FVD normal (± 1,96 desvíos bootstrap) | El intervalo percentil y el "básico" quedan corridos por el sesgo del bootstrap del FVD |
| Caché y máscara alineadas a múltiplos de 8 (2584) | Precaución tras un acceso ilegal a memoria en SDPA que no se pudo reproducir; no volvió a aparecer |
| Almacén de modelos (`src/store.py`) | Pedido durante la ejecución: no reentrenar ni regenerar rollouts |
| Extras: ConvLSTM con 1B tokens y control del LR de M | Para saber si la brecha con gpt2m se cierra con más cómputo y si M ≈ S era un problema de LR |

## Validaciones propias

1. El evaluador reproduce el 2,012 de gpt2m en 2 segmentos y el 2,026 en los 100. ✔
2. KV cache, CUDA graphs y fp16 no cambian los tokens ni la CE más allá del error numérico (≤ 1,4e-5). ✔
3. Las máscaras no filtran información del futuro (test de perturbación con diferencia exactamente 0). ✔
4. H1 es justa: mismos batches (dependen solo de semilla y paso), misma inicialización, LR y tokens. ✔
5. El protocolo de video es el de la sección 2.4: 400 clips fijos, temperatura 1, semillas fijas, real decodificado. ✔
6. Las latencias se midieron con la GPU sin otros trabajos, batch 1, warm-up y mediana, y anotan GPU y TF32. ✔
7. Las conclusiones de H1–H4 salen de tests pareados con IC (`results/hypotheses.md`). ✔
