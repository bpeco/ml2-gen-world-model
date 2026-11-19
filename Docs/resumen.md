# Resumen de lo que se hizo

Pregunta del proyecto: **¿cuánto se puede achicar y acelerar un world model de video de manejo sin perder calidad?** Se trabaja sobre [commaVQ](https://github.com/commaai/commavq): video de conducción a 20 FPS, tokenizado por un VQ-VAE en 128 tokens por frame (grilla de 8 x 16, vocabulario de 1024), con las poses del auto. La referencia es el GPT de comma (307M). Este documento cuenta las etapas, las decisiones, qué se reemplazó y por qué, y dónde está cada resultado. No tiene números: los números están en [`resultados.md`](resultados.md), y cada uno sale de un JSON de `results/`.

## Etapas

1. **Primera versión.** Un GPT alumno (con destilación de `gpt2m`), un MaskGIT propio, un ConvLSTM sobre tokens y dos modelos latentes (JEPA y LeWM sobre tokens), todos con una receta de entrenamiento común, una sola semilla y una evaluación de video (CE, LPIPS y FVD) más un bench de latencia y una cadena de inferencia de punta a punta.
2. **Revisión de fidelidad.** Se encontró que varios modelos no eran lo que decía su fuente, y que algunas conclusiones dependían de eso (la lista está más abajo). También se encontraron problemas en la evaluación.
3. **Reconstrucción, sin GPU.** Se limpió lo que no sobrevive, y se construyó el framework: recetas por modelo, `train.py` genérico, una CLI única, la cola de GPU, la bitácora, el seguimiento, el estado de cada resultado, la estadística de la evaluación, los modelos fieles, las guías, los notebooks y la guía de contribución. Todo con tests en CPU.
4. **Cola de GPU.** Cada modelo se entrena y se evalúa solo con `python -m src.cli` o con la cola (`scripts/queue.toml`), y cada corrida queda en la bitácora con su comando exacto. Es la etapa que llena la tabla de resultados.
5. **Al final:** las corridas 10x de `vjepa2` y `lewm`, solo con todo lo anterior verificado.

La cola de GPU terminó (trabajos 0 a 16): todos los resultados de [`resultados.md`](resultados.md) son `vigente`, con su comando en la bitácora. Las corridas 10× y los trabajos opcionales no se hicieron.

## Decisiones

- **Fidelidad (D1).** Ningún modelo se adapta sin declararlo. Cada modelo tiene una tabla de fidelidad: una fila por decisión, con la fuente y una categoría: `fiel`, `datos` (lo obliga el dato de comma), `presupuesto` (lo obliga el cómputo) o `propio`.
- **Ejecución (D2).** Todo modelo, evaluación, benchmark y test se corre con las herramientas del usuario: la CLI y la cola. Nada de editar código sobre la marcha ni de scripts sueltos. Un bug se arregla con código, test y commit, y se repite el mismo comando.
- **Receta de la fuente.** Cada modelo entrena con la receta de su paper (optimizador, lr y forma del schedule, weight decay, EMA, clip, precisión y batch de la fuente). Lo que la fuente no publica se elige y se declara como `propio`.
- **Presupuesto común.** 780.000 frames vistos por modelo (100M de tokens) con un batch chico, de modo que cada uno hace del orden de 1.500 pasos o más. Si el batch de la fuente no entra en 12 GB se acumula, y el batch efectivo queda declarado. El humo (`entrenar --humo`) elige el lr solo donde cambió el batch, y la elección entra a la receta con un commit.
- **Cada modelo tiene su código.** El modelo, su receta, su stream de inferencia y su export viven junto a él; ajustar uno no mueve a otro. Lo compartido es infraestructura: datos, CLI, cola, bitácora, métricas y reporte.
- **Todo residente en la GPU.** Sin copias entre CPU y GPU por paso, en entrenamiento y en inferencia. Los batches dependen solo de (semilla, paso).
- **Evaluación (D14).** Bootstrap por segmento (los 4 clips de un segmento no son independientes), bootstrap pareado de diferencias de FVD, piso de FVD en cada tabla (real contra real, 400 contra 400), temperatura elegida por familia en el holdout (nunca en val) y tres semillas para los modelos que se comparan.
- **Latentes aparte (D12).** Los modelos latentes no se miden con FVD a través de una lectura a tokens. Tienen su tabla: error contra copiar el último latente, probes contra un encoder sin entrenar y contra píxeles, sorpresa, recuperación de acciones con CEM y latencia por paso.
- **Un resultado es una fila.** La tabla de resultados es modelo → performance, con una fila por modelo. Otro dato es otro modelo o variante que convive con los demás. Cada resultado tiene un estado: `vigente`, `protocolo anterior` o `pendiente`.
- **Solo lo final en el repo.** Los planes de trabajo (`plan-*.md`) quedan en el historial de git; en el repo quedan este resumen, la [guía de uso](uso.md), las [guías por modelo](modelos/README.md) y [`../CONTRIBUTING.md`](../CONTRIBUTING.md).

## Qué se reemplazó y por qué

| Antes | Ahora | Por qué |
|---|---|---|
| JEPA sobre tokens | **`vjepa2`** (V-JEPA 2) | nunca fue V-JEPA: no tenía tubelets, ni máscaras multibloque, ni target EMA con L1, ni la receta del paper |
| LeWM sobre tokens (y `lewm_px`) | **`lewm`**, reescrito según el repo oficial desde píxeles | `lewm_px` solo había arreglado la entrada; el predictor, la historia, el frameskip, el batch, SIGReg y el decoder diferían del oficial. Las 12 corridas latentes de tokens se borraron |
| ConvLSTM sobre tokens con cabeza de cross-entropy | **`convlstm_px`**, el de Shi et al. 2015 sobre píxeles | el anterior ganaba el frente de Pareto pero no era Shi et al.; se borró entero, con su código, corridas, stream, export y tests |
| MaskGIT propio (`choice_temp` 1,0, `p_clean`, posición factorizada) | **`genie`**: la dinámica de Genie con decodificación MaskGIT | decisión del usuario: el MaskGIT de video fiel es el de Genie. El `maskgit` propio queda como ablación declarada |
| GPT con posición factorizada y receta común | **`gpt`** con el diseño de comma escalado (posición absoluta) y la receta de IRIS | la posición factorizada pasa a ser una ablación propia |
| Una receta de entrenamiento para todos | una receta por modelo, de su fuente | comparar arquitecturas con la receta de cada paper, no con una receta común |
| `copy` y `copy_mix` | `transicion` y `congelado` (con alias) | nombres que dicen lo que hacen |
| `.sh` con la lógica de las corridas | `python -m src.cli` y `scripts/queue.toml` | un solo punto de entrada, con bitácora |

## Modelos y cola

| Modelo | Qué es | Guía |
|---|---|---|
| `gpt2m`, `unigram`, `transicion`, `congelado`, `real` | referencias; `real` es el piso de FVD (400 contra 400) | [`referencias.md`](modelos/referencias.md) |
| `gpt` (`gpt_s_kd`) | GPT-S destilado, receta de IRIS | [`gpt.md`](modelos/gpt.md) |
| `genie` (`genie_s`) | MaskGIT fiel (Genie), con 1, 2, 4, 8, 12 y 25 pasadas | [`genie.md`](modelos/genie.md) |
| `convlstm_px` | ConvLSTM de Shi sobre píxeles | [`convlstm_px.md`](modelos/convlstm_px.md) |
| `lewm`, `vjepa2`, `vjepa2_ac` | modelos latentes | [`lewm.md`](modelos/lewm.md), [`vjepa2.md`](modelos/vjepa2.md), [`vjepa2_ac.md`](modelos/vjepa2_ac.md) |
| `gpt_s`, `gpt_m`, `genie_m`, `maskgit_s` | implementados y entrenables, fuera de la cola (opcionales) | [`gpt.md`](modelos/gpt.md), [`genie.md`](modelos/genie.md), [`maskgit.md`](modelos/maskgit.md) |

La cola incluye además las semillas 1 y 2 de `genie_s` y `convlstm_px`. Las corridas 10x de `vjepa2` y `lewm` van al final.

## Dónde está cada resultado

| Qué | Dónde |
|---|---|
| La tabla modelo → performance, y su estado | [`resultados.md`](resultados.md) |
| JSON de cada medición (CE, video, latentes, latencia, cadena, temperatura, humo) | `results/` (`eval_*`, `video_*`, `latent_*`, `bench/`, `infer/`, `temperatura_*`, `humo_*`, `train_*`) |
| El comando y el commit que produjo cada resultado | dentro de cada JSON y en `results/bitacora.jsonl` |
| Estado de cada archivo de resultados | `results/estado.json` (`python -m src.cli resultados`) |
| Tablas y figuras generadas | `python -m src.cli reporte` (`results/summary.md` y figuras) |
| Los checkpoints | `checkpoints/<run>/` y el repo público de Hugging Face [`KevinSoulier/ml2-gen-world-model`](https://huggingface.co/KevinSoulier/ml2-gen-world-model): el código baja solo lo que falte y verifica el sha256 de `results/checkpoints.json` ([Modelos entrenados](uso.md#modelos-entrenados-hugging-face)) |
| Cómo reentrenar un modelo | [`modelos/README.md`](modelos/README.md) y [`uso.md`](uso.md) |
| Los papers del camino de la investigación | [`world-models-conduccion-papers.md`](world-models-conduccion-papers.md) |
| La presentación de avance | [`../pre-entrega/index.html`](../pre-entrega/index.html) |

## Lo que encontró correr todo por la CLI

Toda la cola (trabajos 0 a 16) corrió con `python -m src.cli` y quedó en la bitácora. Cada vez que algo falló o dio un resultado raro, el arreglo fue cambio de código, test que lo reproduce y commit. Después se repitió el mismo comando con `bitacora repetir`. Estas fueron las fricciones:

| fricción | arreglo | commit |
|---|---|---|
| El monitor del entrenamiento usaba ventanas de un largo distinto al de la receta, y el humo de Genie hubo que repetirlo | el monitor usa el largo de la receta | `2d562f0` |
| El humo aceptaba solo algunos múltiplos del lr, y el óptimo quedaba en el borde de la grilla | `--humo-candidatos` acepta cualquier `x<k>` | `9251f61` |
| `cola correr --dry-run` agregaba `--dry-run` al final de cada comando: con un `--` en el comando iba al script y la acción corría de verdad. Además faltaba un paso para pasar a `vigente` lo que cerraba cada bloque | el `--dry-run` va antes del `--`; `bitacora vigentes` cierra cada bloque | `6efc402` |
| El decoder de visualización del trabajo 12 (`vjepa2_ac demo --entrenar-decoder`) se quedó sin memoria: el historial de la loss retenía el buffer del MSE, unos 24 MiB por paso | el historial es un buffer en el device | `3211f38` |
| La bitácora le atribuía a una acción larga resultados que había escrito otra, concurrente, o que `vigentes` solo había re-sellado | una acción no reclama lo de otra; tocar solo el sello no es escribir; manda el sello del JSON | `5405b87`, `ec563b9` |
| `vjepa2 bench` caía en el stream de tokens (V-JEPA 2 sin acciones no tiene stream) | mide el encoder sobre un clip | `8d866e0` |
| La estadística, el piso, la temperatura y el humo no tenían estado: `bitacora verificar` no los revisaba y los docs no podían citarlos | entran en el estado de resultados | `8623b9a` |
| La cadena de Genie con 4 pasadas se guardaba como `genie_s_s4`, que se lee como la semilla 4, y el reporte no la encontraba | las pasadas se nombran `_p<n>`; los archivos viejos se apartaron con `resultados --quitar` | `0d80802`, `4df35c8`, `a0ecc4d` |
| `bench --device npu` pasaba el dry-run y fallaba al correr | falla al planear y dice cómo medir la cadena del codec con ONNX Runtime | `4df35c8` |
| Las figuras del reporte eran de las familias viejas; el destilado no figuraba como autorregresivo; la correlación de H4 se informaba con 3 puntos | figuras con Genie y ConvLSTM, el destilado autorregresivo, correlación solo con 5 puntos o más | `4df35c8`, `3881107` |
| Compartir los modelos entrenados: Git LFS gastaba la cuota del dueño del repo | Hugging Face, con bajada sin login y subida automática al terminar de entrenar | `a957766`, `b3e602f` |

## Qué falta

- **Trabajos opcionales de la cola (17 a 20):** los GPT de cero, que cierran H1 y H4; `genie_m`; y el MaskGIT propio con sus ablaciones.
- **H3:** medir los transformers en fp32 y comparar calidad y latencia con fp16 y CUDA graphs.
- **Otros dispositivos:** medir la inferencia en CPU (todas las latencias de `results/bench/` y `results/infer/` son de la RTX 3080 Ti), en la T4 y en la laptop (iGPU y NPU).
- **Corridas 10×** de `vjepa2` y `lewm`.
