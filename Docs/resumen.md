# Resumen de lo que se hizo

Pregunta del proyecto: **¿cuánto se puede achicar y acelerar un world model de video de manejo sin perder calidad?** Se trabaja sobre [commaVQ](https://github.com/commaai/commavq): video de conducción a 20 FPS, tokenizado por un VQ-VAE en 128 tokens por frame (grilla de 8 x 16, vocabulario de 1024), con las poses del auto. La referencia es el GPT de comma (307M). Este documento cuenta las etapas, las decisiones, qué se reemplazó y por qué, y dónde está cada resultado. No tiene números: los números están en [`resultados.md`](resultados.md), y cada uno sale de un JSON de `results/`.

## Etapas

1. **Primera versión.** Un GPT alumno (con destilación de `gpt2m`), un MaskGIT propio, un ConvLSTM sobre tokens y dos modelos latentes (JEPA y LeWM sobre tokens), todos con una receta de entrenamiento común, una sola semilla y una evaluación de video (CE, LPIPS y FVD) más un bench de latencia y una cadena de inferencia de punta a punta.
2. **Revisión de fidelidad.** Se encontró que varios modelos no eran lo que decía su fuente, y que algunas conclusiones dependían de eso (la lista está más abajo). También se encontraron problemas en la evaluación.
3. **Reconstrucción, sin GPU.** Se limpió lo que no sobrevive, y se construyó el framework: recetas por modelo, `train.py` genérico, una CLI única, la cola de GPU, la bitácora, el seguimiento, el estado de cada resultado, la estadística de la evaluación, los modelos fieles, las guías, los notebooks y la guía de contribución. Todo con tests en CPU.
4. **Cola de GPU.** Cada modelo se entrena y se evalúa solo con `python -m src.cli` o con la cola (`scripts/queue.toml`), y cada corrida queda en la bitácora con su comando exacto. Es la etapa que llena la tabla de resultados.
5. **Al final:** las corridas 10x de `vjepa2` y `lewm`, solo con todo lo anterior verificado.

Hasta que la cola termine, todo resultado figura como `pendiente` en [`resultados.md`](resultados.md); el código es el definitivo.

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
| Los checkpoints | `checkpoints/<run>/`, locales: no se distribuyen. Cada uno entrena lo que necesita; las corridas se registran en `results/checkpoints.json` |
| Cómo reentrenar un modelo | [`modelos/README.md`](modelos/README.md) y [`uso.md`](uso.md) |
| Los papers del camino de la investigación | [`world-models-conduccion-papers.md`](world-models-conduccion-papers.md) |
| La presentación de avance | [`../pre-entrega/index.html`](../pre-entrega/index.html) |

## Qué falta

- Correr la cola de GPU y pasar los resultados a `vigente`.
- Escribir las conclusiones y re-evaluar las hipótesis con los datos nuevos: hoy figuran como "a evaluar".
- Medir la inferencia en la T4 y en la laptop (iGPU y NPU).
- Las corridas 10x de `vjepa2` y `lewm`.
