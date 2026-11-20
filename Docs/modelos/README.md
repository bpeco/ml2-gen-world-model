# Guías por modelo

Una guía por modelo, para reentrenarlo, evaluarlo, medirlo y mostrarlo con la CLI (`python -m src.cli <modelo> <acción>`). La guía general del framework (CLI, cola, bitácora, seguimiento, dispositivos) está en [`../uso.md`](../uso.md); lo que se hizo y por qué, en [`../resumen.md`](../resumen.md); las reglas para sumar algo, en [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md).

## Modelos

Esta tabla sale del código (`python -m src.docs_gen`): lista lo que declara cada modelo en `ACCIONES` y `EVALUACION`.

<!-- auto:modelos -->
| modelo | evaluación | acciones | guía |
|---|---|---|---|
| `gpt` | video | receta, entrenar, continuar, evaluar, bench, demo, estado | [gpt.md](gpt.md) |
| `maskgit` | video | receta, entrenar, continuar, evaluar, bench, demo, exportar, estado | [maskgit.md](maskgit.md) |
| `lewm` | latente | receta, entrenar, continuar, evaluar, bench, demo, exportar, estado | [lewm.md](lewm.md) |
| `genie` | video | receta, entrenar, continuar, evaluar, bench, demo, exportar, estado | [genie.md](genie.md) |
| `vjepa2` | latente | receta, entrenar, continuar, evaluar, bench, demo, exportar, estado | [vjepa2.md](vjepa2.md) |
| `vjepa2_ac` | latente | receta, entrenar, continuar, evaluar, bench, demo, exportar, estado | [vjepa2_ac.md](vjepa2_ac.md) |
| `convlstm_px` | video | receta, entrenar, continuar, evaluar, bench, demo, estado | [convlstm_px.md](convlstm_px.md) |
| `plantilla` | video | receta, entrenar, continuar, evaluar, bench, demo, estado | [plantilla.md](plantilla.md) |
| `gpt2m` | video | evaluar, bench | [referencias.md](referencias.md) |
| `unigram` | video | evaluar, bench | [referencias.md](referencias.md) |
| `transicion` | video | evaluar, bench | [referencias.md](referencias.md) |
| `congelado` | video | evaluar, bench | [referencias.md](referencias.md) |
| `real` | video | evaluar | [referencias.md](referencias.md) |
| `codec` | video | bench, exportar | [referencias.md](referencias.md) |
<!-- /auto -->

Los kinds entrenables (`gpt`, `genie`, `maskgit`, `convlstm_px`, `lewm`, `vjepa2`, `vjepa2_ac`) tienen guía propia. `gpt2m`, los baselines (`unigram`, `transicion`, `congelado`), `real` y `codec` están juntos en [`referencias.md`](referencias.md). [`plantilla.md`](plantilla.md) explica el modelo mínimo que se copia al sumar uno nuevo.

| Guía | Qué es | Fuente |
|---|---|---|
| [`gpt.md`](gpt.md) | GPT alumno autoregresivo, con destilación opcional del maestro `gpt2m` | commaVQ e IRIS |
| [`genie.md`](genie.md) | Dinámica de Genie decodificada con MaskGIT: el MaskGIT de video fiel | Genie (arXiv 2402.15391) |
| [`maskgit.md`](maskgit.md) | MaskGIT propio con sus ablaciones (opcional, fuera de la cola) | MaskGIT (arXiv 2202.04200), diseño propio |
| [`convlstm_px.md`](convlstm_px.md) | ConvLSTM de Shi sobre píxeles | Shi et al. 2015 (arXiv 1506.04214) |
| [`lewm.md`](lewm.md) | LeWorldModel desde píxeles, con SIGReg | LeWM (arXiv 2603.19312) |
| [`vjepa2.md`](vjepa2.md) | V-JEPA 2: preentrenamiento autosupervisado | arXiv 2506.09985 |
| [`vjepa2_ac.md`](vjepa2_ac.md) | V-JEPA 2-AC: predictor con acciones sobre un encoder congelado | arXiv 2506.09985 |
| [`referencias.md`](referencias.md) | `gpt2m`, baselines, `real` (piso de FVD) y `codec` | commaVQ y el propio repo |

## Cómo se lee una guía

Todas siguen el mismo orden:

1. **Qué es y su fuente.**
2. **Fidelidad:** una fila por decisión, con la categoría `fiel`, `datos` (lo obliga el dato de comma), `presupuesto` (lo obliga el cómputo) o `propio` (la fuente no lo fija). Es la tabla `FIDELIDAD` del módulo del modelo.
3. **Receta de entrenamiento:** la de la fuente, escalada al presupuesto común.
4. **Una subsección por acción,** con sus opciones y un ejemplo para copiar.
5. **Qué archivos escribe.**
6. **Problemas frecuentes.**
7. **Notebook** del modelo.

Las tablas de fidelidad, las recetas, las acciones y las opciones se generan con `python -m src.docs_gen` y se verifican con `python -m src.docs_gen --check`: no se editan a mano (todo lo que está entre un marcador `auto` de apertura y el de cierre se reescribe). El texto y los ejemplos son manuales y los prueba `python scripts/check_docs.py` con `--dry-run`.

## Opciones comunes

Valen para todas las acciones de todos los modelos (las propias de cada acción están en su subsección).

<!-- auto:opciones_comunes -->
| opción | qué hace |
|---|---|
| `--run` | nombre de la corrida (default: <modelo>_<tamaño>[_<variante>][_s<semilla>]) |
| `--tamaño`, `--tamano` | tamaño del modelo (s, m, l; vjepa2: t o s) |
| `--semilla` | semilla de entrenamiento (la de la corrida a evaluar si no es 0) |
| `--variante` | sufijo del nombre de la corrida (_<variante>) |
| `--device` | cuda, cpu, rocm, dml, npu o mps (Apple Silicon); default: el que elige el repo |
| `--limite` | recorta la acción: pasos (entrenar), segmentos/clips (evaluar), frames (bench, demo) |
| `--dry-run` | imprime los comandos que ejecutaría y no corre nada |
| `--esperar` | si la GPU está ocupada, espera en vez de encolar |
| `--encolar` | no correr ahora: dejar la acción en la cola |
| `--permitir-sucio` | ya no hace falta: una acción de GPU corre con cambios sin commitear (queda anotado; no cuenta como vigente) |
<!-- /auto -->

`--set` acepta `receta.<campo>=<valor>` y `config.<campo>=<valor>`; el campo se valida contra el dataclass antes de lanzar nada y queda en la bitácora como desvío de la fuente. Lo que viene después de un `--` pasa tal cual al script que envuelve la acción.

## Convenciones

- **Presupuesto común:** 780.000 frames vistos (100M de tokens a 129 tokens por frame) para todos los modelos, con el batch de cada fuente (escalado si no entra en 12 GB). `--presupuesto 780000` o `--presupuesto 10x` (10 veces el de la receta).
- **Nombres de corrida:** `<modelo>_<tamaño>[_<variante>][_s<semilla>][_x<k>]`. Por ejemplo `genie_s`, `gpt_s_kd` (variante `kd`), `genie_s_s1` (semilla 1) y `vjepa2_s_x1` (primera extensión). Los modelos sin tamaño llevan solo el modelo (`lewm`, `convlstm_px`).
- **Una corrida no se pisa:** `entrenar` se niega si ya existe `checkpoints/<run>/last.pt` o `final.pt`; para retomarla o extenderla está `continuar`.
- **Una sola acción de GPU a la vez:** si la GPU está ocupada la acción queda en la cola (`results/queue.json`) o espera con `--esperar`.
- **Árbol limpio:** una acción de GPU corre aunque haya cambios sin commitear en `src/` o `scripts/`, pero el resultado no cuenta como vigente.
- **Sin GPU:** `--device cpu` o `CUDA_VISIBLE_DEVICES=-1` corren todo en CPU, sirve para probar con `--limite`.
- **`--dry-run`:** imprime los comandos que ejecutaría y no corre nada (los ejemplos de las guías se prueban así).
