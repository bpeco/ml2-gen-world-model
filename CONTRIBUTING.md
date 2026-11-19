# Cómo contribuir

Este repo compara world models de manejo con una regla central: **cada modelo es fiel a su fuente, y todo resultado sale de la CLI**. Antes de tocar nada, leé [`Docs/uso.md`](Docs/uso.md) (el framework) y [`Docs/resumen.md`](Docs/resumen.md) (qué se hizo y por qué). Las guías de cada modelo están en [`Docs/modelos/`](Docs/modelos/README.md).

Todos los comandos usan `python -m ...` con el Python del entorno virtual del repo (`.venv\Scripts\python` en Windows, `.venv/bin/python` en Linux y Mac); ver [la instalación por sistema](Docs/uso.md#instalación-y-uso-por-sistema).

## Reglas

### Regla de fidelidad

Ningún modelo se adapta sin declararlo. Cada decisión de diseño o de entrenamiento va en la lista `FIDELIDAD` del módulo del modelo, con el valor de la fuente, el valor de acá, la categoría y la cita (paper, sección o tabla, archivo y línea del código oficial):

| Categoría | Cuándo |
|---|---|
| `fiel` | es lo que dice la fuente |
| `datos` | lo obliga el dato de comma (resolución, tokens VQ, 20 FPS, poses) |
| `presupuesto` | lo obliga el cómputo (batch, pasos, tamaño) |
| `propio` | la fuente no lo fija y lo elegimos nosotros; un diseño propio se nombra como propio |

Los desvíos que cambian un resultado principal se consultan antes con el equipo. Cada modelo declara su receta con `recipe()` (`src/recipes.py`, `src/schedules.py`): la de su paper, escalada al presupuesto común y con lo forzado marcado. Ejemplos: [`src/models/genie.py`](src/models/genie.py) y [`src/models/lewm.py`](src/models/lewm.py).

### Regla de ejecución

Todo resultado que entra al repo sale de `python -m src.cli ...` o de la cola (`python -m src.cli cola correr`), con el árbol limpio y su entrada en `results/bitacora.jsonl`. Sin scripts sueltos, sin celdas sueltas y sin parches a mano durante una corrida.

- Un hiperparámetro se cambia con `--set` (queda como desvío) o con un commit a la receta; nunca editando un archivo a mano.
- Un bug se arregla por el camino normal: un test que lo reproduce, el arreglo, `python -m src.cli verificar` en verde y un commit; después se repite el mismo comando con `python -m src.cli bitacora repetir <id>`.
- Las acciones de GPU no corren con cambios sin commitear en `src/` o `scripts/`; `--permitir-sucio` deja pasar la acción, pero el resultado no cuenta como `vigente`.
- Los checkpoints de la tabla y sus decoders se comparten por Hugging Face: la CLI los sube al terminar de entrenar (con `hf auth login`) y los baja cuando faltan ([`Docs/uso.md`](Docs/uso.md#modelos-entrenados-hugging-face)). Nunca entran a git; los humos, `last.pt` y `pre_decay.pt` quedan locales.

### Reglas de experimentos

- Los hiperparámetros se eligen en el holdout (`results/clips_holdout.json`), nunca en val.
- El presupuesto y la semilla se declaran: 780.000 frames vistos por modelo, con el batch de la fuente (escalado si no entra en 12 GB).
- Las latencias se miden con la GPU libre y con el protocolo único (batch 1, p50 y p99).
- Cada resultado lleva su estado (`vigente`, `protocolo anterior` o `pendiente`; `python -m src.cli resultados`), el piso de FVD y los intervalos por segmento.
- Los docs no llevan números escritos a mano: lo que se muestra sale de un JSON de `results/` o dice "pendiente" (lo controla `scripts/check_docs.py`).

## Sumar un modelo

La CLI, la cola, la evaluación, el reporte y los notebooks toman el modelo del registro: no hay que tocar nada más.

1. **Copiá la plantilla.** `src/models/_plantilla.py` es un modelo mínimo pero real (entrena en CPU en segundos). Copiala a `src/models/<tu_modelo>.py` y copiá su test (`tests/test_<tu_modelo>.py`, con el patrón de [`tests/test_gpt.py`](tests/test_gpt.py)). Cada modelo tiene su propio código: no se comparte arquitectura entre modelos distintos; lo compartido es infraestructura.
2. **Implementá** `train_loss`, la generación o el stream (o `encode` y `predict` para los latentes, ver `src/models/latent.py`), la receta `recipe()`, `ACCIONES` y `EVALUACION` (`"video"` o `"latente"`). La interfaz está en `src/models/base.py`.
3. **Declará la fidelidad.** `FIDELIDAD` con la cita de la fuente en cada fila (regla de fidelidad).
4. **Declará `CONTRATO`:** la configuración mínima con la que corren los tests de contrato (como en la plantilla).
5. **Registralo** en `KINDS` de `src/models/__init__.py`.
6. **Verificá.** `python -m pytest tests/test_contrato.py tests/test_<tu_modelo>.py` y después `python -m src.cli verificar`. El contrato exige receta, fidelidad con fuente, que no filtre el futuro, que sobreajuste un batch, que reanude bit a bit desde `last.pt`, que el stream dé lo mismo que `generate` y que cada acción declarada dé un plan válido con `--dry-run` ([`tests/test_contrato.py`](tests/test_contrato.py)). Un modelo sin receta o sin fidelidad falla con un mensaje.
7. **Escribí la guía** `Docs/modelos/<tu_modelo>.md` (copiá la de la plantilla, [`Docs/modelos/plantilla.md`](Docs/modelos/plantilla.md)) y regenerá sus tablas: `python -m src.docs_gen`. `python -m src.docs_gen --check` tiene que dar 0 diferencias. Regenerá también el notebook del modelo.
8. **Sumá sus corridas** a `scripts/queue.toml`, como llamadas a la CLI, y revisá la cola con `python -m src.cli cola correr --dry-run`.
9. **Sumá su fila** a la tabla de [`Docs/resultados.md`](Docs/resultados.md), con sus valores en "pendiente" hasta que haya corrida.

## Sumar una evaluación

Una función con el decorador `@evaluador` de [`src/evaluadores.py`](src/evaluadores.py), que devuelve un valor por clip:

```python
from src.evaluadores import evaluador

@evaluador("nitidez", aplica_a=("video",), menor_es_mejor=False, sintetico=mis_datos, referencia=mi_referencia)
def nitidez(pred, real):
  ...   # (N,) una medida por clip
```

Cuatro piezas son obligatorias: la función, datos sintéticos (`sintetico(rng) -> (pred, real, seg)`), una referencia trivial (`referencia(real)`: copiar el último frame, la media) y un intervalo de confianza por segmento (por defecto `src.estadistica.ic_media_segmento`). `evaluadores.verificar(nombre)` comprueba que el modelo sintético le gane a la referencia. Un test con datos sintéticos va en `tests/`. La evaluación aparece en la CLI y en el reporte sin tocar el núcleo (`evaluadores.para_modelo(kind)` y `evaluadores.tabla()`).

## Sumar un backend

Una entrada en [`src/infer/backends.py`](src/infer/backends.py) con el decorador `@backend`: el nombre (minúsculas, sin espacios, es el tag de dispositivo), cómo se detecta y si corre con torch (`torch_device`), con ONNX Runtime (`ort_provider`) o con los dos.

```python
from src.infer import backends

@backends.backend("mi_gpu", descripcion="...", torch_device="xpu", instalar="pip install ...")
def _detectar_mi_gpu():
  return hay_mi_gpu(), "motivo si no hay"
```

Todos los `--device` del repo salen de ese registro: el backend nuevo aparece en `bench`, en el notebook de inferencia y en el reporte por dispositivo. `backends.disponibles()` dice cuáles se pueden usar en la máquina.

## Nombres

- **Corridas:** `<modelo>_<tamaño>[_<variante>][_s<semilla>][_x<k>]`. Por ejemplo `genie_s`, `gpt_s_kd`, `genie_s_s1`, `vjepa2_s_x1`. Los modelos sin tamaño llevan solo el modelo (`lewm`, `convlstm_px`).
- **Resultados:** `<tipo>_<run>[__<variante>].json` (`eval_`, `video_`, `latent_`, `train_`...). Con `--pasos`, el run lleva `_p<pasos>` (`genie_s_p8`).
- **Dispositivos:** tags en minúsculas y sin espacios (`cuda`, `rocm`, `dml`, `npu`, `cpu`).
- **Código y docs:** en español, indentación de 2 espacios; los comandos de los docs son `python -m ...`.

## Checklist de PR

- [ ] `python -m src.cli verificar` en verde (pytest, bitácora, docs y notebooks).
- [ ] La fidelidad está completa: cada decisión con su fuente y su categoría; los desvíos que cambian un resultado principal, consultados.
- [ ] La guía y el notebook están regenerados (`python -m src.docs_gen --check` da 0 diferencias).
- [ ] Los resultados tienen su intervalo por segmento, su referencia y su entrada en la bitácora, con el árbol limpio.
- [ ] No quedan huérfanos en `results/` ni en `checkpoints.json`.
- [ ] Los docs no tienen números escritos a mano: `python scripts/check_docs.py` da 0 problemas.
- [ ] Ningún checkpoint ni dato pesado entró al commit: los modelos van a Hugging Face.
