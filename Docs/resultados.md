# Resultados: modelo → performance

> **Resultados en re-ejecución; el código es el definitivo.** Los modelos se reconstruyeron para ser fieles a sus fuentes y todavía no se corrió la cola de GPU: **todos los valores de este documento están "pendiente"**. Ningún número viejo figura como actual. Cada celda se completa con el valor de un JSON de `results/` producido por `python -m src.cli` (o la cola) con su entrada en la bitácora; mientras no exista, dice "pendiente". Qué se hizo y por qué: [`resumen.md`](resumen.md).

## Cómo leer las tablas

- **Una fila por modelo.** Un dato distinto (otro número de pasos, otra semilla, otra receta) es otro modelo o variante, con su propia fila.
- **Solo datos actuales.** Un valor solo aparece si hay un JSON `vigente` que lo respalde (`python -m src.cli resultados`); la última columna de cada fila apunta a la corrida y a su guía.
- **Del resultado a "cómo reentrenar":** cada fila tiene un link a la acción `entrenar` de la guía del modelo.
- **Video** se mide con 400 clips fijos (100 segmentos por 4 arranques), 1 s de contexto y 3 s generados, con intervalos por bootstrap por segmento y el piso de FVD (real contra real, 400 contra 400) en cada tabla. **CE** con el protocolo de comma (`ce_all` incluye los BOS).
- **Los modelos latentes** no se miden con FVD: tienen su tabla, con referencias al lado.
- **La temperatura** de cada familia (`gpt`, `genie`) sale del barrido en el holdout, nunca de val.
- **Semillas:** `genie_s` y `convlstm_px` se entrenan con 3 semillas; la fila de semillas es media ± desvío.

## Estado de cada modelo

| modelo | entrenamiento | evaluación | latencia | estado |
|---|---|---|---|---|
| `gpt_s_kd` | pendiente | pendiente | pendiente | pendiente |
| `genie_s` | pendiente | pendiente | pendiente | pendiente |
| `convlstm_px` | pendiente | pendiente | pendiente | pendiente |
| `lewm` | pendiente | pendiente | pendiente | pendiente |
| `vjepa2` | pendiente | pendiente | pendiente | pendiente |
| `vjepa2_ac` | pendiente | pendiente | pendiente | pendiente |
| referencias (`unigram`, `transicion`, `congelado`, `gpt2m`, `real`) | no se entrenan | pendiente | pendiente | pendiente |

## 1. Modelos de video: modelo → performance

<!-- tabla:video -->
| modelo | variante | CE (`ce_all`) | LPIPS 1 s | LPIPS 3 s | FVD 1 s | FVD 3 s | piso de FVD | corrida | cómo reentrenar |
|---|---|---|---|---|---|---|---|---|---|
| real (piso de FVD) | 400 contra 400 | no aplica | no aplica | no aplica | pendiente | pendiente | pendiente | `real` | no se entrena |
| unigram |  | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | `unigram` | [guía](modelos/referencias.md#unigram-evaluar) |
| transicion |  | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | `transicion` | [guía](modelos/referencias.md#transicion-evaluar) |
| congelado | T = 0 | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | `congelado` | [guía](modelos/referencias.md#congelado-evaluar) |
| gpt2m | maestro, 307M | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | `gpt2m` | [guía](modelos/referencias.md#gpt2m-evaluar) |
| gpt_s_kd | S destilado | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | `gpt_s_kd` | [guía](modelos/gpt.md#entrenar) |
| genie_s | 1 paso | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | `genie_s_p1` | [guía](modelos/genie.md#entrenar) |
| genie_s | 2 pasos | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | `genie_s_p2` | [guía](modelos/genie.md#entrenar) |
| genie_s | 4 pasos | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | `genie_s_p4` | [guía](modelos/genie.md#entrenar) |
| genie_s | 8 pasos | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | `genie_s_p8` | [guía](modelos/genie.md#entrenar) |
| genie_s | 12 pasos | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | `genie_s_p12` | [guía](modelos/genie.md#entrenar) |
| genie_s | 25 pasos | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | `genie_s_p25` | [guía](modelos/genie.md#entrenar) |
| genie_s | 3 semillas, 8 pasos: media ± desvío | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | `genie_s_p8, genie_s_s1_p8, genie_s_s2_p8` | [guía](modelos/genie.md#entrenar) |
| convlstm_px |  | no aplica | pendiente | pendiente | pendiente | pendiente | pendiente | `convlstm_px` | [guía](modelos/convlstm_px.md#entrenar) |
| convlstm_px | 3 semillas: media ± desvío | no aplica | pendiente | pendiente | pendiente | pendiente | pendiente | `convlstm_px, convlstm_px_s1, convlstm_px_s2` | [guía](modelos/convlstm_px.md#entrenar) |
| gpt_s | opcional, fuera de la cola | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | `gpt_s` | [guía](modelos/gpt.md#entrenar) |
| gpt_m | opcional, fuera de la cola | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | `gpt_m` | [guía](modelos/gpt.md#entrenar) |
| genie_m | opcional, fuera de la cola | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | `genie_m` | [guía](modelos/genie.md#entrenar) |
| maskgit_s y ablaciones | opcional, fuera de la cola | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | `maskgit_s` | [guía](modelos/maskgit.md#entrenar) |
<!-- /tabla -->

Las referencias `unigram`, `transicion` y `congelado` son los baselines; `gpt2m` es el maestro y el techo de calidad; `real` es el piso de FVD. La lista de corridas de cada fila sigue la convención de nombres de [`../CONTRIBUTING.md`](../CONTRIBUTING.md#nombres).

## 2. Modelos latentes: modelo → performance

Los latentes se leen contra sus referencias (copiar el último latente, un encoder sin entrenar, píxeles), no contra el video.

<!-- tabla:latentes -->
| modelo | variante | error latente / copiar el último latente | probes contra encoder sin entrenar | probes contra píxeles | sorpresa | acciones con CEM | latencia por paso | cómo reentrenar |
|---|---|---|---|---|---|---|---|---|
| lewm | presupuesto común | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | [guía](modelos/lewm.md#entrenar) |
| vjepa2 | presupuesto común (`vjepa2_s`) | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | [guía](modelos/vjepa2.md#entrenar) |
| vjepa2_ac | sobre `vjepa2_s` | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | [guía](modelos/vjepa2_ac.md#entrenar) |
| lewm | 10x, al final | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | [guía](modelos/lewm.md#entrenar) |
| vjepa2 | 10x, al final | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | [guía](modelos/vjepa2.md#entrenar) |
| vjepa2_ac | sobre el 10x, al final | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | [guía](modelos/vjepa2_ac.md#entrenar) |
<!-- /tabla -->

Las corridas 10x de `vjepa2` y `lewm` se hacen al final, con todo lo demás verificado; quedan en filas aparte.

## 3. Latencia: modelo → performance

Un solo protocolo: batch 1, un paso por stream, fp16 con CUDA graphs donde ande, p50 y p99, con la GPU libre. El titular es a ritmo de cámara (20 FPS), después de explicar el piso de unos 20 ms del timer (`bench --cadena --diagnostico`).

<!-- tabla:latencia -->
| modelo | dispositivo | modelo solo p50 | modelo solo p99 | cadena cerrada p50 | cadena a 50 ms p99 | frames tarde | cómo medir |
|---|---|---|---|---|---|---|---|
| `codec` | GPU | pendiente | pendiente | no aplica | no aplica | no aplica | [guía](modelos/referencias.md#codec-bench) |
| `transicion` | GPU | pendiente | pendiente | pendiente | pendiente | pendiente | [guía](modelos/referencias.md#transicion-bench) |
| `gpt2m` | GPU | pendiente | pendiente | no aplica | no aplica | no aplica | [guía](modelos/referencias.md#gpt2m-bench) |
| `gpt_s_kd` | GPU | pendiente | pendiente | pendiente | pendiente | pendiente | [guía](modelos/gpt.md#bench) |
| `genie_s` (4 pasos) | GPU | pendiente | pendiente | pendiente | pendiente | pendiente | [guía](modelos/genie.md#bench) |
| `convlstm_px` | GPU | pendiente | pendiente | pendiente | pendiente | pendiente | [guía](modelos/convlstm_px.md#bench) |
| `lewm` | GPU | pendiente | pendiente | no aplica | no aplica | no aplica | [guía](modelos/lewm.md#bench) |
| `vjepa2` | GPU | pendiente | pendiente | no aplica | no aplica | no aplica | [guía](modelos/vjepa2.md#bench) |
| `vjepa2_ac` | GPU | pendiente | pendiente | no aplica | no aplica | no aplica | [guía](modelos/vjepa2_ac.md#bench) |
<!-- /tabla -->

Otros dispositivos (T4 de Colab, CPU, laptop con ROCm, DirectML y NPU) son otras filas con su dispositivo: se miden con `bench --device <dispositivo>` y quedan pendientes hasta que se corran ([`uso.md`](uso.md#dispositivos-e-inferencia)).

## 4. Hipótesis y conclusiones: a evaluar

Las conclusiones anteriores dependían de modelos que se reemplazaron, así que se quitaron. Las hipótesis se vuelven a evaluar con los modelos y las recetas nuevos; el estado de cada una es "a evaluar".

| hipótesis | qué se compara | estado |
|---|---|---|
| H1: destilar ayuda | `gpt_s_kd` contra `gpt_s` de cero (opcional) | a evaluar |
| H2: pocos pasos de decodificación bastan | `genie_s` con 1, 2, 4, 8, 12 y 25 pasos, calidad y latencia | a evaluar |
| H3: fp16 con CUDA graphs acelera sin perder calidad | fp32 contra fp16 con graphs en los modelos de tokens | a evaluar |
| H4: la CE ordena igual que el FVD | correlación de rangos entre CE y FVD a 1 s en los modelos con verosimilitud, con su intervalo | a evaluar |

## 5. Limitaciones conocidas

- El presupuesto común (780.000 frames) es chico frente al de los papers; cada receta declara cuánto se escaló en su tabla de fidelidad.
- Una sola GPU de 12 GB y un trabajo a la vez; las latencias de otros dispositivos están pendientes.
- Los modelos latentes se entrenan con los frames reconstruidos por el VQ-VAE de comma, no con video crudo.

## 6. Cómo reproducir

Todo sale de la CLI o de la cola ([`uso.md`](uso.md)).

```bash
python -m src.cli cola correr --dry-run         # la cola completa, expandida
python -m src.cli cola correr                   # la ejecuta, de a un trabajo
python -m src.cli resultados                    # estado de cada resultado
python -m src.cli reporte                       # tablas y figuras desde results/
python -m src.cli bitacora verificar            # cada resultado vigente tiene su comando
```
