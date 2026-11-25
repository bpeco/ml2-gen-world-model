# Movimiento: dónde estamos y plan de acción (25/11/2026)

## Cuadro comparativo

Fuentes:
- Giro y FVD a 1/2/3 s: set de movimiento, 200 clips (100 curvas), semilla 0, fp16. El FVD con 200 clips tiene un ruido
  de ±60-110.
- FVD val a 1/3 s: el del paper, sobre los 100 segmentos de validación oficial (`results/summary.json`).
- Latencias: RTX 3080 Ti (la GPU del paper), B = 1, fp16 con CUDA graphs (`results/infer/latencia_*`).
- Throughput de entrenamiento: `results/train_*.json`, medido en la GPU que se indica.

Unidades:
- 1 shard = 2.500 min = 3 M frames = 387 M tokens (129 por frame).
- commaVQ completo = 100.000 min = 40 shards ≈ 15.500 M tokens.

### A. Calidad

| modelo | qué es / optimización | params | tokens de entrenamiento (% de 1 shard) | giro 1-2 s | **giro 2-3 s [IC]** | lado correcto | FVD 1/2/3 s (200 clips) | FVD val 1/3 s (paper) |
|---|---|---|---|---|---|---|---|---|
| **GPT2M (commaVQ)** | benchmark, maestro | 307 M | no publicado (comma) | 0,85 | **0,79 [0,53; 1,10]** | 0,95 | 138 / 184 / 220 | 67 / 105 |
| GPT-S kd | destilado de GPT2M (paper) | 12,4 M | 101 M (26 %) | 0,07 | 0,03 [0,01; 0,06] | 0,68 | 424 / 626 / 721 | 207 / 469 |
| **kd + FT 500 pasos lr 3e-4** | **+ más KD** | 12,4 M | 121 M (31 %) | **0,32** | **0,21 [0,10; 0,35]** | **0,78** | **363** / 503 / 635 | – |
| kd + FT 1000 pasos coseno | + más KD, lr → 0 | 12,4 M | 142 M (37 %) | 0,17 | 0,13 [0,07; 0,20] | 0,78 | 380 / 528 / 625 | – |
| kd + on-policy K = 8 | + on-policy | 12,4 M | 121 M (31 %) | 0,02 | 0,02 [0,00; 0,04] | 0,68 | 504 / 714 / 923 | – |
| kd T 1,1 | muestreo | 12,4 M | 101 M (26 %) | 0,17 | 0,12 [0,05; 0,21] | 0,60 | 462 / 704 / 872 | – |
| GPT-S d050 | sin maestro (paper) | 12,4 M | 50 M (13 %) | 0,01 | 0,00 [−0,02; 0,02] | 0,49 | 471 / 570 / 611 | 235 / 479 |
| d050 + FT 500 pasos | + más datos, sin maestro | 12,4 M | 71 M (18 %) | 0,01 | 0,00 [−0,01; 0,02] | 0,49 | 455 / 519 / 565 | – |
| GPT-S d025 | sin maestro (paper) | 12,4 M | 25 M (7 %) | no medido | no medido | – | – | 414 / 615 |
| Genie-S p4 | paper | 15,0 M | 101 M (26 %) | 0,00 | 0,00 [−0,02; 0,01] | 0,47 | 479 / 571 / 670 | 232 / 413 |
| Genie-M p4 | paper, más params | 34,8 M | 101 M (26 %) | no medido | no medido | – | – | 244 / 406 |
| MaskGIT-S p1 | paper (ablación) | 12,4 M | 101 M (26 %) | 0,00 | −0,02 [−0,04; 0,00] | 0,45 | 419 / 529 / 762 | 228 / 607 |
| MaskGIT-S p4 | ídem | 12,4 M | 101 M (26 %) | 0,00 | 0,00 [−0,01; 0,01] | 0,49 | 430 / 506 / 674 | 220 / 468 |
| MaskGIT-S p8 | ídem | 12,4 M | 101 M (26 %) | 0,00 | 0,00 [−0,01; 0,01] | 0,47 | 454 / 469 / 559 | 247 / 380 |
| MaskGIT p4 + FT p_clean 0,8 | A: contexto más limpio | 12,4 M | 142 M (37 %) | 0,00 | 0,00 [−0,01; 0,01] | 0,51 | 428 / 479 / **525** | – |
| MaskGIT p4 + FT ctx limpio | A | 12,4 M | 142 M (37 %) | 0,00 | 0,00 [−0,02; 0,01] | 0,50 | 474 / 561 / 649 | – |
| MaskGIT-S shard completo, p8 | más datos, sin maestro | 12,4 M | 379 M (98 %) | no medido | no medido | – | – | 379 / 524 |
| ConvLSTM px | paper | 8,0 M | 101 M (26 %) | no medido | no medido | – | – | 1790 / 2400 |
| Frame congelado | baseline | – | – | 0 (por construcción) | 0 | – | – | 929 / 978 |

LeWM y V-JEPA 2 predicen latentes y no generan video, así que no tienen giro ni FVD.

### B. Costo: inferencia y entrenamiento

El margen es lo que queda de los 50 ms por frame (20 FPS) después de la cadena completa (modelo + decoder del codec).

| modelo | ms/frame modelo (3080 Ti) | cadena p99 (3080 Ti) | margen a 50 ms | entrenamiento medido (tok/s, GPU) | h / 1 shard | h / commaVQ (40 shards) |
|---|---|---|---|---|---|---|
| GPT2M | 291 | – | **no es tiempo real (×6)** | no lo entrenamos | ~8 (estimado*) | ~330 (~14 días)* |
| GPT-S (kd y derivados) | 38 | **48,4** | **1,6 ms (ninguno)** | 34,8 k con maestro (3080 Ti) · 12,4 k con maestro (4060) · 36,2 k sin maestro (4060) | 3,1 (3080 Ti, KD) · 8,7 (4060, KD) · 3,0 (4060, sin maestro) | 124 · 347 · 119 |
| Genie-S p4 | 5,7 | 15,9 | **34 ms (×3)** | 255 k (3080 Ti) | 0,4 | 17 |
| Genie-M p4 | 9,4 | 20,2 | **30 ms (×2,5)** | 94 k (3080 Ti) | 1,1 | 46 |
| MaskGIT-S p4 / p8 | 6,4 / 10,4 | ~16 / ~20 (cadena = modelo + ~10 del codec) | ~30-34 ms | 89 k (4060) | 1,2 | 48 |
| ConvLSTM px | 16,5 | – | ~20 ms | 19 k (3080 Ti) | 5,6 | 222 |

\* Estimado: un paso de KD de GPT-S en la 3080 Ti rinde ~24 TFLOP/s efectivos, casi todo del forward del maestro.
Con eso, entrenar GPT2M (6 × 307 M FLOP por token) daría ~13 k tok/s. Tampoco entra en los 8 GB de la 4060.

## Lo que muestran los cuadros

1. **El giro aparece solo donde hay maestro.**
   - Todos los modelos sin destilación están en 0, sean MaskGIT, Genie o GPT, y tengan más pasos o una arquitectura
     distinta.
   - d050 con más datos sigue en 0.
2. **Más entrenamiento con maestro sí sirve.** kd pasó de 0,03 a 0,21 en giro y de 0,68 a 0,78 en lado correcto con
   solo +20 M tokens (5 % de un shard). Su CE sigue bajando, y kd vio apenas el 26 % de un shard. **Es la única
   palanca que funcionó.**
3. **No sirvieron para el giro:** muestreo (C), contexto limpio y p_clean (A), on-policy (4) y annealing hasta 0.
   - **p_clean 0,8 sí sirve para el FVD a 3 s de MaskGIT** (el mejor FVD a 3 s de los modelos chicos: 525). Es útil si
     MaskGIT termina siendo la base.
4. **La tensión central es la latencia:**
   - **GPT-S, el único chico que gira, no tiene margen:** su cadena llega a 48,4 de 50 ms. No se le pueden agregar
     parámetros sin acelerar el decode.
   - **Genie y MaskGIT tienen margen de sobra (~30 ms, 2,5-3×), pero no giran, y nunca tuvieron maestro.**
5. **Entrenar con el commaVQ completo no es práctico con destilación:**
   - KD de GPT-S: 124 h en la 3080 Ti y 347 h en la 4060, dominadas por el forward del maestro.
   - Sin maestro sí es abordable para Genie-S (17 h), Genie-M (46 h) y MaskGIT (48 h), pero sin maestro no giran.
     Esto se puede verificar barato con MaskGIT del shard completo (abajo).

## Plan de acción

### Fase 1: decidir el camino (hoy, ~3 h de GPU)

| # | qué | pregunta que responde | costo |
|---|---|---|---|
| 1a | **Giro de lo que ya está entrenado:** MaskGIT del shard completo (p4, p8) y Genie-M p4 | ¿Más datos (4×) o más parámetros (2,3×) dan giro sin maestro? | ~45 min, sin entrenar |
| 1b | **Curva de escala de KD en GPT-S:** seguir kd hasta +780 k frames (semilla 1), con giro cada ~800 pasos | ¿El giro sigue subiendo con más destilación o se estanca en ~0,2? | ~2 h |
| 1c | **Bench de latencia de GPTs más anchos y menos profundos** (sin entrenar), p. ej. 4×512, 4×768, 6×512 | ¿Hay un GPT con más parámetros que entre en 50 ms? La latencia depende más de la cantidad de capas que del ancho (GPT2M: 12 ms por capa con dim 1024; GPT-S: 6 ms con dim 384) | ~15 min |

### Fase 2: escalar lo que funciona (según la fase 1)

- **Si 1b sigue subiendo:** destilar GPT-S (o el GPT de 1c que entre en 50 ms) con 1-2 shards de datos: ~3-6 h en la
  3080 Ti o 9-17 h en la 4060.
  - Para bajar el costo del maestro: precalcular sus top-k logits una vez por shard y reutilizarlos.
- **Si Genie o MaskGIT no giran ni con más datos (1a): destilarles el maestro.** Es la apuesta con más upside porque
  tienen ~30 ms de margen para crecer, pero es investigación: el maestro es autorregresivo y MaskGIT predice en
  paralelo.
  - Opción simple: usar como etiqueta blanda la distribución del maestro en teacher forcing.
  - Prueba corta de 1000 pasos (~30 min) antes de invertir más.
- **Si 1b se estanca:** el límite es la capacidad. Pasar a un GPT más ancho que entre en 50 ms (1c), o al camino de
  Genie/MaskGIT destilado.

### Fase 3: condicionamiento por pose (E) sobre el mejor modelo

- **Sin pose futura:** es la evaluación justa.
- **Con pose futura:** sirve de techo. Se reporta aparte.

### Descartado para el movimiento

Muestreo (B y C), MaskGIT con contexto limpio y p_clean (A; p_clean queda solo como mejora de FVD), on-policy tal como
está, annealing hasta 0, F y "más datos sin maestro" en GPT-S (d050).
