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
| kd + FT 500 pasos, semilla 1 (fase 1) | réplica de semilla del FT de 500 pasos | 12,4 M | 121 M (31 %) | 0,17 | 0,11 [0,05; 0,18] | 0,74 | 369 / 526 / 656 | – |
| kd + KD +260 k frames (fase 1, 1b) | más KD, semilla 1 | 12,4 M | 135 M (35 %) | 0,11 | 0,07 [0,03; 0,12] | 0,77 | – | – |
| kd + KD +520 k frames (1b) | ídem | 12,4 M | 168 M (43 %) | 0,06 | 0,03 [0,01; 0,07] | 0,79 | – | – |
| kd + KD +780 k frames (1b) | ídem | 12,4 M | 201 M (52 %) | 0,09 | 0,02 [0,00; 0,04] | 0,80 | 377 / 550 / 672 | – |
| **kd + secuencias del maestro, semilla 0 (2B′)** | KD + secuencias de GPT2M | 12,4 M | 121 M (31 %) | 0,23 | **0,18 [0,10; 0,29]** | 0,76 | 351 / 454 / **537** | – |
| kd + secuencias del maestro, semilla 1 (2B′) | ídem | 12,4 M | 121 M (31 %) | 0,20 | 0,15 [0,07; 0,24] | 0,78 | 358 / 519 / 630 | – |
| GPT 3×768 con KD desde cero (2A′) | 2× params, misma latencia | 24,8 M | 101 M (26 %) | 0,01 | 0,00 [−0,01; 0,01] | 0,59 | 377 / 446 / 555 | – |
| GPT 3×768 + FT de kdB | 3×768 + 500 pasos KD y secuencias | 24,8 M | 121 M (31 %) | 0,06 | 0,02 [0,01; 0,04] | 0,72 | 352 / 440 / **514** | – |
| kd + on-policy K = 8 | + on-policy | 12,4 M | 121 M (31 %) | 0,02 | 0,02 [0,00; 0,04] | 0,68 | 504 / 714 / 923 | – |
| kd T 1,1 | muestreo | 12,4 M | 101 M (26 %) | 0,17 | 0,12 [0,05; 0,21] | 0,60 | 462 / 704 / 872 | – |
| GPT-S d050 | sin maestro (paper) | 12,4 M | 50 M (13 %) | 0,01 | 0,00 [−0,02; 0,02] | 0,49 | 471 / 570 / 611 | 235 / 479 |
| d050 + FT 500 pasos | + más datos, sin maestro | 12,4 M | 71 M (18 %) | 0,01 | 0,00 [−0,01; 0,02] | 0,49 | 455 / 519 / 565 | – |
| GPT-S d025 | sin maestro (paper) | 12,4 M | 25 M (7 %) | no medido | no medido | – | – | 414 / 615 |
| Genie-S p4 | paper | 15,0 M | 101 M (26 %) | 0,00 | 0,00 [−0,02; 0,01] | 0,47 | 479 / 571 / 670 | 232 / 413 |
| Genie-M p4 | paper, más params | 34,8 M | 101 M (26 %) | 0,00 | 0,01 [0,00; 0,02] | 0,50 | 464 / 533 / 609 | 244 / 406 |
| MaskGIT-S p1 | paper (ablación) | 12,4 M | 101 M (26 %) | 0,00 | −0,02 [−0,04; 0,00] | 0,45 | 419 / 529 / 762 | 228 / 607 |
| MaskGIT-S p4 | ídem | 12,4 M | 101 M (26 %) | 0,00 | 0,00 [−0,01; 0,01] | 0,49 | 430 / 506 / 674 | 220 / 468 |
| MaskGIT-S p8 | ídem | 12,4 M | 101 M (26 %) | 0,00 | 0,00 [−0,01; 0,01] | 0,47 | 454 / 469 / 559 | 247 / 380 |
| MaskGIT p4 + FT p_clean 0,8 | A: contexto más limpio | 12,4 M | 142 M (37 %) | 0,00 | 0,00 [−0,01; 0,01] | 0,51 | 428 / 479 / **525** | – |
| MaskGIT p4 + FT ctx limpio | A | 12,4 M | 142 M (37 %) | 0,00 | 0,00 [−0,02; 0,01] | 0,50 | 474 / 561 / 649 | – |
| MaskGIT p4 + FT 1000 pasos (control 1d) | batch 8 × 2 | 12,4 M | 142 M (37 %) | 0,00 | −0,01 [−0,02; 0,01] | 0,49 | 432 / 488 / 643 | – |
| MaskGIT p4 + FT con maestro (1d) | KD opción A | 12,4 M | 142 M (37 %) | −0,01 | 0,00 [−0,01; 0,01] | 0,47 | 438 / 497 / 644 | – |
| MaskGIT p4 + secuencias del maestro (2B′) | datos del maestro | 12,4 M | 142 M (37 %) | 0,00 | 0,00 [−0,01; 0,01] | 0,47 | 426 / 471 / 636 | – |
| MaskGIT-S shard completo, p4 | más datos, sin maestro | 12,4 M | 379 M (98 %) | 0,00 | 0,00 [−0,01; 0,01] | 0,62 | 515 / 620 / 773 | – |
| MaskGIT-S shard completo, p8 | ídem | 12,4 M | 379 M (98 %) | 0,00 | 0,00 [0,00; 0,01] | 0,67 | 582 / 684 / 776 | 379 / 524 |
| ConvLSTM px | paper | 8,0 M | 101 M (26 %) | no medido | no medido | – | – | 1790 / 2400 |
| Frame congelado | baseline | – | – | 0 (por construcción) | 0 | – | – | 929 / 978 |

LeWM y V-JEPA 2 predicen latentes y no generan video, así que no tienen giro ni FVD.

### B. Costo: inferencia y entrenamiento

El margen es lo que queda de los 50 ms por frame (20 FPS) después de la cadena completa (modelo + decoder del codec).

| modelo | ms/frame modelo (3080 Ti) | cadena p99 (3080 Ti) | margen a 50 ms | entrenamiento medido (tok/s, GPU) | h / 1 shard | h / commaVQ (40 shards) |
|---|---|---|---|---|---|---|
| GPT2M | 291 | – | **no es tiempo real (×6)** | no lo entrenamos | ~8 (estimado*) | ~330 (~14 días)* |
| GPT 3×768 (kd3x768, fase 2) | ~37 (estimado: 49,7 ms en la 4060, 0,97× GPT-S) | ~47 (estimado) | ~3 ms | 10,5 k con maestro (4060) | 10,2 (4060, KD) | ~410 |
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

## Plan detallado por fases

### Objetivo y punto de partida

**Objetivo:** un modelo de tiempo real (cadena p99 ≤ 50 ms en la 3080 Ti) que sostenga el giro, acercándose al 0,79
de GPT2M sin perder FVD.

**Hoy:** el mejor modelo de tiempo real es kd + FT, con giro 0,21 y 48,4 ms.

**Hipótesis de trabajo:**
- **H1, la dirección del giro viene del maestro.** Medido: sin maestro, ningún modelo gira.
- **H2, los alumnos están subentrenados.** kd vio el 26 % de un shard y su CE sigue bajando; +5 % de un shard llevó el
  giro de 0,03 a 0,21.
- **H3, el margen de latencia está en Genie y MaskGIT** (~30 ms), no en GPT-S (1,6 ms).
- **H4, el maestro se puede transferir a un alumno no autorregresivo** (Genie y MaskGIT). Sin probar.

**Reglas de todas las fases:**
- Un cambio por experimento, contra su control (mismo init, clips, semilla y presupuesto).
- Antes de corridas de más de ~1 h de GPU, o de usar el maestro sobre datos nuevos, consultar con una estimación.
- Todo se registra en `results.md`.
- Métricas por corrida:
  - giro f1 / 0-1 / 1-2 / 2-3 s con IC y lado correcto;
  - |flow| y tokens por horizonte;
  - FVD/LPIPS a 1/2/3 s con diferencia pareada contra el control;
  - ms/frame.
- Set de evaluación: `clips_motion.json` (data-0001). **data-0001 no se usa nunca para entrenar.**
- Los tiempos que siguen son de la 4060 (la máquina de los experimentos); entre paréntesis, la 3080 Ti cuando se
  conoce. Batch 8 × accum 2 para todo lo que usa maestro en la 4060.

---

### Fase 1: decidir el camino (~4 h de GPU en total; no se bajan datos)

Cuatro experimentos independientes. 1a y 1c no entrenan.

#### 1a. Más datos o más parámetros, sin maestro (~45 min)

**Qué:** rollouts en `clips_motion` (T 1,0), flow, giro y FVD de modelos que ya están entrenados:
- `maskgit_s_abs_shard0`, p4 y p8: 4× datos (98 % del shard);
- `genie_m`, p4: 2,3× parámetros.

**Contra:** `mg_p4/p8_motion` y `genie_s_p4_motion`.

**Pregunta:** ¿más datos o más params dan giro sin maestro? Contrasta H1 a otra escala.

**Decisión:**
- Si el giro 2-3 s tiene IC por encima de 0 y el lado correcto es ≥ 0,6: más datos o más params alcanzan, y Genie/MaskGIT
  sin maestro con commaVQ entero (17-48 h) vuelve a ser opción.
- Si quedan en 0, H1 se refuerza.

#### 1b. Curva de escala de la destilación de GPT-S (~2 h)

**Qué:**
- KD desde `gpt_s_kd` (`final.pt`), lr 3e-4 constante (el del control que dio 0,21), **semilla 1** para no repetir
  ventanas;
- presupuesto +780 k frames (2440 pasos), lo que duplica lo que vio kd;
- checkpoints cada ~800 pasos (+260 k frames), y en cada uno rollout + flow + giro **sin FVD** (~10 min);
- FVD solo en el último.

**Contra:** `kdOP_ctrl` (+160 k frames, semilla 0) y `gpt_s_kd`.

**Pregunta:** ¿el giro sigue subiendo con los frames de destilación (H2) o se estanca en ~0,2?

**Resultado:** curva de giro 2-3 s y lado correcto contra frames vistos.

**Decisión:**
- **Sigue subiendo** (el último punto supera a 0,21 con IC que no se solapa, o pendiente positiva en los 3 puntos):
  vale la pena escalar datos (2A).
- **Se estanca:** el límite es la capacidad de GPT-S, y conviene más parámetros (1c, 2B).

#### 1c. Bench de latencia de GPTs más grandes (~15 min, sin entrenar)

**Qué:** `src.bench` con pesos al azar, fp16 y CUDA graphs, B = 1:
- configuraciones 4×512, 4×640, 4×768, 6×512, 6×640 y 8×384, contra 6×384 (GPT-S);
- en la 4060 se mide la latencia relativa a GPT-S (en la 4060 GPT-S tarda 51 ms y en la 3080 Ti, 38; el 131 ms de la primera versión del plan estaba mal);
- **la confirmación final es en la 3080 Ti**, con la cadena completa.

**Pregunta:** ¿hay un GPT con más parámetros que entre en la cadena de 50 ms? La latencia depende sobre todo de la
cantidad de capas (128 pasos secuenciales por frame): por capa y por frame, 6 ms en GPT-S (dim 384) y 12 ms en GPT2M
(dim 1024).

**Resultado:** la tabla params → ms y la config más grande con cadena estimada ≤ 47 ms.

**Decisión:** define el alumno GPT de la fase 2. Si ninguna entra, el alumno sigue siendo 6×384.

#### 1d. MaskGIT-S con maestro (~1 h de implementación + ~40 min de GPU)

**Qué:** `train_loss` de MaskGIT con maestro, opción A (etiquetas blandas en teacher forcing):
- El maestro corre sobre la secuencia real (igual que en kd).
- En cada posición enmascarada, la pérdida es α·CE + (1 − α)·τ²·KL(maestro ‖ MaskGIT), con α 0,5 y τ 1, igual que kd.
- El orden de los tokens del maestro (raster, con BOS por frame) se mapea a la grilla de MaskGIT.

FT desde `maskgit_s_abs`: 1000 pasos, lr 3e-4, warmup 100, semilla 0 (la receta de mgA). Rollouts p4 y p8.

**Contra:** `mgA_ctrl` (el mismo FT sin maestro, ya está).

**Pregunta:** ¿el maestro le transfiere dirección a un alumno no autorregresivo (H4)?

**Decisión, en orden de fuerza:**
1. Giro 2-3 s con IC por encima de 0 y lado correcto ≥ 0,6: **éxito**, y se abre el camino 2B.
2. Solo el giro del frame 1 o de 0-1 s pasa a positivo (hoy son −0,15 y −0,01): **señal débil**; probar con más pasos
   o α más bajo en 2B antes de descartar.
3. Nada cambia: H4 falla con la opción A. La opción B (secuencias generadas por el maestro) queda como último recurso
   por su costo (~8 h por 100 k frames).

Si 1d da éxito, la misma implementación se aplica a Genie (`genie_s`) en la fase 2.

#### Cierre de la fase 1

Una tabla en `results.md` y la decisión de qué rama sigue.

| resultado de 1b | resultado de 1d | sigue |
|---|---|---|
| sube | éxito | 2A y 2B en paralelo; la final la gana el mejor giro con cadena ≤ 50 ms |
| sube | falla | 2A (GPT; si 1c lo permite, el GPT más grande) |
| se estanca | éxito | 2B (Genie/MaskGIT con maestro, más datos y más params) |
| se estanca | falla | 2A con el GPT más grande de 1c, y revisar la opción B de destilación |

---

### Resultado de la fase 1 (detalle en `results.md`, "Fase 1 del plan de acción")

- **1a:** sin maestro no hay giro sostenido, ni con 4× datos (MaskGIT shard 0) ni con más params (Genie-M).
- **1b:** **se estanca y baja**. Con la misma semilla, el giro 2-3 s da 0,11 / 0,07 / 0,03 / 0,02 a +160 / 260 / 520 /
  780 k frames de KD. El 0,21 de kdOP_ctrl es la semilla buena: la réplica con semilla 1 da 0,11.
- **1c:** 3×768 (24,8 M) cuesta 0,96× la latencia de GPT-S, y 4×512 (15 M) cuesta 0,91×.
- **1d:** MaskGIT con maestro (opción A) no cambia nada, y el KL queda plano en 2,2.

**Celda:** "se estanca × falla". La fase 2 de abajo queda **revisada**. Escalar datos de KD con teacher forcing (2A tal
como estaba, la infraestructura 2.0 y bajar shards) no se justifica: más frames de destilación no subieron el giro
sostenido, lo bajaron.

### Fase 2 revisada (propuesta, pendiente de aprobación)

El diagnóstico: el alumno aprende del maestro el giro del arranque (f1 0,57, lado 0,80), pero en el rollout largo se
apaga. Es sesgo de exposición: con teacher forcing el alumno nunca ve sus propios errores. Que más datos lo empeoren
es consistente con esto. Hay dos vías que atacan causas distintas, y van en una sola cola, de a una.

**2B′. Destilación por secuencias (opción B) en piloto, ~3 h. Va primero.**
- **Datos:** GPT2M genera 10 frames a T 1,0 desde 10 frames reales de contexto, en ventanas del shard 0 de
  entrenamiento (nunca data-0001). Se guarda la secuencia generada y el top-16 de los logits del maestro en cada paso.
  Son ~2 h para ~25 k frames generados, según la estimación de ~8 h por 100 k.
- **Corridas, cada una contra su control:**
  - `kdB_gpt`: FT de gpt_s_kd, 500 pasos, mitad KD real y mitad secuencias del maestro. Control: kdOP_ctrl y kdR1.
  - `mgB`: FT de maskgit_s_abs sobre las secuencias del maestro (CE sobre tokens del maestro). Control: mgT_ctrl.
- **Por qué:** el alumno ve trayectorias largas que giran, y no solo la condicional de un paso. Es la única vía que
  queda para MaskGIT/Genie.
- **Éxito:** GPT, giro 2-3 s por encima del rango 0,11–0,21 de los FT con KD (2 semillas); MaskGIT, giro 2-3 s con IC
  por encima de 0.

**2A′. Capacidad: alumno 3×768 con KD desde cero, ~2,5–3 h.**
- Mismo presupuesto que kd: ~780 k frames del shard 0, el mismo maestro, con snapshots cada 0,25 del presupuesto.
- Control: gpt_s_kd, el mismo presupuesto y la misma receta.
- **Pregunta:** ¿el doble de parámetros, a la misma latencia, sostiene mejor el giro?
- Antes de pasar a la 3080 Ti hay que medir la cadena completa.

**Después:** si alguna vía da éxito, se escala solo esa (más secuencias del maestro, o 4×512/3×768 con más pasos), y
recién ahí vuelve a tener sentido la infraestructura 2.0. Si ninguna da éxito, el modelo final de tiempo real es
kdOP_ctrl / kdR1 (giro 0,1–0,2, FVD −60/−110/−75 contra kd) y se pasa a la fase 3.

### Resultado de la fase 2 revisada (detalle en `results.md`)

- 2B′: las secuencias del maestro no suben el giro (0,18 / 0,15 contra 0,21 / 0,11), pero mejoran el FVD a 3 s
  (kdB_s0: 537, −97 contra su control).
- 2A′: el 3×768 mejora mucho el FVD (555 a 3 s, −166 contra kd) a la misma latencia, pero no gira (0,00; lado 0,59).
- Candidato a modelo final de tiempo real: **kdB_s0**. El FT corto de kdB sobre kd3x768 da FVD 514 a 3 s pero giro 0,02: no cambia el candidato.

### Fase 2 (versión original, antes de la fase 1): escalar la vía ganadora

#### 2.0. Infraestructura común

Se hace solo si la fase 2 va a usar más de ~0,5 shard.

**Datos nuevos:**
- Bajar data-0002 y data-0003 (2 shards, ~6 M frames) con `python -m src.data --shard`. data-0001 sigue reservado para
  evaluar.
- Hay que verificar que no compartan rutas con data-0001 ni con el val oficial (filtrar segmentos de rutas compartidas,
  igual que en `clips_motion`).

**Maestro precalculado** (`src/maestro_topk.py`, nuevo):
- **Ventanas fijas:** por segmento, ventanas de 20 frames con paso 20 (60 por segmento, 3 M frames por shard). La
  distribución del maestro depende de qué contexto vio, así que con logits precalculados las ventanas tienen que ser
  siempre las mismas. Se pierde el desplazamiento aleatorio de las ventanas; se acepta.
- **Qué se guarda:** por token, las k = 16 probabilidades más altas (fp16) y sus índices (int16): 64 B por token, unos
  25 GB por shard. Hace falta verificar el espacio en disco.
- **Costo:** un forward del maestro por shard, ~5 h en la 4060 (~2,5 h en la 3080 Ti), **una sola vez**, reutilizable
  para GPT-S, MaskGIT y Genie.
- **KL con top-k:** sobre las 16 clases más un bucket con la masa restante (KL esparsa).
- **Validación antes de usarlo en serio:** 500 pasos de kd con top-k contra 500 pasos con KD completo (`kdOP_ctrl`).
  Tiene que dar giro y FVD equivalentes (IC solapados).

**Ganancia:** sin el forward del maestro, un shard de KD cuesta ~3 h para GPT-S y ~1,3 h para MaskGIT en la 4060,
contra ~9 h hoy.

#### 2A. Rama GPT

**Alumno:** GPT-S (6×384), o la config más grande de 1c que entre en 50 ms. Si cambia la arquitectura, se entrena
desde cero con KD; si es GPT-S, se sigue desde el mejor checkpoint de 1b.

**Corridas:**
1. **KD con 1 shard nuevo:** data-0002 con maestro top-k (~3 h). Giro cada 0,25 shard.
2. **Si sigue subiendo:** +1 shard (data-0003, ~3 h).
3. **Variante de pérdida:** α 0,2 (más peso al maestro) contra α 0,5, 500 pasos cada una (~40 min). Se hace solo si
   la curva se aplana.

**Controles:** el mismo presupuesto sin maestro (sirve para separar maestro de datos; se puede achicar a 0,5 shard) y
una segunda semilla del mejor.

**Éxito:** giro 2-3 s ≥ 0,35 y lado correcto ≥ 0,85, con FVD a 3 s igual o mejor que `kdOP_ctrl` y cadena ≤ 50 ms.

#### 2B. Rama Genie/MaskGIT con maestro

Se hace si 1d dio éxito o señal débil.

**Alumnos:** MaskGIT-S y Genie-S con maestro (la pérdida de 1d). Después, más params con el margen de ~30 ms: Genie-M
(35 M, ya existe y su cadena es de 20 ms), o un MaskGIT-M a medida, medido antes con el bench.

**Corridas:**
1. **MaskGIT-S con maestro, 1 shard** (~1,3 h con top-k). Giro cada 0,25 shard.
2. **Genie-S con maestro:** el mismo presupuesto.
3. **El mejor de los dos en tamaño M:** 1-2 shards.
4. **Combinar con p_clean 0,8**, que mejoró el FVD a 3 s de MaskGIT (−124). Es un cambio aparte, contra su control.

**Pasos de muestreo:** se evalúa con 4 y 8. El de 8 da mejor FVD a 3 s y sigue en tiempo real.

**Éxito:** el mismo criterio que 2A. Además, si iguala a 2A en giro, gana por margen de latencia.

#### Cierre de la fase 2

- Elegir el modelo final por giro 2-3 s, con FVD como desempate y cadena ≤ 50 ms como restricción dura.
- Confirmar con 3 semillas.
- Medir la latencia de la cadena en la 3080 Ti.

---

### Fase 3: condicionamiento por pose (E) sobre el modelo final

**Implementación:**
- Embedding lineal de las 6 velocidades normalizadas por frame. Los NaN pasan a 0 y se agrega un flag de dato
  faltante.
- La proyección se inicializa en cero, así que al arrancar el modelo es idéntico al de partida.
- Se suma al embedding de los tokens de cada frame. `train.py` ya tiene el camino `uses_poses`.

**Entrenamiento:** FT corto desde el modelo final con poses (~0,25 shard).

**Evaluación, reportada por separado:**
1. **Sin pose futura** (solo la del contexto). Es la prueba justa de si la señal ayuda a sostener el giro.
2. **Con la pose futura real.** Es el techo de un modelo condicionado por acción, que ya es otra tarea.

**Control:** el mismo FT sin poses.

---

### Fase 4: evaluación final y documentación

1. **Set de validación oficial (100 segmentos):** FVD/LPIPS a 1/3 s, comparable con el paper. Giro en `clips_motion`
   y en el set de 10-20°.
2. **3 semillas** del modelo final y de su control.
3. **Latencia de la cadena en la 3080 Ti:** p50/p99 y cumplimiento de 20 FPS.
4. **Figuras:** giro contra latencia con todo, curva de giro contra frames destilados y figura de la métrica.
5. **Informe aparte** (adenda o sección nueva), **sin tocar `paper/informe.pdf` ni `main.tex`**.

---

### Calendario y costo estimado (4060)

| fase | GPU | trabajo mío | requiere OK |
|---|---|---|---|
| 1a + 1c | ~1 h | – | no (no entrena) |
| 1b | ~2 h | – | sí (> 1 h) |
| 1d | ~40 min | ~1 h de implementación | no (< 1 h) |
| 2.0 | bajar 2 shards + ~10 h de precálculo del maestro (2 shards) + validación de top-k | ~2 h | sí (maestro sobre datos nuevos) |
| 2A o 2B | ~4-8 h según la rama (con top-k) | – | sí |
| 3 | ~2 h | ~1-2 h | sí |
| 4 | ~3 h | ~2 h | sí |

**Total:** ~25-30 h de GPU repartidas en varios días. La fase 1 se puede hacer hoy.

### Descartado para el movimiento

Muestreo (B y C), MaskGIT con contexto limpio y p_clean (A; p_clean queda solo como mejora de FVD), on-policy tal como
está, annealing hasta 0, F y "más datos sin maestro" en GPT-S (d050).
