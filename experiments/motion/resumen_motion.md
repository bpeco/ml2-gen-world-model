# Movimiento (giro): resumen de todo lo probado (27/11/2026)

Esto cubre las ramas `v2` y `motion-experiments`. El detalle de cada corrida está en `results.md`; el plan y los
cuadros, en `plan_accion.md`.

## El problema y cómo lo medimos

En las curvas, los modelos chicos del paper arrancan girando y en ~1 s **congelan la escena** o la hacen
**parpadear**: el video deja de girar con el auto.

**Métrica principal: giro sostenido 2-3 s** (`docs/metrica_giro.md`, con tests de sanidad).
- Se mide el flow horizontal RAFT en la franja superior, alineado con el sentido del giro real de cada curva y
  relativo al real.
- 1 = gira como el auto; 0 = no gira o parpadea.
- El IC 95 % es por bootstrap sobre clips.
- Se acompaña con:
  - **lado correcto:** la fracción de curvas en las que el video gira para el mismo lado;
  - **FVD a 1/2/3 s** en el set de movimiento;
  - el **FVD val 1/3 s** del paper.

Set: `clips_motion` (200 clips de data-0001: 100 curvas ≥ 20° y 100 rectas), que nunca se usa para entrenar. El FVD
con 200 clips tiene un ruido de ±60–110; las diferencias se dan pareadas, con p por bootstrap.

## Cuadro de todos los modelos

Latencias:
- ms del modelo en la 3080 Ti (GPU del paper), B = 1, fp16 + CUDA graphs;
- en la 4060, GPT-S tarda 51 ms y 3×768 tarda 50 ms;
- el tiempo real pide una cadena ≤ 50 ms, que es el modelo + ~10 ms del codec.

Tokens de entrenamiento como % de 1 shard (387 M tokens).

### Modelos del paper (punto de partida)

| modelo | params | tokens (% shard) | giro 2-3 s [IC] | lado correcto | FVD 1/2/3 s (movimiento) | FVD val 1/3 s | ms modelo | ¿tiempo real? |
|---|---|---|---|---|---|---|---|---|
| GPT2M (comma, maestro) | 307 M | no publicado | **0,79 [0,53; 1,10]** | 0,95 | 138 / 184 / 220 | 67 / 105 | 291 | no (×6) |
| GPT-S kd | 12,4 M | 101 M (26 %) | 0,03 [0,01; 0,06] | 0,68 | 424 / 626 / 721 | 207 / 469 | 38 | sí, justo (cadena p99 48,4) |
| GPT-S d050 | 12,4 M | 50 M (13 %) | 0,00 [−0,02; 0,02] | 0,49 | 471 / 570 / 611 | 235 / 479 | 38 | sí, justo |
| GPT-S d025 | 12,4 M | 25 M (7 %) | no medido | – | – | 414 / 615 | 38 | sí, justo |
| Genie-S p4 | 15,0 M | 101 M (26 %) | 0,00 [−0,02; 0,01] | 0,47 | 479 / 571 / 670 | 232 / 413 | 5,7 | sí, sobra (~34 ms) |
| Genie-M p4 | 34,8 M | 101 M (26 %) | 0,01 [0,00; 0,02] | 0,50 | 464 / 533 / 609 | 244 / 406 | 9,4 | sí, sobra (~30 ms) |
| MaskGIT-S p1 | 12,4 M | 101 M (26 %) | −0,02 [−0,04; 0,00] | 0,45 | 419 / 529 / 762 | 228 / 607 | ~4 | sí, sobra |
| MaskGIT-S p4 | 12,4 M | 101 M (26 %) | 0,00 [−0,01; 0,01] | 0,49 | 430 / 506 / 674 | 220 / 468 | 6,4 | sí, sobra |
| MaskGIT-S p8 | 12,4 M | 101 M (26 %) | 0,00 [−0,01; 0,01] | 0,47 | 454 / 469 / 559 | 247 / 380 | 10,4 | sí, sobra |
| MaskGIT-S shard completo p4 / p8 | 12,4 M | 379 M (98 %) | 0,00 / 0,00 | 0,62 / 0,67 | 515 / 620 / 773 · 582 / 684 / 776 | – / 379 / 524 (p8) | 6,4 / 10,4 | sí, sobra |
| ConvLSTM px | 8,0 M | 101 M (26 %) | no medido | – | – | 1790 / 2400 | 16,5 | sí |
| Frame congelado | – | – | 0 (por construcción) | – | – | 929 / 978 | – | – |

### Rama v2: entradas de movimiento en MaskGIT (FVD val, 400 clips; giro no medido)

| modelo | qué cambia | params | FVD val 1/3 s p4 | FVD val 1/3 s p8 | resultado |
|---|---|---|---|---|---|
| v2 peso3 | CE ×3 en tokens que cambian | 12,4 M | 248 / 653 | 262 / 683 | se mueve más, peor video |
| v2 pself | scheduled sampling 50 % | 12,4 M | 250 / 627 | 281 / 618 | el que más se mueve, poco realista |
| v2 motion | embedding "igual/distinto al frame anterior" | 12,4 M | 219 / 517 | 285 / 477 | más quieto, sin mejora |
| v2 motioncb | diferencia del codebook con el frame anterior | 12,4 M | 216 / 470 | 254 / 430 | ≈ base |
| v2 flow (2 semillas) | transformer de flujo, desde cero | 13,0 M | 594 / 865 · 531 / 1876 | 512 / 793 · 452 / 1865 | acumula error |
| v2 ft_cb (control) | motioncb + 1000 pasos | 12,4 M | 219 / 451 | 257 / 434 | ≈ base |
| v2 ft_flow | motioncb + flujo | 13,0 M | 267 / 971 | 263 / 1045 | bien a 1 s, diverge |
| v2 ft_l_cb | control con flow_limpio | 12,4 M | **222 / 415** | 300 / 429 | el mejor p4 (1 semilla) |
| v2 ft_l_flow | flujo + flow_limpio | 13,0 M | 286 / 795 | 341 / 657 | sigue divergiendo |
| v2 ft_l_cb_ss | control + scheduled sampling 25 % | 12,4 M | 260 / 465 | 322 / 471 | más quieto |
| v2 ft_l_flow_ss | flujo + las dos correcciones | 13,0 M | 259 / 580 | 328 / 473 | empata con su control |

En todas las variantes de MaskGIT que se midieron en el set de movimiento, el giro sostenido dio 0. Más movimiento
(tokens que cambian) no es más giro: es parpadeo.

### Rama motion-experiments (set de movimiento; T 0,9 para GPT)

| modelo | qué cambia | params | tokens (% shard) | giro 1-2 s | **giro 2-3 s [IC]** | lado correcto | FVD 1/2/3 s | ms (3080 Ti) |
|---|---|---|---|---|---|---|---|---|
| kd T 1,0 / 1,1 / T 1,0 top-p 0,95 | muestreo (C) | 12,4 M | 101 M (26 %) | ≤ 0,17 | ≤ 0,12 | 0,60–0,68 | FVD peor (+87 a +151 a 3 s) | 38 |
| MaskGIT ctx limpio / p_clean 0,8 | contexto limpio (A) | 12,4 M | 142 M (37 %) | 0,00 | 0,00 | 0,50 / 0,51 | 474 / 561 / 649 · 428 / 479 / **525** | 6,4 |
| kd + on-policy K = 8 | el alumno ve sus rollouts | 12,4 M | 121 M (31 %) | 0,02 | 0,02 [0,00; 0,04] | 0,68 | 504 / 714 / 923 | 38 |
| **kd + 500 pasos KD** (semilla 0) | más KD | 12,4 M | 121 M (31 %) | 0,32 | **0,21 [0,10; 0,35]** | 0,78 | 363 / 503 / 635 | 38 |
| kd + 500 pasos KD (semilla 1) | réplica | 12,4 M | 121 M (31 %) | 0,17 | 0,11 [0,05; 0,18] | 0,74 | 369 / 526 / 656 | 38 |
| kd + 1000 pasos coseno | annealing | 12,4 M | 142 M (37 %) | 0,17 | 0,13 [0,07; 0,20] | 0,78 | 380 / 528 / 625 | 38 |
| kd + KD +260 / +520 / +780 k frames | escala de KD (1b) | 12,4 M | 35 / 43 / 52 % | 0,11 / 0,06 / 0,09 | 0,07 / 0,03 / 0,02 | 0,77–0,80 | – / – / 377·550·672 | 38 |
| d050 + 500 pasos | más datos, sin maestro | 12,4 M | 71 M (18 %) | 0,01 | 0,00 [−0,01; 0,02] | 0,49 | 455 / 519 / 565 | 38 |
| MaskGIT + FT con maestro (1d) | KD en MaskGIT | 12,4 M | 142 M (37 %) | −0,01 | 0,00 [−0,01; 0,01] | 0,47 | 438 / 497 / 644 | 6,4 |
| MaskGIT + secuencias del maestro | datos del maestro | 12,4 M | 142 M (37 %) | 0,00 | 0,00 [−0,01; 0,01] | 0,47 | 426 / 471 / 636 | 6,4 |
| **kd + secuencias del maestro (kdB_s0)** | KD + secuencias de GPT2M | 12,4 M | 121 M (31 %) | 0,23 | **0,18 [0,10; 0,29]** | 0,76 | 351 / 454 / **537** | 38 |
| kd + secuencias (semilla 1) | réplica | 12,4 M | 121 M (31 %) | 0,20 | 0,15 [0,07; 0,24] | 0,78 | 358 / 519 / 630 | 38 |
| GPT 3×768 con KD | 2× params, misma latencia | 24,8 M | 101 M (26 %) | 0,01 | 0,00 [−0,01; 0,01] | 0,59 | 377 / 446 / 555 | ~37 (est.) |
| **GPT 3×768 + FT de kdB** | ídem + 500 pasos | 24,8 M | 121 M (31 %) | 0,06 | 0,02 [0,01; 0,04] | 0,72 | 352 / 440 / **514** | ~37 (est.) |

Costo de entrenamiento (h por shard, en la 4060):
- GPT-S con maestro: 8,7 h; sin maestro: 3,0 h.
- 3×768 con maestro: 10,2 h.
- MaskGIT: 1,2 h. Genie-S: 0,4 h en la 3080 Ti; Genie-M: 1,1 h.
- Generar secuencias del maestro: ~2 h cada 25 k frames.

## Qué funcionó

1. **Destilar de GPT2M es la única fuente de dirección.** Todo modelo sin maestro queda en giro 0, sea cual sea la
   arquitectura, los datos (hasta 4×) o los parámetros (Genie-M).
2. **Un FT corto con KD** (500 pasos, lr 3e-4) sobre kd lleva el giro sostenido de 0,03 a 0,11–0,21 (según la
   semilla), el lado correcto de 0,68 a ~0,76 y mejora el FVD (−60 a −120).
3. **Sumarle secuencias generadas por el maestro** a ese FT:
   - el giro queda igual (0,18 / 0,15);
   - las semillas se emparejan;
   - el FVD a 3 s baja a 537 (−97 con p 0,01 contra su control).
   - **kdB_s0 es el mejor modelo de tiempo real para el giro.**
4. **El doble de parámetros a la misma latencia** (3×768: menos capas, más anchas) mejora mucho el video. Tiene el
   mejor FVD a 3 s de tiempo real (514–555, contra 721 de kd), pero no sostiene el giro.
5. **p_clean 0,8 en MaskGIT** mejora el FVD a 3 s (525) sin girar.
6. **Infraestructura que queda:**
   - la métrica de giro con tests;
   - el set de movimiento con poses;
   - la destilación por secuencias (`src/maestro_seq.py`, `--seq-maestro`);
   - KD en MaskGIT;
   - el bench de latencia por configuración;
   - el probador con los modelos nuevos.

## Qué descartamos (y por qué)

| idea | resultado |
|---|---|
| Pesar más los tokens que cambian, scheduled sampling, embeddings de movimiento y módulo de flujo (v2, MaskGIT) | más movimiento = más parpadeo, no giro; el flujo extrapola sus errores |
| Temperatura / top-p (C) | sube algo el giro a costa del FVD y del lado correcto |
| Contexto limpio en MaskGIT (A) | no gira (p_clean sí sirve para el FVD) |
| On-policy (el alumno aprende sobre sus rollouts) | empeora todo (FVD +200 a 3 s) |
| Annealing a lr 0 | no suma giro |
| Más datos de KD con teacher forcing (hasta +780 k frames) | **baja** el giro sostenido (0,11 → 0,02); contradice la hipótesis |
| Más datos / más params sin maestro (MaskGIT shard completo, Genie-M, d050 + FT) | giro 0 |
| Maestro en MaskGIT, como pérdida (KL) o como datos | giro 0; la KL no baja (objetivo raster incompatible) |
| Escalar a varios shards con el maestro precalculado (fase 2 original) | no se justifica tras 1b |

## Qué se puede mejorar de cada modelo

| modelo | estado | qué le falta / qué probaría |
|---|---|---|
| GPT-S kd (paper) | giro 0,03, FVD 3 s 721 | Superado: con 500 pasos de KD + secuencias pasa a 0,18 y 537 a la misma latencia (kdB_s0). |
| **kdB_s0 (candidato final)** | giro 0,18, FVD 3 s 537 | Pose (fase 3); 3 semillas para fijar el número; FVD val del paper (no medido); confirmar la latencia de la cadena. |
| GPT 3×768 | mejor video, sin giro | Más pasos de FT o un 4×512 (más profundo, 0,91× latencia): la profundidad parece importar para la dinámica. |
| GPT-S d050 / d025 | sin dirección | Solo sirven como control sin maestro; no vale la pena seguir. |
| Genie-S / Genie-M | buen FVD, giro 0, mucho margen de latencia | El maestro no transfiere a decodificación paralela. Solo con condicionamiento por pose (sin maestro) podrían girar: es la mejor chance de esta familia. |
| MaskGIT-S (y v2) | buen FVD a 3 s con p8 / p_clean, giro 0 | Igual que Genie: pose como entrada. p_clean 0,8 y ft_l_cb son las mejores bases. |
| MaskGIT shard completo | aprende el signo del frame 1, nada más | No escala solo con datos. |
| ConvLSTM px | FVD muy malo | Fuera de la discusión de movimiento. |
| GPT2M | el techo (0,79) | No es tiempo real (291 ms); solo como maestro. |

## Qué queda por probar

1. **Fase 3: condicionar por pose** (plan abajo). Es la única palanca grande sin probar, y la única que puede hacer
   girar a Genie/MaskGIT.
2. **Evaluación en val oficial** (FVD val 1/3 s de kdB_s0, kdOP_ctrl, 3×768 + FT y MaskGIT p_clean) para compararlos
   con la tabla del paper y verlos en el probador. Son ~1,5 h de GPU.
3. **3 semillas del modelo final.** Ya vimos que la semilla mueve el giro de 0,11 a 0,21.
4. **Alumno 4×512** (15 M, 0,91× latencia), por si la profundidad es lo que importa para el giro (~3 h).

## Plan de la fase 3: condicionamiento por pose

**Idea:** hoy el modelo tiene que deducir del video cuánto está girando el auto, y no lo sostiene. Las poses
(`data/train_poses.npy`: 6 velocidades por frame, alineadas con los tokens, 0,1 % NaN; `data/valp_poses.npy` para el
set de movimiento) le dan esa información en forma explícita.

**Implementación** (~1,5–2 h mías, sin GPU):
- GPT: una proyección lineal de 7 → 384 (las 6 velocidades normalizadas + un flag de NaN), **inicializada en cero**.
  Se suma a los embeddings de los 129 tokens del frame f, con la pose de la transición f−1 → f. Las estadísticas de
  normalización viajan en el checkpoint (como en `latent.py`). Con la proyección en cero, el modelo arranca idéntico a
  kdB_s0.
- `train.py` ya tiene el camino `uses_poses`.
- `next_frame` / `rollouts`, con tres modos de pose para los frames generados:
  - **none:** sin pose (flag de faltante). Es el control de sanidad.
  - **hold:** se mantiene la última pose del contexto (el auto sigue girando igual). Es **la prueba justa**: solo usa
    información del pasado.
  - **real:** la pose futura real. Es el techo de un modelo condicionado por acción, que ya es otra tarea.
- Tests: con la proyección en cero, los tokens son idénticos a kdB_s0; la pose del frame f no afecta los logits de
  frames anteriores.

**Corridas** (una cambio por vez, mismo init kdB_s0, semilla 0):

| corrida | qué | GPU (4060) |
|---|---|---|
| kdP | FT de kdB_s0 con poses, 1000 pasos de KD sobre datos reales, lr 3e-4 | ~50 min |
| kdP_ctrl | el mismo FT sin poses | ~50 min |
| rollouts | kdP en hold (con FVD), real (con FVD) y none; kdP_ctrl con FVD | ~1 h |
| (recomendado) 2ª semilla de kdP y kdP_ctrl | por la varianza de semilla | +1 h 40 |

**Criterios:**
- **Éxito:** en **hold**, giro 2-3 s ≥ 0,35 con IC por encima de kdP_ctrl, lado correcto ≥ 0,85, FVD a 3 s no peor
  que kdP_ctrl, y sin giros inventados en las rectas (flow horizontal en rectas ≈ real).
- **Techo:** si en **real** gira (~0,6–0,8) pero en hold no, el modelo puede generar el giro y lo que falla es
  predecir hacia dónde va el auto. Eso abre un modelo condicionado por acción como producto.
- **none** tiene que dar ≈ kdP_ctrl.

**Si funciona:** la misma proyección en Genie-S y MaskGIT p_clean (~1 h de GPU cada uno, sin maestro), que tienen
~30 ms de margen de latencia.

**Total:** ~1,5–2 h de implementación + **~2 h 40 de GPU con una semilla** (~4 h 20 con dos). Requiere OK (> 1 h).
