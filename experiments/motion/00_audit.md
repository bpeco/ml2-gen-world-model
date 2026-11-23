# Fase 0: auditoría del movimiento

Rama `motion-experiments`, creada desde `v2` (c06ead0). En esta fase no se cambió código.

## Resumen

1. **Modelos en juego, todos sobre tokens commaVQ (128 por frame, 20 FPS, 20 frames de contexto y 60 de rollout):**
   - `gpt2m`, el maestro de comma (307M);
   - `gpt_s_kd`, el alumno destilado (12,4M);
   - `gpt_s_d025` y `gpt_s_d050`, GPT-S de cero con 25 % y 50 % del presupuesto. No existe un GPT-S de cero con el presupuesto completo;
   - `maskgit_s_abs`, el MaskGIT del paper, con semillas 0 y 1, y `maskgit_s_abs_shard0`, entrenado con el shard completo;
   - `genie_s` y `genie_m`.
2. **Evaluación:** 400 clips del val oficial (100 segmentos con 4 arranques, `results/clips_eval.json`), con FVD y LPIPS a 1/2/3 s y diferencias pareadas por bootstrap de segmento (`src.estadistica`). Hay además un holdout de 50 segmentos de train para elegir hiperparámetros.
3. **Experimentos de movimiento ya hechos en `v2`, todos sobre MaskGIT-S:** tasa de cambio de tokens; `peso_cambio=3`; scheduled sampling (`p_self` 0,5 y 0,25); `motion_in`; `motion_cb`; un transformer de flujo, de cero y como ajuste fino; `flow_limpio`; un barrido de temperatura y de `choice_temp`; un análisis sintético del latente.
4. **Sampler de MaskGIT:** `decode_frame` en [src/models/maskgit.py:66](../../src/models/maskgit.py#L66). `next_frame` lo llama en [maskgit.py:330](../../src/models/maskgit.py#L330) y `MaskGITStream._frame` en [src/infer/stream.py](../../src/infer/stream.py).
5. **Loss de destilación:** `lm_loss` en [src/models/base.py:152](../../src/models/base.py#L152) (`alpha·CE + (1−alpha)·tau²·KL`), llamada desde `GPT.train_loss` en [src/models/gpt.py:120](../../src/models/gpt.py#L120). El maestro corre sobre el mismo contexto real (teacher forcing).
6. **Checkpoints:** todos en `checkpoints/<run>/final.pt`. Los ajustes finos de `v2` parten de `maskgit_s_v2_motioncb`.
7. **Bugs:** no encontré off-by-one, ni errores de máscara, de orden de tokens o de contexto deslizante (detalle abajo).
8. **Hay dos desajustes reales entre entrenamiento e inferencia:**
   - MaskGIT entrena con el contexto enmascarado (64 % en promedio, `p_clean=0`) y genera con el contexto limpio;
   - el alumno se destila solo con contexto real.
9. **`pose.npy`:** un valor por frame (N, 1200, 6), alineado con los tokens. Son velocidades, no deltas de pose: probablemente [vx, vy, vz] en m/s y [roll, pitch, yaw] en rad/s. **El val oficial no tiene poses.**
10. **Los datos que ya existen anticipan el diagnóstico.** En MaskGIT el déficit de movimiento está desde el frame 1 y crece con los pasos de decodificado: es el sampler (H2). En el alumno GPT el frame 1 está cerca del real y el movimiento se apaga con el horizonte: es drift o exposure bias (H3).

## 1. Bugs típicos

| Chequeo | Resultado | Evidencia |
|---|---|---|
| Off-by-one en el target | **OK** | GPT: `x = seq[:, :-1]`, `y = seq[:, 1:]` ([gpt.py:122](../../src/models/gpt.py#L122)). MaskGIT predice el token en su propia posición; el frame t está enmascarado y es el target, que es lo correcto para un modelo enmascarado. |
| Máscara y posiciones | **OK** | Máscara `block`: `f[j] <= f[i]` con `f = pos // 129`, causal entre frames y bidireccional dentro de cada uno ([backbone.py:36](../../src/models/backbone.py#L36)). Posiciones absolutas de 0 a P−1, re-prefill del contexto en cada frame, igual que en las ventanas de entrenamiento. En `v2` verificamos que el camino con caché y CUDA graphs da los mismos logits que la pasada completa. |
| Orden de tokens 8×16 | **OK** | Los modelos tratan el frame como una secuencia de 128; el decoder de comma lo reordena por dentro. La sonda del análisis sintético usa `view(8, 16)` row-major y separa dx de dy (R² 0,96 y 0,94): si el orden estuviera mal, no los separaría. |
| Contexto deslizante y caché KV | **OK** | `ctx[:, -ctx_frames:]` y re-prefill en cada frame ([base.py:72-76](../../src/models/base.py#L72)). No se pierden frames al pasar los 20: siempre se usan los últimos 19 más el nuevo. |
| Destilación: top-k y temperatura | **OK** | La KL es sobre los 1025 logits completos, sin truncar ni renormalizar; `tau=1`, `alpha=0,5`. |
| Schedule de MaskGIT | **Fiel a MaskGIT** | Schedule coseno: con 8 pasos el paso 1 fija solo 3 tokens (125 siguen enmascarados) y los demás se agregan de a poco. El frame arranca 100 % enmascarado. `choice_temp=1,0` (MaskGIT usa 4,5); ya probamos 4,5 y no mejoró el FVD. |
| **Desajuste: contexto enmascarado al entrenar** | **Problema real, no es un bug** | MaskGIT entrena las 20 posiciones de frame con una tasa coseno por frame (media 64 %), así que el contexto siempre llega parcialmente tapado. Al generar llega limpio. `p_clean` existe pero está en 0 en el modelo del paper. Lo encontramos en `v2` para el flujo, y afecta también a la red base. |
| **Desajuste: teacher forcing en el alumno** | **Problema real (H3)** | `gpt_s_kd` se destiló solo con contexto real. |
| Temperatura del alumno | Para tener en cuenta | `gpt_s_kd` genera con T=0,9, elegida en el holdout por FVD a 3 s: 0,7 → 805, 0,8 → 645, 0,9 → 571, 1,0 → 681, 1,1 → 867. El movimiento no se midió en ese barrido. |
| El codec no es idempotente | Para tener en cuenta en las métricas | Re-codificar un frame decodificado cambia el 14 % de los tokens, y ruido de píxel σ=4 cambia el 51 %. **La tasa de cambio de tokens mezcla movimiento con el ruido del codec**, por eso hace falta la métrica de flow (RAFT). |
| Referencia de gpt2m | Débil | Su tasa de cambio se midió con **8 clips** (`gpt2m_ej`), no con 400. |

## 2. `pose.npy`

| | |
|---|---|
| Archivo | `data/train_poses.npy` (2500, 1200, 6) float32, en el mismo orden que `train_tokens.npy` (verificado al extraerlo, `src/data.py`) |
| Frecuencia | Un valor por frame (1200 frames de 20 FPS por segmento) |
| NaN | 0,08 % de los frames |
| Col 0 | Media 17,6, desvío 9,5, p99 36; autocorrelación 1,00. **Velocidad de avance en m/s** (mediana ~62 km/h). |
| Col 1, 2 | Desvío 0,12 y 0,06; autocorrelación 0,996 y 0,98. Velocidad lateral y vertical (m/s). |
| Col 3, 4 | Desvío 0,017 y 0,018; autocorrelación 0,86. Roll y pitch rate (rad/s), con ruido. |
| Col 5 | Desvío 0,060, p99 0,24 rad/s (14°/s); autocorrelación 0,999. **Yaw rate en rad/s.** El giro acumulado en 3 s es de 9° en el p90 y de 50° en el p99. |
| Por verificar | El signo del yaw (izquierda o derecha) y el marco de referencia. Se verifica en la Fase 1 contra el flow horizontal. |

**Problema para la Fase 1:** el val oficial (`val.zip`, 100 segmentos) **no tiene poses**, así que no se puede armar sobre él el set balanceado entre rectas y curvas. Hay tres opciones:

- **(a) El holdout de train** (50 segmentos con poses). Ya se usó para elegir la temperatura, y 4 de sus 50 rutas aparecen también en train.
- **(b) Bajar `data-0001.tar.gz`** (~500 MB, `python -m src.data --val-poses`, ya implementado). Arma un val con poses y sin rutas de train ni del val oficial.
- **(c) Estimar el yaw del val oficial a partir del flow de RAFT**, sin poses. Así no se puede validar la métrica de ego-motion.

## 3. Qué del plan ya está hecho y qué no

| Ítem | Estado | Resultado existente |
|---|---|---|
| 1.1 Tasa de cambio de tokens | **Hecho** (`src/metrics_motion.py`, `results/motion/`) | Falta gpt2m con 400 clips. |
| 1.2 \|flow\| con RAFT y error de dirección | Falta | |
| 1.3 Ego-motion contra el yaw | Falta | Requiere poses en el set de evaluación. |
| 2A Arrancar del frame t−1 y re-enmascarar | Falta | |
| 2B Temperatura y `choice_temp` | **Hecho para el FVD**, sin medir movimiento | MaskGIT: T 0,9–1,0 es lo mejor; `choice_temp` 4,5 no mejora. Schedule lineal y confianza sin Gumbel: falta. |
| 2B GPT: temperatura | **Hecho para el FVD** | T=0,9. Top-k y top-p: falta. |
| 2C Barrido de pasos (1/2/4/8/12) | **Hecho** (en el paper) | El movimiento cae de forma monótona con los pasos (abajo). Falta 16 y la curva de movimiento contra ms. |
| 3A Loss pesada por movimiento | **Parcial** | `peso3` (w=3, de cero, sin normalizar): movimiento p8 de 0,24 a 0,47, pero FVD 3 s de 380 a 683. Falta el barrido 2/4/8 como ajuste fino normalizado. |
| 3B Contexto corrompido con tokens al azar | **Variante distinta hecha** | Con reconstrucciones del propio modelo, no tokens al azar: `pself` 0,5 se mueve mucho y es poco realista; `ft_l_cb_ss` 0,25 queda más estable pero más quieto. Falta la versión con tokens al azar. |
| 3C Condicionamiento por pose | Falta para MaskGIT y GPT | Solo lo usan los modelos latentes (`lewm`). |
| 3D Cabeza auxiliar de movimiento | Falta | |
| Arquitectura (ya probado, negativo) | **Hecho** | `motion_in`, `motion_cb`, transformer de flujo de cero y como ajuste fino, `flow_limpio` + scheduled sampling: ninguno le gana a su control (ver `results/v2/`). |
| 4 Destilación on-policy | Falta | |

## 4. Lo que ya dicen los datos sobre las hipótesis

Fracción de tokens que cambian respecto del frame anterior, con 400 clips del val salvo donde se indica. Real: 0,66 en el frame 1 y ~0,60 en promedio.

| Modelo | Frame 1 | Frame 5 | 0–1 s | 1–2 s | 2–3 s | Ratio 2–3 s |
|---|---|---|---|---|---|---|
| gpt2m (8 clips) | 0,62 | 0,63 | 0,60 | 0,62 | 0,64 | 1,07 |
| gpt_s_kd (T 0,9) | 0,51 | 0,43 | 0,36 | **0,16** | **0,10** | 0,17 |
| gpt_s_d050 (de cero, ½ presupuesto) | 0,55 | 0,52 | 0,49 | 0,42 | 0,38 | 0,63 |
| MaskGIT p1 | 0,62 | 0,58 | 0,57 | 0,56 | 0,57 | 0,94 |
| MaskGIT p2 | 0,49 | 0,42 | 0,42 | 0,41 | 0,41 | 0,68 |
| MaskGIT p4 | 0,38 | 0,26 | 0,26 | 0,25 | 0,24 | 0,40 |
| MaskGIT p8 | **0,30** | 0,16 | 0,16 | 0,15 | 0,15 | 0,24 |
| MaskGIT p12 | 0,27 | 0,12 | 0,13 | 0,11 | 0,11 | 0,18 |
| Genie-S p4 | 0,57 | 0,44 | 0,39 | 0,30 | 0,28 | 0,48 |

**Lectura preliminar, que la Fase 1 tiene que confirmar con flow:**

- **MaskGIT: el problema está desde el frame 1 y casi no crece con el horizonte.** No es drift: es el sampler o el modelo. La relación con los pasos es monótona y fuerte (0,62 con p1 y 0,27 con p12 en el frame 1), lo que apoya **H2**: el decodificado por confianza fija primero los tokens que copian. Pero p1 tiene peor FVD, así que más pasos no es "peor" en todo sentido.
- **Alumno GPT: el frame 1 está cerca del real (0,51 contra 0,66) y el movimiento se apaga con el horizonte** (de 0,36 a 0,10). Apoya **H3** (exposure bias o drift), no el sampler. El GPT-S de cero con ½ presupuesto decae mucho menos (de 0,49 a 0,38). **Esto contradice en parte la idea de que destilar ayuda al movimiento**, aunque esa comparación mezcla presupuesto y destilación.
- **H1** (la CE premia copiar) tiene evidencia parcial: pesar los tokens que cambian duplica el movimiento de MaskGIT, pero empeora el FVD. La loss contribuye, pero no es la causa principal en MaskGIT, porque con 1 paso el mismo modelo se mueve como el real.
- **Advertencia:** la tasa de cambio de tokens no distingue movimiento de parpadeo del codec. Un modelo que cambia tokens al azar da ratio 1. Por eso la métrica de flow es necesaria antes de sacar conclusiones.

## 5. Decisiones que necesito antes de la Fase 1

1. **El set balanceado entre rectas y curvas:** opción (a), (b) o (c) de la sección 2. Recomiendo **(b)**: son ~500 MB de descarga y unos minutos de extracción, y es el único set limpio con poses.
2. **N de clips:** propongo **200** (100 rectas y 100 curvas, por el yaw acumulado en 3 s). RAFT-small sobre 200 × 60 frames por modelo son unos minutos en la 4060.
3. **gpt2m con 400 clips (o los N nuevos):** cuesta ~291 ms por frame en la 3080 Ti, unos 0,5 s en la 4060. Con 200 clips × 60 frames en batch 1, ~1,7 h; con batch es bastante menos. Pide confirmación porque corre el maestro.
