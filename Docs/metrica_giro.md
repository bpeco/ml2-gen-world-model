# Métrica de giro: ¿el video generado gira como el auto?

Un proxy simple del ego-motion lateral (yaw) para world models de manejo a baja resolución (commaVQ, 128×256). No
entrena nada: usa RAFT-small preentrenado sobre los frames decodificados y se valida contra la pose del dataset. No es la
primera métrica de ego-motion para world models de manejo (ver [Trabajo relacionado](#7-trabajo-relacionado)). Lo que
aporta es que es barata y reproducible, que separa el giro del parpadeo del codec y que su validación con la pose está a
la vista.

Código: [src/metrica_giro.py](../src/metrica_giro.py) (métrica, IC y sensibilidad) y
[src/metrics_flow.py](../src/metrics_flow.py) (flow por frame). Tests:
[tests/test_metrica_giro.py](../tests/test_metrica_giro.py). Set de clips:
[src/clips_motion.py](../src/clips_motion.py) → `results/clips_motion.json`.

## 1. Definición

**Set de clips.** 200 clips de 20 frames de contexto + 60 de rollout (3 s a 20 FPS), de data-0001 de commaVQ, sin
rutas de train ni del val oficial, uno por ruta. La pose (`pose.npy`, velocidades por frame) da el yaw rate (col 5,
rad/s). El giro de la pose en el rollout es yaw_3s = Σ yaw_rate / 20 (en grados).
- **Curvas (100):** |yaw_3s| ≥ 20° y velocidad media ≥ 8 m/s, 50 a cada lado. Segmentos y arranques al azar
  (semilla 0).
- **Rectas (100):** |yaw_3s| < 1,5° y |giro en el contexto| < 1,5°. A cada curva se le asigna la recta de velocidad
  más parecida, porque el |flow| crece con la velocidad. Las rectas se usan para |flow| y tokens, no para el giro.

**Flow horizontal por frame.** Para cada par de frames consecutivos del rollout (el primero contra el último frame del
contexto), RAFT-small (torchvision, 12 iteraciones) sobre los frames decodificados a 128×256 da el campo (u, v). Se
toma

  u_t = media de u en la franja superior de 64 filas (de 128).

La franja superior está lejos del capó y tiene poca paralaje. Ahí el flow horizontal lo domina la rotación de la
cámara: un yaw de ω rad/s mueve toda la escena lejana ≈ f·ω/20 px por frame, sin importar la profundidad. Signo: yaw
positivo (giro a la derecha, en el marco del dispositivo con z hacia abajo) → la escena se corre a la izquierda → u < 0.

**Alineación con el sentido del giro real.** Para cada curva, s = signo(Σ_t u_t^real) en los 60 frames, y

  giro_t = s · u_t.

Es positivo si la imagen se mueve como en el giro real y negativo si gira al revés. El parpadeo sin dirección da 0 en
promedio.

**Normalización y agregación.** Para un tramo de frames [a, b):

  giro[a, b) = ( media_{curvas} media_{t∈[a,b)} giro_t^modelo ) / ( media_{curvas} media_{t∈[a,b)} giro_t^real )

Es un cociente de medias, no la media de cocientes por clip: los clips con poco giro real harían explotar los
cocientes. 1 = gira tanto como el real, en promedio; 0 = no gira. Se reporta en el frame 1 y en los tramos 0-1, 1-2 y
2-3 s. El **giro sostenido** es el tramo 2-3 s.

**Intervalo de confianza.** Bootstrap por clip (1000 réplicas): se remuestrean las curvas con reposición, las mismas en
el numerador y el denominador, y se toman los percentiles 2,5 y 97,5 del cociente.

**Lado correcto.** La fracción de curvas en que signo(Σ_t u_t^modelo) = s. 0,5 = azar.

## 2. Por qué no alcanza con lo obvio

- **El promedio con signo crudo se cancela.** Hay 50 curvas a cada lado, así que la media de u_t sobre las curvas es
  ≈ 0 también en el real. Hay que alinear cada clip con su propio sentido de giro.
- **|flow| no distingue giro de parpadeo.** El parpadeo del codec también produce flow, solo que sin dirección
  coherente. En la Fase 1, MaskGIT p1 tiene casi el mismo |flow| en rectas que en curvas (0,65 / 0,69 px), cuando el
  real pasa de 0,85 a 2,0 px. Su |flow| en curvas es el 29-43 % del real, pero su giro es 0.
- **La tasa de cambio de tokens engaña todavía más.** Re-codificar un frame real (sin movimiento) cambia el 13,5 % de
  los tokens. MaskGIT p1 cambia tantos tokens como el real (0,55-0,58 frente a 0,57-0,76), con un giro de −0,02 a 2-3 s
  y el lado correcto en el 45 % de las curvas. GPT-S d050 sostiene 0,4-0,6 de tasa de tokens con giro 0.

| modelo (curvas, 2-3 s) | tokens / real | \|flow\| / real | giro | lado correcto |
|---|---|---|---|---|
| GPT2M | 1,00 | 0,92 | 0,79 [0,53; 1,10] | 0,95 |
| GPT-S kd (T 0,9) | 0,17 | 0,13 | 0,03 [0,01; 0,06] | 0,68 |
| GPT-S d050 (T 0,9) | 0,59 | 0,33 | 0,00 [−0,02; 0,02] | 0,49 |
| MaskGIT p1 | 0,82 | 0,43 | −0,02 [−0,04; 0,00] | 0,45 |
| MaskGIT p8 | 0,20 | 0,15 | 0,00 [−0,01; 0,01] | 0,47 |

(Figura: `experiments/motion/figs/fig_metrica_giro.png`, ver sección 8.)

## 3. Validación ya hecha

- **Contra la pose (real):**
  - corr(u_t, yaw rate por frame) = −0,77, con el signo esperado.
  - En las curvas, la corr. entre Σ u_t en 3 s y yaw_3s es −0,97 (−0,96 sobre los 200 clips).
- **Desfasaje:** la correlación por frame es máxima en el desfasaje 0 y casi plana en ±4 frames (el yaw es suave).
  No hay corrimiento entre la pose y los tokens.
- **Piso del codec:** RAFT entre un frame y su re-codificación (x vs dec(enc(x)), nada se mueve) da |flow| 0,17 px y
  flow horizontal 0,03 px. Es el 3 % del flow horizontal real entre frames consecutivos (1,15 px).
- **Proxy de avance:** la expansión radial no funciona (corr. con la velocidad 0,05) y no se usa.

## 4. Tests de sanidad

`python -m pytest tests/test_metrica_giro.py -q` (RAFT y el decoder sobre 16 curvas reales, 8 a cada lado; necesita
CUDA; tarda ~80 s en la RTX 4060). Cada test construye un video con el movimiento conocido. Pasan los 5.

| test | construcción | se espera | resultado |
|---|---|---|---|
| desplazamiento lineal | un frame real corrido k px por frame (k = 1, 2, 4), recortado de un lienzo espejado | u ≈ k; giro(k)/giro(1) ≈ k | u = 1,03 / 2,03 / 4,11 px; giro(2)/giro(1) = 1,97 y giro(4)/giro(1) = 4,00 |
| tiempo invertido | el clip real al revés | giro ≈ −1 | −1,00 [IC −1,08; −0,94] |
| congelado | el último frame del contexto repetido | giro ≈ 0 | 0,0002 |
| parpadeo puro | escena fija con el 30 % de los tokens re-muestreados al azar en cada frame (tasa de cambio 0,51; real 0,63) | IC que contiene el 0 | −0,03 [IC −0,56; 0,69] |
| flip horizontal | "modelo" = real a media velocidad (cada frame repetido dos veces); modelo y real espejados | giro entre 0,3 y 0,7 e igual con y sin flip | 0,63 sin flip y 0,62 con flip |

Notas:
- **El "modelo lento" da 0,63, no 0,5.** En 3 s recorre solo la primera mitad del clip real, y el giro real no es
  uniforme en el tiempo. El test pide simetría, no el valor exacto.
- **El parpadeo puro tiene un IC muy ancho.** RAFT sobre tokens al azar devuelve ~2,4 px de flow horizontal por par
  (el real tiene 1,3) y un sesgo distinto en cada clip. En promedio no hay giro, pero con pocas curvas un modelo que
  parpadea mucho puede dar un valor puntual alejado de 0.
- **Primera versión del test, fallida:** usaba 8 curvas, todas del mismo lado, y dio 0,72. Ver "Balance de lados" en
  las limitaciones.

## 5. Sensibilidad

`python -m src.metrica_giro --sens --names ...` → `results/motion/giro.json` (clave `sensibilidad`). Mide el giro a
2-3 s de los 8 modelos de la Fase 1.

| variante | n curvas | GPT2M | GPT-S kd | d050 | Genie p4 | MaskGIT p1 | p2 | p4 | p8 |
|---|---|---|---|---|---|---|---|---|---|
| franja 32 filas (≥20°) | 100 | 0,80 | 0,03 | −0,00 | −0,01 | −0,02 | −0,01 | −0,00 | 0,00 |
| **franja 64 filas (≥20°, la definición)** | 100 | **0,79** | **0,03** | −0,00 | −0,00 | −0,02 | −0,01 | −0,00 | 0,00 |
| franja 128 filas (≥20°) | 100 | 0,73 | 0,02 | −0,00 | −0,00 | −0,01 | −0,01 | 0,00 | 0,00 |
| umbral 10° (set principal + 50 curvas de 10-20°) | 150 | 0,74 | 0,03 | −0,00 | −0,00 | −0,01 | −0,00 | 0,00 | 0,00 |
| umbral 20° | 100 | 0,79 | 0,03 | −0,00 | −0,00 | −0,02 | −0,01 | −0,00 | 0,00 |
| umbral 30° | 36 | 0,72 | 0,04 | 0,00 | −0,00 | −0,02 | −0,01 | −0,00 | 0,00 |

**El ranking no cambia: GPT2M > kd > el resto** en las 6 variantes. Los otros 6 modelos quedan entre −0,02 y +0,00 y
se reordenan entre sí, pero sus diferencias están dentro del IC (±0,02), así que es ruido y no un cambio de ranking.

- **Franja:** con 128 filas (toda la imagen) el giro de GPT2M baja de 0,79 a 0,73. La mitad inferior agrega paralaje
  y capó, que diluyen el flow horizontal del giro. Con 32 filas da casi lo mismo que con 64. La franja de 64 filas
  calculada de nuevo coincide exactamente con la caché de `src.metrics_flow`.
- **Umbral:** las curvas de 10-20° vienen de `results/clips_motion_suave.json`: 50 curvas (25 por lado) de rutas que no
  están en el set principal, |yaw| de 10,1 a 19,4° (mediana 12,4°).
  - **Diferencia de velocidad:** su velocidad mediana es 12,9 m/s, contra 9,6 m/s en el set principal. No hay rectas
    para emparejar, pero el giro es relativo al real del mismo clip.
  - **GPT2M:** sumar esas curvas lo baja de 0,79 a 0,74.
  - **kd:** sigue en 0,03.

## 6. Limitaciones

- **Balance de lados.** El flow horizontal puede tener un sesgo constante hacia un lado que no es giro:
  - RAFT sobre artefactos deriva hacia un lado.
  - Los modelos chicos derivan en las rectas de −0,10 a −0,16 px respecto del real; GPT2M, −0,01.

  Con curvas balanceadas, ese sesgo se cancela al multiplicar por s. Sin balance, se suma al giro: el primer test de
  parpadeo, con 8 curvas todas a la izquierda, dio 0,72.

  Los sets de evaluación están balanceados (50/50 en el principal y 25/25 en el de 10-20°). Ahí el giro agregado
  coincide con el promedio de los dos lados (±0,01): por ejemplo, para kd el lado derecho da +0,10 y el izquierdo
  −0,03. **Cualquier set nuevo tiene que estar balanceado por lado**, o reportar el giro como el promedio de los
  lados.
## 6. Limitaciones

- **Solo mide el yaw.** No mide el ego-motion longitudinal: la expansión no funcionó (corr. 0,05 con la velocidad), y
  un modelo que gira bien pero no avanza no se detecta.
- **No mide a los otros agentes** (autos, peatones).
- **RAFT tiene su propio error** sobre frames reconstruidos por el codec (borrosos, con parpadeo). El piso de 0,03 px
  horizontal acota el sesgo sin movimiento, pero no el error con movimiento grande.
- **Es un cociente de medias:** dice cuánto gira el modelo en promedio, no si cada clip gira bien. Para eso están el
  lado correcto y la corr. por clip.
- **100 curvas.** Los IC a 2-3 s son anchos para los modelos que giran (GPT2M: [0,53; 1,10]) y angostos para los que
  no (≈ ±0,02).
- **Es relativa al real del mismo clip.** No dice nada en clips sin pose ni en curvas más suaves que el umbral.

## 7. Trabajo relacionado

Leídos en arXiv (noviembre 2026). Las descripciones son las de cada paper.

| métrica | qué mide | ¿dirección? | ¿modelo entrenado? | diferencia con la nuestra |
|---|---|---|---|---|
| VBench, Dynamic Degree ([2311.17982](https://arxiv.org/abs/2311.17982)) | si el video es estático o no: media del 5 % superior de \|flow\| de RAFT entre frames, umbralizada; el puntaje es la fracción de videos no estáticos | no (solo magnitud) | no (RAFT preentrenado) | Es nuestro "\|flow\|" binarizado. No distingue giro de parpadeo ni compara con el real. |
| FVMD ([2407.16124](https://arxiv.org/abs/2407.16124)) | distancia de Fréchet entre histogramas de velocidad y aceleración de 400 puntos seguidos con PIPs++ (ventanas de 16 frames, 256×256) | en parte (histograma de ángulos en 8 bins) | no (PIPs++ preentrenado), pero necesita un set de referencia | Es una distancia entre distribuciones, no por clip: no dice si el modelo gira para el lado del auto. |
| Vista, Trajectory Difference ([2405.17398](https://arxiv.org/abs/2405.17398)) | L2 entre la trayectoria 2D estimada del video generado y la real, en 2 s | sí (posiciones 2D) | sí: un modelo de dinámica inversa entrenado (según ACT-Bench, no público) | Sirve para condicionar en la acción; no es reproducible sin el estimador. |
| ACT-Bench ([2412.05337](https://arxiv.org/abs/2412.05337)) | ACT-Estimator (I3D + transformer) predice la clase de acción y la trayectoria; consistencia de la instrucción + ADE/FDE | sí | sí: entrenado en nuScenes (cámara frontal); la publicación del estimador está anunciada | Hay que entrenar un estimador en otro dominio y resolución (nuScenes). |
| GEM ([2412.11198](https://arxiv.org/abs/2412.11198)) | ADE de la trayectoria estimada con DROID-SLAM + Depth Anything V2 + GeoCalib contra la real | sí (implícito en ADE) | no entrena, pero usa SLAM + profundidad métrica | El SLAM sobre 128×256 con parpadeo del codec es frágil, y es caro. |
| DrivingGen ([2601.01528](https://arxiv.org/abs/2601.01528)) | "Trajectory Alignment": ADE y DTW entre la trayectoria extraída (PnP con SIFT + RANSAC y UniDepthV2; SLAM + profundidad en el apéndice) y la de entrada | sí (implícito) | no entrena; geometría + profundidad preentrenada | Mismo problema que GEM a baja resolución. El paper reporta que la alineación de trayectoria es débil, en parte por artefactos del video que rompen el SLAM. |
| VehDyn ([2609.33264](https://arxiv.org/abs/2609.33264)) | odometría visual (DPVO) → ADE/FDE, RMSE de velocidades y de roll/pitch/yaw, y consistencia dinámica (círculo de fricción) en un set simulado CARLA-CarSim | sí (incluye yaw) | no entrena (DPVO preentrenado); el set y las métricas se publican después de la aceptación | Lo más cercano en espíritu (mide el yaw contra la verdad), pero sobre video simulado de alta resolución y con odometría visual completa. |

La diferencia nuestra es de escala y de supuestos, no de idea: un solo número físico (flow horizontal lejano), sin
odometría ni profundidad, que funciona con frames de 128×256 decodificados de tokens. Se valida directamente contra la
pose del mismo dataset y viene con su piso de ruido (el codec). Lo que pierde frente a ADE/FDE: la escala métrica, el
avance y la trayectoria completa.

## 8. Figura

`experiments/motion/figs/fig_metrica_giro.png`: a la izquierda, el giro alineado por horizonte (real, GPT2M, GPT-S kd,
MaskGIT p1) en las curvas; a la derecha, la tasa de cambio de tokens de los mismos modelos con el piso del codec.
MaskGIT p1 está arriba en tokens y en 0 en giro.
