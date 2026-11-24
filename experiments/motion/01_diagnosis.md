# Fase 1: diagnóstico de movimiento (sin cambiar modelos)

Set: `results/clips_motion.json` (200 clips de data-0001, ninguna ruta de train ni del val oficial; 100 rectas y 100
curvas con |yaw| ≥ 20° en los 3 s del rollout, rectas emparejadas por velocidad, v ≈ 9,6 m/s). 20 frames de contexto
+ 60 de rollout, semilla 0, fp16. Flow con RAFT-small sobre los frames decodificados (`src/metrics_flow.py`).
Figuras: `figs/fig_fase1_*.png` (generadas con `figuras.py`).

## Resumen

1. **Solo el maestro GPT2M se mueve como el auto real.** En curvas mantiene el 96 % del giro real en el primer
   segundo y el 79 % a 2-3 s; el 95 % de las curvas giran para el lado correcto.
2. **Ningún MaskGIT ni Genie gira, con ningún número de pasos.** El flow horizontal alineado con el giro real es
   0 ± 0,02 desde el frame 2 hasta el 60, y aciertan el lado del giro en el 45-49 % de las curvas (azar).
3. **MaskGIT p1 no tiene movimiento real, solo parpadeo.** Cambia tantos tokens como el real (0,55 en rectas y 0,57 en
   curvas, contra 0,57/0,72), pero:
   - su |flow| es casi igual en rectas (0,65 px) que en curvas (0,69 px), cuando en el real pasa de 0,85 a 2,0 px;
   - el error de dirección del flow es de ~90° (azar);
   - su giro acumulado no se correlaciona con el yaw de la pose (+0,32, signo equivocado; el real da −0,97).

   Más pasos solo bajan el parpadeo (p8: 0,14 de tokens, igual al piso del codec), no agregan dirección.
4. **El alumno GPT-S kd arranca girando y se apaga.** En el frame 1 tiene el 43 % del giro real con la dirección
   correcta (error de 61° contra ~90° de los otros chicos, corr. con el yaw −0,68). Después decae a 13 % en 0,5 s, 7 %
   a 1-2 s y 3 % a 2-3 s, y la tasa de tokens cae por debajo del piso del codec (0,09): congela la escena. Es la firma
   del sesgo de exposición (H3).
5. **GPT-S d050 se mueve, pero sin dirección.** Su |flow| se sostiene (0,5 px) y su tasa de tokens no colapsa (0,38),
   pero el giro alineado es 0 y la corr. con el yaw es 0,03: es parpadeo como el de MaskGIT p1/p2, no movimiento.
6. **El problema está desde el frame 1 en todos los chicos.** En el frame 1, el |flow| va del 30 % (p8) al 53 % (kd)
   del real, contra el 96 % de GPT2M. Además:
   - MaskGIT, Genie y d050 tienen el giro del frame 1 **negativo** (−13 a −24 % del real): el primer frame generado
     retrocede respecto al último del contexto.
   - Solo kd empeora claramente con el horizonte; el resto ya está en su meseta a los 0,3 s.
7. **La tasa de cambio de tokens engaña; el flow no.** El piso del codec (re-codificar un frame real) cambia el 13,5 %
   de los tokens, pero solo mueve 0,17 px de flow y 0,03 px en horizontal. MaskGIT p1 iguala al real en tokens y tiene
   giro cero.
8. **FVD en este set tiene mucho ruido** (IC del 95 % de ±60-110). Entre los modelos chicos los intervalos se solapan
   a 1 y 2 s. A 3 s p8 (559) le gana a p1 (762), en la misma dirección que el paper. GPT2M está muy separado
   (138/184/220).

**Consecuencia para las prioridades:**
- **MaskGIT (A/B):** el objetivo no es que cambie más tokens. Lo que falta es que el cambio tenga dirección (giro
  alineado > 0). La métrica principal de A–F pasa a ser el **giro sostenido** (flow horizontal alineado, 2-3 s, sobre
  el real), acompañado de |flow| y tokens.
- **Alumno (C/D):** el objetivo es que el giro del frame 1, que ya existe, no se apague.

## Tabla (curvas; rectas en `results.md`)

En la tabla, "giro" es el flow horizontal alineado con el sentido del giro real de cada clip, relativo al real.

| modelo | ms/frame | giro f1 | giro 0-1 s | giro 1-2 s | giro 2-3 s | lado correcto | corr. giro 3 s con yaw | \|flow\| 0-1/1-2/2-3 s (/real) | tokens 0-1/1-2/2-3 s (/real) | FVD 1/2/3 s |
|---|---|---|---|---|---|---|---|---|---|---|
| real | – | 1 | 1 | 1 | 1 | 1,00 | −0,97 | 2,40/1,91/1,65 px | 0,76/0,73/0,69 | – |
| GPT2M | 291 | 0,99 | 0,96 | 0,85 | 0,79 | 0,95 | −0,84 | 0,96/0,90/0,92 | 1,00/1,00/1,00 | 138/184/220 |
| GPT-S kd (T 0,9) | 38 | 0,43 | 0,19 | 0,07 | 0,03 | 0,68 | −0,68 | 0,31/0,19/0,13 | 0,66/0,32/0,17 | 424/626/721 |
| GPT-S d050 (T 0,9) | 38 | −0,16 | −0,01 | 0,01 | 0,00 | 0,49 | 0,03 | 0,29/0,29/0,33 | 0,78/0,64/0,59 | 471/570/611 |
| Genie-S p4 | 6 | −0,13 | 0,00 | 0,00 | 0,00 | 0,47 | −0,01 | 0,22/0,21/0,24 | 0,59/0,40/0,39 | 479/571/670 |
| MaskGIT p1 | 4 | −0,24 | −0,02 | 0,00 | −0,02 | 0,45 | 0,32 | 0,29/0,35/0,43 | 0,76/0,77/0,82 | 419/529/762 |
| MaskGIT p2 | 5 | −0,22 | −0,02 | 0,00 | −0,01 | 0,46 | 0,28 | 0,22/0,27/0,34 | 0,56/0,55/0,59 | 407/509/731 |
| MaskGIT p4 | 6 | −0,19 | −0,01 | 0,00 | 0,00 | 0,49 | 0,28 | 0,16/0,18/0,22 | 0,35/0,32/0,34 | 430/506/674 |
| MaskGIT p8 | 10 | −0,17 | −0,01 | 0,00 | 0,00 | 0,47 | 0,30 | 0,11/0,12/0,15 | 0,22/0,19/0,20 | 454/469/559 |

Notas sobre las columnas:
- **ms/frame:** B = 1, GPU del paper (`results/summary.json`). En la RTX 4060 del laptop: MaskGIT p1/p2/p4/p8
  9/10/13/18 ms, Genie p4 10 ms, GPT-S 131 ms, GPT2M 916 ms. d050 usa la latencia de kd, que tiene la misma
  arquitectura.
- **"lado correcto"** es el signo del flow horizontal acumulado en 3 s comparado con el real.
- **FVD:** los 200 clips (I3D), con el IC del 95 % en `results/video_<name>_motion.json`.

## Detalles y advertencias

- **Signo del yaw verificado:** corr(flow horizontal real, yaw rate) = −0,77, plana entre los desfasajes −4 y +4 con el
  máximo en 0. Yaw positivo = giro a la derecha.
- **"giro alineado":** flow horizontal (franja superior de 64 filas) multiplicado por el signo del giro acumulado
  *real* de ese clip, promediado sobre las 100 curvas. El promedio con signo crudo se cancela, porque hay 50 curvas
  por lado, y |flow horizontal| suma giro y parpadeo. Un modelo que parpadea da 0; uno que gira para el lado contrario
  da negativo.
- **Frame 1 negativo** en MaskGIT, Genie y d050. Es una observación, no está explicada todavía. La hipótesis más simple
  es que el primer frame se parece más a un frame algo anterior del contexto (o a un promedio del contexto) que al
  último, y por eso el flow del último contexto al frame 1 apunta hacia atrás. No lo persigo salvo que A/B lo
  necesiten.
- **Picos de |flow|** en GPT2M (rectas a ~0,7 s, curvas a ~2,1 s) y en el real (rectas a ~2 s): son clips puntuales con
  flow enorme (cortes o reflejos). Afectan las medias por frame, no las conclusiones por tramo.
- **Expansión (proxy de avance):** no sirve. Su corr. con la velocidad de la pose es 0,05, así que no se usa.
- **ms/frame del rollout por lotes** (`ms_per_frame_batch`, B = 50 en la 4060) está en
  `results/rollouts/<name>.json`, pero no se usa para el criterio de tiempo real.
