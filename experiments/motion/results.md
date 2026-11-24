# Experimentos de movimiento: resultados

Set de evaluación: `results/clips_motion.json` (200 clips de data-0001: 100 rectas y 100 curvas; detalles en
`01_diagnosis.md`), semilla 0, fp16.

**Métrica principal: giro sostenido.** Es el flow horizontal alineado con el sentido del giro real de cada curva, a
2-3 s, relativo al real: 1 = gira como el auto, 0 = no gira o parpadea al azar. La acompañan:
- |flow| y tasa de cambio de tokens por horizonte;
- FVD/LPIPS a 1/2/3 s (en A–F, con la diferencia pareada por bootstrap contra el control);
- ms/frame (B = 1, GPU del paper).

Piso del codec: el 13,5 % de tokens cambiados equivale a 0,17 px de |flow| y 0,03 px de flow horizontal.

## Fase 1: línea de base (sin cambios)

Para cada modelo, la fila "curvas" trae el giro sostenido y la fila "rectas" trae "–". En las columnas de |flow| y
tokens, cada celda es "valor en curvas / valor en rectas", ambos relativos al real.

| experimento | cambio | giro f1 / 0-1 / 2-3 s (curvas) | \|flow\| 2-3 s /real (curvas / rectas) | tokens 2-3 s /real (curvas / rectas) | FVD 1/2/3 s | LPIPS 1/2/3 s | ms/frame | conclusión |
|---|---|---|---|---|---|---|---|---|
| gpt2m_motion | maestro, T 1,0 | 0,99 / 0,96 / **0,79** | 0,92 / 0,95 | 1,00 / 1,07 | 138 / 184 / 220 | 0,216 / 0,293 / 0,344 | 291 | El único que gira y lo sostiene; el techo de referencia. |
| gpt_s_kd_motion | alumno kd, T 0,9 | 0,43 / 0,19 / 0,03 | 0,13 / 0,19 | 0,17 / 0,16 | 424 / 626 / 721 | 0,373 / 0,430 / 0,458 | 38 | Arranca girando bien y se apaga: congela la escena (sesgo de exposición). |
| gpt_s_d050_motion | GPT-S directo, T 0,9 | −0,16 / −0,01 / 0,00 | 0,33 / 0,56 | 0,59 / 0,64 | 471 / 570 / 611 | 0,396 / 0,464 / 0,511 | 38 | Se mueve, pero sin dirección: es parpadeo. |
| genie_s_p4_motion | Genie-S, 4 pasos | −0,13 / 0,00 / 0,00 | 0,24 / 0,42 | 0,39 / 0,44 | 479 / 571 / 670 | 0,351 / 0,417 / 0,454 | 6 | No gira. |
| mg_p1_motion | MaskGIT-S, 1 paso | −0,24 / −0,02 / −0,02 | 0,43 / 0,78 | 0,82 / 0,95 | 419 / 529 / 762 | 0,393 / 0,493 / 0,575 | 4 | Iguala al real en tokens, pero no gira: es parpadeo, no movimiento. |
| mg_p2_motion | MaskGIT-S, 2 pasos | −0,22 / −0,02 / −0,01 | 0,34 / 0,58 | 0,59 / 0,67 | 407 / 509 / 731 | 0,379 / 0,482 / 0,567 | 5 | No gira. |
| mg_p4_motion | MaskGIT-S, 4 pasos | −0,19 / −0,01 / 0,00 | 0,22 / 0,39 | 0,34 / 0,38 | 430 / 506 / 674 | 0,361 / 0,458 / 0,529 | 6 | No gira. |
| mg_p8_motion | MaskGIT-S, 8 pasos | −0,17 / −0,01 / 0,00 | 0,15 / 0,26 | 0,20 / 0,23 | 454 / 469 / 559 | 0,346 / 0,428 / 0,485 | 10 | No gira y casi no cambia (tokens en el piso del codec); el mejor FVD a 3 s entre los chicos. |

Figuras:
- `figs/fig_fase1_giro.png`: giro alineado por horizonte.
- `figs/fig_fase1_flow.png`: |flow| por horizonte.
- `figs/fig_fase1_tok.png`: tasa de cambio de tokens por horizonte.
- `figs/fig_fase1_vs_ms.png`: giro sostenido contra ms/frame, con la línea de 50 ms.

## Prioridades A–F

(pendiente)
