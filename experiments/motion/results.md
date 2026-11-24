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

## 0. ¿A qué se parece el frame 1 generado? (sin entrenar)

`experiments/motion/diag_frame1.py` → `results/motion/diag_frame1.json`. Curvas (n=100). El frame 1 decodificado se
compara con los frames de contexto decodificados t-1 (el último), t-2, …, con el promedio en píxeles de t-1 y t-2 y con
el promedio de los 20 frames del contexto.

**Distancias:** "Δ" es la distancia a la media(t-1, t-2) menos la distancia a t-1. Si es negativo, el frame se parece
más a la mezcla que al último frame. En el real, la mezcla siempre queda más lejos (Δ > 0).

| modelo | LPIPS t-1 | LPIPS t-2 | LPIPS media(t-1,t-2) | ΔLPIPS (mezcla − t-1) | L1 t-1 | L1 media(t-1,t-2) | ΔL1 | L1 media del contexto | giro f1 |
|---|---|---|---|---|---|---|---|---|---|
| real | 0,101 | 0,136 | 0,125 | +0,024 | 9,47 | 9,58 | +0,11 | 15,9 | 1 |
| GPT2M | 0,101 | 0,135 | 0,125 | +0,024 | 9,50 | 9,59 | +0,09 | 16,0 | 0,99 |
| GPT-S kd | 0,121 | 0,158 | 0,146 | +0,025 | 9,87 | 10,31 | +0,44 | 16,8 | 0,43 |
| GPT-S d050 | 0,147 | 0,165 | 0,157 | +0,010 | 11,31 | 11,00 | **−0,31** | 16,4 | −0,16 |
| Genie-S p4 | 0,142 | 0,161 | 0,153 | +0,011 | 11,64 | 11,30 | **−0,34** | 16,7 | −0,13 |
| MaskGIT p1 | 0,149 | 0,168 | 0,157 | +0,008 | 12,11 | 11,67 | **−0,44** | 15,7 | −0,24 |
| MaskGIT p4 | 0,116 | 0,143 | 0,129 | +0,013 | 10,06 | 9,91 | **−0,15** | 15,7 | −0,19 |
| MaskGIT p8 | 0,099 | 0,132 | 0,113 | +0,014 | 8,89 | 9,08 | +0,19 | 15,8 | −0,17 |

**Conclusiones:**

1. **Ningún modelo copia un frame viejo ni el promedio del contexto.** El frame 1 está siempre más cerca de t-1 que de
   t-2 o de cualquier otro frame, y lejísimos del promedio del contexto (LPIPS ~0,37 contra ~0,10-0,15).

2. **El giro negativo del frame 1 se explica como una mezcla de t-1 y t-2.** Los modelos con giro negativo en el frame
   1 (d050, Genie, MaskGIT) quedan relativamente mucho más cerca de esa mezcla que el real:
   - ΔLPIPS de +0,008 a +0,014 contra +0,024 del real.
   - En L1 directamente están más cerca de la mezcla que de t-1, salvo p8.

   Con el giro de −0,13 a −0,24 del real, o sea un "retraso" de 1/8 a 1/4 de frame, esto es consistente con que el
   frame 1 se arme en parte con lo que había en cada posición en t-2. Es una copia posicional de la historia en lugar
   de un desplazamiento del contenido, y el flow de t-1 al frame 1 apunta un poco hacia atrás.

   **Advertencia:** el L1 favorece a los frames borrosos frente a cualquier promedio. Por eso pesa más el ΔLPIPS, que
   castiga el borroneado y aun así muestra que la mezcla queda relativamente cerca.

3. **kd y GPT2M no tienen este efecto** (ΔLPIPS igual al real), y son los dos únicos con giro positivo en el frame 1.

## Prioridades A–F

Corridas: `scripts/motion_AC.sh`. Columnas como en la Fase 1. "|flow| / tokens" son de las curvas y van como "0-1 s /
2-3 s" en px y en fracción del real; el real da 1,62 / 1,27 px. "ΔFVD" es la diferencia pareada contra el control
(bootstrap por clip, `results/diferencia_*.json`), como "valor (p)" a 1/2/3 s; negativo = mejor.

### C. Temperatura y top-p de kd (sin entrenar)

Control: `gpt_s_kd_motion` (T 0,9). Mismo checkpoint, clips y semilla.

| experimento | cambio | giro f1 / 0-1 / 2-3 s [IC 2-3 s] | lado correcto | \|flow\| px / tokens | FVD 1/2/3 s | ΔFVD 1/2/3 s (p) | LPIPS 1/2/3 s | ms/frame | conclusión |
|---|---|---|---|---|---|---|---|---|---|
| gpt_s_kd_motion | control, T 0,9 | 0,43 / 0,19 / 0,03 [0,01; 0,06] | 0,68 | – / 0,16 | 424 / 626 / 721 | – | 0,373 / 0,430 / 0,458 | 38 | control |
| kdC_t10_motion | T 1,0 | 0,44 / 0,22 / 0,06 [0,02; 0,11] | 0,68 | 0,72 / 0,37 · 0,40 | 445 / 633 / 825 | +22 (0,33) / +7 (0,75) / **+104 (0,01)** | 0,409 / 0,479 / 0,505 | 38 | Un poco más de giro tardío; peor FVD a 3 s. |
| kdC_t11_motion | T 1,1 | 0,46 / 0,24 / **0,12** [0,05; 0,21] | 0,60 | 0,89 / 0,83 · 0,99 | 462 / 704 / 872 | +38 (0,17) / **+78 (0,03)** / **+151 (0,01)** | 0,452 / 0,542 / 0,586 | 38 | El giro a 2-3 s se cuadruplica, pero los tokens llegan al nivel del real y el lado correcto baja: es en gran parte parpadeo. |
| kdC_t10_p95_motion | T 1,0 + top-p 0,95 | 0,40 / 0,20 / 0,03 [0,01; 0,05] | **0,73** | 0,60 / 0,17 · 0,14 | 437 / 667 / 808 | +13 (0,55) / +41 (0,21) / **+87 (0,01)** | 0,379 / 0,444 / 0,470 | 38 | Mejor lado correcto, pero el giro se apaga igual que el control. |

**Conclusión C:**
- **El muestreo no arregla el apagado del giro.** La temperatura cambia la cantidad de cambio, no la dirección: a más T,
  más giro tardío, más tokens, menos lado correcto y peor FVD. Top-p hace lo contrario.
- **Se mantiene T 0,9 para kd** (mejor FVD a 3 s; el ΔFVD de todas las variantes es significativo a 3 s).
- **La temperatura de d050 también queda en 0,9.** Es el valor del paper y, por lo visto en kd, subirla no da
  dirección.
- **La hipótesis "kd pierde el giro por sesgo de exposición" sigue en pie**, ya que no se arregla en el muestreo.

### A. MaskGIT con contexto limpio (fine-tune de 1000 pasos desde maskgit_s_abs, igual que v2)

FT: `--presupuesto 320000 --lr 3e-4 --warmup 100 --seed 0`, unos 9 min solo en la GPU.
- **Control:** `mgA_ctrl`, el mismo fine-tune sin cambios. Hace falta porque el fine-tune solo ya mueve el FVD:
  mgA_ctrl contra mg_p8 da ΔFVD a 3 s de +81 (p 0,01) y contra mg_p4 da −24 (p 0,40).
- **p_clean 1,0 literal no es posible:** con atención causal por bloques, todo frame enmascarado tendría contexto
  limpio y no habría pérdida. Se implementó como `ctx_limpio`: un solo frame objetivo enmascarado por clip, con
  contexto 100 % limpio. Ve unas 20× menos tokens enmascarados por paso que el control.

| experimento | cambio | giro f1 / 0-1 / 2-3 s [IC 2-3 s] | lado correcto | \|flow\| px / tokens | FVD 1/2/3 s | ΔFVD 1/2/3 s (p) | LPIPS 1/2/3 s | ms/frame | conclusión |
|---|---|---|---|---|---|---|---|---|---|
| mgA_ctrl_p8_motion | control | −0,15 / −0,01 / 0,00 [−0,01; 0,01] | 0,47 | 0,25 / 0,25 · 0,22 | 443 / 493 / 641 | – | 0,346 / 0,428 / 0,480 | 10 | control |
| mgA_ctrl_p4_motion | control | −0,20 / −0,01 / 0,00 [−0,01; 0,01] | 0,48 | 0,36 / 0,36 · 0,36 | 437 / 502 / 650 | – | 0,356 / 0,438 / 0,496 | 6 | control |
| mgA_ctxl_p8_motion | ctx_limpio | −0,14 / −0,01 / +0,01 [0,00; 0,01] | 0,48 | 0,29 / 0,19 · 0,17 | 535 / 541 / 607 | **+92 (0,01)** / +48 (0,09) / −33 (0,20) | 0,340 / 0,399 / 0,431 | 10 | No gira. Peor FVD a 1 s. |
| mgA_ctxl_p4_motion | ctx_limpio | −0,14 / −0,01 / 0,00 [−0,02; 0,01] | 0,50 | 0,43 / 0,31 · 0,37 | 474 / 561 / 649 | +37 (0,11) / **+59 (0,05)** / 0 (0,94) | 0,360 / 0,421 / 0,456 | 6 | No gira. |
| mgA_pc08_p8_motion | p_clean 0,8 | −0,12 / −0,01 / 0,00 [−0,01; 0,01] | 0,47 | 0,24 / 0,17 · 0,14 | 477 / 509 / 543 | +34 (0,08) / +15 (0,72) / **−98 (0,01)** | 0,334 / 0,397 / 0,431 | 10 | No gira; mejora el FVD a 3 s. |
| mgA_pc08_p4_motion | p_clean 0,8 | −0,14 / −0,01 / 0,00 [−0,01; 0,01] | 0,51 | 0,35 / 0,26 · 0,26 | 428 / 479 / 525 | −9 (0,65) / −23 (0,44) / **−124 (0,01)** | 0,349 / 0,412 / 0,454 | 6 | No gira; el mejor FVD a 3 s de MaskGIT hasta ahora. |

**Conclusión A: contradice la hipótesis.**
- **Ninguna variante gira:** giro 2-3 s = 0 con IC dentro de ±0,02 y lado correcto ~0,5, o sea azar.
- **El frame 1 apenas pasa de −0,15…−0,20 a −0,12…−0,14.** Entrenar con contexto más limpio no le da a MaskGIT la
  dirección del movimiento: no es un problema de diferencia entre entrenamiento e inferencia, el modelo no aprendió
  el ego-movimiento.
- **Efecto lateral real: p_clean 0,8 mejora el FVD a 3 s** (−98/−124, p 0,01), con menos tokens cambiados. Es
  estabilidad (menos deriva), no movimiento.
- **No se aplica E a MaskGIT** (la condición era "si A funciona").

Figuras de C y A (`python experiments/motion/figuras.py --tag AC`):
- `figs/fig_AC_giro.png`: giro alineado por horizonte. Las 3 variantes de kd arrancan igual (~0,9 px en el frame 1)
  y se apagan; las de MaskGIT quedan en 0 desde el frame 3.
- `figs/fig_AC_vs_ms.png`: giro sostenido contra latencia. Los 3 MaskGIT se superponen en (6 ms, 0).
- `figs/fig_AC_tok.png` y `figs/fig_AC_flow.png`: tokens y |flow| por horizonte.

### Métrica: tests y sensibilidad

- **Los 5 tests de sanidad pasan.** Están en `tests/test_metrica_giro.py`; los valores, en `docs/metrica_giro.md` §4.
- **El ranking no cambia** con la franja de 32/64/128 filas ni con el umbral de curva de 10/20/30°. GPT2M > kd > el
  resto en las 6 variantes (§5).
- **Hallazgo: los modelos chicos derivan a la izquierda.** En las rectas, el flow horizontal de los modelos chicos
  está entre −0,10 y −0,16 px respecto del real.
  - En los sets balanceados (50/50) esto se cancela, y el giro reportado no cambia (±0,01 contra el promedio por
    lado).
  - Con un set desbalanceado se confundiría con giro: el primer test de parpadeo, con 8 curvas a un mismo lado, dio
    0,72. Queda como limitación en el doc (§6).
- **Set de 10-20°** (`results/clips_motion_suave.json`, 50 curvas): rollouts de los 8 modelos de la Fase 1.
  - **Tiempo de GPT2M:** con bs 25 tardó 88 min en 50 clips. Con bs 20 el set principal tarda 57 min en 200 clips, así
    que bs 25 probablemente desborda la VRAM. Usar bs ≤ 20.
