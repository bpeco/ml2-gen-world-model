# Hipótesis

## H1: destilado vs. de cero (mismo tamaño, datos, semilla y 100M tokens)

| Tamaño | Métrica | De cero | Destilado − de cero [IC 95 %] | p (Wilcoxon) |
|---|---|---:|---:|---:|
| S | CE val | 2.957 | -0.0073 [-0.0080, -0.0065] | 5.3e-18 |
| S | CE último frame | 2.763 | -0.0134 [-0.0147, -0.0122] | 5e-18 |
| S | LPIPS 1 s | 0.354 | +0.0022 [-0.0055, +0.0100] | 0.57 |
| S | FVD 1 s | 230 [199, 260] | destilado 229 [196, 263] | |
| M | CE val | 2.963 | -0.0319 [-0.0336, -0.0302] | 3.9e-18 |
| M | CE último frame | 2.770 | -0.0437 [-0.0462, -0.0414] | 3.9e-18 |
| M | LPIPS 1 s | 0.343 | -0.0117 [-0.0184, -0.0052] | 3.7e-06 |
| M | FVD 1 s | 237 [207, 268] | destilado 213 [185, 242] | |

## H2: MaskGIT con pocos pasos

Tamaño S: GPT 156.1 ms/frame, LPIPS 1 s 0.354, FVD 1 s 230.

| Pasos | ms/frame | < 50 ms | LPIPS 1 s | MaskGIT − GPT [IC 95 %] | FVD 1 s | FVD 3 s |
|---:|---:|:-:|---:|---:|---:|---:|
| 1 | 5.3 | sí | 0.352 | -0.0014 [-0.0090, +0.0063] | 235 | 489 |
| 2 | 7.0 | sí | 0.320 | -0.0337 [-0.0414, -0.0262] | 204 | 446 |
| 4 | 10.1 | sí | 0.286 | -0.0680 [-0.0751, -0.0610] | 220 | 376 |
| 8 | 16.1 | sí | 0.264 | -0.0891 [-0.0964, -0.0821] | 307 | 366 |

Tamaño M: GPT 211.0 ms/frame, LPIPS 1 s 0.343, FVD 1 s 237.

| Pasos | ms/frame | < 50 ms | LPIPS 1 s | MaskGIT − GPT [IC 95 %] | FVD 1 s | FVD 3 s |
|---:|---:|:-:|---:|---:|---:|---:|
| 1 | 8.4 | sí | 0.378 | +0.0350 [+0.0280, +0.0419] | 291 | 476 |
| 2 | 10.4 | sí | 0.340 | -0.0027 [-0.0097, +0.0041] | 240 | 474 |
| 4 | 14.4 | sí | 0.298 | -0.0445 [-0.0509, -0.0382] | 214 | 380 |
| 8 | 22.6 | sí | 0.269 | -0.0736 [-0.0798, -0.0677] | 266 | 341 |

## H3: precisión y CUDA graphs

| Modelo | Métrica | fp32 | fp16 | Diferencia |
|---|---|---:|---:|---:|
| gpt2m | ce_all | 2.02565 | 2.02567 | +1.37e-05 |
| gpt_s | ce_all | 2.95675 | 2.95676 | +3.13e-06 |
| maskgit_s | ce_last | 3.02237 | 3.02238 | +1.03e-05 |
| gpt_s | LPIPS 1 s (fp16 − fp32) | | | -0.0007 [-0.0043, +0.0028] |
| gpt_s | FVD 1 s | 233 | 230 | |
| maskgit_s_s8 | LPIPS 1 s (fp16 − fp32) | | | +0.0001 [-0.0008, +0.0011] |
| maskgit_s_s8 | FVD 1 s | 309 | 307 | |

| Modelo | fp32 | fp16 | fp16 + graphs | Aceleración total |
|---|---:|---:|---:|---:|
| gpt2m | 2057.2 | 1751.3 | 735.2 | 2.8× |
| gpt_s | 474.5 | 467.0 | 156.1 | 3.0× |
| gpt_m | 652.9 | 603.5 | 211.0 | 3.1× |
| random_gpt_l | 972.5 | 890.1 | 337.2 | 2.9× |
| maskgit_s | 44.9 | 39.0 | 16.1 | 2.8× |
| maskgit_m | 68.5 | 48.1 | 22.6 | 3.0× |
| convlstm | 1.1 | 1.1 | – | 1.0× |

int8 dinámico en CPU (ms/frame):

- gpt_s: fp32 844 · int8 1048
- convlstm: fp32 9 · int8 8
- maskgit_s: fp32 375 · int8 357

## H4: CE vs. FVD

Spearman entre CE y FVD a 1 s: ρ = 1.00 (p < 0,001, n = 9). Entre CE y LPIPS a 1 s: ρ = 0.93 (p < 0,001).

## H5: latente vs. tokens (misma arquitectura, solo cambia la pérdida)

| Modelo | CE val | CE último frame | LPIPS 1 s | FVD 1 s | FVD 3 s | ms/frame | Params (M) |
|---|---:|---:|---:|---:|---:|---:|---:|
| JEPA-S (V-JEPA: celda + EMA) | 3.708 | 3.525 | 0.441 | 460 | 1188 | 2.9 | 11.6 |
| JEPA-S, control CE | 2.959 | 2.739 | 0.358 | 265 | 826 | 2.9 | 11.6 |
| MaskGIT-S 1 paso | – | 3.022 | 0.352 | 235 | 489 | 5.3 | 11.5 |
| LeWM (vector + SIGReg) | 6.637 | 6.683 | 0.501 | 1016 | 1080 | 1.4 | 21.7 |
| LeWM, control CE | 6.574 | 6.615 | 0.597 | 1379 | 1945 | 1.4 | 21.7 |

Diferencias pareadas (latente − control CE; > 0 = el latente es peor):

- JEPA-S (V-JEPA: celda + EMA): CE val +0.749 [+0.708, +0.795], p = 3.9e-18; CE último frame +0.786 [+0.742, +0.834], p = 3.9e-18; LPIPS 1 s +0.083 [+0.074, +0.093], p = 2.4e-40.
  - contra MaskGIT-S 1 paso: LPIPS 1 s +0.089 [+0.080, +0.097], p = 1.2e-53.
  - rollout latente (ẑ al contexto, sin re-codificar): LPIPS 1 s 0.438, FVD 1 s 553, FVD 3 s 2990, 3.1 ms/frame; latente − token: -0.003 [-0.011, +0.006], p = 0.49.
- LeWM (vector + SIGReg): CE val +0.064 [+0.052, +0.076], p = 8.5e-17; CE último frame +0.068 [+0.056, +0.081], p = 7.1e-17; LPIPS 1 s -0.096 [-0.106, -0.086], p = 8.3e-44.
  - contra MaskGIT-S 1 paso: LPIPS 1 s +0.149 [+0.139, +0.159], p = 2.9e-64.
  - rollout latente (ẑ al contexto, sin re-codificar): LPIPS 1 s 0.481, FVD 1 s 991, FVD 3 s 1029, 1.0 ms/frame; latente − token: -0.021 [-0.028, -0.013], p = 7.3e-08.

## H6: acciones (holdout de train separado por ruta, poses reales)

| Modelo | Error latente con pose real | con acción nula | nula − real [IC 95 %] | CEM: velocidad (m/s) | CEM: giro (rad/s) | ms por plan |
|---|---:|---:|---:|---:|---:|---:|
| JEPA-S (V-JEPA: celda + EMA) | 0.2189 | 0.2210 | +0.0020 [+0.0014, +0.0027] | 9.49 | 0.1979 | 22666 |
| JEPA-S con SIGReg (celda + SIGReg) | 0.2276 | 0.2287 | +0.0011 [+0.0008, +0.0015] | 13.67 | 0.1464 | 22660 |
| JEPA-S, target codebook | 0.2274 | 0.2305 | +0.0030 [+0.0023, +0.0038] | – | – | – |
| JEPA-S, control CE | 2.6201 | 2.6710 | +0.0508 [+0.0431, +0.0592] | – | – | – |
| LeWM (vector + SIGReg) | 0.0957 | 0.0958 | +0.0001 [+0.0000, +0.0002] | 19.44 | 0.0770 | 535 |
| LeWM con EMA (vector + EMA) | 1.2594 | 1.2981 | +0.0388 [+0.0372, +0.0403] | 22.56 | 0.0526 | 541 |
| LeWM, control CE | 8.3662 | 10.1410 | +1.7749 [+1.6308, +1.9218] | 11.82 | 0.0505 | 536 |
| LeWM, λ = 0,03 | 0.0558 | 0.0558 | +0.0000 [-0.0000, +0.0001] | 19.99 | 0.0682 | 538 |
| LeWM, λ = 0,3 | 0.4288 | 0.4355 | +0.0067 [+0.0061, +0.0073] | 14.21 | 0.0756 | 550 |
| Referencia: repetir la última pose | | | | 0.13 | 0.0040 | |
| Referencia: pose media de train | | | | 8.55 | 0.0320 | |

## H7: granularidad del latente × antiaplastamiento

| Latente | Antiaplastamiento | Corrida | CE último frame | LPIPS 1 s | FVD 1 s | z_std final | Probe ẑ: r velocidad | r giro | Sorpresa teletransporte | Sorpresa salto 2 s | Rectitud |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| celda | EMA | jepa_s | 3.525 | 0.441 | 460 | 0.78 | 0.52 | 0.11 | 8.16× | 5.04× | -0.21 |
| celda | SIGReg | jepa_s_sigreg | 4.033 | 0.541 | 806 | 0.26 | 0.53 | 0.16 | 6.86× | 4.35× | -0.24 |
| vector | EMA | lewm_s_ema | 6.673 | 0.519 | 1021 | 1.89 | 0.47 | 0.07 | 4.51× | 1.52× | -0.30 |
| vector | SIGReg | lewm_s | 6.683 | 0.501 | 1016 | 0.92 | 0.40 | 0.14 | 19.86× | 5.52× | -0.30 |
| tokens | — | gpt_s | | | | | 0.27 | 0.37 | | | |
| tokens | — | maskgit_s | | | | | 0.67 | 0.30 | | | |

Barrido de λ (LeWM, 100M tokens):

| λ | CE último frame | LPIPS 1 s | FVD 1 s | z_std final | Probe ẑ: r velocidad | Sorpresa teletransporte |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 6.679 | – | – | 0.88 | 0.49 | 31.91× |
| 0.09 | 6.683 | 0.501 | 1016 | 0.92 | 0.40 | 19.86× |
| 0.3 | 6.667 | – | – | 0.97 | 0.17 | 3.81× |
