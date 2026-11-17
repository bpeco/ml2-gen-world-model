Pareto: no dominada en (ms/frame, FVD a 1 s) entre las configuraciones principales (100M tokens); las ablaciones y los extras no compiten. Latencia de los transformers con fp16 + CUDA graphs.

| Configuración | Familia | Params (M) | CE | CE último frame | LPIPS@1s | FVD@1s [IC 95%] | FVD@3s | ms/frame | FPS | Tiempo real (<50 ms) | Pareto |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:-:|:-:|
| Unigrama | baseline | – | 6.846 | 6.854 | 0.571 | 1470.9 [1375.9, 1565.9] | 1523.9 | – | – | – | – |
| Copia (tabla por celda) | baseline | – | 3.48 | 3.331 | 0.522 | 729.4 [674.4, 784.4] | 1241.4 | 0.47 | 2132.2 | sí | sí |
| Frame congelado | baseline | – | – | – | 0.235 | 928.6 [870.7, 986.4] | 978.2 | – | – | – | – |
| gpt2m (comma) | gpt2m | 307.0 | 2.026 | 1.751 | 0.171 | 67.1 [55.6, 78.5] | 105.2 | 735.17 | 1.4 | no | sí |
| ConvLSTM | convlstm | 8.0 | 2.879 | 2.618 | 0.287 | 162.8 [140.1, 185.5] | 293.3 | 1.1 | 907.6 | sí | sí |
| GPT-S | gpt | 11.5 | 2.957 | 2.763 | 0.354 | 229.6 [198.9, 260.3] | 734.6 | 156.08 | 6.4 | no | no |
| GPT-S destilado | gpt_kd | 11.5 | 2.949 | 2.749 | 0.356 | 229.2 [195.6, 262.7] | 616.9 | 156.08 | 6.4 | no | no |
| GPT-M | gpt | 26.3 | 2.963 | 2.77 | 0.343 | 237.5 [206.7, 268.2] | 629.6 | 210.96 | 4.7 | no | no |
| GPT-M destilado | gpt_kd | 26.3 | 2.931 | 2.726 | 0.331 | 213.1 [184.5, 241.7] | 516.7 | 210.96 | 4.7 | no | no |
| MaskGIT-S 1 pasos | maskgit | 11.5 | – | 3.022 | 0.352 | 234.9 [206.5, 263.3] | 489.4 | 5.31 | 188.4 | sí | no |
| MaskGIT-S 2 pasos | maskgit | 11.5 | – | 3.022 | 0.32 | 203.5 [175.6, 231.5] | 445.7 | 7.02 | 142.4 | sí | no |
| MaskGIT-S 4 pasos | maskgit | 11.5 | – | 3.022 | 0.286 | 220.2 [190.1, 250.3] | 375.9 | 10.14 | 98.6 | sí | no |
| MaskGIT-S 8 pasos | maskgit | 11.5 | – | 3.022 | 0.264 | 307.3 [274.0, 340.7] | 366.0 | 16.07 | 62.2 | sí | no |
| MaskGIT-M 1 pasos | maskgit | 26.3 | – | 3.134 | 0.378 | 291.3 [260.6, 321.9] | 475.9 | 8.42 | 118.8 | sí | no |
| MaskGIT-M 2 pasos | maskgit | 26.3 | – | 3.134 | 0.34 | 239.8 [209.6, 270.1] | 473.9 | 10.42 | 96.0 | sí | no |
| MaskGIT-M 4 pasos | maskgit | 26.3 | – | 3.134 | 0.298 | 213.8 [183.5, 244.1] | 379.5 | 14.43 | 69.3 | sí | no |
| MaskGIT-M 8 pasos | maskgit | 26.3 | – | 3.134 | 0.269 | 266.0 [237.6, 294.4] | 341.4 | 22.58 | 44.3 | sí | no |
| GPT-S, posición absoluta | ablación | 12.4 | 3.158 | 3.416 | 0.36 | 284.8 [252.7, 316.9] | 726.8 | 156.08 | 6.4 | no | – |
| MaskGIT-S 8 pasos, posición absoluta | ablación | 12.4 | – | 3.183 | 0.274 | 252.7 [224.0, 281.4] | 432.6 | 16.07 | 62.2 | sí | – |
| MaskGIT-S 8 pasos, máscara Genie | ablación | 11.5 | – | 3.26 | 0.325 | 337.5 [296.1, 379.0] | 725.8 | 16.07 | 62.2 | sí | – |
| MaskGIT-S 8 pasos, sin p_clean | ablación | 11.5 | – | 5.07 | 0.849 | 3294.7 [3133.5, 3455.8] | 4006.9 | 16.07 | 62.2 | sí | – |
| ConvLSTM, 1B tokens | convlstm | 8.0 | 2.685 | 2.398 | 0.259 | 117.3 [100.6, 134.0] | 249.6 | 1.1 | 907.6 | sí | – |
| GPT-S, contexto 4 | contexto | 11.5 | – | – | 0.345 | 236.0 [202.9, 269.1] | 691.1 | 155.18 | 6.4 | no | – |
| MaskGIT-S 2 pasos, contexto 4 | contexto | 11.5 | – | – | 0.322 | 207.0 [177.8, 236.1] | 446.9 | 6.75 | 148.1 | sí | – |
| MaskGIT-S 4 pasos, contexto 4 | contexto | 11.5 | – | – | 0.284 | 229.7 [197.4, 261.9] | 352.4 | 9.99 | 100.1 | sí | – |
| GPT-S, contexto 8 | contexto | 11.5 | – | – | 0.345 | 220.6 [192.5, 248.6] | 661.5 | 155.98 | 6.4 | no | – |
| MaskGIT-S 2 pasos, contexto 8 | contexto | 11.5 | – | – | 0.318 | 204.9 [175.2, 234.6] | 420.6 | 6.87 | 145.6 | sí | – |
| MaskGIT-S 4 pasos, contexto 8 | contexto | 11.5 | – | – | 0.284 | 237.1 [207.2, 266.9] | 373.2 | 10.06 | 99.4 | sí | – |
| JEPA-S (V-JEPA, latente por celda) | jepa | 11.6 | 3.708 | 3.525 | 0.441 | 460.0 [411.8, 508.3] | 1187.7 | 2.92 | 342.9 | sí | no |
| LeWM (LeWorldModel, un vector por frame) | lewm | 21.7 | 6.637 | 6.683 | 0.501 | 1016.3 [961.5, 1071.2] | 1080.0 | 1.36 | 732.6 | sí | no |
| JEPA-XS (4 capas, D 256) | jepa | 3.7 | 3.858 | 3.676 | 0.463 | 510.1 [457.0, 563.2] | 1120.5 | 1.89 | 529.9 | sí | no |
| JEPA-XXS (3 capas, D 192) | jepa | 1.7 | 3.924 | 3.724 | 0.463 | 545.7 [496.2, 595.2] | 1264.2 | 1.2 | 834.7 | sí | no |
| LeWM-XS (ViT de 6 capas, predictor de 4) | lewm | 10.9 | 6.637 | 6.683 | 0.508 | 996.9 [937.7, 1056.1] | 1024.8 | 0.81 | 1231.5 | sí | no |
| JEPA-S, control CE | jepa | 11.6 | 2.959 | 2.739 | 0.358 | 265.3 [234.5, 296.2] | 826.3 | 2.92 | 342.9 | sí | – |
| JEPA-S con SIGReg | jepa | 11.6 | 4.256 | 4.033 | 0.541 | 806.2 [754.9, 857.4] | 849.5 | 2.92 | 342.9 | sí | – |
| JEPA-S, target codebook | jepa | 11.4 | 4.336 | 3.863 | 0.488 | 582.3 [534.9, 629.6] | 970.8 | 2.92 | 342.9 | sí | – |
| LeWM, control CE | lewm | 21.7 | 6.574 | 6.615 | 0.597 | 1379.2 [1294.7, 1463.8] | 1944.8 | 1.36 | 732.6 | sí | – |
| LeWM con EMA | lewm | 21.7 | 6.628 | 6.673 | 0.519 | 1020.6 [961.4, 1079.7] | 1043.2 | 1.36 | 732.6 | sí | – |
| LeWM, λ = 0,03 | lewm | 21.7 | 6.634 | 6.679 | – | – | – | 1.36 | 732.6 | sí | – |
| LeWM, λ = 0,3 | lewm | 21.7 | 6.624 | 6.667 | – | – | – | 1.36 | 732.6 | sí | – |
| JEPA-S, rollout latente | jepa | 11.6 | – | – | 0.438 | 553.4 [503.8, 602.9] | 2989.6 | 3.06 | 326.7 | sí | – |
| LeWM, rollout latente | lewm | 21.7 | – | – | 0.481 | 990.9 [934.4, 1047.4] | 1029.2 | 1.03 | 966.9 | sí | – |

CE vs. FVD@1s (autoregresivas): Spearman ρ = 1.00, p < 0,001, n = 9.
