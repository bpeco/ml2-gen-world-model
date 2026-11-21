Pareto: no dominada en (ms/frame, FVD a 1 s) entre las configuraciones principales (780.000 frames); las semillas 1 y 2, las ablaciones y los extras no compiten. Latencia de los transformers con fp16 + CUDA graphs. De punta a punta: cámara → encoder → modelo → decoder → frame en la GPU, a ritmo de cámara (un frame cada 50 ms), p99 de 3000 frames (results/infer/latencia_*.json). Estado: solo las filas vigentes entran en la frontera de Pareto y en las figuras (results/estado.json). IC del FVD y de LPIPS: bootstrap por segmento (los 4 clips de un segmento no son independientes). Piso (400 contra 400): FVD de los clips reales de evaluación contra otros 400 reales de los mismos segmentos (results/piso_fvd.json); es el comparable con la columna de FVD. Piso en mitades: real contra real con dos mitades disjuntas por segmento (n/2 por lado), y entre paréntesis el FVD del modelo con esas mismas mitades.

| Configuración | Familia | Params (M) | CE | CE último frame | LPIPS@1s | FVD@1s [IC 95% por segmento] | FVD@3s | Piso FVD@1s (400 contra 400) | Piso en mitades (modelo en mitades) | ms/frame | FPS | Tiempo real (<50 ms) | Pareto | De punta a punta p99 (ms) | Estado |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|:-:|---:|---|
| Frame congelado | baseline | – | 4.892 | 4.828 | 0.235 | 928.6 [860.7, 996.4] | 978.2 | 103.3 | 316.8 (1003.7) | – | – | – | – | – | vigente |
| convlstm_px | convlstm | 8.0 | – | – | 0.447 | 1789.9 [1629.7, 1950.2] | 2400.2 | 103.3 | 316.8 (1848.6) | 16.52 | 60.5 | sí | no | 32.2 | vigente |
| convlstm_px_s1 | convlstm | 8.0 | – | – | 0.451 | 1554.8 [1427.7, 1681.9] | 2376.2 | 103.3 | 316.8 (1622.9) | 16.52 | 60.5 | sí | – | – | vigente |
| convlstm_px_s2 | convlstm | 8.0 | – | – | 0.457 | 1695.5 [1542.8, 1848.1] | 2401.6 | 103.3 | 316.8 (1757.8) | 16.52 | 60.5 | sí | – | – | vigente |
| genie_m | genie | 34.8 | – | 3.042 | – | – | – | 103.3 | – | – | – | – | – | – | vigente |
| genie_m_p1 | genie | 34.8 | – | 3.042 | 0.332 | 245.6 [200.9, 290.4] | 729.8 | 103.3 | 316.8 (345.8) | 5.72 | 174.9 | sí | no | – | vigente |
| genie_m_p12 | genie | 34.8 | – | 3.042 | 0.259 | 312.3 [271.7, 353.0] | 409.0 | 103.3 | 316.8 (400.1) | 18.85 | 53.0 | sí | no | – | vigente |
| genie_m_p2 | genie | 34.8 | – | 3.042 | 0.304 | 218.6 [181.9, 255.3] | 504.2 | 103.3 | 316.8 (312.1) | 6.93 | 144.3 | sí | sí | – | vigente |
| genie_m_p25 | genie | 34.8 | – | 3.042 | 0.255 | 331.5 [289.3, 373.6] | 452.9 | 103.3 | 316.8 (418.9) | 34.51 | 29.0 | sí | no | – | vigente |
| genie_m_p4 | genie | 34.8 | – | 3.042 | 0.277 | 244.4 [207.6, 281.2] | 405.8 | 103.3 | 316.8 (332.3) | 9.27 | 107.8 | sí | no | 20.6 | vigente |
| genie_m_p8 | genie | 34.8 | – | 3.042 | 0.266 | 277.0 [236.7, 317.3] | 402.6 | 103.3 | 316.8 (367.7) | 14.06 | 71.1 | sí | no | – | vigente |
| genie_s | genie | 15.0 | – | 3.032 | – | – | – | 103.3 | – | – | – | – | – | – | vigente |
| genie_s_p1 | genie | 15.0 | – | 3.032 | 0.384 | 278.8 [232.7, 324.9] | 1307.3 | 103.3 | 316.8 (385.9) | 3.27 | 305.9 | sí | sí | – | vigente |
| genie_s_p12 | genie | 15.0 | – | 3.032 | 0.259 | 302.0 [258.9, 345.1] | 425.3 | 103.3 | 316.8 (387.5) | 12.72 | 78.6 | sí | no | – | vigente |
| genie_s_p2 | genie | 15.0 | – | 3.032 | 0.32 | 227.8 [188.1, 267.4] | 683.1 | 103.3 | 316.8 (325.6) | 4.05 | 246.6 | sí | sí | – | vigente |
| genie_s_p25 | genie | 15.0 | – | 3.032 | 0.255 | 325.8 [281.5, 370.0] | 444.1 | 103.3 | 316.8 (414.1) | 23.88 | 41.9 | sí | no | – | vigente |
| genie_s_p4 | genie | 15.0 | – | 3.032 | 0.281 | 232.4 [193.9, 270.8] | 413.0 | 103.3 | 316.8 (321.1) | 5.7 | 175.5 | sí | no | 21.7 | vigente |
| genie_s_p4_r | genie | – | – | – | 0.283 | 242.1 [199.7, 284.5] | 434.0 | 103.3 | 316.8 (331.6) | – | – | – | – | – | vigente |
| genie_s_p8 | genie | 15.0 | – | 3.032 | 0.265 | 276.1 [233.9, 318.3] | 425.0 | 103.3 | 316.8 (363.0) | 9.3 | 107.5 | sí | no | – | vigente |
| genie_s_s1 | genie | 15.0 | – | 3.062 | – | – | – | 103.3 | – | – | – | – | – | – | vigente |
| genie_s_s1_p8 | genie | 15.0 | – | 3.062 | 0.268 | 255.7 [214.6, 296.8] | 389.4 | 103.3 | 316.8 (343.5) | – | – | – | – | – | vigente |
| genie_s_s2 | genie | 15.0 | – | 3.04 | – | – | – | 103.3 | – | – | – | – | – | – | vigente |
| genie_s_s2_p8 | genie | 15.0 | – | 3.04 | 0.269 | 248.3 [210.6, 286.0] | 393.1 | 103.3 | 316.8 (337.3) | – | – | – | – | – | vigente |
| gpt2m | gpt2m | 307.1 | 2.026 | 1.751 | 0.171 | 67.1 [50.3, 83.8] | 105.2 | 103.3 | 316.8 (130.0) | 290.75 | 3.4 | no | sí | – | vigente |
| gpt_s_d025 | gpt | 12.4 | 3.959 | 4.736 | 0.383 | 414.4 [372.7, 456.1] | 615.3 | 103.3 | 316.8 (506.4) | – | – | – | – | – | vigente |
| gpt_s_d050 | gpt | 12.4 | 3.299 | 3.229 | 0.345 | 234.8 [198.5, 271.0] | 478.6 | 103.3 | 316.8 (329.7) | – | – | – | – | – | vigente |
| gpt_s_kd | gpt_kd | 12.4 | 2.986 | 2.773 | 0.295 | 206.7 [165.0, 248.5] | 468.5 | 103.3 | 316.8 (299.4) | 37.9 | 26.4 | sí | sí | 48.4 | vigente |
| gpt_s_kd_r | gpt | – | – | – | 0.293 | 207.1 [167.3, 246.9] | 461.8 | 103.3 | 316.8 (300.7) | – | – | – | – | – | vigente |
| lewm | lewm | 18.0 | – | – | – | – | – | 103.3 | – | 8.98 | 111.4 | sí | – | – | vigente |
| lewm_lrx3 | lewm | 18.0 | – | – | – | – | – | 103.3 | – | – | – | – | – | – | vigente |
| maskgit_s_abs | maskgit | 12.4 | – | 3.179 | – | – | – | 103.3 | – | – | – | – | – | – | vigente |
| maskgit_s_abs_p1 | maskgit | 12.4 | – | 3.179 | 0.338 | 227.8 [190.5, 265.0] | 607.4 | 103.3 | 316.8 (325.5) | 3.93 | 254.6 | sí | – | – | vigente |
| maskgit_s_abs_p12 | maskgit | 12.4 | – | 3.179 | 0.268 | 283.8 [247.7, 319.9] | 370.5 | 103.3 | 316.8 (371.5) | 12.98 | 77.0 | sí | – | – | vigente |
| maskgit_s_abs_p2 | maskgit | 12.4 | – | 3.179 | 0.327 | 214.2 [179.8, 248.5] | 544.9 | 103.3 | 316.8 (308.5) | 4.71 | 212.3 | sí | – | – | vigente |
| maskgit_s_abs_p4 | maskgit | 12.4 | – | 3.179 | 0.298 | 220.1 [180.9, 259.3] | 467.6 | 103.3 | 316.8 (312.7) | 6.39 | 156.6 | sí | – | – | vigente |
| maskgit_s_abs_p8 | maskgit | 12.4 | – | 3.179 | 0.278 | 247.3 [211.7, 283.0] | 380.4 | 103.3 | 316.8 (335.4) | 10.35 | 96.6 | sí | – | – | vigente |
| Real (piso) | baseline | – | – | – | 0.0 | 0.0 [0.0, 0.0] | 0.0 | 103.3 | 316.8 (0.0) | – | – | – | – | – | vigente |
| real_eval_b | otro | – | – | – | 0.0 | 0.0 [0.0, 0.0] | 0.0 | 103.3 | 319.2 (0.0) | – | – | – | – | – | vigente |
| Transición por celda | baseline | – | 3.48 | 3.331 | 0.522 | 729.4 [656.3, 802.5] | 1241.4 | 103.3 | 316.8 (821.5) | 0.34 | 2925.8 | sí | sí | 20.6 | vigente |
| Unigrama | baseline | – | 6.846 | 6.854 | 0.571 | 1470.9 [1340.6, 1601.2] | 1523.9 | 103.3 | 316.8 (1532.2) | – | – | – | – | – | vigente |
| vjepa2_ac | vjepa2 | 21.6 | – | – | – | – | – | 103.3 | – | 65.22 | 15.3 | no | – | – | vigente |
| vjepa2_s | vjepa2 | 43.5 | – | – | – | – | – | 103.3 | – | 46.1 | 21.7 | sí | – | – | vigente |

CE vs. FVD@1s (autoregresivas, vigentes): Spearman ρ = 0.90, p = 0,037, n = 5.
