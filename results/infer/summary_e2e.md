# Inferencia de punta a punta

Video en la CPU → encoder VQ → world model → decoder VQ → video en la CPU, todo residente en el dispositivo (fp16 + CUDA graphs), batch 1. Ver `Docs/plan-inferencia.md`. Lo genera `python -m src.infer.report_e2e`.

- **En línea:** llega un frame de cámara, se codifica, entra al estado del modelo y se predice y decodifica el siguiente.
- **Imaginación:** el modelo genera a partir de su propia salida; el encoder solo corre al principio. Es el modo de los rollouts de la tabla principal (FVD).

### NVIDIA GeForce RTX 3080 Ti (`rtx3080ti`)

Codec solo, batch 1 (ms, mediana; codec = frame en el host → encoder → decoder → host):

| Variante | Encoder | Decoder | Codec |
|---|---:|---:|---:|
| fp32 | 7.84 | 14.21 | 21.93 |
| fp16 | 7.50 | 11.93 | 19.51 |
| fp16_graphs | 3.74 | 6.11 | 9.89 |

Fidelidad (fp16 + graphs contra fp32, 200 frames del video de cámara de comma): tokens iguales 99.71 % (fp32 contra los de comma: 99.87 %), PSNR del decoder 61.0 dB, LPIPS de la reconstrucción contra el frame de entrada 0.102 (fp32) y 0.102 (fp16), entre las dos 0.0008.

Cadena completa, batch 1, 3000 frames en bucle cerrado salvo indicación (ms por frame; tarde = más de 50 ms):

| Configuración | FVD@1s | En línea p50 | En línea p99 | Imaginación p50 | Imaginación p99 | A 50 ms p99 | Tarde (%) | Etapas en línea (enc / modelo / dec) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| solo codec (encoder + decoder) | – | 10.0 | 10.1 | – | – | 20.2 | 0.00 | 3.8 / 0.0 / 6.2 |
| Transición por celda | 729.4 | 10.1 | 10.3 | 6.6 | 7.3 | – | 0.00 | 3.8 / 0.1 / 6.2 |
| MaskGIT-S 1 paso | 234.9 | 14.8 | 14.9 | 11.1 | 11.2 | – | 0.00 | 3.8 / 4.7 / 6.2 |
| MaskGIT-S 2 pasos | 203.5 | 16.1 | 16.3 | 12.5 | 12.8 | – | 0.00 | 3.8 / 6.1 / 6.2 |
| MaskGIT-S 4 pasos | 220.2 | 18.9 | 19.0 | 15.3 | 15.5 | 21.1 | 0.00 | 3.8 / 8.9 / 6.1 |
| MaskGIT-S 8 pasos | 307.3 | 24.6 | 24.8 | 21.1 | 22.3 | – | 0.00 | 3.8 / 14.7 / 6.1 |
| MaskGIT-S 12 pasos | 373.6 | 30.4 | 30.5 | 26.9 | 28.6 | – | 0.00 | 3.8 / 20.4 / 6.1 |
| MaskGIT-M 1 paso | 291.3 | 17.9 | 18.0 | 14.2 | 14.4 | – | 0.00 | 3.9 / 7.7 / 6.3 |
| MaskGIT-M 2 pasos | 239.8 | 19.7 | 19.9 | 16.1 | 16.4 | – | 0.00 | 3.9 / 9.6 / 6.2 |
| MaskGIT-M 4 pasos | 213.8 | 23.5 | 23.7 | 19.8 | 20.4 | – | 0.00 | 3.8 / 13.5 / 6.2 |
| MaskGIT-M 8 pasos | 266.0 | 31.2 | 31.4 | 27.6 | 28.8 | 29.0 | 0.00 | 3.8 / 21.2 / 6.1 |
| MaskGIT-M 12 pasos | 326.2 | 39.0 | 39.3 | 39.3 | 44.9 | – | 0.13 | 3.8 / 29.0 / 6.1 |
| GPT-S | 229.6 | 167.3 | 170.9 | 164.3 | 170.5 | – | 100.00 | 3.8 / 157.3 / 6.1 |

Predicción a k frames en línea (k pasos del modelo y un decode; p50 / p99 en ms):

| Configuración | k | p50 | p99 |
|---|---:|---:|---:|
| MaskGIT-S 4 pasos | 5 | 54.8 | 58.3 |
| MaskGIT-S 4 pasos | 20 | 189.1 | 202.8 |

Primer segundo (contexto incompleto, sin graphs, y la captura de los graphs al llenarse): el peor frame es de 655 ms (GPT-S). No entra en las mediciones de arriba.

![Pareto de punta a punta](../fig_pareto_e2e.png)

### Calidad a 1 paso en línea

El modelo ve los frames reales de val (50 segmentos × 400 frames) y en cada uno predice el siguiente. Referencia: repetir el último frame (frame congelado), acierto 38.9 % y LPIPS 0.075.

| Configuración | Temperatura | Acierto de tokens (%) | LPIPS [IC 95%] |
|---|---:|---:|---:|
| Transición por celda | 1 | 16.9 | 0.143 [0.136, 0.151] |
| Transición por celda | 0 | 38.9 | 0.075 [0.070, 0.081] |
| MaskGIT-S 4 pasos | 1 | 30.9 | 0.103 [0.096, 0.110] |
| MaskGIT-S 4 pasos | 0 | 38.6 | 0.076 [0.071, 0.081] |
| MaskGIT-M 8 pasos | 1 | 33.5 | 0.097 [0.090, 0.103] |
| MaskGIT-M 8 pasos | 0 | 38.7 | 0.075 [0.070, 0.081] |

Camino nuevo contra la tabla (MaskGIT-S 4 pasos, 400 clips): tokens iguales 100.0 %, primer frame 100.0 %; FVD@1s 220.2 [190.1, 250.3] contra 220.2 [190.1, 250.3].
