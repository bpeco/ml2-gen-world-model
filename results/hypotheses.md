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

int8 dinámico en CPU (ms/frame):

- gpt_s: fp32 844 · int8 1048
- maskgit_s: fp32 375 · int8 357

## H4: CE vs. FVD

Spearman entre CE y FVD a 1 s: ρ = 1.00 (p < 0,001, n = 7). Entre CE y LPIPS a 1 s: ρ = 0.86 (p = 0.014).
