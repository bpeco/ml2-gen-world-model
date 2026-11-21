# Optimizar el decode de GPT y gpt2m

Surgió de revisar las métricas de la presentación (`presentacion/`) el 20 de noviembre de 2026. **Estado (21/11/2026): implementado y medido en la 3080 Ti, en la rama `optimizar-decode-gpt` (commit `8b3ee3c`).** La hipótesis original (overhead del host) resultó falsa. El tiempo lo dominaba un kernel de atención con la GPU casi vacía, y cambiarlo bajó el GPT-S de 154 a 40 ms y gpt2m de 767 a 313 ms (ver "Resultado"). <!-- ok-numero --> Las secciones de más abajo son la propuesta original. Cuando se decida qué hacer con esto, este documento se borra o se archiva: lo vigente son los JSON de `results/`.

## Resultado: la atención de decode (21/11/2026)

**Diagnóstico.** `python -m src.cli gpt bench --tamaño s --variante kd -- --perfil` (nuevo: desglose por kernel con `torch.profiler`, en `results/perfil/`) mostró que la atención del decode era el 81 % del tiempo de GPU del GPT-S: 128,7 de 159 ms por frame, en 768 llamadas (6 capas × 128 tokens) a `fmha_cutlassF`, de 167,5 µs cada una (`results/perfil/gpt_s_kd_perfil.json`). <!-- ok-numero -->

- Con caché, cada paso atiende 1 query contra las 2584 posiciones de la caché estática, con máscara.
- Este torch (2.11+cu128, Windows) no trae FlashAttention. Con máscara, SDPA elige el kernel mem-efficient, que lanza un bloque por cabeza (6 a 16 de los 80 SM de la 3080 Ti), y cada bloque recorre la caché en serie.
- Además, SDPA convierte la máscara bool a float en cada capa y en cada token.
- Eso explica la evidencia de abajo:
  - el costo es ~200 µs por capa y por token en S, M, L y gpt2m, porque todos tienen head_dim 64 y la misma caché;
  - no depende del contexto, porque se recorre la caché entera igual;
  - la 4060 gana, porque los kernels con poca ocupación dependen del reloj por SM y de la L2.

**Cambio** (`src/models/backbone.py`, `src/models/gpt2m.py`):
- `cached_attention` hace la misma cuenta que SDPA (la del backend math, el que fuerza `decode_n_tokens` de comma) con `baddbmm`, `softmax` y `bmm`, que se reparten sobre las posiciones de la caché.
- La máscara aditiva se calcula una vez, en `setup_cache`.
- Se usa con caché y T ≤ 129: el decode del GPT y las pasadas de MaskGIT. El prefill sigue por SDPA.
- En gpt2m, el `forward` de la atención de comma se reemplaza solo mientras la caché está enganchada (`next_frame`), sin tocar `third_party`.

| Medición en la 3080 Ti (p50, p99 entre paréntesis) | Antes | Después |
|---|---|---|
| GPT-S destilado, modelo solo | 154,0 (158,6) y 155,0 (160,0) | **40,1 (48,3) y 40,0 (47,1)** |
| gpt2m, modelo solo | 767,1 (779,2) | **312,7 (317,7) y 312,3 (326,4)** |
| MaskGIT-S, 4 pasos, modelo solo (pesos al azar) | 9,32 (10,60) | **6,40 (7,46)** |
| GPT-S, cadena cerrada | 172,6 (187,0) | **49,1 (57,0)**: 9,6 % de frames tarde |
| GPT-S, cadena a ritmo de cámara | 165,1 (165,4) | **49,2 (58,1)**: 14,1 % de frames tarde |
<!-- ok-numero -->

- **Tiempo de GPU por frame** (`results/perfil/gpt_s_kd_perfil_attn.json`): de 159 a 38,7 ms en el GPT-S. La atención de decode baja de 167,5 µs a ~13 µs por capa (un gemv, un softmax y un gemm). Lo que queda son matmul (53 %), layer norm, indexado y elementwise: ~13.000 kernels chicos por frame. En gpt2m queda en 307 ms de GPU por frame, con la atención en 5 % (`results/perfil/gpt2m_perfil_attn.json`). <!-- ok-numero -->
- **Equivalencia** (`gpt evaluar --tamaño s --variante kd --cadena --limite 5`, T = 0,9): el FVD es el mismo, 205,3 / 350,2 / 468,5 contra 206,7 / 357,8 / 468,5 a 1, 2 y 3 s, dentro de los IC (`results/video_gpt_s_kd_stream.json` contra `results/video_gpt_s_kd.json`). El LPIPS es igual en toda la curva. Los tokens coinciden en un 92,7 % (`results/infer/imagine_check_gpt_s_kd.json`): el redondeo fp16 del kernel nuevo cambia algún token, y a T = 0,9 eso se propaga por el rollout. <!-- ok-numero -->
- **La cadena del GPT-S queda en el borde.** Encoder 3,8 + modelo 39,2 + decoder 6,1 ms dan 49,1 ms de p50, con el p99 en 57 a 58 ms: entra en los 50 ms en la mediana, pero no siempre. <!-- ok-numero -->
- **El cociente con MaskGIT-S baja.** En la misma GPU, el GPT-S era 16,5 veces más lento que MaskGIT-S con 4 pasos (154 contra 9,32 ms) y ahora lo es 6,3 veces (40 contra 6,4 ms). <!-- ok-numero -->
- **Dónde están los números.**
  - Después: `results/bench/gpt_s_kd_attn__fp16_graphs{,_r2}`, `gpt2m_attn__fp16_graphs{,_r2}`, `maskgit_s_abs_p4_attn__fp16_graphs_s4` y `results/infer/latencia_gpt_s_kd_attn__rtx3080ti.json`.
  - Antes, con el código de `main` (`79b6cae`): `gpt_s_kd_base_hoy`, `gpt2m_base_hoy`, `maskgit_s_abs_p4_base_hoy` y `latencia_gpt_s_kd_base_hoy`.
  - Los JSON viejos de la tabla no se tocaron.
  - Lo de después está medido con el árbol limpio, pero en una rama: cuenta como `vigente` cuando el commit llegue a `main`.
- **Para la presentación** (slides 1, 3, 4, 7, 8 y 11, y la tabla de respaldo): cambian las cifras del GPT-S, gpt2m y MaskGIT-S, el título de la slide 8 ("El GPT chico no llega a 20 FPS") y la conclusión 1. La decodificación paralela sigue siendo más rápida, pero la brecha con el GPT era en gran parte de implementación. Las cifras de la 4060 quedaron viejas.

## Primer intento: el frame entero en un CUDA graph (21/11/2026)

Se implementó `GPTStream` como estaba propuesto (`sample_logits(noise=)`, `next_frame(noise=, graphs=)`, `Gpt2m._ensure_cache`, `GPTStream` y el despacho en `make_stream`) y se midió con la CLI, en la 3080 Ti, fp16 con graphs, batch 1 y contexto lleno. **El frame entero en un graph no reduce la latencia: los ~130 ms fijos no eran overhead de Python del host.**

| Medición (p50, p99 entre paréntesis) | Código viejo, hoy | Frame en un graph | Medición vieja guardada |
|---|---|---|---|
| GPT-S destilado, modelo solo | 154,0 (158,6) y 155,0 (160,0) | 153,0 (157,0) y 153,4 (158,6) | 158,2 |
| gpt2m, modelo solo | 767,1 (779,2) | 770,5 (774,7) | 733,0 |
| GPT-S, cadena cerrada | 172,6 (187,1) | 163,6 (174,2) | 168,1 |
| GPT-S, cadena a ritmo de cámara (50 ms) | 165,1 (165,4) | 171,2 (193,0) | no guardada |

- Las diferencias son de ±1 % en el modelo solo y de ±5 % en la cadena, que cambian de signo de una modalidad a otra: ruido. Con el código viejo, la medición de hoy (154–155 ms) reproduce la guardada (158 ms), así que la máquina está en el mismo estado que cuando se midió.
- **Equivalencia con pesos reales** (`gpt evaluar --tamaño s --variante kd --cadena`, T = 0,9, fp16): `tokens_equal` = 1,0 y `clips_identical` = 1,0 sobre 400 clips de 60 frames, contra `results/rollouts/gpt_s_kd.npy`. El FVD del camino nuevo es idéntico (206,7 / 357,8 / 468,5 a 1, 2 y 3 s). La captura del frame entero de gpt2m (~48.000 nodos) funciona sin partirla.
- **Dónde están los números.** `results/bench/gpt_s_kd_framegraph__fp16_graphs{,_r2}`, `gpt_s_kd_base_hoy__fp16_graphs{,_r2}`, `gpt2m_framegraph__fp16_graphs` y `gpt2m_base_hoy__fp16_graphs`; `results/infer/latencia_gpt_s_kd_framegraph__rtx3080ti.json` y `latencia_gpt_s_kd_base_hoy__rtx3080ti.json`. Los JSON viejos (`gpt_s_kd__fp16_graphs`, `gpt2m__fp16_graphs`, los de la 4060 y `latencia_gpt_s_kd__rtx3080ti`) no se tocaron. Las corridas hechas con otros procesos en la CPU (la primera tanda dio 181, 870 y 415 ms) se repitieron en silencio; sus líneas quedan en `results/bitacora.jsonl`.
- Hecho con el árbol sucio (el código no estaba commiteado): la bitácora lo anota y estos resultados no cuentan como `vigente`.

**Qué significa.** El tiempo de decode es de GPU, no del host: el paso de decode ya corría como graph, y juntar los 128 en uno no cambia nada. El perfil de la sección anterior mostró qué lo dominaba. El frame en un graph se queda igual, porque saca a la CPU del camino: con la CPU cargada, el decode viejo midió 415 ms en vez de 154. <!-- ok-numero -->

## Resumen

- El decode de `gpt` y `gpt2m` no está optimizado como el de MaskGIT y Genie. `MaskGITStream` y `GenieStream` (`src/infer/stream.py`) capturan el frame entero (prefill, pasos y muestreo) en un solo CUDA graph. `gpt` y `gpt2m` usan `TokenStream` y `next_frame`, que capturan un solo paso de decode: el bucle de 128 tokens, el muestreo y las copias a los buffers estáticos corren en Python, token por token.
- Eso suma unos 130 ms fijos por frame, y hace que la comparación AR contra decodificación paralela no sea pareja. Afecta a los 733 ms de gpt2m, a los 131 y 158 ms del GPT-S, al "10 veces más rápido" y al título de la slide 8.
- Propuesta: un `GPTStream` con el frame entero en un graph, igual que las otras streams. Después hay que medirlo en la 3080 Ti y en la 4060 Laptop (sección "Prueba en la 3080 Ti"). No se sabe todavía cuánto baja: hay una estimación, no una medición.

## Evidencia

Los 733 ms no son un error de cálculo ni un valor suelto: salen del JSON y se repiten (735 ms el 15/11 en el commit `a655ed3`, 733 ms el 18/11 en `results/bench/gpt2m__fp16_graphs.json`, con p99 casi igual a p50). Esta sección concluía que ese tiempo lo domina el host y no la GPU; **la medición del 21/11 lo refutó** (ver "Resultado"): la evidencia es compatible con un decode limitado por la latencia de los kernels seriales de la GPU.

- **El tiempo del GPT-S no depende del contexto.** En el commit `a655ed3`, `gpt_s__fp16_graphs`, `_ctx4` y `_ctx8` daban 156,1 / 155,2 / 156,0 ms con el contexto completo, de 4 y de 8 frames. Si fuera cómputo de la GPU, cambiaría.
- **El GPT-S sigue a la CPU.** Es más rápido en la 4060 Laptop (`results/bench/gpt_s_kd_4060__fp16_graphs.json`, 131,2 ms) que en la 3080 Ti (`results/bench/gpt_s_kd__fp16_graphs.json`, 158,2 ms), aunque la 4060 es una GPU mucho más lenta. Genie-S con 4 pasos hace lo contrario: 5,7 ms en la 3080 Ti y 9,9 ms en la 4060.
- **La latencia es lineal en los parámetros, con una ordenada de unos 130 ms.** En la 3080 Ti (commit `a655ed3`): 11,5M parámetros, 156 ms; 26M, 211 ms; 89M, 337 ms; 307M, 735 ms. Son unos 130 ms fijos más unos 2 ms por millón de parámetros. Los 130 ms fijos son ~1 ms por token de overhead, repetido 128 veces.
- **gpt2m queda muy lejos del límite de memoria.** 733 ms son 5,7 ms por token: unos 150 GB/s efectivos, contra 912 GB/s de la placa. La 3080 Ti le gana solo 1,25 veces a la 4060 (916,3 ms en `results/bench/gpt2m_4060__fp16_graphs.json`), que tiene unas 3,5 veces menos ancho de banda. En lotes de 16, `results/video_gpt2m.json` (`ms_per_frame_batch`) da 1845 ms por lote, o 115 ms por clip: con batch 1 el paso está limitado por latencia, no por memoria.
- **La cadena no pierde tiempo en buffers.** El pipeline CUDA ya es residente en la GPU (estado del stream, cachés KV y tokens en el device, `codec.py` con un graph por etapa, un solo cruce pinned por sentido en `bench_e2e.py`). En `results/infer/latencia_genie_s_p4__rtx3080ti.json`, el p50 de la cadena cerrada es 15,58 ms y la suma de sus etapas, 3,80 + 5,57 + 6,14 = 15,52 ms. El camino por ONNX Runtime (DirectML, NPU) sí pasa por la CPU en cada etapa, pero no se usa en las slides.
- **Estimación, no medición.** Sin los ~130 ms fijos, el GPT-S quedaría en unos 25–60 ms y gpt2m en unos 350–600 ms. gpt2m seguiría muy lejos de tiempo real; el GPT-S podría quedar en el borde de los 50 ms.

## Qué cambiar

Los nombres son los de las funciones, no líneas, porque el código se está limpiando en otras ramas.

1. `src/models/base.py`: `sample_logits(logits, temperature, generator, noise=None)`. Si llega `noise`, lo usa en lugar de `exponential_` (mismo cálculo, `argmax(probs / noise)`).
2. `src/models/gpt.py` y `src/models/gpt2m.py`: `next_frame(..., noise=None, graphs=None)`.
   - `noise` es un tensor `(128, B, 1024)` con una tirada por token, en el mismo orden en que hoy consume el generador. Sin `noise`, nada cambia: los rollouts por lotes de la evaluación usan este camino y no se tocan.
   - `graphs=False` fuerza el camino sin graph por paso, porque los graphs no se anidan.
   - `Gpt2m` necesita `_ensure_cache(B, dtype, dev)` (que llame a `_setup`), como `GPT._ensure_cache`, para asignar las cachés antes de capturar.
3. `src/infer/stream.py`: `GPTStream(TokenStream)`, calcada de `MaskGITStream` y `GenieStream`.
   - `_noise(B)` genera las 128 tiradas con el generador del stream, una por una (`buf[i].exponential_(1, generator=gen)`), así los tokens coinciden con los de `next_frame` con el mismo generador. Con `temperature == 0` no hace falta ruido.
   - `_frame(ctx, noise)` llama a `next_frame(..., noise=noise, graphs=False)`.
   - Con el contexto lleno y graphs activos, `_sample_next` lo corre dentro de un `GraphedStep` (`src/models/cudagraph.py`). Si no, cae a `next_frame`, como las otras streams.
   - Llamar a `m._ensure_cache` antes de capturar y rehacer el graph si `m._graphs` cambió.
4. `make_stream` (`src/infer/stream.py`): despachar `GPT` y `Gpt2m` a `GPTStream`. `src/bench.py` y `src/infer/bench_e2e.py` ya pasan por `make_stream`, así que miden la ruta nueva sin cambios.
5. Tests, siguiendo `test_maskgit_fused_frame_equals_next_frame` y `test_graphs_equal_eager_cuda` de `tests/test_infer.py`:
   - en CPU, `next_frame(noise=...)` con el ruido sacado del mismo generador da los mismos tokens que `next_frame(generator=...)`, para `random:gpt_s` y para un `Gpt2m` chico;
   - en CPU, `GPTStream` sin graphs da los mismos tokens que `TokenStream`;
   - en CUDA (se saltea sin GPU), con graphs da lo mismo que en eager.
6. Riesgos:
   - El graph de un frame de gpt2m es grande: 128 pasos, 24 capas y unos 17 kernels por capa son unos 50.000 nodos (estimación). Hay que vigilar el tiempo de captura y la memoria; si no entra, capturar en 2 a 4 trozos de 32 pasos.
   - El primer segundo (contexto incompleto) sigue por el camino sin graph, como las otras streams.

Opcional, solo gpt2m, después de medir lo anterior: la atención del decode usa una máscara booleana sobre las 2584 posiciones de la caché estática, y el prefill también lleva máscara en vez de `is_causal`. Se puede probar sin máscara sobre la parte usada y con `--int8` (ya soportado en `src/bench.py`).

## Prueba en la 3080 Ti

Antes: el código tiene que estar commiteado y la GPU libre. Una acción de GPU con el árbol sucio corre, pero queda anotada y no cuenta como `vigente` (`CONTRIBUTING.md`).

1. Tests: `python -m pytest tests/test_infer.py`. Los de CUDA corren solos en esa máquina.
2. Equivalencia con la GPU: `python -m src.cli gpt evaluar --tamaño s --variante kd --cadena`. Escribe `results/infer/imagine_check_gpt_s_kd.json`, y `tokens_equal` tiene que dar 1,0 (como `results/infer/imagine_check_genie_s_p4.json`).
3. Latencia, con los mismos comandos con los que se midieron las cifras viejas (están en `results/bitacora.jsonl`):
   - `python -m src.cli gpt bench --tamaño s --variante kd`
   - `python -m src.cli gpt bench --tamaño s --variante kd --cadena --ritmo 50`
   - `python -m src.cli gpt2m bench`
4. Qué comparar. Las cifras viejas quedan en git:

   | Modelo | Modelo solo (p50), 3080 Ti | Cadena cerrada (p50) |
   |---|---|---|
   | `gpt_s_kd` | 158,2 ms | 168,1 ms |
   | `gpt2m` | 733,0 ms | no medida |

   <!-- ok-numero -->
5. En la 4060 Laptop, igual que en la bitácora: `python -m src.cli gpt bench --tamaño s --variante kd --aleatorio -- --name gpt_s_kd_4060` y `python -m src.cli gpt2m bench -- --name gpt2m_4060 --frames 50`.
6. Repetir cada medición al menos dos veces y mirar p99 contra p50.
7. Cómo leerlo:
   - Si desaparecen los ~130 ms fijos, la hipótesis era correcta, y hay que regenerar todo (sección siguiente).
   - Si el tiempo casi no baja, la hipótesis era incorrecta. Entonces hay que perfilar con `torch.profiler` (kernels por paso) y mirar si domina la atención o el muestreo.
   - Si baja la cifra de gpt2m pero sigue por encima de los 350 ms, es el límite de la atención y las capas, no del bucle de Python.

## Después de tener las cifras

Regenerar, en este orden:

1. `python -m src.cli reporte`, para `results/summary.json` y `results/summary.md`.
2. `python presentacion/generar_datos.py`, `python presentacion/armar.py` y `python paper/hacer_figuras.py`.
3. Buscar con `grep` los valores viejos (733, 732,9, 158, 131, 916) y revisar uno por uno: `Docs/resultados.md`, `README.md`, `presentacion/guion.md` (a mano), `paper/main.tex` (abstract, introducción y la sección de latencia: "Los GPT no llegan a tiempo real" y "Variación entre GPUs") y `paper/tablas/principal.tex`.

## Qué cambia en el mensaje si el GPT-S entra a 50 ms

- El título de la slide 8 ("El GPT chico no llega a 20 FPS; MaskGIT y Genie sí") y la conclusión 1 dejan de valer tal cual.
- El "10 veces más rápido" de MaskGIT-S con 4 pasos contra el GPT-S pasa a ser una diferencia mucho menor.
- Lo que sí se mantiene: el GPT genera 128 tokens en serie y la decodificación paralela sigue siendo más rápida, pero la brecha deja de ser un resultado de implementación.
- gpt2m sigue sin llegar a tiempo real en cualquier caso.

## Otros hallazgos de la revisión de la presentación

Son independientes de la optimización, y quedan para decidir con el equipo.

1. **GPU sin etiquetar.** El GPT-S aparece con 131 ms (slides 4 y 8; 4060) y con 158 ms (slides 7 y 11; 3080 Ti), y gpt2m con 733 ms (slides 1, 3 y 7; 3080 Ti) y 916 ms (slide 8, "casi un segundo"; 4060). En la slide 8, el 131 ms de la 4060 está junto al 22 ms de la 3080 Ti, sin GPU en ninguno. La slide 6 dice "siempre la misma GPU".
2. **Spearman (respaldo R3).** El p de 0,04 sale de `scipy.spearmanr`, que usa una aproximación t (`src/report.py`, `results/summary.json`). Con n = 5 y rho de 0,90, el p exacto por permutación es 0,083 a dos colas. <!-- ok-numero -->
3. **MaskGIT a 20 FPS.** MaskGIT-S se midió solo como modelo solo, en la 4060, sin encoder, decoder ni p99. `Docs/resultados.md` dice que con la cadena completa sostienen 20 FPS solo Genie y el ConvLSTM. El título de la slide 8 y la conclusión 2 lo extienden a MaskGIT.
4. **Diferencia pareada de la slide 9.** El "−81 a 3 s" sale de la re-corrida `results/video_gpt_s_kd_r.json`, y la tabla y la slide 8 usan `results/video_gpt_s_kd.json`. Restando se obtiene −88. Falta aclarar que es la re-corrida pareada.
5. **p-valores en el piso.** Los "p = 0,01" son el mínimo del bootstrap con 200 remuestreos (2/201 = 0,00995): debería decir "p ≤ 0,01". El "p ≥ 0,33" de Genie-M contra Genie-S es en rigor 0,328.
6. **Latentes (slide 10).** "Lo mismo que el encoder sin entrenar" no vale para todos: LeWM entrenado da R² de 0,08 a 0,10 contra 0,04 sin entrenar, y V-JEPA 2-S entrenado da 0,03 contra 0,09. <!-- ok-numero -->
7. **"Los mejores chicos" (slide 8).** El rango 370–468 incluye al GPT-S destilado, que a 3 s no es de los mejores; los mejores van de 370 a 425.
8. **Notas del orador.** `presentacion/guion.md` conserva un `[pendiente: falta el rollout de MaskGIT-S]` en la slide 7, aunque el sprite ya está.

Verificado sin error: los FVD y sus IC, las diferencias pareadas, los 780.000 frames, el LPIPS, los porcentajes del dataset, el ajuste de escala, las temperaturas, el piso real contra real, los 400 clips, el encoder (3,8 ms) y el decoder (6,1 ms), el p99 de 22 ms de la cadena de Genie-S, y los conteos de las slides 1, 4 y 7. `datos.js`, `presentacion.html` y `notas.js` se regeneran idénticos a los commiteados.
