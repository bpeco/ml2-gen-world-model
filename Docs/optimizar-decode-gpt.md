# Optimizar el decode de GPT y gpt2m

Surgió de revisar las métricas de la presentación (`presentacion/`) el 20 de noviembre de 2026. **Estado (21/11/2026): implementado, medido en la RTX 3080 Ti y en `main`.** La hipótesis original (overhead del host) resultó falsa: el tiempo lo dominaba un kernel de atención con la GPU casi vacía. Cambiarlo, junto con el ruido del muestreo en un solo lanzamiento, bajó el GPT-S destilado de 154 a 38 ms por frame y gpt2m de 767 a 291 ms, sin tocar los modelos (ver "Resultado"). <!-- ok-numero --> La presentación ya está actualizada: la optimización es un tema aparte, al final de los resultados (slides 11 y 12; la 12 muestra los SM antes y ahora), con el perfil por kernel en el respaldo 3. El informe también (`paper/main.tex`, sección "Optimizar la atención del decode", con la figura `paper/figs/optimizacion.pdf`). El mecanismo (qué es un kernel, por qué el kernel de antes usaba 6 de los 80 SM y cómo funciona `cached_attention`), con diagramas, está en [`Docs/atencion-decode.md`](atencion-decode.md). Las secciones de más abajo son la propuesta original. Cuando se decida qué hacer con esto, este documento se borra o se archiva: lo vigente son los JSON de `results/`.

## Resultado (21/11/2026)

Cada paso tiene su commit y está medido con la CLI, con el árbol limpio, en la RTX 3080 Ti (modelo solo, p50, batch 1, fp16; ms por frame). Los commits de los pasos 0 y 2 están en la rama `pasos-optimizacion`, unida a `main` con `-s ours`: no cambia el código de `main`, y así esas mediciones cuentan como vigentes. <!-- ok-numero -->

| Paso | Commit | GPT-S destilado | gpt2m | Qué cambia en el hardware |
|---|---|---|---|---|
| 0. PyTorch eager | `42f563f` (`--sin-graphs`) | 430,2 | 1595,1 | La CPU lanza desde Python ~13.000 kernels por frame (gpt2m: ~47.600); con graphs, esos kernels ocupan ~153 ms de GPU, así que el resto es lanzar |
| 1. Un CUDA graph por token (`GraphedStep`) | `79b6cae` (lo que había) | 154,0 y 155,0 | 767,1 | La CPU dispara 128 graphs por frame; el muestreo de cada token sigue afuera |
| 2. El frame entero en un graph (`GPTStream`) | `e142b9c` | 156,0 y 157,6 | 730,4 | La CPU dispara un graph, más 128 lanzamientos del ruido del muestreo; no cambia nada: el límite ya era la GPU |
| 3. Atención explícita (`cached_attention`) | `8b3ee3c` | 40,1, 40,0, 44,7 y 44,3 | 312,7 y 312,3 | La atención pasa a 4 kernels cortos (el q·kᵀ, en 1008 bloques), sin un bloque por cabeza que recorra la caché en serie |
| 4. El ruido del muestreo en un lanzamiento (`frame_noise`) | `c796313` | **37,6, 37,6 y 37,9** | **289,7 y 290,8** | La CPU ya no demora el graph del frame |
<!-- ok-numero -->

**Diagnóstico** (`python -m src.cli gpt bench --tamaño s --variante kd -- --perfil`: desglose por kernel con `torch.profiler` y el grid de cada kernel del trace de CUPTI, en `results/perfil/`).
- Con el código del paso 2 (`f58358b`, `results/perfil/gpt_s_kd_perfil.json`), la atención era el 81 % del tiempo de GPU del GPT-S: 123,7 de 153,1 ms por frame, en 768 llamadas de 161 µs a `fmha_cutlassF` (el kernel mem-efficient de SDPA). <!-- ok-numero -->
- En las 762 llamadas del decode, ese kernel lanza un grid de 1 × 6 × 1: **6 bloques, uno por cabeza, para los 80 SM** de la GPU, con una ocupación media del 1,2 %. Cada bloque recorre en serie las 2584 posiciones de la caché estática. En gpt2m son 16 bloques, y la atención era el 70 % (`results/perfil/gpt2m_perfil.json`). <!-- ok-numero -->
- Este torch (2.11+cu128, Windows) no trae FlashAttention. Con máscara y una sola query, SDPA no tiene un kernel que reparta la caché entre bloques.
- Eso explica la evidencia de la propuesta:
  - el costo era ~200 µs por capa y por token en S, M, L y gpt2m (todos tienen head_dim 64 y la misma caché);
  - no dependía del contexto (se recorre la caché entera igual);
  - la 4060 le ganaba a la 3080 Ti (los kernels con poca ocupación dependen del reloj por SM y de la L2).

**Cambios.**
- `cached_attention` (`src/models/backbone.py`, y en gpt2m un `forward` propio solo mientras la caché está enganchada, sin tocar `third_party`) hace la misma cuenta que SDPA, la del backend math que fuerza `decode_n_tokens` de comma.
  - Son 4 kernels por capa: q·kᵀ con `baddbmm` (1008 bloques), el softmax, el producto con v y una copia.
  - Tardan 14,9 µs en total, contra 161 (`results/perfil/gpt_s_kd_perfil_attn.json`). <!-- ok-numero -->
  - Se usa con caché y T ≤ 129: el decode del GPT y las pasadas de MaskGIT. El prefill sigue por SDPA.
  - La máscara aditiva se calcula una vez, en `setup_cache`.
- `frame_noise` (`src/models/base.py`): las 128 tiradas Exp(1) de un frame salen de una sola llamada al generador.
  - Con 128 llamadas, la CPU hacía 128 lanzamientos antes del graph del frame y la GPU los esperaba.
  - Según el núcleo en que quedaba el proceso, el GPT-S daba 40 o 44,5 ms: muy estable dentro de una corrida, distinto entre corridas.
  - Con una sola llamada quedan 37,6 a 37,9 ms en todas las corridas. <!-- ok-numero -->
  - `next_frame` tira igual, así que el stream y `next_frame` siguen dando los mismos tokens con el mismo generador.

**Tiempo de GPU por frame, después:** 38,9 ms en el GPT-S y 292,3 en gpt2m. Lo que queda son ~12.900 kernels chicos por frame en el GPT-S (matmul, layer norm, indexado, elementwise). Bajar de ahí pide fusionarlos (`torch.compile`, que en Windows necesita Triton) o menos pasos por frame. <!-- ok-numero -->

**Cadena completa del GPT-S** (`results/infer/latencia_gpt_s_kd__rtx3080ti.json`):
- Encoder 3,8 + modelo 38,2 + decoder 6,1 ms: 48,2 ms de p50 y 48,4 de p99, tanto en bucle cerrado como a ritmo de cámara, con 0 % de frames tarde. <!-- ok-numero -->
- Entra en los 50 ms, sin margen. Antes daba 172,6 ms de p50 (`latencia_gpt_s_kd_base_hoy`). <!-- ok-numero -->
- Una corrida anterior con el paso 3 (`latencia_gpt_s_kd_attn`) dio p99 de 57 a 58 ms y 10 a 14 % de frames tarde. Coincidió con scripts de Python en la CPU: las latencias se miden con la máquina quieta. <!-- ok-numero -->

**Otros modelos.**
- MaskGIT-S usa la misma red y también bajó, con 4 pasos de 9,32 a 6,39 ms (`maskgit_s_abs_p4_base_hoy` contra `maskgit_s_abs_p4`, pesos al azar). <!-- ok-numero -->
  - Con 1 a 12 pasos va de 3,9 a 13,0 ms. Ahora está medido en la 3080 Ti; antes, solo en la 4060. <!-- ok-numero -->
  - El GPT-S pasa de ser 16,5 veces más lento que MaskGIT-S con 4 pasos a serlo 5,9 veces. <!-- ok-numero -->
- Genie no cambia: su atención es otra clase (`Attn`, en `src/models/genie.py`), que llama a SDPA directamente y no pasa por `cached_attention`. Tarda 13,5 µs por llamada, el 14 % de su tiempo de GPU (`results/perfil/genie_s_p4_perfil.json`; ese perfil es anterior a que el bench guardara los bloques de cada kernel). <!-- ok-numero -->

**Calidad.** El modelo es el mismo; cambian el redondeo fp16 del kernel y la secuencia de números al azar, no la distribución.
- `gpt evaluar --tamaño s --variante kd --cadena --limite 5` (T = 0,9) da un FVD de 219,4 / 384,8 / 472,9 a 1, 2 y 3 s, contra 206,7 / 357,8 / 468,5 de los rollouts de la tabla. <!-- ok-numero -->
- La diferencia pareada (`results/diferencia_gpt_s_kd_stream_vs_gpt_s_kd.json`) es +12,6 (p = 0,18), +27,0 (p = 0,05) y +4,4 (p = 0,76). Son dos muestras del mismo modelo. A 1 y 3 s no hay diferencia; a 2 s queda en el borde (IC del 95 % de 1,4 a 52,5), y con tres horizontes no alcanza para distinguirla del ruido. <!-- ok-numero -->
- La tabla de calidad no se rehace.

**Dónde están los números y su estado.**
- Las latencias canónicas son lo vigente, y de ahí salen el reporte y la presentación: `gpt_s_kd__fp16_graphs`, `gpt2m__fp16_graphs`, `maskgit_s_abs_p{1,2,4,8,12}` y `latencia_gpt_s_kd__rtx3080ti`.
- Los pasos intermedios conviven con su nombre: `*_eager`, `*_base_hoy`, `*_framegraph` y `*_attn`.
- Las mediciones de la RTX 4060 Laptop del GPT-S, gpt2m y MaskGIT-S son del código de antes y pasaron a `protocolo anterior`.
  - Las de MaskGIT-S estaban guardadas sin etiqueta de dispositivo, como si fueran de la 3080 Ti, y ahora se llaman `maskgit_s_abs_4060_p*`.
  - Hay que volver a medirlas en esa máquina.
- Genie en la 4060 sigue vigente.

**Informe (21/11/2026).** `paper/main.tex` ya usa las cifras nuevas: todas las latencias en la RTX 3080 Ti (la figura de calidad contra latencia y la tabla principal dejaron la columna de la 4060, que era del código de antes), y una subsección nueva con la optimización. Lo nuevo está marcado con `\editado{}` y se ve en azul en `main_azul.tex`. La sección "Anticipar en línea" quedó como estaba (4060, código de antes). Falta compilarlo (Overleaf o tectonic).

## Primer intento: el frame entero en un CUDA graph (21/11/2026)

Se implementó `GPTStream` como estaba propuesto (`sample_logits(noise=)`, `next_frame(noise=, graphs=)`, `Gpt2m._ensure_cache`, `GPTStream` y el despacho en `make_stream`) y se midió con la CLI, en la 3080 Ti, fp16 con graphs, batch 1 y contexto lleno. **El frame entero en un graph no reduce la latencia: los ~130 ms fijos no eran overhead de Python del host.**

| Medición (p50, p99 entre paréntesis) | Código viejo, hoy (`79b6cae`) | Frame en un graph (`e142b9c`) | Medición vieja guardada |
|---|---|---|---|
| GPT-S destilado, modelo solo | 154,0 (158,6) y 155,0 (160,0) | 156,0 (168,5) y 157,6 (169,9) | 158,2 |
| gpt2m, modelo solo | 767,1 (779,2) | 730,4 (735,3) | 733,0 |
| GPT-S, cadena cerrada | 172,6 (187,0) | 163,5 (163,9) | 168,1 |
| GPT-S, cadena a ritmo de cámara (50 ms) | 165,1 (165,4) | 163,6 (163,9) | no guardada |
<!-- ok-numero -->

- Las diferencias son de unos pocos por ciento y cambian de signo de una medición a otra: ruido. Con el código viejo, la medición de hoy (154–155 ms) reproduce la guardada (158 ms). <!-- ok-numero -->
- **Equivalencia con pesos reales**: con el frame en un graph, `tokens_equal` = 1,0 y `clips_identical` = 1,0 sobre los 400 clips, contra `results/rollouts/gpt_s_kd.npy` (corrida de la otra sesión, con el árbol sucio). La captura del frame entero de gpt2m (~48.000 nodos) funciona sin partirla.
- Las primeras corridas, hechas con otros procesos en la CPU (181, 870 y 415 ms) y con el árbol sucio, se repitieron; sus líneas quedan en `results/bitacora.jsonl`. <!-- ok-numero -->

**Qué significa.** El tiempo de decode es de GPU, no del host: el paso de decode ya corría como graph, y juntar los 128 en uno no cambia nada. El perfil de la sección anterior mostró qué lo dominaba. El frame en un graph se queda igual, porque saca a la CPU del camino: con la CPU cargada, el decode viejo midió 415 ms en vez de 154. <!-- ok-numero -->

## Resumen

- El decode de `gpt` y `gpt2m` no está optimizado como el de MaskGIT y Genie. `MaskGITStream` y `GenieStream` (`src/infer/stream.py`) capturan el frame entero (prefill, pasos y muestreo) en un solo CUDA graph. `gpt` y `gpt2m` usan `TokenStream` y `next_frame`, que capturan un solo paso de decode: el bucle de 128 tokens, el muestreo y las copias a los buffers estáticos corren en Python, token por token.
- Eso suma unos 130 ms fijos por frame, y hace que la comparación AR contra decodificación paralela no sea pareja. Afecta a los 733 ms de gpt2m, a los 131 y 158 ms del GPT-S, al "10 veces más rápido" y al título de la slide 8.
- Propuesta: un `GPTStream` con el frame entero en un graph, igual que las otras streams. Después hay que medirlo en la 3080 Ti y en la 4060 Laptop (sección "Prueba en la 3080 Ti"). No se sabe todavía cuánto baja: hay una estimación, no una medición.

## Evidencia

Los 733 ms no son un error de cálculo ni un valor suelto: salen del JSON y se repiten (735 ms el 15/11 en el commit `a655ed3`, 733 ms el 18/11 en `results/bench/gpt2m__fp16_graphs.json`, con p99 casi igual a p50; ese JSON hoy tiene la medición de después de la optimización). Esta sección concluía que ese tiempo lo domina el host y no la GPU; **la medición del 21/11 lo refutó** (ver "Resultado"): la evidencia es compatible con un decode limitado por la latencia de los kernels seriales de la GPU.

- **El tiempo del GPT-S no depende del contexto.** En el commit `a655ed3`, `gpt_s__fp16_graphs`, `_ctx4` y `_ctx8` daban 156,1 / 155,2 / 156,0 ms con el contexto completo, de 4 y de 8 frames. Si fuera cómputo de la GPU, cambiaría.
- **El GPT-S sigue a la CPU.** Es más rápido en la 4060 Laptop (`results/bench/gpt_s_kd_4060__fp16_graphs.json`, 131,2 ms) que en la 3080 Ti (`results/bench/gpt_s_kd__fp16_graphs.json`, 158,2 ms el 18/11; hoy ese JSON tiene la medición de después de la optimización), aunque la 4060 es una GPU mucho más lenta. Genie-S con 4 pasos hace lo contrario: 5,7 ms en la 3080 Ti y 9,9 ms en la 4060.
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

1. **GPU sin etiquetar.** *(Resuelto el 21/11: la presentación usa solo la RTX 3080 Ti.)* El GPT-S aparece con 131 ms (slides 4 y 8; 4060) y con 158 ms (slides 7 y 11; 3080 Ti), y gpt2m con 733 ms (slides 1, 3 y 7; 3080 Ti) y 916 ms (slide 8, "casi un segundo"; 4060). En la slide 8, el 131 ms de la 4060 está junto al 22 ms de la 3080 Ti, sin GPU en ninguno. La slide 6 dice "siempre la misma GPU".
2. **Spearman (respaldo R3).** El p de 0,04 sale de `scipy.spearmanr`, que usa una aproximación t (`src/report.py`, `results/summary.json`). Con n = 5 y rho de 0,90, el p exacto por permutación es 0,083 a dos colas. <!-- ok-numero -->
3. **MaskGIT a 20 FPS.** *(En parte resuelto el 21/11: MaskGIT-S está medido en la 3080 Ti, 6,4 ms con 4 pasos; sigue sin cadena completa ni p99.)* MaskGIT-S se medía solo como modelo solo, en la 4060, sin encoder, decoder ni p99. `Docs/resultados.md` dice que con la cadena completa sostienen 20 FPS solo Genie y el ConvLSTM. El título de la slide 8 y la conclusión 2 lo extienden a MaskGIT.
4. **Diferencia pareada de la slide 9.** El "−81 a 3 s" sale de la re-corrida `results/video_gpt_s_kd_r.json`, y la tabla y la slide 8 usan `results/video_gpt_s_kd.json`. Restando se obtiene −88. Falta aclarar que es la re-corrida pareada.
5. **p-valores en el piso.** Los "p = 0,01" son el mínimo del bootstrap con 200 remuestreos (2/201 = 0,00995): debería decir "p ≤ 0,01". El "p ≥ 0,33" de Genie-M contra Genie-S es en rigor 0,328.
6. **Latentes (slide 10).** "Lo mismo que el encoder sin entrenar" no vale para todos: LeWM entrenado da R² de 0,08 a 0,10 contra 0,04 sin entrenar, y V-JEPA 2-S entrenado da 0,03 contra 0,09. <!-- ok-numero -->
7. **"Los mejores chicos" (slide 8).** El rango 370–468 incluye al GPT-S destilado, que a 3 s no es de los mejores; los mejores van de 370 a 425.
8. **Notas del orador.** *(Resuelto el 21/11: ya no está el pendiente.)* `presentacion/guion.md` conserva un `[pendiente: falta el rollout de MaskGIT-S]` en la slide 7, aunque el sprite ya está.

Verificado sin error: los FVD y sus IC, las diferencias pareadas, los 780.000 frames, el LPIPS, los porcentajes del dataset, el ajuste de escala, las temperaturas, el piso real contra real, los 400 clips, el encoder (3,8 ms) y el decoder (6,1 ms), el p99 de 22 ms de la cadena de Genie-S, y los conteos de las slides 1, 4 y 7. `datos.js`, `presentacion.html` y `notas.js` se regeneran idénticos a los commiteados.
