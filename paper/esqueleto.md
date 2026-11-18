# Esqueleto del paper y de la presentación

Borrador de trabajo, al 18/11/2026. Los números salen de los JSON vigentes de `results/` (cola hasta el trabajo 11, RTX 3080 Ti). Lo marcado **[pendiente]** depende de trabajos de la cola que todavía no corrieron.

---

## Título tentativo

*¿Cuánto se puede achicar un world model de manejo? Calidad contra latencia de modelos autoregresivos, de decodificación paralela y de píxeles sobre commaVQ*

## La lógica del paper, en un párrafo

Un world model de manejo sirve como simulador si genera video **creíble** y **rápido** (a 20 FPS, menos de 50 ms por frame). La referencia pública, el GPT de comma (307M), tiene buena calidad pero es grande y autoregresivo token a token. Nos preguntamos cuánta calidad se pierde al **achicar** el modelo unas 20–25 veces (destilación) y al **cambiar la forma de decodificar** (MaskGIT: todos los tokens de un frame en pocas pasadas paralelas). Comparamos contra baselines triviales y contra un modelo de píxeles, con un protocolo de evaluación único (FVD, LPIPS, CE con intervalos por bootstrap) y un protocolo de latencia único. Como contraste, probamos dos modelos latentes no generativos (LeWM, V-JEPA 2) y vemos que, con nuestro presupuesto, no aprenden una representación útil de la dinámica.

## Aportes (los que podemos defender)

1. Un **benchmark reproducible sobre commaVQ**: split, 400 clips fijos, horizontes de 1, 2 y 3 s, intervalos por bootstrap por segmento y temperatura elegida en un holdout. Todo se reproduce con un comando y queda en una bitácora. *Antes de decir "el primero", chequear Scholar y arXiv.*
2. Una **frontera calidad–latencia** con modelos chicos (8–15M) contra el maestro de 307M. **[pendiente: latencia]**
3. Un **análisis del número de pasos de decodificación paralela**: con 4 pasos se alcanza la calidad de 25, y con 1–2 pasos la calidad se degrada con el horizonte.
4. Lecciones de evaluación: el LPIPS premia el video estático, y comparar contra "real contra real" necesita cuidado (ver más abajo).

---

## 1. Introducción (0,75 pág.)

- **Para qué sirve un world model de manejo:** simular, entrenar políticas sin el auto real y anticipar. comma.ai entrena políticas con uno ("Learning to Drive from a World Model", 2025).
- **Por qué importa la velocidad:** un simulador en lazo cerrado tiene que ir al ritmo de la cámara. A 20 FPS hay 50 ms por frame, y 128 tokens autoregresivos cuestan 128 pasadas.
- **Pregunta:** ¿cuánto se puede achicar y acelerar sin perder calidad?
- **Qué hacemos**, en una frase por eje: tamaño (destilación), decodificación (MaskGIT/Genie), representación (tokens, píxeles o latentes).
- **Resultados principales**, en 3 viñetas cuando estén los números finales.

## 2. Trabajo relacionado (0,5 pág.)

- **Tokens + transformer:** IRIS (la receta de commaVQ) y GAIA-1.
- **Decodificación paralela:** MaskGIT y Genie (ST-transformer + MaskGIT para video).
- **Píxeles:** ConvLSTM (Shi et al. 2015) como baseline clásico. DIAMOND y DiT como el camino de difusión, que mencionamos y no probamos.
- **Latentes / JEPA:** V-JEPA 2 y LeWM. No son generativos: predicen embeddings.
- **Destilación** para achicar modelos (soft targets).
- Fuente: `Docs/world-models-conduccion-papers.md`.

## 3. Datos y protocolo (0,75 pág.)

- **commaVQ:** frames comprimidos por un VQ-VAE a 8×16 = 128 tokens (vocabulario de 1024), a 20 FPS, con poses.
  - Train: shard 0, 2.500 segmentos de 1 min.
  - Val: los 100 segmentos oficiales.
  - Holdout aparte para elegir la temperatura.
- **Presupuesto común:** 780.000 frames vistos por modelo. Cada modelo entrena con la receta de su paper (tabla de fidelidad), y el lr se ajusta con una corrida corta de 300 pasos ("humo") solo donde cambió el batch.
- **Evaluación de video:**
  - 400 clips (100 segmentos × 4 arranques), 1 s de contexto y 3 s generados.
  - **FVD** (I3D, ventanas de 16 frames que terminan a 1, 2 y 3 s) como métrica principal.
  - **LPIPS** por horizonte como complemento.
  - **CE** en tokens solo para los modelos autoregresivos (la de MaskGIT no es comparable).
  - Intervalos por bootstrap por segmento.
- **Temperatura:** elegida por familia en el holdout, nunca en val. GPT: T = 0,9. Genie: T = 1,0, porque el 2,0 del paper da FVD 1739.
- **Latencia:**
  - **Modelo solo:** un frame, batch 1, contexto lleno, fp16 con CUDA graphs, p50 y p99, con la GPU libre y siempre la misma (3080 Ti).
  - **Cadena:** encoder → modelo → decoder, medida lo más rápido posible y a ritmo de cámara (% de frames tarde).
  - También se registran los parámetros y la memoria pico.

## 4. Modelos (0,75 pág.)

| modelo | qué es | params | por qué está |
|---|---|---|---|
| `gpt2m` | GPT de comma, autoregresivo por token | 307M | maestro y techo de calidad |
| `gpt_s_kd` | GPT-S destilado de `gpt2m` (receta IRIS) | 12M | **eje tamaño** |
| `gpt_s` | GPT-S de cero | 12M | control de H1 y primer punto de escala **[pendiente: trabajo 17]** |
| `gpt_m` | GPT-M de cero | ~25–30M | segundo punto de escala (sección 6, Chinchilla) **[pendiente: trabajo 18]** |
| `genie_s` | dinámica de Genie + decodificación MaskGIT, 1–25 pasos | 15M | **eje decodificación paralela** |
| `maskgit_s` | MaskGIT con el backbone del GPT | ~12M | comparación AR vs paralelo con la misma arquitectura **[pendiente: PC de Tomás]** |
| `convlstm_px` | ConvLSTM de Shi sobre píxeles | 8M | **eje representación**: píxeles |
| `unigram`, `transicion`, `congelado` | baselines triviales | – | qué tan difícil es la tarea |
| `lewm`, `vjepa2` | modelos latentes | 18M / 43M | contraste no generativo |

## 5. Resultados (1,5–2 pág.)

### 5.1 Calidad de video (tabla principal)

| modelo | CE | FVD 1 s | FVD 3 s | LPIPS 1 s |
|---|---|---|---|---|
| gpt2m | 2,03 | 67 | 105 | 0,17 |
| gpt_s_kd | 2,99 | 207 | 469 | 0,30 |
| genie_s, 4 pasos | – | 232 | 413 | 0,28 |
| genie_s, 8 pasos | – | 276 | 425 | 0,27 |
| genie_s, 25 pasos | – | 326 | 444 | 0,26 |
| congelado | 4,89 | 929 | 978 | 0,24 |
| transición | 3,48 | 729 | 1241 | 0,52 |
| convlstm_px | – | 1790 | 2400 | 0,45 |
| unigram | 6,85 | 1471 | 1524 | 0,57 |

Puntos a tocar:
- **El costo de achicar:** con 20–25 veces menos parámetros, el FVD a 3 s pasa de 105 a unos 410–470, es decir, unas 4 veces peor. Aun así, los modelos chicos quedan muy por encima de todos los baselines.
- **GPT vs Genie se cruzan:** el GPT destilado gana a 1 s y Genie gana a 3 s. Hipótesis: el error compuesto token a token acumula más drift. **Chequear si los intervalos se solapan** (trabajo 16, diferencia pareada).
- **ConvLSTM es peor que congelar el frame:** la regresión a la media en píxeles da video borroso. Es el argumento a favor de los tokens discretos.
- **El LPIPS no alcanza:** `congelado` tiene mejor LPIPS a 1 s que todos los modelos chicos y, sin embargo, su FVD es 929. Por eso el FVD es la métrica principal.

### 5.2 Pasos de decodificación (H2): figura

- FVD a 1/2/3 s contra el número de pasos (1, 2, 4, 8, 12, 25).
- Con 1 paso: 279 → 1307 (colapsa con el horizonte). Con 2 pasos: 228 → 683. Con 4 pasos o más: estable, alrededor de 413–444.
- Un detalle: a 1 s, más pasos dan peor FVD pero mejor LPIPS (fidelidad contra diversidad). Mencionarlo, sin sobreinterpretar.
- **[pendiente: ms por número de pasos]** para cerrar "con 4 pasos se gana X veces en velocidad sin perder calidad".

### 5.3 Calidad contra latencia (figura central) [pendiente: trabajo 14]

- FVD a 3 s contra ms/frame (p50), con una línea vertical en 50 ms, la frontera de Pareto marcada y el tamaño del punto proporcional a los parámetros.
- La misma figura para la cadena completa, con el codec incluido: ¿quién llega a 20 FPS de punta a punta?
- H3: fp32 contra fp16 + CUDA graphs (aceleración). La pérdida de calidad, si se mide.

### 5.4 Horizonte (drift)

- FVD y LPIPS contra segundos (las curvas ya existen en `video_*.json`). Muestran el compounding error de cada familia.

### 5.5 ¿La CE predice el FVD? (H4)

- En los 5 modelos con CE (gpt2m, gpt_s_kd, transición, congelado, unigram), el orden por CE coincide con el orden por FVD (Spearman = 1). Con n = 5 es evidencia débil; decirlo así.

### 5.6 Destilación (H1) [pendiente: trabajo 17]

- `gpt_s_kd` contra `gpt_s`: CE y FVD.

### 5.7 Contraste: modelos latentes (párrafo corto o apéndice)

- Probes lineales de pose (velocidad, giro) sobre los embeddings:
  - **LeWM:** R² de velocidad 0,08, contra 0,04 de un encoder sin entrenar y 0,14 de los píxeles crudos.
  - **V-JEPA 2:** 0,03, peor que el encoder sin entrenar (0,09).
  - El giro da ≈ 0 en todos los casos.
- LeWM predice casi igual que copiar el último latente.
- **Lectura:** con 780k frames, estos métodos no aprenden la dinámica. Es un resultado negativo honesto y una limitación del presupuesto, no una conclusión sobre los métodos.

## 6. Discusión y limitaciones (0,5 pág.)

- **El "piso" de FVD (real contra real, 103 a 1 s) no es una cota inferior:** gpt2m da 67. El piso compara dos sets de escenas distintas, mientras que cada modelo se compara con la continuación real de su propio contexto. Presentarlo como "referencia entre escenas", o sacarlo.
- **Presupuesto:** 780k frames es mucho menos que lo de los papers (gpt2m vio 3M de minutos). Los modelos chicos están subentrenados, y la brecha con el maestro mezcla tamaño y datos.
- **Escala de entrenamiento (Chinchilla) y por qué para inferencia conviene sobreentrenar:**
  - Chinchilla (Hoffmann et al. 2022) ajusta L(N, D) = E + A/N^α + B/D^β y encuentra que, para un cómputo fijo, lo óptimo es D ≈ 20 tokens por parámetro. Los coeficientes son de texto: no se trasladan a tokens de video, y hay que ajustarlos con datos propios.
  - Nuestros chicos ven D/N ≈ 8 (`gpt_s_kd`: 12,4M parámetros y 100,6M tokens), así que están ~2,5× por debajo del óptimo de Chinchilla. `gpt2m` está del otro lado: entrenado sobre hasta ~4,6·10¹¹ tokens, D/N ≈ 1500.
  - Para un simulador en tiempo real, el óptimo no es Chinchilla: el entrenamiento se paga una vez y la latencia se paga en cada frame. Conviene el modelo más chico que entre en los 50 ms, entrenado mucho más allá de 20 tokens por parámetro (Sardana et al. 2024, *Beyond Chinchilla-Optimal*). Es lo que hizo comma con `gpt2m`.
  - **Mini-estudio [pendiente]:** `gpt_s` con presupuestos de 0,25×, 0,5× y 1× (corridas separadas, cada una con su coseno completo) y `gpt_m` con 1×. Se ajusta L(D) = E + B/D^β para N fijo y se extrapola la CE a 3× y 10×. Con H4 (CE ↔ FVD), se traduce a un FVD esperado. Si sobra GPU, una corrida 3× verifica la predicción.
  - Lectura para el paper: la brecha con el maestro mezcla capacidad (N) y datos (D). El estudio separa cuánto se recupera solo con entrenar más, sin tocar la latencia.
- Una sola GPU (3080 Ti) para la latencia. La 4060 de laptop va como fila aparte.
- El FVD es ruidoso: los intervalos son anchos (±80). Hay 3 semillas solo para genie_s y convlstm_px **[pendiente: trabajos 15 y 16]**.
- Los modelos latentes se entrenan sobre frames reconstruidos por el VQ-VAE, no sobre video crudo.
- No probamos difusión (DIAMOND, DiT), que es hacia donde fue comma.

## 7. Conclusión (0,25 pág.)

- Respuesta a la pregunta, con números: cuánto se achica, cuánto se acelera y cuánto se pierde.
- Qué haríamos con más cómputo: más datos para los chicos, difusión y speculative decoding.

## Figuras (orden de prioridad)

1. **FVD contra ms/frame**, con Pareto y la línea de 50 ms. **[pendiente: trabajo 14]**
2. **FVD contra número de pasos de Genie**, con una curva por horizonte.
3. **FVD contra horizonte** para todos los modelos.
4. **Tira de frames:** real, gpt2m, gpt_s_kd, genie_s (4 pasos) y convlstm_px a 0,5, 1, 2 y 3 s. Es la figura cualitativa; sale de `demo`.
5. (Apéndice) CE contra FVD.

## Pendientes que bloquean el paper

| qué | dónde | bloquea |
|---|---|---|
| Latencia (trabajo 14) | 3080 Ti | figura 1, H2, H3 |
| `gpt_s` de cero (trabajo 17) | 3080 Ti | H1, estudio de escala |
| `gpt_m` de cero (trabajo 18) | 3080 Ti | estudio de escala (2 tamaños) |
| `gpt_s` con 0,25× y 0,5× del presupuesto | 3080 Ti o PC de Tomás | curva L(D) para extrapolar |
| Semillas y estadística (trabajos 15 y 16) | 3080 Ti | si GPT vs Genie y 4 vs 8 pasos son diferencias reales |
| `maskgit_s` | PC de Tomás | AR vs paralelo con la misma arquitectura |
| `python -m src.cli reporte` | CPU | tablas y figuras |
| Confirmar que no hay un benchmark previo sobre commaVQ | Scholar y arXiv | aporte 1 |

---

## Presentación (10–12 slides)

Reutiliza lo que ya está en `pre-entrega/index.html`: animaciones del flujo, carrera de velocidad y roadmap.

1. **Título** y para qué sirve un simulador de manejo.
2. **El problema:** calidad y velocidad (50 ms por frame).
3. **commaVQ:** video → 128 tokens por frame → video (la animación de la pre-entrega).
4. **Tres ejes:** tamaño (destilar), decodificación (MaskGIT) y representación (tokens, píxeles o latentes).
5. **Cómo medimos:** FVD/LPIPS/CE con bootstrap, temperatura en el holdout, latencia p50.
6. **Video lado a lado:** real, gpt2m, chicos y ConvLSTM. Es la slide que más impacta.
7. **Tabla de calidad:** el costo de achicar.
8. **Pasos de MaskGIT:** con 4 alcanza; con 1–2 hay drift.
9. **Figura central:** calidad contra latencia, quién llega a 20 FPS.
10. **Lo que no funcionó:** ConvLSTM borroso, latentes sin señal y el LPIPS engañoso.
11. **Conclusiones y limitaciones.**
12. **Reproducibilidad:** un comando y una bitácora.
