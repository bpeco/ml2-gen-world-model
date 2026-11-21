# Guion de la presentación final

Diez minutos más preguntas; con las dos slides de optimización (11 y 12), el texto completo lleva unos 11:50, así que hay que recortar unos 110 segundos. Los segundos están estimados a unas 140 palabras por minuto, redondeados de a 5. Las slides de respaldo (después de la 14) no cuentan: son para las preguntas. Este texto es también el de las notas del orador: en la presentación, la tecla N las muestra con el tiempo objetivo y un cronómetro (`python presentacion/armar.py` las actualiza).

| Slide | Segundos |
|---|---|
| 0 · Portada | 20 |
| 1 · El problema | 35 |
| 2 · Pipeline | 60 |
| 3 · Qué cambió | 60 |
| 4 · Cómo se genera un frame | 40 |
| 5 · Qué comparamos | 35 |
| 6 · Cómo medimos | 30 |
| 7 · Video lado a lado | 40 |
| 8 · Calidad contra latencia | 70 |
| 9 · Pasos y horizonte | 55 |
| 10 · Lo que no funcionó | 65 |
| 11 · Optimización: cómo corre en la GPU | 50 |
| 12 · Optimización: los SM de la GPU | 50 |
| 13 · Conclusiones | 50 |
| 14 · Limitaciones | 50 |
| **Total** | **710 (11:50)** |

Teclas: flechas o clic para avanzar, R repite la animación de la slide, N muestra las notas, T reinicia el cronómetro, F pone pantalla completa, Fin va a la última slide.

---

## 0 · Portada (20 s)

Un world model predice cómo sigue un video de manejo. Si lo hace bien, sirve como simulador para el software de manejo. comma.ai ya entrena así a openpilot. A la derecha, un frame real y los 128 tokens en que lo comprime commaVQ.

## 1 · El problema (35 s)

*La línea de tiempo se arma sola.*

El problema es la velocidad. La cámara saca un frame cada 50 milisegundos. gpt2m, el GPT que publicó comma, tarda 291 milisegundos en generar uno, y eso ya con la optimización de la GPU que contamos al final: mientras tanto, la cámara grabó 6. Son 3,4 frames por segundo. Además, genera los 128 tokens de cada frame de a uno: son 128 pasadas en fila.

En la pre-entrega nos preguntamos cuánto se puede achicar y acelerar un world model sin perder calidad. Hoy les contamos qué encontramos.

## 2 · Pipeline (60 s)

*La animación arranca sola: primero el carril de arriba, después el de abajo.*

Lo gris es de comma y lo usamos tal cual. Lo naranja es lo que entrenamos nosotros.

Arriba, el entrenamiento. El dataset de comma ya viene comprimido en tokens, así que el encoder no corre al entrenar. Nuestro modelo agarra los tokens de los frames anteriores y aprende a predecir los del frame siguiente, con cross-entropy. El GPT-S destilado suma un maestro: el GPT de comma, fijo. El alumno aprende a imitar sus probabilidades.

Abajo, la generación. Entra un segundo de video real. El encoder de comma lo pasa a tokens en unos 4 milisegundos. Nuestro modelo genera los 128 tokens del frame siguiente, y el decoder los vuelve imagen en unos 6. El frame nuevo entra al contexto: así generamos 60 frames, tres segundos.

El ConvLSTM y los modelos latentes quedan afuera de este esquema: uno predice píxeles y los otros no tienen decoder.

## 3 · Qué cambió (60 s)

Este es el roadmap de la pre-entrega. Lo completamos, pero en el camino cambiamos cosas.

Se mantuvo la pregunta, el dataset, el encoder y el decoder de comma, y gpt2m como referencia y maestro.

Cambió la métrica principal: ahora es FVD, porque la cross-entropy no se puede comparar entre modelos con máscara y autorregresivos. Para la decodificación paralela usamos la dinámica de Genie, como en su paper, y un MaskGIT con la misma red que el GPT. El ConvLSTM ahora trabaja sobre píxeles. Y medimos de nuevo la latencia de comma con un protocolo fijo: 767 milisegundos, y 291 después de optimizar la inferencia. El 1,25 segundos de la pre-entrega no tenía el hardware documentado.

Sumamos Genie-M, la cadena completa, la estadística, los modelos latentes y la optimización de la inferencia, que contamos al final, en dos slides.

## 4 · Cómo se genera un frame (40 s)

*Las dos grillas corren con el mismo reloj.*

Las dos son la misma red, de 12,4 millones de parámetros, y corren con el mismo reloj, en cámara lenta. A la izquierda, el GPT genera token a token: cada pasada agrega un token y mira solo los anteriores. Son 128 pasadas por frame: 38 milisegundos.

A la derecha, MaskGIT arranca con el frame enmascarado y en cada paso fija los tokens más confiables, mirando el frame entero. Con 4 pasos tarda 6,4. Mientras el GPT completa un frame, MaskGIT genera 5; a los 50 milisegundos de la cámara, el GPT terminó uno y MaskGIT, 7.

## 5 · Qué comparamos (35 s)

Comparamos seis familias. Las que entrenamos usan la receta de su paper y el mismo presupuesto: 780 mil frames. El GPT de comma es la referencia, y el GPT-S es su versión 25 veces más chica. MaskGIT-S usa la misma red que el GPT-S: entre los dos cambia solo la forma de decodificar. Genie suma atención espacio-temporal. El ConvLSTM predice píxeles. Las referencias simples miden qué tan difícil es la tarea, y los latentes son un contraste: no generan video.

## 6 · Cómo medimos (30 s)

Todos se miden igual. Cada modelo recibe un segundo de video real y genera tres, alimentándose de sus propias predicciones. Son 400 clips de validación. La métrica principal es FVD: compara la distribución de los videos generados con la de los reales, con el movimiento incluido. La medimos a 1, 2 y 3 segundos. La latencia es el tiempo de generar un frame con batch 1, y para comparar usamos siempre la misma GPU.

## 7 · Video lado a lado (40 s)

*Primero corre el segundo de contexto real en los seis.*

Este es el mismo clip para todos. Ahora cada modelo genera a la velocidad que medimos. gpt2m tarda 291 milisegundos por frame: en estos tres segundos llega a 10 frames. Los chicos van al ritmo de la cámara.

*Empieza la segunda pasada.*

Ahora los pasamos todos a 20 frames por segundo, para comparar la calidad. gpt2m sigue de cerca la escena real: mantiene los postes y la calle. Los chicos de tokens generan una calle creíble desde el principio, pero se apartan de la escena real. El ConvLSTM da un promedio borroso.

## 8 · Calidad contra latencia (70 s)

*El gráfico se arma por partes.*

Este es el resultado central. En el eje horizontal, milisegundos por frame, en escala logarítmica. En el vertical, FVD a 3 segundos: más abajo es mejor. A la izquierda de la línea punteada se genera en tiempo real.

gpt2m tiene la mejor calidad, pero tarda 291 milisegundos por frame. El GPT-S destilado es 25 veces más chico y, con la inferencia optimizada, entra: 38 milisegundos. Pero queda justo: con encoder y decoder, 48 milisegundos, sin margen.

Las curvas son la decodificación paralela; cada punto es un número de pasos. Todas entran en los 50 milisegundos. MaskGIT-S con 4 pasos tiene la misma calidad que el GPT-S y es 6 veces más rápido. Con 8 a 12 pasos es el mejor modelo chico a 3 segundos.

A la derecha, el costo de achicar: el FVD pasa de 105 a entre 370 y 468. Y con Genie-S a 4 pasos, la cadena completa, con encoder y decoder, tiene un p99 de 22 milisegundos: ningún frame llega tarde.

## 9 · Pasos y horizonte (55 s)

A la izquierda, el FVD a 3 segundos según los pasos; el sombreado es el intervalo del 95 %. Con 1 o 2 pasos, el modelo decide casi todos los tokens a la vez, sin ver los de al lado, y el error se acumula: Genie con 1 paso pasa de 279 a 1307 entre 1 y 3 segundos. Desde 4 pasos se estabiliza y alcanza la calidad de 25, que es lo que usa el paper.

A la derecha, el FVD según el horizonte. El GPT-S es el mejor chico a 1 segundo, pero se degrada más que los de decodificación paralela con 4 o más pasos. A 3 segundos gana la decodificación paralela con 8 o más pasos: MaskGIT-S con 8 le saca 81 de FVD, y la diferencia pareada es significativa.

## 10 · Lo que no funcionó (65 s)

Cuatro cosas no funcionaron, y las contamos como resultado.

Primero, predecir píxeles. El ConvLSTM queda peor que repetir el último frame: su FVD a 1 segundo es 1790, contra 929 del frame congelado. Con este presupuesto, promedia y sale borroso.

Segundo, los modelos latentes. Una regresión lineal desde sus embeddings explica a lo sumo el 10 % de la velocidad: lo mismo que el encoder sin entrenar, y menos que los píxeles crudos.

Tercero, el LPIPS. Con LPIPS, el frame congelado le gana a todos los modelos chicos. Un video quieto se parece píxel a píxel a la continuación real, aunque no sea creíble.

Cuarto, agrandar sin más datos. Genie-M tiene más del doble de parámetros y empata con Genie-S. Cada modelo vio 780 mil frames; gpt2m vio 4.600 veces más. El GPT-S tiene 8 tokens por parámetro y Genie-M, unos 3; Chinchilla sugiere unos 20.

## 11 · Optimización: cómo corre en la GPU (50 s)

*Los cuatro pasos aparecen de a uno. Un clic en una barra vuelve a ese paso.*

Ahora, otro tema: cómo llegamos a esas latencias. No es aprendizaje automático, pero cambió los resultados. Es el mismo GPT-S, con los mismos pesos.

En PyTorch normal, la CPU lanza desde Python cada operación: unos 13 mil kernels por frame, y tarda 430 milisegundos. Con un CUDA graph por token, la CPU dispara cada paso con una sola llamada: 154. Llevar el frame entero a un solo graph no cambió nada, 156: el límite ya no era la CPU. Entonces medimos kernel por kernel, y la atención era el 81 % del tiempo de GPU. Cambiarla bajó el frame a 40 milisegundos, y tirar el ruido del muestreo de una sola vez, a 38, con la misma calidad.

## 12 · Optimización: los SM de la GPU (50 s)

*La animación corre las dos versiones con el mismo reloj.*

La GPU tiene 80 multiprocesadores, los SM. En el GPT-S, con la caché y un solo token, PyTorch elegía un kernel de atención que lanza 6 bloques, uno por cabeza: 74 SM quedaban sin trabajo, y cada bloque recorría la caché en 41 vueltas, una detrás de la otra. Eran unos 160 microsegundos por capa.

Escribimos la misma cuenta con multiplicaciones de matrices de cuBLAS. El producto de q por k se reparte en mil bloques y usa toda la GPU: los cuatro kernels tardan 15 microsegundos. Con eso, y el ruido del muestreo en un solo lanzamiento, el GPT-S pasó de 156 a 38 milisegundos por frame, y gpt2m, de 730 a 291.

## 13 · Conclusiones (50 s)

*Las conclusiones aparecen de a una.*

Uno: medir dónde se iba el tiempo cambió el resultado. Sin tocar el modelo, el GPT de 12 millones de parámetros bajó de 156 a 38 milisegundos por frame. Llega a 20 frames por segundo, pero justo: con encoder y decoder, 48 milisegundos.

Dos: con la misma red, decodificar en pocos pasos paralelos entra en los 50 milisegundos con margen, 6,4 milisegundos con 4 pasos, y con la misma calidad a 3 segundos, o mejor, con 8 a 12 pasos. A 1 segundo, el GPT sigue siendo mejor.

Tres: el número de pasos se elige en inferencia, sin reentrenar. Permite elegir el punto entre calidad y latencia. Con menos de 4 pasos, el video se degrada con el horizonte.

## 14 · Limitaciones (50 s)

Estos resultados valen para pocos datos: 780 mil frames por modelo, el 0,65 % de commaVQ. También para video de una cámara a 128 por 256, y para latencias con batch 1 en una RTX 3080 Ti, que dependen de los kernels que trae PyTorch. La temperatura de MaskGIT deja sus números conservadores, y el FVD es ruidoso. No evaluamos difusión, ni acciones en los modelos generativos.

Lo que sigue es entrenar los chicos con todo commaVQ, 150 veces más datos. No cambia la latencia y permite medir cuánto de la brecha cierran los datos. Después, condicionar el modelo en las acciones del auto, para usarlo como simulador de una política. Gracias.

---

## Respaldo, para las preguntas

- R1, tabla completa: todos los modelos con CE, LPIPS, FVD a 1, 2 y 3 s, drift y latencia en la RTX 3080 Ti.
- R2, cadena completa: encoder, modelo y decoder por etapa, con el p99 y los frames tarde a ritmo de cámara.
- R3, perfil por kernel: tiempo de GPU en la atención y en el resto de la red, antes y después de la atención explícita, para el GPT-S y gpt2m. El kernel de antes y sus bloques.
- R4, datos y escala: el GPT-S de cero con 25 % y 50 % de los datos, el ajuste y la extrapolación (con la advertencia: dos puntos y E fijo). Correlación entre CE y FVD.
- R5, FVD real contra real: por qué no es una cota inferior y por qué leemos el FVD solo para comparar.
- R6, temperatura: FVD en el holdout según la temperatura; la del paper de Genie triplica el FVD.
- R7, figura del informe, FVD según el horizonte: todos los modelos, con las referencias simples.
- R8, figura del informe, frames del clip 0: incluye MaskGIT-S y el frame congelado.
- R9, arquitecturas: enlaces a la hoja de cada modelo (arquitecturas.html, de Tomás).
