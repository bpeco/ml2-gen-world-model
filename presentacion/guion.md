# Guion de la presentación final

Diez minutos más preguntas. Los segundos están estimados a unas 140 palabras por minuto. Las slides de respaldo (después de la 12) no cuentan: son para las preguntas. Este texto es también el de las notas del orador: en la presentación, la tecla N las muestra con el tiempo objetivo y un cronómetro (`python presentacion/armar.py` las actualiza).

| Slide | Quién | Segundos |
|---|---|---|
| 0 · Portada | Bautista | 20 |
| 1 · El problema | Bautista | 35 |
| 2 · Pipeline | Bautista | 60 |
| 3 · Qué cambió | Bautista | 50 |
| 4 · Cómo se genera un frame | Bautista | 50 |
| 5 · Qué comparamos | Tomás | 35 |
| 6 · Cómo medimos | Tomás | 30 |
| 7 · Video lado a lado | Tomás | 50 |
| 8 · Calidad contra latencia | Tomás | 65 |
| 9 · Pasos y horizonte | Kevin | 50 |
| 10 · Lo que no funcionó | Kevin | 60 |
| 11 · Conclusiones | Kevin | 45 |
| 12 · Limitaciones | Kevin | 25 |
| **Total** | | **575 (9:35)** |

Bautista: 3:35. Tomás: 3:00. Kevin: 3:00.

Teclas: flechas o clic para avanzar, R repite la animación de la slide, N muestra las notas, T reinicia el cronómetro, F pone pantalla completa, Fin va a la última slide.

---

## 0 · Portada (Bautista, 20 s)

Un world model predice cómo sigue un video de manejo. Si lo hace bien, sirve como simulador para el software de manejo. comma.ai ya entrena así a openpilot. A la derecha, un frame real y los 128 tokens en que lo comprime commaVQ.

## 1 · El problema (Bautista, 35 s)

*La línea de tiempo se arma sola.*

El problema es la velocidad. La cámara saca un frame cada 50 milisegundos. gpt2m, el GPT que publicó comma, tarda 733 milisegundos en generar uno: mientras tanto, la cámara grabó 15. Es 1,4 frames por segundo. Además, genera los 128 tokens de cada frame de a uno: son 128 pasadas en fila.

En la pre-entrega nos preguntamos cuánto se puede achicar y acelerar un world model sin perder calidad. Hoy les contamos qué encontramos.

## 2 · Pipeline (Bautista, 60 s)

*La animación arranca sola: primero el carril de arriba, después el de abajo.*

Lo gris es de comma y lo usamos tal cual. Lo naranja es lo que entrenamos nosotros.

Arriba, el entrenamiento. El dataset de comma ya viene comprimido en tokens, así que el encoder no corre al entrenar. Nuestro modelo agarra los tokens de los frames anteriores y aprende a predecir los del frame siguiente, con cross-entropy. El GPT-S destilado suma un maestro: el GPT de comma, fijo. El alumno aprende a imitar sus probabilidades.

Abajo, la generación. Entra un segundo de video real. El encoder de comma lo pasa a tokens en unos 4 milisegundos. Nuestro modelo genera los 128 tokens del frame siguiente, y el decoder los vuelve imagen en unos 6. El frame nuevo entra al contexto: así generamos 60 frames, tres segundos.

El ConvLSTM y los modelos latentes quedan afuera de este esquema: uno predice píxeles y los otros no tienen decoder.

## 3 · Qué cambió (Bautista, 50 s)

Este es el roadmap de la pre-entrega. Lo completamos, pero en el camino cambiamos cosas.

Se mantuvo la pregunta, el dataset, el encoder y el decoder de comma, y gpt2m como referencia y maestro.

Cambió la métrica principal: ahora es FVD, porque la cross-entropy no se puede comparar entre modelos con máscara y autorregresivos. Para la decodificación paralela usamos la dinámica de Genie, como en su paper, y un MaskGIT con la misma red que el GPT. El ConvLSTM ahora trabaja sobre píxeles. Y medimos de nuevo la latencia de comma con un protocolo fijo: 733 milisegundos. El 1,25 segundos de la pre-entrega no tenía el hardware documentado.

Sumamos Genie-M, la cadena completa, la estadística y los modelos latentes.

## 4 · Cómo se genera un frame (Bautista, 50 s)

*Las dos grillas corren con el mismo reloj.*

Las dos son la misma red, de 12,4 millones de parámetros, y corren con el mismo reloj, en cámara lenta. A la izquierda, el GPT genera token a token: cada pasada agrega un token y mira solo los anteriores. Son 128 pasadas por frame: 131 milisegundos.

A la derecha, MaskGIT arranca con el frame enmascarado y en cada paso fija los tokens más confiables, mirando el frame entero. Con 4 pasos tarda 12,8. A los 50 milisegundos, el GPT lleva 48 tokens y MaskGIT ya generó 3 frames. Mientras el GPT completa uno, MaskGIT genera 10.

## 5 · Qué comparamos (Tomás, 35 s)

Comparamos seis familias. Las que entrenamos usan la receta de su paper y el mismo presupuesto: 780 mil frames. El GPT de comma es la referencia, y el GPT-S es su versión 25 veces más chica. MaskGIT-S usa la misma red que el GPT-S: entre los dos cambia solo la forma de decodificar. Genie suma atención espacio-temporal. El ConvLSTM predice píxeles. Las referencias simples miden qué tan difícil es la tarea, y los latentes son un contraste: no generan video.

## 6 · Cómo medimos (Tomás, 30 s)

Todos se miden igual. Cada modelo recibe un segundo de video real y genera tres, alimentándose de sus propias predicciones. Son 400 clips de validación. La métrica principal es FVD: compara la distribución de los videos generados con la de los reales, con el movimiento incluido. La medimos a 1, 2 y 3 segundos. La latencia es el tiempo de generar un frame con batch 1, y para comparar usamos siempre la misma GPU.

## 7 · Video lado a lado (Tomás, 50 s)

*Primero corre el segundo de contexto real en los seis.*

Este es el mismo clip para todos. Ahora cada modelo genera a la velocidad que medimos. gpt2m tarda 733 milisegundos por frame: en estos tres segundos llega a 4 frames. El GPT-S destilado llega a 18. Genie y el ConvLSTM van al ritmo de la cámara.

*Empieza la segunda pasada.*

Ahora los pasamos todos a 20 frames por segundo, para comparar la calidad. gpt2m sigue de cerca la escena real: mantiene los postes y la calle. Los chicos de tokens generan una calle creíble desde el principio, pero se apartan de la escena real. El ConvLSTM da un promedio borroso.

**[pendiente: falta el rollout de MaskGIT-S; lo tiene Tomás]**

## 8 · Calidad contra latencia (Tomás, 65 s)

*El gráfico se arma por partes.*

Este es el resultado central. En el eje horizontal, milisegundos por frame, en escala logarítmica. En el vertical, FVD a 3 segundos: más abajo es mejor. A la izquierda de la línea punteada se genera en tiempo real.

gpt2m tiene la mejor calidad, pero en esta GPU tarda casi un segundo por frame. El GPT-S destilado es 25 veces más chico y sigue lejos: 131 milisegundos, porque sigue haciendo 128 pasadas.

Las curvas son la decodificación paralela; cada punto es un número de pasos. Todas entran en los 50 milisegundos. MaskGIT-S con 4 pasos tiene la misma calidad que el GPT-S y es 10 veces más rápido. Con 8 a 12 pasos es el mejor modelo chico a 3 segundos.

A la derecha, el costo de achicar: el FVD pasa de 105 a entre 370 y 468. Y con encoder y decoder, la cadena completa tiene un p99 de 22 milisegundos. Ningún frame llega tarde.

## 9 · Pasos y horizonte (Kevin, 50 s)

A la izquierda, el FVD a 3 segundos según los pasos; el sombreado es el intervalo del 95 %. Con 1 o 2 pasos, el modelo decide casi todos los tokens a la vez, sin ver los de al lado, y el error se acumula: Genie con 1 paso pasa de 279 a 1307 entre 1 y 3 segundos. Desde 4 pasos se estabiliza y alcanza la calidad de 25, que es lo que usa el paper.

A la derecha, el FVD según el horizonte. El GPT-S es el mejor chico a 1 segundo, pero se degrada más que los de decodificación paralela con 4 o más pasos. A 3 segundos gana la decodificación paralela con 8 o más pasos: MaskGIT-S con 8 le saca 81 de FVD, y la diferencia pareada es significativa.

## 10 · Lo que no funcionó (Kevin, 60 s)

Cuatro cosas no funcionaron, y las contamos como resultado.

Primero, predecir píxeles. El ConvLSTM queda peor que repetir el último frame: su FVD a 1 segundo es 1790, contra 929 del frame congelado. Con este presupuesto, promedia y sale borroso.

Segundo, los modelos latentes. Una regresión lineal desde sus embeddings explica a lo sumo el 10 % de la velocidad: lo mismo que el encoder sin entrenar, y menos que los píxeles crudos.

Tercero, el LPIPS. Con LPIPS, el frame congelado le gana a todos los modelos chicos. Un video quieto se parece píxel a píxel a la continuación real, aunque no sea creíble.

Cuarto, agrandar sin más datos. Genie-M tiene más del doble de parámetros y empata con Genie-S. Cada modelo vio 780 mil frames; gpt2m vio 4.600 veces más. El GPT-S tiene 8 tokens por parámetro y Genie-M, unos 3; Chinchilla sugiere unos 20.

## 11 · Conclusiones (Kevin, 45 s)

*Las conclusiones aparecen de a una.*

Uno: un world model de manejo se puede achicar de 307 a 12 o 15 millones de parámetros y generar a 20 frames por segundo solo si cambia la forma de decodificar. La cadena completa usa menos de la mitad del presupuesto.

Dos: el GPT chico, token a token, no llega. Con la misma red, MaskGIT con 4 pasos da la misma calidad 10 veces más rápido.

Tres: el FVD a 3 segundos de los mejores chicos es de 3,5 a 4,5 veces el de gpt2m.

Cuatro: vieron el 0,02 % de los datos de gpt2m, y varios resultados indican que están limitados por datos. Cuánto mejorarían con más, no lo podemos estimar con dos puntos.

## 12 · Limitaciones (Kevin, 25 s)

Las limitaciones: el presupuesto de datos, la temperatura de MaskGIT, que deja sus números conservadores, la latencia, que depende de la GPU, y el ruido del FVD. No evaluamos difusión, ni acciones en los modelos generativos. Lo que sigue es entrenar los chicos con todo commaVQ, 150 veces más datos, sin cambiar la latencia. Gracias.

---

## Respaldo, para las preguntas

- R1, tabla completa: todos los modelos con CE, LPIPS, FVD a 1, 2 y 3 s, drift y latencia en las dos GPU.
- R2, cadena completa: encoder, modelo y decoder por etapa, con el p99 y los frames tarde a ritmo de cámara.
- R3, datos y escala: el GPT-S de cero con 25 % y 50 % de los datos, el ajuste y la extrapolación (con la advertencia: dos puntos y E fijo). Correlación entre CE y FVD.
- R4, FVD real contra real: por qué no es una cota inferior y por qué leemos el FVD solo para comparar.
- R5, temperatura: FVD en el holdout según la temperatura; la del paper de Genie triplica el FVD.
- R6, figura del informe, FVD según el horizonte: todos los modelos, con las referencias simples.
- R7, figura del informe, frames del clip 0: incluye MaskGIT-S y el frame congelado.
- R8, arquitecturas: enlaces a la hoja de cada modelo (arquitecturas.html, de Tomás).
