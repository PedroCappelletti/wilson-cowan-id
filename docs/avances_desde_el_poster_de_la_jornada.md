---
type: note
status: inbox
project: "[[Investigación Neurociencia]]"
fecha: 2026-10-14
tags: [gray-box, wilson-cowan, avances, poster, dataset-ampliado, warm-start]
---

# Avances desde el póster de la Jornada

Qué hay de nuevo respecto del póster de la III Jornada de Tecnología y Salud, ordenado para mostrarlo como avance. Primero lo que ya estaba hecho y no entró en el póster, después lo que se hizo pasado el congreso, y al final lo que quedó planteado sin resultados.

Relacionado: [[Para la próxima reunión]] · [[Dataset ampliado - figuras]] · [[Arranque desde el white-box]] · [[Reporte - con y sin ruido]].

---

## 0. Punto de partida: lo que dijo el póster

1. Sin ruido y desde el arranque ignorante, el backbone ampliado es el mejor, por delante de toda corrección entrenable, e identifica β dentro del 6 % hasta σ = 0,01.
2. En la planta con actuador, una corrección que usa entradas pasadas y presentes (el filtro entrenable de 8 canales) cierra cerca de la mitad de esa diferencia sin forma impuesta.
3. Igualando el arranque, la agnóstica sobre refractariedad también identifica: de 26 a 5,5 % de error de β con σ = 0,01. En el actuador no se repite.

Todo con un dataset de 20 escenarios (13 de entrenamiento, 7 de test), sin validación, una semilla, y evaluando la última época.

---

## 1. Lo que no entró en el póster

Estaba hecho antes del congreso y quedó en el complemento.

### 1.1 Reproducir no es identificar

El modelo tiene dos caminos para tapar el término faltante: representarlo en la corrección, o deformar β hasta que la trayectoria coincida. Los dos bajan el NRMSE; solo el primero sube el $R^2$ y deja β cerca del verdadero.

La agnóstica sobre refractariedad, sin ruido, reproduce a 2,84 % de NRMSE con 25,6 % de error de β y $R^2 = -0{,}49$. El white-box solo queda en 11,5 %. Reproduce bien y deja los parámetros peor que si no se hubiera corregido nada. La distinción importa si el modelo se va a reusar en lazo cerrado o en otra preparación.

### 1.2 Un escenario decidía las comparaciones

El promedio de los siete escenarios de test es sin ponderar, y `box_a1.2` es el más difícil y el más disperso: mediana de 27,7 % contra 8 a 13 % del resto. Con un séptimo del peso aportaba cerca de un tercio del promedio, y alcanzaba para dar vuelta una comparación. Era el único escalón de test, con una amplitud que el entrenamiento nunca mostró. Eso motivó el dataset ampliado (2.1).

### 1.3 El filtro entrenable por dentro

- **Es una red con la primera capa restringida.** Resume los 800 valores pasados del comando en pocas sumas pesadas.
- **Cuántos canales usa de verdad.** La dimensión efectiva es 2,7 con 4 canales, 3,9 con 8 y 7,1 con 16. El mínimo teórico es 4 (versión filtrada y valor actual de P y de Q). Los 16 canales usan más direcciones y aun así reproducen peor.
- **El núcleo no se puede interpretar como física.** Muestras vecinas del comando tienen correlación 0,990, así que núcleos muy distintos dan la misma salida: un núcleo suave y uno en zigzag difieren en 370 % y sus salidas en 2,6 %. Por eso los núcleos entrenados no se parecen a la exponencial del actuador.
- **La penalización L2 no lo arregló.** Reparte la energía entre retardos vecinos en vez de concentrarla, y las corridas sin ruido divergieron al final del entrenamiento.

![[nucleos-4-y-8-canales.png]]

### 1.4 El warm start, completo

- En el actuador no identifica en ninguna variante: entre 27 y 37 % de error de β, igual o peor que el white-box.
- Congelar β y entrenar solo la corrección es lo peor en las dos plantas. La ganancia viene de arrancar desde el white-box y seguir entrenando todo junto.
- Con σ = 0,05 la identificación en refractariedad solo se sostiene con la ventana de 20 ms.

### 1.5 Modos de falla que se resolvieron

- **La forma exacta colapsaba al white-box** porque el coeficiente de refractariedad quedaba trabado en el borde de su restricción. Se arregló arrancando desde el white-box: de 5,92 a 1,70 %.
- **El estado de filtro daba τ a la mitad** (0,56 contra 1 ms) por un transitorio falso al inicio de cada ventana. Calculando el estado inicial exacto sobre todo el comando bajó de 12,16 a 2,96 %, con τ = 0,98 ms.
- **Cuatro corridas divergían al final** del entrenamiento y se evaluaba la última época. Se resolvió después del congreso, con la selección por validación (2.1).
- **Sin suavizado, la corrección con memoria persigue el ruido**: $R^2$ de −17,2 contra −1,44 con la media móvil.

### 1.6 El parámetro más difícil es $w_{II}$

El análisis de Fisher predijo, sin entrenar, que $w_{II}$ sería el menos identificable (número de condición 1182). Un barrido de ruido lo confirmó: con σ = 0,10, $w_{II}$ llega a 41 % de error mientras el resto queda debajo de 16 %. Un estímulo ponderado hacia ese cuello de botella baja su error de 93 a 19 %.

### 1.7 El ruido es poco realista

El ruido blanco actual da 41 y 27 dB de SNR dentro de la banda de la señal. Una estimación sobre las grabaciones reales da una cota de unos −7 dB. Casi todo el ruido simulado cae fuera de banda y la media móvil lo saca con facilidad.

---

## 2. Lo nuevo después del congreso

### 2.1 Dataset ampliado con validación

- **49 escenarios**: 28 de entrenamiento, 7 de validación (uno por familia de estímulos) y 14 de test (dos por familia). Los 20 originales están adentro sin cambios.
- **El modelo que se evalúa es el de mejor validación**, no el de la última época. Eso elimina el problema de las corridas que divergían al final, y las elecciones de arquitectura pasan a hacerse por validación y no mirando el test.
- **Por qué `box_a1.2` pesaba tanto.** La familia de escalones tiene dos regímenes: hasta amplitud 0,8 la actividad pico queda por debajo de 0,17, y desde 1,0 salta a 0,7–0,8. Los escalones de entrenamiento viejos estaban en el régimen bajo y el de test en el alto. Ahora hay escalones de entrenamiento en los dos.

### 2.2 Desde cero contra desde el white-box, en los tres niveles de ruido

66 corridas: las doce configuraciones del póster, cada una desde cero y desde el white-box, con y sin ruido.

**Refractariedad: con warm start, la agnóstica identifica β igual o mejor que la forma exacta.** Error de β / $R^2$:

| agnóstica | sin ruido | σ = 0,01 | σ = 0,05 |
|---|---|---|---|
| 5 ms, desde cero | 18,1 % / 0,47 | 19,2 % / 0,50 | 14,8 % / 0,31 |
| 5 ms, warm start | **3,4 % / 0,83** | **5,6 % / 0,81** | **5,7 % / 0,79** |
| 20 ms, desde cero | 17,7 % / 0,05 | 17,7 % / 0,03 | 15,6 % / 0,27 |
| 20 ms, warm start | **2,7 % / 0,79** | **3,9 % / 0,72** | **6,1 % / 0,81** |
| forma exacta | 3,8 % / 0,97 | 3,5 % / 0,97 | 8,4 % / 0,90 |

La forma exacta sigue recuperando mejor el término faltante, pero en β la agnóstica con warm start la iguala o la supera, y reproduce igual. La forma exacta desde cero colapsa al white-box en los tres niveles.

**Actuador: el warm start mejora la reproducción, pero ata β al error del white-box.** Sin ruido, la historia de 100 baja de 6,40 a 4,35 % de NRMSE y el filtro de 4 canales de 6,46 a 4,14 %. El error de β queda en 24 a 32 %, alrededor del del white-box, y el $R^2$ cerca de cero.

![[ampliado-arranque-sin-ruido.png]]
![[ampliado-arranque-n01.png]]
![[ampliado-arranque-n05.png]]

### 2.3 El dataset del póster contra el ampliado

Sobre los 7 escenarios de test que comparten:
- **Con σ = 0,05 la mejora es grande, sobre todo en identificación.** Desde cero, el white-box de refractariedad pasa de 17,2 a 6,1 % de error de β, la forma exacta de 17,7 a 8,4 %, y el estado de filtro de 22,1 a 11,3 % con $R^2$ de 0,71 a 0,95. Con warm start, la agnóstica de 5 ms pasa de 18,8 a 5,7 % y de $R^2$ −1,79 a 0,79.
- **Con σ = 0,01** las agnósticas y las historias identifican mejor, y los backbones expandidos un poco peor.
- **Sin ruido** mejora sobre todo la reproducción: la forma exacta baja de 1,70 a 0,58 %.

![[ampliado-vs-poster-n05.png]]
![[ampliado-vs-poster-wrm-n05.png]]

### 2.4 Dos predicciones que no se cumplieron

Antes de ampliar el dataset se habían anotado dos predicciones falsables. Ninguna se cumplió con claridad:
- **El óptimo de canales del filtro no subió.** Desde cero, por validación, 4, 8 y 16 canales quedan a 0,3 puntos. Con warm start el de 8 queda apenas mejor.
- **El óptimo de ventana no se alargó.** Por validación, la de 5 ms queda apenas por delante de la de 20 ms.

Conclusión: el tamaño del dataset no era lo que limitaba esas dos elecciones.

### 2.5 Cuánto contamina la perturbación a β, sin conocer su forma

Un white-box por familia de estímulos, en el control sin perturbación y en las dos plantas (66 corridas). Si el backbone alcanzara, todas las familias darían el mismo β. Lo que se dispersa por encima del control es lo que aporta la perturbación.

| dispersión de β entre familias | control | refractariedad | actuador |
|---|---|---|---|
| sin ruido | 1,9 % | 7,7 % | **29,5 %** |
| σ = 0,01 | 4,8 % | 8,5 % | **23,6 %** |
| σ = 0,05 | 13,5 % | 17,0 % | **25,3 %** |

- **El actuador contamina β de forma distinta según el estímulo**: entre 5 y 15 veces la dispersión del control. Lo peor está en los estímulos de pulsos rápidos.
- **La limitación:** en refractariedad todas las familias sobreestiman $w_{II}$ en la misma dirección (de +35 a +112 %). Un sesgo común no aumenta la dispersión, así que este diagnóstico no lo ve. Sirve para la contaminación que depende del estímulo, no para la que es igual en todos.

![[consistencia-beta-sin-ruido.png]]

---

## 3. Planteado, sin resultados todavía

- **Memoria en la corrección.** La historia del comando alcanza cuando lo oculto depende solo del comando, como en el actuador. Para una adaptación que depende del estado, o para algo no observado, haría falta otra representación (historia del estado, estado explícito, o una red recurrente tipo LSTM o GRU o un observador). Hay una comparación de diez arquitecturas y una escalera de tres plantas propuesta.
- **Ruido más parecido al real.** Ruido $1/f^\chi$ con la SNR fijada dentro de banda, después ruido sobre la señal medida, y al final ruido de proceso. El generador ya está escrito.
- **Observar el LFP en vez de I y E.** El mismo modelo, pero ajustado contra una sola señal que combina las dos poblaciones. Hay cuatro proxies de la literatura y un piloto armado y probado, en `experimental/lfp/`, marcado como por verse.
- **Separar el backbone de la perturbación sin conocer su forma.** Sin ningún supuesto no se puede. Hay cinco supuestos que no tocan la forma de la perturbación: previos sobre β, condiciones experimentales donde β no cambia, intervenciones conocidas, consistencia entre familias (probado en 2.5) y medir la ambigüedad en vez de resolverla.

El detalle de cada uno, con lo que hay que decidir, está en [[Para la próxima reunión]].

---

## Dónde está todo

- Figuras: [[Dataset ampliado - figuras]] (y el PDF en `attachments/`), y en el repo `wilson-cowan-id`, `results/figures/ampliado/`.
- Resultados: `results/escalado/e13_*` (arranque y dataset ampliado) y `e14_*` (consistencia entre familias); tabla en `results/consistencia_beta.md`.
- Complemento del póster: `Desktop/complemento/complemento.pdf`, fuente de la sección 1.
