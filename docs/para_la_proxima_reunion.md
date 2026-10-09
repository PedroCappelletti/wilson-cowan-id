---
type: note
project: "[[Investigación Neurociencia]]"
status: active
fecha: 2026-10-04
tags: [reunion, agenda, gray-box, memoria, arquitectura, identificabilidad]
---

# Para la próxima reunión

Arriba, el punteo de todo lo que hay que discutir. Abajo, la explicación de cada tema, en el mismo orden.

Relacionado: [[Próximos pasos desde el 10-09]] · [[El filtro de la variante K - núcleos y qué probar]] · [[Arranque desde el white-box]].

## Punteo

1. **Memoria en la corrección: cuándo alcanza la historia del comando y cuándo hace falta una red recurrente.**
	- Alcanza cuando lo oculto depende solo del comando (actuador). No alcanza si depende del estado (adaptación) o de algo no observado (ruido de proceso).
	- La historia del estado realimenta el error en corrida libre, pero en la adaptación ese costo no se puede evitar con ninguna arquitectura.
	- Comparación de diez arquitecturas, incluidas la LSTM y la GRU.
	- A decidir: si se arma la escalera de tres plantas, qué representación entra primero para la adaptación, y si el caso no observado va con LSTM o GRU o con un observador continuo.
2. **Dataset ampliado y arranque desde el white-box (resultados de la tanda e13).**
	- 49 escenarios con validación. La familia de escalones tiene dos regímenes, y el escalón de test viejo caía en el que el entrenamiento no mostraba.
	- En refractariedad, el warm start hace identificar a la corrección agnóstica. En el actuador mejora la reproducción pero ata β al error del white-box.
	- A decidir: cómo se reporta, y si cambia algo del mensaje del póster.
3. **Ruido más parecido al real.**
	- El ruido actual da 27 a 41 dB de SNR dentro de banda; las grabaciones reales, alrededor de −7 dB.
	- Propuesta: ruido $1/f^\chi$ con la SNR fijada dentro de banda, después ruido sobre S, y al final ruido de proceso.
	- A decidir: si se pasa a reportar por SNR dentro de banda, qué rango de $\chi$ y de SNR se barre, y cuándo entra el ruido de proceso.
4. **Observar S, el potencial de campo, en vez de I y E (piloto armado, por verse).**
	- La literatura se divide entre una diferencia con signo (potencial de membrana) y una suma de magnitudes de corrientes.
	- A decidir: si entra en la línea principal, qué hipótesis de S se toma, y si se corre el piloto.
5. **Cómo separar el backbone de la perturbación sin conocer la forma de la perturbación.**
	- Sin ningún supuesto es imposible: cualquier cambio de β que imite parte de la perturbación lo puede compensar la red.
	- Hay supuestos que no tocan la forma de la perturbación: previos sobre β, condiciones donde β no cambia, intervenciones que lo cambian de forma conocida, consistencia de β entre familias de estímulos, y medir la ambigüedad en vez de resolverla.
	- A decidir: qué se prueba en simulación, y **qué condiciones experimentales podría registrar el equipo**.

---

## 1. Cuándo alcanza la historia del comando y cuándo hace falta memoria de verdad

### De dónde sale

Nunca usamos una LSTM ni otra red recurrente de tiempo discreto. Lo más cercano fue la latent ODE (`LatentGrayBox`): un estado oculto $z$ con $\dot z = h_\psi(z, x, P, Q)$ y la corrección $g_\phi(x, z)$. Dio $R^2 = -0{,}51$ contra $\Delta f$, y fue la corrida más cara de todas. Se dejó de lado y se pasó a darle a la red la historia del comando (variantes H y K). En el actuador funcionó.

### Por qué funciona en el actuador

El estado oculto del actuador cumple $\tau \dot P_{lag} = P - P_{lag}$. Depende **solo del comando**, y de forma lineal: es una convolución del comando con una exponencial. Un filtro lineal sobre la historia del comando lo representa de manera exacta.

### El matiz

Las plantas simuladas son deterministas, arrancan siempre del mismo punto y olvidan su pasado lejano. En ellas **todo** el estado es función de la historia del comando. La pregunta útil no es si esa historia alcanza, sino qué tan complicada es la función. Salen tres casos:

| caso | ejemplo | de qué depende lo oculto | qué haría falta |
|---|---|---|---|
| 1 | actuador (`act1`) | del comando, linealmente | historia del comando. Hecho |
| 2 | adaptación, $\dot A = (E - A)/\tau_a$ | del estado observado | historia de $(I, E)$, o $\hat A$ como estado explícito |
| 3 | ruido de proceso, otra población, una entrada desconocida | de algo no observado | estado recurrente que se corrija con lo observado (latent ODE como observador, LSTM, GRU) |

En el caso 2, $A$ también es función del comando, pero pasando por toda la dinámica no lineal de Wilson-Cowan. Desde la historia de $E$ es un filtro lineal. Ya está medido lo que pasa sin memoria: con $g(I, E, P, Q)$, el $R^2$ cae de 0,97 con $\tau_a = 1$ ms a 0,54 con 30 ms y a 0,33 con 100 ms.

En el caso 3 ninguna ventana finita alcanza, ni de comando ni de estado. La biestabilidad que apareció en la familia de escalones (ver el punto 2) también cae acá, porque con histéresis el estado depende del pasado lejano. El F7 ya midió el piso: con ruido de proceso, el oráculo da $R^2$ de 0,12.

### La objeción: la historia del estado acumula error

En la corrida libre, la historia de $(I, E)$ que recibiría la corrección es la de **las propias predicciones** del modelo. Un error en $t$ entra a la corrección en los $K$ pasos siguientes y se realimenta. Además hay un desajuste entre entrenamiento y uso:
- En el entrenamiento por ventanas, la historia anterior al arranque de cada ventana es dato observado.
- En la corrida libre es predicción.
- Con ruido se entrena con historia ruidosa y se usa con historia limpia.

La historia del comando no tiene este problema: es exógena y exacta en la corrida libre. Es una ventaja del actuador, además de la linealidad.

**Pero en la adaptación ese costo no se puede evitar.** La planta calcula $A$ desde la $E$ verdadera, y un modelo en corrida libre solo tiene su $E$ predicha. Cualquier representación arrastra el error de $E$ hacia $A$:
- **Historia de $(I, E)$:** el error entra por $K$ canales retrasados.
- **$\hat A$ como estado del ODE**, análogo al estado de filtro pero sobre $E$: el error entra por un solo canal, y filtrado.
- **LSTM o latent ODE:** el error entra porque el estado se actualiza con la $E$ predicha.

La única opción que no realimenta el error es la historia del comando, y es justamente la que tiene que aprender una función mucho más difícil.

Entonces, la comparación justa en la adaptación no es estado contra comando, sino cómo se maneja un error que existe sí o sí:
- **Ventana de entrenamiento larga respecto de $\tau_a$.** Con ventanas de 5 ms y $\tau_a$ de 30 o 100 ms, casi toda la historia que ve la corrección es dato. El entrenamiento nunca le muestra sus propios errores, y la corrida libre los castiga.
- **Estado explícito contra historia:** un canal de realimentación filtrado, contra $K$ canales.
- **Estabilidad.** Con historia del estado, el modelo pasa a ser una ecuación con retardo, y puede volverse inestable en corrida libre aunque cada ventana corta ande bien.

### Propuesta

Una escalera de tres plantas que convierta «el filtro funciona en el actuador» en una afirmación con límite conocido. Dos de las plantas ya existen en el repo (`Actuator` y `Adaptation` en `uncertainty.py`).

| planta | predicción |
|---|---|
| `act1` | historia del comando ✓ |
| adaptación, $\tau_a$ = 30 y 100 ms | la historia del comando necesita $K$ grande y no alcanza. Entre las opciones con estado, gana la que entrene con ventanas largas respecto de $\tau_a$ |
| ruido de proceso | toda corrección feedforward falla. Solo un estado recurrente con corrección por observación se acerca al piso del oráculo |

Encaja con lo que ya estaba pendiente: la adaptación figuraba como planta de control para la penalización de antigüedad del núcleo del filtro.

### Comparación de arquitecturas para darle memoria a la corrección

Todas cumplen el mismo rol: la corrección $g$ dentro de $\dot x = f_{WC}(x, P, Q; \beta) + g$. Lo que cambia es de dónde sacan la memoria.

| arquitectura | de dónde saca la memoria | ¿realimenta el error en corrida libre? | estado en el proyecto |
|---|---|---|---|
| sin memoria, $g(I, E)$ | de ningún lado | solo por el estado actual, como el backbone | medida, techo conocido |
| historia cruda del comando (H) | ventana de $K$ comandos | no | medida |
| filtro entrenable sobre el comando (K, FIR) | ventana de $K$ comandos, resumida en $n$ canales | no | medida |
| estado explícito con física (backbone expandido, $\hat P_{lag}$ o $\hat A$) | una variable de estado escrita a mano | sí, por un canal filtrado | medida en el actuador |
| historia del estado (NARX) | ventana de $K$ valores pasados de $(I, E)$ | sí, por $K$ canales | sin implementar |
| convolución temporal dilatada (TCN) | ventana larga con pocos pesos, de comando o de estado | solo si mira el estado | sin implementar |
| latent ODE (estado oculto continuo) | estado $z$ que el modelo propaga | sí | medida, $R^2 = -0{,}51$ |
| GRU o LSTM como corrección | estado oculto discreto, actualizado paso a paso | sí | sin implementar |
| neural CDE | estado continuo manejado por la trayectoria de la entrada | sí, si la entrada incluye el estado | sin implementar |
| observador (ODE-RNN, latent ODE con codificador) | estado oculto que se corrige con lo observado | sí, pero se corrige con los datos | sin implementar |

**Historia del comando, cruda o filtrada.**
- **Ventajas.** El comando es exógeno y exacto, así que el error del modelo nunca entra por ahí. Es feedforward: entrena barato y estable, y no hay estado que inicializar. Con el FIR, el núcleo se puede graficar.
- **Desventajas.** Solo sirve si lo oculto depende del comando de forma simple (caso 1). El largo $K$ es un hiperparámetro y fija la memoria máxima.
- **Problemas conocidos.** Con estímulos suaves la historia tiene rango bajo y el núcleo no está determinado: muestras vecinas con correlación 0,990, la demo del zigzag. Además, a más capacidad hay más formas de tapar el término faltante sin aprenderlo; $K = 400$ reproduce mejor y recupera menos física que $K = 100$.

**Estado explícito con física.**
- **Ventajas.** Pocos parámetros (12 o 13). Recupera $\beta$ y el término faltante, y se puede invertir para el controlador.
- **Desventajas.** Hay que saber la forma del mecanismo. No es una corrección agnóstica.
- **Problemas conocidos.** Hay que inicializar el estado oculto en cada ventana: el primer intento metía un transitorio falso que sesgaba $\hat\tau$ a 0,56 contra 1,0.

**Historia del estado (NARX).**
- **Ventajas.** Resuelve el caso 2 con un mapa lineal sobre lo observado, sin estado que inicializar.
- **Desventajas.** Realimenta el error por $K$ canales retrasados. El modelo pasa a ser una ecuación con retardo.
- **Problemas posibles.**
  - Inestabilidad en corrida libre aunque cada ventana corta ande bien.
  - Desajuste entre entrenar con historia observada (y ruidosa) y usarla con historia predicha.
  - Con ruido, el regresor viene con ruido, y eso sesga los pesos hacia cero (errores en las variables).

**Convolución temporal dilatada (TCN).** Bai, Kolter y Koltun, 2018.
- **Ventajas.** Llega a campos receptivos de cientos o miles de pasos con pocos pesos, que es lo que pide una adaptación con $\tau_a$ = 100 ms (2000 pasos de 0,05 ms). Sigue siendo feedforward.
- **Desventajas.** El núcleo se lee peor que en un FIR de una capa.
- **Problemas posibles.** Hereda el mal condicionamiento del FIR. Si mira el estado, hereda además la realimentación del error.

**Latent ODE, o Neural ODE aumentada.**
- **Ventajas.** Es continua, así que encaja con el integrador y con el resto del modelo. Sirve en principio para el caso 3.
- **Desventajas.** Fue la corrida más cara del proyecto y dio $R^2 = -0{,}51$ y $-3{,}55$.
- **Problemas conocidos.** El estado inicial por ventana no se observa (se usa $z = 0$). Además, $z$ puede absorber error de $\beta$ con total libertad.

**GRU o LSTM.** Cho et al., 2014; Hochreiter y Schmidhuber, 1997.
- **Ventajas.** Son el estándar para memoria larga en tiempo discreto. Las compuertas permiten guardar información por muchos pasos. La LSTM separa la celda de memoria del estado de salida y suele aguantar dependencias más largas que la GRU; la GRU tiene menos parámetros.
- **Desventajas y problemas posibles.**
  - Son de tiempo discreto. Dentro de un integrador RK4, el estado de la red se actualiza una vez por paso y no en las cuatro etapas, así que el modelo deja de ser una ODE bien definida y el resultado depende de $dt$.
  - Con $dt$ = 0,05 ms, una memoria de 20 a 100 ms son 400 a 2000 pasos. Entrenar con retropropagación a través de esa cantidad de pasos es lento y el gradiente se degrada, así que habría que submuestrear la entrada de la red.
  - El estado oculto es opaco. No hay núcleo que graficar ni forma de saber qué aprendió.
  - Para el controlador, el estado oculto complica la linealización por realimentación, que hoy asume un modelo de estado conocido.
  - La ambigüedad con $\beta$ crece con la capacidad, y una LSTM tiene mucha.

**Neural CDE.** Kidger et al., 2020.
- **Ventajas.** Es el análogo continuo de una red recurrente: el estado evoluciona como $dz = f_\theta(z)\, dX$, manejado por la trayectoria de la entrada. Encaja con el integrador sin el problema de la GRU o la LSTM, y está pensada para series con muestreo irregular, como los datos reales a 125 o 250 Hz.
- **Desventajas.** Más compleja de implementar. No sé de ningún uso en gray-box con backbone físico; sería algo a buscar.
- **Problemas posibles.** Los mismos de capacidad y opacidad que una GRU o una LSTM.

**Observador: ODE-RNN o latent ODE con codificador.** Rubanova, Chen y Duvenaud, 2019; GRU-ODE-Bayes, De Brouwer et al., 2019.
- **Ventajas.** Es la única familia que corrige su estado oculto con lo que observa, que es lo que pide el caso 3. Si lo oculto lo maneja algo no observado, ninguna otra arquitectura de esta tabla puede seguirlo.
- **Desventajas.** En corrida libre pura no hay observaciones con qué corregir, así que su ventaja aparece en predicción a horizonte corto o en un esquema de filtrado. Es la misma situación de los datos reales, donde lo que transfiere es la predicción a 128 ms.
- **Problemas posibles.** Es cara. Mezcla identificación con estimación de estado, lo que hace más difícil decir qué parte del acierto es del modelo y qué parte de la corrección por datos.

**Lectura de conjunto.** Para los casos 1 y 2 hay opciones feedforward o con un estado explícito chico, que son más baratas, más estables y más legibles que una LSTM. La LSTM, la GRU, la neural CDE y el observador solo se justifican en el caso 3, y entre ellas las continuas (neural CDE, observador sobre la latent ODE) encajan mejor con un modelo que se integra con RK4. Si se prueba una LSTM, conviene hacerlo como referencia de caja negra y no como corrección dentro del integrador.

### Para decidir

- [ ] ¿Vale la pena la escalera, o alcanza con presentar el actuador diciendo explícitamente cuál es su límite?
- [ ] Para la adaptación, ¿qué entra primero: historia de $(I, E)$, $\hat A$ como estado explícito, o las dos?
- [ ] ¿El caso 3 se hace con una LSTM o GRU sobre la historia observada, o con la latent ODE planteada como observador? La segunda es continua y encaja con el resto del modelo, pero ya fue la corrida más cara.
- [ ] ¿Va antes o después de las penalizaciones del núcleo? Las dos cosas usan la planta de adaptación.

---

## 2. Dataset ampliado y arranque desde el white-box

Tanda completa el 6-10: 66 corridas (`e13_*`).

Desde el 3-10 corre la tabla del póster entera sobre un dataset de 49 escenarios: 28 de entrenamiento, 7 de validación y 14 de test (tags `e13_*`, unas 55 h). El modelo que se evalúa es el de mejor validación, no el de la última época.

Al armarlo apareció algo: la familia de escalones tiene **dos regímenes**. Hasta amplitud 0,8 la actividad pico de $E$ queda por debajo de 0,17, y desde 1,0 salta a 0,7 u 0,8. El escalón de test viejo (1,2) caía en el régimen alto, que el entrenamiento (0,4 y 0,8) nunca había mostrado. Por eso pesaba tanto en el promedio.

Cada configuración se corre dos veces: desde cero (β = 1,0) y desde el β del white-box de la misma planta y el mismo ruido (warm start). Las figuras están en `wilson-cowan-id/results/figures/ampliado/` (las genera `scripts/figura_ampliado.py`).

### Contra el dataset del póster, desde cero

Sobre los 7 escenarios de test que comparten los dos datasets. Con σ = 0,01:
- Las correcciones agnósticas identifican mejor y recuperan más del término faltante. En refractariedad, el error de β baja de 24–26 % a 18–19 %, y el $R^2$ sube con las dos ventanas.
- Las historias del comando identifican mejor y reproducen apenas peor.
- Los dos backbones expandidos reproducen igual o mejor, pero identifican un poco peor: el estado de filtro pasa de 3,2 a 7,5 % de error de β.

Con σ = 0,05 la mejora es mucho más grande, sobre todo en identificación: el white-box de refractariedad pasa de 17,2 a 6,1 %, la forma exacta de 17,7 a 8,4 %, el estado de filtro de 22,1 a 11,3 % (con $R^2$ de 0,71 a 0,95), y las agnósticas de refractariedad pasan de $R^2$ negativo a positivo. Sin ruido, la mejora es sobre todo en reproducción (la forma exacta, de 1,70 a 0,58 %).

### Desde cero contra warm start

**Refractariedad.** El warm start hace que la corrección agnóstica identifique β.

| σ = 0,01 | error de β | $R^2$ contra $\Delta f$ |
|---|---|---|
| agnóstica 5 ms, desde cero → warm start | 19,2 → 5,6 % | 0,50 → 0,81 |
| agnóstica 20 ms, desde cero → warm start | 17,7 → 3,9 % | 0,03 → 0,72 |
| forma exacta (referencia) | 3,5 % | 0,97 |

En los tres niveles de ruido, error de β y $R^2$:

| agnóstica | sin ruido | σ = 0,01 | σ = 0,05 |
|---|---|---|---|
| 5 ms, desde cero | 18,1 % / 0,47 | 19,2 % / 0,50 | 14,8 % / 0,31 |
| 5 ms, warm start | **3,4 % / 0,83** | **5,6 % / 0,81** | **5,7 % / 0,79** |
| 20 ms, desde cero | 17,7 % / 0,05 | 17,7 % / 0,03 | 15,6 % / 0,27 |
| 20 ms, warm start | **2,7 % / 0,79** | **3,9 % / 0,72** | **6,1 % / 0,81** |
| forma exacta (warm start) | 3,8 % / 0,97 | 3,5 % / 0,97 | 8,4 % / 0,90 |
| forma exacta, desde cero | 8,7 % / −0,17 | 12,1 % / −0,17 | 6,0 % / −0,17 |

- Con warm start, la agnóstica identifica β igual o mejor que la forma exacta, y reproduce igual. La forma exacta sigue recuperando mejor el término faltante.
- Con σ = 0,05 el white-box ya identifica 6,1 %, y el warm start conserva ese valor mientras la red aprende el término. Sin ruido y con σ = 0,01, en cambio, lo mejora (de 8,7 y 14,0 % a 3–6 %).
- La forma exacta desde cero colapsa al white-box en los tres niveles (mismo NRMSE, mismo $R^2$), igual que en el dataset del póster: necesita el warm start.

**Actuador.** El warm start mejora la reproducción y ata β al error del white-box.
- La validación mejora en todas las correcciones con historia o filtro, con los dos ruidos. Con σ = 0,01, el filtro de 8 canales pasa de 5,56 a 3,82 %.
- El error de β queda cerca del del white-box (23 % con σ = 0,01, 31 % con σ = 0,05). Donde desde cero identificaba mejor, empeora: la historia de 400 pasa de 15,1 a 25,2 % con σ = 0,01.
- El $R^2$ queda cerca de cero en todas.
- El estado de filtro con warm start identifica peor que desde cero con σ = 0,05 (19,5 contra 11,3 %). Sin ruido y con σ = 0,01 no cambia.
- Sin ruido la mejora en reproducción es la más grande: la historia de 100 pasa de 6,40 a 4,35 %, la de 400 de 8,94 a 6,08 %, y el filtro de 4 canales de 6,46 a 4,14 %.

**Lectura.** El warm start conviene cuando el white-box ya está cerca del β verdadero (refractariedad) y no cuando está deformado (actuador). Eso lleva directo al tema 5.

**Las dos predicciones del dataset ampliado.** Ninguna se cumplió con claridad:
- El óptimo de canales del filtro no subió desde cero; con warm start, el de 8 queda apenas mejor que el de 4.
- El óptimo de ventana no se alargó: por validación, la de 5 ms queda apenas por delante de la de 20 ms.

Todo con una semilla.

### Para decidir

- [ ] ¿Cómo se reporta la comparación entre arranques? ¿Cambia algo del mensaje del póster?
- [ ] ¿Hacen falta semillas antes de sacar conclusiones sobre los óptimos de canales y de ventana?

---

## 3. Ruido más parecido al real

### Qué hay hoy

Ruido blanco gaussiano sobre $I$ y $E$, sumado por muestra a 20 kHz, con $\sigma$ = 0,01 y 0,05, y una media móvil de 7 muestras antes de entrenar. Casi toda su potencia cae fuera de la banda donde vive la señal. Dentro de banda la SNR queda en 27 y 41 dB, contra una cota de unos −7 dB en las grabaciones reales (`scripts/snr_real_vs_sim.py`). Además el ruido real no es blanco, no se mide sobre $I$ y $E$ por separado, y parte de él está dentro del sistema y no solo en la medición.

### Qué dice la literatura

- **El espectro del LFP y del EEG es aperiódico, $1/f^\chi$**, más picos oscilatorios (Donoghue et al. 2020, Nat Neurosci). El exponente depende mucho del rango donde se ajusta:
  - EEG humano entre 30 y 45 Hz: $\chi \approx 1{,}9$ en vigilia, entre 3,5 y 4,7 en sueño (Lendner et al. 2020, eLife).
  - ECoG humano: $\chi \approx 2$ por debajo de unos 75 Hz y $\approx 4$ por encima (Miller et al. 2009, PLoS Comput Biol).
  - LFP de ratón entre 4 y 200 Hz: un ajuste de dos exponentes da 0,14 y 1,98 en CA1 y 0,84 y 3,88 en giro dentado (Kühn et al. 2026, eNeuro).
  - Para una banda de 1 a 40 Hz, un $\chi$ entre 1 y 2 es razonable, pero no encontré un valor verificado.
- **El exponente se relaciona con el balance excitación-inhibición.** Más inhibición empina la pendiente entre 30 y 50 Hz (Gao, Peterson y Voytek 2017, NeuroImage). Un ruido $1/f$ no es neutro respecto de lo que se quiere identificar.
- **Generarlo es estándar.** Se le da forma al espectro de un ruido blanco (Timmer y König 1995, A&A 300:707) o se filtra (Kasdin 1995, Proc IEEE 83:802).
- **Ruido de proceso en Wilson-Cowan.**
  - Ecuaciones de Langevin derivadas del tamaño finito de la población (Bressloff 2010, Phys Rev E 82:051903). El ruido intrínseco amplifica oscilaciones por debajo del umbral.
  - Alternativa más simple: ruido Ornstein-Uhlenbeck en la entrada de las dos poblaciones (Krishnakumaran, Pavuluri y Ray 2024, J Neurosci).
- **Ruido de línea**, 50 Hz y armónicos en Argentina. Los datos que nos pasaron ya están filtrados entre 0,5 y 19 Hz, así que ahí no aparece.
- **Del amplificador**, del orden de 2 a 10 µV rms según el equipo. Son números de hoja de datos, sin verificar en fuente primaria.

### Propuesta, de lo más barato a lo más caro

1. **Ruido de observación $1/f^\chi$ con SNR definida dentro de banda.** No cambia las trayectorias, así que el $R^2$ contra $\Delta f$ sigue valiendo, y se puede sumar a los datasets que ya existen. Lo importante es fijar el nivel por SNR dentro de la banda de la señal, medida igual que en `snr_real_vs_sim.py`, y no por $\sigma$ por muestra. Barrer la SNR hasta llegar a la de los datos reales. Ya está escrito en `experimental/lfp/ruido.py`.
2. **Ruido sobre $S$ y no sobre $I$ y $E$.** Se combina con el tema 4, y es lo que pasa en un registro.
3. **Ruido de proceso**, OU en la entrada o Langevin. Hay que regenerar las trayectorias con integración estocástica y redefinir $\Delta f$ para que el ruido no cuente como término faltante. El F7 ya midió el piso: con ruido de proceso, el oráculo da $R^2$ de 0,12.

**Un problema de escalas.** El simulador vive en ms con $\tau_e$ = 1 ms, y los datos reales tienen una banda de 0,5 a 19 Hz con $\tau$ del orden de 20 a 40 ms. Un 50 Hz o un exponente medido en Hz no se traslada directo. Conviene definir el ruido respecto de la banda de la señal: la SNR dentro de banda y la forma del espectro relativa a esa banda.

### Para decidir

- [ ] ¿Pasamos a fijar el ruido por SNR dentro de banda en vez de $\sigma$ por muestra? Cambiaría cómo se reportan todos los resultados con ruido.
- [ ] ¿Qué rango de $\chi$ y de SNR barremos? ¿Alguien del equipo tiene una estimación de la SNR de los registros reales mejor que la cota de −7 dB?
- [ ] ¿El ruido de proceso entra ahora o queda para cuando haya datos reales?

---

## 4. Observar S, el potencial de campo, en vez de I y E (piloto, por verse)

### La idea

El mismo modelo, pero en vez de ver $I$ y $E$ por separado ve una sola señal $S = h(I, E, P, Q)$, que es lo que se mide en un registro. Todo lo demás queda igual: variantes, arranque ignorante, selección por validación.

### Qué expresión usar para S

No hay una sola. La literatura se divide en dos hipótesis físicas que difieren en el **signo de la inhibición**:

| proxy | fórmula | hipótesis | fuente |
|---|---|---|---|
| diferencia | $E - I$ | potencial de membrana | decisión D3 de los datos reales |
| entrada neta a E | $w_{EE}E - w_{EI}I + P$ | despolarización de las piramidales | análogo de Jansen-Rit 1995 y del DCM (Moran et al. 2007) |
| suma de corrientes | $\lvert w_{EE}E + P\rvert + \lvert w_{EI}I\rvert$ | dipolos extracelulares del mismo signo | Mazzoni et al. 2008 |
| suma simple | $-(E + I)$ | lo mismo, con pesos iguales | Krishnakumaran, Raees y Ray 2022, sobre un Wilson-Cowan |

Mazzoni et al. 2015 proponen además $\text{AMPA}(t - 6\ \text{ms}) - 1{,}65\cdot\text{GABA}(t)$. En nuestro simulador ese retardo sería seis constantes de tiempo y más largo que una ventana de entrenamiento, así que no se traslada.

No encontré ningún trabajo que analice qué se pierde en identificabilidad al observar una sola combinación de $E$ e $I$. Es justamente lo que el piloto mide.

### Qué hay que resolver

- **Sin $I$ y $E$ observados, el estado al arranque de cada ventana no se conoce.** Pasa a ser una incógnita por ventana, con penalización de continuidad, como ya hace `train_real_output.py` con los datos reales.
- **Los proxies que usan pesos meten $\beta$ en la observación.** Un error en $w_{EE}$ o $w_{EI}$ cambia lo que se predice como $S$.
- **Con datos reales, la ganancia del proxy es otra incógnita**, y puede confundirse con los pesos.

### Lo que ya está armado

Una prueba piloto en `wilson-cowan-id/experimental/lfp/`, marcada en el README como en evaluación, con datos y resultados en carpetas aparte. Genera $S$ con cualquiera de los cuatro proxies y ruido $1/f^\chi$ sobre $S$, entrena con $S$ solamente y mide:
- el NRMSE de $S$;
- el error de parámetros;
- el $R^2$ contra $\Delta f$;
- si reconstruye $I$ y $E$ sin haberlos visto.

Por ahora soporta las variantes white-box, agnóstica, historia, filtro y forma exacta. Solo se probó que corre; no hay ninguna corrida completa.

### Para decidir

- [ ] ¿Se incluye en la línea principal, o queda como estudio aparte?
- [ ] ¿Qué hipótesis de $S$ tomamos como principal: diferencia con signo o suma de magnitudes? ¿Alguien del equipo sabe cuál corresponde mejor al registro que tenemos?
- [ ] ¿Se corre el piloto apenas se libere la máquina, con white-box y forma exacta en refractariedad y los cuatro proxies?

---

## 5. Cómo separar el backbone de la perturbación sin conocer la forma de la perturbación

### El problema

El white-box ajusta β para reproducir datos que generó el backbone **más** la perturbación. Toda parte de la perturbación que se pueda imitar moviendo β, el white-box la absorbe. Después el warm start arranca de ese β ya deformado, y la red solo tapa lo que β no pudo imitar. Eso explica lo del tema 2: el warm start identifica en refractariedad, donde el white-box queda cerca del β verdadero, y ata β al error en el actuador, donde queda deformado.

La parte imitable está medida (`exp_a_set_design.py`): con la librería de estímulos, **el 67 % de $\Delta f$ se puede imitar con un único cambio de β**.

### Sin ningún supuesto, no se puede

Si el modelo con $(\beta, g)$ reproduce los datos, el modelo con $(\beta + \delta,\ g - S\delta)$ los reproduce igual, donde $S$ son las sensibilidades del backbone respecto de β. Sobre esos datos son indistinguibles. Hace falta **algún** supuesto. Hoy hay uno implícito: que β explique todo lo que pueda.

Además, cualquier supuesto sobre la forma de la perturbación queda descartado, porque en datos reales no se conoce. Elegir el régimen donde la perturbación es chica tampoco sirve, porque supone saber cómo es. Las perturbaciones que sí se conocen se escriben en las ecuaciones, como venimos haciendo. La red queda para lo que no se conoce, y eso tiene el mismo problema.

### Supuestos que no tocan la forma de la perturbación

1. **Supuestos sobre β.** Valores de la literatura o de mediciones independientes (constantes de tiempo medidas aparte, rangos de conectividad), como una penalización $\lVert \beta - \beta_0 \rVert^2$. No dicen nada sobre qué falta en el modelo.
2. **Condiciones donde β no cambia.** Si el mismo circuito se registra en condiciones donde lo no modelado puede variar (otra intensidad de luz, otra sesión, otra potencia), β queda como lo común y la red se lleva lo que varía. Hace falta saber qué se mantiene constante, que es un supuesto sobre el experimento y no sobre el mecanismo.
3. **Intervenciones que cambian β de forma conocida.** Por ejemplo, un fármaco que baja la inhibición. Si β cambia de una manera que se conoce y lo demás no, eso también separa.
4. **Consistencia de β entre subconjuntos.** Ajustar el white-box por separado en cada familia de estímulos:
	- si el backbone alcanzara, todas darían el mismo β;
	- si dan β distintos, algo no modelado actúa distinto según el estímulo, y la dispersión mide cuánto contamina a β, sin saber qué es;
	- se puede usar además para darle más peso a las familias donde β es estable.
5. **Medir la ambigüedad en vez de resolverla.** Reportar el conjunto de β que reproducen igual de bien cuando la red se reajusta. Dice qué parámetros son identificables con una corrección agnóstica y cuáles no, sin prometer más de lo que los datos dan. Ya estaba en el plan como «Eje 2: medir la ambigüedad entre β y $g$».

### El mensaje honesto

Con una corrección agnóstica, β se identifica solo en la medida en que la perturbación no se pueda imitar moviéndolo. Esa medida se puede estimar, y el diseño del experimento la puede mejorar.

### Qué se puede hacer ya, en simulación

- La consistencia de β entre las siete familias del dataset ampliado (punto 4): white-box cortos, 7 por planta.
- El conjunto de β compatibles para el actuador (punto 5), que es donde el problema se ve.

### Para decidir

- [ ] ¿Qué de esto se prueba primero en simulación?
- [ ] **Para el equipo experimental:** ¿se pueden registrar condiciones distintas del mismo animal o preparado, como intensidades de luz, potencias, sesiones o fármacos (puntos 2 y 3)? Esto define si la separación es posible con datos reales, y conviene planificarlo antes de registrar.
- [ ] ¿Hay mediciones independientes de algún parámetro de β que se puedan usar como valor previo (punto 1)?
- [ ] ¿Cómo se reporta β en el trabajo: un valor, o un conjunto de valores compatibles con los datos?
