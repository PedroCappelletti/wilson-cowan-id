---
title: "Avances: actuador con recorte, x0 entrenable y referencia black-box"
---

Este documento resume el trabajo posterior a los informes de historia pasada del comando y de sesgo por ventanas. Cubre cuatro frentes. El actuador pasó de una saturación suave a un recorte duro. Se midió qué parte del error de $\hat\beta$ viene del ruido en las condiciones iniciales y qué parte de la estructura del modelo. El lazo cerrado se evaluó por la distancia al lazo de la planta y no solo por el error de seguimiento. Se agregó una referencia black-box.

Salvo aclaración, cada valor es la media de tres semillas. El NRMSE es el de los 13 escenarios de test que interpolan, en %. El error de $\hat\beta$ es el error relativo medio de los diez parámetros, en %. σ es el desvío del ruido de medición.

# 1. Cambios de diseño

**Dos líneas de perturbación.** La línea de diseño es la que se venía usando, con refractariedad $r = 0{,}10$ y un actuador simétrico de 1 ms con nivel 3. Sus valores se eligieron por dificultad, para poner a prueba las arquitecturas. Es optimista frente a ChR2, cuyo $\tau_{off}$ va de 11 a 14 ms. La línea realista usa los valores de la bibliografía, con filtro asimétrico y un barrido de τ. Quedó postergada por tiempo. Su cola se cortó con 16 de 50 corridas (e18), que quedan como exploración y no se reportan. En el paper va como trabajo futuro.

**Recorte duro en el actuador.** La saturación $3\tanh(P_f/3)$ comprime aun por debajo del nivel. Un comando de 2 sale como 1,75 y uno de 3 como 2,28. Esa compresión agrega una no linealidad que no se quiere en la planta. Se reemplazó por

$$P_{ef} = \min\left(3,\ \max(-3,\ P_f)\right), \qquad \tau\,\dot P_f = P - P_f, \qquad \tau = 1\ \text{ms}.$$

El dataset nuevo es `actc1_2s`, con las mismas semillas y niveles de ruido que `act1_2s`. La línea con tanh (e16 a e20) queda como comparación. Los comandos de entrenamiento están en $[0, 2]$, así que el recorte no actúa nunca durante el entrenamiento.

**Historia pasada solo como planta.** La variante Hp recibe los retardos del comando sin el comando actual, y así el modelo queda afín en $u_k$. En este paper se evalúa solo como planta. Usarla dentro del controlador queda para trabajo futuro, porque el paper no modifica el controlador.

# 2. Lazo abierto con recorte (e21)

| modelo | σ = 0,01 NRMSE / $\hat\beta$ / R² | σ = 0,05 NRMSE / $\hat\beta$ / R² |
|---|---|---|
| white-box | 7,28 / 20,1 / − | 7,90 / 16,6 / − |
| agnóstica, warm start | 7,34 / 17,6 / −0,02 | 7,90 / 22,1 / −0,12 |
| historia 5 ms, warm start | 4,07 / 18,6 / 0,17 | 6,24 / 27,4 / 0,27 |
| filtro 8 canales, warm start | 4,23 / 18,9 / 0,13 | 6,72 / 27,6 / 0,15 |
| historia pasada, warm start | 4,06 / 28,8 / −0,73 | 6,30 / 34,2 / −0,77 |
| estado de filtro, desde cero | 2,26 / 0,4 / 1,00 | 5,87 / 12,2 / 0,98 |
| estado de filtro, warm start | 2,25 / 0,5 / 1,00 | 5,86 / 10,3 / 0,99 |

R² se mide contra el término faltante exacto. Con el recorte, el estado de filtro recupera β casi exacto con σ = 0,01, y $\hat\tau$ = 0,997 ms. Con la tanh, la compresión terminaba absorbida en wEE, wEI y ae, y el error de $\hat\beta$ quedaba entre 7 y 9 %. Las correcciones con red quedan como con tanh. El nivel del recorte no recibe gradiente porque los datos nunca llegan a 3, y queda en su valor inicial de 10.

# 3. Capacidad de cada red con β verdadero fijo (e20)

Con β fijo en el valor verdadero se entrena solo la red. El R² contra el término exacto mide cuánto del término faltante puede representar cada arquitectura.

| red | σ = 0,01 | σ = 0,05 |
|---|---|---|
| forma refractaria, agnóstica B100 | 0,95 | 0,89 |
| historia 5 ms | 0,84 | 0,79 |
| filtro entrenable | 0,74 | 0,69 |
| historia pasada | 0,19 | 0,11 |
| agnóstica en el actuador | −0,20 | −0,19 |

Sin el comando actual, la red no reconstruye el estado del filtro. Con β libre la historia pasada parecía equivalente a la de 5 ms, porque $\hat\beta$ absorbía parte del término.

# 4. x0 entrenable

Con ventanas que arrancan del estado medido, el ruido entra como condición inicial exacta y sesga $\hat\beta$. La segunda etapa arranca del modelo ya entrenado y libera el estado inicial de cada ventana, con un término de continuidad entre ventanas consecutivas.

**Refractariedad y actuador con tanh (e19).** En refractariedad con σ = 0,05, la forma exacta baja de 7,6 a 3,6 % de error de $\hat\beta$ y de 6,91 a 5,50 % de NRMSE. La agnóstica baja de 3,5 a 3,3 % con σ = 0,01 y de 6,9 a 5,6 % con σ = 0,05. En el actuador con tanh, las redes quedan entre 24 y 35 %. Las ventanas de 20 ms no mejoran a las de 5 ms y se descartaron.

**Recorte, correcciones con red (e22).** Error de $\hat\beta$, de e21 a e22.

| modelo | σ = 0,01 | σ = 0,05 |
|---|---|---|
| historia 5 ms | 18,6 → 18,6 | 27,4 → 23,3 |
| historia pasada | 28,8 → 28,3 | 34,2 → 30,2 |
| filtro 8 canales | 18,9 → 19,7 | 27,6 → 28,3 |
| white-box | 20,1 → 20,1 | 16,6 → 16,6 |

El NRMSE de las redes baja entre 0,1 y 0,35 puntos. En la selección por validación compite también el modelo de partida. En el white-box ninguna época de la segunda etapa valida mejor que el punto de partida, así que queda igual. Sin esa regla se había reportado un empeoramiento de 20 a 35 %, que era un artefacto de la selección. Los x0 libres bajan la pérdida de entrenamiento absorbiendo el desajuste del modelo, y la validación empeora.

**Recorte, estado de filtro (e23).** Se liberan solo los x0 de E e I. El estado del filtro de cada ventana sale de filtrar el comando entero con $\hat\tau$, que se conoce sin ruido.

| | σ = 0,01 | σ = 0,05 |
|---|---|---|
| error de $\hat\beta$ | 0,46 → 0,16 | 10,3 → 1,4 |
| NRMSE | 2,25 → 2,22 | 5,86 → 5,40 |
| $\hat\tau$ (ms) | 0,99 | 0,98 |

Con σ = 0,05 el error baja en las tres semillas, de 7,5, 10,3 y 13,0 % a 1,7, 1,1 y 1,4 %. Los dos resultados juntos separan las fuentes del sesgo. En el estado de filtro, el sesgo con ruido alto venía de las condiciones iniciales. En las correcciones con red es estructural, porque les falta el estado del filtro.

# 5. Lazo cerrado

Cada modelo identificado reemplaza a la planta bajo el controlador IMC del paper, sin modificarlo. Además del RMSE de seguimiento se mide la distancia entre el lazo con el modelo y el lazo con el simulador, en E e I, sobre el último 80 % de la corrida. Los valores son medianas de tres semillas, para σ = 0,01 / 0,05.

| planta en el lazo | RMSE theta-gamma | distancia theta-gamma | distancia gamma 50 ms |
|---|---|---|---|
| simulador | 0,057 | − | − |
| white-box | 0,045 / 0,043 | 0,034 / 0,035 | 0,061 / 0,062 |
| agnóstica | 0,044 / 0,042 | 0,034 / 0,035 | 0,061 / 0,062 |
| historia con comando actual | 0,059 / 0,063 | 0,049 / 0,054 | 0,068 / 0,066 |
| historia sin comando actual | 0,070 / 0,079 | 0,057 / 0,066 | 0,076 / 0,070 |
| filtro 8 canales | 0,077 / 0,078 | 0,065 / 0,071 | 0,067 / 0,067 |
| estado de filtro, nivel entrenado (10) | 0,056 / 0,056 | 0,035 / 0,038 | 0,062 / 0,087 |
| estado de filtro, nivel fijo en 3 | 0,057 / 0,058 | 0,010 / 0,033 | 0,007 / 0,059 |

**El RMSE de seguimiento no ordena modelos.** El white-box y la agnóstica siguen a la referencia mejor que el simulador, y sus lazos quedan a 0,034 del lazo con el simulador, igual que el estado de filtro con el nivel entrenado. Las correcciones con historia del comando siguen casi como el simulador y quedan más lejos, entre 0,049 y 0,065.

**Corrección de la simulación.** En una versión anterior las correcciones con historia aparecían trabadas en theta-gamma, con RMSE de 0,137 y 0,172. Era un error de la simulación del lazo. El tiempo se acumula sumando el paso, y desde los 1292 ms el código dejaba de registrar el comando aplicado, así que la red veía una historia en cero. Corregido el registro, ninguna se traba. Las referencias más cortas que 1292 ms no estaban afectadas.

**Nivel de saturación.** Con theta-gamma el controlador pide comandos de hasta ±11. El nivel del recorte no se identifica con comandos de entrenamiento en $[0, 2]$. Con el nivel fijo en su valor de diseño y sin reentrenar, el estado de filtro reproduce el lazo de la planta con σ = 0,01. Con σ = 0,05 mejora menos, por el error de $\hat\beta$. En la línea con tanh fijar el nivel no ayuda, porque las ganancias ya absorbieron la compresión.

**Referencias en rango.** Con theta de 8 Hz y gamma de 40 Hz, construidas para que el comando quede dentro del rango de entrenamiento, el estado de filtro reproduce el lazo del simulador con distancia menor a 0,001. El white-box y la agnóstica quedan a 0,001 a 0,003. Las redes con historia quedan entre 0,03 y 0,06.

**Planta sin saturación.** Se repitió el lazo con el simulador con filtro y sin recorte. Los modelos son los mismos e21, porque con comandos en $[0, 2]$ los datos de entrenamiento de las dos plantas coinciden. A la forma exacta se le quitó el recorte.

| | theta-gamma | gamma 50 ms |
|---|---|---|
| simulador con recorte, RMSE | 0,0570 | 0,0676 |
| simulador sin recorte, RMSE | 0,0560 | 0,0633 |
| estado de filtro sin recorte, distancia | 0,007 / 0,029 | 0,017 / 0,079 |
| historia con comando actual, distancia | 0,048 / 0,053 | 0,071 / 0,064 |

Con la forma exacta de e23 (x0 entrenable) la distancia con σ = 0,05 baja a 0,013 en theta-gamma y a 0,019 en gamma, con el nivel de diseño y sin saturación. Con $\hat\beta$ corregido, el lazo con el modelo coincide con el lazo con el simulador en los dos niveles de ruido.

La saturación cambia poco el seguimiento del simulador, como ya se había visto con el simulador solo. La forma exacta sin recorte reproduce el lazo sin saturación con σ = 0,01. Las redes quedan a la misma distancia que con recorte, así que su distancia no viene de la saturación.

# 6. Referencia black-box (e24)

La referencia black-box es una Neural ODE sin el modelo de Wilson-Cowan, $\dot x = g(I, E, P, Q)$. La variante BBH recibe además la historia del comando de la variante de 5 ms. Las dos usan una red de 64 unidades y 3000 épocas, el doble que las correcciones, porque aprenden el campo entero. Se entrenan desde cero, porque no hay backbone desde el cual arrancar. No tienen $\hat\beta$ ni R² contra el término faltante.

La tabla muestra el NRMSE de test, media de tres semillas. El white-box es la referencia sin corrección.

| planta, σ | BB | BBH | white-box | corrección con red | forma exacta |
|---|---|---|---|---|---|
| recorte, 0,01 | 6,50 | 3,47 | 7,28 | 4,07 | 2,25 |
| recorte, 0,05 | 7,34 | 6,19 | 7,90 | 6,24 | 5,86 |
| refractariedad, 0,01 | 2,88 | 3,20 | 3,29 | 2,13 | 2,14 |
| refractariedad, 0,05 | 5,85 | 5,99 | 5,91 | 5,66 | 5,70 |

En el actuador, la black-box con historia queda a la par de la corrección con historia de 5 ms, y con σ = 0,01 la supera. Las formas estructuradas siguen siendo las más precisas en las dos plantas. La ventaja del gray-box frente a la black-box no está en la precisión en lazo abierto de las correcciones con red. Está en los parámetros identificables y en las formas estructuradas. La black-box todavía no puede usarse como planta en el lazo cerrado.

# 7. Paper y repositorios

- La introducción del borrador se reescribió con la saturación dura. Las contribuciones incluyen la recuperación de β con 0,5 % de error, la falla de la historia sin comando actual, la distancia entre lazos y la reproducción del lazo en rango y con el nivel de diseño. La cuarta contribución ya refleja e22. Falta sumar e23 y la black-box, y recortar el texto, que ronda las 1200 palabras.
- El plan del paper tiene las tablas de e21, e20, e19, e22 y de distancia entre lazos.
- El repositorio limpio recibió el recorte, la distancia entre lazos, e19 sin las ventanas de 20 ms, e20 y los 86 checkpoints del lazo cerrado. e22, e23, la planta sin saturación y la black-box están solo en el repositorio de trabajo.

# 8. Pendientes

- Llevar la black-box al lazo cerrado.
- Declarar en Métodos la excepción al warm start de la black-box y que e19 corrió sobre la línea con tanh.
- Unificar la métrica de 13 y de 14 escenarios.
- Decidir el alcance de la planta con las dos perturbaciones juntas.
- Actualizar el documento del controlador con el recorte y la distancia entre lazos.
- Decidir qué migra al repositorio limpio.
