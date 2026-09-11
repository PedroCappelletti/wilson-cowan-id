# Próximos pasos desde el 10/9/2026

## Qué disparó esta revisión

Preparando el póster se verificó en el código que en las dos correcciones que
funcionan la red está **apagada**:

```python
"S": dict(use_correction=False, correction_inputs="x", structured=True)   # forma exacta
self.core = GrayBoxWC(..., use_correction=False)                          # estado de filtro
```

Los checkpoints lo confirman: `e2_B_v2` tiene 6 tensores de red y 1228
parámetros, `e2_lag2_v2` tiene 0 y 12. Las tres barras que compara la figura de
resultados no son variantes de una misma red. Una es una red y las otras dos son
física escrita a mano.

Eso deja una objeción de fondo. Si lo único que reproduce la dinámica es
codificar la forma analítica del término faltante, hubo que saber la respuesta de
antemano, y sobre datos reales no se la sabe. La sección 22 del manual ya lo
había anotado como advertencia:

> un gray-box estructurado que ajusta bien **no garantiza** que los parámetros
> físicos que reporta sean los correctos, si la forma propuesta es incompleta. Y
> en datos reales siempre lo es.

Así que el orden de prioridades cambia. **Hacer funcionar `g` pasa a ser el eje
principal.** La rama estructural queda como referencia, porque marca el techo
alcanzable, y como la vía para el control, porque es la única que el IMC puede
cancelar de forma exacta y explícita.

## Lo que ya está medido y no hay que volver a probar

| resultado | dónde | número |
|---|---|---|
| $\Delta f$ verdadero no es función del estado observado | F5 | techo del oráculo state-only, $R^2 = -0{,}11$ |
| regularizar `g` está agotado | manual, 15.2 | suave ayuda, fuerte perjudica, ninguna la vuelve física |
| memoria genérica no sirve acá | `e2_lat_w100` | 14,92 % NRMSE, $R^2 = -0{,}51$ |
| idem con ventana larga | `e2_lat_w400` | 13,28 % NRMSE, $R^2 = -3{,}55$ |

De la primera fila se sigue lo más importante para el eje 1: **más capas o más
unidades sobre `g(I,E)` no pueden pasar ese techo.** Es una limitación de
información, no de capacidad ni de optimización. Cualquier plan que empiece por
agrandar la red empieza mal.

## El dato que ordena todo: la planta es un sistema de Wiener

`src/wilson_cowan/uncertainty.py:169` define el actuador como un filtro lineal
seguido de una no linealidad estática:

```
dP_lag/dt = (P − P_lag) / τ        τ = 1 ms
P_eff     = A · tanh(P_lag / A)    A = 3
```

El docstring ya dice qué parte es aprendible y cuál no:

> Aprendible por g_phi: PARCIAL. La saturación sí (es función de P). El lag NO,
> porque su estado es una variable independiente que g_phi no ve.

La primera línea es una convolución: la salida de un filtro de primer orden es
el comando convolucionado con un núcleo exponencial. Y `g` no ve el estado del
filtro, pero **sí puede ver el comando entero**, que es una señal conocida y
exógena. Ahí está la salida sin escribir la física a mano.

Con $dt = 0{,}05$ ms, un milisegundo son 20 muestras y los 200 ms de cada
trayectoria son 4000. El campo receptivo tiene que cubrir la memoria más lenta
que pueda haber en los datos, y se fija por arriba, no a partir de $\tau$.

---

## La regla: a `g` no se le da física, se le da la chance de aprenderla

Vale la pena precisar qué dice el teorema de aproximación universal, porque es
lo que justifica el eje 1 y a la vez lo que explica por qué `g(I,E)` fracasó.

El teorema garantiza que una red con una capa oculta y suficientes neuronas
aproxima, con la precisión que se quiera, cualquier función continua **de sus
entradas** sobre un dominio acotado. No dice nada sobre aproximar algo que no
es función de sus entradas.

Y ese es exactamente el caso acá. $\Delta f$ depende de $P_{lag}$, que es un
estado con dinámica propia: para un mismo $(I,E)$ el $\Delta f$ verdadero toma
valores distintos según la historia del comando. **No es una función de
$(I,E)$**, así que no hay cantidad de neuronas que alcance. El techo medido de
$R^2 = -0{,}11$ es eso y no un problema de capacidad.

De ahí se sigue lo que ordena el eje 1: **lo que hay que cambiar es el conjunto
de entradas, no el tamaño de la red.** Con la historia del comando adentro,
$\Delta f$ sí pasa a ser una función continua de las entradas, y recién ahí el
teorema aplica y predice que se puede aprender.

### Qué cuenta como darle física y qué no

| lo que recibe `g` | ¿es física? |
|---|---|
| $(I,E)$, el estado observado | no, es la señal medida |
| $P,Q$ y su historia | no, es el estímulo que uno mismo comanda |
| que el núcleo es exponencial | **sí** |
| el valor de $\tau$ | **sí** |
| la forma $\tanh$ de la saturación | **sí** |
| $K$, el largo del campo receptivo | es un límite de capacidad, ver abajo |

Las tres del medio no entran en ningún experimento del eje 1. El núcleo de la
convolución arranca libre: si converge a una exponencial, eso es un resultado y
no un supuesto.

**Corrección sobre $K$.** La primera versión de este plan fijaba $K = 100$
muestras «derivado de $5\tau$». Eso es darle física, aunque sea poca: usa el
valor de $\tau$, que es justo lo que la red tendría que descubrir. Se
reemplaza por un $K$ generoso y agnóstico, **400 muestras (20 ms)**, muy por
encima de cualquier constante de tiempo plausible del dataset, y se deja que el
núcleo aprendido decaiga solo. El largo efectivo pasa a ser una medición en vez
de un supuesto.

### El resultado que hay que buscar

Graficar el núcleo aprendido y compararlo con $e^{-t/\tau}$. Si la red recupera
la exponencial y un $\tau$ cerca de 1 ms sin que se le haya dicho nada, el
titular deja de ser «el gray-box reproduce la dinámica» y pasa a ser **«la
corrección descubrió la física omitida»**. Esa es la diferencia entre el eje 1 y
el eje 3, y es medible.

---

## Eje 1 — Darle a `g` la historia del comando (prioridad alta)

**Memoria en la entrada, no estado recurrente.** Hay dos formas de darle
memoria a un modelo y acá solo sirve una. La primera es un estado interno propio
que el modelo infiere y propaga: eso es la latent ODE, ya medida en
$R^2 = -0{,}51$ y la corrida más cara de todas. La segunda es ensanchar la
entrada con la historia del comando, sin ningún estado interno. Va la segunda,
por tres motivos.

El comando **se conoce exactamente**, porque es lo que uno manda. $P_{lag}$ es
un funcional determinista de esa historia, así que la historia es un estadístico
suficiente para el estado oculto: no hay nada que inferir, solo que leer.

Sin estado interno **desaparece el problema de inicialización**, que es un modo
de falla ya documentado acá. El primer intento con estado de filtro, `e2_lag`,
dio 12,16 % porque arrancar cada ventana con $\hat P = P(t_0)$ metía un
transitorio falso que sesgaba $\hat\tau$ a 0,56 contra 1,0 real. Se arregló
recorriendo el filtro sobre el comando entero (`filtered_inputs`), y ahí
`e2_lag2` bajó a 2,96 %. Una ventana de comandos pasados no tiene ese problema
de entrada, porque no hay nada que inicializar.

Y sigue siendo feedforward, así que el entrenamiento se mantiene barato y
estable. En jerga de identificación, es pasar de un mapa que solo ve el estado a
uno de **memoria finita sobre el comando**, tipo NARX o FIR.

**El límite.** Esto alcanza porque la dinámica que falta está manejada por una
entrada conocida. Si el estado oculto tuviera dinámica autónoma, o lo manejara
algo que no observamos, la ventana no alcanzaría y ahí sí haría falta estado
recurrente. Para la adaptación, por ejemplo, el estado lo maneja $E$, que sí se
observa, así que una ventana sobre la historia de $(I,E)$ también serviría.

La apuesta principal. Cuatro escalones, en orden, y no se pasa al siguiente sin
medir el anterior.

### 1.1 `g(I,E,P,Q)`, el comando instantáneo

Ya está implementado: es la variante `A`, `correction_inputs="xpq"`. **Nunca se
corrió en el protocolo de escalado**, solo en los experimentos viejos sobre la
planta combinada, que además no registran NRMSE ni $R^2$, solo error de
parámetros.

Sigue siendo sin memoria, así que no puede representar el lag. Se corre igual
porque separa dos preguntas que hoy están mezcladas: cuánto de la mejora viene
de ver el comando y cuánto de tener memoria. Y porque la saturación **sí** es
función de $P$, así que debería capturarla.

Predicción: mejora sobre `e2_B` (15,40 %) pero no llega a resolver el lag. Si
empata con B, el comando instantáneo no aporta nada y se salta al 1.2.

### 1.2 El mismo MLP, con la historia del comando en la entrada

El test limpio del principio. No cambia la arquitectura: es el mismo perceptrón
de siempre, con más entradas.

```
entrada = [ I, E, P(t-K..t), Q(t-K..t) ]  ->  MLP  ->  Δf
```

Con la historia adentro, $\Delta f$ **es** una función continua de la entrada,
así que el teorema de aproximación universal aplica sin asteriscos y predice que
una red suficientemente ancha la puede representar. Si esto falla, no falla el
teorema: falla la optimización o el diseño del experimento, y hay que mirar ahí
antes que la arquitectura.

Con $K = 400$ y dos canales son 802 entradas, así que la primera capa crece a
unos 26 mil pesos. Es la versión cara y sin ninguna estructura impuesta, y por
eso mismo es la que mide el principio en su forma pura.

### 1.3 Convolución, la misma información con menos parámetros

```
P,Q (últimas K muestras) -> Conv1d (núcleo libre) -> no linealidad -> concat con (I,E) -> MLP -> Δf
```

Misma información que 1.2 y mucho menos peso, porque la convolución comparte el
núcleo a lo largo del tiempo en vez de aprender un peso por retardo. Es un
supuesto de invarianza temporal, no un supuesto de física: no se le dice qué
forma tiene el núcleo.

Es también el que permite el resultado interesante, porque el núcleo aprendido
se puede graficar y comparar con la exponencial.

Criterio de éxito para 1.2 y 1.3, sobre `act1`: **$R^2$ de la corrección
positivo** y NRMSE por debajo del white-box (15,23 %).

Banco de prueba para el campo receptivo: la perturbación `Adaptation` tiene una
perilla continua entre capturable y no capturable, ya medida con `g(I,E,P,Q)`:

| $\tau_a$ | $R^2$ |
|---|---|
| 1 ms | 0,97 |
| 30 ms | 0,54 |
| 100 ms | 0,33 |

Si el campo receptivo está bien dimensionado, esa caída se tiene que aplanar al
crecer $K$. Es la validación más limpia y no necesita tocar el actuador.

### 1.4 Capacidad, y recién acá

Más capas, más unidades, otra activación. Solo si 1.2 y 1.3 mejoraron
parcialmente y hay motivo para creer que falta capacidad y no información.
Nunca antes: sobre `g(I,E)` está probado que no sirve, y el teorema explica por
qué no podía servir.

## Eje 2 — Medir la ambigüedad entre $\beta$ y `g` (barato y falsable)

Correr **forma exacta + red** sobre `refrac1`, que es la planta donde la forma
asumida **sí está completa**.

Implementación: una entrada más en `VARIANTS` y una opción más en el argparse de
`esc_run.py`. Los dos flags ya son independientes en `GrayBoxWC` (`structured`
crea $r_i, r_e, \alpha$; `use_correction` crea `self.g`; el `forward` suma las
dos cosas). Unas cinco líneas.

Costo: `e1_S2` tardó 13 min.

Qué mide cada resultado:

- Si $\lVert g \rVert \to 0$ y el $R^2$ se queda en 0,96, el grey-box **sabe
  cuándo parar**: con la física completa la red no inventa. Es un argumento
  fuerte para datos reales.
- Si el $R^2$ cae y los parámetros físicos derivan, es la primera **medición**
  de la ambigüedad entre $\beta$ y `g`, que hoy el complemento afirma pero no
  mide, apoyándose solo en la tabla de pérdidas de la sección 22.

Las dos salidas son publicables. Por eso va antes que lo caro.

---

## Eje 3 — Cerrar la rama estructural: forma exacta + retardo

Es la opción B+C del roadmap del manual, sin cambios: cuatro parámetros físicos
($r_i$, $r_e$, sat, $\tau$), sin red. El manual la llama «el camino que más se
parece a lo que el proyecto quiere de punta a punta», y la razón es el control:
conserva la cancelación exacta y explícita en el IMC.

No es el camino generalizable, y conviene decirlo así en cualquier informe. Es
la referencia contra la que se mide el eje 1: si la `g` convolucional se le
acerca, el resultado interesante es justamente que no hizo falta escribir la
física.

---

## Eje 4 — El caso combinado

Sigue abierto, como dice el abstract. Los números a superar están en la planta
completa: white-box 14,0 % y gray-box 13,1 %, los dos escritos a mano en
`scripts/figuras_escalado.py` y **sin JSON que los respalde**. Antes de usarlos
como línea de base hay que regenerarlos, porque hoy no son reproducibles.

Se corre después de que el eje 1 o el 3 funcionen sobre las plantas aisladas, no
antes: el protocolo de escalado existe precisamente porque la planta combinada
enmascaraba las causas.

---

## Eje 5 — Los datos reales, y qué se puede medir con ellos

Los datos que nos pasaron **están en el repo**, en
`data/processed/real/data8_fs{125,250}.npz`. Son tres estimulaciones chirp de 1 a
10 Hz con sus respuestas, de 16 a 20 s cada una, filtradas entre 0,5 y 19 Hz,
originales a 1250 Hz y remuestreadas a 125 o 250.

Traen **`u` y `s` nada más**: el estímulo y el LFP. No hay estado $(I,E)$. Eso
no es un detalle de formato, cambia qué preguntas se pueden hacer.

### Qué métrica sobrevive y cuál no

**El $R^2$ de la corrección no sobrevive.** Compara $\hat g$ contra el
$\Delta f$ verdadero, y el $\Delta f$ verdadero solo existe porque el simulador
lo genera. Sobre datos reales no hay planta conocida contra la cual restar. La
métrica que responde «¿aprendió el mecanismo?», que es la pregunta central del
proyecto, **no se puede evaluar fuera del simulador**. Conviene decirlo así en
cualquier informe, porque es una limitación del experimento y no del método.

**El NRMSE de corrida libre sobrevive pero es inútil acá**, y eso ya está
medido. La tanda de julio ajustó un ARX lineal $u \to s$ y encontró el techo:

| modo | $R^2$ |
|---|---|
| predicción a un paso, con el pasado real de `s` | 0,99 a 1,00 |
| **simulación libre, solo con `u`** | **0,04 a 0,11** |

`s` está dominada por su propia dinámica recurrente, no por el estímulo, que
explica alrededor del 10 % de la varianza en simulación libre. Ordenar variantes
por corrida libre sobre estos datos es ordenarlas adentro del ruido.

**Lo que sí transfiere es la predicción a horizonte corto.** Con ventanas de
unos 128 ms el WC llega a $R^2 \approx 0{,}85$. Ese es el marco donde la
dinámica es identificable y donde una corrección con memoria puede mostrar que
mejora algo.

### Qué variante se puede probar y cuál no

**El eje 1 tal cual no se puede validar acá.** El retardo del actuador tiene
$\tau = 1$ ms, y estos datos están filtrados a 19 Hz y muestreados a 4 u 8 ms.
El fenómeno que la `g` convolucional aprende en `act1` es literalmente invisible
en esta banda. Si se corre igual, el núcleo aprendido va a estar midiendo otra
memoria, más lenta, no la del actuador.

Lo que sí se puede probar es **la misma arquitectura con otra pregunta**: si
darle a `g` la historia del estímulo mejora la predicción a horizonte corto
sobre `s`. Es un resultado legítimo, pero es una afirmación distinta de la del
eje 1 y no hay que mezclarlas.

### Lo que hay que resolver para correr cualquier cosa

Sin estado, el modelo de observación deja de ser opcional. Hay que ajustar
$s = c_{\text{out}}(E - I)$ con $c_{\text{out}}$ como incógnita más, y decidir
qué hacer con el filtrado de 0,5 a 19 Hz, que le sacó a la señal la componente
de continua sobre la que están definidos los offsets $k_e, k_i$ del backbone.

Las escalas ya están resueltas, aunque a mano. `train_real_output.py` trabaja en
segundos y fija $\tau_e = 20$ ms y $\tau_i = 40$ ms, contra 1 y 2 ms del
simulador. Es el orden correcto para una banda de 0,5 a 19 Hz. Lo que queda
abierto no es la escala sino si esos valores son identificables desde `s`, o si
hay que dejarlos fijos y ajustar solo el resto.

Vale la pena tenerlo presente al comparar: **los parámetros ajustados sobre
datos reales no son comparables numéricamente con los del escalado**, porque el
régimen temporal es otro. Lo comparable son las métricas, no los diez números.

### Protocolo

Tres grabaciones, todas chirp, así que no hay familias de estímulo para separar
como en el escalado. Lo único honesto es **dejar una afuera**: entrenar con dos
y evaluar en la tercera, rotando las tres. Con $n = 3$ el resultado es
indicativo, no concluyente, y así hay que reportarlo.

---

## Lo que parecía un bloqueante y no lo es

Las tres corridas ciegas sobre `refrac1` informan unas 25 horas cada una. **No
es el costo de esas corridas.** `scripts/run_escalado_all.sh` lanza las doce en
**cuatro carriles paralelos**, y el campo `minutos` sale de `time.time()`, o sea
reloj de pared, así que incluye la contención con los otros carriles.

La prueba está en los propios números. Las tres informan 1486,8, 1556,3 y
1575,7 minutos, dentro del 6 % entre sí, **pese a tener ventanas de 100, 200 y
400**, es decir cuatro veces distinta cantidad de trabajo. Cuando una corrida sí
está sola la ventana pesa muchísimo: la latent con ventana 100 tardó 1013,7 min
y con ventana 400, 63,9. Tres tiempos casi iguales para tres cargas distintas es
la firma de tres procesos compartiendo máquina, no de tres cómputos distintos.

Sumando por carril, el batch entero tardó **27,4 horas de reloj**, no 98:

| carril | corridas | horas |
|---|---|---|
| a | `e1_S` + `e2_lat_w100` | 19,2 |
| b | `e1_B_w100` + `e2_wb` | 25,3 |
| c | `e1_B_w200` + `e2_B` + `e2_lat_w400` | 27,4 |
| d | `e1_B_w400` + `e2_lag` | 26,6 |

Consecuencia práctica: **de esa tanda no sale ninguna medida de costo por
corrida utilizable**, ni para arriba ni para abajo. Las únicas limpias son las
dos del 8/9, corridas solas: `e2_B_v2` 41,7 min y `e2_lag2_v2` 61,1 min. Y como
`e2_B` había tardado 21,9 min en la máquina original, esta es alrededor de 1,9×
más lenta para el mismo trabajo.

Eso reordena el plan para bien: el barrido del eje 1 es perfectamente
afordable. Antes de lanzarlo conviene cronometrar una corrida sola de la
variante B sobre cada dataset, para tener una referencia real.

---

## Orden sugerido

1. Cronometrar una corrida sola de la variante B sobre `refrac1` y sobre `act1`,
   para tener una referencia de costo real. Es una tarde, no un bloqueante.
2. Eje 2, forma exacta + red sobre `refrac1`. 13 min, resultado falsable, cierra
   una afirmación que hoy está sin medir.
3. Eje 1.1, la variante A, que ya está escrita.
4. Eje 1.2, el MLP con la historia del comando en la entrada. Es el test limpio
   del principio y la apuesta principal.
5. Eje 1.3, la versión convolucional, si 1.2 funciona pero sale cara o ruidosa.
   Es la que permite graficar el núcleo aprendido.
6. Eje 3, la rama estructural completa.
7. Eje 4, la planta combinada, regenerando antes sus líneas de base.

## Lo que no haría

- Agrandar `g(I,E)`. Tiene un techo medido de $R^2 = -0{,}11$ y no se rompe con
  capacidad.
- Volver a la regularización de `g`. El barrido la agotó.
- Latent ODE genérica. Dio $R^2 = -0{,}51$ y es la más cara de todas.
