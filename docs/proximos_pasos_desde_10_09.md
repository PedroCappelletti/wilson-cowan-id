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
| el ruido de proceso es irreducible | F7 | $R^2$ del oráculo 0,12 con estado, 0,11 con comando |
| el escalado corre sin ruido de observación | `gen_uncertain_dataset.py` | `noise_std = 0.0` en los dos datasets |

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

### 1.2 y 1.3 medidos el 11-09: el eje 1 funciona

Corridas sobre `act1`, ventana 100, 1500 épocas, semilla 0:

| variante | entrada de `g` | NRMSE | $R^2$ | err. param |
|---|---|---|---|---|
| white-box | sin `g` | 15,23 | −0,006 | 30,7 |
| B | $(I,E)$ | 15,40 | −1,875 | 34,4 |
| A | $(I,E,P,Q)$ | 15,18 | −0,714 | 32,9 |
| H, $K=100$ | $(I,E)$ + 100 retardos | 12,92 | +0,250 | 26,3 |
| H, $K=400$ | $(I,E)$ + 400 retardos | **11,48** | +0,146 | 22,7 |
| forma estructural | sin `g`, 3 parámetros | 2,96 | 0,942 | 5,1 |

Las dos corridas con historia rompen el techo state-only de $R^2 = -0{,}11$ y
son las primeras correcciones genéricas que le ganan al white-box en NRMSE. El
error de parámetros baja monótonamente con el campo receptivo.

**La inversión entre las dos.** $K=400$ reproduce mejor y recupera menos física
que $K=100$. El plan predecía que el campo receptivo más grande ganaría en las
dos métricas. Es la firma de la ambigüedad entre $\beta$ y `g`: con más
capacidad hay más formas de tapar el término faltante sin aprenderlo. Eso
convierte al eje 2 en la medición que corresponde hacer ahora.

### La prioridad, fijada el 11-09: NRMSE

El objetivo del gray-box es reproducir la dinámica. El $R^2$ de la corrección
pasa a ser diagnóstico, no criterio de aceptación, y una variante que baje el
NRMSE con $R^2$ pobre se toma como avance. Eso reordena lo que sigue: los ejes
que atacan el NRMSE (1.3, 1.4) suben, y el eje 2 se corre por lo que explica,
no por lo que decide.

Criterio nuevo, sobre `act1`: **NRMSE por debajo de 11,48 %**, el mejor gray-box
medido, con el white-box (15,23 %) como piso de comparación y la forma
estructural (2,96 %) como referencia de cuánto falta. El $R^2$ se reporta
siempre y no bloquea nada.

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

Implementado el 11-09 como variante `K` (`correction_inputs="xconv"`): un FIR
de cuatro canales sobre los $2K$ retardos, seguido de la misma MLP. Con $K=400$
son 3200 pesos en el filtro contra los 26 mil de la primera capa densa de `H`.
El ancho del FIR se controla con `--n-fir`.

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

Más capas, más unidades, otra activación, y $K$ más grande. Habilitada: 1.2
mejoró y la mejora crece con el campo receptivo, así que ahora hay motivo para
creer que falta capacidad y no información. La condición sigue siendo que se
barra sobre `xhist` o `xconv`, nunca sobre `g(I,E)`, que tiene techo medido.

El barrido mínimo son tres ejes, uno por vez: $K \in \{400, 800\}$,
`hidden` $\in \{32, 64\}$ y `n_fir` $\in \{4, 8\}$.

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

## Eje 6 — El ruido, que hasta acá está apagado

Todo el escalado corre sobre datos limpios. Los dos datasets se generan con
`noise_std = 0.0` (`scripts/gen_uncertain_dataset.py:53`), a propósito: con
ruido encima no se puede saber si el residuo que `g` tiene que aprender es el
término omitido o la medición. Los 11,48 % de NRMSE de `H400` son sobre
trayectorias exactas.

El ruido se estudió antes, en una línea aparte y sobre el white-box de 10
parámetros. Barrido de ruido de observación gaussiano, con el error máximo de
parámetro en porcentaje:

| $\sigma$ | 10 parámetros | subconjunto de 4 |
|---|---|---|
| 0 | 0,8 | 0,1 |
| 0,01 | 4,5 | 0,6 |
| 0,05 | 19,3 | 1,2 |
| 0,10 | 41,3 | 8,9 |

Las dos columnas ya tienen el suavizado aplicado: una media móvil sobre $I$ y
$E$ antes de ajustar, de 7 muestras hasta $\sigma = 0{,}05$ y de 11 por encima.
Lo que separa las columnas es fijar los parámetros no identificables en lugar de
pelearlos, y el efecto es grande: el problema no era el ruido sino la
combinación de ruido y parámetros mal condicionados.

El método sin suavizar, sobre otro conjunto de escenarios y por eso no
directamente comparable, llegaba a 106 % de error máximo en $\sigma = 0{,}10$
(`scripts/noise_final.py:38`).

Hay un caso que no se mitiga y conviene tenerlo presente como piso. El ruido de
proceso, el que entra dentro de la ODE, da $R^2$ del oráculo de 0,12 con el
estado y 0,11 con estado más comando. No es función de nada, así que ninguna
corrección lo puede aprender, ni con historia ni con capacidad. Es el control
negativo del proyecto.

**Lo que hay que medir.** Repetir `H400` y `K` sobre `act1` con
$\sigma = 0{,}01$ y $\sigma = 0{,}05$. La hipótesis es que la historia cruda
sufre más que el FIR, porque 802 entradas con 26 mil pesos en la primera capa
es exactamente la configuración que ajusta ruido, mientras que el FIR promedia
$K$ muestras y eso es un filtro pasabajos por construcción. Si se confirma, el
argumento a favor de la convolución deja de ser solo el costo.

Va después del eje 1.4 y antes de los datos reales, porque los datos reales
tienen ruido y no se sabe cuánto.

## Para después del congreso (anotado el 28-09)

Hasta el viernes 2-10 el trabajo es validar con ruido lo que ya está y
actualizar el póster. Queda para después:

- **Cuántos canales hacen falta, sin barrerlos.** Hoy el ancho del filtro se
  elige por barrido (4, 8, 16) mirando el test. Tres alternativas, de la más
  barata a la más cara:

  1. **SVD de la historia del comando. No cuesta ninguna corrida.** El banco FIR
     mira 400 muestras por canal; si esa matriz tiene rango efectivo bajo, los
     canales de más caen en direcciones casi nulas y el barrido sobra. Tomarle la
     SVD al regresor y leer cuántos valores singulares son apreciables cierra la
     pregunta en una tarde, y contrasta de frente la hipótesis de por qué 16
     anda peor que 8. La maquinaria ya existe para los parámetros en
     `exp_subset_selection.py` (QR con pivoteo y SVD sobre la matriz de
     sensibilidad); sería la misma idea sobre el regresor del filtro.
  2. **Esparsidad de grupo**, si la SVD dice que hay algo que podar. Cada canal
     es un grupo y una penalización L2,1 manda canales enteros a cero exacto.
     Se entrena una vez con 16 y se cuentan los que sobreviven. El repo está a
     un paso: `--wd-fir` ya aplica L2 sobre el FIR, que encoge todo pero no
     anula nada. **Dos advertencias:** las corridas con L2 que ya existen
     salieron inestables (`filtro, 4 canales, L2 1e-3` divergió en varias y una
     llegó a 187 % de error de parámetros), y se cambia un barrido discreto de
     tres valores por un λ continuo que también hay que elegir, aunque se pueda
     elegir con validación.
  3. **Dropout sobre los canales: no.** Sirve contra la coadaptación en redes
     sobreparametrizadas con datos de sobra. Acá el problema es el opuesto, los
     canales de más no se coadaptan sino que ven direcciones casi nulas, así que
     matarlos al azar agrega ruido al gradiente sin tocar la causa.

- **La familia de escalones es demasiado chica, y un escenario decide
  comparaciones.** `box_a1.2` es el único escalón de test; en entrenamiento solo
  están `box_a0.4` y `box_a0.8`. Es el caso de extrapolación y el escenario más
  difícil y más disperso de los siete: sobre las 63 corridas de `act1` su mediana
  es 27,7 % contra 8 a 13 % de los demás, con desvío 8,0. Con un séptimo del peso
  aporta cerca de un tercio del promedio, y alcanza para invertir una comparación
  (el filtro de 8 canales con arranque desde el white-box parece mejorar solo con
  `σ = 0,01`; sacando ese escenario el orden es monótono y la diferencia
  desaparece). Haría falta una familia de amplitudes graduada con más de una en
  test. **Es la intervención más cara:** rehacer los datasets cambia su
  `data_sha256` e invalida las 63 corridas. Antes conviene lo barato, que es
  reportar la mediana o separar el escenario de extrapolación, porque
  `por_escenario` ya está guardado en cada JSON y no hace falta correr nada.

  **Dos predicciones falsables**, para que esto no quede en «más datos es mejor»:

  - *El óptimo de canales del filtro debería subir.* El banco FIR mira 400
    muestras de comando, y con estímulos suaves esa historia es de rango bajo:
    los canales de más caen en direcciones casi nulas, donde el gradiente no
    distingue nada. Más estímulos y más variados abren direcciones
    independientes y vuelven identificables los canales extra. O sea que «16 es
    peor que 8» puede ser una afirmación sobre tener 13 escenarios y no sobre
    que 16 canales sean demasiados.
  - *El óptimo de ventana debería alargarse.* Con 4000 muestras y 13
    trayectorias, la ventana de 5 ms da 507 ventanas de gradiente por época y la
    de 20 ms solo 117. La ventana larga acerca el objetivo de entrenamiento al
    criterio de corrida libre, pero paga en condicionamiento y en cantidad de
    muestras; más trayectorias compensan justo esa parte.

  Si al duplicar los escenarios ninguno de los dos óptimos se mueve, el tamaño
  del dataset no era el límite y conviene mirar otra cosa.

  **Ojo con cómo se amplía, que esto ya se midió** (`exp_a_set_design.py`).
  Optimizar *un* estímulo bajó la fracción imitable del 71,4 % del mejor de
  librería al 59,5 %. Pero armar un dataset con 20 variantes de ese diseño
  óptimo subió la fracción **conjunta** a 73,9 %, peor que la librería (67,1 %).
  Cuando se exige un único δθ que explique todos los escenarios a la vez, la
  degeneración se rompe sola si los escenarios son **distintos entre sí**, y las
  siete familias de la librería hacen ese trabajo gratis. O sea que hay dos
  palancas y la segunda pesa más: que cada estímulo sea bueno, y que sean
  complementarios. Engordar la familia de escalones es la palanca débil.

  Conviene separar los dos objetivos, que empujan distinto:
  - *Identificabilidad:* familias nuevas, o correr `exp_a_set_design.py`, que
    optimiza el conjunto y ya está escrito.
  - *El peso de `box_a1.2`:* ese sí pide más miembros en la familia de
    escalones, porque el problema es que la extrapolación la decide un punto
    único. Es diseño del conjunto de **test**, no de identificabilidad.
- **Hecho el 3-10: dataset ampliado con validación.** `refrac1_amp` y
  `act1_amp` (más sus `_n01` y `_n05`), generados con
  `scripts/esc_gen_ampliado.py` sobre `build_scenarios_ampliado` de
  `gen_multi_dataset.py`. Son 49 escenarios: 28 de entrenamiento, 7 de
  validación (uno por familia) y 14 de test (dos por familia). Los 20 originales
  están adentro sin cambios, con el mismo rol y trayectorias idénticas bit a bit,
  así que el test viejo es un subconjunto del nuevo. El chirp, que antes solo
  estaba en test, tiene ahora tres escenarios de entrenamiento que cubren entre
  todos su banda. `box_a1.2` queda marcado con `is_extrap` y se reporta aparte
  (`nrmse_test_interp`, `nrmse_extrap`).

  **Lo que explica el peso de `box_a1.2`.** La familia de escalones tiene dos
  regímenes: hasta amplitud 0,8 la actividad pico de E queda por debajo de 0,17,
  y desde 1,0 salta a 0,7 u 0,8. En el dataset original los dos escalones de
  entrenamiento (0,4 y 0,8) estaban en el régimen bajo y el de test (1,2) en el
  alto, así que no era solo extrapolar en amplitud sino caer en un régimen que
  la familia nunca mostró. Con `box_a1.0` en entrenamiento eso cambia. El
  escalón de validación (0,7) y el de test nuevo (0,5) están en el régimen bajo.

  **Cómo se usa la validación.** `esc_run.py` mide el NRMSE de corrida libre
  sobre los 7 de validación cada `--val-every` épocas (100 por defecto) y otra
  vez después de L-BFGS, y se queda con el estado de menor validación
  (`MejorVal`, en `graybox_train.py`, compartido con `fit_aug`). El JSON guarda
  `nrmse_val_mejor`, `ep_mejor_val`, `nrmse_val_ultima` y la historia. Eso
  también cubre las corridas que divergían al final: ya no se evalúa la última
  época. Las elecciones de arquitectura se hacen por `nrmse_val_mejor`, nunca
  por test.

  Los datasets viejos no traen `is_val`, y con ellos todo corre igual que antes.
- **Partición con validación.** Hoy los datasets tienen 13 escenarios de
  entrenamiento y 7 de test, sin validación, y todas las elecciones de
  arquitectura (400 contra 800 retardos, 4 contra 8 canales, el ancho de la red)
  se hicieron mirando el test. Separar 3 escenarios de entrenamiento como
  validación, uno por tipo de estímulo, y rehacer la comparación final.
  **Además el presupuesto de ajuste fue desparejo:** la ventana se barrió solo
  para la agnóstica (100, 200 y 400 muestras), la historia y el filtro barrieron
  su propio hiperparámetro, y la forma exacta y el estado de filtro corrieron sin
  ningún barrido, en los valores por defecto. Eso juega en contra del resultado
  principal y no a favor, porque las dos variantes que ganan son las únicas sin
  afinar; donde sí infla es adentro de la familia entrenable.
- **Penalización de la antigüedad del núcleo**, el centroide
  $\sum_k k\,w_k^2 / \sum_k w_k^2$. Solo con su intensidad elegida por NRMSE de
  validación y probada también en una planta con memoria larga (adaptación con
  $\tau_a$ de 30 o 100 ms), para que no le pase a la red el valor de $\tau$.
- **Ruido realista.** El ruido blanco actual da 27 y 41 dB dentro de la banda de
  la señal; las grabaciones reales, una cota de unos −7 dB. Falta ruido dentro de
  banda a niveles alrededor del real y una componente de ruido de proceso.
- **Semillas** de las configuraciones que se comparen, para saber qué diferencias
  son reales.
- **La penalización L2 del filtro no está medida.** Las tres corridas con L2
  (`e5_K400_wd1e-4`, `e5_K400_wd1e-3`, `e8_K400wd4_n01`) divergieron al final del
  entrenamiento: la pérdida terminó entre 3 y 12 veces por encima de su mínimo, y
  el pipeline evalúa el modelo de la última época. Ninguna otra corrida con ruido
  lo hizo. Sospecha sin verificar: `torch.optim.Adam` con `weight_decay` mezcla la
  penalización con el gradiente antes de escalarlo; probar `AdamW`. Aparte, evaluar
  el modelo de menor pérdida de entrenamiento y no el de la última época, que no
  usa el test. `scripts/divergencia.py` detecta los casos.
  Dato a favor: la única L2 con ruido que no divergió, `e8_K400wd3_n05`
  ($\lambda = 10^{-3}$, $\sigma = 0{,}05$), dio NRMSE 11,16, la mejor reproducción de
  todas las correcciones agnósticas a ese nivel y 0,7 puntos mejor que el estado de
  filtro. Una semilla. Si se sostiene con semillas y con `AdamW`, la penalización
  pasa a ser la opción con ruido.
- **Arranque desde el white-box** (corrido el 29 y 30-09, fuera del póster por
  decisión del 30-09). Entrenar todo junto, pero partiendo del β del white-box ya
  ajustado en lugar del arranque ignorante. En refractariedad la corrección
  agnóstica pasa a identificar β (con $\sigma = 0{,}01$, error de 26,4 a 5,5 % con
  ventana de 20 ms y de 23,7 a 4,2 % con 5 ms; $R^2$ contra $\Delta f$ de −0,46 a
  0,79). Con $\sigma = 0{,}05$ solo se sostiene con ventana de 20 ms. En el
  actuador no identifica en ninguna variante y la reproducción cambia en los dos
  sentidos sin patrón. Congelar β es peor que no congelar. Falta: semillas, la
  versión sin ruido que divergió (`e10_B400_wrm`), el estado de filtro desde el
  white-box (`fit_aug` no acepta `--init-params`), y probar si la diferencia entre
  plantas es por cuán deformado sale el β del white-box. Tablas en
  `results/tabla_arranque.md` (`scripts/tabla_arranque.py`); corridas `e10_*` y
  `e11_*`, colas `cola_dos_etapas.py` y `cola_arranque_wb.py`.

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

Los puntos 1, 2 y 3 de la versión anterior (costo de la variante B, eje 1.1 y
eje 1.2) se corrieron el 11-09 y están arriba.

1. Eje 1.3, la variante `K`. Es el mejor candidato a bajar el NRMSE, porque el
   mismo campo receptivo cuesta un orden de magnitud menos de parámetros, y de
   paso da el núcleo aprendido para graficar contra $e^{-t/\tau}$.
2. Eje 1.4, el barrido de capacidad y de $K$, empezando por $K=800$, que es la
   extrapolación directa de la tendencia medida.
3. Eje 6, las mismas dos variantes con ruido de observación. Decide si el FIR
   gana por costo o también por robustez.
4. Eje 2, forma exacta + red sobre `refrac1`. 13 min. Ahora explica la inversión
   entre $K=100$ y $K=400$ en vez de decidir nada.
5. Eje 3, la rama estructural completa.
6. Eje 4, la planta combinada, regenerando antes sus líneas de base.

## Lo que no haría

- Agrandar `g(I,E)`. Tiene un techo medido de $R^2 = -0{,}11$ y no se rompe con
  capacidad.
- Volver a la regularización de `g`. El barrido la agotó.
- Latent ODE genérica. Dio $R^2 = -0{,}51$ y es la más cara de todas.
