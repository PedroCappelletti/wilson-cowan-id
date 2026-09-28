# Informe de la sesión del 11/9/2026

Para leer de una. Tres partes: **lo que está corriendo**, **dos errores que
aparecieron y hay que saber**, y **la evaluación de los datos que ofrecieron los
investigadores**.

---

## 1. Lo que quedó corriendo

`bash scripts/run_eje1.sh` lanza tres entrenamientos sobre `act1`, la planta con
el actuador. **En secuencia a propósito**, uno por vez: en paralelo el campo
`minutos` mide contención y deja de servir como costo, que es lo que arruinó la
tanda de agosto.

| tag | qué prueba | estimado |
|---|---|---|
| `e2_A` | `g(I,E,P,Q)`, comando instantáneo, sin memoria | ~45 min |
| `e2_H100` | historia de 100 muestras (5 ms) | ~60 min |
| `e2_H400` | historia de 400 muestras (20 ms) | ~70 min |

Total del orden de **tres horas**. Los resultados quedan en
`results/escalado/e2_{A,H100,H400}.json` y los logs en `logs/escalado/`.

### Qué mirar cuando terminen

El número que decide es el **$R^2$ de la corrección**, no el NRMSE. Las
referencias sobre `act1`:

| | NRMSE | $R^2$ |
|---|---|---|
| white-box (`e2_wb`) | 15,23 | −0,01 |
| ciego `g(I,E)` (`e2_B`) | 15,40 | −1,87 |
| techo del oráculo state-only | | **−0,11** |
| estado de filtro (`e2_lag2`) | 2,96 | 0,94 |

La hipótesis es que **`A` no debería pasar de −0,11**, porque sigue sin memoria,
y que **`H400` sí**, porque con la historia del comando $\Delta f$ pasa a ser
función de la entrada. Si `H400` da $R^2$ positivo, el eje 1 funciona y la
corrección aprendió la física sin que se la escribiéramos. Si `H100` queda en el
medio, además queda medido que lo que importa es el campo receptivo.

### El costo, medido y no estimado

Cronometré el bucle de Adam de cada variante sobre los datos reales:

| variante | s/época | 1500 épocas, solo Adam |
|---|---|---|
| B | 0,80 | 20 min |
| A | 0,82 | 21 min |
| H, K=100 | 1,16 | 29 min |
| H, K=400 | 1,36 | 34 min |

La historia sale barata: a K=400 es 1,7× el costo de `B` y 81 MB de tensor. Los
estimados de arriba aplican el factor 2,1× que sale de comparar esos 20 min con
los 41,7 min reales de `e2_B_v2`, que incluyen L-BFGS, sensibilidades y
evaluación. Sale de un solo punto, así que es orden de magnitud.

---

## 2. Dos errores que aparecieron

### 2.1. `c_P` nunca se entrenaba, en datos reales

En `train_real_output.py`, la ganancia del estímulo `c_P` estaba declarada como
parámetro, puesta en el optimizador y reportada en cada log, **pero no entraba
en la pérdida**. `fit_windows` integraba con `Pw` crudo y solo `free_rollout`
aplicaba `c_P * u`.

Consecuencia: **el modelo se entrenaba con $P = u$ y se evaluaba con $P = 3u$**,
y `c_P` se quedaba clavada en su valor inicial por no recibir gradiente. Lo
confirmé corriendo 40 épocas más L-BFGS: `c_P = 3.000` exacto mientras `c_out`
se movía de 0,05 a 0,20.

Está arreglado. **Importa para la interpretación**: el $R^2 \approx 0$ del
rollout libre sobre datos reales, que quedó documentado como «el WC no anda en
corrida libre», se midió con esa discrepancia de factor 3 entre entrenamiento y
evaluación. La conclusión probablemente se sostiene igual, porque el techo de
0,04 a 0,11 del ARX lineal es evidencia independiente de que la tarea es dura,
pero **el número del WC hay que volver a medirlo**.

### 2.2. Las 101 horas del escalado eran 27

Ya corregido en el complemento, en el plan y en el README del repo limpio. Las
doce corridas de agosto fueron en cuatro carriles paralelos y `minutos` mide
reloj de pared, así que sumar los doce campos cuenta cuatro veces el mismo
tiempo de máquina. La señal que lo delata: las tres corridas del barrido de
ventana informan entre 24,8 y 26,3 horas **pese a tener ventanas de 100, 200 y
400**, o sea cuatro veces distinta cantidad de trabajo.

---

## 3. Los datos que ofrecieron los investigadores

**Para esta línea de trabajo, no sirven.** No es un problema de formato ni de
preprocesamiento, y la adaptación no es chica.

El motivo de fondo es uno solo: **no hay entrada controlada**. Todo este
proyecto es identificación de un sistema *forzado*, donde aplicamos un estímulo
$u(t)$ conocido y medimos la respuesta, y el objetivo declarado es diseñar el
estímulo antes de aplicarlo. En el dataset que ofrecen la variable que organiza
todo es la conducta del animal, que no comandamos nosotros y que además está en
lazo cerrado con la actividad neuronal: la posición en el laberinto es en parte
consecuencia de la actividad, no una causa exógena. Eso rompe el argumento de
identificabilidad sobre el que se apoya el trabajo entero.

Hay dos problemas más, cada uno serio por su cuenta:

- **La escala temporal.** El imaging de calcio va a decenas de Hz como mucho, y
  el indicador es un pasabajos con decaimiento de cientos de ms. La dinámica que
  modelamos es de decenas de ms. Habría que agregarle al modelo de observación
  un modelo del indicador, encima del que ya nos falta.
- **La pregunta es otra.** Lo que ellos quieren, cómo se reorganiza el patrón
  poblacional con el condicionamiento, es una pregunta de **geometría de
  representaciones**, no de dinámica forzada. CEBRA es la herramienta correcta
  para eso y nosotros no aportaríamos nada mejor.

**Lo único que sí tienen y a nosotros nos falta** son decenas de neuronas por
separado. Nuestro cuello de botella hoy es que solo observamos el LFP, que es
una proyección de rango uno del estado, y con neuronas individuales se podrían
separar poblaciones excitatoria e inhibitoria, o sea observar $E$ e $I$ en vez
de $E - I$. Eso atacaría el eje 5 de lleno. Pero sin estímulo controlado sigue
sin alcanzar.

### Qué pedirles, en dos frases

> Necesitamos registros donde se aplique un **estímulo conocido y controlado por
> nosotros** (optogenético, eléctrico o sensorial) con un curso temporal rico
> (chirp, escalones, ruido de banda ancha, no pulsos idénticos repetidos), y la
> respuesta poblacional medida en simultáneo con muestreo bien por encima de la
> dinámica de interés, o sea electrofisiología, LFP o tasas de disparo a 250 Hz
> o más, no calcio.
>
> Si además las poblaciones excitatoria e inhibitoria se pueden distinguir, eso
> resuelve un problema abierto que hoy tenemos.

Vale la pena preguntarles algo concreto: **si en esa misma preparación pueden
correr sesiones con estimulación optogenética durante el imaging**. Si la
respuesta es sí, el dataset pasa de no servir a ser exactamente lo que hace
falta, salvo por la escala temporal del calcio.

---

## 4. Estado del código

Todo commiteado en `reproducibilidad-semilla-procedencia` y pusheado.

- **`Sg`** (eje 2), forma exacta más red. Probada de punta a punta sobre
  `refrac1`, con `g_rms = 0,0099`, o sea la red activa junto a la corrección
  estructurada.
- **`H`** (eje 1.2), red con historia del comando. `P` y `Q` pasan a ser
  `(...,K)` y el backbone recorta el primer canal, así el integrador no se toca
  y las variantes viejas no se enteran.
- **`scripts/test_variantes_nuevas.py`**, chequea las cinco variantes, que el
  gradiente llegue al backbone, a la red y a los parámetros estructurados, y que
  las viejas sigan con un canal. Pasan todas.
- **`train_real_output.py`**, arreglado el `c_P` y abierto a todas las variantes
  del escalado, con soporte de historia.
- Datasets y checkpoints ahora versionados en los dos repos, y `esc_run.py`
  estampa el sha256 del dataset en cada JSON.

### Lo que queda, y por qué no lo hice

**La métrica de horizonte corto sobre datos reales.** Es la que discrimina, la
corrida libre tiene techo 0,1. Implementarla requiere una decisión: para
predecir a horizonte corto en una grabación *held-out* hay que estimarle el
estado inicial de cada ventana, y hay que decidir si eso se hace optimizando
solo los `x0` con los parámetros congelados (que es lo razonable y análogo al
teacher forcing) o de otra forma. No quise elegirlo por mi cuenta.

**El eje 1.3, la convolución.** Tiene sentido recién si `H` funciona. Si `H400`
da $R^2$ positivo, la convolución es la versión eficiente y la que permite
graficar el núcleo aprendido para compararlo con $e^{-t/\tau}$.
