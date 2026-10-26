---
type: note
status: inbox
project: "[[Investigación Neurociencia]]"
fecha: 2026-10-26
tags: [wilson-cowan, control, imc, gray-box, explicacion]
---

# Controladores: explicación a fondo

Qué hace el controlador del paper, por qué no puede usar los modelos con memoria tal como está, cómo se extendió, y cómo se armaron las comparaciones. Los números están en [[Controladores - resultados]].

Relacionado: [[Conectar el controlador - opciones]] · [[Especificaciones técnicas - pipeline gray-box]] · [[La parte biológica - neuronas, optogenética y Wilson-Cowan]]. Paper: Martínez, Sánchez-Peña, Belluscio, Piriz y García-Violini (2022), *Towards an experimental control of neural activity: the Wilson-Cowan model*, IFAC PapersOnLine 55-40: 223-228.

---

## 1. Qué tiene que hacer el controlador

Llevar la actividad de las dos poblaciones a una forma deseada, manejando la luz en tiempo real. En las pruebas, que E e I sigan dos senoidales de 120 Hz, un ritmo gamma, con la inhibitoria atrasada respecto de la excitatoria (ecuaciones 13 y 14 del paper):

$$
r_E(t) = 0{,}3\,\sin(2\pi\cdot 120\,t) + 0{,}45,\qquad
r_I(t) = 0{,}2\,\sin(2\pi\cdot 120\,t - 0{,}94) + 0{,}25
$$

El controlador mide E e I en cada instante y decide los comandos P (luz sobre la población excitatoria) y Q (sobre la inhibitoria).

---

## 2. El controlador del paper, paso a paso

**La idea: hacer que Wilson-Cowan parezca lineal.** La ecuación 2 del paper agrupa todo lo que está adentro de la sigmoidea en una entrada nueva:

$$
\dot E = \tfrac{1}{\tau_E}\big(-E + u_1\big),\qquad u_1 = f_E(\gamma_{xE} + P),\qquad \gamma_{xE} = w_{EE}E - w_{EI}I
$$

Visto desde $u_1$, cada población es un sistema lineal de primer orden, $1/(\tau s + 1)$, y las dos quedan desacopladas. Toda la no linealidad y el acoplamiento entre poblaciones quedaron adentro de $u_1$.

En cada instante, el controlador hace cuatro cosas:

1. **Un PI decide qué $u$ quiere.** Para la excitatoria, $u_E = k_p(r_E - E) + k_i \int (r_E - E)\,dt$, con $k_p = 5$, $k_i = 5$; para la inhibitoria, $k_p = 10$, $k_i = 5$. Es la forma que toma el diseño IMC de Morari sobre el sistema lineal (ecuaciones 10 a 12).
2. **Satura $u$ al rango que la sigmoidea puede producir**, porque una sigmoidea no da valores fuera de su rango.
3. **Invierte la sigmoidea** para saber qué argumento hace falta: $x = f^{-1}(u)$.
4. **Despeja el comando restando el acoplamiento** (ecuación 6): $P = f_E^{-1}(u_1) - (w_{EE}E - w_{EI}I)$.

**Por qué funciona.** Si el modelo es exacto, el paso 4 cancela la no linealidad y el acoplamiento, y lo que queda entre el PI y la actividad es exactamente el sistema lineal para el que se diseñó el PI. La acción integral corrige además los errores constantes del modelo. Por eso este controlador es tolerante a errores en β, como ya se había visto en la caracterización previa.

**El paso que importa es el 4.** Funciona porque se cumplen dos supuestos:
- **P entra en un solo lugar**, sumado adentro de la sigmoidea, así que se despeja con una inversa.
- **P actúa al instante:** la luz se convierte en corriente sin demora. El paper lo dice explícitamente en la sección 2: desprecia la dinámica de la opsina suponiéndola mucho más rápida que la de las poblaciones, y deja el caso con esa dinámica para una extensión.

---

## 3. Por qué los modelos con memoria no entran en el controlador del paper

| modelo | ¿se puede despejar P con una cuenta? |
|---|---|
| white-box | sí: es el modelo del paper |
| forma exacta | sí: el factor refractario $(1 - rE)$ se divide antes de invertir la sigmoidea, y la saturación se invierte con artanh |
| agnóstica $g(I, E)$ | sí: g no depende de P, así que se resta del objetivo antes de invertir. Esto es «cancelar g» |
| historia y filtro entrenable | **no:** la red recibe el comando actual. P aparece dos veces, en la sigmoidea y adentro de la red, y no hay una inversa que lo despeje |
| estado de filtro | **no:** P no llega al instante. Pasa primero por el retardo del actuador, así que lo que el controlador fija ahora recién afecta a las poblaciones después |

Cualquier modelo de los que tenemos se puede usar de todos modos **con solo su β̂**, ignorando la corrección. El paper lo justifica en la sección 3.2: trata lo no modelado como una incertidumbre acotada y demuestra estabilidad por pequeña ganancia. Así anda, pero no aprovecha lo que aprendió la red.

---

## 4. El controlador extendido

**Qué conserva.** Todo lo que es del paper: el PI con las mismas ganancias, la saturación de $u$, la inversa de la sigmoidea y los mismos topes. Cambia solo el paso 4, el despeje del comando. Código: `IMCExtendido`, en `src/neural_ode/closed_loop_ext.py`.

**Historia y filtro entrenable: despeje numérico con la red.**
- El objetivo es el mismo del paper: que el campo del modelo valga $\tfrac{1}{\hat\tau}(-x + u_{sat})$, la dinámica que pidió el PI.
- En vez de una cuenta, se resuelve $f_{modelo}(x, P, Q, \text{pasado}) = \text{objetivo}$ con el método de Newton en (P, Q). El jacobiano se calcula por diferencias finitas, y el pasado son los comandos que el propio controlador ya mandó, al paso de muestreo con que se entrenó el modelo.
- Se arranca del despeje cerrado del paper o de la solución del paso anterior, la que esté más cerca.
- Si Newton no encuentra un comando mejor que el despeje cerrado dentro de los topes, se usa el despeje cerrado y se cuenta como falla de inversión. **Pasa en alrededor de la mitad de los pasos:** la red no es invertible en el comando en buena parte del rango. Esto es un resultado: dice que estas correcciones sirven para predecir, pero no para despejar el comando.
- Para que fuera viable en 2 s de simulación (400 000 pasos), la red se evalúa con una copia en numpy, verificada contra la de torch con diferencias del orden de $10^{-8}$.

**Estado de filtro: copia interna del actuador.**
1. El despeje cerrado del paper da el comando efectivo que haría falta, $P^*_{ef}$.
2. Se invierte la saturación identificada: $P^*_{lag} = \operatorname{artanh}(\hat\alpha P^*_{ef})/\hat\alpha$.
3. Se compensa el retardo con una copia del filtro que el controlador integra adentro:

$$
P = \hat P + \frac{\hat\tau}{\tau_f}\big(P^*_{lag} - \hat P\big),\qquad \dot{\hat P} = \frac{P - \hat P}{\hat\tau}
$$

Con eso, la copia $\hat P$ sigue al valor deseado con constante $\tau_f$ en vez de $\hat\tau$. $\tau_f = 0{,}2$ ms es un parámetro de diseño: más chico compensa mejor el retardo y pide comandos más grandes. La copia recibe siempre el comando que efectivamente se aplicó.

**Agnóstica y forma exacta:** el paper ya cancela todo el modelo, así que el extendido no les cambia nada.

### Diferencias de estructura, en una tabla

| | original (paper) | extendido |
|---|---|---|
| PI, ganancias, saturación de u, inversa de la sigmoidea | sí | igual |
| despeje del comando | cuenta cerrada (ec. 6) | igual, más un despeje con el modelo completo |
| estados internos del controlador | 2 integradores | 2 integradores, más la copia del actuador (estado de filtro) |
| memoria de comandos pasados | no | sí (historia y filtro) |
| qué parte del modelo usa | β̂, y g(I, E) o la forma exacta si los hay | todo el modelo |

---

## 5. Cómo se simula y cómo se mide

- **Planta:** Wilson-Cowan con los parámetros verdaderos más la perturbación (refractariedad o actuador), integrada con RK4 a 0,005 ms. Es la «planta real» del experimento simulado.
- **Actualización del controlador:** cada 0,005 ms, el mismo paso de la planta, igual para todos. Se probó actualizarlo cada 0,05 ms y, en la planta con actuador, eso empeoraba a todos, incluso al oráculo (0,069 contra 0,093). La comparación habría medido el muestreo y no los modelos.
- **Medición:** E e I se miden directo y sin ruido; no hay filtro de Kalman, a diferencia del paper.
- **Métrica:** la raíz del error cuadrático medio entre E, I y sus referencias, promediada entre las dos poblaciones, sin contar el primer quinto de la simulación (el arranque desde el reposo).
- **Oráculo:** el controlador del paper con los parámetros verdaderos, sin saber de la perturbación. En refractariedad hay un segundo oráculo que además conoce la forma de la refractariedad.

### Las dos referencias

- **Gamma de 50 ms:** las senoidales de 120 Hz encendidas todo el tiempo, 6 ciclos.
- **Theta-gamma de 2 s:** como en la figura 7 del paper. Se genera una onda theta (ruido filtrado entre 4 y 12 Hz, semilla fija) y una ventana $T_w$ que vale 1 en el cuarto de ciclo centrado en cada valle de la theta. Las senoidales de gamma se multiplican por esa ventana: 16 ráfagas de unos 28 ms, y reposo entre ellas.

### El tope del comando

Sin tope, los controladores piden comandos muy fuera de lo realizable. Incluso el oráculo pide P entre −13 y +13, y el extendido con estado de filtro entre −30 y +73. Los estímulos de entrenamiento van de 0 a 2. Una luz no puede ser negativa y tiene un máximo.

Por eso se corrió todo en tres modos:
- **Sin tope**, como en el paper.
- **Tope literal:** P acotado a [0, 2,0] y Q a [0, 1,5], el máximo de cada canal en los estímulos de entrenamiento (lo pone el chirp, que oscila alrededor de un offset). El controlador no se entera.
- **Tope con anti-windup:** el mismo tope, y además el integrador del PI deja de acumular mientras su comando está contra el tope y el error lo empujaría más allá. Es lo que lleva cualquier controlador real con actuación limitada.

**Por qué hace falta el anti-windup.** Cuando el comando choca contra el tope, el error no se corrige y el integrador sigue acumulándolo sin freno. Al salir del tope, el integrador está cargado de más y el sistema oscila. Con el tope literal, en la planta con actuador todos los controladores se rompen (RMSE de 0,40 contra 0,07). No es por el rango del comando, sino por el windup.

### Qué se corrió

Todos los modelos de las dos tandas:
- **200 ms (`e13`):** la tabla completa, tres niveles de ruido, desde cero y con warm start.
- **2 s (`e16`):** las configuraciones clave con tres semillas, más las variantes que completan la tabla con la semilla 0.

Cada uno con los controladores que le corresponden, las dos referencias y los tres modos: **1848 simulaciones.** Código: `scripts/control_comparado.py` y `scripts/cola_control.py`. Tablas y figura: `scripts/resumen_control.py`.

---

## 6. Cómo leer los resultados

- **El oráculo no es el mejor posible.** No sabe del actuador ni de la refractariedad. Un modelo que sí sabe puede superarlo, y eso es lo que muestra el estado de filtro.
- **Un β deformado puede controlar mejor que el verdadero.** En el actuador, varios controladores con β̂ equivocado le ganan al oráculo. La deformación de β absorbe parte del efecto del actuador, que es la misma ambigüedad entre β y la corrección de la identificación, vista desde el control.
- **La refractariedad casi no se nota en el control.** Todos quedan cerca del oráculo, porque la acción integral la compensa sola.
- **Con tope y anti-windup, el comando pasa entre el 40 y el 95 % del tiempo contra el tope.** El límite pasa a ser la autoridad del actuador, no el modelo. Eso explica que en la referencia theta-gamma de 2 s todos los controladores queden parecidos.

---

## 7. Limitaciones

- **Medición perfecta.** Se miden E e I sin ruido. En un experimento real habría que estimarlas a partir del LFP, con el filtro de Kalman del paper.
- **Ganancias del paper sin cambiar.** Están pensadas para actuación sin límite. Con el tope, un rediseño del PI para la autoridad real del actuador probablemente cambiaría los números.
- **τ_f del estado de filtro fijado a mano** (0,2 ms), sin barrer.
- **La historia y el filtro no invierten en la mitad de los pasos.** Para aprovecharlos haría falta un controlador que no invierta el modelo, como un MPC o la IMC clásica con el modelo completo en paralelo, o reentrenarlos con el comando entrando de forma afín.
- **Una sola planta por perturbación,** con parámetros fijos.
