---
type: note
status: inbox
project: "[[Investigación Neurociencia]]"
fecha: 2026-10-15
tags: [gray-box, wilson-cowan, control, imc, decision]
---

# Conectar el controlador a los modelos gray-box: opciones

Qué hace falta para usar cada modelo del escalado dentro del controlador IMC, y qué opciones hay para los que hoy no se pueden conectar. Para decidir. Código del lazo cerrado: `src/neural_ode/closed_loop.py` (`IMCController`, `simulate_closed_loop`).

Relacionado: [[Especificaciones técnicas - pipeline gray-box]] · [[Avances desde el póster de la Jornada]] · [[Para la próxima reunión]].

---

## Cómo calcula el comando el controlador

El IMC hace linealización por realimentación. En cada paso:
1. un PI con acción integral decide qué dinámica quiere para I y para E;
2. invierte la sigmoidea para saber qué entrada tiene que recibir cada población;
3. despeja el comando P (y Q) que produce esa entrada, restando el acoplamiento con los pesos del modelo: $P = u_p - (w_{EE}E - w_{EI}I)$.

El paso 3 es el que importa: el controlador necesita **despejar P del modelo en ese instante**. Si el modelo deja hacerlo con una cuenta, se conecta directo. Si no, hace falta algo más.

---

## Estado por modelo

| modelo | ¿se conecta? | por qué |
|---|---|---|
| white-box (β̂) | **sí, ya** | P entra sumado, se despeja con una cuenta |
| forma exacta (β̂, r̂, saturación) | **sí, ya** | el factor refractario se divide y la saturación se invierte con artanh |
| agnóstica $g(I, E)$ | **sí, ya** | la red no depende de P: se evalúa, se resta, y P se despeja igual |
| cualquier corrección, usando solo su β̂ | **sí, ya** | se ignora la corrección y se usa β̂ como un white-box |
| historia del comando, filtro entrenable | **no** | la red recibe P(t), así que P aparece adentro de la red |
| estado de filtro | **no** | P no llega directo a las poblaciones: pasa por el retardo del actuador |
| latent ODE | **no** | además del problema anterior, el estado oculto no se observa |

### Lo que ya se corrió con los modelos que se conectan (15-10)

`scripts/control_refrac.py` y `scripts/control_act.py` construyen el controlador con cada modelo de la tanda e13, desde cero y con warm start, en los tres niveles de ruido. Todos los controladores enfrentan la misma planta verdadera y tienen los mismos topes de actuación. Las referencias son theta-gamma de 120 Hz, 50 ms. Se mide el RMSE de seguimiento (promedio de I y E, sin el primer quinto del transitorio). Resultados en `results/control_refrac.json` y `results/control_act.json`.

**Refractariedad: identificar bien importa cuando se usa la corrección para cancelar.**

| controlador | error de β | RMSE |
|---|---|---|
| oráculo (θ verdaderos, sin saber de la refractariedad) | 0 | 0,0339 |
| oráculo con la forma de la refractariedad | 0 | 0,0324 |
| white-box y agnósticas usando solo su β̂ | 3 a 19 % | 0,0333 a 0,0352 |
| agnóstica desde cero, β̂ + g cancelada | 15 a 19 % | **0,045 a 0,049** |
| agnóstica con warm start, β̂ + g cancelada | 3 a 6 % | 0,0331 a 0,0355 |
| forma exacta con warm start | 3,5 a 8,4 % | 0,0326 a 0,0327 |

- **Usando solo β̂, el error de β casi no importa.** De 3 a 19 % de error, el RMSE queda dentro de un 4 % del oráculo. La acción integral absorbe el error, como ya había mostrado la caracterización previa.
- **Cancelar la corrección aprendida desde cero empeora el control entre 30 y 45 %**, sobre todo en el lazo de I (RMSE de I de 0,035 a 0,055). La red aprendió a compensar un β deformado; puesta en el controlador junto a ese β, empeora la cancelación.
- **Con warm start, la misma corrección cancelada queda al nivel del oráculo.** Es el primer resultado donde identificar bien β se traduce en controlar mejor.
- **La forma exacta con warm start iguala al oráculo que conoce la forma** (0,0326 contra 0,0324).

**Actuador: el retardo domina y la identificación de β no predice el control.**

| controlador | error de β | RMSE |
|---|---|---|
| oráculo (θ verdaderos, sin saber del actuador) | 0 | **0,0676** |
| white-box | 23 a 32 % | 0,088 a 0,090 |
| correcciones con historia o filtro, solo β̂ | 15 a 36 % | 0,053 a 0,103 |
| estado de filtro, solo β̂ | 7,5 a 19,5 % | 0,053 a 0,078 |
| agnóstica desde cero, β̂ + g cancelada | 25 a 27 % | **0,10 a 0,18** |

- **El actuador sin compensar duplica el error incluso con θ verdaderos** (0,068 contra 0,034 en la planta con refractariedad). Ningún controlador de esta tabla compensa el retardo, y es lo que más pesa.
- **El error de β no ordena a los controladores.** Algunos β deformados controlan mejor que el oráculo (la historia de 100 con warm start da 0,053 con 26 % de error), probablemente porque la deformación absorbe parte del efecto del actuador. Con una semilla, no hay patrón entre arranques.
- **Cancelar la agnóstica $g(I, E)$ empeora mucho**, porque la red no puede representar el retardo y su cancelación mete error.
- Conclusión: en el actuador, para que el modelo mejore el control hay que conectar la parte del modelo que representa el retardo. Esa es justamente la parte que hoy no se conecta (opciones A, B, C y E de abajo).

---

## Opciones para la historia del comando y el filtro entrenable

El problema: con $g(I, E, P_t, P_{t-1}, \dots)$, el comando actual aparece dos veces, en el backbone y adentro de la red. Despejarlo es resolver en cada paso

$$
f_{WC}(x, P_t, Q_t; \hat\beta) + g_\phi(x, P_t, Q_t, \text{pasado}) = \text{objetivo}
$$

en $P_t$ y $Q_t$ a la vez, y la red no tiene inversa analítica.

**Opción A. Resolver la ecuación en cada paso (Newton con el gradiente de la red).**
- A favor: es la cancelación correcta, sin aproximar.
- En contra: dos incógnitas por paso, con unos 10 000 pasos por simulación; más lento.
- Riesgo principal: la red no tiene por qué ser monótona en P. Puede haber dos soluciones o ninguna, y entonces el controlador salta o se vuelve inestable. Hay que chequear la invertibilidad, y eso es un resultado en sí: dice si la corrección agnóstica sirve para control o solo para predecir.

**Opción B. Evaluar la red con el comando del paso anterior.**
- A favor: es simple, se arma en una tarde, y no hay que invertir nada. El paso del lazo es de 0,005 ms y el comando casi no cambia entre un paso y el siguiente.
- En contra: es una aproximación, y mete un error que hay que medir. Si la red es muy sensible a $P_t$, el error puede realimentarse.
- Cómo validarla: comparar con la opción A en algunos casos, o mirar cuánto cambia $g$ entre usar $P_{t-1}$ y $P_t$.

**Opción C. Separar la red en una parte que depende del comando actual y otra que no.**
- Reentrenar la corrección de modo que el comando actual entre de forma afín, por ejemplo $g = g_1(x, \text{pasado}) + g_2(x, \text{pasado})\,P_t$. Así se despeja con una cuenta.
- A favor: cancelación exacta y explícita, como en la agnóstica.
- En contra: hay que reentrenar todas las variantes con historia, y la restricción puede empeorar la reproducción.

**Opción D. Usar solo su β̂ (ya disponible).** Sirve como cota: dice cuánto vale su identificación de β para el control, sin aprovechar la red.

---

## Opciones para el estado de filtro

El problema: el comando pasa por el filtro del actuador, $\dot{\hat P} = (P - \hat P)/\hat\tau$, antes de llegar a las poblaciones. El controlador sabe qué entrada efectiva quiere, $P_{ef}^*$, pero no puede imponerla en el instante: tiene que pedírsela al filtro.

**Opción E. Precompensar el retardo (inversión dinámica).**
- Pedir $P = \hat P + \hat\tau\,\dot P^*_{ef}$, invirtiendo antes la saturación con artanh. Con eso el filtro entrega lo que el controlador quiere.
- A favor: usa la forma física que el modelo identificó; la cancelación es exacta en el modelo.
- En contra: necesita la derivada de la señal del controlador, que amplifica el ruido de medición y puede pedir comandos muy grandes. Hay que acotarla o filtrarla.
- Requiere: que el controlador lleve el estado $\hat P$ del filtro, que se puede simular adentro porque depende solo del comando.

**Opción F. Ignorar el retardo y compensar solo la saturación.**
- Usar β̂ y la saturación identificada, como hace la forma exacta, y dejar el retardo de 1 ms a la acción integral.
- A favor: no hay que derivar nada; ya casi está implementado.
- En contra: no aprovecha la parte del modelo que lo hace bueno. Mide cuánto aporta el retardo al control.

**Opción D aplica igual:** solo su β̂.

---

## Para decidir

- [ ] ¿Vale la pena conectar la historia y el filtro? Mirar primero cuánto separa el control a los modelos que ya se conectan. Si las diferencias son chicas (la acción integral absorbe mucho), quizás no compense.
- [ ] Si se conecta: ¿opción A (exacta y diagnóstico de invertibilidad), B (rápida), o C (reentrenar con estructura afín)?
- [ ] Para el estado de filtro: ¿opción E (precompensar el retardo) o F (solo saturación)?
- [ ] ¿Se miden también referencias distintas de la theta-gamma de 120 Hz? El retardo pesa más cuanto más rápida es la referencia.
