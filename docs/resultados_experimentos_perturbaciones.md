# Resultados de los experimentos con perturbaciones

> Qué se probó, qué dio y qué significa. Todos los números salen de los JSON en
> `results/uncertainty/`.
>
> Contexto previo: `docs/las_dos_perturbaciones.md` (qué son las perturbaciones),
> `docs/neural_ode_entrenamiento_detallado.md` (cómo se entrena).

---

## La pregunta que se quería contestar

Le metimos al simulador dos fenómenos que el modelo no tiene (refractariedad y
actuador no ideal). La pregunta era:

> **¿La corrección neuronal `g_φ` sirve para algo? ¿Recupera los parámetros que
> el desajuste arruina, y aprende la física que falta?**

La respuesta corta: **sí ayuda a identificar, pero no aprende la física.** Y hay
una condición importante que hace que a veces empeore todo. Vamos por partes.

---

## Resumen ejecutivo

| # | experimento | qué contesta | resultado |
|---|---|---|---|
| F1 | Caracterización | ¿cuánto deforma la perturbación? | ε=1 desvía la trayectoria **53.9%**, 4.6× el ruido |
| F2 | Costo de la rigidez | ¿cuánto daña al white-box? | error de parámetros de **1.05% → 59.5%** |
| F3 | Gray-box | ¿la corrección lo arregla? | lo baja a **35.6%** (mejora 40%) |
| F4b | Geometría | ¿por qué no más? | **67%** del desajuste se imita moviendo parámetros |
| F4 | FIM | ¿por dónde se filtra? | **93.4%** por la dirección más débil (`wII`) |
| F5 | Recuperación funcional | ¿aprendió física? | **No.** R² negativo en las tres variantes |
| F6 | Lazo cerrado | ¿sirve para controlar? | ajustar bien ≠ poder cancelar |
| F7 | Controles | ¿cuándo sí funciona? | funciona con física **sin memoria** |

---

## F1 — ¿Cuánto deforma la perturbación?

Antes de nada había que calibrar: si la perturbación es muy chica se confunde con
el ruido, y si es muy grande destruye la dinámica oscilatoria que nos interesa.

| ε | desvío de la trayectoria | veces el ruido | cruces por ciclo |
|---|---|---|---|
| 0.25 | 11.7% | 0.99× | 21 |
| 0.5 | 27.5% | 2.3× | 21 |
| **1.0** | **53.9%** | **4.6×** | **21** |
| 1.5 | 66.9% | 5.7× | 20 |
| 2.0 | 77.5% | 6.6× | 19 |

**Conclusión:** ε=1 es el punto justo. Deforma bastante (54%, bien visible sobre
el ruido) pero el sistema **sigue oscilando igual**: 21 cruces, los mismos que sin
perturbación. Recién en ε=2 empieza a perder ciclos.

---

## F2 — El costo de la rigidez: cuánto daña al modelo sin corrección

Acá se entrena el **white-box** (Wilson-Cowan puro, sin red) contra datos
perturbados. Como al modelo le falta física, tiene que "esconder" el desajuste
deformando los parámetros.

| ε | error medio de parámetros | peor parámetro |
|---|---|---|
| 0 | **1.05%** | `wII` |
| 0.25 | 13.5% | `wII` |
| 0.5 | 27.7% | `wII` |
| **1.0** | **59.5%** | `wII` |
| 2.0 | 93.6% | `wII` |

**Lo que hay que ver acá:**

1. **Sin perturbación el método funciona perfecto** (1.05% de error). O sea que el
   problema no es el entrenamiento ni el optimizador — es el desajuste.
2. El daño **crece de forma ordenada** con ε. No es un artefacto.
3. **Siempre es `wII` el más castigado**, en los cinco niveles. No es casualidad:
   es el parámetro al que el sistema menos responde, así que es donde el error se
   puede esconder más barato. F4 lo confirma.

Esto define el problema: **59.5% de error es inservible.** ¿Puede la red arreglarlo?

---

## F3 — El gray-box: ¿la corrección arregla el problema?

Se entrenan varias versiones del modelo con corrección, todas a ε=1.

| variante | qué hace | error de parámetros | MSE test |
|---|---|---|---|
| white-box | sin red | 59.5% | 1.06e-2 |
| A | `g(I,E,P,Q)` — red libre | 35.9% | 8.51e-3 |
| **B** | **`g(I,E)` — no ve el estímulo** | **35.6%** | 9.63e-3 |
| C | B + penaliza el tamaño de `g` | 31.4% | 8.87e-3 |
| D | B + prohíbe redundancia (λ=1) | 31.1% | 8.54e-3 |
| D (λ=10) | ídem, más fuerte | 44.6% | 1.06e-2 |
| S | corrección **física** (3 parámetros) | 55.4% | 1.06e-2 |

**Resultado principal: la corrección baja el error de 59.5% a ~31-36%, o sea lo
mejora un 40%.** El desajuste deja de comerse los parámetros.

Detalles que importan:

- **C y D son las mejores** (~31%), lo cual tiene sentido: son las que más
  restringen a la red. Pero **D con λ=10 empeora todo** (44.6%): apretar demasiado
  le impide a la red hacer su trabajo. Hay un óptimo intermedio.
- **La corrección estructurada (S) fracasó** en este caso (55.4%). Es esperable:
  su forma física no incluye el retardo del actuador, que es justo lo que domina
  el desajuste. (En una planta que sí tiene su forma exacta, recupera el
  parámetro con 1.3% de error.)

### El resultado más importante de todo el trabajo: el cruce

¿Qué pasa si se usa la corrección **cuando no hace falta** (ε=0, sin perturbación)?

| situación | white-box | gray-box (B) | efecto de la red |
|---|---|---|---|
| **con** desajuste (ε=1) | 59.5% | 35.6% | **mejora 40%** ✅ |
| **sin** desajuste (ε=0) | 1.05% | 18.1% | **empeora 17×** ❌ |

**La red no es gratis.** Si no hay física faltante, la corrección se dedica a
tapar parámetros buenos y los arruina: de 1.05% a 18.1%.

Y lo más traicionero:

> **En los dos casos el MSE mejora** (15% mejor a ε=0, 9% mejor a ε=1).

O sea que **mirando sólo el error de predicción uno concluiría que la red siempre
ayuda** — y estaría equivocado la mitad de las veces. Por eso todo el proyecto
reporta las dos métricas juntas.

---

## F4b — ¿Por qué la corrección no arregla más?

La explicación está en la geometría del problema. Se mide qué fracción del
desajuste `Δf` se puede imitar simplemente **moviendo los 10 parámetros**:

| ε | fracción imitable moviendo parámetros | física genuinamente nueva |
|---|---|---|
| 0.25 | 30.3% | 69.7% |
| 0.5 | 47.9% | 52.1% |
| **1.0** | **67.1%** | **32.9%** |
| 1.5 | 82.2% | 17.8% |
| 2.0 | 91.4% | 8.6% |

**Ésta es la causa raíz de todo.** A ε=1, **dos tercios del desajuste son
ambiguos**: se pueden explicar cambiando los parámetros *o* con la red, y nada en
los datos permite distinguir cuál es la correcta.

Por eso el reparto entre `θ` y `g` es en buena parte arbitrario, y por eso la
corrección recupera sólo parte del error.

Peor: **cuanto más grande la perturbación, más ambigua se vuelve.** A ε=2 el 91%
es imitable. O sea que no alcanza con "perturbar más fuerte" para que el problema
se note mejor — se vuelve más degenerado.

---

## F4 — ¿Por dónde exactamente se filtra el error?

Se descompone el sistema en sus 10 direcciones naturales (SVD de la matriz de
información de Fisher), ordenadas de la más "sensible" a la más "sorda".

- **Número de condición: 12.414.** El sistema responde 12 mil veces más a la
  dirección más fuerte que a la más débil. Está muy mal condicionado.
- **El 93.75% de lo que hace la red** es imitable por un cambio de parámetros.
- Y de esa energía robada, **el 93.4% se va por la dirección más débil de las 10**;
  **el 99.5% por las tres más débiles**.

Si la red robara identificabilidad "al azar", las tres direcciones más débiles se
llevarían el 30%. Se llevan el 99.5%.

**Qué significa:** la red no ataca al azar — se mete exactamente por donde el
sistema es sordo, que es donde no cuesta nada y no se nota en el MSE. Y esa
dirección más débil corresponde a `wII`, **el mismo parámetro que F2 encontró como
el más castigado**. Dos experimentos independientes apuntando al mismo culpable.

---

## F5 — ¿La corrección aprendió física de verdad?

La prueba honesta: comparar lo que aprendió `g_φ` contra el `Δf` **verdadero**
(que conocemos porque nosotros pusimos la perturbación).

| modelo | qué ve `g` | R² logrado | **techo** de esa arquitectura | ¿llegó al techo? |
|---|---|---|---|---|
| A | `I,E,P,Q` | −0.63 | **+0.78** | **no, ni cerca** |
| B | `I,E` | −1.21 | −0.11 | no |
| C | `I,E` | −0.07 | −0.11 | **sí** |

**Todos los R² son negativos**: la corrección aprendida predice el `Δf` real
**peor que decir "el promedio"**. Ninguna aprendió la física.

Pero la columna del techo separa dos fracasos muy distintos, y conviene no
mezclarlos:

**B y C chocan contra un límite estructural.** Como sólo ven `(I,E)`, lo máximo
que podrían lograr es −0.11 — el `Δf` simplemente **no es función del estado**,
porque el retardo del actuador tiene memoria propia (`P_lag`). C llega a −0.07,
o sea que **está en el techo**: no entrenó mal, la tarea era imposible.

**A es el caso interesante.** Como sí ve `P,Q`, su techo es **+0.78**: con esa
información la física *sí* era aprendible en buena medida. Y sin embargo logró
−0.63. **A no chocó contra ningún límite: eligió no aprender la física.**

¿Por qué? Porque teniendo acceso al estímulo le resulta más fácil tapar
parámetros equivocados que descubrir el mecanismo real — es exactamente el modo
de falla que motivó la variante B. El dato que lo confirma: A es también la de
mayor redundancia con los parámetros (0.918).

Es la diferencia entre "la tarea era imposible" (B, C) y "había cómo, pero el
optimizador encontró un atajo" (A).

**Corolario:** identificar ≠ predecir ≠ entender. La corrección recupera 40% del
error de parámetros, mejora 9% la predicción, y aprende 0% de la física. Las tres
cosas se miden distinto y no vienen juntas.

---

## F6 — ¿Sirve para controlar?

Se cierra el lazo con un controlador y se mide el error de seguimiento.

| controlador | RMSE en I | RMSE en E |
|---|---|---|
| 0. planta limpia + parámetros verdaderos | 0.033 | 0.031 |
| 1. oráculo (parámetros verdaderos) | 0.057 | 0.063 |
| 2. white-box | 0.053 | 0.089 |
| 3a. gray-box, **sin** cancelar `g` | 0.055 | 0.078 |
| 3b. gray-box, **cancelando** `g` | **0.206** | 0.065 |

**El resultado sorprendente está en la última fila.** Cancelar la corrección
—usar el conocimiento aprendido para compensar activamente— **mejora el canal E un
18% y destruye el canal I: lo empeora 3.8 veces.**

**Por qué pasa:** la red ajusta bien *en promedio*, pero no representa la física
real (eso es F5, R²≈0). Restarla no cancela nada — inyecta una señal equivocada
que en un canal casualmente ayuda y en el otro rompe todo.

**La lección, que es general:** *ajustar bien no es lo mismo que poder cancelar.*
Un modelo puede predecir aceptablemente y ser inútil —o peligroso— para control.
Si sólo mirábamos el canal E, la conclusión habría sido la opuesta.

---

## F7 — ¿Cuándo sí funciona el gray-box?

Para no quedarnos con "no funcionó", se probaron 12 tipos distintos de física
faltante, midiendo cuánto puede capturar la corrección.

| tipo de física faltante | R² con `g(I,E)` | veredicto |
|---|---|---|
| Refractariedad | **0.97** | capturable ✅ |
| Adaptación rápida (τ=1 ms) | 0.91 | capturable ✅ |
| Heterogeneidad | 0.77 | capturable ✅ |
| Adaptación media (τ=10 ms) | 0.61 | parcial |
| Población oculta | 0.61 | parcial |
| Adaptación lenta (τ=30 ms) | 0.44 | parcial |
| Adaptación muy lenta (τ=100 ms) | 0.31 | pobre |
| Deriva de `wEE` en el tiempo | 0.23 | pobre |
| Ruido de proceso | 0.12 | pobre (control negativo) |
| **El par que usamos** | **−0.11** | **el más difícil** |

**Se ve un patrón limpísimo, y es el hallazgo más generalizable del trabajo:**

> **Lo que decide si el gray-box funciona no es cuán grande es el desajuste, sino
> si tiene memoria propia.**

- Física que depende **sólo del estado actual** → la red la captura muy bien
  (0.9+).
- Física con **memoria propia** → cuanto más lenta su memoria, peor. La serie de
  adaptación lo muestra como una perilla continua: τ=1 ms da 0.91 y τ=100 ms da
  0.31, con el mismo tipo de fenómeno.
- **Ruido de proceso** da 0.12 y es el control negativo: nadie puede aprenderlo, y
  sirve para verificar que la métrica no da falsos positivos.

Y confirma que el par elegido (−0.11) es **deliberadamente el caso más duro de
todos**, no un caso desafortunado.

---

## Qué se aprendió, en cinco frases

1. **La corrección neuronal ayuda a identificar cuando hay física faltante**
   (error de 59.5% → 35.6%), pero **arruina los parámetros cuando no la hay**
   (1.05% → 18.1%). No es gratis.

2. **El MSE no sirve para decidir si usarla:** mejora en los dos casos, incluido
   aquel donde la red está haciendo daño.

3. **La causa raíz es geométrica:** el 67% del desajuste se puede imitar moviendo
   parámetros, así que el reparto entre `θ` y `g` es en buena parte arbitrario. Y
   la red se filtra justo por la dirección más sorda del sistema (93.4% por
   `wII`).

4. **Ajustar bien no es aprender física, y aprender física no es poder
   controlar.** Los tres son distintos y se midieron por separado: 40% de mejora
   en parámetros, 9% en predicción, 0% en física, y un canal del lazo cerrado
   destruido al intentar cancelar.

5. **El predictor de si el gray-box va a funcionar es la memoria**, no el tamaño
   del desajuste. Sin memoria, R² > 0.9; con memoria lenta, R² < 0.35.

---

## Qué haríamos ahora

En orden de prioridad, según lo que los experimentos señalan:

1. **Diseñar los estímulos** para bajar la fracción imitable del 67%. Es atacar la
   causa raíz. Ya se probó y funciona por estímulo (71.4% → 59.5%), pero hay que
   diseñar el **conjunto**, no la señal aislada: 20 variantes de un mismo diseño
   dieron peor (73.9%) que una librería diversa (67.1%).
2. **Darle memoria a la corrección** (un estado interno, tipo GRU). Es la única
   forma de subir del techo de −0.11, porque el problema es estructural.
3. **Correcciones con forma física** en vez de una red genérica: identificables y
   cancelables en el lazo cerrado, que es donde la red genérica falló.
