# ¿El gray-box copia bien la dinámica?

> La pregunta acá **no** es si identifica los parámetros, sino si **reproduce la
> trayectoria**: se le da el estado inicial y el estímulo, tiene que generar los
> 200 ms enteros por su cuenta y parecerse a lo que hizo el simulador perturbado.
>
> Todos los modelos entrenados con datos de la planta **con perturbación** (ε=1) y
> evaluados sobre **estímulos que nunca vieron**.

---

## Cómo se mide

**NRMSE en % del rango de la señal.** El error cuadrático medio, dividido por
cuánto se mueve la señal. Es interpretable directo:

| NRMSE | lectura |
|---|---|
| < 5% | muy bueno, casi indistinguible |
| 5-15% | la forma está bien, con desvíos visibles |
| > 25% | inservible |

Y es **rollout completo**: 4000 pasos seguidos sin reiniciar, 40 veces más largo
que las ventanas de 5 ms con las que se entrenó.

---

## El resultado

| modelo | train | **test** | canal I | canal E |
|---|---|---|---|---|
| oráculo (parámetros **verdaderos**) | — | **21.6%** | — | — |
| white-box (sin red) | 7.1% | **14.0%** | 13.0% | 15.1% |
| gray-box A | 9.2% | 14.2% | 15.7% | 12.7% |
| gray-box B | 13.0% | 15.6% | 16.7% | 14.5% |
| gray-box C | 7.4% | 13.3% | 12.4% | 14.2% |
| **gray-box D** | 7.4% | **13.1%** | 12.3% | 14.0% |

*Referencia de qué significa "aprender bien": el mismo white-box, entrenado sobre
la planta **sin** perturbación, da **2.0%**.*

### Las tres cosas que dice esta tabla

**1. Los dos copian la dinámica razonablemente bien: ~13-14%.** La forma de la
trayectoria está bien, con desvíos visibles pero sin perderse. Nada explota — el
error tampoco se acumula con el tiempo (ver más abajo).

**2. La corrección neuronal casi no ayuda: 14.0% → 13.1%.** Una mejora del **6%**.
Comparalo con lo que la misma corrección logra en identificación de parámetros
(59.5% → 35.6%, una mejora del 40%). **Para copiar la dinámica, la red aporta
casi nada.**

**3. Y el dato más raro de todos:** el white-box entrenado (14.0%) **reproduce la
trayectoria mejor que los parámetros verdaderos** (21.6%). Un modelo con 59.5% de
error en sus parámetros predice mejor que el modelo exacto.

---

## Por qué: los parámetros equivocados son la compensación

Ésta es la explicación de todo lo anterior, y es el punto central del documento.

La planta tiene física que el modelo no tiene (refractariedad + actuador con
retardo). Con los parámetros **verdaderos**, esa física falta y nada la
compensa → 21.6% de error.

Pero el entrenamiento no busca parámetros verdaderos: **busca la trayectoria que
mejor ajusta.** Y encuentra que deformando los parámetros puede *imitar* buena
parte del efecto de la perturbación. Entonces:

```
parámetros verdaderos     → 0% error en θ,  21.6% error en trayectoria
parámetros deformados     → 59.5% error en θ,  14.0% error en trayectoria
                              └── el precio ──┘  └── lo que compra ──┘
```

**Los parámetros "malos" no son un fracaso del entrenamiento: son la
compensación que el modelo eligió pagar.** Sacrifica interpretabilidad para
ganar precisión predictiva.

Y esto no es una interpretación: está medido aparte. **El 67% del desajuste se
puede imitar moviendo los 10 parámetros** (experimento F4b). Ese 67% es
exactamente el margen que el white-box aprovecha para bajar de 21.6% a 14.0%.

---

## Por qué la red aporta tan poco

Si el white-box ya usa los parámetros para tapar el 67% imitable del desajuste,
a la red sólo le queda el **33% restante** — la física genuinamente nueva.

Y ahí está el problema: **ese 33% es justo la parte que la red no puede
aprender.** Es el retardo del actuador, que tiene memoria propia (`P_lag`), y la
red sólo ve el estado `(I,E)`. Dos instantes con el mismo `(I,E)` pueden tener
distinto `P_lag` y distinta derivada — ninguna función de `(I,E)` representa las
dos.

Está medido: el techo teórico de una corrección que sólo ve el estado es
**R² = −0.11** contra el `Δf` verdadero. Negativo. No hay nada que aprender ahí.

```
desajuste total
├── 67% imitable con parámetros  → se lo lleva el backbone (y funciona)
└── 33% física nueva             → le tocaría a la red...
                                   ...pero es justo la parte CON MEMORIA
                                   que g(I,E) no puede representar
```

**La red queda sin trabajo útil.** Por eso mejora un 6% y no un 40%: no es que
esté mal entrenada, es que lo que quedaba disponible para ella era
estructuralmente inalcanzable con la información que recibe.

> Esto explica también por qué **B es la peor de las variantes** (15.6%, peor que
> el white-box). B es la que menos información recibe: no ve el estímulo. Eso la
> hace buena para identificar (no puede tapar parámetros) pero mala para predecir.
> **Las restricciones que ayudan a identificar cuestan precisión de trayectoria.**

---

## Escenario por escenario

El promedio esconde mucha variación. Éstos son los 7 estímulos de test:

| escenario | oráculo | white-box | gray-box D |
|---|---|---|---|
| `box_a1.2` | 38.5% | 33.0% | **30.1%** |
| `square_a1.0_f130` | 35.2% | 11.6% | **10.7%** |
| `aprbs_2` | 6.9% | **5.0%** | 5.5% |
| `prbs_1` | 19.8% | 21.2% | **16.8%** |
| `thetagamma_2` | 8.3% | **4.4%** | 4.5% |
| `poisson_1` | 20.3% | **10.1%** | 10.9% |
| `chirp` | 22.2% | **13.0%** | 13.5% |
| **promedio** | 21.6% | 14.0% | **13.1%** |

**Lo que se ve:**

- **El rango es enorme: de 4.4% a 33%.** En estímulos ricos y rápidos
  (`aprbs`, `thetagamma`) los dos modelos copian la dinámica **muy bien** (~5%).
  En `box_a1.2` —un escalón grande y sostenido— los dos fallan feo (~30%).
- **La razón es dónde se nota la perturbación.** Un escalón grande y sostenido
  lleva la actividad a valores altos, que es donde la refractariedad muerde más
  (el factor `1−r·E` se aparta más de 1) y donde el actuador satura. Un estímulo
  rápido y de amplitud moderada nunca entra en ese régimen, así que el modelo sin
  la física faltante alcanza igual.
- **La corrección gana donde el desajuste es grande y pierde donde es chico.**
  Gana en `box` (−2.9 puntos) y `prbs` (−4.4), pierde levemente en los fáciles.
  Tiene sentido: donde no falta física, la red sólo puede molestar.

---

## ¿Se degrada con el tiempo?

Una preocupación legítima: el modelo se entrenó con ventanas de 5 ms y acá se le
piden 200 ms. ¿Se va acumulando el error?

| modelo | 0-50 ms | 50-100 | 100-150 | 150-200 |
|---|---|---|---|---|
| oráculo | 11.8% | 17.6% | 17.0% | 13.2% |
| white-box | 7.2% | 9.4% | 10.5% | 10.6% |
| gray-box D | 7.1% | 8.5% | 10.5% | 8.8% |

**No hay deriva.** El error sube un poco al principio y después se estabiliza —
no crece sin control. Los modelos son **dinámicamente estables**: se quedan
enganchados al régimen oscilatorio correcto aunque no reproduzcan la fase exacta.

Es un resultado importante y no era obvio: significa que estos modelos **sirven
como planta** para diseñar un controlador, aunque sus parámetros estén mal.

---

## Conclusión

**¿El gray-box copia bien la dinámica? Sí, razonablemente: ~13% de NRMSE, la
forma correcta y sin deriva. Pero el white-box lo hace casi igual (14%), y la
corrección neuronal aporta apenas un 6%.**

La explicación completa en tres pasos:

1. **El desajuste es en gran parte imitable con parámetros** (67%). El white-box
   lo aprovecha: deforma los parámetros y baja el error de trayectoria de 21.6%
   (con los valores verdaderos) a 14.0%.
2. **Eso deja a la red sólo el 33% no imitable** — la física genuinamente nueva.
3. **Y ese 33% es justo la parte con memoria propia**, que una corrección
   `g(I,E)` no puede representar (techo R² = −0.11).

De ahí la asimetría que define todo el trabajo:

> La corrección neuronal es **útil para identificar** (mejora 40% los parámetros,
> porque le saca al backbone la presión de compensar) pero **casi inútil para
> predecir** (mejora 6%, porque lo que quedaba por explicar era inalcanzable).

**Qué haría falta para mejorar la reproducción de la dinámica.** No es entrenar
más ni agrandar la red — el límite no está ahí. Habría que **darle memoria a la
corrección** (un estado interno, tipo GRU), que es la única forma de representar
el retardo del actuador. Es la diferencia entre subir del techo o seguir chocando
contra él.

---

*Números: `results/uncertainty/f2_eps1.json` (white-box), `models/f3_*_eps1.pt`
(gray-box), `f4b_geometria.json` (el 67%), `f5_recovery.json` (el techo R²).
Detalle de los experimentos en `docs/resultados_experimentos_perturbaciones.md`.*
