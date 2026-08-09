# Las dos perturbaciones: qué son, dónde entran, cómo las ve la red

> El simulador dejó de ser Wilson-Cowan puro. Se le agregaron **dos** fenómenos
> físicos que el modelo no tiene, y esa diferencia es lo que la red neuronal
> tiene que aprender. Acá está cada uno en detalle.
>
> Código: `src/wilson_cowan/uncertainty.py`. Números medidos sobre `eps1.npz`.

---

## 1. El punto de partida

El modelo Wilson-Cowan que usamos es:

```
u_e = wEE·E − wEI·I + P − θe          ← entrada total a la población excitatoria
u_i = wIE·E − wII·I + Q − θi          ← ídem, inhibitoria

dE/dt = (1/te)·( −E + S(ae·u_e) − ke )
dI/dt = (1/ti)·( −I + S(ai·u_i) − ki )
```

| símbolo | qué es |
|---|---|
| `E`, `I` | fracción de la población excitatoria / inhibitoria **activa** (0 a 1) |
| `P`, `Q` | **estímulo externo** que nosotros comandamos a cada población |
| `wEE…wII` | pesos sinápticos: cuánto se influyen las poblaciones entre sí |
| `te`, `ti` | constantes de tiempo: qué tan rápido responde cada población [ms] |
| `ae`, `ai` | pendiente de la sigmoidea: cuán abrupta es la transición apagado→encendido |
| `θe`, `θi` | umbral: cuánta entrada hace falta para empezar a responder |
| `S(·)` | sigmoidea, la curva que convierte "entrada total" en "fracción que dispara" |
| `ke`, `ki` | offsets que garantizan que `E=I=0` sea un estado de reposo |

**El problema que motiva todo:** este modelo es demasiado prolijo. Cualquier
desajuste se puede tapar moviendo los 10 parámetros, así que la red neuronal
nunca tiene nada real que aprender y el gray-box no se activa de verdad.

La solución fue meterle al **simulador** dos fenómenos que el **modelo** no
tiene. Se eligieron a propósito por dos vías distintas: uno entra por la
**dinámica interna**, el otro por la **entrada**.

---

## 2. Perturbación 1 — Refractariedad

### La idea física

Una neurona que acaba de disparar **no puede volver a disparar inmediatamente**:
necesita unos milisegundos para recuperarse. Es el *período refractario*.

Entonces, cuando una fracción `E` de la población ya está activa, la fracción
realmente **disponible** para responder no es toda la población, sino:

```
disponible = 1 − r·E
```

`r` es qué proporción de los que están activos quedan bloqueados.

### La matemática

Multiplica la salida de la sigmoidea:

```
dE/dt = (1/te)·( −E + (1 − r_e·E)·S(ae·u_e) − ke )
dI/dt = (1/ti)·( −I + (1 − r_i·I )·S(ai·u_i) − ki )
                      └── esto es lo nuevo ──┘
```

| parámetro | valor (ε=1) | significado |
|---|---|---|
| `r_e` | 0.10 | fracción de la población E bloqueada por cada unidad de actividad |
| `r_i` | 0.10 | ídem para I |

**Cómo leerlo:** con `E = 0.78` (el máximo medido), el factor vale
`1 − 0.10·0.78 = 0.922`. O sea que en el pico de actividad la población responde
un **7.8% menos** de lo que respondería sin refractariedad. Es un efecto de
saturación: cuanto más activa está, más le cuesta activarse más.

### Por qué este y no otro

**No es una perturbación inventada.** Es literalmente el término que Wilson y
Cowan derivaron en su paper de 1972 y que la forma reducida —la que usamos—
descarta por simplicidad. O sea:

> el simulador pasa a ser el **Wilson-Cowan original** y el modelo es el
> **Wilson-Cowan reducido**.

Es un desajuste estructural real y citable, no un capricho.

### Propiedades

- **Respeta el reposo:** en `E=I=0` el factor vale exactamente 1, así que el
  estado de reposo no se mueve. Importante — si no, el desajuste sería un simple
  corrimiento y no un cambio de dinámica.
- **Depende sólo del estado.** Es una función pura de `(I, E)`.

---

## 3. Perturbación 2 — Actuador no ideal

### La idea física

En optogenética uno comanda **intensidad de luz**, pero lo que la neurona recibe
no es eso. Pasa por dos deformaciones:

1. **Retardo (lag).** El canal ChR2 no se abre instantáneamente: es un filtro de
   primer orden con constante de tiempo `τ_act`. La luz sube de golpe, la
   corriente sube exponencialmente.
2. **Saturación.** La opsina no puede entregar corriente infinita. Pasado cierto
   nivel, más luz ya no da más corriente.

### La matemática

**Primero el retardo.** Se agrega una variable de estado nueva `P_lag` que
*persigue* al comando:

```
dP_lag/dt = (P_cmd − P_lag) / τ_act
```

Si el comando se queda quieto, `P_lag` lo alcanza exponencialmente en ~`τ_act`
milisegundos. Ídem para `Q_lag`.

**Después la saturación**, aplicada a lo que salió del filtro:

```
P_eff = A · tanh( P_lag / A )
```

La `tanh` aplana suavemente: para `P_lag ≪ A` es casi idéntica (`≈ P_lag`, no
distorsiona), y para `P_lag ≫ A` se aplana en `±A`.

Y ese `P_eff` es el que entra en la ecuación, **en lugar** del comandado:

```
u_e = wEE·E − wEI·I + P_eff − θe        ← antes acá iba P
```

| parámetro | valor (ε=1) | significado |
|---|---|---|
| `τ_act` | 1.0 ms | qué tan lento reacciona el canal. Más grande = más retardo |
| `A` (`sat`) | 3.0 | nivel donde satura. Más chico = satura antes |
| `P_lag`, `Q_lag` | — | **estados ocultos** (2 nuevos), no observables |

### Cuánto deforma, medido

| medida | valor |
|---|---|
| `P` comandado | rango [0, 1.60] |
| `P_eff` efectivo | rango [0, 1.46] |
| `\|P_eff − P\|` promedio | 0.1015 → **22.8% del comando** |
| de eso, cuánto es **saturación** | 0.0161 |
| de eso, cuánto es **retardo** | el resto (~84%) |

**El retardo domina ampliamente sobre la saturación.** Es el dato más importante
de esta sección y explica por qué la red tiene tanto problema con esta
perturbación (sección 5).

### Por qué este

Rompe el supuesto de que `P, Q` son **perfectamente conocidos** — que es
justamente el supuesto que hoy ancla la escala de los 10 parámetros. Es un test
duro y honesto. Además entra por las **dos entradas** del sistema, que era el
requisito de diseño original.

---

## 4. Cómo se combinan: la perilla ε

Las dos se aplican **juntas**, graduadas por un único número `ε`:

```python
r       = 0.10 · ε          # refractariedad: ε=0 la apaga
sat     = 3.0  / ε          # ε chico -> satura muy tarde -> casi lineal
τ_act   = max(1.0 · ε, 0.15)   # ε chico -> reacciona casi instantáneo
```

- `ε = 0` → Wilson-Cowan puro, sin perturbación.
- `ε = 1` → punto de operación nominal (todos los números de arriba).
- Rango validado: **ε ∈ [0.25, 2.0]**.

Cuánto deforma la trayectoria cada nivel (medido):

| ε | desvío de la trayectoria | veces el ruido de observación |
|---|---|---|
| 0.25 | 11.7% | 0.99× |
| 0.5 | 27.5% | 2.3× |
| **1.0** | **53.9%** | **4.6×** |
| 2.0 | 77.5% | 6.6× |

El nominal `ε=1` está calibrado para quedar **bien por encima del ruido** (4.6×)
sin destruir el régimen oscilatorio: el sistema sigue teniendo 21 cruces por
ciclo, los mismos que sin perturbación. Recién en ε=2 empieza a perder
oscilaciones (19 cruces).

> **Por qué `τ_act` tiene un piso de 0.15 ms.** Bajar `τ_act` vuelve *rígida* la
> ecuación del actuador: cuando se acerca al paso del integrador, el RK4 de paso
> fijo se desestabiliza **en silencio**. Por eso el límite ideal no se alcanza
> bajando ε de forma continua; para "sin perturbación" hay que usar `ε = 0`
> explícitamente, que devuelve otro objeto.

**El resultado neto.** El hueco entre simulador y modelo, medido como la
diferencia de derivadas en el mismo estado:

```
Δf = f_planta(x, P, Q) − f_WC(x, P, Q)        |Δf| promedio = 0.0157
```

Repartido muy desparejo: `|Δf_E| = 0.0150` contra `|Δf_I| = 0.0021`. **El
desajuste vive casi todo en el canal excitatorio**, que es donde entra `P`.

---

## 5. Cómo entra esto en la red

Acá está el punto que conecta con el entrenamiento.

El modelo gray-box es `ẋ = f_WC(x,P,Q;θ) + g_φ(x)`. La red `g_φ` tiene que
aprender ese `Δf`. **Pero sólo recibe el estado `(I, E)`** — no ve la
perturbación, ni `P_eff`, ni los estados ocultos del actuador.

### El techo teórico de cada una

La pregunta previa a entrenar es: *¿existe siquiera una función de `(I,E)` que
represente este `Δf`?* Se mide entrenando la arquitectura exacta de `g_φ` contra
el `Δf` **verdadero** (un oráculo que el entrenamiento real no tiene) y viendo
qué R² alcanza:

| perturbación | ¿es función pura del estado? | techo (R²) |
|---|---|---|
| **Refractariedad** | **Sí.** `(1−r·E)·S(u)` depende sólo de `(I,E)`. | **0.97** |
| **Actuador — saturación** | parcialmente: es función de `P`, que deja rastro en el estado | — |
| **Actuador — retardo** | **No.** `P_lag` es una variable independiente. Dos instantes con el mismo `(I,E)` pueden tener distinto `P_lag`, y por lo tanto distinta derivada: ninguna función de `(I,E)` puede dar las dos. | **0.34** |

> Ese 0.34 está medido con `τ_act = 0.5 ms` (el caso difícil). Con el nominal de
> 1.0 ms el retardo es aún más lento y más difícil de imitar sin memoria.

### El techo de las dos juntas: −0.11

Y acá viene lo que importa, porque **no se promedian**. Medido sobre el par
combinado que se usa en todos los experimentos, el techo de una corrección que
sólo ve el estado es:

```
R² = −0.11        (negativo = peor que predecir la media)
```

O sea: **el `Δf` del par elegido no es función del estado**, punto. La parte
buena (refractariedad) queda enterrada bajo la parte con memoria, que además es
la que domina el desajuste — recordemos que el retardo aporta ~84% de la
deformación del actuador.

Esto **no es un defecto del entrenamiento**: es una limitación estructural, y era
esperable. Una corrección sin memoria no puede representar un fenómeno con
memoria propia. De hecho la corrección realmente entrenada llega a R² = −0.63
contra el `Δf` verdadero: está en el techo, no por debajo.

### Por qué se eligió así a propósito

Si las dos perturbaciones fueran capturables, el experimento sería fácil y no
diría nada sobre los límites del enfoque. La combinación da las dos caras:

- una parte **capturable en principio** (refractariedad, 0.97) → verifica que el
  gray-box funciona cuando puede;
- una parte **demostrablemente fuera de alcance** (el retardo) → marca dónde está
  el techo, y explica por qué la corrección ajusta bien los datos pero **no**
  aprende la física real.

Es la diferencia entre "la red no aprendió" y "no había nada que la red pudiera
aprender con la información que recibe". Acá es lo segundo, y está medido.

> **El detalle metodológico que más importa.** El `.npz` guarda `P_eff` y `Q_eff`
> por separado del comandado, pero **el entrenamiento nunca los usa**: sólo ve
> `P` y `Q`. Si usara el efectivo le estaríamos regalando la mitad de la
> respuesta — sabría el resultado de la perturbación que justamente tiene que
> descubrir. Lo mismo con `dfI`/`dfE`, que están guardados pero se usan sólo para
> evaluar.

---

## 6. Resumen en una tabla

| | **Refractariedad** | **Actuador no ideal** |
|---|---|---|
| **origen físico** | período refractario neuronal | cinética del canal ChR2 + saturación de la opsina |
| **dónde entra** | multiplica la sigmoidea | reemplaza `P,Q` por `P_eff,Q_eff` |
| **ecuación** | `(1 − r·x)·S(u)` | `dP_lag/dt=(P−P_lag)/τ`, `P_eff=A·tanh(P_lag/A)` |
| **parámetros (ε=1)** | `r = 0.10` | `τ_act = 1.0 ms`, `A = 3.0` |
| **estados ocultos** | ninguno | **2** (`P_lag`, `Q_lag`) |
| **efecto medido** | hasta 7.8% menos de respuesta en el pico | 22.8% de desvío promedio en el comando |
| **respeta el reposo** | sí | sí |
| **techo de `g_φ(I,E)`** | **R² = 0.97** | **R² = 0.34** |
| **por qué** | función pura del estado | el retardo tiene memoria propia |

**Y las dos juntas —que es como se usan— tienen techo R² = −0.11.** Ése es el
número que gobierna todos los resultados del proyecto.

---

## Conexiones

- `docs/neural_ode_entrenamiento_detallado.md` — cómo se entrena la red que
  tiene que aprender esto
- `docs/incertidumbre_dinamica_graybox.md` — la calibración de ε y las otras 7
  familias de perturbación disponibles
- `src/wilson_cowan/uncertainty.py` — el código, con 9 familias composables
