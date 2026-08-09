# Incertidumbre dinámica en el simulador: cómo activar el gray-box de verdad

> **Estado:** propuesta de diseño con evidencia numérica. Nada de esto está todavía en `src/`.
> Las mediciones salen de un banco de pruebas que implementa 9 familias candidatas de
> perturbación y mide, para cada una, cuánto deforma la dinámica y si el término `g_φ`
> de `GrayBoxWC` es *capaz* de representarla.

---

## 1. El problema, en una frase

Hoy el simulador que genera los datos y el backbone de la red **son exactamente las mismas
ecuaciones**. Por eso `use_correction=True` no está activado en ningún experimento sintético
(sólo en `train_real_output.py`, sobre datos reales): no hay nada que corregir.

```
  data/processed/control/multi_dataset.npz          src/neural_ode/dynamics.py
  ← generado por src/wilson_cowan/model.py  rhs()    ← GrayBoxWC.forward()

        dI = (1/ti)(-I + S_i(u_i) - ki)                 dI = (1/ti)(-I + S_i(u_i) - ki)
        dE = (1/te)(-E + S_e(u_e) - ke)                 dE = (1/te)(-E + S_e(u_e) - ke)
                                                        + g_φ(I,E,P,Q)   ← no tiene trabajo
```

Un gray-box necesita un **hueco estructural**: que la planta que genera los datos tenga
física que el backbone no modela. Ese hueco es lo que `g_φ` aprende. Sin hueco, `g_φ` no
sólo es inútil sino **dañino**, por la razón de la sección 6.

---

## 2. Dónde se aplica la perturbación (respuesta directa)

**En el simulador, no en la red.** Concretamente, dentro de `rhs()` de
`src/wilson_cowan/model.py` — el campo vectorial que `solve_ivp` integra.

El reparto de roles queda así:

| Pieza | Qué representa | Qué sabe |
|---|---|---|
| `src/wilson_cowan/model.py` (perturbado) | **el cerebro real** / la planta física | toda la física, incluida la perturbación |
| `data/.../multi_dataset.npz` | lo que **medís** en el experimento | sólo `(t, I, E, P, Q)` — la perturbación es invisible |
| `GrayBoxWC` backbone | tu **hipótesis** de modelo (WC reducido) | los 10 parámetros |
| `g_φ` | lo que la hipótesis **no explica** | lo tiene que descubrir de los datos |

La red **nunca ve** la perturbación ni sabe que existe. Ve trayectorias `(I,E)` y el estímulo
`(P,Q)` que vos comandaste. Todo lo que no cierra con Wilson-Cowan puro tiene que salir de
`g_φ`. Eso es exactamente el escenario real: medís actividad y estímulo, y tu modelo del
cerebro es incompleto.

### ¿En qué momento se aplica?

**En todo instante, continuamente.** No es un evento ni un golpe en `t=100`. Es un término
que forma parte permanente del campo vectorial, activo desde `t=0` hasta el final. Está
siempre ahí porque *es física del sistema*, no una molestia externa.

Una consecuencia importante y deseable: todas las familias que propongo **preservan el
reposo `E=I=0` como equilibrio**. Con estímulo nulo el sistema sigue durmiendo en cero,
igual que ahora. La perturbación se "enciende" sola cuando el sistema se activa, porque
depende del estado. Esto evita que cambies silenciosamente el punto de operación y rompas
la convención de `ke`/`ki`.

### ¿Es algo aleatorio? — **No, y esto es lo más importante del documento**

Hay que separar tres cosas que se confunden:

| | Qué es | Dónde entra | ¿`g_φ` lo puede aprender? |
|---|---|---|---|
| **Ruido de observación** (ya lo tenés: `noise_std`) | medís mal | *después* de integrar; no toca la dinámica | No, ni debe |
| **Ruido de proceso** (Langevin/OU) | fluctuación real de la población | *dentro* del integrador; cambia la trayectoria | **No** — medido: R² = 0.04 |
| **Incertidumbre estructural** | física que tu modelo omite | dentro del campo vectorial, determinista | **Sí** — medido: R² = 0.97 |

`g_φ` es una función determinista de `(I,E,P,Q)`. Sólo puede aprender lo que es
**reproducible**: si el sistema vuelve a pasar por el mismo estado con el mismo estímulo,
la corrección tiene que valer lo mismo. El ruido, por definición, no cumple eso.

Lo medí explícitamente: entrené *la arquitectura exacta* de `g_φ`
(`Linear(4,32)-Tanh-Linear(32,32)-Tanh-Linear(32,2)`) contra el Δf verdadero de cada
familia, con estímulos de test nunca vistos. Con ruido de proceso OU el R² de test da
**0.04** — o sea, cero. Con refractariedad da **0.97**.

> **Conclusión de diseño:** la incertidumbre que activa el gray-box tiene que ser
> **determinista y con sentido físico**. El ruido de proceso es interesante como
> *piso irreducible* (el suelo que ninguna corrección puede bajar), no como el mecanismo.

---

## 3. El catálogo: 9 formas de volver menos rígido el simulador

Todas medidas sobre los mismos estímulos del `multi_dataset` (régimen ms, APRBS +
theta-gamma + PRBS + box, con chirp y APRBS nuevos como held-out).

Las columnas:
- **D_traj** — cuánto se deforma la trayectoria respecto de la WC nominal (RMS relativo).
- **|Δf|** — tamaño del término faltante respecto de `|f_WC|`: cuánto trabajo tiene `g_φ`.
- **R²_test** — cuánto de ese término puede capturar `g_φ(I,E,P,Q)` en estímulos nuevos.
  **Es la columna que decide si la familia sirve.**

| # | Familia | Sentido físico | D_traj | \|Δf\| | R²_test | Veredicto |
|---|---|---|---:|---:|---:|---|
| 1 | **Refractariedad** `(1-r·x)·S(u)` | El término del **Wilson-Cowan original (1972)** que la forma reducida tira | 47% | 29% | **0.967** | ✅ capturable |
| 2 | **Sigmoidea heterogénea** | La población real tiene umbrales dispersos → la curva efectiva no es una logística | 52% | 47% | **0.977** | ✅ capturable |
| 3 | **Actuador no ideal** (lag+saturación) | La luz que comandás no es la que llega: dinámica del canal ChR2 + saturación | 80% | 69% | **0.962** | ✅ capturable |
| 4 | **Retardo axonal** `E(t-τ)` | Tiempo de conducción en las conexiones cruzadas | 47% | 24% | **0.983** | ⚠️ ver §5 |
| 5 | **Población no modelada** | Una tercera población / columna vecina acoplada | 17% | 10% | 0.840 | 🟡 parcial |
| 6 | **Depresión sináptica** (Tsodyks) | Los pesos se debilitan con el uso (recursos finitos) | 57% | 42% | 0.714 | 🟡 parcial |
| 7 | **Adaptación (SFA)** | Corriente lenta que frena a E tras disparar mucho | 20% | 13% | 0.671 | 🟡 parcial |
| 8 | **Deriva de `wEE(t)`** | Neuromodulación lenta: el peso cambia en el tiempo | 60% | 33% | 0.333 | ❌ no autónoma |
| 9 | **Ruido de proceso (OU)** | Fluctuaciones de tamaño finito de la población | 86% | 99% | **0.042** | ❌ irreducible |

### Cómo leer los tres bloques

**✅ Capturables (1–3).** Δf es una función pura de `(I,E,P,Q)`. `g_φ` las representa casi
perfecto. Son las que hacen que el gray-box **gane**: hay un hueco real y la red lo tapa.
Sirven para demostrar que el gray-box funciona.

**🟡 Parciales (5–7).** Tienen un **estado oculto propio** (el recurso sináptico `D`, la
corriente de adaptación `A`, la población `J`). Δf depende de ese estado, que no está en
los argumentos de `g_φ`. La red captura la parte reconstruible y falla en el resto. Son las
que muestran el **límite** del gray-box memoryless y motivan la extensión natural (aumentar
el estado, o darle memoria a la corrección).

**❌ No aprendibles (8–9).** La deriva depende explícitamente de `t`, que no es argumento de
`g_φ`. El ruido no es función de nada. Sirven como controles negativos y como piso.

### Un resultado que vale la pena: la memoria aparece cuando la escala de tiempo es lenta

Barrí la constante de tiempo de cada mecanismo con intensidad fija. La adaptación lo muestra
limpísimo (recordá que `te=1 ms`, `ti=2 ms`):

| `τ_a` de la adaptación | 1 ms | 3 ms | 10 ms | 30 ms | 100 ms |
|---|---:|---:|---:|---:|---:|
| R²_test de `g_φ` | 0.973 | 0.956 | 0.826 | 0.541 | 0.332 |

Cuando el estado oculto es **rápido** respecto del sistema, queda esclavizado al estado
visible (`A ≈ E`) y `g_φ` lo reconstruye sin problema. Cuando es **lento**, guarda historia
propia que `(I,E,P,Q)` no contiene, y la corrección memoryless se queda corta.
Esto te da una perilla continua entre "capturable" y "no capturable" con **un solo número**.

El lag del actuador da la imagen espejo, y su interpretación es distinta:

| `τ_act` (lag puro) | 0.5 ms | 1 ms | 2 ms | 5 ms | 10 ms | 20 ms |
|---|---:|---:|---:|---:|---:|---:|
| R²_test de `g_φ` | 0.340 | 0.564 | 0.748 | 0.837 | 0.960 | 0.984 |

Acá el R² **crece** con τ, y tiene explicación: con τ_act muy grande el actuador filtra
tanto que la entrada efectiva se vuelve casi constante, y entonces `Δf ≈ -P + cte`, que sí
es función de `P`. El caso genuinamente difícil es el **intermedio** (`τ_act` comparable al
tiempo de permanencia del APRBS, 2–8 ms): ahí el estado del actuador es una variable
independiente de verdad y el R² se cae a 0.34.

---

## 4. Recomendación: cuáles dos entradas

Interpreto "dos entradas diferentes" como dos puntos de inyección **de naturaleza distinta**,
que además activan las dos componentes de `g_φ`. Mi recomendación:

### Entrada 1 — lado del **estado**: refractariedad

```python
dI = (1/ti) * (-I + (1 - r*I) * S_i(u_i) - ki)
dE = (1/te) * (-E + (1 - r*E) * S_e(u_e) - ke)
```

Por qué esta y no otra:

1. **No es inventada.** Es literalmente el factor refractario del paper original de
   Wilson & Cowan (1972). La forma reducida que usás lo descarta. O sea: la planta es el
   **WC original** y el modelo es el **WC reducido**. Ese es el mismatch estructural más
   defendible que podés escribir en un informe — no estás perturbando arbitrariamente,
   estás *deshaciendo una simplificación conocida y citable*.
2. **Preserva el reposo exactamente**: en `E=I=0` el factor vale 1 y todo queda igual.
3. **Entra en las dos ecuaciones** → activa las dos salidas de `g_φ`.
4. **Es invertible por el controlador** (ver §7): depende sólo del estado, así que el IMC
   puede cancelarla analíticamente si le pasás `ĝ`.

### Entrada 2 — lado de la **entrada**: canal de actuación optogenético

```python
dP_lag = (P_cmd - P_lag) / tau_act        # dinámica del canal (ChR2)
P_eff  = A * tanh(P_lag / A)              # saturación de la opsina
# idem para Q; el modelo sólo ve P_cmd, nunca P_eff
```

Por qué:

1. Entra **literalmente por las dos entradas** `P` y `Q` — la lectura más directa de lo
   que pediste.
2. Es la incertidumbre más realista de la línea del tutor (control optogenético,
   Martínez 2024): en un experimento real vos comandás intensidad de luz, y lo que la
   neurona recibe pasó por la cinética del canal y satura.
3. Tiene **estado propio** con una escala de tiempo ajustable → es tu perilla de dificultad.
   Con `τ_act` chico o grande es capturable; en el rango intermedio no lo es.
4. Rompe el supuesto de que `P,Q` son perfectamente conocidos, que es el que hoy sostiene
   la identificabilidad de la escala (lo decís vos mismo en el encabezado de
   `train_neural_ode_full.py`). Eso lo vuelve un test duro y honesto.

### Intensidades calibradas

Medí el par combinado. `D_abs` es la deformación absoluta, comparable con el `σ` del ruido
de observación de tus barridos (0.01–0.10):

| `r` | `A` | `τ_act` | D_traj | D_abs | E_max | ¿oscila? | R²_test de `g_φ` |
|---:|---:|---:|---:|---:|---:|:---:|---:|
| — | — | — | 0% | 0 | 0.793 | sí (10) | — |
| 0.05 | 4.0 | 0.5 | ~20% | ~0.02 | 0.75 | sí | — |
| **0.10** | **3.0** | **1.0** | **58%** | **0.065** | 0.62 | sí (12) | **0.654** |
| 0.20 | 2.0 | 1.0 | 74% | 0.082 | 0.44 | sí (11) | 0.825 |
| 0.20 | 1.5 | 2.0 | 87% | 0.096 | 0.23 | sí (11) | 0.913 |

**Punto de operación sugerido: `r=0.10, A=3.0, τ_act=1.0`.** Deformación bien por encima
del ruido (3× el `σ=0.02`), el sistema sigue oscilando con un rango de `E` sano, y `g_φ`
puede capturar ~2/3 del hueco: **suficiente para que el gray-box gane claramente, con un
residuo genuino que deja historia para contar.**

Detalle interesante: el R² **sube** cuando subís `r`, porque la refractariedad (capturable)
pasa a dominar sobre el lag del actuador (no capturable). O sea, la proporción entre las dos
entradas te controla directamente qué fracción del problema es resoluble. Es una perilla
experimental, no un accidente.

---

## 5. Dos advertencias sobre los números

**El retardo dio R²≈0.98 en todo el barrido de τ (0.2 a 8 ms), contra lo esperado.** Un
retardo tiene memoria infinita, debería ser el peor caso. La explicación candidata es que la
trayectoria vive sobre una variedad de baja dimensión forzada por el estímulo, y sobre esa
variedad `E(t-τ)` es reconstruible desde `(I,E,P,Q)`. Si es así, es un artefacto de que los
datos no cubren suficiente espacio de estados, y se rompería con estímulos más ricos o
condiciones iniciales variadas. **No lo daría por bueno sin verificarlo** — pero si el
resultado aguanta, es interesante por sí mismo (un gray-box memoryless absorbiendo retardos).

**Los `D_traj` son grandes incluso con perturbaciones suaves** (r=0.05 → 12%). El sistema es
sensible: la fase de la oscilación se corre y el RMS punto a punto castiga mucho. No hay que
leer "58% de deformación" como "el sistema es irreconocible" — mirá `E_max` y el número de
cruces, que muestran que el régimen se conserva.

---

## 6. ⚠️ La trampa que puede romper la tesis del proyecto

Esto es lo más importante para vos, porque toca el resultado central que ya tenés
("identificamos los 10 parámetros con <1.14% de error").

`g_φ` toma **exactamente los mismos argumentos** `(I,E,P,Q)` que el backbone y suma
directamente sobre `dI,dE`. Es un aproximador universal sobre el mismo dominio. Entonces:

> Para *cualquier* `θ̂` equivocado existe un `g` que hace que el campo total sea exacto.
> Con `g_φ` libre, **los 10 parámetros dejan de ser estructuralmente identificables.**

El síntoma esperado: activás la corrección, el MSE mejora, y el error paramétrico **empeora**.
La red aprende a compensar parámetros malos en vez de aprender la física faltante. Es la
patología clásica de los UDE y es exactamente la que puede dinamitar el mensaje del trabajo.

Y acá está lo bueno: **esto conecta directo con la línea Fisher+SVD que ya tenés armada**
(`scripts/fisher_identifiability.py`). Las direcciones singulares débiles que ya
identificaste (σ₁₀ = `wII`, σ₉ = acople `ae–te–wEE`) son *justamente* por donde `g_φ` va a
entrar a robar. Predecir eso con la FIM y confirmarlo con el experimento es el mismo bucle
"predecir → confirmar" que ya te funcionó y que figura como aporte #2 en
`docs/novedad_trabajo_relacionado.md`.

### Los cuatro remedios, de menos a más fuerte

1. **Regularización de norma** `λ‖g‖²` — el prior de "corrección mínima". Es el estándar en
   UDE. Barato, y da una curva `λ` vs error paramétrico.
2. **Restringir los argumentos de `g`.** Si `g = g(I,E)` sin `P,Q`, toda la dependencia del
   estímulo queda forzada dentro del backbone, y como `P,Q` son conocidos y ricos, eso ancla
   la escala de los parámetros. Es la restricción más barata y probablemente la más efectiva.
3. **Proyección ortogonal** — penalizar la componente de `g` a lo largo de
   `span{∂f_WC/∂θ_j}`. Es el remedio principista: le prohibís a la red moverse en las
   direcciones que un cambio de parámetros ya puede explicar. Requiere calcular las
   sensibilidades (que ya calculás para la FIM).
4. **Entrenamiento en dos etapas / alternado** — primero `θ` con `g=0`, después congelás `θ`
   y ajustás `g`, y opcionalmente iterás.

Yo empezaría por (2)+(1) juntos, que son media hora de trabajo, y dejaría (3) como el
experimento de fondo — porque *(3) es el que da material publicable*.

---

## 7. Efecto sobre el controlador (OE3)

El IMC de `closed_loop.py` hace linealización por realimentación: cancela el acoplamiento con
`Q = uq - (wIE·E - wII·I)`. Si la planta real tiene un término extra, la cancelación queda
incompleta y aparece un residuo de seguimiento.

Con un gray-box podés hacer algo que hoy no podés: **meter `ĝ` en la cancelación**. Y acá se
paga la decisión de diseño de la §6, remedio 2:

- Si `g = g(I,E)` (sólo estado) → la inversión es **exacta y explícita**: le restás `ĝ(I,E)`
  al objetivo antes de invertir la sigmoidea. Una línea de código.
- Si `g = g(I,E,P,Q)` → la ecuación de cancelación se vuelve **implícita en `P,Q`** y
  necesitás un punto fijo o un Newton por paso. Mucho más caro y frágil.

O sea: restringir `g` a depender sólo del estado te compra identificabilidad **y** un
controlador invertible. Es la misma decisión pagando dos veces. Es el argumento más fuerte
del documento a favor de esa restricción.

Esto da la comparación de tres vías, que es el resultado end-to-end natural:

| Controlador construido con… | contra la planta perturbada |
|---|---|
| θ verdaderos + física verdadera | cota inferior (ideal inalcanzable) |
| θ̂ del white-box (ignora el hueco) | ¿cuánto degrada el mismatch? |
| θ̂ + `ĝ` del gray-box | ¿cuánto recupera el gray-box? |

---

## 8. Roadmap

Ocho fases. Cada una tiene un criterio de corte explícito, para no seguir de largo si algo
no da.

### F0 — Infraestructura de perturbaciones · *~1 día*

- Nuevo `src/wilson_cowan/uncertainty.py`: perturbaciones como objetos componibles con
  interfaz uniforme (`n_extra_states`, `augment_rhs`, `metadata`).
- `WilsonCowan.__init__` acepta `perturbation=None`. Con `None`, **camino de código idéntico
  al actual**.
- `generate_dataset` guarda los metadatos de la perturbación en el `.npz`, y —cuando hay
  actuador— guarda `P_cmd` **y** `P_eff` por separado. *El entrenamiento sólo puede usar
  `P_cmd`; `P_eff` es únicamente para diagnóstico.* Este es el error más fácil de cometer y
  el que invalidaría todo el experimento.
- Test de regresión: `perturbation=None` reproduce el `multi_dataset.npz` actual bit a bit.

> ⚠️ Dos trampas numéricas concretas: los retardos **no funcionan con `solve_ivp`** (necesitan
> paso fijo con buffer de historia) y el ruido de proceso **no se puede integrar con RK45
> adaptativo** (necesita Euler-Maruyama; el paso adaptativo sobre un SDE da resultados
> silenciosamente incorrectos). Si arrancás con refractariedad + actuador, ninguna de las dos
> te toca — otra razón para empezar por ahí.

**Corte:** el test de regresión pasa.

### F1 — Implementar y caracterizar el par elegido · *~1 día*

- Refractariedad y actuador, con la perilla `ε` calibrada de §4.
- Figura de caracterización: retratos de fase nominal vs perturbado, y el mapa 2D de `Δf`
  sobre el plano `(I,E)` — que después es la referencia contra la que se compara `ĝ`.
- Regenerar `multi_dataset` en tres versiones: `ε=0`, `ε` bajo, `ε` nominal.

**Corte:** el sistema sigue oscilando y `D_abs` está por encima del `σ` de ruido que usás.

### F2 — El costo de la rigidez (baseline sin gray-box) · *~1 día*

Antes de encender `g_φ` hay que **demostrar que el problema existe**. Correr
`train_neural_ode_full.py` sin modificar sobre los datos perturbados, barriendo `ε`.

- Curva: error paramétrico y MSE open-loop de test vs `ε`.
- Hipótesis a chequear: el white-box no explota, **sesga** los parámetros — y apuesto a que
  los sesga sobre todo en las direcciones débiles que ya te marcó la FIM.

**Corte:** hay degradación medible y monótona en `ε`. Si el white-box aguanta perfecto, la
perturbación es demasiado suave: volvé a F1 y subí `ε`.

### F3 — Encender `g_φ` y enfrentar la trampa de identificabilidad · *~2-3 días* ⭐

La fase central. `train_neural_ode_gray.py`, con las cuatro variantes de §6:

| Variante | `g` depende de | Regularización |
|---|---|---|
| A (ingenua) | `I,E,P,Q` | ninguna |
| B | `I,E` | ninguna |
| C | `I,E` | `λ‖g‖²`, barriendo λ |
| D | `I,E` | proyección ortogonal a `∂f/∂θ` |

Reportar siempre **las dos** métricas juntas — MSE y error paramétrico — porque el hallazgo
esperado es que se muevan en direcciones opuestas.

**Corte / resultado esperado:** la variante A mejora el MSE y empeora `θ̂` (la trampa,
confirmada). Alguna de C/D mejora **las dos** respecto del white-box de F2. Si ninguna lo
logra, el mensaje del trabajo cambia y hay que replantear antes de seguir.

### F4 — FIM del híbrido · *~1-2 días*

Extender `fisher_identifiability.py` para calcular la FIM de `θ` **en presencia de** `g`.

- ¿Cómo cambia el número de condición al agregar la corrección?
- ¿Las direcciones por donde `g` roba información son las mismas σ₉, σ₁₀ que ya conocés?
- Predecir *sin entrenar* qué parámetro se va a degradar, y confirmarlo con F3.

Es el mismo bucle predecir→confirmar que ya te funcionó con `wII`. **Es la fase con más
retorno científico por hora invertida.**

### F5 — Recuperación funcional · *~1 día*

Como conocés el simulador, conocés el `Δf` verdadero. Entonces podés preguntar algo que casi
ningún trabajo de UDE puede: **¿la red aprendió la física correcta, o sólo algo que ajusta?**

- Mapa 2D: `ĝ(I,E)` vs `Δf` verdadero sobre el plano de estados, con su R².
- **Dominio de validez**: R² dentro vs fuera de la región visitada en entrenamiento. Es lo
  que determina si el modelo sobrevive al lazo cerrado, donde el controlador va a llevar al
  sistema a estados que el dataset no cubrió.

**Corte:** R² > 0.8 dentro del dominio visitado.

### F6 — Lazo cerrado con cancelación gray-box · *~2 días*

- Extender `IMCController` para aceptar un `ĝ` opcional en la cancelación (trivial si en F3
  ganó una variante con `g=g(I,E)`).
- La tabla de tres vías de §7, sobre la planta perturbada.
- Métrica: RMSE de seguimiento de las referencias theta-gamma.

**Corte:** el gray-box recupera una fracción sustancial de la degradación que el white-box
sufre. Esto es lo que cierra la historia end-to-end.

### F7 — Límites, controles negativos y escritura · *~2 días*

- Familias 🟡 y ❌ como controles: adaptación con `τ_a=30 ms` (memoria genuina), ruido de
  proceso (piso irreducible). Muestran **dónde deja de funcionar** el gray-box memoryless.
- La perilla `τ_a`: R² de `g_φ` vs `τ_a` — la curva de la §3, que resume en un gráfico
  cuándo hace falta darle memoria a la corrección.
- Doc de resultados + figuras, en el formato de los otros `docs/resultado_*.md`.

### Camino crítico

```
F0 → F1 → F2 → F3 → F4 ─┐
                   └→ F5 ┴→ F6 → F7
```

F4 y F5 son independientes entre sí y pueden ir en paralelo. **F3 es el cuello de botella
real**: si la trampa de identificabilidad no se puede domar, el resto del roadmap cambia de
forma. Todo lo anterior a F3 es infraestructura y se puede hacer rápido.

---

## 9. Qué agrega esto al aporte del trabajo

Mirando `docs/novedad_trabajo_relacionado.md`, esto no es una rama lateral: extiende los tres
aportes que ya tenés declarados.

| Aporte actual | Cómo lo extiende |
|---|---|
| Identificar los 10 parámetros de WC | …**bajo mismatch estructural**, que es el caso realista. Hoy el resultado vive en el mundo cómodo donde tu modelo es exacto. |
| Fisher+SVD predice qué es identificable | …**y predice por dónde la corrección neuronal roba identificabilidad.** Un uso nuevo de la misma herramienta. |
| La fragilidad no se propaga al control | …**y el gray-box recupera lo que el mismatch degrada**, con cancelación aumentada. |

Y desactiva el "qué NO hace" que vos mismo le anotaste a El-Gazzar & van Gerven 2025 en la
tabla de related work: *"review; sin resultado empírico ruido/control"*. Con F3–F6 tendrías
exactamente ese resultado empírico faltante — con la ventaja, que casi nadie tiene, de
**conocer el Δf verdadero** y poder medir si la red aprendió la física correcta y no
simplemente algo que ajusta.

---

## 10. Conexiones

- **Evidencia reproducible de este documento:** `scripts/uncertainty_probe/` (ver su README).
  Las 9 familias están implementadas ahí; `probe_learnable.py` es el que produce la columna R²_test.
- Código: `src/wilson_cowan/model.py` · `src/neural_ode/dynamics.py` · `src/neural_ode/closed_loop.py`
- Scripts: `scripts/gen_multi_dataset.py` · `scripts/train_neural_ode_full.py` · `scripts/fisher_identifiability.py`
- Docs: `novedad_trabajo_relacionado.md` · `identificacion_completa_neural_ode.md` · `robustez_ruido_identificacion_completa.md`
