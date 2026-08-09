# Incertidumbre dinámica y gray-box: manual completo

> **Qué es este documento.** La explicación entera de la extensión gray-box del proyecto:
> qué es la incertidumbre dinámica, dónde se aplica, cómo, en qué momento, qué cambia en la
> red, qué se midió y qué dio. Está escrito para leerse de arriba a abajo sin conocimiento
> previo de esta parte.
>
> Documento hermano: `incertidumbre_dinamica_graybox.md` (el catálogo de opciones y el
> roadmap que dio origen a esto). Este documento es **lo que efectivamente se construyó y
> se midió**.

---

## Resumen ejecutivo

### Qué queríamos hacer

El proyecto identifica los 10 parámetros de Wilson-Cowan con una Neural ODE y lo hace muy
bien: ~1 % de error. Pero ese resultado vive en un mundo cómodo, y conviene ver por qué. El
simulador que genera los datos y el modelo que se entrena **son exactamente las mismas
ecuaciones**. Estamos resolviendo "dado que el modelo es perfecto, encontrale los
parámetros". En un experimento real tu modelo del cerebro siempre está incompleto.

Por eso `GrayBoxWC` tenía desde el principio un término de corrección neuronal `g_φ` que
**nunca se usó** en ningún experimento sintético: no tenía nada que aprender.

El objetivo fue entonces: **meterle al simulador física que el modelo no contempla**, para
que la corrección tenga trabajo real, y medir qué pasa.

### Cómo se aplica la incertidumbre (la idea en un párrafo)

Se le agrega al **simulador** —no a la red— física real que el modelo omite. Concretamente
dos cosas: la **refractariedad**, que es el factor `(1-r·x)` del Wilson-Cowan original de
1972 que la versión reducida descarta; y un **canal de actuación optogenético no ideal**,
donde la luz que comandás no es la que llega a la neurona (filtra y satura).

El reparto de papeles queda así:

```
   simulador + perturbación   =  "el cerebro real"      (tiene toda la física)
              ↓ genera
        el .npz de datos      =  lo que medís           (sólo t, I, E, P, Q)
              ↓ entrena
   backbone Wilson-Cowan      =  tu hipótesis           (los 10 parámetros)
        +  g_φ                =  lo que no explica      (lo tiene que descubrir)
```

**La red nunca ve la perturbación.** No sabe que existe, ni de qué tipo es, ni ve sus
estados internos. Ve trayectorias `(I,E)` y el estímulo `(P,Q)` que vos comandaste. Todo lo
que no cierre con Wilson-Cowan puro tiene que salir de `g_φ`.

Se aplica **en todo instante**, no es un golpe en un momento dado: es un término permanente
del campo vectorial, porque es física que siempre estuvo ahí. Y **no es aleatoria**: `g_φ`
es determinista, así que sólo puede aprender lo reproducible. (El ruido está en el catálogo,
pero como control negativo: R² = 0.04, nadie lo puede aprender.)

Una perilla, `ε`, gradúa cuánta incertidumbre hay. Con `ε=0` la planta vuelve a ser
Wilson-Cowan puro; con `ε=1`, el hueco vale un tercio del campo.

### ¿Copia bien el comportamiento la Neural ODE?

Ésta es la pregunta práctica, y la medimos aparte de todo lo demás. La prueba es exigente:
se le da **sólo el estado inicial y el estímulo**, y tiene que generar los 200 ms enteros por
su cuenta, sin volver a mirar el dato real nunca (rollout *open loop*), sobre estímulos que
no vio al entrenar. El error va normalizado al rango de la señal:

| situación | error de trayectoria | correlación con la real |
|---|---|---|
| **sin hueco** (modelo = planta) | **2.0 %** | **0.99** |
| con hueco, white-box | 15.1 % | 0.82 |
| con hueco, gray-box `g(I,E)` | 14.5 % | 0.85 |
| con hueco, gray-box regularizada | 14.0 % | 0.85 |

**La respuesta corta: sí cuando el modelo es correcto, y sólo a medias cuando no lo es.**

Sin hueco la copia es casi perfecta: 2 % de error tras 200 ms rodando sola, correlación 0.99.
Eso confirma que la maquinaria funciona.

Con hueco la reproducción cae a ~15 %: **sigue la forma** de la oscilación (correlación 0.82,
los picos y valles están donde corresponde) pero se desfasa y desajusta la amplitud. Serviría
para entender el sistema, no para predecir con precisión.

Y el dato más informativo: **la corrección neuronal recupera casi la mitad del error de los
parámetros pero casi nada del error de trayectoria** (15.1 % → 14.0 %). O sea que ayuda a
identificar, no a predecir. La razón está en la sección 14 y es el hallazgo central.

### Qué se encontró

Cinco resultados, en orden de importancia.

1. **Dos tercios del hueco son indistinguibles de mover parámetros.** Ésta es la causa de
   todo lo demás. Si la física que falta "se parece" a un cambio de `θ`, entonces el
   white-box la absorbe deformando los parámetros (de 1 % a 59 % de error), y el reparto
   entre `θ` y `g` deja de ser único.

2. **La corrección es útil o dañina según haya hueco o no** — y el ajuste no te avisa. Con
   hueco recupera el 48 % del error paramétrico; sin hueco lo empeora **17 veces** (de 1.05 %
   a 18 %). En los dos casos el MSE mejora. Es la trampa en su forma pura.

3. **Ayuda sin aprender la física.** La corrección aprendida tiene la magnitud correcta
   (0.0170 contra 0.0172 del término verdadero) pero R² ≈ 0 contra la física real: es ~90 %
   redundante con los parámetros.

4. **Y por eso no sirve para cancelar en el controlador.** Meterla en la linealización
   recupera el lazo de E casi hasta el óptimo pero destruye el de I. Ajustar bien y servir
   para cancelar son dos requisitos distintos.

5. **La FIM predijo por dónde iba a fallar, y acertó.** El 93 % de lo que la corrección le
   roba a los parámetros va por σ₁₀ = `wII`, exactamente la dirección que el análisis
   Fisher+SVD del proyecto ya marcaba como la más débil.

### Estado actual (28-jul-2026)

**Terminado y verificado.** Las ocho fases del roadmap (F0–F7), con 38 tests que pasan y todo
reproducible con `bash scripts/run_uncertainty_all.sh`. Una auditoría adversarial de 51
agentes sobre el código encontró dos bugs que cambiaban conclusiones, ya corregidos.

**Tres frentes abiertos, avanzados después del roadmap:**

- **El integrador quedó absuelto** (sección 20). Se sospechaba que el error de `solve_ivp`
  contaminaba los barridos de ruido; se midió y **no**: con σ=0.01 los tres integradores
  coinciden dentro del 3 %. No hay que rehacer nada.
- **El diseño de estímulos funciona** (sección 21): la degeneración baja del 71 % al 59.5 %
  eligiendo bien el estímulo, con la misma amplitud. Pero apareció un matiz importante — lo
  que más rompe la degeneración no es un estímulo bueno sino un conjunto **diverso**.
- **La corrección estructurada** (sección 22) es identificable **si la forma está completa**
  (recupera `sat` con 1.3 % de error), y converge a valores equivocados si le falta una
  componente. Es la misma lección de degeneración, un nivel más abajo.

---

## Parte I — Qué problema resuelve

### 1. El punto de partida

El proyecto identifica los 10 parámetros de Wilson-Cowan con una Neural ODE y lo hace muy
bien: con datos limpios, error medio del orden del 1%. Pero ese resultado vive en un mundo
cómodo, y conviene ver por qué.

El dato se genera con `src/wilson_cowan/model.py`. El modelo que se entrena es el backbone
de `src/neural_ode/dynamics.py`. **Son exactamente las mismas ecuaciones.** El simulador no
tiene ni un gramo de física que el modelo no conozca.

Eso quiere decir que el problema que se está resolviendo es: *dado que el modelo es
perfecto, encontrale los parámetros*. Es un problema legítimo y difícil, pero no es el
problema real. En un experimento de verdad, tu modelo del cerebro **siempre** está
incompleto.

Y tiene una consecuencia concreta en el código: `GrayBoxWC` tiene un término de corrección
neuronal `g_φ` desde el principio, con el flag `use_correction`, y **nunca se usó** en
ningún experimento sintético. No por olvido: porque no tenía nada que aprender. Sumarle una
red a un modelo que ya es exacto sólo puede empeorar las cosas.

### 2. Qué es la incertidumbre dinámica

Es física real que la planta tiene y el modelo no contempla.

No es ruido. No es un error de medición. No es una perturbación externa que golpea al
sistema. Es un **término que forma parte del campo vectorial de la planta**, presente en
todo instante, que el modelo simplemente no incluye porque su estructura es una
simplificación.

El ejemplo canónico —y el que usa este proyecto— es la **refractariedad**. Wilson y Cowan,
en 1972, derivaron sus ecuaciones con un factor `(1 - r·x)` multiplicando la sigmoidea: una
neurona que acaba de disparar no puede volver a disparar enseguida, así que la fracción de
la población *disponible* para disparar no es 1 sino `(1 - r·x)`. La forma reducida que usa
todo el mundo —y que usa este proyecto— **descarta ese factor**.

Entonces la jugada es honesta y citable: **la planta pasa a ser el Wilson-Cowan original, y
el modelo sigue siendo el reducido.** No estamos inventando una perturbación arbitraria;
estamos deshaciendo una simplificación conocida.

### 3. Para qué sirve

Tres cosas, en orden de importancia.

**Le da trabajo real a `g_φ`.** El hueco entre planta y modelo es exactamente lo que la
corrección tiene que aprender. Sin hueco, el gray-box es decoración.

**Convierte el resultado en uno defendible.** "Identificamos 10 parámetros cuando el modelo
es exacto" es mucho más débil que "identificamos 10 parámetros bajo mismatch estructural, y
medimos cuánto cuesta".

**Abre una pregunta que casi nadie puede contestar.** Como nosotros fabricamos el simulador,
conocemos el término faltante exacto. Entonces podemos preguntar: *¿la red aprendió la
física correcta, o simplemente algo que ajusta?* Prácticamente ningún trabajo de UDE puede
responder eso, porque no conoce la verdad.

---

## Parte II — Cómo funciona, en detalle

### 4. Dónde se aplica exactamente

En el simulador, dentro del campo vectorial. Concretamente en
`WilsonCowan.perturbed_field()` de `src/wilson_cowan/model.py`.

El reparto de roles es el corazón del diseño:

| Pieza | Qué representa | Qué sabe |
|---|---|---|
| `src/wilson_cowan/model.py` + perturbación | **el cerebro real** | toda la física |
| el `.npz` generado | **lo que medís** en el experimento | sólo `t, I, E, P, Q` |
| backbone de `GrayBoxWC` | **tu hipótesis** de modelo | los 10 parámetros |
| `g_φ` | lo que la hipótesis no explica | lo tiene que descubrir |

**La red nunca ve la perturbación.** No sabe que existe, no sabe de qué tipo es, no ve sus
estados internos. Ve trayectorias `(I, E)` y el estímulo `(P, Q)` que vos comandaste. Todo
lo que no cierre con Wilson-Cowan puro tiene que salir de `g_φ`.

### 5. Los cinco lugares donde puede entrar

La ecuación base, para la población inhibitoria:

```
u_i = wIE·E - wII·I + Q - thetai          (1) entrada total
dI  = (1/ti)·( -I + S(u_i) - ki )         (2) sigmoidea y relajación
```

Una perturbación puede meterse en cinco lugares distintos, y en `uncertainty.py` cada uno
es un "gancho" (hook) para poder **combinar varias a la vez**:

| Gancho | Dónde entra | Quién lo usa |
|---|---|---|
| `inputs()` | el estímulo que llega ≠ el comandado (`Q → Q_eff`) | actuador optogenético |
| `drive()` | se suma una corriente a `u_i`/`u_e` | adaptación, población oculta |
| `gains()` | multiplica la salida de la sigmoidea | **refractariedad** |
| `weights()` | cambia los pesos sinápticos | depresión sináptica, deriva |
| `deriv()` | suma directo a `dI`/`dE` | ruido de proceso |

Además una perturbación puede tener **estados ocultos propios** (`n_extra`): la corriente de
adaptación, el recurso sináptico, el estado del actuador. Esos estados no se miden y no son
argumentos de `g_φ` — y son justamente los que hacen que una corrección sin memoria se quede
corta. La sección 16 mide exactamente cuánto: para el par elegido, el techo de una corrección
que sólo ve el estado es R² = −0.11, o sea que el término faltante **no es** función del
estado.

### 6. En qué momento se aplica

**En todo instante, siempre.** No es un evento, ni un golpe en `t=100`, ni algo que se
enciende a mitad del experimento. Es física que siempre estuvo ahí.

Hay una propiedad de diseño que se cuidó en todas las familias y que se testea
explícitamente (`tests/test_uncertainty.py::test_reposo_es_equilibrio`): **todas preservan
el reposo `E=I=0` como equilibrio**. Con estímulo nulo el sistema sigue durmiendo en cero,
exactamente igual que antes.

Eso importa porque la perturbación se "enciende sola" cuando el sistema se activa —depende
del estado— y no corre silenciosamente el punto de operación. Si no se cuidara, cambiaría
el equilibrio y rompería la convención de `ke`/`ki` sin que nadie se dé cuenta.

La única excepción es a propósito: el **ruido de proceso** sí saca al sistema del reposo,
porque no depende del estado. Está en el catálogo justamente como control negativo, y hay
un test que verifica ese contraste.

### 7. ¿Es aleatorio? No — y es lo más importante de entender

Hay tres cosas que se confunden todo el tiempo:

| | Qué es | Dónde entra | ¿`g_φ` lo aprende? |
|---|---|---|---|
| **Ruido de observación** (`noise_std`, ya existía) | medís mal | *después* de integrar, no toca la dinámica | No, ni debe |
| **Ruido de proceso** (OU) | fluctuación real de la población | *dentro* del integrador, cambia la trayectoria | **No** — medido R² = 0.04 |
| **Incertidumbre estructural** | física que tu modelo omite | dentro del campo, determinista | **Sí** — medido R² = 0.97 |

La razón es simple: `g_φ` es una función determinista de `(I,E)` o `(I,E,P,Q)`. Sólo puede
aprender lo **reproducible**. Si el sistema vuelve a pasar por el mismo estado con el mismo
estímulo, la corrección devuelve el mismo valor, siempre. El ruido, por definición, no
cumple eso.

Esto no es una opinión: se midió entrenando la arquitectura exacta de `g_φ` contra el Δf
verdadero de cada familia. Ruido de proceso: R² = 0.04. Refractariedad: R² = 0.97.

**Conclusión de diseño: la incertidumbre que activa el gray-box tiene que ser determinista
y con sentido físico.** El ruido sirve como piso irreducible, no como mecanismo.

### 8. Las dos entradas elegidas

**Entrada 1 — lado del estado: refractariedad.**

```python
dI = (1/ti)·( -I + (1 - r·I)·S_i(u_i) - ki )
dE = (1/te)·( -E + (1 - r·E)·S_e(u_e) - ke )
```

Es el término del Wilson-Cowan de 1972. Preserva el reposo exactamente (en `E=I=0` el factor
vale 1). Entra en las dos ecuaciones, así que activa las dos salidas de `g_φ`. Y es
capturable: R² = 0.97, porque `Δf` es función pura del estado.

**Entrada 2 — lado de la entrada: canal de actuación optogenético.**

```python
dP_lag = (P_cmd - P_lag) / tau_act      # cinética del canal ChR2
P_eff  = A · tanh(P_lag / A)            # saturación de la opsina
```

Entra literalmente por las dos entradas `P` y `Q`. Es la incertidumbre más realista de la
línea del tutor: vos comandás intensidad de luz, y lo que la neurona recibe pasó por la
cinética del canal y saturó. Y **rompe el supuesto de que `P,Q` son perfectamente
conocidos**, que es el que hoy ancla la escala de los parámetros.

Tiene estado propio (el lag), así que es la parte difícil: parcialmente no capturable.

**La perilla `ε`.** Una sola, que escala las dos: `default_uncertainty(eps)`. Con `eps=0`
devuelve la identidad; con `eps=1`, `r=0.10`, `A=3.0`, `tau_act=1.0 ms`. El actuador escala
al revés de lo que uno esperaría (`sat = 3/eps`, `tau = 1·eps`) para que al bajar `ε` el
canal se acerque al ideal: satura cada vez más tarde y filtra cada vez menos.

> **Con una salvedad que conviene conocer** (la encontró la auditoría del código): bajar
> `tau_act` vuelve *rígida* la ecuación del actuador, y cuando se acerca al paso del
> integrador el RK4 de paso fijo se desestabiliza. O sea que el límite ideal **no** se
> alcanza bajando `ε` de forma continua. Por eso `tau_act` tiene un piso de 0.15 ms, la
> función avisa si se le pide menos, y para "sin perturbación" hay que usar `ε=0`, que
> devuelve `NoPerturbation`. El rango validado es `ε ∈ [0.25, 2.0]`.

---

## Parte III — Qué cambia en la red

### 9. El modelo pasa de blanco a gris

El modelo entrenado es:

```
ẋ = f_WC(x, P, Q; θ)  +  g_φ(x [, P, Q])
    └─ backbone ─┘        └─ corrección ─┘
    10 parámetros         MLP 2 capas, tanh, 32 ocultas
    interpretables        no interpretable
```

En `dynamics.py` se separó explícitamente en tres métodos —`backbone()`, `g_out()` y
`forward() = backbone + g_out`— porque hacen falta por separado: para medir cuánto aporta
cada parte, para calcular las sensibilidades `∂f/∂θ`, y para que el controlador pueda
cancelar `ĝ`.

### 10. ⚠️ La trampa: `g_φ` compite con los parámetros

Este es el punto que puede romper la tesis del proyecto, y merece leerse despacio.

`g_φ` es un aproximador universal sobre el mismo dominio que el backbone. Entonces:

> Para **cualquier** `θ̂` equivocado existe un `g` que hace que el campo total sea exacto.
> Con `g` libre, los 10 parámetros dejan de ser estructuralmente identificables.

El síntoma esperado era traicionero: **el MSE mejora y el error paramétrico empeora**. Si uno
mira sólo el MSE, parecería que el gray-box es un éxito cuando en realidad rompió lo único
que el proyecto quería lograr. Por eso todos los experimentos reportan las dos métricas
juntas, siempre.

> **Adelanto: la realidad resultó más interesante que esta predicción.** Bajo mismatch real
> la corrección mejora *las dos* métricas (sección 15), porque le saca al white-box la
> obligación de deformar `θ`. Pero lo hace siendo >90 % redundante con los parámetros, y sin
> aprender la física verdadera (sección 16). La trampa existe —se mide con
> `frac_redundante`— pero no se manifiesta como yo esperaba. La causa está en la sección 14.

Se implementaron cuatro variantes para atacarlo:

| Variante | `g` depende de | Regularización | Idea |
|---|---|---|---|
| **A** | `I,E,P,Q` | ninguna | la ingenua — muestra la patología |
| **B** | `I,E` | ninguna | el estímulo sólo lo puede explicar `θ` |
| **C** | `I,E` | `λ‖g‖²` | prior de "corrección mínima" (el estándar UDE) |
| **D** | `I,E` | `λ‖proyección‖²` | prohibirle a `g` ser redundante |

**Por qué B ayuda:** al no ver el estímulo, toda la dependencia de `P,Q` queda forzada dentro
del backbone. Como `P,Q` son conocidos y ricos (APRBS, PRBS, chirp, theta-gamma), eso ancla
la escala de los parámetros.

**Por qué D es la variante principista.** C penaliza que `g` sea *grande*. D penaliza que `g`
sea *redundante*, que es lo que realmente molesta. La construcción:

1. Se calculan las sensibilidades `S = ∂f_backbone/∂θ` en `N` puntos del dataset
   (`backbone_sensitivities`, diferencias centradas sobre los 10 parámetros).
2. Se apila `g` evaluada en esos `N` puntos como un vector en `R^{2N}`, y `S` como una
   matriz `(2N × 10)`.
3. Se busca el `δθ` que mejor imita a `g`: `c = argmin ‖S·c − G‖`.
4. Se penaliza `‖S·c‖²` — la parte de `g` que un cambio de parámetros ya podía explicar.

**El detalle que hace que funcione:** la proyección es en el **espacio de funciones**, no
punto a punto. Punto a punto no serviría: 10 columnas en `R²` generan todo `R²`, y la
penalización mataría `g` por completo. Al proyectar sobre el dataset entero, lo que se
prohíbe es que `g` imite un cambio de parámetros *de forma consistente en todos lados*, que
es exactamente la redundancia que rompe la identificabilidad.

De ahí sale además la métrica más informativa de todo el trabajo:

> **`frac_redundante`** = qué fracción de lo que hace `g_φ` lo podría haber hecho un cambio
> de `θ`. Cerca de 1: la red está compitiendo con los parámetros. Cerca de 0: está
> aportando física genuinamente nueva.

### 11. Qué cambia en el controlador

El IMC de `closed_loop.py` hace linealización por realimentación: cancela el acoplamiento
usando los pesos. Con un gray-box se puede meter `ĝ` en la cancelación.

La derivación es corta. El controlador quiere lograr `dI = (1/ti)(-I + Ulti_I)`. El modelo
aprendido es `dI = (1/ti)(-I + σ_i - ki) + ĝ_I`. Igualando:

```
σ_i = Ulti_I + ki − ti·ĝ_I
```

O sea: **alcanza con correr el objetivo de la sigmoidea por `−ti·ĝ_I`.** Una línea.

Y acá se cobra la decisión de la sección 10: esto es exacto y explícito **sólo porque `ĝ`
depende únicamente del estado**. Si dependiera también de `P,Q`, la ecuación quedaría
implícita en `P,Q` y habría que resolver un punto fijo en cada paso de integración.

> Restringir `g` a depender sólo del estado compra identificabilidad **y** un controlador
> invertible. Es la misma decisión pagando dos veces.

Esa era la teoría al diseñar, y las dos mitades se cumplieron: `g(I,E)` efectivamente da el
menor error paramétrico (sección 15) y la cancelación efectivamente es exacta (verificado a
3e-9 cuando la actuación no satura). Lo que no se cumplió es que **sirviera**: la
restricción también le impide representar la parte del mismatch que entra por el actuador,
que depende de `P,Q` y de un estado oculto. El resultado es que la cancelación empeora el
control (sección 17). Es una tensión de diseño real, no un bug, y está discutida en la
sección 24.

---

## Parte IV — Resultados

### 12. F1 · La perturbación está bien calibrada

La perilla `ε` produce una degradación suave, monótona, bien por encima del ruido y sin
destruir el régimen del sistema:

| ε | deformación de la trayectoria | veces el σ del ruido | \|Δf\|/\|f_WC\| | E_max | cruces por la media |
|---|---|---|---|---|---|
| 0 | 0 % | 0 | 0 % | 0.878 | 21 |
| 0.25 | 11.7 % | 1.0× | 9.5 % | 0.855 | 21 |
| 0.5 | 27.5 % | 2.3× | 18.2 % | 0.832 | 21 |
| **1.0** | **54.0 %** | **4.6×** | **34.2 %** | 0.782 | 21 |
| 1.5 | 66.9 % | 5.7× | 45.7 % | 0.731 | 20 |
| 2.0 | 77.5 % | 6.6× | 52.6 % | 0.676 | 19 |

En el punto nominal el término faltante vale un tercio del campo de Wilson-Cowan y la
deformación es casi 5 veces el ruido de observación que el proyecto usa en sus barridos.
El número de cruces se mantiene en 21: **el ciclo límite se deforma, no desaparece.**

> **Hallazgo lateral, pero que conviene corregir en el proyecto.** El integrador que usan
> los scripts actuales (`solve_ivp` con `rtol=1e-3`) tiene un error de **1.5e-2** contra una
> referencia de alta precisión. Eso es del mismo orden que el `σ=0.01–0.02` de los barridos
> de robustez al ruido, y es un error *sistemático*, no aleatorio. El RK4 de paso fijo del
> camino nuevo es ~30× más preciso. Por eso el baseline `ε=0` se genera con
> `NoPerturbation` y no con `perturbation=None`: si no, la comparación mezclaría el efecto
> de la perturbación con un cambio de integrador. Está fijado en
> `tests/test_uncertainty.py::test_paso_fijo_es_mas_preciso_que_el_camino_historico`.

### 13. F2 · Lo que cuesta la rigidez

Identificación de los 10 parámetros con el white-box de siempre, sobre datos cada vez más
perturbados:

| ε | error medio de θ | error máximo | peor parámetro | MSE open-loop (test) |
|---|---|---|---|---|
| 0 | **1.05 %** | 3.67 % | wII | 9.1e-4 |
| 0.25 | 13.5 % | 70.5 % | wII | 8.3e-3 |
| 0.5 | 27.7 % | 127.8 % | wII | 8.3e-3 |
| 1.0 | 59.5 % | 187.5 % | wII | 1.1e-2 |
| 1.5 | 74.8 % | 199.5 % | wII | 4.3e-2 |
| 2.0 | 93.6 % | 196.2 % | wII | 2.7e-3 |

Dos cosas para destacar.

**El caso ε=0 reproduce el resultado conocido del proyecto** (1.05 % de error medio). O sea
que la maquinaria nueva no cambió nada del resultado previo: sirve de control positivo.

**El peor parámetro es siempre `wII`.** En los cinco niveles de mismatch, sin excepción. Es
exactamente el parámetro que el análisis Fisher+SVD del proyecto ya señalaba como la
dirección singular más débil (σ₁₀). O sea: *el mismatch estructural degrada preferentemente
la dirección que ya era la más frágil*. Es una confirmación independiente del diagnóstico de
identificabilidad, obtenida por un camino totalmente distinto.

El error por parámetro muestra otro patrón: los pesos sinápticos se rompen mucho más que los
parámetros de forma de la sigmoidea. En ε=1: wII 187 %, wEI 101 %, wIE 68 %, wEE 48 %,
frente a ae 18 %, thetae 19 %, ti 21 %. Tiene sentido — la refractariedad multiplica la
salida de la sigmoidea, y eso se imita mucho más fácil reescalando pesos.

### 14. F4b · La geometría del mismatch — el resultado que explica todo lo demás

Esta medición sólo es posible porque conocemos el simulador. Se proyecta el Δf **verdadero**
sobre el espacio que generan las sensibilidades `∂f/∂θ`:

```
Δf  =  S·δθ*         +      residuo
       imitable              física que NINGÚN ajuste de
       moviendo θ            los 10 parámetros puede dar
```

| ε | \|Δf\| | **imitable moviendo θ** | \|parte imitable\| | \|parte nueva\| |
|---|---|---|---|---|
| 0.25 | 0.0089 | 30.3 % | 0.0049 | 0.0074 |
| 0.5 | 0.0125 | 47.9 % | 0.0086 | 0.0090 |
| **1.0** | 0.0168 | **67.1 %** | 0.0138 | 0.0097 |
| 1.5 | 0.0178 | 82.2 % | 0.0161 | 0.0075 |
| 2.0 | 0.0178 | 91.4 % | 0.0170 | 0.0052 |

En el punto nominal, **dos tercios de la física que falta son indistinguibles de un cambio
de parámetros**. Esto reordena la interpretación de todo lo demás:

1. **El sesgo del white-box no es un fallo del optimizador.** Es la mejor respuesta que un
   modelo rígido puede dar: hay una dirección de parámetros que imita el 67 % del hueco, y
   el ajuste la encuentra. Los 187 % de error en `wII` son el *precio estructural* de la
   rigidez, no un problema numérico.
2. **Pone un techo al gray-box.** Sobre esa fracción imitable, el reparto entre `θ` y `g`
   **no es único**: cualquier división ajusta los datos igual de bien. Eso predice que la
   corrección aprendida puede tener el tamaño correcto sin parecerse al Δf verdadero — que
   es exactamente lo que pasa (sección 16).
3. **Dice cuándo hay que regularizar.** Cuanto más grande es `ε`, más parecido a parámetros
   es el mismatch (91 % en ε=2), y por lo tanto más se justifica restringir `g`.

La dirección de parámetros equivalente en ε=1 está dominada por `wEI` (+2.82), `wEE` (+2.48)
y `wIE` (+1.11) — los pesos, consistente con el patrón de degradación de F2.

#### 14.1 · Y depende del estímulo — que es lo que abre la puerta

La fracción imitable no es una constante del problema: es geometría **de la trayectoria**, y
la trayectoria la fija el estímulo. Midiéndola escenario por escenario en ε=1:

| estímulo | \|Δf\| | imitable por θ |
|---|---|---|
| `aprbs_1` | 0.0223 | **71.4 %** |
| `square_a1.0_f50` | 0.0239 | 72.1 % |
| `poisson_0` / `prbs_0` | 0.035 / 0.020 | 74.8 % |
| `chirp` | 0.0348 | 77.8 % |
| `thetagamma_0` | 0.0119 | 85.2 % |
| `box_a0.8` | 0.0049 | 86.4 % |
| `prbs_1` | 0.0396 | 89.3 % |
| `box_a1.2` | 0.0254 | **93.9 %** |

**Un rango de 22.6 puntos**, y el orden tiene sentido físico: los estímulos ricos (APRBS,
PRBS, cuadrada con amplitudes variadas) dejan el mismatch mucho menos confundible con un
cambio de parámetros que un pulso casi constante. Es la misma lógica de excitación
persistente que el proyecto ya usa para identificar `θ`, pero aplicada a un objetivo nuevo:
*separar* `g` de `θ` en vez de sólo informar sobre `θ`.

Y un detalle que vale la pena: **el valor agrupado (67.1 %) es más bajo que el de cualquier
escenario individual** (71–94 %). No es un error — es que cuando se exige un único `δθ` que
explique *todos* los estímulos a la vez, el problema queda más restringido. O sea que la
diversidad de estímulos ya está rompiendo parte de la degeneración por sí sola, sin haber
optimizado nada. Eso sugiere que optimizar el estímulo a propósito (sección 24-A) tiene
recorrido real.

### 15. F3 · Encender la corrección: qué pasa realmente

Con ε=1, comparando contra el white-box del mismo dataset:

| variante | error medio de θ | error máximo | MSE test | \|g\|/\|f_WC\| | **redundante con θ** |
|---|---|---|---|---|---|
| white-box | 59.5 % | 187.5 % | 1.06e-2 | — | — |
| **A** — `g(I,E,P,Q)` libre | 35.9 % | 98.1 % | 8.51e-3 | 0.579 | 0.918 |
| **B** — `g(I,E)` | 35.6 % | **69.2 %** | 9.63e-3 | 0.477 | 0.941 |
| **C** — `g(I,E)` + `λ‖g‖²`, λ=0.1 | **30.8 %** | 82.1 % | 9.26e-3 | 0.201 | 0.914 |
| **C** — λ=1 | 31.4 % | 81.7 % | 8.87e-3 | 0.069 | 0.903 |
| **D** — ortogonal, λ=1 | 31.1 % | 78.3 % | **8.54e-3** | 0.075 | 0.861 |
| **D** — ortogonal, λ=10 | 44.6 % | 101.4 % | 1.06e-2 | 0.010 | **0.385** |

**La predicción de la trampa era equivocada, y el motivo es interesante.** Yo esperaba que
la corrección mejorara el ajuste y empeorara los parámetros. Pasa lo contrario: mejora las
dos cosas. La razón la da F4b: bajo mismatch, el white-box está *obligado* a deformar `θ`
para absorber física que no puede representar. Darle a esa física otro lugar donde ir
**libera** a los parámetros. Con regularización suave el error medio baja de 59.5 % a ~31 %:
**se recupera cerca de la mitad del daño**.

**Pero la corrección es redundante en un ~90 %.** El `frac_redundante` dice que casi todo lo
que hace `g_φ` lo podría haber hecho un cambio de parámetros. Ayuda, pero no por la razón
que uno querría: no está aprendiendo la física faltante, está haciendo un reacomodamiento
parámetro-símil que resulta beneficioso.

**Y ahí aparece el compromiso más informativo de todo el experimento.** La variante D
—penalizar que `g` sea redundante— *funciona* como mecanismo: con λ=10 la redundancia cae de
0.94 a 0.385. Pero el error paramétrico **empeora**, de 31 % a 44.6 %, y la corrección queda
casi anulada (`|g|/|f| = 0.010`).

No es un fallo de la variante: es que **el supuesto detrás de la penalización ortogonal no
se cumple acá**. La penalización asume que la física faltante es ortogonal a las direcciones
de parámetros; F4b midió que el 67 % de esa física *no lo es*. Prohibirle a `g` ser
redundante es, en este problema, prohibirle hacer la mayor parte del trabajo útil.

#### 15.2 · Calibrar `frac_redundante`: el punto cero no es cero

Esto lo detectó la auditoría del código y corrige una lectura que yo tenía mal. Sobre ε=1:

| referencia | `frac_redundante` |
|---|---|
| ruido blanco, sin estructura | 0.002 |
| una red `g(I,E)` con pesos **aleatorios**, sin entrenar | 0.466 |
| **el Δf VERDADERO — la física real** | **0.670** |
| las correcciones aprendidas (A, B, C, D λ=1) | 0.86 – 0.94 |
| D con λ=10 | 0.385 |

Dos lecturas que cambian:

**Una corrección perfecta puntuaría 0.67, no 0.** Así que el objetivo no es *minimizar* esta
métrica sino **acercarla al valor de la física real**. Comparar el 0.94 aprendido contra 0
exagera la patología; compararlo contra 0.67 la mide bien: la corrección aprendida es más
redundante de lo que la física justifica, pero no es pura competencia.

**Y una función suave cualquiera ya arranca en 0.47.** Simplemente por ser suave sobre esta
variedad de datos, una red aleatoria solapa casi la mitad de su energía con las direcciones
de parámetros. Ese es el piso realista, no el cero.

Con esa calibración, lo que hizo D con λ=10 se ve claro: empujó `g` hasta 0.385, **por
debajo del Δf verdadero (0.67) y por debajo incluso de una red aleatoria (0.47)**. No la
purificó: la alejó de donde vive la física real, y el sesgo volvió a los parámetros. En el
límite `λ→∞` la variante D converge al estimador white-box — que es exactamente el punto de
partida que se quería mejorar.

> Esto conecta las dos direcciones de la Parte V: la regularización ortogonal (opción D) sólo
> va a rendir si antes se consigue que el mismatch **sea** mayormente no-imitable — que es
> exactamente lo que persigue el diseño de estímulos (opción A). **A habilita a D.**

Un detalle fino que respalda todo lo anterior: la magnitud aprendida por la variante B es
`0.477`, muy cerca de la relación verdadera `|Δf|/|f_WC| = 0.44`. La variante libre sobrepasa
(0.579) y las regularizadas quedan cortas. La restricción al estado, sola, ya acierta el
tamaño del hueco.

#### 15.0 · El control que cierra la interpretación: ¿y si NO hay nada que corregir?

Lo anterior deja una pregunta abierta: si la corrección ayuda porque libera a `θ` de absorber
física no representable, entonces **cuando no hay física que absorber no debería ayudar — y
debería estorbar.** Es el control decisivo, y se corrió con `ε=0` (planta = modelo, sin
hueco):

| ε=0, sin mismatch | error medio de θ | error máximo | MSE test |
|---|---|---|---|
| **white-box** | **1.05 %** | 3.67 % | 9.14e-4 |
| A — `g(I,E,P,Q)` | 26.7 % | 47.0 % | 1.24e-3 |
| B — `g(I,E)` | 18.1 % | 36.4 % | **7.72e-4** |

**La corrección empeora los parámetros entre 17 y 25 veces** — de 1.05 % a 18–27 %. Y la
variante B **mejora el MSE** (7.72e-4 contra 9.14e-4) mientras destruye la identificación.
Ésa es la trampa en su forma pura: el ajuste dice que todo va bien y los parámetros están
arruinados.

Poniendo los dos regímenes uno al lado del otro:

| | ε=0 (sin hueco) | ε=1 (con hueco) |
|---|---|---|
| white-box | 1.05 % | 59.5 % |
| mejor gray-box | 18.1 % | 30.8 % |
| **efecto de la corrección** | **×17 peor** | **48 % mejor** |

> **Éste es el resultado central del trabajo.** Una corrección neuronal es **dañina justo
> cuando tu modelo ya es correcto, y útil justo cuando no lo es.** Y el problema práctico es
> que **no podés distinguir en cuál de los dos casos estás mirando el ajuste**: en ε=0 el MSE
> mejora igual. Hace falta una métrica que no dependa del ajuste — y por eso importa
> `frac_redundante` (sección 15.2), que es lo único que queda cuando no se conoce el Δf
> verdadero.

#### 15.1 · F4 · Por dónde roba la corrección — el bucle predecir→confirmar

Si `g_φ` compite con los parámetros, ¿por dónde entra? La hipótesis natural: por las
direcciones **más débiles** de la matriz de Fisher, porque son las más baratas — mover el
modelo en una dirección mal condicionada casi no cambia la trayectoria, así que la red puede
hacerlo casi gratis.

El procedimiento: se calcula cuánto mueve la trayectoria la corrección aprendida
(`Δy_g`), se busca el cambio de parámetros equivalente resolviendo `c = argmin ‖J_rel·c − Δy_g‖`
con el mismo jacobiano relativo que usa `scripts/fisher_identifiability.py`, y se descompone
`c` en la base de vectores singulares.

Para la variante B con ε=1:

| | |
|---|---|
| cuánto mueve la trayectoria `g_φ` | 18.3 % de `‖y‖` |
| fracción explicable por un `δθ` | **0.9375** |
| **energía en las 3 direcciones más débiles** | **99.5 %** (azar = 30 %) |
| energía sólo en σ₁₀ | **93.4 %** |

Y σ₁₀ es la dirección `wII` (+1.00, con trazas de `ai` y `ti`) — **exactamente la dirección
que el análisis Fisher+SVD del proyecto ya había identificado como la más débil.**

Esto cierra el bucle predecir→confirmar que ya es el aporte #2 declarado del trabajo, ahora
extendido al modelo híbrido: **la FIM predice, sin entrenar nada, por dónde va a entrar la
corrección neuronal a competir con los parámetros, y acierta con el 93 % de la energía.**

Y explica de paso el resultado de F2: el peor parámetro es siempre `wII`, en los cinco
niveles de mismatch. Es la misma dirección débil, castigada por dos mecanismos distintos —
el sesgo del white-box y el robo de la corrección.

### 16. F5 · ¿Aprendió la física correcta? No

| | A — `g(I,E,P,Q)` | B — `g(I,E)` |
|---|---|---|
| \|Δf\| verdadero (RMS) | 0.0172 | 0.0172 |
| \|g_φ\| aprendida (RMS) | 0.0170 | 0.0129 |
| **R² contra el Δf verdadero (test)** | **−0.63** | **−1.21** |
| techo del oráculo sin memoria (test) | 0.775 | −0.111 |
| amplificación al extrapolar +40 % | ×1.52 | ×1.88 |

El R² es **negativo**: la corrección aprendida es peor que predecir la media. Y sin embargo
su magnitud coincide casi exactamente con la del término verdadero (0.0170 vs 0.0172).
Aprende algo del tamaño correcto y de forma equivocada — exactamente lo que F4b predice
cuando el reparto entre `θ` y `g` no es único.

El techo del oráculo agrega un dato estructural: para `g(I,E)` vale **−0.111**, o sea que el
Δf verdadero **no es función del estado solo**. Tiene que ser así: la mitad del hueco es el
actuador, que depende de `P,Q` y de su propio estado interno. Con `P,Q` entre los argumentos
el techo sube a 0.775.

> Esto deja una tensión de diseño explícita, y es útil que quede escrita: `g(I,E)` es lo que
> conviene para identificabilidad y para que el controlador pueda cancelar; pero es
> justamente lo que **no puede** representar la parte del mismatch que entra por la entrada.

### 17. F6 · Lazo cerrado: la corrección aprendida **empeora** el control

> **Quién es quién, porque se presta a confusión.** La **planta es siempre la misma**: el
> simulador real con la perturbación. **El modelo aprendido no reemplaza al cerebro** — eso se
> hace en otra parte del proyecto (`make_neural_plant`, usado en `eval_closed_loop.py`), pero
> no acá. Lo único que cambia entre filas es **con qué conocimiento se arma el controlador**:
> los 10 parámetros θ̂ (que entran en la inversión de la sigmoidea y en la cancelación del
> acoplamiento) y, opcionalmente, la corrección `ĝ` restada del objetivo. "Usar sólo los
> parámetros" = entrenar con la red y después descartarla; "restar la red" = además evaluarla
> en cada paso de control.

Todas las filas contra la misma planta perturbada y con el mismo rango de actuación para
todos los controladores (ver el recuadro de abajo: esto último no era así al principio y
cambiaba las conclusiones):

| controlador construido con | RMSE_I | RMSE_E | saturación |
|---|---|---|---|
| 0. planta **limpia** + θ verdaderos | 0.0335 | **0.0313** | 0 % |
| 1. oráculo: θ verdaderos, planta perturbada | 0.0570 | 0.0629 | 0 % |
| 2. white-box: θ̂ con 187 % de error | 0.0531 | 0.0886 | 24 % |
| **con la corrección sin regularizar (B)** | | | |
| 3a. gray-box θ̂, sin cancelar ĝ | 0.0704 | 0.0820 | 42 % |
| 3b. gray-box θ̂ + ĝ cancelada | 0.1726 | 0.1142 | 25 % |
| **con la corrección regularizada (D, λ=1)** | | | |
| 3a. gray-box θ̂, sin cancelar ĝ | 0.0550 | 0.0784 | 13 % |
| 3b. gray-box θ̂ + ĝ cancelada | **0.2064** | **0.0645** | 22 % |

Lectura, de arriba abajo. La perturbación por sí sola **duplica** el error de seguimiento
(0.0313 → 0.0629) aun con los parámetros verdaderos: es el costo del hueco estructural, y no
hay identificación que lo arregle. Encima de eso, el error paramétrico del white-box agrega
un 41 % más (0.0629 → 0.0886). Los mejores parámetros del gray-box recuperan parte de eso
(0.0784 con la variante regularizada, que además satura menos: 13 % contra 24 %).

**Meter `ĝ` en la cancelación da un resultado mixto, y el detalle importa.** Con la
corrección sin regularizar empeora todo (0.0820 → 0.1142 en E). Con la regularizada, el lazo
de E **se recupera casi hasta el oráculo** (0.0784 → 0.0645, contra 0.0629 del oráculo) pero
el lazo de I **se destruye** (0.0550 → 0.2064).

O sea: la corrección aprendida acierta lo suficiente en un canal como para ayudar, y se
equivoca lo suficiente en el otro como para arruinarlo. Netamente **no es usable para
cancelación**, y es consistente con F5: el R² contra el Δf verdadero es ~0 (está en el techo
de lo que una corrección state-only puede hacer, y ese techo es cero). La cancelación por
linealización por realimentación usa `ĝ` sola y aislada, y la resta punto a punto — no
tolera que apunte mal aunque el ajuste global sea bueno.

> **La conclusión práctica, que vale para cualquiera que haga gray-box orientado a control:
> que una corrección ajuste bien los datos NO la habilita para cancelación.** Son dos
> requisitos distintos. El ajuste tolera que `θ` y `g` se repartan el trabajo de cualquier
> manera; la cancelación no, porque usa a `g` sola y aislada.

> ⚠️ **Dos bugs que la auditoría encontró acá, y que cambiaban las conclusiones.**
> (1) El re-clampeo del objetivo de la sigmoidea existía sólo en la rama con corrección, así
> que pasar un `ĝ` idénticamente nulo ya "mejoraba" el RMSE un 25 % sin que hubiera ninguna
> corrección — un artefacto del mismo orden que cualquier efecto real. (2) Los topes de
> actuación se derivaban del `θ̂` de cada controlador, así que cada uno enfrentaba la misma
> planta con distinto margen de actuación; el white-box ganaba autoridad de control por
> tener un `ai` estimado chico. Con eso, el white-box parecía empatarle al oráculo
> (0.0628 vs 0.0629). Corregido —mismos topes físicos para todos— el white-box en realidad
> **degrada un 41 %**. La columna "saturación" es el diagnóstico que se agregó para que esto
> no vuelva a pasar inadvertido: sin ella, un `ĝ` divergente produce igual un número finito
> de aspecto sano.

### 18. F7 · Dónde deja de alcanzar una corrección sin memoria

Se ajusta la arquitectura exacta de `g_φ` contra el Δf **verdadero** de cada familia, con
estímulos de test nunca vistos. Eso da el **techo** de lo que la corrección podría aprender
en el mejor de los casos — separando la pregunta "¿es representable?" de "¿el entrenamiento
lo encuentra?".

| familia | tipo | \|Δf\| | techo con `g(I,E)` | techo con `g(I,E,P,Q)` | veredicto |
|---|---|---|---|---|---|
| refractariedad `r=0.1` | capturable | 0.0066 | 0.967 | **0.997** | capturable |
| sigmoidea heterogénea | capturable | 0.0105 | 0.768 | **0.984** | capturable |
| adaptación `τ_a=1 ms` | memoria rápida | 0.0089 | 0.913 | 0.989 | capturable |
| adaptación `τ_a=3 ms` | memoria rápida | 0.0069 | 0.850 | 0.948 | capturable |
| adaptación `τ_a=10 ms` | memoria media | 0.0048 | 0.615 | 0.768 | parcial |
| adaptación `τ_a=30 ms` | memoria lenta | 0.0040 | 0.439 | 0.691 | parcial |
| adaptación `τ_a=100 ms` | memoria lenta | 0.0027 | 0.309 | 0.543 | **pobre** |
| depresión sináptica | estado oculto | 0.0193 | 0.634 | 0.717 | parcial |
| población no modelada | estado oculto | 0.0049 | 0.608 | 0.784 | parcial |
| **el par del roadmap** | refract + actuador | 0.0172 | **−0.111** | 0.775 | parcial |
| deriva de `wEE(t)` | no autónoma | 0.0205 | 0.234 | 0.527 | **pobre** |
| ruido de proceso | irreducible | 0.0201 | 0.119 | **0.112** | **pobre** |

Tres bloques, y cada uno falla por una razón distinta.

**Las capturables.** El Δf es función pura del estado y la entrada: la corrección las
representa casi perfecto. La refractariedad llega a 0.997.

**Las de estado oculto.** El límite lo pone *cuán lenta* es la variable escondida. La
adaptación lo muestra en una curva limpia, con `te=1 ms` y `ti=2 ms` como referencia:

| `τ_a` | 1 ms | 3 ms | 10 ms | 30 ms | 100 ms |
|---|---|---|---|---|---|
| techo con `g(I,E)` | 0.913 | 0.850 | 0.615 | 0.439 | 0.309 |

Cuando el estado oculto es **rápido** respecto del sistema, queda esclavizado al estado
visible y la corrección lo reconstruye. Cuando es **lento**, guarda historia propia que
`(I,E)` no contiene. Es una perilla continua entre "capturable" y "no capturable" con un solo
número, y es el argumento cuantitativo para cuándo hace falta darle memoria a la corrección.

**Las que no se pueden.** El ruido de proceso da 0.112 con todos los argumentos: no es
función de nada, es el piso irreducible. La deriva de `wEE(t)` da 0.234 pero por una razón
**distinta**: depende explícitamente de `t`, y `t` no es argumento de `g_φ`. Son dos maneras
diferentes de que la corrección falle —por azar y por dependencia temporal— y conviene no
confundirlas.

> **El caso que más informa es el par del roadmap: −0.111 con `g(I,E)` y 0.775 con
> `g(I,E,P,Q)`.** O sea que el Δf de la combinación elegida **no es función del estado**, y
> por eso ninguna corrección state-only puede recuperarlo (lo que F5 confirmó por separado).
> La causa es el actuador: entra por `P,Q` y arrastra su propio estado interno.

---

## Parte IV-bis — Trabajo posterior al roadmap

*Las tres direcciones que quedaron recomendadas al cerrar F7, ya ejecutadas.*

### 19. ¿Copia bien el comportamiento? (la pregunta práctica)

Todas las métricas anteriores miran los **parámetros**. Ésta mira lo otro: si el modelo
aprendido, soltado a rodar solo, reproduce la actividad del cerebro simulado.

La prueba es exigente a propósito. Se le da únicamente el **estado inicial y el estímulo**, y
tiene que generar los 200 ms enteros por su cuenta, sin volver a mirar el dato real (rollout
*open loop*, 4000 pasos). Cualquier error se acumula. Y se evalúa en estímulos de **test**,
que no vio al entrenar. Es exactamente lo que pasaría si lo usáramos como planta para
diseñar un controlador.

El error va normalizado al rango de la señal, para que sea interpretable:

| modelo | error en I | error en E | correlación I | correlación E |
|---|---|---|---|---|
| white-box, planta **sin** hueco | **2.0 %** | **2.0 %** | 0.988 | 0.989 |
| white-box, planta **con** hueco | 13.0 % | 15.1 % | 0.825 | 0.821 |
| gray-box B, `g(I,E)` | 16.7 % | 14.5 % | 0.830 | 0.846 |
| gray-box D, regularizada | 12.3 % | 14.0 % | 0.849 | 0.851 |

*(referencia: <5 % reproduce muy bien; 5–15 % la forma es correcta con desvíos; >25 % no
sirve como planta)*

**Sin hueco, la copia es casi perfecta.** 2 % de error después de 200 ms rodando sola y
correlación 0.99. Esto vale como control positivo de toda la maquinaria: cuando el modelo
tiene la estructura correcta, la Neural ODE reproduce el sistema con mucha fidelidad.

**Con hueco, reproduce la forma pero no los detalles.** Correlación 0.82 significa que los
picos y valles caen donde corresponde —el ritmo está bien— pero hay desfase y error de
amplitud que suman ~15 %. Sirve para entender el sistema; no para predecir con precisión.

**Y lo más informativo:** la corrección neuronal recupera casi la mitad del error
*paramétrico* (59.5 % → 31 %) pero prácticamente nada del error de *trayectoria*
(15.1 % → 14.0 %, apenas un 7 % relativo). Ayuda a **identificar**, no a **predecir**.

No es contradictorio, y la explicación es la de la sección 14: la corrección le saca al
backbone la obligación de deformar `θ`, lo cual arregla los parámetros. Pero lo que aprende
es ~90 % redundante con esos mismos parámetros, así que el campo vectorial total queda casi
igual de equivocado que antes. Mueve la responsabilidad de sitio sin agregar física.

### 20. El integrador: absuelto

Durante F1 apareció que el integrador que usan los scripts del proyecto (`solve_ivp` con
`rtol=1e-3`) acumula un error de ~1.5e-2 contra una referencia de alta precisión. Eso cae
justo entre los dos primeros niveles de los barridos de robustez (σ = 0, 0.01, 0.05, 0.10),
y encima es un error **sistemático**, no aleatorio, así que promediar no lo elimina. Había
que descartar que estuviera contaminando un resultado ya establecido.

Se generó el mismo dataset con tres integradores y se identificaron los 10 parámetros con
cada uno:

| integrador | desvío de trayectoria | error de θ con σ=0 | error de θ con σ=0.01 |
|---|---|---|---|
| histórico (`RK45`, `rtol=1e-3`) | 6.8e-3 (máx 1.9e-1) | 0.54 % | 8.14 % |
| paso fijo (RK4) | 1.5e-4 (máx 3.2e-3) | 1.05 % | 8.39 % |
| referencia (`DOP853`, `rtol=1e-11`) | — | 1.07 % | **8.41 %** |

**Veredicto: no contamina.** Al nivel de ruido que importa (σ=0.01) los tres coinciden dentro
del 3 % relativo, y el peor parámetro es `wII` en los tres casos con el mismo valor (~41 %).
**Los barridos de robustez del proyecto son válidos y no hay que rehacerlos.**

Dos observaciones para no malinterpretar la tabla. Con σ=0 el histórico parece *mejor*
(0.54 % contra 1.07 %); es casualidad, no una virtud: su error sistemático se cancela
parcialmente con el sesgo de la estimación. Y el desvío máximo de 0.19 asusta pero es
puntual, en los transitorios rápidos donde el paso adaptativo afloja; el RMS es 30× menor.

Recomendación práctica: usar el paso fijo para lo nuevo (es más preciso y es el mismo
integrador con el que se entrena), pero **no hace falta regenerar nada de lo ya publicado**.

### 21. Diseño de estímulos: se puede romper la degeneración, con un matiz

La sección 14.1 mostró que la fracción imitable depende del estímulo (71 % con APRBS,
94 % con un pulso casi constante). Si se puede elegir, se puede **optimizar**.

Se parametrizó el estímulo como 24 escalones (realizable con optogenética) y se buscó por
evolución, **con la amplitud media fijada** al valor del mejor estímulo de librería, para que
cualquier mejora sea de forma y no de energía. Dos objetivos:

| objetivo | \|Δf\| | imitable | residuo aprovechable |
|---|---|---|---|
| APRBS de librería (referencia) | 0.0223 | 71.4 % | 0.0119 |
| maximizar el residuo | 0.0376 | 72.2 % | **0.0198** (+66 %) |
| minimizar la fracción | 0.0232 | **59.5 %** | 0.0148 (+24 %) |

Las dos cosas funcionan y son distintas: se puede conseguir **66 % más de física aprovechable
con la misma energía**, o **bajar la degeneración 12 puntos**. La segunda es la que ataca la
causa raíz.

**Pero apareció un matiz que cambia la recomendación.** Al armar un dataset con 20 variantes
del estímulo diseñado, la fracción imitable *conjunta* subió a 73.9 % — **peor** que el
dataset de librería (67.1 %). La razón es instructiva: cuando se exige un único `δθ` que
explique *todos* los escenarios a la vez, la degeneración se rompe sola si los escenarios son
**distintos entre sí**. La librería tiene 7 familias diferentes y esa diversidad hace mucho
trabajo gratis; 20 variantes de la misma forma, no.

> **Hay dos palancas y la segunda pesa más:** que cada estímulo sea individualmente bueno, y
> que los estímulos sean **complementarios entre sí**. Optimizar una señal aislada es diseño
> de señal; lo que hace falta es diseñar el **experimento entero** — un conjunto de estímulos
> elegido para que sus degeneraciones no se solapen.

Eso es lo que queda como siguiente paso concreto y es, además, la versión interesante del
problema: *optimal experiment design* para separar `g` de `θ`, no sólo para informar sobre `θ`.

### 22. Corrección estructurada: identificable si la forma está completa

F6 mostró que una corrección de caja negra no sirve para cancelar. La alternativa: reemplazar
el MLP de ~1000 pesos por **tres parámetros con significado físico** — `r_i`, `r_e`
(refractariedad) y `sat` (saturación del actuador). Es identificable, interpretable, y sobre
todo la cancelación del controlador vuelve a ser **exacta y explícita**, porque se conoce la
forma analítica y se puede invertir a mano.

Se implementó (`variant="S"`, con la inversión correspondiente en el IMC) y **falló**: los
tres parámetros convergieron a los extremos (`r → 0`, `sat → ∞`), o sea a "no hay corrección".

El diagnóstico resultó más interesante que el resultado. Fijando `θ` en los valores
verdaderos y ajustando sólo los tres físicos:

| | pérdida |
|---|---|
| sin corrección | 1.36e-3 |
| con la corrección **verdadera** (`r=0.10`, `sat=3.0`) | 7.75e-4 |
| con la corrección que **encuentra el optimizador** (`r=0`, `sat=1.90`) | **5.69e-4** |

El optimizador encuentra un ajuste **mejor que el de la física verdadera**, con parámetros
equivocados. No es un fallo numérico: es que a la corrección le falta una componente —el
**retardo** del actuador, que tiene estado propio y un modelo sin memoria no puede
representar— y usa las componentes que sí tiene para compensar la que le falta.

La confirmación es limpia. Generando una planta que tiene **exactamente** la forma que la
corrección puede representar (refractariedad + saturación, sin retardo):

| parámetro | verdadero | recuperado | error |
|---|---|---|---|
| `sat` | 3.0 | 2.962 | **1.3 %** |
| `r_e` | 0.10 | 0.0912 | 8.8 % |

y la pérdida cae 145× (4.7e-4 → 3.2e-6).

> **Conclusión: la corrección estructurada es identificable y precisa cuando la forma
> asumida está completa, y converge a valores equivocados —con mejor ajuste que la verdad—
> cuando le falta una componente.** Es la misma lección de degeneración que la sección 14,
> un nivel más abajo: con física estructurada el problema no desaparece, se vuelve más
> explícito y por eso más fácil de diagnosticar.

Es una advertencia práctica fuerte para el uso en datos reales: un gray-box estructurado que
ajusta bien **no garantiza** que los parámetros físicos que reporta sean los correctos, si la
forma propuesta es incompleta. Y en datos reales siempre lo es.

---

## Parte V — Qué sigue

### 23. El diagnóstico, en cuatro frases

Antes de las opciones conviene fijar qué aprendimos, porque todas las opciones salen de acá.

1. **El mismatch es mayormente "parámetro-símil".** Dos tercios de la física faltante se
   pueden imitar moviendo los 10 parámetros. Eso hace que el reparto entre `θ` y `g` no sea
   único, y es la causa raíz de todo lo demás.
2. **La corrección ayuda si y sólo si hay mismatch.** Con hueco recupera el 48 % del error
   paramétrico; sin hueco lo empeora 17 veces. Y en los dos casos el MSE mejora, así que el
   ajuste no te dice en cuál estás.
3. **Ayuda sin aprender física.** Le saca al white-box la obligación de deformar `θ`, pero lo
   que aprende es ~90 % redundante con `θ` (contra 67 % de la física real) y tiene R² ≈ 0
   contra el Δf verdadero.
4. **Y por eso no sirve para cancelar.** La linealización por realimentación usa `ĝ` sola y
   aislada; si `ĝ` apunta mal, restarla arruina un canal aunque mejore el otro. Ajustar bien
   ≠ servir para cancelar.

### 24. Las opciones, ordenadas

**A. Diseño de EXPERIMENTOS, no de señales.** ⭐ *Ejecutada a medias — y lo que salió cambió
la recomendación. Es la dirección con más recorrido.*

Optimizar un estímulo aislado **funciona** (sección 21): la degeneración baja del 71 % al
59.5 % con la misma amplitud media. Pero un dataset de 20 variantes de ese único diseño da
73.9 %, peor que la librería, porque **la diversidad entre estímulos rompe más degeneración
que la calidad de cada uno**.

Entonces la formulación correcta no es "diseñar la señal óptima" sino **diseñar el conjunto**:
elegir N estímulos para minimizar la fracción imitable *conjunta*, o sea para que sus
degeneraciones no se solapen. Es la analogía exacta del OED clásico pero con una función
objetivo nueva: en vez de maximizar información de Fisher sobre `θ` (lo que el proyecto ya
hace), **maximizar la separación entre el espacio de `θ` y el del mismatch**.

Ya está escrito el evaluador conjunto (`scripts/exp_a_set_design.py`); falta correrlo y
validar que un dataset generado así mejore de verdad el R² de la corrección contra el Δf
verdadero. Es el aporte con más potencial de publicación de todo esto.

**B. Corrección estructurada en lugar de caja negra.** *Ejecutada — funciona con una condición
importante.*

Tres parámetros físicos (`r_i`, `r_e`, `sat`) en lugar de un MLP de ~1000 pesos, con la
cancelación exacta implementada en el IMC. Resultado (sección 22): **es identificable y
precisa si la forma asumida está completa** (`sat` con 1.3 % de error), y converge a valores
equivocados —con mejor ajuste que la verdad— si le falta una componente.

Lo que falta es cerrar el círculo: agregarle el **retardo** del actuador, que es la componente
faltante. Eso requiere que la corrección tenga estado propio, así que se solapa con la opción
C. Si se hace, es el camino más prometedor para el control: física interpretable + cancelación
exacta.

**C. Darle memoria a la corrección.** *Ahora sí, y con una razón concreta.*

Cuando escribí el roadmap puse esta opción al final, con el argumento de que agregar capacidad
es lo último que conviene cuando falta identificabilidad. **Los resultados la subieron de
prioridad**, y por una razón específica y no genérica: dos experimentos independientes
señalan a la misma componente faltante.

F5 midió que el Δf verdadero **no es función del estado solo** (el oráculo state-only da
R² = −0.11). Y la sección 22 mostró que la corrección estructurada converge a valores
equivocados **precisamente porque le falta el retardo**. Los dos apuntan al mismo lugar: el
actuador tiene dinámica propia y ningún modelo sin memoria la alcanza.

Lo importante es que ahora no hace falta memoria genérica —una latent ODE, una ventana de
historia— sino **un solo estado extra con forma conocida**: `dP_lag/dt = (P_cmd - P_lag)/τ`.
Es un parámetro más, no una red. Mantiene la interpretabilidad y probablemente la
cancelación explícita. Es la versión barata y dirigida de esta opción, y es la que yo haría.

**D. Regularización: ya está respondido, y la respuesta es "poca y calibrada".** *Cerrado.*

Esta opción estaba abierta cuando escribí el roadmap; el barrido la cerró. Resumen:
regularización **suave** (`λ‖g‖²` con λ=0.1–1, o la ortogonal con λ=1) es lo mejor: baja el
error paramétrico medio de 59.5 % a ~31 % y además reduce a la mitad la saturación del
actuador en lazo cerrado. Regularización **fuerte** (ortogonal con λ=10) es
contraproducente: empuja `frac_redundante` a 0.385, por debajo del 0.67 que tiene la física
real, y el error paramétrico vuelve a 44.6 %.

Lo que queda como **receta reutilizable** es la calibración: el objetivo no es minimizar
`frac_redundante` sino acercarla al valor de la física real. Y cuando no se conoce el Δf
verdadero —o sea, siempre, fuera de este montaje— el sustituto es comparar contra el ~0.47
que da una red aleatoria de la misma arquitectura, que sí se puede calcular en cualquier
caso.

**E. Datos de lazo cerrado (estilo DAgger).** F5 midió que la corrección se amplifica ×1.5–1.9
al salir de la región visitada, y el controlador lleva al sistema justamente ahí. El ciclo
natural: correr el lazo cerrado, guardar las trayectorias, reentrenar con ellas, repetir.

**F. Llevarlo a los datos reales.** Ya existe `scripts/train_real_output.py` con la variante
`v1` (gray-box). Ahora hay una receta con fundamento para aplicarla: restringir `g` al
estado, regularizar, y —sobre todo— **reportar `frac_redundante`**, que en datos reales es la
única forma de saber si la corrección está aportando física o compitiendo con los parámetros.
Sin el Δf verdadero esa métrica es lo único que queda.

**G. Revisar el integrador del proyecto.** *Chico pero conviene no dejarlo pasar.*

**Cerrado y resuelto** (sección 20). Se midió y el integrador **no contamina**: con σ=0.01 los
tres integradores coinciden dentro del 3 % relativo. No hay que rehacer los barridos de
robustez. Para lo nuevo conviene el paso fijo, que es 30× más preciso y además es el mismo
integrador con el que se entrena.

### 25. Cómo avanzaría yo

*(actualizado tras ejecutar G, A y B)*

**Lo primero, y es de un día:** cerrar **A** corriendo `exp_a_set_design.py`, que ya está
escrito. Optimiza el **conjunto** de estímulos en vez de uno solo, que es lo que el matiz de
la sección 21 mostró que hace falta. Y después la validación que importa: generar un dataset
con ese conjunto, entrenar, y ver si el R² de la corrección contra el Δf verdadero **se vuelve
positivo**. Si eso pasa, es el resultado que da vuelta la historia: pasaríamos de "la
corrección ayuda sin aprender física" a "con el experimento bien diseñado, aprende física".

**Después, y es lo más prometedor para el control:** completar **B** con el retardo. La
sección 22 dejó el diagnóstico hecho —la corrección estructurada falla *sólo* porque le falta
esa componente— y la sección C explica que es un estado extra con forma conocida, no una red.
Sería una corrección de 4 parámetros físicos, identificable, interpretable y con cancelación
explícita. Es el camino que más se parece a lo que el proyecto quiere de punta a punta.

**Lo que ya no haría:** insistir con regularización de la corrección de caja negra. El barrido
la agotó (sección 15.2): suave ayuda, fuerte perjudica, y ninguna la vuelve física real. El
problema no está en cómo se regulariza `g` sino en que el experimento le permite esconderse —
que es lo que ataca A— o en que la forma es incompleta — que es lo que ataca B.

**Y una advertencia para cuando esto vaya a datos reales (opción F):** ahí no existe el Δf
verdadero, así que **no hay forma de saber si la corrección aprendió física o está compitiendo
con los parámetros mirando el ajuste** — el control de ε=0 muestra que el MSE mejora en los
dos casos. Lo mínimo a reportar es `frac_redundante` contra la referencia de una red
aleatoria. Sin eso, un gray-box sobre datos reales puede estar destruyendo la identificación
y pareciendo un éxito.

### 26. Qué agrega esto al aporte del proyecto

Mirando `docs/novedad_trabajo_relacionado.md`, esto extiende los tres aportes declarados y
agrega uno nuevo:

| aporte | cómo queda |
|---|---|
| Identificar los 10 parámetros de WC | ahora **bajo mismatch estructural**, con la curva de cuánto cuesta la rigidez (1 % → 93 % de error medio) |
| Fisher+SVD predice qué es identificable | confirmado por **dos** caminos independientes: el mismatch degrada `wII` primero en los 6 niveles, y el 93 % de lo que roba la corrección va por σ₁₀ = `wII` |
| La fragilidad no se propaga al control | **matizado**: con topes de actuación iguales para todos, el error paramétrico sí degrada un 41 % |
| *(nuevo)* geometría del mismatch | 67 % de la física faltante es indistinguible de mover parámetros — explica el sesgo del white-box y el techo del gray-box |
| *(nuevo)* el cruce en ε=0 | la corrección es **dañina cuando el modelo ya es correcto** (×17 peor) y útil cuando no lo es (48 % mejor) — y el ajuste no distingue los dos casos |
| *(nuevo)* ajustar ≠ poder cancelar | una corrección con buen ajuste y R² ≈ 0 contra la física real arruina un canal del lazo cerrado aunque mejore el otro |
| *(nuevo)* calibración de la redundancia | el punto cero no es 0: la física real puntúa 0.67 y una red aleatoria 0.47 — minimizar la métrica es alejarse de la física |
| *(nuevo)* identificar ≠ predecir | la corrección recupera el 48 % del error paramétrico y sólo el 7 % del error de trayectoria |
| *(nuevo)* el diseño puede romper la degeneración | 71 % → 59.5 % eligiendo el estímulo, y la **diversidad** del conjunto pesa más que la calidad de cada uno |
| *(nuevo)* límite de la física estructurada | una corrección con forma incompleta ajusta **mejor que la verdad** con parámetros equivocados |

Los últimos son, hasta donde vimos, resultados que nadie reporta — y sólo se pueden obtener
en un montaje donde el Δf verdadero es conocido, que es exactamente la ventaja de haber
construido la incertidumbre a mano en vez de tomarla de datos reales.

Y hay uno que vale para cualquiera que haga gray-box, no sólo para Wilson-Cowan: **una
corrección aprendida que ajusta bien puede no ser física, y no hay forma de notarlo mirando el
ajuste.** Se necesita una métrica geométrica —cuánto de la corrección es redundante con los
parámetros— y un punto de comparación calibrado. Sin eso, el gray-box puede estar destruyendo
la identificación y pareciendo un éxito.

---

## Apéndice — Dónde está cada cosa

**Código nuevo**

| archivo | qué hace |
|---|---|
| `src/wilson_cowan/uncertainty.py` | las 9 familias de perturbación, componibles por 5 ganchos |
| `src/wilson_cowan/model.py` | `perturbed_field()`, `rhs_aug()`, integrador de paso fijo con estado aumentado |
| `src/neural_ode/dynamics.py` | `backbone()` / `g_out()` separados, `correction_inputs`, `backbone_sensitivities()` |
| `src/neural_ode/graybox_train.py` | entrenamiento de las 5 variantes, penalización por proyección, métricas |
| `src/neural_ode/closed_loop.py` | cancelación gray-box y estructurada, plantas con estado oculto, diagnóstico de saturación |

**Experimentos del roadmap (F1–F7)** — `scripts/exp_f1_characterize.py`,
`exp_f2_rigidity_cost.py`, `exp_f3_graybox.py`, `exp_f4_fim_hybrid.py`,
`exp_f4b_geometria_mismatch.py`, `exp_f5_functional_recovery.py`, `exp_f6_closed_loop.py`,
`exp_f7_controls.py`. Consolidado: `scripts/informe_incertidumbre.py`.

**Experimentos posteriores (secciones 19–22)**

| script | qué contesta |
|---|---|
| `exp_reproduccion.py` | ¿copia bien el comportamiento? (rollout open-loop, NRMSE normalizado) |
| `exp_g_integrador.py` | ¿el error del integrador contamina los barridos de ruido? |
| `exp_a_stimulus_design.py` | optimiza UN estímulo (`--objetivo residuo` o `fraccion`) |
| `exp_a_validate.py` | genera un dataset con el estímulo diseñado y lo evalúa |
| `exp_a_set_design.py` | optimiza el CONJUNTO de estímulos (escrito, falta correr) |

**Para reproducir todo de cero:** `bash scripts/run_uncertainty_all.sh` (~2.5 h en 8 núcleos).
Corre F0 a F7 en el orden correcto, en olas de 4 procesos × 2 hilos — que es el reparto que
midió mejor: más hilos por proceso no acelera, así que conviene paralelizar entre corridas.

**Tests** — `tests/test_uncertainty.py` (20) y `tests/test_graybox.py` (9). Protegen: que sin
perturbación nada cambió, que todas preservan el reposo, que el estímulo comandado no se
contamina con el efectivo, y la matemática de la proyección (incluido el orden de aplanado,
que es el error silencioso más fácil de cometer).

**Datos** — `data/processed/uncertain/eps{0,0.25,0.5,1,1.5,2}.npz` y `controls/*.npz`. Cada
uno guarda el Δf verdadero punto a punto, y `P_eff`/`Q_eff` por separado del comandado.

**Resultados** — `results/uncertainty/*.json`, `results/uncertainty/informe.txt`,
`results/figures/{f1_caracterizacion,f4b_geometria,f5_recovery_*,f7_memoria,informe_incertidumbre}.png`.
