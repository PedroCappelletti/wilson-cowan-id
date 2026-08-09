# La Neural ODE y su entrenamiento, de punta a punta

> Cómo está construido el modelo, qué recibe, qué predice, contra qué se compara,
> cómo se optimiza y cómo se podría mejorar. Todo verificado contra el código;
> los números salen de correrlo, no de memoria.
>
> Archivos: `src/neural_ode/dynamics.py` (el modelo),
> `src/neural_ode/integrate.py` (el integrador),
> `src/neural_ode/graybox_train.py` (el entrenamiento).

---

## Respuestas rápidas

Si venís con una pregunta puntual, esto es el resumen; el detalle está más abajo.

| pregunta | respuesta corta |
|---|---|
| ¿Los 10 parámetros y la red se entrenan **en paralelo**? | **Sí, simultáneamente.** Un solo `loss.backward()` reparte gradiente a los tres grupos a la vez, y un solo `opt.step()` los mueve juntos. No hay fases ni alternancia. |
| ¿Por qué entonces tienen learning rates distintos? | Para regular **cuánto se mueve cada uno por época**. Con Adam el lr *es* el tamaño del paso: los pesos cambian hasta 0.05 por época y los de la red sólo 0.003. Le da ventaja al backbone para acomodarse antes de que la red tape el error. |
| ¿Ese freno alcanza? | **No del todo.** Medido: entre las épocas 100 y 200 la red igual se adelanta, llega a valer tanto como el backbone entero y el error paramétrico empeora de 46.9% a 55.4%. Después se revierte parcialmente (sección 5.3). |
| ¿Cómo se reparten el trabajo? | **No se reparten: compiten.** Se suman en la misma derivada y hay una sola loss, así que lo que explica uno el otro ya no lo ve. Y el 67% del error se puede explicar de las dos formas. |
| ¿Qué recibe la red? | Sólo `(I, E)` — el estado. En la variante A recibe además `(P, Q)`. **Nunca** ve la perturbación ni el estímulo efectivo. |
| ¿Qué predice el modelo? | **La derivada**, no la trayectoria: `ẋ = f(x, P, Q)`. La trayectoria sale de integrar esa derivada. |
| ¿Contra qué se compara? | Contra las trayectorias medidas `(I, E)`, en 101 instantes × 507 ventanas × 2 canales = **102.414 números** por época. |
| ¿Cuánto ve de una vez? | Ventanas de **5 ms** (100 pasos). No los 200 ms de la trayectoria. |
| ¿Ve los valores verdaderos? | **No.** Arranca todo en 1.0 y nunca los ve; sólo se usan al final para reportar el error. |

---

## 1. Qué es el modelo

### 1.1 Predice la derivada, no la trayectoria

Ésta es la diferencia con una red que uno esperaría. El modelo **no** mapea
"tiempo → actividad". Mapea:

```
(estado actual, estímulo actual)  →  velocidad de cambio del estado

     f_θ(x, P, Q)  =  ẋ  =  [dI/dt, dE/dt]
```

La trayectoria no se predice: **se construye integrando** esa derivada paso a
paso. Por eso se llama Neural ODE — la red define una ecuación diferencial, no
una función del tiempo.

Tiene una consecuencia práctica importante: el modelo puede correr con cualquier
estímulo, incluso uno que se decida sobre la marcha. Una red que mapeara
`t → (I,E)` quedaría atada al estímulo con el que se entrenó.

### 1.2 Las dos mitades

```
ẋ  =  f_WC(x, P, Q; θ)   +   g_φ(x)
      └── backbone ──┘       └─ corrección ─┘
      10 parámetros          MLP: 2 capas ocultas, 32 neuronas
      interpretables         1218 pesos
```

Los tamaños importan para entender la tensión que domina todo el trabajo: la red
tiene **122 veces más parámetros** que el backbone. Es un aproximador universal
compitiendo con 10 números que tienen significado biológico.

**El backbone** son las ecuaciones de Wilson-Cowan escritas a mano
(`dynamics.py:167`):

```python
u_i = wIE*E - wII*I + Q - thetai
u_e = wEE*E - wEI*I + P - thetae
dI  = (1/ti) * (-I + sigmoid(ai*u_i) - ki)
dE  = (1/te) * (-E + sigmoid(ae*u_e) - ke)
```

Los 10 parámetros son 4 pesos sinápticos (`wEE, wEI, wIE, wII`) y 6 físicos
(`te, ti, ae, ai, thetae, thetai`). `ke` y `ki` **no** se entrenan: se recalculan
en cada forward a partir de `ae,thetae` y `ai,thetai`, para que `E=I=0` siga
siendo un equilibrio aunque los parámetros cambien.

**La corrección** es un MLP chiquito que se suma a la derivada. Arranca en cero
exacto —la última capa se inicializa con pesos en cero (`dynamics.py:140`)— así
que en la época 0 el modelo **es** Wilson-Cowan puro y la red va apareciendo sólo
si le sirve.

### 1.3 Los parámetros no se guardan como uno los lee

Detalle que confunde al leer el código: internamente no vive `wEE`, vive
`raw_w`. Los parámetros positivos se guardan **crudos** y pasan por un softplus:

```python
self.raw_w = nn.Parameter(torch.log(torch.expm1(w)))   # inv_softplus
def weights(self):  return F.softplus(self.raw_w)      # siempre > 0
```

El motivo es que un peso sináptico negativo no tiene sentido físico, y el
softplus lo garantiza **por construcción** en vez de por penalización. El
optimizador trabaja sin restricciones sobre `raw_w`, y la positividad sale gratis.

> **La trampa que esto trae.** La parametrización cambia la escala del gradiente,
> y eso ya rompió una versión del código: la corrección estructurada usaba
> softplus con el crudo muy negativo (para arrancar en r≈0), donde la derivada
> del softplus vale ~0.0025. El parámetro no se movía nunca — no era un problema
> de learning rate, era gradiente muerto. Está documentado en `dynamics.py:69`.

---

## 2. Cómo se arman los datos que entran

### 2.1 El problema de integrar 4000 pasos

Cada trayectoria tiene 4000 pasos (200 ms a dt = 0.05 ms). Integrar eso de un
tirón y hacer backward a través de los 4000 pasos hace que el gradiente explote o
se desvanezca — es el mismo problema que las RNN largas.

### 2.2 La solución: multiple shooting

La trayectoria se corta en **ventanas de 100 pasos (5 ms)**, y cada ventana
**arranca desde el estado medido**, no desde donde el modelo venía
(`graybox_train.py:46`):

```python
x0  = [I[s,a], E[s,a]]              # el estado REAL en el borde de la ventana
Pw  = P[s, a:a+W]                   # el estímulo de esa ventana
tgt = [I[s,a:a+W+1], E[s,a:a+W+1]]  # lo que tiene que reproducir
```

Con los datos actuales eso da **507 ventanas** desde 13 trayectorias de
entrenamiento (40 ventanas por trayectoria).

Reiniciar desde el dato real es lo que hace tratable el problema. El precio es
que el modelo nunca tiene que sostener más de 5 ms por su cuenta durante el
entrenamiento — y eso explica por qué después falla al pedirle 200 ms seguidos
(sección 7.1).

### 2.3 Las 507 ventanas van todas juntas

No hay minibatches. Cada época procesa las 507 ventanas **en paralelo**, como un
batch único, aprovechando que el integrador está vectorizado. Por eso una época
es un solo `forward` + un solo `backward`.

---

## 3. El forward: cómo se construye la predicción

`rollout()` en `integrate.py` integra con **RK4 de paso fijo**, escrito en torch
para que sea diferenciable:

```python
def rk4_step(f, x, P, Q, dt):
    k1 = f(x, P, Q)
    k2 = f(x + 0.5*dt*k1, P, Q)      # P,Q constantes dentro del paso (ZOH)
    k3 = f(x + 0.5*dt*k2, P, Q)
    k4 = f(x + dt*k3, P, Q)
    return x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
```

Dos decisiones que están metidas ahí:

**Paso fijo y no adaptativo.** Un integrador adaptativo (como `solve_ivp`) elige
sus pasos según el error, y esa decisión no es diferenciable. Con paso fijo el
grafo de cómputo es una cadena determinista que el autograd puede recorrer.

**Zero-order hold.** El estímulo se mantiene constante dentro de cada paso, que
es como llega un control muestreado real. Si se interpolara, el modelo aprendería
a explotar una suavidad que el hardware no tiene.

El resultado es un tensor `(101, 507, 2)`: 101 instantes (los 100 pasos más el
inicial), 507 ventanas, 2 canales.

---

## 4. La comparación: contra qué se mide

```python
pred = rollout(model, x0, Pw, Qw, dt)     # (101, 507, 2)
data_loss = ((pred - tgt) ** 2).mean()    # 102.414 numeros
```

Es un **MSE plano** sobre todos los elementos: cada instante, cada ventana y cada
canal pesan lo mismo.

El objetivo `tgt` son las trayectorias `(I, E)` **medidas del simulador**. Y hay
que ser preciso sobre qué se le da al modelo y qué no:

| el modelo recibe | el modelo NO recibe |
|---|---|
| el estado inicial de cada ventana | la perturbación (no sabe que existe) |
| el estímulo **comandado** `P, Q` | el estímulo **efectivo** `P_eff, Q_eff` |
| — | los estados ocultos del actuador |
| — | los valores verdaderos de los 10 parámetros |
| — | el `Δf` verdadero (existe en el `.npz`, sólo se usa para evaluar) |

Que el estímulo efectivo esté prohibido no es un detalle: con el actuador
optogenético, lo que llega a la neurona pasó por un retardo y una saturación. Si
el entrenamiento usara `P_eff` le estaríamos regalando la mitad de la respuesta.

> Un detalle que conviene saber al mirar la loss: el instante 0 de cada ventana
> es el estado inicial *dado*, así que su error es exactamente 0 y entra en el
> promedio. Diluye la loss en ~1%. Es inofensivo pero está ahí.

---

## 5. La optimización: todo junto, no por partes

### 5.1 Un solo backward mueve las tres cosas

Ésta es la pregunta que más se hace, y la respuesta es inequívoca. El bucle
completo (`graybox_train.py:253`) es:

```python
for ep in range(1500):
    opt.zero_grad()
    pred = rollout(model, x0, Pw, Qw, dt)
    data_loss = ((pred - tgt) ** 2).mean()
    pen, diag = penalties()
    (data_loss + pen).backward()          # <- UN solo backward
    clip_grad_norm_(todos_los_params, 10.0)
    opt.step()                            # <- UN solo step
```

**No hay fases, ni alternancia, ni congelamiento.** Los 10 parámetros y los 1218
pesos de la red se mueven en el mismo paso, guiados por la misma loss.

Verificado corriéndolo: después de un único `backward()`, los tres grupos tienen
gradiente no nulo simultáneamente.

| grupo | qué es | `|grad|` medio |
|---|---|---|
| `raw_w` | los 4 pesos sinápticos | 3.96e-4 |
| `raw_te, raw_ti, …` | los 6 físicos | 1.39e-3 |
| `g.parameters()` | la red (1218 pesos) | 4.56e-3 |

### 5.2 Qué significa "la red va 17× más lenta"

```python
groups = [{"params": [model.raw_w],        "lr": 5e-2},   # pesos
          {"params": phys_raw,             "lr": 2e-2},   # fisicos
          {"params": model.g.parameters(), "lr": 3e-3}]   # la red
opt = torch.optim.Adam(groups)
```

**No se refiere al tiempo de cómputo ni a que la red se actualice menos seguido.**
Todos se actualizan en cada época, siempre. Se refiere a **cuánto se mueve cada
parámetro en cada paso** — el tamaño del paso.

Y acá hay un detalle de Adam que lo hace literal. Adam divide el gradiente por su
propia magnitud típica, así que el gradiente crudo **se cancela** y lo que queda
es aproximadamente:

```
paso ≈ ± lr           (independiente de cuán grande sea el gradiente)
```

Con lo cual el `lr` **es** el tamaño del paso, en unidades del parámetro. Medido
sobre las primeras épocas (`|Δ|` promedio por época):

| grupo | lr | paso medido ep 0 | ep 2 | ep 5 |
|---|---|---|---|---|
| pesos | 0.05 | 5.00e-2 | 4.59e-2 | 2.92e-2 |
| físicos | 0.02 | 2.00e-2 | 1.96e-2 | 1.75e-2 |
| red | 0.003 | 1.63e-4 | 2.51e-3 | 1.50e-3 |

El paso coincide con el `lr` casi exactamente. Entonces "17× más lenta" quiere
decir esto: **en una época, un peso sináptico puede cambiar 0.05 y un peso de la
red sólo 0.003.** Para recorrer la misma distancia, la red necesita 17 veces más
épocas.

> Esto también explica la tabla de gradientes de 5.1: aunque la red tiene el
> gradiente **más grande** (4.6e-3 contra 4.0e-4), avanza más lento igual. Con
> Adam el gradiente decide la *dirección*, no la *velocidad*.

**¿Para qué sirve?** Es un intento de darle ventaja al backbone: que los 10
parámetros se acomoden primero, y que la red sólo cubra lo que quede. La sección
siguiente muestra que funciona a medias.

### 5.3 Cómo colaboran realmente: la carrera por explicar el error

Esto es lo que hace especial al gray-box, y también su mayor riesgo.

Los dos términos **se suman** para dar la misma derivada, y hay **una sola loss**.
O sea que no tienen tareas asignadas: compiten por explicar el mismo error. Cada
pedacito de error que tapa uno, el otro ya no lo ve.

```
error observado  =  lo que explica f_WC(θ)  +  lo que explica g_φ
                    └── ajustando θ ──┘        └── ajustando 1218 pesos ──┘
```

El reparto **no está decidido de antemano**: sale de quién llega primero. Y ahí
está el problema, porque **buena parte del reparto es arbitraria**: el 67% del
mismatch se puede imitar moviendo parámetros (sección 6.1). En esa zona
compartida, quien la ocupe depende sólo de la dinámica del entrenamiento.

**Qué pasa de verdad.** Corrí una variante B siguiendo las dos partes en paralelo:

| época | loss | error param | `|g|` | `|backbone|` | `g` respecto del backbone |
|---|---|---|---|---|---|
| 0 | 5.8e-3 | 46.9% | 0.005 | 0.096 | 5% |
| 100 | 1.9e-3 | **55.4%** | 0.028 | 0.035 | **79%** |
| 200 | 8.4e-4 | 49.6% | 0.024 | 0.024 | **99%** |
| 400 | 3.4e-4 | 44.2% | 0.017 | 0.024 | 69% |
| 800 | 2.9e-4 | 44.8% | 0.015 | 0.026 | 57% |

Se leen tres fases:

1. **Arranque (ep 0).** `g ≡ 0` por construcción: el modelo es Wilson-Cowan puro y
   todo el error lo tienen que explicar los parámetros.

2. **La red se adelanta (ep 100-200).** `g` crece de 0.005 a 0.028 y llega a valer
   **tanto como el backbone entero**. Y justo ahí el error paramétrico **empeora**,
   de 46.9% a 55.4%. Es exactamente el modo de falla que se teme: la red tapa el
   error antes de que los parámetros se acomoden, y los parámetros se quedan en
   valores malos porque ya no sienten presión para mejorar.

3. **Reacomodo (ep 200-800).** La red **se achica** (0.028 → 0.015) mientras los
   parámetros mejoran (55.4% → 44.8%). El backbone recupera terreno que la red
   había ocupado.

**La conclusión honesta: el freno de la red ayuda pero no alcanza.** El lr 17×
más chico no evita que la red se adelante entre las épocas 100 y 200; sólo hace
que después se pueda revertir parcialmente. Y el error paramétrico final (44.8%)
es *peor* que el de la época 0 —cuando el modelo no había aprendido nada— lo cual
dice que la mayor parte de lo que la red explica es zona compartida, no física nueva.

> **Por eso las variantes B/C/D existen.** No son adornos: son tres formas de
> restringir a la red para que la competencia sea más pareja. B le saca
> información (no ve el estímulo), C le limita el tamaño, D le prohíbe
> directamente ocupar la zona compartida.
>
> Y por eso el entrenamiento por fases (sección 8.2) es la mejora más prometedora:
> congelar la red durante las primeras ~200 épocas ataca exactamente la fase 2
> de esa tabla, que es donde se hace el daño.

### 5.4 El gradiente atraviesa la integración entera

Esto es lo que hace que sea una Neural ODE y no un ajuste punto a punto. El
backward recorre los 100 pasos del RK4 hacia atrás — cada paso llama 4 veces al
modelo, así que son ~400 evaluaciones encadenadas.

Verificado: si se calcula la loss usando sólo el primer paso, `|grad|` en los
pesos da 2.8e-7; usando los 101 instantes da 4.0e-4. **Tres órdenes de magnitud
de diferencia** — o sea que el error de los pasos 2 a 100 sí contribuye, y el
gradiente realmente se propaga hacia atrás en el tiempo.

### 5.5 El clip de gradiente

`clip_grad_norm_(..., 10.0)` antes de cada paso. Está porque en un rollout de 100
pasos, un parámetro que hace al sistema inestable produce un gradiente enorme, y
sin clip un solo paso puede tirar el entrenamiento a una región de la que no
vuelve.

### 5.6 El refinamiento final con L-BFGS

Después de las 1500 épocas de Adam vienen 60 pasos de L-BFGS
(`graybox_train.py:279`), y acá **sí hay una asimetría** que conviene conocer:

```python
params = [model.raw_w] + phys_raw + struct_raw   # <- la red NO esta
```

L-BFGS optimiza sólo los parámetros físicos, **no la red**. El motivo es que
L-BFGS aproxima la curvatura y funciona bien en pocas dimensiones (10), no en
1218. Además su `closure` recalcula la loss **sin las penalizaciones**, sólo el
término de datos.

O sea que la última palabra sobre los parámetros la tiene un optimizador que ve
la red congelada y no ve la regularización. Es defendible —afina los parámetros
con la corrección ya establecida— pero es una decisión que vale la pena tener
presente al interpretar resultados.

---

## 6. Las variantes y las penalizaciones

### 6.1 El problema que las motiva

La red es un aproximador universal sobre el mismo dominio que el backbone.
Entonces:

> Para **cualquier** `θ` equivocado existe un `g` que compensa el error y
> reproduce los datos igual de bien. Con `g` libre, los 10 parámetros dejan de
> ser identificables.

El síntoma es traicionero: el MSE mejora y los parámetros empeoran. Por eso todo
el proyecto reporta las dos métricas juntas.

### 6.2 Las cinco variantes

| variante | `g` recibe | penalización | idea |
|---|---|---|---|
| `whitebox` | — | — | sin red: el baseline |
| `A` | `I,E,P,Q` | ninguna | la ingenua |
| `B` | `I,E` | ninguna | el estímulo sólo lo puede explicar `θ` |
| `C` | `I,E` | `λ‖g‖²` | prior de "corrección mínima" |
| `D` | `I,E` | `λ‖proyección‖²` | prohibirle a `g` ser redundante |
| `S` | — | — | corrección **física** (3 parámetros, no red) |

**Por qué B ayuda:** al no ver el estímulo, toda la dependencia de `P,Q` queda
forzada dentro del backbone. Como `P,Q` son conocidos y ricos, eso ancla la
escala de los parámetros.

### 6.3 La penalización de la variante D

Es la parte principista y la más fácil de entender mal. No penaliza que `g` sea
*grande*, sino que sea **redundante** con los parámetros:

1. Se calculan las sensibilidades `S = ∂f_backbone/∂θ` en N puntos del dataset
   (diferencias centradas, `dynamics.py`).
2. Se busca el `δθ` que mejor imita a `g`: `c = argmin ‖S·c − G‖`.
3. Se penaliza `‖S·c‖²` — la parte de `g` que un cambio de parámetros ya podía
   explicar.

**El detalle que la hace funcionar:** la proyección es en el **espacio de
funciones** (`g` evaluada en los N puntos a la vez), no punto a punto. Punto a
punto no serviría: 10 columnas en `R²` generan todo `R²` y la penalización
mataría `g` por completo.

Las sensibilidades se recalculan cada 25 épocas, porque dependen de `θ` que va
cambiando.

> **El punto cero no es cero.** Medido en ε=1: el `Δf` **verdadero** ya puntúa
> 0.67 de redundancia, y una red con pesos aleatorios puntúa 0.47 sólo por ser
> una función suave. Una corrección perfecta puntuaría 0.67, no 0. Bajar de ahí
> —lo que hace D con λ grande— **aleja** a `g` de la física real.

---

## 7. Cómo se evalúa

### 7.1 La prueba dura: rollout open-loop

`open_loop_mse()` integra la trayectoria **completa** sin reinicios: se le da
sólo el estado inicial y el estímulo, y tiene que generar los 4000 pasos por su
cuenta.

Es **40 veces más largo** que lo que vio al entrenar (5 ms contra 200 ms). Por
eso un modelo puede tener excelente loss de ventana y fallar acá — y por eso esta
métrica no es redundante con la de entrenamiento.

### 7.2 El split es por escenario, no por puntos

De cada familia de estímulo se reserva un escenario **entero** para test. Así el
test mide si generaliza a un **estímulo nuevo**, no a instantes nuevos de un
estímulo que ya vio. Es una prueba mucho más exigente.

### 7.3 Las métricas que se reportan juntas

- **error paramétrico** (%): qué tan lejos quedaron los 10 parámetros.
- **MSE open-loop** en test: qué tan bien predice.
- **`g_rel`**: tamaño de la corrección respecto del campo de Wilson-Cowan.
- **`frac_redundante`**: cuánto de lo que hace `g` lo podría haber hecho `θ`.

Las dos primeras se reportan siempre juntas, porque **pueden moverse en
direcciones opuestas** y mirar una sola engaña.

---

## 8. Cómo se podría mejorar el entrenamiento

Ordenadas por relación entre lo que cuestan y lo que probablemente den.

### 8.1 Continuidad entre ventanas ⭐

**El hueco más claro que tiene el entrenamiento actual.** Cada ventana arranca
desde el dato medido y termina donde termina; nadie pide que el final de una
ventana coincida con el arranque de la siguiente. El modelo puede tener 507
tramos de 5 ms excelentes y una trayectoria larga mala — que es exactamente lo
que se observa.

El multiple shooting clásico agrega justamente ese término:

```python
loss = data_loss + lam_cont * ((x0[siguiente] - traj[-1][actual]) ** 2).mean()
```

Y no habría que inventarlo: **ya está implementado** en
`scripts/train_real_output.py` (la variable `lam_cont`), pero no se trasladó al
trainer del gray-box. Es de las cosas más baratas de probar.

### 8.2 Entrenamiento por fases o alternado

Hoy todo se mueve junto. Alternativas que atacan directamente la competencia
entre `θ` y la red:

- **Curriculum:** primero sólo el backbone (red congelada), después liberar la
  red. Deja que los parámetros expliquen todo lo que pueden antes de que la red
  entre a tapar.
- **Alternado:** unos pasos de `θ`, unos de `g`, iterando.
- **Descongelado gradual:** subir `lr_g` de 0 a su valor durante las primeras
  épocas, en vez del arranque brusco.

### 8.3 Horizonte creciente

Empezar con ventanas cortas (20 pasos) y alargarlas durante el entrenamiento
hasta 200-400. Es estándar en Neural ODEs: las ventanas cortas dan un problema
bien condicionado al principio, y las largas fuerzan estabilidad al final.
Ataca el mismo problema que 8.1 desde otro ángulo.

### 8.4 Ponderar los dos canales

La loss actual promedia `I` y `E` con el mismo peso, pero no se mueven igual: el
desvío de `E` es 0.107 y el de `I` 0.048, así que **`E` pesa 2,2× más** en el
gradiente sólo por tener más amplitud. Normalizar cada canal por su desvío haría
que los dos cuenten igual.

Lo mismo aplica entre escenarios: los de amplitud grande dominan sobre los
chicos.

### 8.5 Los tres puntos numéricos

- **El instante 0** de cada ventana tiene error exacto 0 y entra en el promedio.
  Sacarlo es una línea y hace la loss ~1% más informativa.
- **L-BFGS sin penalizaciones** (sección 5.6): la última palabra sobre `θ` la
  tiene un optimizador que ignora la regularización. Incluir `pen` en el
  `closure` sería más consistente.
- **Sin scheduler**: el lr es constante las 1500 épocas. Un cosine o un
  `ReduceLROnPlateau` normalmente ayuda en la fase final.

### 8.6 Cosas que faltan como infraestructura

- **Sin early stopping ni validación durante el entrenamiento.** Se corren 1500
  épocas fijas y el modelo final es el último, no el mejor. No hay forma de saber
  si sobreajustó a mitad de camino.
- **Una sola semilla.** Todos los resultados salen de `seed=0`. Sin repeticiones
  no se sabe cuánto de la diferencia entre variantes es real y cuánto es ruido de
  inicialización — y las diferencias reportadas (35.9% contra 35.6%) son del
  orden de lo que podría explicar el azar.
- **Los hiperparámetros no se barrieron.** `window=100`, `lr=5e-2/2e-2/3e-3` y
  `hidden=32` se fijaron por criterio, no por búsqueda.

### 8.7 Lo que cambiaría el planteo, no el entrenamiento

Dos cosas que no son "mejorar el optimizador" sino cambiar el problema, y que el
trabajo ya identificó como las más prometedoras:

- **Diseñar el conjunto de estímulos** para que el mismatch deje de parecerse a
  un cambio de parámetros. Ningún ajuste de entrenamiento arregla una
  degeneración que está en los datos.
- **Corrección con forma física** en vez de red: 3-4 parámetros interpretables
  en lugar de 1218 pesos. Es identificable y no compite con `θ` del mismo modo.

---

## 9. El ciclo completo, de una

```
507 ventanas de 5 ms, desde el estado medido
        ↓
rollout RK4 x 100 pasos  (~400 evaluaciones del modelo encadenadas)
        ↓
pred (101, 507, 2)   vs   tgt (101, 507, 2)     ->  102.414 numeros
        ↓
MSE + penalizaciones
        ↓
UN backward  ->  gradiente a los 10 parametros Y a los 1218 pesos, a la vez
        ↓
clip a norma 10  ->  Adam con 3 learning rates
        ↓
x 1500 epocas
        ↓
60 pasos de L-BFGS  (solo los parametros, sin penalizaciones)
        ↓
evaluacion: rollout completo de 200 ms sobre estimulos NUEVOS
```

---

## Conexiones

- `docs/recorrido_estimulos_y_entrenamiento.md` — de dónde salen los datos
- `docs/graybox_manual_completo.md` — qué se midió y qué dio
- `tests/test_graybox.py` — qué protege cada test de esta maquinaria
