# Recorrido: del estímulo al modelo entrenado

> Qué archivo hace qué, cómo se fabrica un estímulo, cómo entra al simulador y
> cómo llega al entrenamiento. Con las perillas de cada cosa y recetas para
> modificarlas.

---

## 0. El mapa en una tabla

Todo el camino son **seis piezas**, en este orden:

| # | archivo | qué hace |
|---|---|---|
| 1 | `src/wilson_cowan/model.py` (líneas 105–295) | **la librería**: 9 funciones que fabrican estímulos |
| 2 | `scripts/gen_multi_dataset.py` → `build_scenarios()` | **el catálogo**: elige 20 combinaciones concretas de (P, Q) |
| 3 | `src/data/generate.py` → `generate_dataset()` | **el motor**: simula UNA trayectoria y la empaqueta |
| 4 | `scripts/gen_uncertain_dataset.py` | genera el barrido de ε y guarda el Δf verdadero |
| 5 | `src/neural_ode/graybox_train.py` | **el entrenamiento** |
| 6 | `src/neural_ode/integrate.py` | el integrador diferenciable (RK4) que usa el entrenamiento |

---

## 1. Un estímulo NO es un array: es una función del tiempo

Ésta es la idea que hay que entender antes que nada, porque explica cómo está
escrito todo lo demás.

```python
from src.wilson_cowan import box_pulse

P = box_pulse(amplitude=0.8, t_on=10.0, t_off=190.0)   # fabrica la función
P(25.0)   # → 0.8    ¿cuánto vale el estímulo en t = 25 ms?
P(5.0)    # → 0.0    fuera de la ventana
```

**Por qué función y no vector.** El integrador (`solve_ivp` o el RK4) llama al
estímulo en instantes **arbitrarios, repetidos y desordenados** — un paso RK4
evalúa en `t`, `t+h/2`, `t+h/2`, `t+h`. Si el estímulo fuera un array indexado
habría que interpolar; siendo una función, se lo pregunta directo.

**La consecuencia para los estímulos aleatorios.** APRBS, PRBS y Poisson son
aleatorios, pero `P(t)` tiene que devolver **siempre lo mismo para el mismo `t`**.
Si no, el integrador ve un sistema que cambia bajo sus pies y el resultado no es
reproducible. Por eso esas tres **pre-calculan su agenda** a partir de la semilla
al construirse, y después `P(t)` sólo consulta en qué tramo cae `t`:

```python
# aprbs_pulse, model.py:205-222 — resumido
rng = np.random.default_rng(seed)
bordes, valores = [t_on], []          # se arma TODA la agenda al construir
while t < t_off:
    t += rng.uniform(dwell_min, dwell_max)
    bordes.append(min(t, t_off))
    valores.append(rng.uniform(amp_min, amplitude))

def f(t):                              # después sólo se CONSULTA
    i = np.searchsorted(bordes, t, side="right") - 1
    return valores[i]
```

---

## 2. Las 9 familias y sus perillas

Esto es el "delta" de cada estímulo: qué podés cambiar y qué efecto tiene.
Todas están gateadas a una ventana `[t_on, t_off)` y valen 0 fuera.

| función | perillas | qué controla cada una |
|---|---|---|
| `box_pulse` | `amplitude, t_on, t_off` | el escalón más simple. Casi DC: excita poco |
| `square_wave_pulse` | `+ freq, duty` | tren de pulsos. `duty` = fracción del período encendido |
| `aprbs_pulse` | `+ dwell_min, dwell_max, seed, amp_min` | **el caballo de batalla**: escalones de amplitud Y duración al azar |
| `prbs_pulse` | `+ bit_period, seed` | binaria: cada `bit_period` vale 0 o `amplitude` |
| `poisson_pulse` | `+ rate, pulse_width, seed` | pulsos en tiempos aleatorios. `rate` = eventos por ms |
| `theta_gamma_pulse` | `+ f_gamma, f_theta, duty` | ráfagas rápidas moduladas por una envolvente lenta |
| `chirp_pulse` | `+ f0, f1` | barrido de frecuencia f0 → f1 |
| `sine_pulse` | `+ freq, offset` | senoide (no se usa: no es on/off) |
| `multisine_pulse` | `+ freqs, offset` | suma de senoides (idem) |

### Las tres perillas que más importan

**`amplitude`** — cuán fuerte pegás. Para identificar un sistema **no lineal**
hay que recorrer muchas amplitudes, porque la sigmoidea responde distinto en cada
punto de operación. Por eso APRBS (que varía la amplitud tramo a tramo) identifica
mucho mejor que PRBS (que sólo tiene dos niveles).

**`dwell_min` / `dwell_max`** (APRBS) — cuánto dura cada escalón. Esto fija el
contenido de frecuencia. Si los tramos son mucho más largos que `te=1 ms` el
sistema alcanza el equilibrio en cada uno y no ves la dinámica; si son mucho más
cortos, el sistema promedia y no responde. Los valores usados (2–8 ms) están en
la banda donde el sistema sí reacciona.

**`amp_min`** (APRBS) — el piso de amplitud. Con `amp_min=0` cada tramo puede caer
cerca de cero y la excitación "se apaga" seguido. Subirlo mantiene la señal viva
sin perder el barrido de amplitudes.

### La conversión de frecuencias

Todo el proyecto trabaja en **milisegundos**. Las frecuencias se escriben en Hz y
se convierten con un helper (`gen_multi_dataset.py:54`):

```python
def hz(f_hz): return f_hz / 1000.0     # Hz → ciclos por ms
square_wave_pulse(amp, hz(50), ton, toff, 0.4)    # 50 Hz
```

---

## 3. El catálogo: cómo se arman los 20 escenarios

`scripts/gen_multi_dataset.py` → `build_scenarios()` (línea 63). Devuelve una
lista de tuplas `(etiqueta, función_P, función_Q, es_test)`:

```python
S = []
ton, toff = 10.0, 190.0

for amp in (0.4, 0.8, 1.2):
    S.append((f"box_a{amp}",
              box_pulse(amp, ton, toff),              # ← P, a la población E
              box_pulse(0.7 * amp, ton + 5, toff - 5), # ← Q, a la población I
              amp == 1.2))                             # ← ¿es de test?
```

Tres decisiones de diseño que están metidas ahí:

**P y Q descorrelacionados.** `Q` siempre tiene otra amplitud, otro timing y otra
semilla que `P`. Si fueran iguales el problema quedaría mal condicionado: no se
podría separar el efecto de una entrada del de la otra.

**El corte train/test es por escenario entero**, no por puntos sueltos. De cada
familia se reserva uno completo. Así el test mide si generaliza a un **estímulo
nuevo**, no a instantes nuevos de un estímulo que ya vio.

**Sin senoides puras.** Decisión del proyecto: los estímulos tienen que ser on/off
y ≥ 0 para ser realizables con optogenética. El chirp se conserva por su cobertura
espectral.

---

## 4. Cómo entra el estímulo al simulador

El modelo recibe las dos funciones al construirse:

```python
modelo = WilsonCowan(params=PARAMS, P=Pf, Q=Qf, perturbation=pert)
```

Y a partir de ahí hay **dos caminos** según haya perturbación o no.

### Camino A — Wilson-Cowan puro (`rhs`, model.py:329)

```python
u_i = wIE*E - wII*I + self.Q(t) - thetai      # ← acá entra Q
u_e = wEE*E - wEI*I + self.P(t) - thetae      # ← acá entra P
dI  = (1/ti) * (-I + S(u_i) - ki)
dE  = (1/te) * (-E + S(u_e) - ke)
```

El estímulo se suma a la **entrada total** de cada población, antes de la
sigmoidea. Lo integra `solve_ivp` (RK45 de paso adaptativo).

### Camino B — con perturbación (`perturbed_field`, model.py:365)

Acá aparece el **delta que importa para el gray-box**:

```python
P_cmd, Q_cmd = self.P(t), self.Q(t)                       # lo que COMANDÁS
P_eff, Q_eff = pert.inputs(t, I, E, extra, P_cmd, Q_cmd)  # lo que LLEGA
u_e = wEE*E - wEI*I + P_eff - thetae                      # el sistema usa el efectivo
```

Con el actuador optogenético, `P_eff` es `P_cmd` filtrado por el retardo del canal
y achatado por la saturación. **El dataset guarda los dos por separado**, y el
entrenamiento sólo puede usar `P_cmd`:

```python
# generate.py:105 — comentario del código
# P y Q son SIEMPRE los estimulos COMANDADOS. Si hay perturbacion en el canal
# de actuacion, el estimulo que realmente llego (P_eff) va aparte y es SOLO
# para diagnostico: entrenar con el seria hacer trampa.
```

Este camino usa **RK4 de paso fijo** con estado aumentado `[I, E, *estados_ocultos]`,
porque las perturbaciones con memoria (el retardo) y el ruido de proceso no se
pueden integrar con paso adaptativo.

---

## 5. De función a dataset: el muestreo

`generate_dataset()` (`src/data/generate.py:33`) es el que convierte la función en
números:

```python
t_eval = np.linspace(t_span[0], t_span[1], n_eval)   # 0..200 ms, 4000 puntos
sol = modelo.simulate(I0, E0, t_span, t_eval)        # integra
# y devuelve, evaluados en esa grilla:
{"t": ..., "I": ..., "E": ..., "y": E-I, "P": ..., "Q": ...}
```

Con los valores del proyecto: `T_SPAN=(0,200)` ms y `N_EVAL=4000` → **dt = 0,05 ms**.
Veinte escenarios × 4000 puntos = **80 000 muestras** de `(t, I, E, P, Q)`.

Todo se apila y se guarda en un único `.npz`:

```python
np.savez_compressed(OUT_PATH,
    t=t_ref,
    I=np.stack(I_all), E=np.stack(E_all),      # (20, 4000)
    P=np.stack(P_all), Q=np.stack(Q_all),      # (20, 4000)  ← el COMANDADO
    P_eff=..., Q_eff=...,                      # sólo diagnóstico
    dfI=..., dfE=...,                          # el Δf verdadero
    is_test=..., labels=..., dt=..., eps=...)
```

---

## 6. Cómo se entrena

### 6.1 Cargar y partir

```python
data = load_split("data/processed/uncertain/eps1.npz")   # graybox_train.py:332
# separa por is_test: I,E,P,Q (train) y I_te,E_te,P_te,Q_te (test)
```

### 6.2 Trocear en ventanas (multiple shooting)

`make_windows()` (línea 46). Cada trayectoria de 4000 pasos se parte en ventanas
de `W=100` pasos (≈ 5 ms), y **cada ventana arranca desde el estado observado**:

```python
x0  = [I[s,a], E[s,a]]                     # el estado real en el borde
Pw  = P[s, a:a+W]                          # el estímulo de esa ventana
tgt = [I[s,a:a+W+1], E[s,a:a+W+1]]         # lo que tiene que reproducir
```

**Por qué.** Integrar los 4000 pasos de un tirón hace que el gradiente explote o
se desvanezca. Reiniciando desde el dato real cada 100 pasos el problema queda
mucho mejor condicionado. Salen ~500 ventanas de las 13 trayectorias de train.

### 6.3 El rollout diferenciable

`src/neural_ode/integrate.py` — RK4 de paso fijo escrito en torch, así que
`loss.backward()` atraviesa la integración entera:

```python
def rk4_step(f, x, P, Q, dt):
    k1 = f(x, P, Q)
    k2 = f(x + 0.5*dt*k1, P, Q)      # P y Q constantes dentro del paso (ZOH)
    k3 = f(x + 0.5*dt*k2, P, Q)
    k4 = f(x + dt*k3, P, Q)
    return x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
```

El estímulo entra como **zero-order hold**: se mantiene constante durante el paso,
que es como llega un control muestreado real.

### 6.4 El bucle

```python
for ep in range(1500):
    pred = rollout(model, x0, Pw, Qw, dt)       # todas las ventanas a la vez
    loss = ((pred - tgt) ** 2).mean() + penalizaciones
    loss.backward(); opt.step()
# después: 60 pasos de L-BFGS para refinar
```

Con tres grupos de learning rate distintos (`TrainConfig`, línea 167):

| grupo | qué es | lr |
|---|---|---|
| `raw_w` | los 4 pesos sinápticos | 5e-2 |
| `raw_te, raw_ti, …` | los 6 parámetros físicos | 2e-2 |
| `g.parameters()` | la red de corrección | 3e-3 |

**Arranque ignorante**: los 10 parámetros parten de **1.0**. La red nunca ve los
valores verdaderos — sólo se usan al final para reportar el error.

---

## 7. Recetas para modificar cosas

### Cambiar la amplitud o la duración de un estímulo

En `gen_multi_dataset.py` → `build_scenarios()`:

```python
for amp in (0.4, 0.8, 1.2):        # ← acá
    ...
ton, toff = 10.0, 190.0            # ← y acá la ventana
```

Después hay que **regenerar el dataset**: `python scripts/gen_multi_dataset.py`
(o `gen_uncertain_dataset.py` si querés el barrido de ε).

### Agregar una familia nueva de estímulo

1. Escribir la función en `model.py` siguiendo el patrón: devuelve un `f(t)` puro,
   gateado a `[t_on, t_off)`, y si es aleatorio **pre-calcula la agenda**.
2. Exportarla en `src/wilson_cowan/__init__.py`.
3. Agregar los escenarios en `build_scenarios()`, reservando uno como test.
4. Regenerar el dataset.

### Cambiar la resolución temporal

`N_EVAL` en `gen_multi_dataset.py` (línea 47). Ojo: `dt` sale de ahí y el
entrenamiento lo lee del `.npz`, así que se propaga solo. Pero cambiarlo cambia
también cuántos pasos cubre la ventana de multiple shooting (`W=100`).

### Cambiar cuánta física le falta al modelo

`EPS_GRID` en `gen_uncertain_dataset.py` (línea 52). `ε=0` es Wilson-Cowan puro,
`ε=1` el punto nominal. La perilla está definida en
`src/wilson_cowan/uncertainty.py` → `default_uncertainty()`.

### Cambiar el entrenamiento

`TrainConfig` en `graybox_train.py:167`: `window`, `epochs`, `lbfgs_steps`, los
tres learning rates, `lam_norm` y `lam_orth` (las regularizaciones), `hidden`
(tamaño de la red).

---

## 8. El camino entero, de una

```
box_pulse(0.8, 10, 190)                      model.py       una función P(t)
        ↓
build_scenarios()                            gen_multi…     20 pares (P, Q) + flag test
        ↓
generate_dataset(params, P, Q, …)            generate.py    integra y muestrea
        ↓   simulate() → rhs / perturbed_field               ← acá entra al sistema
        ↓
.npz  {I, E, P, Q, is_test, dt, dfI, dfE}                   80 000 muestras
        ↓
load_split() → make_windows(W=100)           graybox_train  ~500 ventanas
        ↓
rollout() RK4 diferenciable                  integrate.py   ZOH del estímulo
        ↓
loss.backward() × 1500 + L-BFGS                             10 parámetros + red
```

---

## Conexiones

- Documento hermano: `graybox_manual_completo.md` (qué es la incertidumbre y qué se midió)
- Reproducir todo: `bash scripts/run_uncertainty_all.sh`
- Los estímulos, comparados entre sí: `docs/informe_estimulos.html`
