# Mapa del código: dónde está cada cosa

> El ciclo completo, de los estímulos al modelo entrenado, y en qué archivo pasa
> cada paso.

---

## El ciclo en una vista

```
  1. ESTÍMULOS            src/wilson_cowan/model.py       (líneas 105-300)
         ↓                  funciones P(t), Q(t)
  2. SIMULADOR            src/wilson_cowan/model.py       (clase WilsonCowan)
         ↓                  integra las ecuaciones WC
     + PERTURBACIONES     src/wilson_cowan/uncertainty.py
         ↓                  la física extra que el modelo NO tiene
  3. DATASET              src/data/generate.py  →  .npz
         ↓                  guarda trayectorias (I, E, P, Q)
  4. MODELO GRAY-BOX      src/neural_ode/dynamics.py      (clase GrayBoxWC)
         ↓                  Wilson-Cowan + red neuronal
  5. INTEGRADOR           src/neural_ode/integrate.py     (43 líneas)
         ↓                  RK4 diferenciable
  6. ENTRENAMIENTO        src/neural_ode/graybox_train.py
         ↓
  7. EXPERIMENTOS         scripts/exp_*.py  →  results/uncertainty/*.json
```

---

## 1. Los estímulos — `src/wilson_cowan/model.py`

Están al principio del archivo, antes de la clase del simulador. Cada uno es una
**función que devuelve otra función** `P(t)`: se configura una vez y después se
la llama en cada instante.

| función | línea | qué genera |
|---|---|---|
| `box_pulse` | 105 | un escalón simple (prende y apaga) |
| `sine_pulse` | 131 | una sinusoide |
| `chirp_pulse` | 157 | frecuencia que barre de `f0` a `f1` |
| `aprbs_pulse` | 199 | escalones de amplitud y duración aleatorias |
| `prbs_pulse` | 259 | escalones binarios aleatorios |
| `theta_gamma_pulse` | 230 | gamma modulado por theta (ritmo cerebral realista) |
| `poisson_pulse` | 276 | pulsos en tiempos aleatorios |
| `square_wave_pulse` | 245 | onda cuadrada |

> **Detalle importante:** las familias aleatorias calculan su cronograma **una
> sola vez** a partir de la semilla, no en cada llamada. Si sortearan en cada
> instante, el estímulo no sería una función del tiempo y el integrador RK4
> —que evalúa el mismo `t` varias veces— vería valores distintos.

**Dónde se eligen los estímulos de cada dataset:** `scripts/gen_multi_dataset.py`
(los 20 escenarios del catálogo, con la marca de cuáles son de test).

---

## 2. El simulador — `src/wilson_cowan/model.py`, clase `WilsonCowan` (línea 302)

Es la "planta": genera los datos que después se tratan como si fueran mediciones
reales.

Lo importante está en la **función derivada** (~línea 370), donde se arma:

```python
u_e = wEE·E − wEI·I + P_eff − θe      # entrada total a cada población
dE  = (1/te)·( −E + g_e·S(ae·u_e) − ke )
```

Ahí mismo están los **5 ganchos** por donde entran las perturbaciones. Cada uno
es un punto distinto de la ecuación:

| gancho | dónde interviene |
|---|---|
| `inputs()` | cambia `P,Q` por `P_eff,Q_eff` ← **actuador** |
| `drive()` | suma una corriente a la entrada |
| `gains()` | multiplica la salida de la sigmoidea ← **refractariedad** |
| `weights()` | cambia los pesos sinápticos |
| `deriv()` | suma directo a `dI/dE` |

---

## 3. Las perturbaciones — `src/wilson_cowan/uncertainty.py` (615 líneas)

El archivo más grande. Contiene **9 familias** de física faltante, todas
combinables. Las dos que se usan por defecto:

- **`Refractoriness`** (línea 137) — el factor `(1−r·E)` del Wilson-Cowan de 1972.
  Entra por `gains()`.
- **`Actuator`** (línea 169) — retardo del canal óptico + saturación. Entra por
  `inputs()` y tiene **2 estados ocultos** propios.

**`default_uncertainty(eps)`** (línea 562) es la función que se usa en todos lados:
devuelve las dos juntas graduadas por una sola perilla. `eps=0` → sin perturbación.

---

## 4. Generar el dataset — `src/data/generate.py`

`generate_dataset()` (línea 33) toma los estímulos + el simulador + la
perturbación, corre las simulaciones y guarda un `.npz`.

**Lo que guarda y por qué importa la distinción:**

| campo | qué es |
|---|---|
| `I, E` | las trayectorias (lo que el modelo tiene que reproducir) |
| `P, Q` | estímulo **comandado** ← lo único que ve el entrenamiento |
| `P_eff, Q_eff` | estímulo **efectivo** (después del actuador) — sólo diagnóstico |
| `dfI, dfE` | el desajuste verdadero — sólo para evaluar |
| `is_test` | qué escenarios se reservan |

> Usar `P_eff` para entrenar sería hacer trampa: le estaríamos dando el resultado
> de la perturbación que justamente tiene que descubrir.

**Los scripts que arman los datasets concretos:**
`scripts/gen_multi_dataset.py` (limpio) y `scripts/gen_uncertain_dataset.py`
(con perturbación, un `.npz` por nivel de ε).

---

## 5. El modelo gray-box — `src/neural_ode/dynamics.py`, clase `GrayBoxWC`

Las dos mitades, separadas a propósito en métodos distintos:

| método | qué es |
|---|---|
| `backbone()` (línea 167) | sólo Wilson-Cowan, los 10 parámetros |
| `g_out()` | sólo la red neuronal (MLP, 2 capas de 32, 1218 pesos) |
| `forward()` | la suma de los dos = la derivada final |
| `backbone_sensitivities()` | `∂f/∂θ` por diferencias finitas |

Están separados para poder medir cuánto aporta cada parte y para que el
controlador pueda cancelar la corrección.

**Dos detalles del archivo:**
- Los parámetros se guardan **crudos** y pasan por `softplus` → siempre positivos
  por construcción.
- La red se inicializa con la última capa en **cero exacto** → en la época 0 el
  modelo es Wilson-Cowan puro.

---

## 6. El integrador — `src/neural_ode/integrate.py` (43 líneas)

El archivo más chico y uno de los más importantes.

```python
def rk4_step(f, x, P, Q, dt):    # línea 21 — un paso de Runge-Kutta 4
def rollout(f, x0, P_seq, Q_seq, dt):   # línea 30 — encadena N pasos
```

**Por qué existe** si ya hay integradores en scipy: éste está escrito en **torch**,
así que el gradiente puede atravesarlo. Es lo que permite entrenar contra
trayectorias en vez de contra derivadas punto a punto.

Es de **paso fijo** a propósito: un integrador adaptativo elige sus pasos según el
error, y esa decisión no es diferenciable.

---

## 7. El entrenamiento — `src/neural_ode/graybox_train.py`

| parte | línea | qué hace |
|---|---|---|
| `make_windows()` | 46 | corta las trayectorias en ventanas de 100 pasos |
| `VARIANTS` | 185 | las 6 configuraciones (whitebox, A, B, C, D, S) |
| `projection_operator()` | — | la penalización de redundancia (variante D) |
| `fit()` | 210 | **el bucle de entrenamiento** |
| `open_loop_mse()` | 139 | evaluación: rollout completo sin reinicios |
| `load_split()` | 332 | carga el `.npz` y separa train/test |

El corazón es el bucle en `fit()`: un `backward()` que mueve **a la vez** los 10
parámetros y los 1218 pesos de la red, y al final 60 pasos de L-BFGS sobre los
parámetros solos.

---

## 8. Los experimentos — `scripts/exp_*.py`

Cada uno responde una pregunta y deja un JSON en `results/uncertainty/`:

| script | pregunta |
|---|---|
| `exp_f1_characterize.py` | ¿cuánto deforma la perturbación? |
| `exp_f2_rigidity_cost.py` | ¿cuánto daña al white-box? |
| `exp_f3_graybox.py` | ¿la corrección lo arregla? ← **el principal** |
| `exp_f4_fim_hybrid.py` | ¿por dónde se filtra el error? |
| `exp_f4b_geometria_mismatch.py` | ¿qué fracción es imitable con parámetros? |
| `exp_f5_functional_recovery.py` | ¿aprendió física de verdad? |
| `exp_f6_closed_loop.py` | ¿sirve para controlar? |
| `exp_f7_controls.py` | ¿con qué tipo de física sí funciona? |

`scripts/informe_incertidumbre.py` junta todos los JSON en un informe de texto.

---

## Si tenés que tocar una sola cosa

| querés cambiar… | andá a… |
|---|---|
| qué estímulos se usan | `scripts/gen_multi_dataset.py` |
| agregar un tipo de estímulo | `src/wilson_cowan/model.py` (~línea 105) |
| qué física le falta al modelo | `src/wilson_cowan/uncertainty.py` |
| la arquitectura de la red | `src/neural_ode/dynamics.py` |
| learning rates, épocas, ventana | `TrainConfig` en `graybox_train.py:166` |
| cómo se mide el resultado | `open_loop_mse()` en `graybox_train.py:139` |

---

*Más detalle: `docs/neural_ode_entrenamiento_detallado.md` (el entrenamiento),
`docs/las_dos_perturbaciones.md` (la física agregada),
`docs/reproduccion_dinamica_graybox.md` (los resultados).*
