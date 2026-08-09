# Wilson-Cowan: la ecuación original y la reducida

> De dónde sale el término de refractariedad, por qué la forma que usamos lo
> descarta, y por qué eso es exactamente el desajuste estructural del proyecto.
>
> Referencia: Wilson & Cowan (1972), *Excitatory and inhibitory interactions in
> localized populations of model neurons*, Biophysical Journal 12(1):1-24.
> Implementación: `src/wilson_cowan/model.py`.

---

## 1. De dónde sale todo: la forma integral

Wilson y Cowan parten de una idea concreta: **la fracción de la población que
puede disparar en el instante `t` no es toda la población.** Los que dispararon
recientemente están en período refractario.

Si `r` es la duración del período refractario, la fracción **bloqueada** es la
que disparó en la ventana `[t−r, t]`:

$$
\text{bloqueada}(t) = \int_{t-r}^{t} E(t')\,dt'
$$

Y entonces la ecuación original (antes de cualquier simplificación) es:

$$
E(t+\tau) \;=\; \underbrace{\left[1 - \int_{t-r_e}^{t} E(t')\,dt'\right]}_{\text{fracción disponible}} \cdot\; S_e\!\left(\int_{-\infty}^{t}\alpha(t-t')\left[w_{EE}E(t') - w_{EI}I(t') + P(t')\right]dt'\right)
$$

Se lee así: **la actividad futura = (los que pueden disparar) × (qué fracción de
ésos efectivamente dispara)**. El segundo factor es la sigmoidea, que convierte
la entrada total en una tasa.

---

## 2. La forma diferencial *con* refractariedad

Aplicando *time coarse-graining* (promediar en una ventana corta), la integral
del período refractario se aproxima por `r_e·E(t)` y las dos ecuaciones quedan:

$$
\boxed{
\begin{aligned}
\tau_e \frac{dE}{dt} &= -E + \left(1 - r_e E\right)\, S_e\!\left(w_{EE}E - w_{EI}I + P - \theta_e\right)\\[4pt]
\tau_i \frac{dI}{dt} &= -I + \left(1 - r_i I\right)\, S_i\!\left(w_{IE}E - w_{II}I + Q - \theta_i\right)
\end{aligned}}
$$

**Ésta es la ecuación "original"** en el sentido que usamos en el proyecto: la
que conserva el factor refractario.

| símbolo | qué es |
|---|---|
| `E`, `I` | fracción activa de cada población (0 a 1) |
| `r_e`, `r_i` | **período refractario** — el término que la forma reducida elimina |
| `τ_e`, `τ_i` | constantes de tiempo |
| `w_EE … w_II` | pesos sinápticos |
| `P`, `Q` | estímulo externo a cada población |
| `θ_e`, `θ_i` | umbrales |
| `S(·)` | sigmoidea |

---

## 3. La forma reducida (la que usamos como modelo)

Se hace `r_e = r_i = 0`, o sea **se supone que la población siempre está
íntegramente disponible**. El paréntesis vale 1 y desaparece:

$$
\boxed{
\begin{aligned}
\frac{dE}{dt} &= \frac{1}{\tau_e}\left[-E + S(a_e\,u_e) - k_e\right], \qquad u_e = w_{EE}E - w_{EI}I + P - \theta_e \\[4pt]
\frac{dI}{dt} &= \frac{1}{\tau_i}\left[-I + S(a_i\,u_i) - k_i\right], \qquad\; u_i = w_{IE}E - w_{II}I + Q - \theta_i
\end{aligned}}
$$

con

$$
S(z) = \frac{1}{1+e^{-z}}, \qquad k_e = \frac{1}{1+e^{\,a_e\theta_e}}, \qquad k_i = \frac{1}{1+e^{\,a_i\theta_i}}
$$

**Dos detalles de esta versión** (`model.py`):

- **El umbral entra dentro de `u`**, no como argumento aparte de la sigmoidea.
  Por eso `sigmoid(u, a)` sólo recibe la ganancia.
- **`k_e` y `k_i` son offsets de reposo:** son el valor de la sigmoidea con
  entrada nula. Restarlos hace que `E = I = 0` sea un equilibrio exacto — sin
  ellos el sistema tendría actividad de fondo aun sin estímulo. No son
  parámetros libres: se recalculan de `a` y `θ`.

### Valores nominales

| | `τ` | `w_EE` | `w_EI` | `w_IE` | `w_II` | `a` | `θ` |
|---|---|---|---|---|---|---|---|
| **E** | 1.0 | 6.4 | 4.8 | — | — | 1.2 | 2.8 |
| **I** | 2.0 | — | — | 6.0 | 1.2 | 1.0 | 4.0 |

---

## 4. La diferencia, en una línea

$$
\Delta f \;=\; \underbrace{f_{1972}}_{\text{con }(1-rE)} - \underbrace{f_{\text{reducido}}}_{\text{sin él}} \;=\; -\,r\,x\cdot S(a\,u)
$$

**El desajuste es proporcional a la actividad.** En reposo (`E = I = 0`) vale
exactamente cero — las dos versiones coinciden. Cuanto más activa está la
población, más se separan.

Con `r = 0.10` y el pico medido `E = 0.78`, el factor vale `1 − 0.078 = 0.922`:
la población responde un **7.8% menos** que en la forma reducida.

---

## 5. Por qué esto es el corazón del proyecto

El experimento central usa esta asimetría:

| | qué ecuación usa |
|---|---|
| **el simulador** (la "planta") | la de 1972, **con** `(1 − r·x)` |
| **el modelo** (lo que entrenamos) | la reducida, **sin** ese término |

Y ésa es la gracia del planteo: **el desajuste no es una perturbación
inventada.** Es literalmente el término que el propio paper deriva y que la forma
de trabajo descarta por simplicidad. La física que le falta al modelo es física
real, publicada y citable — no un ruido agregado a mano para que el problema
parezca difícil.

Además tiene dos propiedades que lo hacen un buen banco de pruebas:

- **Preserva el equilibrio de reposo** → el desajuste es un cambio genuino de
  dinámica, no un corrimiento constante que se absorbería ajustando `θ`.
- **Es función pura del estado** `(I, E)` → una corrección `g_φ(I,E)` *puede*
  representarlo en principio. Medido: **R² = 0.97**.

> En el proyecto la refractariedad va acompañada de una segunda perturbación (el
> actuador optogenético), que sí tiene memoria propia y **no** es capturable. La
> combinación es deliberada: una parte aprendible y una que no, para medir dónde
> está el techo. Ver `docs/las_dos_perturbaciones.md`.

---

## 6. En código

```python
# La forma REDUCIDA — el modelo (model.py, dentro de rhs)
u_e = wEE*E - wEI*I + P - thetae
dE  = (1.0/te) * (-E + sigmoid(u_e, ae) - ke)

# La forma de 1972 — el simulador, vía el gancho gains()
g_e = 1.0 - r_e * E                            # uncertainty.py:162
dE  = (1.0/te) * (-E + g_e*sigmoid(u_e, ae) - ke)
```

La refractariedad no está escrita como una ecuación aparte: entra por el gancho
`gains()`, que multiplica la salida de la sigmoidea. Por eso agregarla o
sacarla es cambiar un solo objeto, sin tocar el simulador.
