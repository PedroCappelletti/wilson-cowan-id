# Informe completo — Identificación de Wilson-Cowan y control con Neural ODE

Documento único y detallado de **todo lo hecho**: contexto, conceptos en lenguaje simple, la
librería de estímulos, la identificación paramétrica (PINN y Neural ODE), el control en lazo cerrado,
la robustez bajo ruido, y la bitácora de cada experimento con sus errores y correcciones. Pensado
para leerse a la par del informe visual `docs/informe_integral.html`.

> **Nota de lectura:** las secciones 1–4 son introductorias y en lenguaje simple. Las 5–11 son los
> resultados con su detalle técnico. El **Apéndice** es la bitácora cronológica completa (cada
> intento, error y corrección).

---

## 0. Resumen ejecutivo

| Resultado | Estado |
|---|---|
| Librería de estímulos ampliada (5 generadores nuevos tipo pulso + chirp) | ✅ |
| Identificación de los 4 pesos **en limpio** (PINN y Neural ODE) | ✅ wII ≤ 1.7% (PINN) · 0.42% (Neural ODE) |
| Neural ODE aprende la dinámica y generaliza | ✅ rollout MSE test 8.5e-5 |
| Controlador conectado a **ambas plantas** (simulador y Neural ODE) | ✅ RMSE ~idéntico |
| Controlador con parámetros **identificados (θ̂)** vs reales | ✅ no degrada el control |
| **Robustez bajo ruido** (identificación) | ✅ θ̂ máx ≤1.2% hasta σ=0.05 y 8.9% a σ=0.10 (vs 106% ingenuo) |
| Realismo interno de Q grande (no satura) | ✅ |
| Variación paramétrica · gray-box real · head-to-head PINN/NODE bajo ruido · calibración a unidades físicas | ⏳ pendiente |

**En una frase:** la cadena completa *identificar → diseñar controlador → controlar* funciona de
punta a punta, con varios estímulos realistas, con parámetros identificados, sobre la planta real y
sobre el modelo aprendido, y es robusta al ruido.

---

## 1. Contexto y objetivo

Proyecto del Concurso de Iniciación a la I+D ITBA 2026. Título: *"Identificación de modelos
poblacionales de neuronas mediante redes neuronales con validación orientada al control en lazo
cerrado"*. Tutor: Ricardo Sánchez-Peña.

**Objetivo central:** identificar los **parámetros** del modelo de Wilson-Cowan (no solo reproducir
la dinámica) a partir de datos sintéticos con ruido, y validar la identificación conectándola a un
controlador. Tres fases:

- **OE1** — simulador de Wilson-Cowan en Python + datasets sintéticos con ruido.
- **OE2** — identificación con **dos arquitecturas comparadas** (PINN y Neural ODE) + baseline clásico.
- **OE3** — validación orientada al control: conectar el modelo identificado al controlador IMC y ver
  si se preserva la sincronización theta-gamma bajo ruido y variación paramétrica.

Este informe cubre la ampliación de la librería de estímulos (transversal a OE1/OE2) y el grueso de
OE2 y OE3.

---

## 2. Conceptos en lenguaje simple

- **Wilson-Cowan (el "simulador" / sistema real):** ecuaciones de dos poblaciones de neuronas —una
  excitatoria **E**, una inhibitoria **I**. Es nuestro "cerebro de mentira": genera los datos. Sus
  **parámetros** (wEE, wEI, wIE, wII) dicen cuánto se influyen las poblaciones.
- **Estímulo (P, Q):** la señal externa que inyectamos (P→E, Q→I) para "mover" el sistema.
- **Identificar parámetros:** a partir de los datos (cómo se movieron I y E), deducir los valores de
  los pesos. Es el problema "al revés".
- **PINN:** red que mapea `tiempo → [I, E]` y deduce los parámetros, con la física como castigo en
  el entrenamiento. Una red por trayectoria (el estímulo queda implícito).
- **Neural ODE:** red que aprende la **regla de cambio** `f(estado, estímulo) → velocidad`. El
  estímulo es **entrada explícita**, así que **un solo modelo** sirve para cualquier estímulo. Por
  eso funciona como "gemelo digital" y como planta del controlador.
- **Planta:** el sistema que el controlador maneja (el simulador real o el Neural ODE aprendido).
- **Controlador (IMC):** genera los estímulos justos para que I y E **sigan una referencia** (un
  ritmo deseado), como un termostato. Lleva un modelo de la planta adentro (sus pesos).
- **Lazo abierto vs cerrado:** abierto = aplico estímulo y miro; cerrado = el controlador corrige el
  estímulo sobre la marcha según el resultado.

---

## 3. Antes vs ahora

| | **Antes** | **Ahora** |
|---|---|---|
| Estímulos | Solo **pulso cuadrado** (box) | **Librería**: APRBS, theta-gamma, onda cuadrada, PRBS, Poisson, chirp + box |
| Identificar wII **en limpio** | Ya daba bien (PINN conjunta: **0.1%**) | Reconfirmado con estímulos nuevos (0.4–1.7%) |
| Identificar wII **con ruido alto** | **62%** de error (se rompía) | **8.9%** — la mejora real |
| Neural ODE | Existía, sin entrenar/usar | **Entrenado** y funcionando como planta |
| Controlador | Solo vs simulador, con parámetros **reales** | vs **ambas plantas**, con parámetros **reales E identificados** |
| Ruido | Solo en la PINN (wII 62%) | En **toda la cadena** identificación → control (wII 8.9%) |

---

## 4. La librería de estímulos

Antes se usaba solo el pulso cuadrado, que "mueve poco" al sistema. Se agregaron 5 generadores
nuevos (en `src/wilson_cowan/model.py`), todos **on/off, ≥ 0 (realizables con optogenética)** y
**funciones puras del tiempo** (reproducibles con `seed`; los aleatorios pre-calculan su "agenda" y
`f(t)` solo consulta, necesario porque el integrador evalúa en tiempos salteados).

| Generador | Qué es | Para qué |
|---|---|---|
| `aprbs_pulse` | Escalones de amplitud y duración aleatorias (+ piso `amp_min`) | Barre **amplitud × frecuencia**; el más rico para no lineal |
| `theta_gamma_pulse` | Ráfagas gamma bajo envolvente theta | El régimen propio del proyecto |
| `square_wave_pulse` | Onda cuadrada / tren de pulsos (freq + duty) | DBS / optogenética |
| `prbs_pulse` | Secuencia binaria pseudo-aleatoria | Clásico de identificación (Ljung) |
| `poisson_pulse` | Pulsos en tiempos aleatorios (ancho configurable) | Naturalista (estadística de spikes) |

Senoide/multiseno quedaron como *legacy* (descartados: no on/off). El chirp se conserva por su
cobertura espectral.

---

## 5. Identificación paramétrica

### 5.1 El recorrido de la PINN (qué falló y qué funcionó)

Partiendo de un arranque ignorante (todos los pesos en 1.0; verdaderos 6.4 / 4.8 / 6.0 / 1.2):

- **Intento 1 — Adam conjunto, w_physics=1:** wEE≈1.7 (74% error). `L_data` domina; el gradiente
  físico es muy débil y Adam lo suprime → la red memoriza la trayectoria sin mover los pesos.
- **Intento 2 — Adam conjunto, w_physics=10:** mínimo degenerado — satisface la física con una
  trayectoria incorrecta y parámetros incorrectos a la vez.
- **Intento 3 — Dos etapas (ajustar trayectoria, luego pesos):** fracaso, y la **raíz** es
  instructiva: el `forward` de la PINN no usa los pesos; entrena la trayectoria sin física en la
  etapa 1, y en la etapa 2 sus derivadas autograd ya satisfacen la física con pesos errados →
  gradiente ≈ 0. Separar las etapas rompe el acoplamiento necesario.
- **Solución (PINN canónica que funciona):** entrenamiento **conjunto** (autograd + física en la
  pérdida a la vez) + **multi-trayectoria** (θ compartido) + **Fourier features** (rompen el sesgo
  espectral) + balance `w_data` alto. Con eso, en limpio: **wII 0.1%** (todos < 0.3%).

Nota: diferencias finitas + scipy (mínimos cuadrados no lineales) **no es una PINN** — es el método
clásico de *derivative matching* (tipo SINDy), que el concurso pide como **comparación**.

### 5.2 Por qué wII es el difícil (y cómo se resuelve)

wII aparece solo en la entrada inhibitoria `u_i = wIE·E − wII·I + Q − thetai`. Cuesta por dos motivos:
1. Su sensibilidad es ∝ I: si I es chico, los datos no informan sobre wII.
2. Compite con `wIE·E` en el mismo término: si E e I van correlacionados, (wIE, wII) son inseparables.

→ Hay que provocar **I grande y decorrelado de E**. En este sistema el umbral inhibitorio es alto
(thetai=4), así que **Q debe ser grande (~3–5, no ~1)** para mover I por su cuenta. (La idea de
"variar Q para wII" ya existía en el proyecto; el aporte fue empujar Q mucho más y **cuantificarlo**.)

**Proxy de identificabilidad (sin entrenar):** el cociente `s2/s1` de los valores singulares de la
nube de puntos (E, I) mide si llenan el plano 2D. Alto = mejor = wII más identificable. box ≈ 0.13;
diseño "Q grande" ≈ 0.64 (≈5× mejor). Predice el resultado en segundos, antes de gastar horas
entrenando.

### 5.3 Resultados de identificación (en limpio)

| Método | wEE | wEI | wIE | wII |
|---|---|---|---|---|
| PINN conjunta (previo) | 0.1% | 0.3% | 0.0% | 0.1% |
| PINN Q-grande (`train_wii`) | 1.67% | 4.47% | 0.24% | 1.66% |
| **Neural ODE** (mezcla de estímulos) | 0.11% | 0.37% | 0.09% | **0.42%** |

El Neural ODE identifica incluso mejor que la PINN (integrar a través del rollout es muy potente).

---

## 6. El Neural ODE como planta (lazo abierto)

Se entrenó **un único** Neural ODE `f(x, P, Q)` sobre 20 trayectorias con todos los estímulos nuevos
(multiple shooting, ventanas cortas). No memoriza trayectorias: aprende la regla local, con el
estímulo como entrada. Por eso una mezcla de estímulos lo entrena mejor (más cobertura del espacio).

**Resultado (lazo abierto, mismo estímulo al simulador y al Neural ODE):**
- rollout MSE train **3.36e-5**, test held-out **8.54e-5** (solo ~2.5× el de train → generaliza a
  estímulos no vistos).

Ver `informe_neural_ode.html` (figuras de I(t), E(t) superpuestas).

---

## 7. Control en lazo cerrado

### 7.1 Qué parámetros usa el controlador

El IMC (port del MATLAB del tutor) usa parámetros de Wilson-Cowan en dos lugares:
- **Pesos** (wEE,wEI,wIE,wII) → en la **cancelación** (feedback linearization):
  `P = up − (wEE·E − wEI·I)`, `Q = uq − (wIE·E − wII·I)`.
- **Sigmoidea** (ae,ai,thetae,thetai,ke,ki) → en la inversa y los límites de saturación.
- **No usa** te, ti (son de la dinámica de la planta).

Por eso la calidad de la identificación impacta en el control.

### 7.2 La matriz 2×2 (planta × parámetros del controlador), por tipo de referencia

Se evaluó el seguimiento de referencias de 3 tipos (theta-gamma, APRBS, chirp), 2 formas cada una,
con las 4 configuraciones {simulador, Neural ODE} × {θ̂, reales}. RMSE de seguimiento de E:

| Referencia | Sim+θ̂ | Sim+reales | NODE+θ̂ | NODE+reales |
|---|---|---|---|---|
| theta-gamma 120 Hz | 3.14e-2 | 3.13e-2 | 3.13e-2 | 3.13e-2 |
| theta-gamma 160 Hz | 4.16e-2 | 4.15e-2 | 4.15e-2 | 4.15e-2 |
| APRBS lento | 2.11e-2 | 2.11e-2 | 2.11e-2 | 2.11e-2 |
| APRBS rápido | 2.96e-2 | 2.96e-2 | 2.96e-2 | 2.97e-2 |
| chirp 80→200 | 4.03e-2 | 4.03e-2 | 4.03e-2 | 4.02e-2 |
| chirp 200→80 | 3.48e-2 | 3.48e-2 | 3.48e-2 | 3.47e-2 |

Las 4 columnas son casi idénticas en todas las referencias → **usar θ̂ en vez de los reales no
degrada el control, y la planta aprendida se comporta como el simulador.** El RMSE varía con la
**dificultad de la referencia** (160 Hz / chirp más difíciles), no con la configuración.

### 7.3 Por qué el control anda aunque θ esté errado

El IMC tiene dos partes: (1) **cancelación feedforward** (usa θ; con θ errado queda imperfecta) y
(2) **realimentación con acción integral** (NO usa θ; mide el error y empuja hasta llevarlo a cero).
Mientras el lazo sea estable, la acción integral compensa el error de modelo. Por eso el control es
mucho más robusto que la identificación — con la salvedad de que un θ tan malo que desestabilice el
lazo sí fallaría (no es magia infinita).

---

## 8. Robustez bajo ruido (OE3)

Para cada nivel de ruido de observación se identifica con datos ruidosos (θ̂), se arma el controlador
con θ̂ y se cierra el lazo sobre la planta verdadera.

**Método ingenuo (escenarios moderados, sin preprocesar) vs mejorado (estímulo fuerte + suavizado
adaptativo de las observaciones):**

| σ (ruido) | k suav. | θ̂ máx ingenuo | θ̂ máx **mejorado** | Control RMSE I / E |
|---|---|---|---|---|
| 0.00 | 7 | 0.4% | **0.1%** | 3.347e-2 / 3.133e-2 |
| 0.01 | 7 | 2.1% | **0.6%** | 3.348e-2 / 3.142e-2 |
| 0.05 | 7 | 19.3% | **1.2%** | 3.371e-2 / 3.127e-2 |
| 0.10 | 11 | 106.1% | **8.9%** | 3.418e-2 / 3.064e-2 |

**Dos resultados:**
1. El **control** se mantiene cerca del ideal (~3.3e-2) en todos los niveles, incluso con θ̂ malo
   (acción integral).
2. La **identificación** bajo ruido mejora drásticamente con tres palancas: **(a) suavizar** las
   observaciones (palanca dominante: quita el ruido de alta frecuencia preservando la dinámica),
   **(b) estímulo fuerte** (Q grande, mejora el SNR de wII), **(c) suavizado adaptativo** (la ventana
   crece con el ruido — sobre-suavizar con poco ruido mete sesgo).

Ver `informe_ruido.html`.

---

## 9. PINN vs Neural ODE

- **No fue una comparación perfectamente controlada:** la Neural ODE dio 0.42% y la PINN 1.66%, pero
  en setups distintos. *En nuestras corridas* la Neural ODE rindió mejor en limpio.
- **Fortalezas distintas:** la Neural ODE integra a través de la dinámica (muy potente), pero es
  sensible al ruido en los estados (por eso necesitó suavizado); la PINN suaviza la trayectoria con
  su red (denoising nativo), podría ser más robusta al ruido sin preprocesar.
- **Comparar las dos ES un objetivo (OE2)** → no se descarta la PINN. Pendiente: un head-to-head
  **bajo ruido** sobre el mismo dataset.

---

## 10. Realismo de Q grande

**Chequeo interno (positivo):** con Q ~4–5, la tasa inhibitoria I llega a ~0.33–0.53 y **nunca se
satura** (0% del tiempo con I>0.9). Q~5 vs thetai=4 = drive externo apenas sobre el umbral →
respuesta moderada (~½ del máximo), no patológica → régimen válido.

**Caveat:** las unidades del modelo son adimensionales; una afirmación física cuantitativa (mW/mm²)
requiere calibrar el modelo a unidades reales. Cualitativamente, estimular fuerte y selectivamente
poblaciones inhibitorias es estándar con optogenética (opsinas en interneuronas PV/SST).

---

## 11. Estado actual y pendientes

**Demostrado** (todo con la cadena identificar → controlar):
- Librería de estímulos ampliada y realista.
- Identificación de los 4 pesos con buena excitación (no automática: depende del diseño del estímulo).
- Neural ODE que aprende la dinámica y generaliza.
- Controlador andando con ambas plantas, con parámetros reales e identificados, robusto a la forma de
  la referencia y al ruido.

**Pendiente:**
- **Variación paramétrica** (la otra mitad de OE3): control cuando la planta real difiere de la
  identificada (deriva).
- **Head-to-head PINN vs Neural ODE bajo ruido** sobre el mismo dataset.
- **Gray-box real** (`use_correction=True`): la corrección neuronal g_φ, para datos reales (el cerebro
  no es exactamente Wilson-Cowan).
- **Calibración a unidades físicas** para cuantificar el realismo de Q.

---

# Apéndice — Bitácora cronológica detallada

Registro de cada experimento, error y corrección.

## A. Identificación PINN (foco inicial)
Intentos 1–4 resumidos en §5.1: Adam (domina datos) → Adam w_physics alto (mínimo degenerado) →
dos etapas (gradiente nulo por desacople) → **conjunto con autograd + multi-trayectoria + Fourier**
(funciona). Receta complementaria heredada: warmup "datos primero" → congelar la red → lr alto para θ.

## B. Ampliación de estímulos e identificación de wII

- **Iter 1 — Comparación 7 familias, params a ojo, config pesada.** box gana (wII 6.7%); APRBS 12.5%;
  theta-gamma 11.3%. *Error:* params de los nuevos sin calibrar (APRBS quedaba cerca de cero).
  *Cambio:* afinar + bajar costo.
- **Iter 2 — 3 familias, retuneadas, config liviana.** box 5.7%, theta-gamma 9.5%, APRBS 17.9%.
  *Error mío:* cambié dos cosas a la vez (estímulo + config) → no atribuible. *Cambio:* experimento
  dedicado a wII variando solo las trayectorias.
- **Iter 3 — Diseño "P off, Q fuerte".** FALLÓ antes de entrenar: I_max ≈ 0.04. *Por qué:* Q~1.4 no
  vence thetai=4; I lo maneja E. *Cambio:* proxy s2/s1 + subir Q.
- **Iter 4 — Proxy + diseño "Q GRANDE".** proxy: box 0.13 → Q-grande 0.64. *Cambio:* lanzar con Q~3–5.
- **Iter 5 — Entrenamiento wII (Q-grande) → ÉXITO.** wII 1.66% (vs 5.7% box). El cuello de botella era
  el diseño de excitación, no la red.

## C. Neural ODE como planta + control

- **Iter 6 — Portar estímulos nuevos al dataset de control (ms).** *Error detectado:* Poisson con
  pulsos angostos (1.5 ms) daba E≈0 (en un modelo de tasas los pulsos angostos se promedian).
  *Corrección:* pulsos anchos (4–6 ms).
- **Iter 7 — Entrenar Neural ODE.** rollout MSE train 3.36e-5 / test 8.54e-5; θ̂ máx 0.42%.
- **Iter 8 — Lazo cerrado: planta aprendida vs verdadera.** RMSE casi idéntico (~3.3e-2).
- **Iter 9 — OE3 honesto: controlador con θ̂.** Ctrl(θ̂)·verdadera ≈ ideal → la identificación no
  degrada el control.
- **Iter 10 — Informe Simulador vs Neural ODE** (lazo abierto/cerrado). Corrección pedida: el
  controlador usa θ̂, no los reales.
- **Iter 11 — Informe integral por tipo de referencia** (matriz 2×2). Las 4 configs casi idénticas.
  *Ajuste de visualización (later):* curvas con grosor decreciente porque se superponen.

## D. Robustez bajo ruido

- **Iter 12 — Barrido de ruido (ingenuo).** θ̂ se rompe: 0.4 / 2.1 / 19.3 / 106% a σ=0/0.01/0.05/0.10.
  Control robusto igual (~3.3e-2). *Hallazgo:* control ≫ más robusto que identificación (acción integral).
- **Iter 13 — Diagnóstico de palancas a σ=0.10.** A baseline 106% · B +suavizado 15.4% · C estímulo
  fuerte 32.4% · **D fuerte+suavizado 10.6%**. El suavizado es la palanca dominante.
- **Iter 14 — Barrido con config D (k=7).** θ̂ 0.1 / 0.6 / 1.2 / 10.6%.
- **Iter 15 — Refinamiento σ=0.10.** strong k=11 8.9% (mejor). *Tradeoff:* más suavizado baja wEI/wEE
  pero sube wII (borra la dinámica rápida de I).
- **Iter 16 — k=11 → tradeoff → suavizado ADAPTATIVO (definitivo).** k crece con el ruido (7 hasta
  σ=0.05, 11 a σ=0.10) → lo mejor de cada nivel: 0.1 / 0.6 / 1.2 / 8.9%.

## E. Realismo

- **Iter 17 — Realismo de Q grande (chequeo interno).** I no satura (0% > 0.9), llega a ~½ máx →
  régimen válido. Caveat: falta calibración a unidades físicas para una afirmación cuantitativa.
