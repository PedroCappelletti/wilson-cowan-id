# Costo computacional del Neural ODE — qué reducir y cómo estimarlo

> **Pregunta que dispara esta nota:** ya sabemos qué estímulos identifican mejor (OED). Ahora: ¿cuánto **cuesta** computacionalmente el Neural ODE, qué **variables** podemos tocar para bajar ese costo (y el peso de la red) **sin perder calidad**, y — como con Fisher+SVD para identificabilidad — **¿hay un estimador a-priori del costo**? Sí lo hay, y en parte **reusa la misma maquinaria SVD/Jacobiano**. Código: `src/neural_ode/dynamics.py`, `integrate.py`.

---

## 1 · Dónde está el costo realmente (desglose de *nuestra* arquitectura)
Antes de optimizar, medir dónde se va el cómputo. Nuestro Neural ODE es **gray-box**: backbone con la estructura exacta de WC + corrección neuronal **opcional**.

| Componente | Tamaño real | Costo | ¿Se puede tocar? |
|---|---|---|---|
| **Backbone WC** (white-box) | **~10 escalares** (4 pesos + te,ti,ae,ai,θe,θi) | casi **gratis** (un puñado de mult/sigmoides por eval) | no; es la física |
| **Corrección `g_φ`** (MLP 4→32→32→2) | **1 282 parámetros** | el único bloque "de red"; solo activo si `use_correction=True` | **sí** (ancho, capas, pruning) |
| **Integrador RK4** | **4 evaluaciones de f por paso** | **dominante**: NFE = 4 × pasos × trayectorias × épocas | **sí** (dt, orden, ventana) |

> **Punchline (clave para no optimizar lo que no toca):** en **identificación pura sobre datos de WC** corremos con `use_correction=False` → la red es **~10 escalares** y su costo es **despreciable**. El costo real está en la **INTEGRACIÓN** (cuántos pasos × cuántas trayectorias × cuántas épocas), no en "el tamaño de la red". El tamaño de la red solo pesa cuando activamos el gray-box (`use_correction=True`, para datos que no son WC puro) o en la **latencia de control**.

## 2 · Son DOS costos distintos (no confundirlos)
- **(a) Costo de identificación** = entrenamiento, **offline**, se paga una vez. Lo domina la integración diferenciable (NFE) y la memoria del backprop a lo largo del rollout.
- **(b) Costo de control** = inferencia, **online / tiempo real**, se paga en cada paso del lazo. Lo domina el **forward pass de la planta** (si la planta es el Neural ODE, `make_neural_plant`): el IMC llama a `plant_rhs` una vez por sub-paso RK4 → con `dt=0.005 ms, tf=50 ms` son **~40 000 evaluaciones** por corrida de lazo. Acá el **peso de la red SÍ importa** (latencia, hardware — ver Su 2026 - SoC para neuromodulación closed-loop).

Minimizar cada uno pide palancas distintas (tabla §3). Lo que abarata la identificación (menos épocas, menos datos) no cambia la latencia de control, y viceversa.

---

## 3 · Variables que podemos cambiar (palancas)

| Variable                                             | Baja el costo de…      | Efecto en calidad                                 | Riesgo / límite                                                |
| ---------------------------------------------------- | ---------------------- | ------------------------------------------------- | -------------------------------------------------------------- |
| **Tamaño del dataset** (nº trayectorias, longitud T) | identificación (a)     | menos datos → más varianza en θ̂                  | buscar el **codo** de la curva de aprendizaje                  |
| **Densidad de muestreo** (dt de observación)         | identificación (a)     | submuestrear pierde dinámica rápida               | limitado por el contenido de frecuencia (Nyquist) del estímulo |
| **Paso de integración `dt`**                         | (a) y (b)              | dt grande → menos pasos, menos NFE                | **estabilidad/precisión**: limitado por la **rigidez** (§4)    |
| **Orden del solver** (RK4 vs RK2 vs adaptativo)      | (a) y (b)              | orden alto = menos pasos a igual precisión        | RK4 = 4 evals/paso; a veces RK2 (2 evals) alcanza              |
| **Ventana de rollout** (multiple shooting)           | identificación (a)     | ventanas cortas = menos memoria y mejor gradiente | demasiado corta pierde dinámica lenta                          |
| **Ancho/profundidad de `g_φ`** (`hidden`)            | control (b) + gray-box | red chica puede no capturar el mismatch           | solo relevante si `use_correction=True`                        |
| **Pruning / cuantización** de `g_φ`                  | control (b)            | degradación si se poda de más                     | podar hasta el **rango efectivo** (§4)                         |
| **Precisión numérica** (float64→32→16)               | (a) y (b)              | menos precisión puede frenar la convergencia      | WC no es muy rígido → float32 suele bastar                     |
| **`use_correction` on/off**                          | ambos                  | off = white-box exacto (para datos WC)            | on solo cuando los datos NO son WC puro                        |
| **Épocas / early-stopping / optimizador**            | identificación (a)     | cortar antes → θ̂ peor                            | L-BFGS converge en menos iters que Adam al final               |

---

## 4 · ¿Hay un estimador a-priori del costo, como FIM+SVD para identificabilidad? **Sí**
La pregunta clave. La respuesta: hay tres proxies que se calculan **sin barrer entrenamientos completos**, y dos **reusan el Jacobiano/SVD** que ya computamos para el FIM.

### (a) Rigidez (*stiffness*) → predice `dt` → predice NFE — *el análogo directo del número de condición*
- La rigidez de `ẋ=f_θ(x,P,Q)` se mide con los **autovalores del Jacobiano de estado** `∂f/∂x` (una matriz 2×2 acá). El **ratio de rigidez** `|λ_max|/|λ_min|` dice cuán chico debe ser `dt` para integrar estable.
- **Es el gemelo computacional del FIM+SVD:** el FIM+SVD toma la SVD de la sensibilidad a los **parámetros** (∂trayectoria/∂θ) y su número de condición predice **identificabilidad**; la rigidez toma los autovalores del Jacobiano de **estado** (∂f/∂x) y su spread predice **costo de integración (NFE)**. Misma idea (espectro de una derivada), distinta derivada.
- **Acción:** calcular `∂f/∂x` (ya tenemos autograd) a lo largo de una trayectoria → si el spread es chico (WC no es muy rígido), `dt` grande y RK4/RK2 alcanzan → NFE bajo. Si crece con Q-alta o cierto régimen, ahí hay que refinar `dt`.

### (b) NFE (*number of function evaluations*) → la métrica de costo estándar de Neural ODEs
- En la literatura de Neural ODEs (Chen et al. 2018) el **NFE** es *la* medida de costo (los solvers adaptativos lo reportan). Con RK4 de paso fijo es **determinista**: `NFE = 4 × pasos`. Sirve como métrica objetivo en los barridos (§5) sin cronometrar hardware.

### (c) Rango efectivo del Jacobiano/activaciones → predice la red mínima (target de pruning) — *reusa la SVD*
- El **rango efectivo** (participation ratio de los valores singulares) de la matriz de sensibilidad o de las activaciones de `g_φ` estima la **dimensión intrínseca** del problema → cuántas neuronas ocultas *realmente* hacen falta. Es la **misma SVD** del análisis de identificabilidad, leída para otro fin: allá cuenta direcciones de parámetro recuperables; acá, direcciones de estado/feature necesarias.
- **Acción:** SVD de las activaciones → si el rango efectivo ≪ `hidden=32`, se puede podar sin perder calidad. Da el **target de pruning a-priori** en vez de barrer tamaños a ciegas.

### (d) Curvas de aprendizaje (para el costo de datos)
- Para "cuántos datos", el estimador es empírico: **MSE vs tamaño de dataset** (curva de aprendizaje / scaling). No hay atajo cerrado como la rigidez, pero el **punto de saturación** se detecta con pocos puntos (log-log) sin barrer todo.

> **Resumen de la analogía:** *sí, hay estimadores a-priori del costo, y dos comparten maquinaria con Fisher+SVD.* FIM+SVD(∂y/∂θ) → **identificabilidad**. Autovalores(∂f/∂x) → **costo de integración (rigidez/NFE)**. SVD/rango efectivo(activaciones) → **red mínima / pruning**.

---

## 5 · Experimentos: lo que ya hay y lo que falta

**Ya hecho (relevante al costo):**
- Elegimos `use_correction=False` para identificación → red mínima por diseño (backbone exacto). El "peso de red" ya está en su piso para datos WC.
- Excitación Q-alta + multi-trayectoria + Fourier features (PINN) → **más info por muestra** (mejor SNR de wII) → menos datos para igual calidad. Es reducción de costo de datos vía diseño de estímulo (OED, ver Resultado - Subset selection, regularización y diseño de estímulo).

**Antes → después (números medidos):**

| Palanca | Antes | Después | Efecto |
|---|---|---|---|
| dt de integración | 0.05 (N_EVAL 4000, 4000 pasos/tray) | **0.2** (N_EVAL 1000), estímulos suaves | 4× menos pasos |
| tiempo por época (medido) | ~510 ms | **~140 ms** | ~3.7× más rápido |
| épocas (NODE 10 params) | 6000 | **2000** (converge ~ép. 1250) | 3× menos épocas |
| red (identificación pura) | — | `use_correction=False` (~10 escalares) | costo de red ≈ 0 |
| solver | RK4 fijo | RK4 fijo (adaptativo evaluado: **no aporta**, WC no es rígido) | — |

**Peso de los datos (para dimensionar):**
- Identificación: **12 escenarios × 4000 pasos × 4 señales (I,E,P,Q)** ≈ **1.5 MB** en memoria (float64); ~140 KB por `.npz`, **~15 MB** los 32 juntos. Multiple shooting WINDOW=100 → **39 ventanas/tray** → ~312 ventanas de train.
- Control: **19 trayectorias × 2000 pasos = 864 KB**.
- Redes: NODE white-box **~10 escalares** (con corrección `g_φ`: MLP **1 282 params**); PINN **~30 k params** (64×4 + 128 Fourier features).

**A hacer (mapea a Plan de trabajo - Neural ODE Ejes 2 y 3):**

| Experimento | Variable barrida | Métrica | Estimador a-priori que lo predice |
|---|---|---|---|
| Curva de aprendizaje | tamaño del dataset | rollout MSE, error θ̂ | curva log-log (saturación) |
| Barrido de densidad temporal | dt de observación | MSE vs Nyquist | contenido de frecuencia del estímulo |
| Barrido de `dt` de integración + orden de solver | dt, RK2/RK4/adaptativo | NFE vs MSE | **rigidez** `|λ_max|/|λ_min|` de ∂f/∂x |
| Frontera ancho×profundidad (gray-box) | `hidden`, capas | nº params vs MSE (el "codo") | **rango efectivo** (SVD activaciones) |
| Pruning post-entrenamiento | % podado de `g_φ` | MSE vs sparsity | rango efectivo (target) |
| Precisión numérica | float64/32/16 | MSE, tiempo | rigidez (float bajo falla si rígido) |
| **Latencia de control** vs tamaño de red | `hidden` | ms/paso del lazo, RMSE | FLOPs de `g_φ` |

---

> ⚠️ **Límite del lever de dt grande (hallazgo empírico, 2026-07-02):** el dt grande es seguro **solo para estímulos suaves / de banda limitada** (chirp). Con estímulos **conmutados/discontinuos** (pulsos, escalones, on/off: poisson, aprbs, square, theta-gamma) a dt≈0.2 la identificación **colapsa** (wII a 250–400 %) por el mismatch ZOH + desalineado temporal. Es la misma lección de las discontinuidades del §4/§5. → Para esos estímulos, dt fino. Ver Resultado - Identificación NODE por familia y el límite del dt grande.

## 6 · Recomendaciones prácticas (priorizadas)
1. **No optimizar la red en identificación pura:** con `use_correction=False` el cuello no es la red, es la integración. Primero medir NFE y rigidez.
2. **Estimar la rigidez una vez** (autovalores de ∂f/∂x): si es baja (probable en WC), subir `dt` y/o pasar a RK2 → menos NFE gratis.
3. **Curva de aprendizaje** para fijar el dataset en el codo (no juntar datos de más).
4. **Multiple shooting con ventanas cortas** para bajar memoria de backprop sin perder identificabilidad.
5. Para el **gray-box y el control**: usar el **rango efectivo** como target de pruning, y medir **latencia vs tamaño** porque ahí el peso de la red sí paga (hardware, tiempo real).

## 🔗 Conexiones
- Plan (Ejes 2 footprint y 3 datos): Plan de trabajo - Neural ODE
- Analogía de estimador: Fundamentos teóricos - Identificabilidad, SVD, Fisher y OED (FIM+SVD) · Identificabilidad (Fisher + SVD) - explicación visual
- Modelo y control: Neural ODEs · Modelo Wilson-Cowan · Controlador IMC - cómo funciona y por qué es robusto
- Por qué importa el footprint (hardware): Su 2026 - SoC para neuromodulación closed-loop
- Proyecto: Investigación Neurociencia

## 📚 Fuentes
- Código: `wilson-cowan-id/src/neural_ode/dynamics.py` (GrayBoxWC), `integrate.py` (RK4/rollout).
- NFE como métrica de costo: Chen et al. 2018 (*Neural ODEs*). Stiffness en Neural ODEs: Kim et al. 2021 (*Stiff Neural ODEs*).
