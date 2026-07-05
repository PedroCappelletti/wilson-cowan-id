# Identificación Wilson-Cowan sobre datos reales — informe de la primera tanda

> **Fecha:** 2026-07-03/04. **Datos:** `data8_filtered.mat` (3 grabaciones chirp/LFP).
> **Objetivo:** empezar la identificación paramétrica de Wilson-Cowan (WC) con la
> Neural ODE gray-box sobre datos **experimentales reales**, y ver qué se puede
> replicar. Este documento explica **qué se hizo, qué se encontró y qué sigue**.
>
> **TL;DR:** el hallazgo central es que **el estímulo `u` gobierna solo ~10 % de la
> respuesta `s` en modo simulación**; el resto es dinámica interna/estocástica. Por eso
> la identificación debe hacerse en **modo predicción a horizonte corto** (donde la
> dinámica SÍ es identificable, R²≈0.85–0.99) y **no** exigiendo que un rollout libre
> reproduzca toda la traza (techo real ~0.1). Esto no es un fallo del modelo: es una
> propiedad medible de los datos, y encaja con la teoría de excitación/identificabilidad
> del proyecto (ver [fundamentos_teoricos.md](fundamentos_teoricos.md)).

---

## 1. Los datos (verificado)

Del `.mat` (MATLAB v5, arrays planos, sin structs/cells):

| variable | shape | qué es |
|----------|-------|--------|
| `fsd` | (1,1) uint16 | fs = **1250 Hz** |
| `u1,u2,u3` | (N,1) f64 | estímulos **chirp 1→10 Hz**, no negativos `u=0.1+0.1·sin(fase)` ∈ [0,0.2] |
| `s1,s2,s3` | (N,1) f64 | respuestas (LFP), escalares, filtradas pasa-banda **[0.5,19] Hz** |

N = 23938 / 24822 / 20287 (≈ 19.1 / 19.9 / 16.2 s). Sin NaN/Inf.
Conversión con `scripts/load_real_data.py` → `data/processed/real/data8_fs{250,125}.npz`
(decimado con anti-alias; se trabajó a 125 Hz para iterar rápido, 250 Hz para producción).
Detalle de estructura y cuidados de conversión en [plan_datos_reales.md §0.1](plan_datos_reales.md).

Diagnóstico espectral (`results/figures/datos_reales_diagnostico.png`): la respuesta
**rastrea la frecuencia instantánea del chirp** (cresta del espectrograma que sube de
~1 a ~6-7 Hz) → hay respuesta forzada al estímulo.

---

## 2. El problema y las decisiones

Se identifica WC **por salida**: solo se observa el escalar `s` (LFP), no el estado
`[I,E]`. Decisiones tomadas con el usuario/tutor:

- **Entrada:** `u → E` (excitatoria). `P = c_P·u`, `Q = 0`, `c_P` libre.
- **Salida:** `s` = LFP = `c_out·(E−I)`, offset `b = 0` (la señal ya está sin DC).
- **Comparación** en el dominio filtrado [0.5,19] Hz (mismo que `s`).
- **Gray-box central:** V0 = WC puro (10 params); V1 = WC + corrección neuronal `g_φ`
  para lo no-WC.

Esto **rompe el pipeline existente** (`train_neural_ode_full.py`), que asume estado
completo `[I,E]` y hace multiple shooting reseteando desde el estado medido. Con datos
reales el estado inicial de cada ventana es una **variable latente**. Se escribió un
trainer nuevo por salida: `scripts/train_real_output.py`.

---

## 3. Validación por partes (lo que se corrió)

Se validaron los supuestos con experimentos baratos, en orden de menor a mayor
compromiso, antes de invertir en el pipeline completo.

### 3.1. ¿`s` está gobernada por `u`? — coherencia
Coherencia `u→s` alta a baja frecuencia (**pico 0.96 @ ~1.7 Hz**), decae al subir.
(El chirp es no estacionario → la coherencia de Welch está sesgada, pero confirma que
hay relación entrada-salida.)

### 3.2. Techo con un modelo lineal ARX `u→s`
Un ARX (auto-regresivo con entrada exógena) ajustado por mínimos cuadrados:

| orden (na, nb) | R² **teacher-forcing** (in-sample) | R² **teacher-forcing** cruzado (fit rec0 → rec1/2) |
|---|---|---|
| (0, 40) — solo FIR de `u` | 0.14 | 0.10 |
| (2, 20) | **0.987** | **0.987 / 0.988** |
| (4, 40) | ~1.00 | 0.998 / 0.999 |

Parecía inmejorable (R²≈0.99, y **cruzado** casi igual → parámetros reproducibles entre
grabaciones). **Pero ese R² es predicción a 1 paso** (usa el pasado *real* de `s`).

### 3.3. El hallazgo central — predicción vs **simulación libre**
El **mismo** ARX, corrido en **simulación** (solo `u`, realimentando su propia salida —
exactamente lo que hace el rollout del WC):

| modo | R² |
|------|----|
| teacher-forcing (predicción 1 paso, usa pasado real de `s`) | **0.99–1.00** |
| **simulación libre** (solo `u`) | **0.04–0.11** |

Ver **`results/figures/real_teacherforcing_vs_simlibre.png`**: arriba la predicción a 1
paso calca `s`; abajo la simulación libre produce una oscilación limpia con la frecuencia
correcta pero **desfasada y sin la estructura fina** de `s`.

**Conclusión:** `s` está **dominada por su propia dinámica recurrente/estocástica**, no
por `u`. El estímulo explica solo ~10 % de la varianza en modo simulación. Sin la parte
AR (na=0) el R² es 0.14 → la dinámica interna hace casi todo el trabajo.

### 3.4. El WC en las dos métricas
Con multiple shooting por ventanas (estado latente por ventana):

| métrica | R² WC | comentario |
|---------|-------|------------|
| **predicción a horizonte corto** (ventana ~16 pasos = 128 ms) | **≈ 0.85** | la dinámica WC SÍ es identificable |
| **rollout libre** de toda la traza (solo `u`) | ≈ 0.01–0.05 | cerca del techo real de simulación (~0.1) |

O sea: el WC en rollout libre daba ~0 **porque la tarea tiene techo ~0.1**, no porque el
trainer o la hipótesis `u→E` estén mal. En el marco correcto (predicción a horizonte
corto) el WC alcanza R²≈0.85.

---

## 4. Por qué el rollout libre "no anda" (y por qué eso está bien)

Tres cosas se confundieron al principio y conviene dejarlas claras:

1. **Colapso a salida-cero:** ajustar una señal oscilatoria con MSE en el tiempo, si la
   fase no coincide casi exacto, hace que "predecir 0" tenga menos error que una
   oscilación desfasada → el gradiente cae al mínimo trivial. (Se mitiga con loss
   normalizada / de correlación.)
2. **Degeneración con estado latente:** ventanas cortas + `x0` latente ajustan
   localmente **incluso con un modelo inestable/incorrecto** (los resets enmascaran la
   inestabilidad); el rollout libre después diverge. (Se mitiga con continuidad fuerte,
   regularización de parámetros y curriculum de longitud/ventana.)
3. **El techo real:** aun arreglando 1 y 2, la simulación libre desde `u` no puede pasar
   de R²~0.1 **porque `u` no gobierna a `s`**. Esto es lo que confirmó el ARX (§3.3):
   no es un problema numérico, es la física de los datos.

Detalles de saturación de la sigmoidea congelando `c_P`, explosión de `c_out`, etc., en
el historial de experimentos (bitácora).

---

## 5. Implicaciones para los dos protocolos pedidos

- **Protocolo A — generalización entre trayectorias (leave-one-out):** viable **en modo
  predicción**. El techo cruzado es ≈ 0.99 y los parámetros son reproducibles entre las
  3 grabaciones. Es el protocolo fuerte para estos datos.
- **Protocolo B — continuación temporal (forecast por rollout libre):** **limitado por
  el techo de simulación (~0.1)**. "Continuar la trayectoria" desde `u` solo no es
  alcanzable con alta precisión — propiedad de los datos (input débil), no del modelo.
  La métrica honesta de forecast es a **horizonte corto** (predecir los próximos k pasos
  dado el pasado reciente), no continuación indefinida.

Esto **no invalida** el objetivo: la identificación paramétrica se hace en modo
predicción, que es el marco estándar (prediction-error method, Ljung). Lo que cambia es
el criterio de éxito y la expectativa sobre el rollout libre.

---

## 6. Conexión con la teoría del proyecto (identificabilidad)

Que `u` sea un driver débil y que el régimen sea **casi lineal de 2º orden** implica que,
de los 10 parámetros WC, solo ~4-5 combinaciones efectivas quedan restringidas por los
datos (2 polos de la dinámica + ganancias `c_P, c_out`). Es decir, **habrá direcciones
planas** en la FIM del ajuste real — exactamente el fenómeno descrito en
[fundamentos_teoricos.md §3.1](fundamentos_teoricos.md) (curvatura del paisaje de error,
SVD, Cramér-Rao). El análisis FIM/SVD sobre el óptimo real es, por lo tanto, parte
central del entregable (O4), y se espera que **muestre y explique** la baja
identificabilidad de varios parámetros.

---

## 7. Estado y próximos pasos

**Hecho y validado ✅**
- Conversión y verificación del `.mat`; datasets `data8_fs{250,125}.npz`.
- Diagnóstico espectral + coherencia.
- Baseline ARX (predicción y simulación) y el hallazgo predicción-vs-simulación.
- Trainer por salida (`scripts/train_real_output.py`) con multiple shooting.
- WC identificable a horizonte corto (R²≈0.85); rollout libre cerca del techo ~0.1.

**Abierto / próximos pasos**
1. **Reencuadrar el trainer al marco de predicción a k-pasos** (prediction-error):
   optimizar predicciones de horizonte corto en vez de matching de simulación libre, y
   reportar R² **vs horizonte** (curva de teacher-forcing → simulación) como métrica.
2. **Inicialización principista:** anclar la WC a los polos del ARX orden-2 (la WC
   linealizada debe reproducir esos 2 polos) → arranque bien condicionado.
3. **Protocolo A** completo (leave-one-out en modo predicción, 3 folds) con barras de
   parámetros y su dispersión.
4. **V1 (gray-box):** ver si `g_φ` captura parte de la dinámica intrínseca que `u` no
   explica — pero cuidando que **no "tape" al WC** (que la parte WC siga interpretable).
   Ojo: como `s` no está gobernada por `u`, `g_φ` sin entrada no puede inventar la
   estocástica; su rol es corregir la estructura, no predecir el ruido.
5. **FIM/SVD (O4)** sobre el óptimo real → identificabilidad e incertidumbre (CRB).
6. **Preguntas al tutor:** ¿se esperaba que `u` fuera un driver débil? ¿el criterio de
   éxito es predicción (identificación) o reproducción de trayectoria? ¿hay más
   grabaciones / mayor amplitud de estímulo para mejorar la excitación?

---

## 8. Archivos

- **Datos/conversión:** `scripts/load_real_data.py`, `data/processed/real/data8_fs*.npz`.
- **Trainer:** `scripts/train_real_output.py` (modos `single`/`loo`/`forecast`).
- **Figuras:**
  - `results/figures/datos_reales_diagnostico.png` — u, s y espectrogramas.
  - `results/figures/real_teacherforcing_vs_simlibre.png` — **el hallazgo central**.
- **Plan y decisiones:** [plan_datos_reales.md](plan_datos_reales.md).
- **Teoría de identificabilidad:** [fundamentos_teoricos.md](fundamentos_teoricos.md).
