# Plan de trabajo — Identificación Wilson-Cowan sobre **datos reales**

> Estado: propuesta inicial (2026-07-03). Datos entregados: `data8_filtered.mat`.
> Objetivo global: usar la Neural ODE gray-box para **identificar parámetros**
> WC a partir de datos reales y ver **qué trayectorias puede replicar**.

---

## 0. Qué nos dieron (hechos verificados)

Del `.mat` (`info` + inspección):

- **3 grabaciones sucesivas**, estímulo tipo **chirp 1→10 Hz**: pares `(u_i, s_i)`.
- `u_i` = **estímulo aplicado**, chirp **no negativo** en `[0, 0.2]` (offset ≈ 0.1,
  o sea `u ≈ 0.1 + 0.1·sin(fase_chirp)` — estilo optogenético, prende/apaga).
- `s_i` = **respuesta medida**, un **escalar**, filtrada pasa-banda **[0.5, 19] Hz**
  (por eso es de media ~0; amplitud ±0.05). Es el análogo de la salida `y = E − I`.
- **fs = 1250 Hz**; duraciones **19.15 / 19.86 / 16.23 s**.
- Diagnóstico (ver `results/figures/datos_reales_diagnostico.png`): la respuesta
  **rastrea la frecuencia instantánea del chirp** (cresta del espectrograma sube de
  ~1 a ~6-7 Hz) → es una **respuesta forzada**; el mapa `u → s` es identificable.

Datos ya convertidos: `scripts/load_real_data.py` → `data/processed/real/data8_fs250.npz`
(decimados ×5 a **250 Hz**, con anti-alias; la banda útil llega a 19 Hz, sobra margen).

### 0.1. Estructura exacta del `.mat` (verificada — para no errar la conversión)

MATLAB v5. Solo **arrays numéricos planos** (sin structs ni cells anidados):

| variable | shape | dtype | qué es |
|----------|-------|-------|--------|
| `fsd` | (1,1) | uint16 | frecuencia de muestreo = **1250 Hz** |
| `info` | (1,) | `<U162` | string descriptivo (Latin-1: la `ñ` aparece como byte suelto) |
| `u1,u2,u3` | (N,1) | float64 | estímulos chirp, **N = 23938 / 24822 / 20287** |
| `s1,s2,s3` | (N,1) | float64 | respuestas (mismo N que su `u`) |

Chequeos de sanidad (pasan): **sin NaN/Inf**; `u` exactamente en `[0, 0.2]`, arranca en
`0.1`, `(max−min)/2 = 0.1000` → `u = 0.1 + 0.1·sin(fase_chirp)` (no negativo).

**Cuidados al convertir:**
1. `u_i,s_i` son **vectores columna `(N,1)`** → aplanar a 1-D (`ravel`).
2. `fsd` es **uint16** → castear a float antes de dividir.
3. La **decimación IIR** mete un leve *ringing* que lleva `u` a ≈ −0.001 (artefacto de
   filtro). Si se requiere `u ≥ 0` estricto: decimar con FIR (`ftype="fir"`), clipear a
   `[0,0.2]`, o submuestrear por paso simple (la señal es de banda acotada).
4. `info` está en Latin-1 → decodificar así si se quiere el texto limpio (cosmético).

---

## 1. El cambio de fondo: de **estado completo** a **observación de salida**

Todo el pipeline sintético (`train_neural_ode_full.py`, `make_windows`, multiple
shooting) asume que **medimos `[I, E]`**: resetea `x0` desde el estado observado en
cada ventana y compara contra `[I, E]`. **Con datos reales eso no existe**: solo
tenemos el escalar `s ≈ y = E − I`. Consecuencias:

1. El estado inicial `[I₀, E₀]` de cada grabación es **desconocido** → pasa a ser
   variable a estimar (o se arranca en el equilibrio de reposo).
2. **Multiple shooting** tal cual no aplica (no hay estados para resetear). Se pasa a
   **single/multiple shooting por salida**: el `x0` de cada ventana es una **variable
   latente** que se optimiza, con penalización de continuidad entre ventanas.
3. Aparece un **modelo de observación**: `s` es el **LFP (potencial extracelular)**,
   modelado como `s = c_out·(E − I) + b`. Como `s` está pasa-banda (media 0), el offset
   `b` se pierde → efectivamente `s ≈ c_out·PB(E − I)`.
4. Hay que fijar/estimar **escalas** que en lo sintético no existían: ganancia de
   salida `c_out`, ganancia(s) de entrada, y la **escala de tiempo** (ver §2).

Esto es identificación de sistemas no lineal **por salida** estándar (gray-box),
pero **no** es lo que corre hoy el repo → hace falta un trainer nuevo.

---

## 2. Decisiones a tomar (con recomendación)

| # | Decisión | Opciones | Recomendación |
|---|----------|----------|---------------|
| D1 | **Escala de tiempo** | (a) ajustar en segundos reales y dejar que `te,ti` absorban la escala; (b) reescalar el tiempo a unidades adimensionales | **(a)** — `te,ti` libres; la resonancia observada (~3.5 Hz) fija el orden (~tens of ms). Honesto y simple. |
| D2 ✅ | **Qué población recibe `u`** | (a) `u→E`; (b) `u→I`; (c) ambas | **RESUELTO: (a) `u→E`** (estímulo excitatorio, `P=c_P·u`, `Q=0`), `c_P` libre. |
| D3 ✅ | **Modelo de salida** | `s = c_out·(E−I)(+b)` | **RESUELTO: `s` = LFP** = `c_out·(E−I)`; `c_out` libre, **`b`=0** (señal ya sin DC). |
| D4 | **Filtro del modelo** | comparar crudo vs aplicar el mismo pasa-banda [0.5,19] al `ŷ` simulado | **Aplicar el mismo pasa-banda** a `ŷ` antes de comparar (comparación justa; el modelo no debe "explicar" DC filtrado). |
| D5 ✅ | **White-box (V0) y gray-box (V1)** | V0 = WC puro (10 params); V1 = WC + corrección `g_φ` | **RESUELTO: ambos son objetivo.** V0 = identificación WC honesta; **V1 = central**, la corrección `g_φ` captura lo que **no** es estrictamente WC. Validar V1 con un test rápido antes de invertir en el pipeline (ver F1.5). |
| D6 | **Estado inicial `[I₀,E₀]`** | (a) reposo (0,0); (b) latente estimado | **(b)** por grabación (barato, y el transitorio inicial importa). Descartar los primeros ~0.5 s del loss. |
| D7 | **fs de trabajo** | 1250 / 500 / 250 Hz | **250 Hz** (×5): Nyquist 125 ≫ 19 Hz; rollout ~4-5k pasos (viable). |
| D8 | **Regularización / identificabilidad** | FIM/SVD + prior suave hacia rangos fisiológicos | Reusar el análisis FIM ya hecho: reportar SVD del Jacobiano y qué combinaciones quedan planas; regularizar solo si hace falta. |

**Preguntas abiertas para el tutor** (Sánchez-Peña): ¿hay un valor de referencia
esperado para `te,ti` / la frecuencia de resonancia con el que contrastar? ¿La
comparación se juzga en el dominio filtrado (D4)? ¿Qué tanto peso darle a que WC puro
(V0) ajuste vs. dejar que `g_φ` (V1) tome el residuo?

---

## 3. Objetivos

1. **O1 — Identificar** un juego de parámetros WC (10, más ganancias `c_P,c_out`) que
   reproduzca la respuesta medida a partir del estímulo, **sin ver `[I,E]`**.
2. **O2 — Generalización entre trayectorias**: que **los mismos** parámetros expliquen
   una grabación no usada en el ajuste (protocolo A, §5).
3. **O3 — Continuación temporal**: ajustar en la primera parte de una grabación y
   **predecir la cola** (protocolo B, §5) — como el chirp barre frecuencia en el
   tiempo, la cola contiene frecuencias no vistas → test fuerte de extrapolación.
4. **O4 — Identificabilidad**: FIM/SVD de la solución real (qué se determina bien, qué
   combinaciones quedan planas) y **incertidumbre** vía Cramér-Rao.
5. **O5 — Gray-box (central)**: usar la corrección `g_φ` (V1) para capturar la dinámica
   que **no** es estrictamente WC. Cuantificar cuánto mejora V1 sobre V0 (mide el
   mismatch estructural) y verificar que `g_φ` **complementa** sin "tapar" a WC (que la
   parte WC siga siendo interpretable, no que `g_φ` explique todo).

---

## 4. Pipeline / fases

- **F0 — Datos** ✅ conversión `.mat→.npz` + diagnóstico espectral. *(hecho)*
- **F1 — Trainer por salida** (nuevo, p.ej. `scripts/train_real_output.py`):
  - reusar `GrayBoxWC` + `rollout` (RK4 diferenciable, ZOH del estímulo);
  - añadir ganancias `c_P` (entrada) y `c_out` (salida) + `x0` latente;
  - loss = MSE entre `PB(c_out·(E−I))` y `s` (D4), + continuidad de ventanas;
  - Adam (lr separados como en la receta que funcionó) + refinamiento L-BFGS.
- **F1.5 — Validación rápida del gray-box** (antes de invertir en todo el pipeline):
  test mínimo de que `g_φ` sobre WC mejora el ajuste de una grabación de forma estable
  (no diverge, no "tapa" a WC). Decide si V1 vale la pena antes de F3/F4.
- **F2 — Ajuste de una grabación** (sanity): ¿alcanza WC puro (V0) a seguir la
  respuesta? Reportar `NRMSE`, ajuste en tiempo y espectrograma modelo vs real.
- **F3 — Protocolo A** (cross-trajectory / leave-one-out, §5) — V0 y V1.
- **F4 — Protocolo B** (split temporal / forecast, §5) — V0 y V1.
- **F5 — Identificabilidad** (FIM/SVD/CRB sobre el óptimo real).
- **F6 — V0 vs V1**: cuánto aporta `g_φ`, y chequeo de que WC siga interpretable.
- **F7 — Informe + figuras + bitácora**, y sincronización al vault.

---

## 5. Los dos protocolos de validación (lo que pidió el usuario)

**Protocolo A — Generalización entre trayectorias (leave-one-out).**
Ajustar **un solo juego de parámetros** sobre 2 grabaciones (p.ej. rec1+rec2) y medir
el error en la tercera (rec3), rotando la que se deja afuera (3 folds). Solo se
re-estiman por-grabación las condiciones iniciales latentes `x0`; **los parámetros son
compartidos**. Responde: *¿los mismos parámetros sirven para un chirp que el modelo no
vio?* Limitación: las 3 son de la **misma familia** de chirp → diversidad acotada (a
declarar en el informe).

**Protocolo B — Continuación temporal (forecast).**
Dentro de una grabación, ajustar con el **primer 70 %** (parámetros + `x0`) y hacer
**rollout libre** sobre el 30 % restante, comparando contra `s`. Como el chirp sube en
frecuencia con el tiempo, la cola tiene **dinámica no vista** → es el test más exigente
para un modelo dinámico. Responde: *¿el modelo continúa la trayectoria?* Reportar el
error de forecast vs. el error in-sample y vs. un baseline (persistencia / lineal).

*Recomendación:* correr **ambos**. A mide reproducibilidad de parámetros entre ensayos;
B mide poder predictivo dinámico. Son complementarios; juntos son el argumento fuerte.

---

## 6. Métricas y criterios de éxito

- **NRMSE** y **R²/varianza explicada** de `s` (in-sample, held-out, forecast).
- **Ajuste espectral**: cresta modelo-vs-real en el espectrograma (que la cresta del
  chirp coincida) — más informativo que el MSE punto a punto en señal oscilatoria.
- **Consistencia de parámetros** entre folds del protocolo A (dispersión baja = buena
  identificabilidad práctica).
- **FIM/SVD**: número de condición y direcciones planas sobre el óptimo real.
- Éxito razonable esperado: R² held-out claramente > baseline lineal; parámetros
  estables entre folds; forecast que sigue la fase varios ciclos antes de desfasar.

---

## 7. Riesgos y mitigaciones

- **No identificabilidad de escala** (`c_P·wEE`, `c_out`, `ae` acoplan): fijar un ancla
  (p.ej. `c_out=1` o `ae` fijo) y reportar el resto relativo; usar FIM para detectarlo.
- **WC puro no alcanza** a seguir la respuesta (mismatch estructural): cuantificar con
  V1; **no** perseguir deriva temporal — el modelo es **tiempo-invariante** (tutor).
- **Sensibilidad al `x0` latente / transitorio**: descartar los primeros ~0.5 s del loss.
- **Costo de rollout** en 5k pasos × backprop: multiple shooting por ventanas + 250 Hz.
- **Sobreajuste con solo 3 grabaciones**: parámetros compartidos + leave-one-out honesto.

---

## 7bis. Validación temprana (hallazgos, 2026-07-03)

Antes de pelear con el trainer WC completo, se validaron los supuestos con
experimentos baratos (`scripts/load_real_data.py` + diagnósticos):

1. **`s` está gobernada por `u`.** Coherencia `u→s` alta a baja frecuencia (pico
   **0.96 @ ~1.7 Hz**); la respuesta rastrea la frecuencia instantánea del chirp
   (espectrograma). La relación entrada-salida existe y es fuerte.
2. **⚠️ Distinción CRÍTICA — predicción vs simulación** (el hallazgo central):
   - Un ARX **teacher-forcing** (predice el próximo dato usando el pasado REAL de `s`)
     da **R² ≈ 0.99** (orden 2) — y **cruzado** (fit rec0 → rec1/2) también ≈ 0.99.
   - El **MISMO ARX en simulación libre** (solo `u`, realimentando su propia salida —
     lo que hace el rollout del WC) **colapsa a R² ≈ 0.04–0.11**.
   - Sin la parte AR (na=0), R² = 0.14 → la dinámica recurrente hace casi todo.
3. **Interpretación (importante y honesta):**
   - **`s` está dominada por su dinámica interna/estocástica, NO por `u`.** El estímulo
     explica solo **~10 %** de la varianza *en modo simulación*. Ver figura
     `real_teacherforcing_vs_simlibre.png`.
   - Por lo tanto el **rollout libre desde `u` tiene techo ~0.1** para *cualquier*
     modelo manejado solo por `u` (WC incluido). El WC en rollout libre daba ~0.01–0.05
     → estaba **cerca de ese techo real**; el trainer NO estaba roto, la tarea tiene
     techo bajo en simulación libre.
   - En cambio, en **predicción a horizonte corto** la dinámica SÍ es identificable: el
     WC en ventanas de ~16 pasos (multiple shooting) alcanza **R²_ventana ≈ 0.85**.
   - **Marco correcto de identificación = error de predicción / horizonte corto**
     (prediction-error method), NO el matching de simulación libre de toda la traza.
   - El régimen es **casi lineal de 2º orden** → 10 params WC con **direcciones planas**
     (~4-5 efectivas) → conecta con FIM/SVD.
4. **Implicación para los dos protocolos:**
   - **Protocolo A (cross-trajectory):** funciona en modo predicción (techo cruzado
     ≈ 0.99). Los parámetros son reproducibles entre ensayos.
   - **Protocolo B (forecast/continuación libre):** limitado por el techo de simulación
     (~0.1). "Continuar la trayectoria" desde `u` solo NO es alcanzable con alta
     precisión — es una propiedad de LOS DATOS (input débil), no del modelo. La métrica
     honesta de forecast es a **horizonte corto**, no continuación indefinida.

**Baselines oficiales:** predicción-1-paso ARX R²≈0.99; simulación-libre ARX R²≈0.1.
La WC debe compararse contra el baseline **del mismo modo** (predicción vs simulación).

---

## 8. Entregables

- `scripts/load_real_data.py` ✅ y `data/processed/real/data8_fs250.npz` (+ `fs125`) ✅.
- `results/figures/datos_reales_diagnostico.png` ✅.
- Diagnósticos de viabilidad (coherencia + techo ARX) ✅.
- `scripts/train_real_output.py` (F1) + checkpoint.
- Figuras: ajuste temporal, espectrograma modelo/real, forecast (B), barras de
  parámetros por fold (A), FIM/SVD del óptimo real.
- `docs/identificacion_datos_reales.md` (informe) + entrada de bitácora + sync al vault.
