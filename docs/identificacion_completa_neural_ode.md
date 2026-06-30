# Identificación completa con la Neural ODE (caso realista)

> Identificar **los 10 parámetros** de Wilson-Cowan con la Neural ODE, sin
> regalarle ninguno: arranque ignorante (todo en 1.0) y solo se le muestran las
> trayectorias `(I, E)` y los estímulos `(P, Q)` — como pasaría en un caso real.

---

## 1. Motivación

Hasta ahora la Neural ODE (`GrayBoxWC`) identificaba **solo los 4 pesos sinápticos**
(`wEE, wEI, wIE, wII`) y recibía el resto de los parámetros físicos *fijos*, tomados
del propio dataset (es decir, conocidos de antemano). En un escenario real no se
conoce **ninguno** de los parámetros: solo se mide la actividad neuronal y se sabe
qué estímulo se aplicó. Este experimento extiende la identificación a todos los
parámetros físicos para reproducir ese escenario.

## 2. Qué parámetros se identifican

| Grupo | Parámetros | ¿Se identifica? |
|-------|-----------|-----------------|
| Pesos sinápticos | `wEE, wEI, wIE, wII` | Sí (ya se hacía) |
| Constantes de tiempo | `te, ti` | **Sí (nuevo)** — salen de la velocidad de relajación |
| Ganancias sigmoide | `ae, ai` | **Sí (nuevo)** — ancladas porque `P,Q` entran como `ae·P` / `ai·Q` y son conocidos |
| Umbrales sigmoide | `thetae, thetai` | **Sí (nuevo)** — offset de la entrada total |
| Offsets de reposo | `ke, ki` | **No son libres** — se derivan: `ke = σ(−ae·thetae)`, `ki = σ(−ai·thetai)` |

Total: **10 parámetros libres**. `ke, ki` no se entrenan; se recalculan en cada paso
a partir de `ae, thetae` / `ai, thetai` para que el reposo `E=I=0` siga siendo un
punto de equilibrio del modelo (se conserva la estructura física de Wilson-Cowan).

## 3. Cambios en el código

### 3.1. Modelo — `src/neural_ode/dynamics.py`

Se agregó el flag `learnable_params` a `GrayBoxWC`:

- `EXTRA = ("te", "ti", "ae", "ai", "thetae", "thetai")`: parámetros físicos que
  pasan a ser entrenables.
- Cuando `learnable_params=True`, esos 6 parámetros se guardan "crudos" como
  `nn.Parameter` y pasan por `softplus` para mantenerse positivos (igual que los
  pesos). Cuando es `False`, se comportan como antes (buffers fijos).
- `ke, ki` dejan de ser buffers fijos y se **recalculan dentro de `forward`** a
  partir de los `ae, thetae` / `ai, thetai` actuales.
- Nuevo método `params_dict()`: devuelve todos los parámetros identificados
  (pesos + físicos).

El núcleo de la dinámica (las ecuaciones de Wilson-Cowan) **no cambió**: sigue
siendo el mismo `forward` gray-box, solo que ahora los parámetros físicos pueden
venir de `nn.Parameter` en lugar de constantes.

### 3.2. Script de entrenamiento — `scripts/train_neural_ode_full.py`

Variante de `train_neural_ode.py` con:

- **Arranque ignorante**: todo (pesos y físicos) inicia en `1.0`. La red nunca ve
  los valores verdaderos; estos se usan **solo al final** para reportar el error.
- `learnable_params=True`, `use_correction=False` (identificación pura, sin la
  corrección neuronal `g_φ`).
- **Multiple shooting**: cada trayectoria se parte en ventanas de 100 pasos (~5 ms),
  cada una integrada desde el estado observado (estabilidad del entrenamiento).
- Optimizador Adam con dos grupos: pesos (`LR=5e-2`) y físicos (`LR=2e-2`),
  seguido de refinamiento L-BFGS.
- Checkpoint en `results/models/neural_ode_full.pt`.

## 4. Datos y estímulos

Dataset: `data/processed/control/multi_dataset.npz` — **datos limpios**
(`noise_std = 0.0`), 20 trayectorias (13 train / 7 test held-out).

**Estímulos de entrenamiento (13):** `box` (a0.4, a0.8), `square`
(a0.6×f50/f100/f130, a1.0×f50/f100), `aprbs` (×2), `prbs` (×1), `thetagamma` (×2),
`poisson` (×1).

**Test / held-out (7):** `box_a1.2`, `square_a1.0_f130`, `aprbs_2`, `prbs_1`,
`thetagamma_2`, `poisson_1`, `chirp` (este último, un barrido de frecuencia, solo
aparece en test → mide generalización a un estímulo nunca visto).

La riqueza de la librería (muchas amplitudes × muchas frecuencias, todos `≥ 0` y
conmutados, realizables con optogenética) es lo que hace identificables a los 10
parámetros.

## 5. Resultados

Arranque ignorante (todo en 1.0) → tras Adam + L-BFGS:

| Parámetro | Verdadero | Estimado | Error |
|-----------|-----------|----------|-------|
| `wEE` | 6.40 | 6.374 | 0.40% |
| `wEI` | 4.80 | 4.775 | 0.51% |
| `wIE` | 6.00 | 6.048 | 0.80% |
| `wII` | 1.20 | 1.187 | 1.08% |
| `te` | 1.00 | 0.994 | 0.64% |
| `ti` | 2.00 | 1.995 | 0.26% |
| `ae` | 1.20 | 1.204 | 0.36% |
| `ai` | 1.00 | 0.994 | 0.65% |
| `thetae` | 2.80 | 2.798 | **0.08%** |
| `thetai` | 4.00 | 4.046 | 1.14% |

- **Error paramétrico máximo: 1.14%.**
- **MSE open-loop** (rollout completo, sin resets): train `2.8e-5`, test held-out `5.5e-4`.
- **Convergencia**: en la época 0 todo está en ~1.0; para la **época ~1250** ya está
  prácticamente en el valor final (loss aplanado en `~9.4e-6`). Las 6000 épocas
  configuradas resultaron excesivas → conviene bajar a ~2000.

## 6. Validación en lazo abierto y lazo cerrado

Con el modelo identificado completo (`results/models/neural_ode_full.pt`) se corrió
`scripts/eval_full_identified.py`, que lo evalúa de dos maneras.

### 6.1. Lazo abierto (rollout vs. real)

Se integra el modelo aprendido sobre cada trayectoria del dataset **sin resets**
(rollout completo) y se compara contra la real. MSE por estímulo:

| Estímulo | Split | MSE |
|----------|-------|-----|
| `box_a0.4` | train | 1.3e-7 |
| `box_a0.8` | train | 5.2e-7 |
| `box_a1.2` | **TEST** | 3.0e-3 |
| `square_a0.6_f50` | train | 3.4e-7 |
| `square_a0.6_f100` | train | 6.8e-7 |
| `square_a0.6_f130` | train | 5.2e-7 |
| `square_a1.0_f50` | train | 5.2e-5 |
| `square_a1.0_f100` | train | 1.2e-5 |
| `square_a1.0_f130` | **TEST** | 9.7e-6 |
| `aprbs_0` | train | 4.6e-5 |
| `aprbs_1` | train | 8.2e-5 |
| `aprbs_2` | **TEST** | 2.0e-6 |
| `prbs_0` | train | 3.1e-5 |
| `prbs_1` | **TEST** | 7.6e-4 |
| `thetagamma_0` | train | 2.3e-6 |
| `thetagamma_1` | train | 2.4e-5 |
| `thetagamma_2` | **TEST** | 1.6e-6 |
| `poisson_0` | train | 1.2e-4 |
| `poisson_1` | **TEST** | 8.3e-5 |
| `chirp` | **TEST** | 1.9e-5 |

**MSE medio:** train `2.8e-5`, test held-out `5.5e-4`.

![Lazo abierto — modelo vs. real (chirp)](../results/figures/open_loop_full.png)

- El **chirp** (held-out, barrido de frecuencia nunca visto) se reproduce con MSE
  `1.9e-5`: en la figura, la curva del modelo se superpone a la real.
- El peor caso es `box_a1.2` (`3.0e-3`): es **extrapolación de amplitud** — el `box`
  más fuerte en train era `a0.8`, así que `a1.2` cae fuera del rango entrenado. Es el
  límite esperable de un modelo de datos, no un fallo de identificación.

### 6.2. Lazo cerrado (controlador IMC)

El modelo identificado se enchufa al controlador IMC (port de
`simulador_wilson_cowan_con_control.m`) de dos formas, comparadas contra el caso
nominal (todo verdadero). RMSE de seguimiento de las referencias theta-gamma
(descartando el transitorio inicial):

| Caso | RMSE I | RMSE E |
|------|--------|--------|
| Nominal (ctrl verdadero / planta verdadera) | 3.348e-2 | 3.134e-2 |
| Planta **aprendida** (ctrl verdadero) | 3.346e-2 | 3.128e-2 |
| Controlador **θ̂** (todo identificado) / planta verdadera | 3.341e-2 | 3.132e-2 |

![Lazo cerrado — seguimiento theta-gamma](../results/figures/closed_loop_full.png)

- **El modelo aprendido es una planta controlable válida**: usado como planta del
  controlador, el seguimiento es idéntico al de la planta verdadera (RMSE `3.346e-2`
  vs `3.348e-2`).
- **El control no se degrada con los parámetros identificados**: un controlador
  construido **enteramente** con θ̂ (los 10 parámetros, físicos + pesos) sobre la
  planta verdadera da RMSE `3.341e-2`, indistinguible del nominal. El error de
  identificación (<1.14%) es tan chico que no se propaga al control.
- En la figura, las tres curvas se superponen y siguen las referencias con el leve
  retardo de fase propio del controlador a 120 Hz.

> Salvedad de unidades: el lazo cerrado corre en la convención del controlador
> original (ms, refs 120 Hz) mientras que la identificación corrió en segundos.
> Reconciliar unidades sigue pendiente (ver nota en `closed_loop.py`); acá se replica
> la convención existente para ser consistente con `eval_closed_loop.py`.

## 7. Conclusiones

1. **Todos los parámetros son identificables.** La red recuperó los 10 desde un
   arranque ignorante, sin ver los verdaderos.
2. **El acoplamiento `ae·wEE` no degenera.** `ae` y los pesos quedaron en su valor
   real por separado (no en una combinación equivalente). Lo permitió que `P, Q`
   sean conocidos (anclan la escala de las ganancias) sumado a la riqueza de
   estímulos.
3. **Costo mínimo respecto a identificar solo pesos.** El experimento de 4 pesos
   daba 0.42% de error máximo; identificando los 10, sube a 1.14%. Degradación
   pequeña para más del doble de incógnitas.
4. **Salvedad importante:** estos resultados son con **datos limpios**
   (`noise_std = 0`). El test held-out ya muestra un MSE un orden mayor que train.
   Con ruido, identificar 10 parámetros (todos compartiendo la misma señal ruidosa)
   será notablemente más duro que identificar solo 4.

5. **El error de identificación no se propaga al control.** Tanto la planta aprendida
   como un controlador construido con θ̂ siguen las referencias theta-gamma igual que
   el caso nominal (validación orientada al control, OE3).

## 8. Próximos pasos

1. **Repetir con ruido** — el test realista de verdad; cuantificar cuánto se degrada
   la identificación completa frente a la de solo pesos.
2. ~~Figura de trayectorias~~ — hecho (§6.1, lazo abierto del chirp held-out).
3. **Bajar `EPOCHS` a ~2000** en el script, ya que converge mucho antes.
4. **Reconciliar unidades** entre identificación (segundos) y lazo cerrado (ms).

## 9. Archivos

- Modelo: `src/neural_ode/dynamics.py` (flag `learnable_params`)
- Entrenamiento: `scripts/train_neural_ode_full.py`
- Evaluación lazo abierto + cerrado: `scripts/eval_full_identified.py`
- Logs: `results/train_neural_ode_full.log`, `results/eval_full_identified.log`
- Checkpoint: `results/models/neural_ode_full.pt`
- Figuras: `results/figures/open_loop_full.png`, `results/figures/closed_loop_full.png`
