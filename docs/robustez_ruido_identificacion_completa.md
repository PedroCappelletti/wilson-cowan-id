# Robustez al ruido de la identificación completa (10 parámetros)

> Lleva la identificación de **los 10 parámetros** de Wilson-Cowan al **caso realista
> con ruido**, predice qué se puede recuperar con un diagnóstico de identificabilidad
> (Fisher + SVD), y mide cómo el error de identificación se propaga al control.
> Continuación de [identificacion_completa_neural_ode.md](identificacion_completa_neural_ode.md).

---

## 1. Motivación

En limpio (σ=0) la Neural ODE identifica los 10 parámetros con error máx 1.14%. La
pregunta realista es: **¿qué pasa con ruido de observación?** Y, sobre todo, ¿qué se
puede recuperar y qué no? Esta sesión:

1. Barre el ruido σ = [0, 0.01, 0.05, 0.10] e identifica los 10 parámetros, reportando
   el error **por cada parámetro** y comparándolo contra identificar solo los 4 pesos.
2. Predice ese resultado con la **Matriz de Información de Fisher (FIM) + SVD** —
   la generalización a 10 dimensiones del proxy `s2/s1` que ya usaba el proyecto.
3. Mide la **propagación al control**: arma el controlador IMC con el θ̂ ruidoso y mide
   el seguimiento en lazo cerrado.

Se mantiene `use_correction=False` (la corrección neuronal gray-box `g_φ` no se toca:
con datos WC puros solo degradaría la identificabilidad).

## 2. Método

- **Excitación fuerte y diversa** (`strong_scenarios`, de `noise_improve.py`): 12
  trayectorias con Q grande en varias (excita I → mejora el SNR de wII).
- **Suavizado adaptativo** (reutilizado de `noise_final.py`): promedio móvil de las
  observaciones ruidosas con ventana **k que crece con σ** (k=7 hasta σ=0.05, k=11 a
  σ=0.10). Sobre-suavizar con poco ruido mete sesgo; por eso es adaptativo.
- **Identificación**: arranque ignorante (todo en 1.0), `learnable_params=True`,
  multiple shooting, Adam (1500 épocas) + L-BFGS.
- **Comparación**: el mismo pipeline identificando **solo los 4 pesos**
  (`learnable_params=False`).

Script: `scripts/noise_full_sweep.py`. Diagnóstico FIM: `scripts/fisher_identifiability.py`.

## 3. Diagnóstico de identificabilidad (Fisher + SVD)

Para un estímulo dado, la trayectoria predicha `y(θ) = [I(t), E(t)]` apilada tiene
sensibilidad `J = ∂y/∂θ` (jacobiano por modo-forward, evaluado en θ verdadero). Con
ruido i.i.d. de varianza σ², la FIM es `JᵀJ / σ²`, cuyos autovalores son los valores
singulares de `J` al cuadrado. Se trabaja en **sensibilidad relativa** (columnas
escaladas por θ_j → cambios fraccionales, adimensional).

**Espectro de valores singulares** (normalizados al mayor) y **número de condición**:

| Modo | σ₁ | σ₂ | σ₃ | σ₄ | σ₅ | σ₆ | σ₇ | σ₈ | σ₉ | σ₁₀ |
|------|----|----|----|----|----|----|----|----|----|-----|
| rel  | 1.00 | 0.172 | 0.044 | 0.020 | 0.015 | 0.011 | 6.6e-3 | 3.9e-3 | 2.4e-3 | **7.5e-4** |

Número de condición σ₁/σ₁₀ ≈ **1.3e3** (moderado: todo identificable en limpio,
consistente con el 1.14%).

**Direcciones mal condicionadas** (vectores singulares de los σ chicos) → qué
**combinaciones** de parámetros son poco identificables:

| Modo | rel | Combinación dominante |
|------|-----|-----------------------|
| σ₁₀ | 7.5e-4 | **wII** (+0.91), ae (+0.25), te (+0.21) |
| σ₉ | 2.4e-3 | **ae** (−0.61), **te** (−0.46), **wEE** (+0.40), thetae (+0.35) |
| σ₈ | 3.9e-3 | **ai** (−0.71), wEI (+0.36), **ti** (−0.35), thetai (+0.35) |

![Espectro FIM y vectores singulares](../results/figures/fisher_svd.png)

**Predicción:** el parámetro menos identificable es **wII** (domina la dirección más
débil y tiene la menor sensibilidad relativa). Le siguen **ti**, las ganancias **ae/ai**
y **thetai**, atrapados en los acoples `ae–te–wEE` y `ai–ti–thetai`. Los mejor
identificados deberían ser **thetae**, **wIE** y **wEE**. → bajo ruido, esperamos que
**wII se degrade primero y más**.

## 4. Robustez al ruido — error por parámetro

Error de θ̂ (%) por parámetro y nivel de ruido (suavizado adaptativo):

| Parámetro | σ=0 | σ=0.01 | σ=0.05 | σ=0.10 |
|-----------|-----|--------|--------|--------|
| `wEE` | 0.05 | 0.78 | 4.78 | 3.33 |
| `wEI` | 0.08 | 0.92 | 1.23 | 4.25 |
| `wIE` | 0.14 | 0.19 | 0.57 | **1.18** |
| `wII` | 0.83 | 4.53 | 19.32 | **41.34** |
| `te` | 0.16 | 0.00 | 2.92 | 5.44 |
| `ti` | 0.68 | 1.44 | 4.55 | 15.88 |
| `ae` | 0.03 | 0.72 | 7.54 | 9.05 |
| `ai` | 0.35 | 0.92 | 4.76 | 11.53 |
| `thetae` | 0.03 | 0.35 | 3.53 | **4.05** |
| `thetai` | 0.03 | 0.48 | 2.18 | 5.21 |
| **máx (10)** | **0.83** | **4.53** | **19.32** | **41.34** |
| **máx (solo 4 pesos)** | 0.14 | 0.65 | 1.19 | **8.92** |
| k suavizado | 7 | 7 | 7 | 11 |

![Error por parámetro vs σ; 10 params vs solo 4 pesos](../results/figures/noise_param_error.png)

**La predicción de la FIM se cumple:**
- **`wII` se degrada primero y más** (0.83 → 41.34%): es el peor a todos los niveles,
  exactamente la dirección singular más débil.
- **`ti`** es el segundo peor (15.88%), seguido de **`ai`** (11.53%) y **`ae`** (9.05%) —
  los parámetros de los acoples `ai–ti–thetai` y `ae–te–wEE` que marcó la SVD.
- Los **mejor identificados** son **`wIE`** (1.18%), **`thetae`** (4.05%) y **`wEE`**
  (3.33%), también como predijo el ranking de sensibilidad.

**Identificar 10 es mucho menos robusto que identificar 4.** A σ=0.10, el error máx
de los 10 params es **41%** (por wII) vs **8.9%** identificando solo los 4 pesos.
Agregar los 6 parámetros físicos como incógnitas reparte el "presupuesto de ruido" y
agrava las direcciones débiles: al liberar `ai, ti, thetai` (que compiten con wII en la
ecuación de I), wII pierde mucha identificabilidad. **Lección práctica:** si se conocen
los parámetros físicos, conviene fijarlos; si no, regularizar o fijar wII / las
ganancias es lo más rentable según la FIM.

## 5. Propagación al control

Para cada σ se arma el controlador IMC **enteramente con el θ̂ ruidoso** (los 10
parámetros; ke,ki derivados) y se corre el lazo cerrado sobre la planta verdadera.
RMSE de seguimiento de las referencias theta-gamma (baseline ideal con parámetros
reales: I=3.348e-2, E=3.134e-2):

| σ | θ̂ máx (10) | RMSE I | RMSE E |
|---|-----------|--------|--------|
| 0.00 | 0.83% | 3.348e-2 | 3.134e-2 |
| 0.01 | 4.53% | 3.360e-2 | 3.137e-2 |
| 0.05 | 19.32% | 3.333e-2 | 3.061e-2 |
| 0.10 | **41.34%** | 3.291e-2 | 2.982e-2 |

![RMSE de control vs σ](../results/figures/noise_control_rmse.png)

**La acción integral absorbe el error de identificación aun con θ̂ malo.** Con wII al
**41%** de error (σ=0.10), el seguimiento sigue clavado en ~3.3e-2 / ~3.0e-2,
indistinguible del ideal (incluso marginalmente mejor en E, dentro del ruido). El IMC
tiene cancelación feedforward (usa θ, queda imperfecta con θ̂ malo) **y** realimentación
con acción integral (no usa θ, empuja el error a cero): mientras el lazo sea estable,
la integral compensa el error de modelo. → **el control es mucho más robusto que la
identificación** (resultado OE3, ahora confirmado para la identificación completa).

## 6. Conclusiones

1. **La FIM+SVD predice la degradación.** El ranking de sensibilidad relativa y las
   direcciones singulares débiles anticiparon, sin entrenar, qué parámetros se rompen
   bajo ruido: wII primero, luego ti / ai / ae. Es la generalización del proxy s2/s1.
2. **wII es el cuello de botella** de la identificación completa (aparece solo en la
   entrada inhibitoria, baja sensibilidad, compite con wIE y ahora también con ai/ti).
3. **Identificar 10 params es notablemente menos robusto que identificar 4** (41% vs
   8.9% a σ=0.10). El conocimiento previo de los parámetros físicos es valioso; si no
   se tiene, la FIM dice qué fijar/regularizar.
4. **El control no se degrada** con θ̂ ruidoso en ningún nivel: la acción integral
   absorbe el error de modelo. La cadena identificar → controlar es robusta al ruido
   también en el caso de identificación completa.

## 7. Reconciliación de unidades (ms)

La cadena trabaja **en milisegundos de punta a punta** — decisión ya tomada en
`gen_multi_dataset.py` ("régimen ms"): dataset de control, identificación
(te=1 ms, ti=2 ms; frecuencias en Hz → ciclos/ms vía f/1000) y lazo cerrado (tf=50 ms,
refs 120 Hz = 0.12 ciclos/ms) usan la misma unidad. Los parámetros son numéricamente
idénticos en ambos lados; lo único que difiere es el paso de integración (dt≈0.05 ms en
identificación vs dt=0.005 ms en el lazo), que es una elección numérica, no de unidades.
Se actualizó la nota (antes "pendiente de decisión") en `src/neural_ode/closed_loop.py`
y se verificó que `eval_closed_loop.py` sigue dando RMSE ~3.3e-2 (sin cambios).

## 8. Tests

`tests/test_neural_ode.py` (nuevos, además de `tests/test_wilson_cowan.py`):
- `learnable_params` expone los 6 raw físicos + pesos; `params_dict()` devuelve 10.
- `ke,ki` derivados conservan el equilibrio en reposo (E=I=0).
- El forward con `learnable_params=True` en los verdaderos coincide con el modo fijo
  (no se tocó el núcleo de la dinámica).
- Recuperación de los **4 pesos** (caso bien condicionado) desde arranque ignorante.
- Maquinaria de los 10 params: la pérdida baja y todos reciben gradiente finito.
- Regresión del lazo cerrado (convención ms): el IMC sigue la referencia.

Un test inicial que pedía recuperar los 10 params con **una sola trayectoria** falló de
forma instructiva: la pérdida baja pero el error paramétrico **sube** — la firma exacta
de mala identificabilidad que predice la FIM. Se reescribió honestamente.

## 9. Próximos pasos

1. **Regularización / fijado guiado por FIM**: fijar o regularizar wII (y opcionalmente
   las ganancias) y ver cuánto recupera la identificación completa bajo ruido.
2. **Head-to-head PINN vs Neural ODE bajo ruido** sobre el mismo dataset (la PINN
   suaviza nativamente con su red).
3. **Gray-box real** (`use_correction=True`) recién cuando los datos no sean WC puro.

## 10. Archivos

- Diagnóstico FIM+SVD: `scripts/fisher_identifiability.py` → `results/figures/fisher_svd.png`, `results/fisher_identifiability.log`
- Barrido de ruido + control: `scripts/noise_full_sweep.py`
- Figuras: `results/figures/noise_param_error.png`, `results/figures/noise_control_rmse.png`
- Resultados crudos: `results/noise_full_sweep.json`, `results/noise_full_sweep.log`
- Tests: `tests/test_neural_ode.py`
- Housekeeping: `scripts/train_neural_ode_full.py` (EPOCHS 6000 → 2000)
