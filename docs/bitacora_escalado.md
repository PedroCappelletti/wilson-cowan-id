# Bitácora del escalado progresivo

> Registro cronológico de las etapas 0–2 del plan
> (`docs/plan_escalado_progresivo.md`): qué se corrió, con qué configuración,
> qué dio y qué se decidió en cada paso. Fecha de inicio: 2026-08-09.
>
> Objetivo rector: **copiar la dinámica** (NRMSE de rollout open-loop sobre
> estímulos de test). Métricas secundarias: R² de la corrección contra el Δf
> verdadero y error de parámetros.

---

## Preparación

**Infraestructura nueva:**

- `scripts/esc_gen_datasets.py` — genera los datasets con UNA perturbación:
  `refrac1.npz` (solo refractariedad, r=0.10) y `act1.npz` (solo actuador,
  τ=1.0 ms, sat=3.0). Mismos valores que `default_uncertainty(eps=1)` para que
  todo sea comparable con los experimentos F.
- `scripts/esc_eval.py` — evaluación unificada: para cualquier modelo reporta
  NRMSE de rollout (test), R² de la corrección implícita contra el Δf
  verdadero, y error de parámetros.

**Decisiones de diseño:**

- La corrección "implícita" se mide como `f_modelo − f_WC(θ̂)` con los θ̂
  aprendidos: así la misma métrica sirve para la red aditiva (donde coincide
  con `g`), para la corrección estructurada y para los modelos con memoria.
- Los datasets aislados usan el mismo catálogo de 20 escenarios (13 train / 7
  test) y el mismo integrador que los eps*.npz existentes.

*(las corridas se registran abajo a medida que terminan)*

**Datasets generados** (`scripts/esc_gen_datasets.py`, 20 escenarios cada uno,
13 train / 7 test, sin ruido de observación):

- `refrac1.npz` — |Δf| va de ~0.0001 (estímulos suaves) a 0.021 (`prbs_1`):
  la refractariedad solo muerde donde la actividad es alta, como se esperaba.
- `act1.npz` — |Δf| más parejo y más grande (hasta 0.035 en `chirp`): el
  retardo del actuador deforma siempre que el estímulo se mueve.

---

## Etapa 0 — Referencia limpia ✅

White-box de F2 (entrenado sobre `eps0.npz`) evaluado con el pipeline nuevo:

| métrica | valor |
|---|---|
| NRMSE rollout test | **2.04%** (I 2.03 / E 2.04) |
| error medio de parámetros | 1.05% |

Coincide con lo documentado: **el piso es ~2%**. El único escenario flojo es
`box_a1.2` (10.8%), el escalón grande — ya era el caso duro antes. R² contra Δf
da NaN porque en eps0 el Δf verdadero es idénticamente cero (correcto).

## Etapa 1 — Solo refractariedad

Corridas lanzadas (todas sobre `refrac1.npz`, 1500 épocas + L-BFGS):

| tag | variante | ventana | qué contesta |
|---|---|---|---|
| `e1_wb` | white-box | 100 (5 ms) | ¿cuánto daña la refractariedad sola? |
| `e1_B_w100` | gray-box B `g(I,E)` | 100 | la corrección ciega, config de siempre |
| `e1_B_w200` | gray-box B | 200 (10 ms) | ¿ayuda entrenar con ventanas más largas? |
| `e1_B_w400` | gray-box B | 400 (20 ms) | ídem, más largo |
| `e1_S` | estructural `(1−r·x)·S(u)` | 100 | 1b: la forma exacta, ¿clava r? |

**Primer resultado — `e1_wb` (white-box, 22 min):**

| métrica | valor | contra referencia |
|---|---|---|
| NRMSE rollout test | **5.93%** | piso limpio 2.04%; con las DOS perturbaciones era 14.0% |
| R² corrección implícita | −0.27 | (whitebox: no hay corrección, mide el sesgo de θ̂ deformados) |
| error medio de parámetros | 11.5% | con las dos perturbaciones era 59.5% |

Lectura: **la refractariedad sola es un problema mucho más benigno** que el par
completo — el white-box ya la compensa bastante deformando poco los parámetros
(11.5%). El margen de mejora para el gray-box en NRMSE es 5.93% → ~2% (piso).

**`e1_S` (estructural, 140 min — lenta por competencia de CPU):** dio
**idéntica al white-box** (NRMSE 5.92%, mismos parámetros). El checkpoint
muestra por qué: aprendió `r_i = r_e = 0.0` exactos y saturación → ∞, o sea
colapsó a WC puro en vez de encontrar el `r = 0.10` verdadero. Causa probable:
`r` se parametriza `clamp(raw, 0, 2)`; si al principio el gradiente lo empuja
por debajo de 0 (mientras los θ todavía están lejos), queda pegado al borde
**sin gradiente** y nunca sale. Es una falla de optimización, no estructural —
la forma acá es exacta. **Acción:** relanzar `e1_S2` con `r` inicial 0.05 y
warm-start de θ desde el white-box.

**Barrido de ventana del gray-box B (terminado):**

| tag | ventana | NRMSE test | I / E | R² g vs Δf | err. param | wII |
|---|---|---|---|---|---|---|
| `e1_wb` | 100 | 5.93% | 5.91 / 5.96 | −0.27 | 11.5% | 72% |
| `e1_B_w100` | 100 (5 ms) | **2.99%** | 3.53 / 2.45 | 0.02 | 21.3% | 59% |
| `e1_B_w200` | 200 (10 ms) | 3.31% | 3.36 / 3.26 | −0.39 | 24.6% | 64% |
| `e1_B_w400` | 400 (20 ms) | **2.84%** | 2.82 / 2.85 | −0.49 | 25.6% | 61% |

Lecturas:

1. **La corrección ciega funciona para copiar la dinámica:** baja de 5.93% a
   ~2.9%, o sea recupera el **~75% del hueco** hasta el piso limpio de 2.04%.
   Por escenario (`e1_B_w100`): `prbs_1` 0.9%, `poisson_1` 0.9%, `chirp` 1.0%,
   `box_a1.2` 9.8% (el escalón grande sigue siendo el caso duro, como en el
   piso limpio, 10.8%).
2. **La ventana casi no importa** en NRMSE (2.84–3.31%, dentro del ruido entre
   seeds). La sospecha de que entrenar con 5 ms era el cuello de botella **no se
   confirmó** — al menos con refractariedad sola.
3. **Pero R² ≈ 0 en todas:** la red copia la trayectoria sin aprender la física.
   El backbone deforma θ (error sube de 11.5% a 21–26%, con `wII` ~60%) y la red
   compensa el resto. Es la ambigüedad θ/g de F4b otra vez: predecir bien ≠
   entender. Para el objetivo actual (copiar la dinámica) esto es aceptable; para
   cancelar en lazo cerrado no.
4. Costo: las corridas B tardaron ~25 h cada una — no por sí mismas (una sola
   tarda ~25 min) sino por correr 4 carriles × 4 hilos sobre 8 cores con
   ventanas largas. Nota para el futuro: **no más de 2 corridas en paralelo**.

**Criterio de éxito de la etapa 1** (NRMSE ≤ ~4% y R² > 0.9): la mitad. NRMSE
✅ (2.8–3.0%), R² ❌ (≈0). El smoke test de `e1_S2` (estructural con r
inicial 0.05 y warm-start desde el white-box, solo 2 épocas + L-BFGS) dio
**NRMSE 1.48% y R² 0.965** — la forma exacta bien inicializada cumple las dos.

**`e1_S2` completa (13 min):**

| métrica | valor |
|---|---|
| NRMSE rollout test | **1.70%** (I 1.68 / E 1.71) — por debajo del piso limpio 2.04% |
| R² corrección vs Δf | **0.964** (techo teórico 0.97) |
| error medio de parámetros | **2.42%** (wII 16.5%, el resto < 1.5%) |
| refractariedad aprendida | **r_e = 0.102** (verdadero 0.10); r_i = 0.0 (verdadero 0.10) |

`r_i` no se identificó: el canal I casi no siente la refractariedad (|Δf_I| es
~7× menor que |Δf_E|), así que no hay señal para moverlo y quedó en el borde.
No afecta la predicción. Por escenario: todos < 1% salvo `box_a1.2` (8.8%).

**Conclusión etapa 1: cumplida.** Dos caminos válidos según el objetivo:
- *copiar la dinámica sin saber la forma:* gray-box B ciego, ~2.9%, R² ≈ 0;
- *copiar Y entender:* estructural con warm-start, 1.7%, R² 0.96, r exacto.

Y una lección de método que se repitió en las dos etapas: **las correcciones
con forma física fallan por inicialización, no por forma** — arrancar en el
borde de un `clamp` mata el gradiente. Warm-start de θ desde el white-box +
parámetro físico inicial > 0 lo arregla.

## Etapa 2 — Solo actuador

Corridas lanzadas (todas sobre `act1.npz`):

| tag | variante | ventana | qué contesta |
|---|---|---|---|
| `e2_wb` | white-box | 100 | ¿cuánto daña el actuador solo? |
| `e2_B` | gray-box B | 100 | control negativo: `g(I,E)` no debería poder |
| `e2_lag` | **lag estructural** (τ̂, sat aprendibles) | 100 | 2a: la forma física, ¿lo clava? |
| `e2_lat_w100` | **latente** (z de dim 2 aprendido) | 100 | 2b: memoria genérica |
| `e2_lat_w400` | latente | 400 | 2b con ventana larga (z arranca en 0 por ventana: necesita tiempo para cargarse) |

**Código nuevo:** `src/neural_ode/memory.py` — `LagGrayBox` (estado aumentado
[I,E,P̂,Q̂] con dP̂/dt=(P−P̂)/τ̂ y saturación aprendible) y `LatentGrayBox`
(ż = h_ψ(x,z,P,Q), corrección g_φ(x,z), ambas redes con salida inicial 0).
Entrenador `fit_aug` con multiple shooting; el estado oculto se inicializa por
ventana (filtro asentado en el comando / z=0) y la loss solo mira (I,E).

**Resultados (todas terminadas):**

| tag | variante | NRMSE test | I / E | R² vs Δf | err. param | extras aprendidos |
|---|---|---|---|---|---|---|
| `e2_wb` | white-box | 15.23% | 13.96 / 16.51 | −0.01 | 30.7% | — |
| `e2_B` | gray-box `g(I,E)` | 15.40% | 15.51 / 15.29 | −1.88 | 34.4% | — |
| `e2_lag` | lag estructural | **12.16%** | 12.43 / 11.89 | **0.65** | 39.2% | τ̂=0.56 (real 1.0), sat→∞ (real 3.0) |
| `e2_lat_w100` | latente z∈ℝ² | 14.92% | 14.91 / 14.93 | −0.51 | 38.9% | — |
| `e2_lat_w400` | latente, W=400 | 13.28% | 17.14 / 9.42 | −3.55 | 52.6% | — |

Lecturas:

1. **El actuador solo daña más que la refractariedad sola** (white-box 15.2%
   contra 5.9%), y de hecho tanto como las dos juntas (14.0%). Confirma que el
   retardo es lo que domina el problema original.
2. **`g(I,E)` no puede, como se predijo** (15.4%, R² −1.9): control negativo
   correcto.
3. **La forma física con memoria es la única que aprende algo real:** R² 0.65
   —el mejor de todo el escalado hasta acá— y baja el NRMSE a 12.2%. Pero
   está lejos de clavarlo, y el checkpoint dice por qué: **τ̂ se quedó en 0.56
   (la mitad del real) y la saturación se apagó** (α pegado al borde inferior
   del clamp, mismo defecto que `e1_S`).
4. **El latente genérico no ayuda** (13.3–14.9%) y arruina los parámetros
   (52% con W=400). Con z arrancando en 0 en cada ventana de 5 ms, la memoria no
   llega a cargarse; con W=400 mejora E pero rompe I. Necesita más trabajo
   (inicialización de z, o encoder desde la historia) — no es prioridad.

**Diagnóstico del `e2_lag`** — dos defectos de entrenamiento, no de modelo:

- *Transitorio falso por ventana.* En multiple shooting cada ventana arrancaba
  con `P̂ = P(t0)` (filtro "asentado"). Pero el filtro real nunca está asentado
  cuando el comando se mueve: la ventana arranca con un salto de P̂ que no
  existe en la planta, y el modelo lo minimiza achicando τ̂. Como el estado del
  actuador **depende solo del comando (conocido entero) y de τ̂**, se puede
  computar exacto filtrando toda la trayectoria — sin suponer nada.
  Implementado: `LagGrayBox.filtered_inputs` (solución exacta del ZOH,
  diferenciable en τ̂), usado por `fit_aug` para el estado inicial de cada
  ventana y por `esc_eval` en el R² teacher-forced.
- *α en el borde muerto.* Arrancaba en 0.01 y cayó al clamp 1e-3, donde no hay
  gradiente. Ahora arranca en 0.1.

**`e2_lag2` completa (28 min):**

| métrica | `e2_lag` (antes) | `e2_lag2` (corregida) |
|---|---|---|
| NRMSE rollout test | 12.16% | **2.96%** (I 3.01 / E 2.91) |
| R² corrección vs Δf | 0.65 | **0.942** |
| error medio de parámetros | 39.2% | **5.11%** (todos < 10%) |
| τ̂ aprendido (real 1.0) | 0.56 | **0.984** |
| saturación (real 3.0) | ∞ | ∞ (no identificada) |

**Con el estado inicial exacto por ventana el modelo con memoria física clava
el actuador**: de 15.2% (white-box) a 2.96%, con τ̂ dentro del 2%. La
saturación volvió a apagarse: α cayó al borde del clamp aun arrancando en 0.1.
Es consistente con lo medido en F1 (la saturación aporta ~16% de la deformación
del actuador, el retardo el resto) — con τ̂ bien puesto queda muy poca señal
para α, y la que queda la absorben los θ. No impide llegar al 3%. Por
escenario: todo < 3% salvo `box_a1.2` (12.8%), el escalón grande, que es donde
la saturación sí muerde — coherente.

**Conclusión etapa 2: cumplida para 2a** (criterio ≤ ~4%: 2.96% ✅). 2b (latente
genérico) quedó en 13.3% y necesita otro diseño (no se insistió: no era el
cuello de botella).

---

## Cierre de la bitácora

Las tres etapas quedaron con receta validada. El informe consolidado, con la
tabla comparativa y la recomendación para las etapas 3–4, está en
`docs/resultados_escalado.md`.
