# Resultados del escalado progresivo (etapas 0–2)

> Qué se probó, qué dio y qué se decidió. Plan en
> `docs/plan_escalado_progresivo.md`, proceso paso a paso en
> `docs/bitacora_escalado.md`, números en `results/escalado/*.json`.
>
> **Objetivo rector:** copiar la dinámica del simulador (NRMSE de rollout
> open-loop de 200 ms sobre 7 estímulos de test que el modelo nunca vio).
> Métricas secundarias: R² de la corrección contra el Δf verdadero (¿aprendió
> la física?) y error de parámetros.

---

## Resumen ejecutivo

Se volvió a hacer el gray-box de a una perturbación por vez, en lugar de las dos
juntas. Resultado: **las dos perturbaciones se copian a ~2–3% de NRMSE por
separado**, cuando con las dos juntas el mejor modelo daba 13.1%.

| etapa | planta | mejor modelo | NRMSE test | R² física | err. param |
|---|---|---|---|---|---|
| 0 | WC puro (piso) | white-box | **2.04%** | — | 1.05% |
| 1 | + refractariedad | estructural + warm-start (`e1_S2`) | **1.70%** | 0.96 | 2.4% |
| 1 | + refractariedad | gray-box ciego `g(I,E)` (`e1_B_w400`) | 2.84% | −0.49 | 25.6% |
| 2 | + actuador (retardo+sat) | lag estructural, estado exacto (`e2_lag2`) | **2.96%** | 0.94 | 5.1% |
| *ref.* | *las dos juntas (trabajo previo)* | *gray-box D* | *13.1%* | *≈0* | *31%* |

Las tres cosas que dice esta tabla:

1. **Cuando la corrección tiene la forma correcta y arranca bien, clava la
   física:** r = 0.102 (real 0.10) y τ = 0.984 (real 1.0). Y de paso deja los
   parámetros WC casi intactos (2–5% de error, contra 31–60% antes).
2. **La corrección ciega `g(I,E)` copia la dinámica sin entenderla** (2.84%,
   R² ≈ 0): tapa la refractariedad deformando parámetros. Sirve para predecir,
   no para cancelar en lazo cerrado.
3. **Los fracasos fueron de optimización, no de modelo.** Las dos correcciones
   estructurales fallaron primero (colapsaron a WC puro / τ a la mitad) por
   arrancar en el borde muerto de un `clamp` y por un transitorio falso en
   multiple shooting. Arregladas las dos cosas, funcionan.

---

## Etapa 0 — el piso

White-box entrenado sobre planta limpia, evaluado con el pipeline nuevo:
**2.04%**. Todos los escenarios < 1% salvo `box_a1.2` (10.8%), el escalón grande
sostenido — es el caso duro en *todas* las etapas y hay que leerlo aparte.

## Etapa 1 — solo refractariedad

Planta = WC + `(1 − r·x)·S(u)`, r = 0.10. Es física **sin memoria**: una
función pura de (I,E), capturable en principio (techo R² 0.97).

| tag | modelo | ventana | NRMSE | R² | err. param |
|---|---|---|---|---|---|
| `e1_wb` | white-box | 5 ms | 5.93% | −0.27 | 11.5% |
| `e1_S` | estructural, arranque r≈0 | 5 ms | 5.92% | −0.27 | 11.5% |
| `e1_B_w100` | gray-box `g(I,E)` | 5 ms | 2.99% | 0.02 | 21.3% |
| `e1_B_w200` | gray-box `g(I,E)` | 10 ms | 3.31% | −0.39 | 24.6% |
| `e1_B_w400` | gray-box `g(I,E)` | 20 ms | 2.84% | −0.49 | 25.6% |
| **`e1_S2`** | **estructural, warm-start + r₀=0.05** | 5 ms | **1.70%** | **0.964** | **2.4%** |

**Qué se aprendió:**

- La refractariedad sola es benigna: el white-box la compensa a 5.9% con poca
  deformación (11.5%). El daño del trabajo previo (14%) venía casi todo del
  actuador.
- **La ventana de entrenamiento no importa** (2.84–3.31%, dentro del ruido). La
  hipótesis del plan —que entrenar con 5 ms era el cuello de botella— **no se
  confirmó**.
- El gray-box ciego recupera el 75% del hueco (5.93 → 2.9%, piso 2.04%) pero
  con **R² ≈ 0**: la red y los parámetros se reparten el trabajo de forma
  arbitraria (la ambigüedad θ/g de F4b). Copia bien, no entiende.
- La estructural con la forma exacta **falló primero** (`e1_S` = white-box:
  aprendió r = 0 exacto). Causa: `r = clamp(raw, 0, 2)` arrancando en 0.01;
  mientras los θ están lejos el gradiente lo empuja bajo cero y queda pegado
  al borde sin gradiente. Con warm-start de θ desde el white-box y r₀ = 0.05,
  clava r_e = 0.102 y baja a 1.70% — **por debajo del piso** limpio, porque el
  piso lo fijaba un white-box con 1% de error de parámetros. `r_i` quedó en 0:
  el canal I casi no siente la refractariedad (|Δf_I| ~7× menor), no hay
  señal para moverlo, y no afecta la predicción.

## Etapa 2 — solo actuador

Planta = WC con `dP_lag/dt = (P−P_lag)/τ`, τ = 1 ms, y saturación
`3·tanh(P_lag/3)`. Es física **con memoria**: `g(I,E)` no puede por diseño.

| tag | modelo | NRMSE | R² | err. param | τ̂ (real 1.0) |
|---|---|---|---|---|---|
| `e2_wb` | white-box | 15.23% | −0.01 | 30.7% | — |
| `e2_B` | gray-box `g(I,E)` (control negativo) | 15.40% | −1.88 | 34.4% | — |
| `e2_lat_w100` | latente z∈ℝ² aprendido, 5 ms | 14.92% | −0.51 | 38.9% | — |
| `e2_lat_w400` | latente, 20 ms | 13.28% | −3.55 | 52.6% | — |
| `e2_lag` | lag estructural, h₀ "asentado" | 12.16% | 0.65 | 39.2% | 0.56 |
| **`e2_lag2`** | **lag estructural, h₀ exacto** | **2.96%** | **0.942** | **5.1%** | **0.984** |

**Qué se aprendió:**

- **El actuador solo daña tanto como las dos juntas** (15.2% vs 14.0%): el
  retardo es el problema real del trabajo previo. Confirma F7.
- `g(I,E)` no puede (15.4%): control negativo correcto.
- **El latente genérico no ayudó** (13–15%) y arruina los parámetros. Con z
  arrancando en 0 en cada ventana de 5 ms la memoria no llega a cargarse. No se
  insistió porque no era el cuello de botella; queda como trabajo futuro (z₀
  desde un encoder de la historia, o ventanas con solapamiento).
- La forma física con memoria mejoró de entrada (12.2%, R² 0.65 — lo único con
  R² positivo hasta ahí) pero **τ̂ quedó en 0.56, la mitad**. Causa: en multiple
  shooting cada ventana arrancaba con el filtro "asentado" en el comando
  (`P̂ = P(t₀)`), un salto que la planta no tiene, y el modelo lo minimiza
  achicando τ̂. Como el estado del actuador depende **solo del comando (conocido
  entero) y de τ̂**, se computa exacto filtrando toda la trayectoria
  (`LagGrayBox.filtered_inputs`, solución exacta del ZOH, diferenciable en τ̂).
  Con eso: **2.96%, τ̂ = 0.984, parámetros al 5%.**
- La **saturación no se identifica** (α cae al borde del clamp aun arrancando en
  0.1). Coherente con F1: es ~16% de la deformación del actuador y con τ̂ bien
  puesto queda poca señal, que los θ absorben. Es lo que deja `box_a1.2` en
  12.8% (donde la saturación sí muerde) mientras el resto queda < 3%.

---

## Las lecciones transversales

1. **De a una perturbación por vez funciona.** El "13%" del trabajo previo no
   era un techo del gray-box: era la suma de un problema con memoria atacado sin
   memoria más dos defectos de optimización que las dos perturbaciones juntas
   ocultaban.
2. **La forma física gana cuando existe, pero hay que inicializarla bien.** Los
   dos fracasos estructurales tuvieron la misma anatomía: un parámetro físico
   arrancando en el borde de un `clamp` (gradiente cero) mientras los θ, aún
   lejos, se llevan todo el gradiente. Receta que funcionó dos veces:
   *warm-start de θ desde el white-box + parámetro físico inicial claramente
   positivo.*
3. **En multiple shooting, el estado oculto no se supone: se computa.** Si
   depende de entradas conocidas (como el actuador), filtrarlas entero cuesta
   nada y elimina un sesgo grande. Si no (latente genérico), hay que darle al
   modelo una forma de cargarlo — pendiente.
4. **NRMSE bueno ≠ física aprendida.** El gray-box ciego llega a 2.8% con
   R² −0.5. Para usar el modelo como planta de simulación alcanza; para cancelar
   en lazo cerrado (F6) no.
5. **La ventana de entrenamiento no fue la perilla.** 5/10/20 ms dan lo mismo.
   No hace falta seguir barriéndola.

---

## Recomendación para las etapas 3 y 4

**Etapa 3 — las dos juntas.** Combinar lo que funcionó: backbone WC + término
refractario `(1−r·x)` + estado de actuador con τ̂, todo aprendible, warm-start
desde el white-box, estado inicial exacto por ventana. Número a batir: **13.1%**.
Expectativa razonable: 3–4%. Si sale, el "hueco" original queda cerrado con un
modelo interpretable y cancelable. Como segunda variante, la misma cosa pero con
`g(I,E,z)` en lugar del término refractario explícito, para medir cuánto cuesta
no conocer la forma de la parte sin memoria.

**Etapa 4 — full black-box.** Sigue teniendo sentido como cota: `ẋ = NN(x,P,Q)`
con y sin estados latentes, mismos datos y métricas. Con la referencia de la
etapa 2 en mano, la pregunta ya no es "¿el gray-box ayuda?" sino "¿cuánto paga
el prior WC frente a una red pura?". Advertencia por el resultado de 2b: la
versión con latentes va a necesitar el arreglo de inicialización de z antes de
que el número sea justo.

**Cosas para no repetir:** no más de 2 corridas en paralelo (4 carriles × 4
hilos en 8 cores multiplicó los tiempos por 50); no barrer más la ventana;
guardar `hidden` y demás flags de arquitectura en los checkpoints.

---

*Código: `scripts/esc_gen_datasets.py`, `scripts/esc_run.py`,
`scripts/esc_eval.py`, `src/neural_ode/memory.py`. Checkpoints en
`results/escalado/models/`, logs en `logs/escalado/`.*
