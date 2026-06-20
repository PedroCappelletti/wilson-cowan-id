# Guía de los archivos principales

Mapa del código: qué hace cada módulo de `src/` (la "biblioteca") y cada script de `scripts/` (los
"puntos de entrada" que se corren). Para el contexto de qué se hizo y por qué, ver
`informe_completo.md`.

---

## Vista general

```
src/                      ← la biblioteca (lógica reutilizable; no se corre directo)
  wilson_cowan/model.py   ← el modelo + la librería de estímulos
  data/                   ← generar y cargar datasets
  pinn/                   ← la PINN (red, pérdidas, entrenamiento)
  neural_ode/             ← el Neural ODE + el controlador (lazo cerrado)
  utils/                  ← config y semillas
scripts/                  ← se corren con `python scripts/<x>.py`; usan src/
```

Regla mental: **`src/` define las herramientas; `scripts/` las usa** para generar datos, entrenar,
evaluar y armar informes.

---

## Código fuente (`src/`)

### `wilson_cowan/model.py` — el modelo y los estímulos
El corazón de todo. Contiene:
- **`WilsonCowanParams`** — los parámetros del modelo (wEE, wEI, wIE, wII, te, ti, umbrales, ganancias)
  y los offsets de reposo ke/ki.
- **`WilsonCowan`** — la clase del modelo: `rhs()` (la derivada [dI/dt, dE/dt]) y `simulate()`
  (integra la trayectoria, equivalente al `ode45` del MATLAB).
- **La librería de estímulos** (funciones que devuelven `f(t)`): `box_pulse`, `square_wave_pulse`,
  `aprbs_pulse`, `prbs_pulse`, `theta_gamma_pulse`, `poisson_pulse`, `chirp_pulse`, `zero_input`
  (+ `sine_pulse`, `multisine_pulse`, legacy/descartadas).
- **`sigmoid`**, **`plot_results`** (figura de 3 paneles).

### `data/` — datasets
- **`generate.py`** → `generate_dataset(params, P, Q, …)`: simula **una** trayectoria y la devuelve
  como dict (t, I, E, y, P, Q + metadatos), con opción de ruido. + `save_dataset`/`load_dataset` (.npz).
- **`dataset.py`** → `WilsonCowanDataset`: envoltorio tipo `torch.Dataset` (para cargar datos en
  PyTorch).

### `pinn/` — la PINN (identificación por física en la pérdida)
- **`network.py`** → `PINN`: red que mapea `t → [I, E]` con **Fourier features** (rompen el sesgo
  espectral) y guarda los **pesos a identificar** (`raw_w`, vía softplus para que sean > 0).
- **`losses.py`** → `data_loss` (pega a los datos), `physics_loss` (residuo de la ODE con derivadas
  por **autograd**), `initial_condition_loss`.
- **`train.py`** → `TrainConfig` (hiperparámetros) y `Trainer`: entrena **una** trayectoria (warmup
  "datos primero", congelar la red, parada por meseta, validación temporal).
- **`multitraj.py`** → `MultiTrajPINN` (N redes, una por trayectoria, + **un θ compartido**) y
  `MultiTrajectoryTrainer`: identifica con **varias** trayectorias a la vez (rompe la degeneración wII).

### `neural_ode/` — el Neural ODE y el control
- **`dynamics.py`** → `GrayBoxWC`: el modelo de estados `f(x, P, Q) → [dI/dt, dE/dt]`. Dos modos:
  `use_correction=False` (solo ecuaciones WC con pesos aprendibles = **white-box**) y `True`
  (WC + corrección neuronal = **gray-box**).
- **`integrate.py`** → `rk4_step`, `rollout`: integrador RK4 **diferenciable** (para entrenar el
  Neural ODE y para correr el lazo).
- **`closed_loop.py`** → `IMCController` (el controlador IMC: usa los pesos para cancelar +
  realimentación con acción integral), `make_true_plant` / `make_neural_plant` (la planta verdadera
  o la aprendida), `simulate_closed_loop` (corre el lazo), `theta_gamma_refs` (la referencia).

### `utils/`
- **`config.py`** → `load_config` (lee YAML). **`seed.py`** → `set_seed`, `set_plot_style`.

---

## Scripts (`scripts/`)

### Generación de datos
| Script | Qué hace |
|---|---|
| `generate_data.py` | "Panel de control" para generar **una** trayectoria (parámetros a mano). |
| `gen_multi_dataset.py` | Genera el **dataset de control** multi-variante (20 trayectorias, todos los estímulos nuevos, régimen ms) → entrada del Neural ODE. |

### Identificación con PINN
| Script | Qué hace |
|---|---|
| `train_pinn.py` | PINN una trayectoria (pasos 4.0/4.1/4.2: forward, inverso estable, inverso ignorante). |
| `train_multi.py` | PINN multi-trayectoria (θ compartido) — recupera wII. |
| `pinn_joint_sweep.py` | PINN canónica conjunta (autograd + multi-trayectoria), barrido de ruido. |
| `train_wii.py` | Experimento dedicado a **wII** con el diseño **Q-grande**. |

### Estímulos: comparación y gráficos
| Script | Qué hace |
|---|---|
| `compare_estimulos.py` | Compara familias de estímulo por **error paramétrico** (identifica con cada una). |
| `demo_estimulos.py` | Demo de cobertura del espacio de estados por estímulo (retratos de fase). |
| `graficos_estimulos.py` | Gráficos **antes vs ahora** de los estímulos + respuesta del sistema. |

### Neural ODE y control
| Script | Qué hace |
|---|---|
| `train_neural_ode.py` | Entrena el Neural ODE (multiple shooting) sobre el dataset de control. |
| `eval_closed_loop.py` | Corre el lazo cerrado: controlador sobre planta verdadera y aprendida, con pesos reales o θ̂. |

### Robustez bajo ruido
| Script | Qué hace |
|---|---|
| `noise_sweep.py` | Barrido de ruido de la PINN (método de dos etapas + FD). |
| `noise_robustness.py` | Ruido en la cadena Neural ODE → control (método ingenuo). |
| `noise_improve.py` | Diagnóstico: qué palanca (suavizado / estímulo fuerte) mejora θ̂ a ruido alto. |
| `noise_refine.py` | Refinamiento del suavizado a ruido alto (k=7 vs 11, set XL). |
| `noise_final.py` | Barrido final con suavizado **adaptativo** (mejor resultado). |

### Informes (generan HTML)
| Script | Qué hace |
|---|---|
| `informe_neural_ode.py` | Simulador vs Neural ODE (lazo abierto y cerrado). |
| `informe_integral.py` | **Informe integral** de todo el trabajo (`docs/informe_integral.html`). |
| `evaluate.py` | Evaluación/figuras auxiliares de un checkpoint. |

---

## Flujo típico (cómo se encadenan)

```
1. gen_multi_dataset.py        → data/processed/control/multi_dataset.npz
2. train_neural_ode.py         → results/models/neural_ode.pt   (modelo aprendido)
3. eval_closed_loop.py         → corre el controlador con ese modelo
4. noise_final.py              → robustez bajo ruido
5. informe_integral.py         → docs/informe_integral.html (visual de todo)
```

Para la rama PINN: `gen` de trayectorias → `train_multi.py` / `train_wii.py` (identifica θ) →
`compare_estimulos.py` (compara estímulos).

---

## Dónde tocar para…

- **Agregar un estímulo nuevo:** `src/wilson_cowan/model.py` (definir el generador) + exportarlo en
  `src/wilson_cowan/__init__.py`.
- **Cambiar qué trayectorias se generan:** la `ZONA EDITABLE` de `gen_multi_dataset.py` (o el script
  de identificación correspondiente).
- **Cambiar la arquitectura de la PINN / Neural ODE:** `src/pinn/network.py` / `src/neural_ode/dynamics.py`.
- **Cambiar el controlador:** `src/neural_ode/closed_loop.py`.
- **Cambiar hiperparámetros de entrenamiento:** `TrainConfig` (PINN) o la `ZONA EDITABLE` del script.
