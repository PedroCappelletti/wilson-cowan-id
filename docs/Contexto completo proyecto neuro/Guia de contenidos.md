---
type: resource
topic: Wilson-Cowan — guía de estudio integral
source: "Contexto completo proyecto neuro (paquete de estudio); wilson-cowan-id; teoria-wilson-cowan"
tags: [resource, neurociencia, wilson-cowan, guia-estudio, moc]
---

# Guía de contenidos — Proyecto Wilson-Cowan (paquete de estudio)

> [!abstract] Qué es este documento
> El **mapa maestro** de todo lo que hay que entender para dominar el proyecto de identificación de Wilson-Cowan con Neural ODE / PINN. **No desarrolla** los temas: los **lista, ordena y conecta**, y te dice en qué archivo del paquete está cada uno. Pensalo como el índice + la ruta de estudio. El desarrollo vive en los PDF (bloques 1, 2 y 4) y en los `.md` (bloque 3) de esta misma carpeta.

> [!tip] Cómo usar este paquete en el vuelo
> 1. Leé esta guía entera primero (10 min): te da el panorama y el porqué de cada pieza.
> 2. Seguí la **[Ruta de estudio sugerida](#🛫-ruta-de-estudio-sugerida-para-el-vuelo)** (más abajo), que **mezcla los 3 bloques** en el orden que minimiza "saltos al vacío".
> 3. Si preferís lo canónico, seguí los bloques 1 → 2 → 3 → 4 en orden.
> 4. Cada tema tiene un marcador de dependencia (**⟵ requiere …**) para que sepas qué leer antes.

---

## Estructura del paquete

```
Contexto completo proyecto neuro/
├── Guia de contenidos.(md/pdf)      ← este archivo
├── 1 - Teoria matematica/           ← PDFs (desde LaTeX), con bibliografía
├── 2 - Teoria biologica/            ← PDFs (desde LaTeX), con bibliografía
├── 3 - Archivos del proyecto/       ← Markdown con diagramas (manual del código)
├── 4 - Resultados y conclusiones/   ← PDFs (desde LaTeX): experimentos, datos reales, novedad
└── figuras/                         ← imágenes usadas por los documentos
```

**Convención de dificultad:** 🟢 base / puente · 🟡 núcleo del proyecto · 🔴 avanzado / fino.

---

## 🧮 Bloque 1 — Teoría matemática

> El andamiaje de cálculo, álgebra lineal, sistemas dinámicos y machine learning que sostiene el método. Ordenado de los cimientos (M1) al diagnóstico fino (M6). Los documentos M3, M5 y M6 son el corazón; M1–M2–M4 son el puente desde tu base de Bioingeniería.

### M1 · Repaso: cálculo y álgebra lineal para el proyecto 🟢
**Archivo:** `1 - Teoria matematica/M1_repaso_calculo_algebra.pdf`
Puente. Solo lo que se usa después.
- Derivada parcial, gradiente, regla de la cadena.
- **Jacobiano** (derivada de una función vectorial) y **Hessiano** (curvatura). ⟵ base de M6.
- Matrices: producto, transpuesta, simetría, rango.
- **Autovalores y autovectores**; matrices simétricas definidas positivas.
- **Formas cuadráticas** y su geometría (elipsoides). ⟵ base de "valle plano" en M6.
- Normas de vector y matriz; ortogonalidad; **número de condición**.
- Serie de Taylor de 2.º orden (aproximación cuadrática del error).

### M2 · Ecuaciones diferenciales y sistemas dinámicos 🟢🟡
**Archivo:** `1 - Teoria matematica/M2_edos_sistemas_dinamicos.pdf`
La lente para *leer* Wilson-Cowan como sistema. ⟵ requiere M1.
- Qué es una EDO; problema de valor inicial; campo vectorial / flujo.
- **Espacio de estados** y **retrato de fase** (plano de fase para 2D).
- Puntos de equilibrio; **linealización** (Jacobiano de estado ∂f/∂x) y clasificación por autovalores.
- Estabilidad; nulclinas.
- **Oscilaciones y ciclo límite**; **bifurcación de Hopf** (cómo nace un ritmo). ⟵ conecta con B3.
- **Rigidez (stiffness)** de un sistema: qué es y por qué fija el paso de integración. ⟵ conecta con M3.

### M3 · Integración numérica y costo computacional 🟡
**Archivo:** `1 - Teoria matematica/M3_integracion_numerica.pdf`
Cómo se resuelve una EDO en la compu y qué cuesta. ⟵ requiere M2.
- Método de Euler; **Runge-Kutta 4 (RK4)**; paso fijo vs adaptativo (Dormand-Prince / `ode45`/`RK45`).
- Error local y global; orden de un método.
- **NFE** (número de evaluaciones de f) como medida de costo.
- Rigidez → paso estable; por qué WC admite paso grande (y la salvedad de los estímulos conmutados).
- Integrador **diferenciable** (por qué importa para entrenar). ⟵ conecta con M5.

### M4 · Fundamentos de machine learning y optimización 🟢🟡
**Archivo:** `1 - Teoria matematica/M4_fundamentos_ml_optimizacion.pdf`
Puente al SciML. ⟵ requiere M1.
- Aprendizaje supervisado; parámetros; **función de pérdida** (MSE).
- **Descenso por gradiente**; tasa de aprendizaje; **Adam** y **L-BFGS** (por qué se usan los dos).
- Sobreajuste, train/validación, regularización L2 (una primera mirada; el detalle en M6).
- **Diferenciación automática (autograd):** modo *reverse* (gradientes) vs *forward* (Jacobianos, `jacfwd`). ⟵ base de M5 y M6.
- Warmup, congelamiento de capas, parada por meseta (trucos usados en el repo).

### M5 · Aprendizaje científico: Neural ODE y PINN 🟡🔴 ⭐
**Archivo:** `1 - Teoria matematica/M5_neural_ode_pinn.pdf`
El método de identificación. ⟵ requiere M2, M3, M4.
- Redes neuronales: **MLP**, activaciones, **sesgo espectral** y **Fourier features**.
- **Neural ODE:** aprende la dinámica ẋ = f_θ(x); backprop *a través del solver*; método *adjoint*.
- Enfoque **gray-box / UDE:** física conocida (WC) + corrección neuronal opcional; parámetros interpretables.
- **Multiple shooting:** por qué partir la trayectoria en ventanas estabiliza el entrenamiento.
- **PINN:** aprende la solución t↦x; **residuo físico** por autograd; datos + física en la pérdida.
- **Reparametrización softplus** (positividad) y offsets ke,ki derivados.
- **PINN vs Neural ODE:** tabla comparativa y cuándo cada una (control ⟵ conecta con P4 y B5).

### M6 · Identificabilidad: sensibilidad, SVD, Fisher, Cramér-Rao y OED 🔴 ⭐
**Archivo:** `1 - Teoria matematica/M6_identificabilidad_fisher_svd_oed.pdf`
El diagnóstico que decide *qué se puede recuperar*. ⟵ requiere M1, M4.
- **Matriz de sensibilidad** J = ∂y/∂θ (Jacobiano de salida); sensibilidad relativa.
- **SVD** de J: valores/vectores singulares; direcciones planas; número de condición.
- **Matriz de información de Fisher (FIM)** = JᵀJ/σ²; por qué "Fisher = curvatura del error".
- **Cota de Cramér-Rao (CRB):** la incertidumbre *correcta* (diagonal de la inversa), no ‖columna‖.
- Identificabilidad **estructural vs práctica**; **profile likelihood**.
- **Selección de subconjuntos** (QR con pivoteo): qué parámetro fijar.
- **Regularización** Tikhonov / ridge / MAP; el continuo λ→∞ ≡ fijar.
- **Diseño óptimo de experimentos (OED):** A/D/E-óptimo; **excitación persistente**; por qué el estímulo *es* la información.
- El hallazgo del proyecto: **wII** vive en la dirección plana (predicho sin entrenar, confirmado con ruido).

---

## 🧠 Bloque 2 — Teoría biológica

> La motivación y el sustrato biológico: de la neurona al ritmo cerebral, el modelo de Wilson-Cowan como modelo de *poblaciones*, y por qué querríamos identificarlo y controlarlo (optogenética / neuromodulación). Aquí vive el modelo neurobiológico central (B3).

### B1 · La neurona, la sinapsis y la tasa de disparo 🟢
**Archivo:** `2 - Teoria biologica/B1_neurona_sinapsis_tasa_disparo.pdf`
De la célula a la variable que modela WC.
- Neurona: soma, dendritas, axón; potencial de membrana y **potencial de acción** (spike), en breve.
- **Sinapsis**: excitatoria vs inhibitoria; neurotransmisores (glutamato / GABA), muy general.
- De **spikes** a **tasa de disparo (firing rate)**: promediar en tiempo/población.
- **Curva f–I** (input–output de una neurona) → justifica la sigmoidea.
- Por qué modelar **poblaciones** y no neuronas individuales.

### B2 · Poblaciones neuronales y el balance excitación/inhibición 🟢🟡
**Archivo:** `2 - Teoria biologica/B2_poblaciones_balance_EI.pdf`
El nivel de descripción de WC. ⟵ requiere B1.
- **Modelos de tasa (rate models)** y de **campo medio (mean-field)**: qué promedian y qué pierden.
- Poblaciones **E** e **I**; conectividad recurrente (los 4 pesos wEE, wEI, wIE, wII).
- La **sigmoidea** como función de activación poblacional (ganancia y umbral).
- **Balance E/I**: por qué el tira-y-afloja E↔I genera dinámica rica; su relevancia funcional y clínica.
- Ubicación de WC en la jerarquía de modelos (Hodgkin-Huxley → integrate-and-fire → rate → neural mass → neural field).

### B3 · El modelo de Wilson-Cowan 🟡🔴 ⭐
**Archivo:** `2 - Teoria biologica/B3_modelo_wilson_cowan.pdf`
El corazón biológico-matemático del proyecto. ⟵ requiere B2, M2.
- Historia y motivación: **Wilson & Cowan (1972)**; qué problema resolvían.
- **Derivación** de las dos ecuaciones (relajación + sigmoidea de la entrada total).
- Significado biológico de **cada uno de los 10 parámetros** (pesos, constantes de tiempo, ganancias, umbrales) y de los offsets ke, ki.
- Estímulos externos **P (→E)** y **Q (→I)**; salida y = E − I.
- **Régimen oscilatorio:** cómo el acople E/I produce un **ciclo límite** (retrato de fase, nulclinas, Hopf). ⟵ usa M2.
- Parámetros por defecto del simulador y por qué generan oscilación sostenida.
- Relación con **ritmos** (gamma/theta) y variantes del modelo (neural field). ⟵ conecta con B4.

### B4 · Ritmos cerebrales y acoplamiento theta-gamma 🟡
**Archivo:** `2 - Teoria biologica/B4_ritmos_cerebrales_theta_gamma.pdf`
Qué produce el modelo y por qué importa. ⟵ requiere B3.
- Bandas: delta, theta, alpha, beta, **gamma**; qué son y a qué se asocian.
- **Acoplamiento theta-gamma** (ráfagas gamma moduladas por theta): el régimen propio del proyecto.
- Cómo se mide la actividad poblacional: **EEG / LFP**; y = E − I como *proxy* de potencial.
- Función de los ritmos (comunicación, memoria) y relevancia clínica (epilepsia, Parkinson).

### B5 · Neuroestimulación y control en lazo cerrado 🟡 ⭐
**Archivo:** `2 - Teoria biologica/B5_neuroestimulacion_control.pdf`
El *para qué* del proyecto. ⟵ requiere B3, B4; conecta con M5 y P4.
- **Optogenética**: qué es, por qué el estímulo es **on/off y ≥ 0** (limita las señales realizables). ⟵ explica la librería de estímulos.
- **DBS (estimulación cerebral profunda)** y neuromodulación de lazo abierto vs **lazo cerrado (closed-loop)**.
- Por qué **identificar** el modelo antes de **controlar**: gemelo digital → controlador.
- El objetivo del proyecto: inducir/mantener un ritmo (theta-gamma) con un controlador robusto; por qué la fragilidad de identificar wII **no** rompe el control.
- Antecedente cercano: *Adaptive Stimulus Design*; encuadre de novedad.

---

## 💻 Bloque 3 — Archivos del proyecto

> Manual del repositorio `wilson-cowan-id`: qué hace cada archivo y cada función, con diagramas (Mermaid) del flujo. En Markdown para que se lea/renderice en Obsidian. Ordenado de la vista general (P0) al detalle por módulo.

### P0 · Arquitectura general y flujo de datos 🟢 ⭐
**Archivo:** `3 - Archivos del proyecto/P0_arquitectura_y_flujo.md`
- Mapa `src/` (biblioteca) vs `scripts/` (puntos de entrada); regla mental.
- Diagrama del **pipeline** completo (generar datos → entrenar → evaluar → informe).
- Diagrama de dependencias entre módulos.
- Dónde tocar para cada tipo de cambio.
- `configs/`, `pyproject.toml`, `requirements.txt`, `tests/`.

### P1 · `src/wilson_cowan` — modelo y librería de estímulos 🟡
**Archivo:** `3 - Archivos del proyecto/P1_src_wilson_cowan.md`
- `WilsonCowanParams` (los 10 parámetros + ke/ki derivados), `sigmoid`.
- `WilsonCowan`: `rhs()` (la derivada) y `simulate()` (integración, ≈ `ode45`).
- **Librería de estímulos** función por función: `box_pulse`, `aprbs_pulse`, `theta_gamma_pulse`, `square_wave_pulse`, `prbs_pulse`, `poisson_pulse`, `chirp_pulse`, `sine/multisine` (legacy), `zero_input`.
- `plot_results` (figura de 3 paneles).

### P2 · `src/data` — generación y datasets 🟡
**Archivo:** `3 - Archivos del proyecto/P2_src_data.md`
- `generate.py`: `generate_dataset`, ruido, `save/load_dataset` (.npz).
- `dataset.py`: `WilsonCowanDataset` (envoltorio tipo `torch.Dataset`).

### P3 · `src/pinn` — la PINN 🔴
**Archivo:** `3 - Archivos del proyecto/P3_src_pinn.md`
- `network.py`: `PINN` (Fourier features + pesos a identificar vía softplus).
- `losses.py`: `data_loss`, `physics_loss` (autograd), `initial_condition_loss`.
- `train.py`: `TrainConfig`, `Trainer` (warmup, congelar red, parada por meseta).
- `multitraj.py`: `MultiTrajPINN` y `MultiTrajectoryTrainer` (θ compartido, rompe la degeneración de wII).

### P4 · `src/neural_ode` — dinámica, integración y control 🔴 ⭐
**Archivo:** `3 - Archivos del proyecto/P4_src_neural_ode.md`
- `dynamics.py`: `GrayBoxWC` (white-box vs gray-box, `use_correction`).
- `integrate.py`: `rk4_step`, `rollout` (integrador RK4 diferenciable).
- `closed_loop.py`: `IMCController`, `make_true_plant` / `make_neural_plant`, `simulate_closed_loop`, `theta_gamma_refs`.

### P5 · `src/utils` y configuración 🟢
**Archivo:** `3 - Archivos del proyecto/P5_src_utils_config.md`
- `config.py` (`load_config`), `seed.py` (`set_seed`, `set_plot_style`), `configs/default.yaml`.

### P6 · `scripts/` — puntos de entrada (mapa por propósito) 🟡
**Archivo:** `3 - Archivos del proyecto/P6_scripts.md`
- Agrupados: generación de datos · identificación PINN · estímulos · Neural ODE y control · robustez al ruido · experimentos de identificabilidad (Fisher/subset/OED) · informes.
- Qué script corre cada experimento del informe (mapa experimento ↔ script ↔ figura).

---

## 📊 Bloque 4 — Resultados y conclusiones

> Qué se hizo y qué salió: los experimentos del proyecto, el trabajo con **datos reales experimentales** (lo más reciente), y las conclusiones y la novedad. Cierra el círculo teoría → código → resultados.

### R1 · Experimentos y resultados 🟡 ⭐
**Archivo:** `4 - Resultados y conclusiones/R1_experimentos_y_resultados.pdf`
Recorrido por cada experimento (objetivo → hipótesis → método → resultado → por qué importa):
- Simulador y datasets; identificación en limpio (4 pesos → 10 params, error máx 1.14 %).
- Robustez al ruido + diagnóstico Fisher/SVD (wII: 0.8 % → 41 % a σ=0.10, predicho y confirmado).
- Remedios (fijar/regularizar wII, QR-pivoteo, OED por familia) y OED ponderado (V-shape, óptimo ρ=0.5).
- Familia y límite del dt grande; costo computacional (WC no rígido → RK4 ~5× más barato).
- Validación orientada al control: la fragilidad de wII **NO se propaga**.
- Tabla-resumen experimento → resultado → novedad.

### R2 · Datos reales experimentales 🔴 ⭐
**Archivo:** `4 - Resultados y conclusiones/R2_datos_reales_experimentales.pdf`
Lo más reciente del proyecto. ⟵ requiere M5, M6.
- Los datos (LFP + chirp, `data8`), conversión, identificación WC **por salida** (estado latente por ventana).
- El **hallazgo central**: el estímulo gobierna solo ~10 % de la respuesta → hay que identificar en **modo predicción a horizonte corto** (R²≈0.85), no exigiendo un rollout libre (techo real ~0.1).
- Protocolos A/B, conexión con la FIM (direcciones planas), y próximos pasos.

### R3 · Conclusiones, novedad y trabajo futuro 🟡
**Archivo:** `4 - Resultados y conclusiones/R3_conclusiones_novedad.pdf`
- Síntesis del proyecto; encuadre honesto de novedad ("métodos de X aplicados a Y con Z"); qué se puede afirmar vs qué queda abierto; conclusiones por eje; y el trabajo futuro.

---

## 🛫 Ruta de estudio sugerida para el vuelo

> Orden lineal que mezcla los bloques minimizando prerequisitos faltantes. Cada tramo deja listo lo que necesita el siguiente.

**Tramo 1 — El porqué (motivación biológica).** Arrancá enganchándote con la biología:
`B1` → `B2` → `B5` (leé B5 sin el detalle de control; volvés después).

**Tramo 2 — La lente matemática mínima.** Para poder leer el modelo como sistema:
`M1` → `M2`.

**Tramo 3 — El modelo central.** Ya tenés biología + dinámica:
`B3` (Wilson-Cowan) → `B4` (los ritmos que produce) → `P1` (verlo en código).

**Tramo 4 — El método.** Cómo se identifica:
`M3` (integración) → `M4` (ML) → `M5` (Neural ODE / PINN) → `P2`, `P3`, `P4` (el código del método y el control).

**Tramo 5 — El diagnóstico fino.** La parte más conceptual y el aporte del trabajo:
`M6` (Fisher + SVD + OED) → releer `B5` (control robusto) → `P6` (mapa experimento↔script) → `P0` (cerrar con la vista de arquitectura).

**Tramo 6 — Resultados y cierre.** Con todo el marco puesto, ver qué salió:
`R1` (experimentos y resultados) → `R2` (datos reales experimentales) → `R3` (conclusiones y novedad).

> [!note] Atajo si vas corto de tiempo
> Núcleo mínimo para "entender el proyecto": **B3 + M5 + M6 + R1 + P0**. Con eso tenés el modelo, el método, el diagnóstico, los resultados y el mapa del código.

---

## 🔗 Correspondencia con material que ya existe

Este paquete **sintetiza y completa** lo que ya está; si querés ir a la fuente:

| Tema del paquete | Ya existía en… |
|---|---|
| M5, M6 (versión previa) | `teoria-wilson-cowan/02_neural_ode_pinn.tex`, `01_fundamentos_matematicos.tex` y sus PDF |
| M6 (nota viva) | Vault: *Fundamentos teóricos - Identificabilidad, SVD, Fisher y OED*; *Identificabilidad (Fisher + SVD) - explicación visual* |
| M5 | Vault: *Neural ODEs*; *PINN vs Neural ODE* |
| B3 | Vault: *Modelo Wilson-Cowan* |
| B2, B4 | Vault: *Neurociencia computacional* (hub) |
| B5 / control | Vault: *Control en lazo cerrado (IMC)*; *Controlador IMC - cómo funciona y por qué es robusto* |
| P0–P6 | Repo: `docs/estructura_codigo.md`, `docs/informe_completo.md` y el código comentado |
| R1, R3 (resultados / novedad) | Repo: `docs/informe_wilson_cowan.md`, `docs/novedad_trabajo_relacionado.md`; Vault: notas *Resultado - …*, *Novedad y trabajo relacionado* |
| R2 (datos reales) | Repo: `docs/identificacion_datos_reales.md`, `docs/plan_datos_reales.md` |

---

## Checklist de lectura

- [ ] B1 · Neurona, sinapsis, firing rate
- [ ] B2 · Poblaciones y balance E/I
- [ ] B5 · Neuroestimulación y control (primera pasada)
- [ ] M1 · Repaso cálculo / álgebra lineal
- [ ] M2 · EDOs y sistemas dinámicos
- [ ] B3 · Modelo de Wilson-Cowan ⭐
- [ ] B4 · Ritmos cerebrales y theta-gamma
- [ ] P1 · Código del modelo y estímulos
- [ ] M3 · Integración numérica
- [ ] M4 · Fundamentos de ML
- [ ] M5 · Neural ODE y PINN ⭐
- [ ] P2 · Datos · P3 · PINN · P4 · Neural ODE y control
- [ ] M6 · Identificabilidad (Fisher + SVD + OED) ⭐
- [ ] B5 · relectura (control robusto)
- [ ] P6 · scripts · P0 · arquitectura
- [ ] R1 · Experimentos y resultados ⭐
- [ ] R2 · Datos reales experimentales ⭐
- [ ] R3 · Conclusiones y novedad
