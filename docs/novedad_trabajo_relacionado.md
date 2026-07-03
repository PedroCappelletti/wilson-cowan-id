# Novedad y trabajo relacionado

> Bitácora viva de **qué es posiblemente novedoso** en lo que hacemos y **qué ya está publicado** (related work). Para la sección de *related work* del informe y para charlar con el tutor. Se va actualizando.

> ⚠️ **Búsqueda limitada** (pocos queries, sin bases pagas). Para afirmar novedad en serio: búsqueda sistemática (Scholar + arXiv + IEEE Xplore) **y preguntarle al tutor** (conoce su línea mejor que cualquier búsqueda).

---

## 🧭 Síntesis: estado del arte y cómo nos diferenciamos
Todo lo previo, en una tabla. La columna "qué NO hace" es donde entramos nosotros.

| Línea de trabajo | Referencia clave | Qué aporta | Qué NO hace | Nuestra diferencia |
|---|---|---|---|---|
| **Modelo** | Wilson-Cowan 1972 | El modelo E/I | — | Es nuestro objeto de estudio |
| **Control del WC** (línea del tutor) | Martínez, LPV optogenético 2024 | Marco de control (LPV, optogenético) sobre WC | — | Marco de control que el proyecto **complementa**: aportamos la identificación de parámetros desde datos, aguas arriba |
| **Métodos de identificación SciML** | SINDy, PINN, Neural ODE, RK-PINN | Identifican dinámica/params | Genéricos; no sobre WC completo con ruido | Aplicados a los **10 params de WC** con arranque ignorante |
| **Identificabilidad práctica** | Profile likelihood; (Column) Subset Selection | Deciden qué es identificable / qué fijar | No aplicado a WC + Neural ODE | Diagnóstico **Fisher+SVD** que señala **wII**, confirmado con ruido |
| **Diseño de estímulos (OED)** | Adaptive Stimulus Design (RNN); OED | Estímulos que mejoran la identificación | Otro modelo; **sin control** | WC específico + **cierre del lazo de control** |
| **Control aprendido / closed-loop neuro** | Fehrman, Madondo, Steffen, Koopman-MPC | Control sobre modelos neuronales | Suelen ser caja negra / otros modelos | Params **interpretables** + validación end-to-end |
| **Gray-box / UDE** | El-Gazzar & van Gerven 2025 | Marco WC + red aprendible | Review; sin resultado empírico ruido/control | **Resultado empírico** + diagnóstico de identificabilidad |

**Nuestro diferencial en una frase:** *somos, hasta donde vimos, el primer trabajo que identifica los 10 parámetros de Wilson-Cowan con una Neural ODE, diagnostica con Fisher+SVD qué se puede recuperar (wII es el límite), y demuestra que la fragilidad de la identificación no se propaga al control en lazo cerrado.* Los **métodos** (subset selection, OED) son prestados y citados; la **aplicación + el hallazgo + la validación de control** son el aporte.

**Antecedente más peligroso a citar/diferenciar:** [Adaptive Stimulus Design for Dynamic RNN](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6349832/) — ya muestra que estímulos óptimos recuperan mejor constantes de tiempo y pesos en un modelo neuronal. No nos invalida (modelo distinto, sin control), pero hay que nombrarlo.

---

## 🟢 Qué parece novedoso (nuestro aporte como paquete)
Las piezas por separado están publicadas; la novedad está en la **combinación** y el resultado empírico sobre Wilson-Cowan:

1. **Identificar los 10 parámetros de WC** con una Neural ODE desde arranque ignorante (no solo los 4 pesos). No encontré esto hecho.
2. **Bucle predecir → confirmar:** Fisher+SVD predice *sin entrenar* que **wII** es el menos identificable, y el barrido de ruido lo **confirma** (41% a σ=0.10). Conecta un método de OED con un resultado empírico en *este* modelo.
3. **La fragilidad de la identificación NO se propaga al control:** cuantificado end-to-end, la acción integral del IMC absorbe el error aun con wII al 41%.

## 📚 Trabajo relacionado (ya publicado — no es novedad por sí solo)
- **Identificabilidad de neural mass models = problema conocido** — [bioRxiv 480012](https://www.biorxiv.org/content/10.1101/480012v1.full) (identifiability en NMM de conectoma).
- **WC como UDE / con parámetros libres** — [El-Gazzar & van Gerven 2025](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2025.1677930/full). Ver Bibliografía del proyecto - Wilson-Cowan ID.
- **Fisher / información en sistemas dinámicos** — [arXiv 2506.18498](https://arxiv.org/pdf/2506.18498); OED con Fisher+SVD (Plate et al. 2024, ya citado).
- **Neural ODE para identificación + control** — [MPC con Neural ODE](https://www.sciencedirect.com/science/article/abs/pii/S0098135423002375), [identificación de sistemas realimentados](https://arxiv.org/abs/2503.22601), [Neural ODEs como políticas de control](https://arxiv.org/abs/2210.11245).
- **Control del WC optogenético (línea del tutor)** — [Martínez, LPV optogenético 2024](https://pubmed.ncbi.nlm.nih.gov/38653250/): marco de control LPV/optogenético sobre WC. Es la **línea de control que el proyecto complementa** — nuestra pieza es la identificación de parámetros desde datos, aguas arriba de ese control. *(Encuadre a confirmar con el tutor.)*
- Candidatos cercanos a chequear a mano: el review de UDEs (El-Gazzar) y la review del tutor [12] "Dynamical models from a closed-loop control perspective".

---

## 🔬 Direcciones novedosas a explorar (ideas del equipo)
Experimentos que reforzarían la contribución. Todos se apoyan en el diagnóstico Resultado - Robustez al ruido (10 params).

### A. Fijar / regularizar wII y ver si mejora el resto
**Hipótesis:** wII domina la dirección singular más débil (σ₁₀); si lo fijás (o lo regularizás fuerte), "sacás" esa dirección plana y el resto debería identificarse mejor bajo ruido.
- [ ] Fijar wII en su valor verdadero → re-identificar los otros 9 bajo ruido → ¿baja el error de ti, ai, ae?
- [ ] En vez de fijar, **regularizar** wII (penalización/prior) con distintas fuerzas → curva error-resto vs fuerza de regularización.
- **Métrica:** error por parámetro de los otros 9, vs el caso "10 libres".

### B. Qué parámetros conviene fijar para mejorar cuáles (guiado por FIM)
**La FIM ya da la hipótesis:** las direcciones débiles agrupan parámetros acoplados → fijar uno del grupo debería liberar a los otros.
- Direcciones débiles observadas: σ₁₀ = **wII**; σ₉ = acople **ae–te–wEE**; σ₈ = acople **ai–ti–thetai**.
- [ ] Probar fijar un representante de cada acople (ej. `ae`, `ai`) y medir la mejora del resto del grupo.
- [ ] Construir una **tabla "qué fijar → a quién ayuda"** derivada de los vectores singulares y validada empíricamente.
- **Idea de fondo:** convertir la FIM en una receta accionable de "conocimiento previo mínimo" (qué N parámetros hay que conocer para identificar el resto con robustez X).

### C. Identificabilidad dependiente del estímulo (chirp, APRBS, PRBS, theta-gamma…)
**Sí, el estímulo cambia la identificabilidad:** la FIM depende de la trayectoria, que depende del estímulo → distintos estímulos vuelven identificables distintos parámetros. Es *optimal input design* (Ljung; Plate).
- [ ] Calcular la **FIM por familia de estímulo** (box, square, APRBS, PRBS, theta-gamma, poisson, chirp) → ¿qué estímulo hace más identificable a wII? ¿a las constantes de tiempo?
- [ ] Barrido de ruido **por estímulo** (no solo con la mezcla) → confirmar la predicción de la FIM.
- [ ] **Diseñar la mezcla óptima:** combinar los estímulos que cubren direcciones complementarias → maximizar identificabilidad conjunta. (Antecedente propio: box `s2/s1≈0.13` vs Q-grande `≈0.64`.)
- **Métrica:** FIM (valores singulares / número de condición) y error por parámetro, por estímulo y por mezcla.

> Estas tres alimentan el **Eje 3** de Plan de trabajo - Neural ODE. Juntas cuentan una historia potente: *"la identificabilidad de WC es estructural y dependiente del estímulo; la FIM dice qué conocer y qué inyectar para identificar el resto"* — que es más novedoso que solo "identificamos 10 params".

---

## 🧱 Base metodológica (¡las ideas A/B/C ya tienen métodos establecidos!)
Research (jul-2026): las tres ideas **NO son nuevas como método** — son metodologías con nombre propio. Buena noticia: tenés **fundamentos citables** y no reinventás nada. La novedad queda en **aplicarlas a WC + Neural ODE** y en el resultado (wII) + la validación por control.

### Para A y B (fijar / regularizar / qué fijar) → *Parameter Subset Selection* y *Profile Likelihood*
- **Column Subset Selection sobre la matriz de sensibilidad** — [arXiv 2205.04203](https://arxiv.org/pdf/2205.04203): identifica qué ejes de parámetro caen más cerca de las direcciones mal condicionadas de la FIM y los **fija en su prior**, dejando activos los bien condicionados. **Es literalmente la idea B**, con receta numérica robusta.
- **Selecting Sensitive Parameter Subsets (FIM-based)** — [PMC6056202](https://pmc.ncbi.nlm.nih.gov/articles/PMC6056202/): estima el subconjunto sensible, fija el resto en valores preliminares. (Idea A/B en biomecánica.)
- **Profile Likelihood** (Raue et al. 2009) — [Bioinformatics 25(15):1923](https://academic.oup.com/bioinformatics/article/25/15/1923/213246): el método canónico de identificabilidad *práctica*; barre cada parámetro y detecta los no identificables (perfil plano). Estándar para decidir qué fijar.
- **Regularización guiada por autovectores de la FIM** — se agregan términos de regularización sobre las direcciones no identificables para volver *todos* los params prácticamente identificables. **Es la variante "regularizar en vez de fijar" de la idea A.** (ver framework en [PMC12463036](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12463036/))
- **Subset profiling** (combinaciones identificables) — [ScienceDirect S0025556414001631](https://www.sciencedirect.com/science/article/abs/pii/S0025556414001631).

### Para C (identificabilidad dependiente del estímulo) → *Optimal Input / Experimental Design (OED)*
- **Adaptive Stimulus Design for Dynamic Recurrent Neural Network Models** — [PMC6349832](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6349832/): ⭐ **el más cercano a la idea C** — estímulos diseñados óptimamente recuperan **mejor las constantes de tiempo y los pesos de conexión** de un modelo neuronal. Léelo para posicionar/diferenciar.
- **OED para identificabilidad práctica y discriminación de modelos** — [arXiv 2506.11311](https://arxiv.org/html/2506.11311v1).
- **D-optimal input design para sistemas no lineales** — minimiza la covarianza de los estimados (inversa de la FIM) eligiendo el input. Marco general (Ljung; tesis D-optimal). Nota: para no lineales el OED es difícil (la FIM depende de momentos de orden alto del input) → se hace numérico.
- **OED con ruido de observación** — [arXiv 2504.19233](https://arxiv.org/pdf/2504.19233).
- Neuro-específico: estimación de params de neural mass model desde EEG — [Springer s12021-018-9369-x](https://link.springer.com/article/10.1007/s12021-018-9369-x).

### Cómo lo usamos (encuadre honesto para el informe)
- **Método = prestado y citable** (subset selection / profile likelihood / OED). No lo vendemos como nuestro.
- **Novedad = la aplicación**: primer uso (hasta donde vimos) de este arsenal sobre **Wilson-Cowan identificado con Neural ODE**, con el hallazgo concreto (wII como dirección débil) y la **validación orientada al control** (la fragilidad no se propaga). El diferencial vs [PMC6349832] es el cierre del lazo de control y el modelo WC específico.
- ⚠️ Igual: búsqueda acotada → confirmar con búsqueda sistemática y con el tutor.

## 🔗 Conexiones
- Proyecto: Investigación Neurociencia · Plan: Plan de trabajo - Neural ODE
- Resultados: Resultado - Robustez al ruido (10 params) · Resultado - Identificación completa 10 parámetros
- Concepto: Identificabilidad (Fisher + SVD) - explicación visual
- Bibliografía: Bibliografía del proyecto - Wilson-Cowan ID
