# Fundamentos teóricos — Identificabilidad, SVD, Fisher y OED

> **Qué es esta nota:** el "temario" de la matemática que hay que entender para leer y defender el trabajo de identificación (Experimentos A/B/C, Resultado - Subset selection, regularización y diseño de estímulo). Cada sección tiene **definición → intuición → cómo aparece en el proyecto**. Para la versión con analogías visuales, ver Identificabilidad (Fisher + SVD) - explicación visual.

> [!tip] Orden de estudio sugerido
> **§1 SVD** → **§2 sensibilidad/Jacobiano** → **§3 Fisher** → **§4 Cramér-Rao** → **§5 identificabilidad** → luego los remedios (**§6 subset selection**, **§7 regularización**, **§8 OED**). Las piezas de ML (**§9**) se pueden leer aparte. La cadena lógica es: *sensibilidad → Fisher → SVD de Fisher → identificabilidad → qué hacer al respecto*.

---

## 1. Descomposición en valores singulares (SVD)

**Definición.** Toda matriz $A \in \mathbb{R}^{m\times n}$ se factoriza como
$$A = U\,\Sigma\,V^\top,$$
con $U$ ($m\times m$) y $V$ ($n\times n$) **ortonormales** y $\Sigma$ diagonal con los **valores singulares** $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$. Las columnas de $V$ (**vectores singulares derechos**) son direcciones en el *espacio de entrada*; las de $U$ (izquierdos), en el de salida.

**Datos clave:**
- $\sigma_i^2$ = autovalores de $A^\top A$. Los $v_i$ = autovectores de $A^\top A$.
- **Geometría:** $A$ manda la esfera unitaria a un **elipsoide** cuyos semiejes miden $\sigma_i$ en la dirección $u_i$. Un $\sigma_i$ chico = esa dirección "casi se aplasta".
- **Número de condición:** $\kappa(A) = \sigma_1/\sigma_n$. Grande = matriz mal condicionada (hay direcciones casi planas).
- **Rango numérico:** cuántos $\sigma_i$ son "no despreciables".

**En el proyecto.** Aplicamos SVD a la **matriz de sensibilidad** $J$ (§2). Los $v_i$ con $\sigma_i$ chico son **combinaciones de parámetros poco identificables**. El $\sigma_{10}$ (el más chico, dominado por **wII**) es el "valle" del paisaje. `κ≈1.2e3` = moderadamente mal condicionado.

---

## 2. Sensibilidad y el Jacobiano $\partial y/\partial\theta$

**Definición.** Si el modelo predice una trayectoria $y(\theta)=[I(t),E(t)]$ apilada, la **matriz de sensibilidad** es
$$J = \frac{\partial y}{\partial \theta} \in \mathbb{R}^{N_\text{out}\times N_\text{par}},$$
donde $J_{kj}$ dice cuánto cambia la salida $k$ si movés el parámetro $j$. Se evalúa en $\theta$ verdadero.

**Sensibilidad relativa (log).** Escalamos cada columna por $\theta_j$: $\tilde J_{:,j} = \theta_j\,\partial y/\partial\theta_j$. Así comparamos **cambios fraccionales** (adimensional), equivalente a derivar respecto de $\log\theta_j$. Necesario porque los parámetros tienen escalas distintas (wII≈1.2 vs thetae≈2.8).

**Intuición.** Una columna con norma chica = ese parámetro **casi no afecta** la salida → difícil de estimar. Pero ¡ojo! norma de columna grande **no** garantiza identificabilidad si esa sensibilidad está **correlacionada** con la de otro parámetro (§4).

**En el proyecto.** $J$ se calcula por **autograd en modo forward** (`jacfwd`, §9). Es el insumo de Fisher y de la subset selection.

---

## 3. Matriz de Información de Fisher (FIM)

**Definición.** Para el modelo $y = f(\theta) + \varepsilon$ con ruido $\varepsilon\sim\mathcal N(0,\sigma^2 I)$, la log-verosimilitud es $\ell(\theta) = -\frac{1}{2\sigma^2}\lVert y - f(\theta)\rVert^2 + \text{cte}$. La FIM mide la **curvatura esperada** de $\ell$:
$$\mathcal I(\theta) = \mathbb E\!\left[-\frac{\partial^2\ell}{\partial\theta\,\partial\theta^\top}\right] = \frac{J^\top J}{\sigma^2}.$$

**Interpretación.**
- FIM grande en una dirección = la log-verosimilitud es **muy curva** ahí = los datos son **muy informativos** sobre esa combinación de parámetros = pico agudo = bien identificable.
- FIM chica (autovalor chico) = **valle plano** = muchos $\theta$ dan casi la misma verosimilitud = mal identificable.
- Sus autovalores son $\sigma_i(J)^2/\sigma^2$ → **la SVD de $J$ ES el análisis de la FIM** (por eso decimos "Fisher + SVD").

**En el proyecto.** Trabajamos con la FIM **relativa** ($\tilde J^\top\tilde J$). Su SVD nos da el espectro (§1) y las direcciones débiles. Método tomado de **Plate et al. 2024** (OED para UDEs).

---

## 4. Cota de Cramér-Rao (CRB)

**Teorema.** Para cualquier estimador **insesgado** $\hat\theta$,
$$\operatorname{Cov}(\hat\theta) \succeq \mathcal I(\theta)^{-1}.$$
En particular, la varianza mínima del parámetro $j$ es $\operatorname{Var}(\hat\theta_j)\ge (\mathcal I^{-1})_{jj}$, y su **desvío estándar mínimo** $\ge \sqrt{(\mathcal I^{-1})_{jj}}$.

**Lo crucial — por qué la diagonal de la INVERSA.** Hay dos medidas que se confunden:
- $\mathcal I_{jj}$ (o $\lVert$columna$_j\rVert$): sensibilidad **marginal**, ignora correlaciones.
- $(\mathcal I^{-1})_{jj}$: incertidumbre **real** de $\hat\theta_j$ teniendo en cuenta que se estima **junto con los demás**. Si $\theta_j$ está correlacionado con otro, $(\mathcal I^{-1})_{jj}$ se dispara aunque $\mathcal I_{jj}$ sea grande.

**En el proyecto (Exp C).** La métrica ingenua $\lVert$columna$\rVert$ decía "box es el mejor estímulo para wII" (señal grande). La **CRB** = $\sqrt{\operatorname{diag}(\text{FIM}^{-1})}$ lo corrige: **chirp** es el mejor (menor CRB), porque box da señal grande pero **correlacionada** (E≈I). Moraleja: identificabilidad = sensibilidad **decorrelacionada**, no amplitud.

---

## 5. Identificabilidad: estructural vs práctica

- **Estructural:** ¿se puede recuperar $\theta$ con datos **perfectos, sin ruido e infinitamente ricos**? Es propiedad de la *estructura* del modelo (p. ej. dos parámetros que solo aparecen como producto **nunca** se separan).
- **Práctica:** ¿se puede recuperar con los datos **reales** (finitos, con ruido)? Es lo que miden la FIM/CRB y el **profile likelihood**.

> **wII es estructuralmente identificable** (lo logramos con multi-trayectoria), pero **prácticamente mal identificable bajo ruido** (CRB alta, se rompe al 41 % a σ=0.10).

**Profile likelihood (Raue et al. 2009).** Se fija $\theta_j$ en distintos valores, se re-optimiza el resto, y se traza $\ell$ vs $\theta_j$. Un perfil **plano** = no identificable; uno con **valle marcado** = identificable. La regularización (§7) es una versión "suave" de esto.

---

## 6. Selección de subconjunto identificable (parameter subset selection)

**Idea.** Si liberar todos los parámetros es mal condicionado, **fijá** los peor identificables y estimá solo el subconjunto bien condicionado. ¿Cuáles fijar? Los que dominan las direcciones débiles.

**Método: QR con pivoteo de columnas** (Golub & Van Loan; Chu & Hahn 2007). Se factoriza $\tilde J\,P = QR$ con permutación $P$ que ordena las columnas de **más a menos** independientes (pivotes $|R_{ii}|$ decrecientes). La **cola** = candidatos a fijar. Es equivalente a una selección greedy por ortogonalización.

**En el proyecto (Exp B).** QR-pivoteo dio `thetae > … > ae > wII` → **wII** último, cola {wEI, ae, wII}. Empíricamente: fijar **wII o ai** ≈ óptimo; fijar **ae** no sirve (está en otra dirección débil); **fijar de más (wII+ai) es contraproducente**. → Regla: fijar un representante **mínimo** de cada acople débil.

---

## 7. Regularización: Tikhonov / ridge / MAP

**Definición.** En vez de fijar, penalizar la desviación de un prior $\theta_0$:
$$\hat\theta = \arg\min_\theta \; \lVert y - f(\theta)\rVert^2 + \lambda\,\lVert \theta - \theta_0\rVert^2.$$

**Interpretaciones equivalentes:**
- **Ridge / Tikhonov:** suma $\lambda I$ a la FIM → la vuelve invertible / mejor condicionada.
- **MAP bayesiano:** es el máximo a posteriori con **prior gaussiano** $\theta\sim\mathcal N(\theta_0,\tau^2 I)$, con $\lambda=\sigma^2/\tau^2$.
- **Bias-variance:** agrega **sesgo** (hacia $\theta_0$) a cambio de **menos varianza** → conviene justo en las direcciones mal condicionadas.

**Límites:** $\lambda=0$ = mínimos cuadrados (10 libres); $\lambda\to\infty$ = **fijar** $\theta$ en $\theta_0$ (= §6). O sea **la regularización interpola entre "todo libre" y "fijar"**.

**En el proyecto (Exp A2).** Regularizar wII: el error del resto baja monótono con λ y satura en el valor de "fijar"; **λ óptimo ≈ 1.0**. Confirma la equivalencia λ→∞ = fijar.

---

## 8. Diseño óptimo de experimentos (OED) e input design

**Idea central.** La FIM **depende del estímulo** $u(t)$. Distintas entradas hacen identificables distintas direcciones → se puede **diseñar** el estímulo para maximizar la información.

**Criterios de optimalidad** (funcionales escalares de la FIM):
- **A-optimal:** minimizar $\operatorname{tr}(\mathcal I^{-1})$ = varianza promedio (suma de CRB²).
- **D-optimal:** maximizar $\det(\mathcal I)$ = mínimo volumen del elipsoide de confianza.
- **E-optimal:** maximizar $\lambda_\min(\mathcal I)$ = achicar la **peor** dirección (la más débil).

**Excitación persistente (Ljung 1999, cap. 13).** Para que la FIM sea de rango completo, la entrada debe ser "suficientemente rica" (excitar suficientes frecuencias/estados). Señales **broadband** (chirp, PRBS, ruido) son persistentemente excitantes; un escalón casi-DC no.

**En el proyecto (Exp C).** Broadband (chirp/poisson/prbs) >> box/square por decorrelación. La **mezcla** de estímulos complementarios mejora la identificabilidad conjunta a ruido moderado; a ruido extremo el cuello de botella (wII) necesita **peso extra** en su dirección (Q-alta) → OED ponderado, no cobertura uniforme.

---

## 9. Piezas de ML / dinámica (contexto del método)

- **Autograd (diferenciación automática).** Modo *reverse* = gradientes (barato con muchas salidas→1 pérdida). Modo *forward* (`jacfwd`) = **Jacobiano** $\partial y/\partial\theta$ (eficiente cuando #params ≤ #salidas). Es como calculamos $J$ para la FIM **sin diferencias finitas** (exacto). Ver Neural ODEs.
- **Neural ODE.** El modelo aprende la **dinámica** $\dot x = f_\theta(x,u)$ (la *regla*), no la trayectoria. Se integra para obtener la solución y se backpropaga por el integrador. Contraste con la PINN en PINN vs Neural ODE.
- **Multiple shooting** (Bock & Plitt). Partir la trayectoria en ventanas cortas, cada una arrancando del estado **observado**, para evitar gradientes mal condicionados en horizontes largos. Es lo que estabiliza el entrenamiento del Neural ODE.
- **Reparametrización con softplus.** $\theta = \text{softplus}(\text{raw})>0$ para forzar positividad sin restricciones duras. `ke,ki` se **derivan** de las ganancias/umbrales para preservar el equilibrio E=I=0.
- **RK4.** Integrador de paso fijo, diferenciable, usado tanto para entrenar como para el lazo de control.

---

## 10. Cómo se conecta todo (glue)

```
u(t) (estímulo, §8)  →  trayectoria y(θ)  →  J = ∂y/∂θ (§2, autograd §9)
        │                                          │
        │                                     FIM = JᵀJ/σ² (§3)  →  SVD (§1)
        │                                          │
        └──────────── OED (§8) ────────────  direcciones débiles / CRB (§4)
                                                   │
                             identificabilidad práctica (§5)
                                                   │
                        ┌──────────────────────────┼──────────────────────────┐
                   fijar subset (§6)        regularizar (§7)          rediseñar u (§8)
                     [Exp B]                    [Exp A]                   [Exp C]
```

Todo el trabajo de esta etapa (A/B/C) es **aplicar §6-§8 usando el diagnóstico §1-§5**. La FIM predice, sin entrenar, qué fijar, qué regularizar y qué estímulo usar — y los experimentos lo confirman.

---

## 11. Qué leer (recursos)

- **Álgebra lineal / SVD:** Strang, *Linear Algebra*; Golub & Van Loan, *Matrix Computations* (SVD, QR con pivoteo).
- **Identificación de sistemas / OED / PE:** Ljung, *System Identification: Theory for the User* (1999), cap. 13.
- **FIM + SVD para modelos dinámicos / UDEs:** Plate et al. 2024 (OED for UDEs) — el método que usamos.
- **Subset selection:** Chu & Hahn 2007 (*Parameter set selection...*, AIChE J.).
- **Identificabilidad práctica / profile likelihood:** Raue et al. 2009 (Bioinformatics).
- **OED clásico (A/D/E-optimal):** Franceschini & Macchietto 2008 (Chem. Eng. Sci.).
- **Cramér-Rao / estimación:** cualquier texto de estadística (Casella & Berger) o teoría de estimación (Kay).
- **Multiple shooting:** Bock & Plitt 1984.

## 🔗 Conexiones
- Resultados que aplican esta teoría: Resultado - Subset selection, regularización y diseño de estímulo · Resultado - Robustez al ruido (10 params)
- Explicación visual (analogías 1→2→10 variables): Identificabilidad (Fisher + SVD) - explicación visual
- Métodos ML: Neural ODEs · PINN vs Neural ODE
- Proyecto: Investigación Neurociencia · Plan: Plan de trabajo - Neural ODE
- Bibliografía anotada: Bibliografía del proyecto - Wilson-Cowan ID
