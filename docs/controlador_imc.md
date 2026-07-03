# Controlador IMC — cómo funciona y por qué es robusto al error paramétrico

> **Para qué esta nota:** entender *en detalle* el controlador que usamos y **por qué el control sigue andando aunque los parámetros identificados (θ̂) tengan error** — la clave del resultado OE3. Complementa (no repite) el resumen de resultados de Control en lazo cerrado (IMC): acá está la **mecánica de control** y el **argumento de robustez**. Código: `src/neural_ode/closed_loop.py`.

---

## 1 · Qué controla y qué es la "planta"
- **Objetivo de control:** que las poblaciones **I(t)** y **E(t)** de Wilson-Cowan **sigan una referencia** rI(t), rE(t) (p. ej. un ritmo theta-gamma) manipulando los **estímulos externos P (a E) y Q (a I)**.
- **Planta = lo que el controlador maneja.** En el código es un callable `plant_rhs(I,E,P,Q)->(dI,dE)` que puede ser:
  - la **WC verdadera** (`make_true_plant`), o
  - el **Neural ODE aprendido** (`make_neural_plant`).
  - Controlador y planta están **desacoplados** → se enchufa el modelo aprendido sin tocar el controlador. Esto es lo que permite la matriz 2×2 de OE3.

## 2 · Arquitectura del controlador (dos capas)
El IMC es un **IMC con linealización por realimentación** (*feedback linearization*). Tiene dos partes con roles muy distintos — y esta separación **es** el motivo de la robustez.

### (a) Capa lineal PI — realimentación con acción integral · **NO usa θ**
Estados integradores del controlador (uno por lazo):

$$\dot Z_1 = r_I - I, \qquad \dot Z_2 = r_E - E$$

Salidas PI (proporcional + integral del error):

$$U^{lti}_I = k_{pI}(r_I - I) + k_{iI}Z_1, \qquad U^{lti}_E = k_{pE}(r_E - E) + k_{iE}Z_2$$

(en el código `kp_I=10, ki_I=5, kp_E=5, ki_E=5`). **Nada de esto depende de los parámetros del modelo**: es puro error medido `r − y` y su integral.

### (b) Capa no lineal — sigmoidea inversa + cancelación · **SÍ usa θ**
Se satura `U^{lti}` al dominio válido, se pasa por la **sigmoidea inversa** para obtener `up, uq`, y se **cancela el acoplamiento** de WC con los pesos:

$$Q = u_q - (w_{IE}\,E - w_{II}\,I), \qquad P = u_p - (w_{EE}\,E - w_{EI}\,I)$$

Idea: WC tiene el término de acoplamiento `wIE·E − wII·I` dentro de la sigmoidea; si le **restás** ese término vía el estímulo, la dinámica en lazo cerrado queda **lineal y desacoplada** → los lazos PI la controlan trivialmente. Usa `wEE,wEI,wIE,wII` y la sigmoidea (`ae,ai,θe,θi,ke,ki`). **No usa te, ti** (las constantes de tiempo no entran en la cancelación).

> Integración: RK4 sobre el estado aumentado `[Z1, Z2, I, E]` (`simulate_closed_loop`).

---

## 3 · Por qué es robusto al error paramétrico (el argumento central)
Si el controlador se construye con **θ̂ ≠ θ** (parámetros identificados con error), la cancelación es **imperfecta**: queda un residuo

$$\Delta Q = (\hat w_{IE}-w_{IE})E - (\hat w_{II}-w_{II})I \quad (\text{análogo para }\Delta P)$$

Ese residuo entra al lazo como una **perturbación de carga** (una señal aditiva no cancelada). Y acá está la clave:

**La acción integral rechaza perturbaciones y errores de modelo constantes/lentos, sin conocer θ.**
- El integrador `Z` sigue acumulando mientras exista error `r − y ≠ 0`. En régimen permanente fuerza `error → 0` **cualquiera sea** el valor exacto de los pesos, siempre que **el lazo permanezca estable** (principio del modelo interno: un integrador en el lazo → error de seguimiento nulo ante referencias/perturbaciones de tipo escalón).
- Por eso el **control es mucho más robusto que la identificación**: la identificación necesita el valor *correcto* de wII; el control solo necesita que el lazo no se desestabilice. La cancelación imperfecta degrada el **transitorio**, no el **régimen permanente**.

**En una frase:** *el error de θ̂ se convierte en una perturbación acotada que el integrador absorbe; θ̂ solo afecta la calidad del feedforward, no la capacidad de seguir la referencia.*

### La salvedad (cuándo SÍ fallaría)
- Si θ̂ está **tan** errado que la cancelación deja una dinámica que **desestabiliza el lazo** (polos a la derecha), la acción integral ya no salva nada → el control falla. La robustez es **local/condicional a la estabilidad**, no infinita.
- También asume **realimentación de estado completo** (I, E medidos directos, sin filtro). El código replica el original sin EKF; con ruido de medición fuerte, un observador/EKF sería la capa extra (lo menciona el paper del tutor).

---

## 4 · La evidencia empírica (lo que confirma el argumento)
Detalle numérico en Control en lazo cerrado (IMC). Resumen:

- **Matriz 2×2** {simulador, Neural ODE} × {θ̂, real}: RMSE de seguimiento **casi idéntico** en toda referencia → usar θ̂ **no degrada** el control y la planta aprendida se comporta como el simulador.
- **Bajo ruido:** aunque θ̂ ingenuo llegue a **106% de error a σ=0.10**, el control RMSE se mantiene **~3.3e-2** (nivel ideal) en todos los niveles de ruido. **La fragilidad de la identificación no se propaga al control.**
- Esto es lo que vuelve *interesante* el hallazgo de identificabilidad: **wII es casi irrecuperable bajo ruido, pero para controlar no hace falta recuperarlo bien.**

---

## 5 · Conexión con el resto del proyecto
- Es la **validación orientada al control (OE3)**: cierra el lazo identificar → diseñar → controlar de Investigación Neurociencia.
- Da vuelta la lectura pesimista de Resultado - Robustez al ruido (10 params) (wII 41%): sí, wII es un cuello de botella para *identificar*, pero **para el objetivo final (controlar) no es fatal**.
- **Deriva paramétrica (descartada):** se consideró probar el control cuando la planta cambia en el tiempo, pero el tutor confirmó que el modelo es **tiempo-invariante** → el caso *time-varying* queda **fuera de alcance** (complejidad desproporcionada). El caso relevante es el error de estimación **estático**, ya validado (la acción integral lo absorbe).

## 🔗 Conexiones
- Resumen de resultados de control: Control en lazo cerrado (IMC)
- Proyecto: Investigación Neurociencia · Plan: Plan de trabajo - Neural ODE
- Diagrama del lazo: Diagramas - Pipelines Wilson-Cowan (diagrama 4)
- Modelo controlado: Modelo Wilson-Cowan · Identificación: PINN vs Neural ODE
- Guion donde se usa: Guion del informe y la presentación (sección 7)
- Paper relacionado (control en hardware): Su 2026 - SoC para neuromodulación closed-loop

## 📚 Fuentes
- Código: `wilson-cowan-id/src/neural_ode/closed_loop.py` (port fiel del `.m` del tutor); scripts `eval_closed_loop.py`, `noise_final.py`.
- `wilson-cowan-id/docs/informe_completo.md` §7–§8.
