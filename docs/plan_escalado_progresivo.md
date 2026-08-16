# Plan: escalar la complejidad de a poco

> **Cambio de objetivo.** Hasta ahora la métrica reina era el error de parámetros.
> A partir de acá lo que importa es **copiar la dinámica del simulador** (NRMSE de
> rollout completo sobre estímulos no vistos). La identificación paramétrica pasa
> a segundo plano.

---

## Por qué reordenamos (diagnóstico en 3 líneas)

Metimos las dos perturbaciones juntas y el resultado fue mediocre para predecir:
la red mejoró el rollout apenas 6% (14.0% → 13.1% NRMSE). La causa está medida:
el 67% del desajuste lo imita el backbone moviendo parámetros, y el 33% restante
es el **retardo del actuador, que tiene memoria** — inalcanzable para una
corrección `g(I,E)` (techo R² = −0.11). Saltamos directo al caso más difícil.

**La lección de F7:** física sin memoria → capturable (refractariedad: R² 0.97);
física con memoria → hay que cambiar la arquitectura, no entrenar más.

---

## El principio del plan

Una sola perturbación a la vez, y no avanzar de etapa hasta cumplir el criterio
de éxito de la anterior. En cada etapa: primero la red "a ciegas", después (si
hace falta) darle información o estructura.

**Métricas en toda etapa** (siempre las mismas, para comparar entre etapas):

1. **NRMSE de rollout** (200 ms, estímulos de test) — la principal.
2. R² de `g` contra el `Δf` verdadero — ¿aprendió la física o tapó parámetros?
3. Error de parámetros — secundaria, sólo para vigilar el trade-off.

---

## Etapa 0 — Referencia limpia (ya la tenemos)

Planta sin perturbación, white-box. Rollout **2.0%**. Es el piso: ningún modelo
va a hacer mejor que esto. Verificar que sigue dando eso con el pipeline actual.

## Etapa 1 — Sólo refractariedad (la capturable)

Planta = WC + `(1−r·x)·S(u)` con `r = 0.10` (ε=1 sólo en refractariedad; actuador
apagado). Es el caso donde el gray-box **tiene que** funcionar: techo R² = 0.97.
Si acá no anda, el problema es de entrenamiento, no estructural — y es la etapa
donde vale la pena ajustar perillas.

**1a. Gray-box ciego:** `ẋ = f_WC(x,P,Q;θ) + g_φ(I,E)`. Perillas a barrer, en
orden de sospecha:

- **Largo de ventana de entrenamiento** (hoy 5 ms; probar 10/20/40 y curriculum
  corto→largo). Entrenar con ventanas cortas optimiza derivadas locales, no
  rollout — sospechoso principal del gap train/test.
- **Loss de rollout**: agregar un término de trayectoria larga (o fine-tune final
  con ventanas largas / multiple shooting).
- **Balance backbone/red**: congelar θ en los valores del white-box limpio vs.
  co-entrenar; penalización de redundancia (λ de la variante D, ya vimos que hay
  un óptimo intermedio).
- Tamaño de red, normalización de entradas, dt del integrador.

**1b. Con ayuda (sólo si 1a se queda corto):** darle a la red información sin
regalarle la respuesta, en escala creciente:

1. entrada extra `S(ae·u_e), S(ai·u_i)` (la sigmoidea evaluada, que la red no
   tiene por qué redescubrir);
2. forma estructural `g = −ρ_x · x · S(u)` con `ρ` aprendible (la corrección
   física S, que en F3 recuperó 1.3% cuando la forma era exacta — acá lo es).

**Criterio de éxito:** rollout ≤ ~4% (cerca del piso de 2%) y R² de `g` vs `Δf`
> 0.9. Si se cumple, tenemos la receta de entrenamiento validada.

## Etapa 2 — Sólo actuador (la que tiene memoria)

Planta = WC + lag/saturación (refractariedad apagada). Acá `g(I,E)` no puede por
diseño; el objetivo es elegir **cómo darle memoria** a la corrección:

- **2a. Estructural (la apuesta):** agregar al modelo el estado del filtro,
  `dP̂/dt = (P − P̂)/τ̂`, con `τ̂` (y quizá la saturación) aprendibles. Es la forma
  física exacta → debería clavarlo, y de paso valida el mecanismo de estados
  aumentados.
- **2b. Aprendida:** estado latente genérico — Neural ODE aumentada
  (`ż = h_ψ(z, x, P, Q)`, `g = g_φ(x, z)`) o corrección recurrente (GRU). Es lo
  que generaliza a física desconocida.

**Criterio de éxito:** 2a con rollout ≤ ~4%; 2b acercándose (≤ 6-8%) — la
comparación 2a vs 2b mide cuánto cuesta no conocer la forma.

## Etapa 3 — Las dos juntas, con la arquitectura ganadora

Recién acá volver al caso completo (ε=1 en ambas), con la receta de la etapa 1 y
la memoria de la etapa 2. Comparar contra el 13.1% que ya tenemos: ése es el
número a batir.

## Etapa 4 (rama paralela) — Full black-box

Si el gray-box no mejora el rollout de forma convincente (o directamente en
paralelo, porque es barato), probar sin backbone físico:

- **4a.** `ẋ = NN(x, P, Q)` — Neural ODE pura, mismos datos y métricas.
- **4b.** Versión con estados latentes aumentados (para la memoria del actuador).

Sirve como cota: si la black-box copia la dinámica mejor que el gray-box, el
backbone WC está **estorbando** para predecir (aunque siga siendo mejor para
interpretar). Si da peor, el prior físico paga. Cualquiera de los dos resultados
es informativo y publicable.

---

## Orden de trabajo propuesto

1. Etapa 1a (barrido de ventana/loss de rollout — es lo que más chances tiene de
   mover la aguja para *todo* lo demás).
2. Etapa 1b sólo si hace falta.
3. Etapa 2a → 2b.
4. Etapa 3.
5. Etapa 4 en paralelo desde que la etapa 1 esté cerrada (reusa el mismo
   pipeline de datos y evaluación).

**Infraestructura mínima previa:** las perturbaciones ya existen como clases
separadas en `uncertainty.py` (`Refractoriness`, `Actuator`), así que sólo hace
falta generar los datasets de cada etapa con la perturbación aislada, y un
script de evaluación único que reporte las 3 métricas para cualquier modelo.
