# Log de approaches — PINN problema inverso Wilson-Cowan

Registro de qué falló en cada intento de identificación paramétrica
(wEE=6.4, wEI=4.8, wIE=6.0, wII=1.2 partiendo de inicio ignorante 1.0).

---

## Intento 1 — Adam conjunto, w_physics=1.0

**Resultado:** wEE≈1.7 (verdadero=6.4), error ~74%

**Por qué falló:** `L_data` domina sobre `L_physics`. La red memoriza la trayectoria sin mover `raw_w`. El gradiente físico es demasiado débil y Adam lo suprime frente al de datos.

**Cambio:** subir `w_physics` a 10.

---

## Intento 2 — Adam conjunto, w_physics=10.0

**Resultado:** datos=1.5e-2 (alto), fis=1.25e-4 (bajísimo), wEE≈1.4

**Por qué falló:** La red encontró un mínimo degenerado — satisface la física con una trayectoria INCORRECTA combinada con parámetros incorrectos. `L_data` ya no guía. Paisaje no-convexo con mínimos locales profundos donde la física se cumple mal pero la red "cree" que está bien.

**Cambio:** separar en dos etapas (primero ajustar trayectoria, luego identificar parámetros).

---

## Intento 3 — Dos etapas con Adam

**Etapa 1** (w_data=1, w_physics=0, 15k epochs): datos=7.95e-6 — trayectoria perfecta.
**Etapa 2** (MLP congelado, solo `raw_w`, w_physics=1, 20k epochs): wEE: 1.000 → 1.010. Fracaso.

**Por qué falló (raíz real):** `PINN.forward()` **no usa `raw_w`** — la red sólo mapea t → [I, E] usando el MLP y las Fourier features. En Etapa 1, la red aprende valores E_NN(t) ≈ E_true(t), pero sus derivadas autograd dE_NN/dt son las de la función interpolante aprendida, *no* las de la ODE física.

En Etapa 2, el residuo físico es:

```
residual_E = dE_NN/dt  −  rhs_E(I_NN, E_NN, wEE)
```

Si `dE_NN/dt` coincide accidentalmente con `rhs_E(wEE=1.0)` (parámetros incorrectos), el residuo es pequeño y el gradiente w.r.t. `raw_w` ≈ 0 aunque wEE sea completamente incorrecto. La red, al entrenar sin física en Etapa 1, puede encontrar una interpolación cuyas derivadas internas satisfagan la física en el punto de inicio equivocado.

**Conclusión:** separar etapas rompe el acoplamiento necesario entre la interpolación y la física. No se puede corregir después con Adam porque el gradiente es nulo.

**Cambio:** entrenamiento **conjunto** con L-BFGS.

---

## Intento 4 — L-BFGS conjunto (implementación actual)

**Idea:** optimizar MLP + `raw_w` *simultáneamente* con L-BFGS (full-batch, curvatura de segundo orden, line search de Wolfe fuerte).

**Por qué debería funcionar:**
- El MLP no puede "escapar" a una interpolación con derivadas físicamente incorrectas porque la física penaliza en cada paso de optimización.
- L-BFGS usa el Hessiano aproximado (BFGS), que maneja mejor el paisaje mal condicionado donde Adam se queda atascado.
- Full-batch (sin minibatch) asegura que el gradiente es determinista, requisito de L-BFGS.
- La literature de PINNs recomienda Adam para pre-entrenamiento y L-BFGS para convergencia fina; aquí vamos directo a L-BFGS con inicialización estándar.

**Hiperparametros:** LBFGS_STEPS=3000, max_iter=20 (internos), lr=0.5, strong_wolfe, N_COLLOC=10000 fijo.

---

## Nota sobre scipy.optimize / diferencias finitas

Usar diferencias finitas de la trayectoria + scipy.optimize (mínimos cuadrados no lineales) **no es una PINN**. La PINN requiere:
1. Red neuronal como aproximador de la solución.
2. Derivadas vía autograd a través de la red.
3. Física como término de la función de pérdida durante el entrenamiento.

Scipy/diferencias finitas es el método clásico de "derivative matching" (similar a SINDy), que el concurso pide implementar como **comparación** con la PINN, no como método principal.
