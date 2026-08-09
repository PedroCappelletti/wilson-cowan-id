# Banco de pruebas: incertidumbre dinámica para el gray-box

Evidencia numérica de `docs/incertidumbre_dinamica_graybox.md`. **No toca `src/`**: son
scripts exploratorios que reimplementan las perturbaciones candidatas por fuera del
simulador, para decidir cuáles vale la pena llevar a `src/wilson_cowan/`.

Integrador propio (RK4 de paso fijo con buffer de historia) porque `solve_ivp` no sirve
para retardos.

| Script | Qué contesta | Costo |
|---|---|---|
| `probe_uncertainty.py` | Cuánto deforma cada familia la dinámica (D_traj, \|Δf\|, régimen) | ~2 min |
| `probe_learnable.py` | **La pregunta clave**: ¿puede `g_φ(I,E,P,Q)` representar el Δf de cada familia? Entrena la arquitectura exacta de `dynamics.py` contra el Δf verdadero, con estímulos held-out | ~6 min |
| `probe_scaling.py` | Dónde se rompe la ausencia de memoria al barrer las constantes de tiempo | ~10 min |
| `probe_calib.py` | Intensidad recomendada del par refractariedad + actuador | ~5 min |

```bash
.venv/bin/python scripts/uncertainty_probe/probe_learnable.py
```

Las 9 familias implementadas están en `probe_uncertainty.py` (`FAMILIES`): refractariedad,
sigmoidea heterogénea, adaptación, depresión sináptica, población oculta, retardo axonal,
actuador optogenético, deriva de `wEE(t)` y ruido de proceso OU.

**La métrica que decide** es `R2_test` de `probe_learnable.py`. Está calibrada: las dos
familias que son *demostrablemente* funciones sin memoria del estado dan R²≈0.97, que es el
techo del estimador. Un R² muy por debajo de eso significa memoria oculta real, no ruido
de medición.
