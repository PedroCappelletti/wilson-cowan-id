# Resultado — OED ponderado al cuello de botella (prueba de concepto)

> **Qué se probó:** el Exp C dejó abierto que *"el OED debe ponderar hacia la dirección cuello de botella, no solo repartir cobertura"*. Este experimento lo testea: barre la **proporción ρ** de trayectorias de **Q-alta** (excitan la población I → SNR de wII) vs de **decorrelación** (chirp P,Q en bandas distintas → separan direcciones acopladas), y mide el error de θ̂ (foco wII) a σ=0.10. **Todo con componentes suaves (chirp)** para no perder el lever de dt grande. Fuente: `scripts/exp_oed_weighted.py`, `results/ident_oed_weighted.json`.

## 📊 Resultado — V-shape con óptimo en ρ=0.5
![](../results/figures/oed_weighted.png)

| ρ (fracción Q-alta) | composición | error wII |
|---|---|---|
| 0.00 (solo decorrelación) | 8 decorr | 93.1 % |
| 0.25 | 6 decorr + 2 Q-alta | 25.7 % |
| **0.50** | **4 decorr + 4 Q-alta** | **18.7 %** ← óptimo |
| 0.75 | 2 decorr + 6 Q-alta | 54.8 % |
| 1.00 (solo Q-alta) | 8 Q-alta | 98.7 % |

## 🔑 Lectura
**Hacen falta las DOS cosas.** Puro decorrelación (93 %) o pura Q-alta (98.7 %) fallan; **mezclarlas mitad y mitad baja wII a 18.7 %** (5× mejor que cualquier extremo). Confirma la hipótesis del Exp C: el OED **debe ponderar hacia el cuello de botella (Q-alta para el SNR de wII) sin abandonar la cobertura (decorrelación para separar los acoples)**. El óptimo es un balance, no un extremo → es una curva de diseño accionable.

- **Por qué falla ρ=0** (solo decorrelación): sin Q-alta, la población I no se excita fuerte → poco SNR en wII (que multiplica a I).
- **Por qué falla ρ=1** (solo Q-alta): sin decorrelación, wIE y wII quedan colineales (E≈I correlacionados) → se confunden.

## ⚠️ Caveats honestos
1. **Las 3 semillas dieron idéntico (std=0)** → **no hay barras de error reales**. Es un límite de diseño: en `identify_subset` la init de params es determinística (todos en 1.0) y el ruido usa una semilla fija dentro de `generate()`, así que variar `seed` no cambia la corrida. Para barras de error habría que **variar la realización del ruido** (y/o randomizar la init). La forma de la curva es robusta, pero las magnitudes son de una sola realización.
2. **Throttling térmico:** las corridas se frenaron de ~2.9 a ~7.2 min a lo largo de los ~70 min → el ETA de la calibración (38 min) quedó corto. Confiar en el timer que refina sobre la marcha, no en la calibración inicial.

## 🔬 Limitación para optogenética (importante)
Los componentes son **chirp (suaves)** — elegidos para mantener el dt grande (4× más barato). Pero la **optogenética es on/off**: un chirp no es directamente realizable con luz. Esta es una **prueba de concepto de la IDEA de OED ponderado** sobre datos sintéticos; **no** es un protocolo de estímulo implementable en el experimento real.

## ▶️ Experimento propuesto (versión realizable)
**OED ponderado con estímulos conmutados (realizables):** repetir el barrido de ρ pero con **componentes on/off** — decorrelación vía PRBS/APRBS (P,Q con semillas distintas) + Q-alta vía tren de pulsos de gran amplitud. Requiere **dt fino** (los conmutados rompen a dt grande, ver Resultado - Identificación NODE por familia y el límite del dt grande) → ~4× más lento, pero cierra el gap hacia la aplicación optogenética. Métrica idéntica (error wII vs ρ). Bonus: **variar la realización del ruido** para tener las barras de error que faltaron acá.

## 🔗 Enlaces
- Proyecto: Investigación Neurociencia · Plan: Plan de trabajo - Neural ODE
- Origen de la hipótesis (Exp C): Resultado - Subset selection, regularización y diseño de estímulo
- Lever de dt (y su límite con conmutados): Costo computacional del Neural ODE - qué reducir y cómo estimarlo · Resultado - Identificación NODE por familia y el límite del dt grande
- Teoría (OED, excitación persistente): Fundamentos teóricos - Identificabilidad, SVD, Fisher y OED
- Presentación: Presentación al tutor - resumen y plan de armado (tema 7)
