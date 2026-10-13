# Consistencia de β entre familias de estímulos (tanda e14)

Un white-box por familia, entrenado y validado solo con esa familia. «error» = promedio de |desvío relativo| sobre parámetros y familias; «dispersión» = desvío estándar entre familias, promediado sobre parámetros; «todas» = el white-box con las siete familias juntas. En %.

| planta | ruido | familias | error | dispersión | error de «todas» |
|---|---|---|---|---|---|
| sin perturbación (control) | σ = 0,01 | 7 | 3.9 | 4.8 | 0.5 |
| refractariedad | σ = 0,01 | 7 | 13.1 | 8.5 | 14.0 |
| actuador | σ = 0,01 | 7 | 34.2 | 23.6 | 22.7 |
| sin perturbación (control) | σ = 0,05 | 7 | 16.5 | 13.5 | 7.0 |
| refractariedad | σ = 0,05 | 7 | 16.0 | 17.0 | 6.1 |
| actuador | σ = 0,05 | 7 | 28.2 | 25.3 | 31.5 |
| sin perturbación (control) | sin ruido | 7 | 1.9 | 1.9 | 0.3 |
| refractariedad | sin ruido | 7 | 13.1 | 7.7 | 8.7 |
| actuador | sin ruido | 7 | 37.5 | 29.5 | 25.7 |
