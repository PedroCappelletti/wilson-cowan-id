# Tanda e16 (datos de 2 s): resultados

Media ± desvío sobre semillas (n = 3; n = 1 donde no hay ±). NRMSE sobre los 13 escenarios de test que interpolan; extrapolación = box_a1.2.

## σ = 0,01

| planta | configuración | arranque | n | NRMSE | extrap. | error de β | R² | validación |
|---|---|---|---|---|---|---|---|---|
| Refractariedad | white-box | desde cero | 3 | 3.29 ± 0.01 | 23.97 ± 0.07 | 10.46 ± 0.07 | -0.17 ± 0.00 | 3.06 ± 0.01 |
| Refractariedad | agnóstica, ventana 5 ms | desde cero | 3 | 2.28 ± 0.02 | 25.33 ± 0.88 | 18.32 ± 0.27 | 0.42 ± 0.02 | 1.84 ± 0.03 |
| Refractariedad | agnóstica, ventana 5 ms | warm start | 3 | 2.13 ± 0.00 | 28.33 ± 0.49 | 3.50 ± 0.25 | 0.78 ± 0.01 | 1.70 ± 0.02 |
| Refractariedad | agnóstica, ventana 20 ms | desde cero | 3 | 2.38 ± 0.07 | 26.96 ± 1.43 | 21.71 ± 2.03 | -0.04 ± 0.10 | 1.95 ± 0.06 |
| Refractariedad | agnóstica, ventana 20 ms | warm start | 3 | 2.14 ± 0.00 | 26.90 ± 1.40 | 3.96 ± 0.31 | 0.74 ± 0.01 | 1.71 ± 0.02 |
| Refractariedad | forma exacta | warm start | 3 | 2.14 ± 0.00 | 10.33 ± 6.45 | 2.83 ± 0.07 | 0.97 ± 0.00 | 1.72 ± 0.02 |
| Actuador | white-box | desde cero | 3 | 7.26 ± 0.04 | 32.43 ± 0.05 | 25.15 ± 0.25 | -0.00 ± 0.00 | 6.57 ± 0.03 |
| Actuador | agnóstica | desde cero | 1 | 7.15 | 32.77 | 26.84 | -0.46 | 6.16 |
| Actuador | agnóstica | warm start | 1 | 7.04 | 32.59 | 20.16 | 0.03 | 6.10 |
| Actuador | historia, 5 ms | desde cero | 3 | 4.52 ± 0.21 | 32.47 ± 0.08 | 25.19 ± 2.26 | -0.02 ± 0.02 | 3.69 ± 0.22 |
| Actuador | historia, 5 ms | warm start | 3 | 3.91 ± 0.03 | 22.24 ± 13.11 | 24.30 ± 1.21 | 0.23 ± 0.05 | 3.15 ± 0.08 |
| Actuador | historia, 20 ms | desde cero | 1 | 5.63 | 33.35 | 31.86 | 0.11 | 4.72 |
| Actuador | historia, 20 ms | warm start | 1 | 4.51 | 33.06 | 26.64 | 0.26 | 3.86 |
| Actuador | filtro, 4 canales | desde cero | 1 | 5.60 | 30.92 | 26.39 | -0.26 | 4.98 |
| Actuador | filtro, 4 canales | warm start | 1 | 4.18 | 31.74 | 28.11 | 0.08 | 3.47 |
| Actuador | filtro, 8 canales | desde cero | 3 | 4.93 ± 0.24 | 32.36 ± 1.77 | 26.86 ± 0.51 | -0.37 ± 0.09 | 4.14 ± 0.10 |
| Actuador | filtro, 8 canales | warm start | 3 | 4.18 ± 0.03 | 31.95 ± 0.47 | 26.72 ± 1.41 | 0.12 ± 0.05 | 3.48 ± 0.04 |
| Actuador | filtro, 16 canales | desde cero | 1 | 4.87 | 34.68 | 25.62 | -0.37 | 4.25 |
| Actuador | filtro, 16 canales | warm start | 1 | 4.16 | 31.15 | 24.23 | -0.07 | 3.59 |
| Actuador | estado de filtro | desde cero | 3 | 2.51 ± 0.00 | 27.79 ± 3.39 | 6.66 ± 0.28 | 0.96 ± 0.00 | 1.95 ± 0.01 |
| Actuador | estado de filtro | warm start | 3 | 2.52 ± 0.01 | 32.14 ± 2.85 | 8.65 ± 1.33 | 0.96 ± 0.00 | 1.93 ± 0.01 |

## σ = 0,05

| planta | configuración | arranque | n | NRMSE | extrap. | error de β | R² | validación |
|---|---|---|---|---|---|---|---|---|
| Refractariedad | white-box | desde cero | 3 | 5.91 ± 0.05 | 20.44 ± 0.14 | 5.42 ± 0.36 | -0.17 ± 0.00 | 5.19 ± 0.01 |
| Refractariedad | agnóstica, ventana 5 ms | desde cero | 3 | 5.70 ± 0.02 | 22.27 ± 0.08 | 17.01 ± 1.70 | 0.23 ± 0.09 | 4.87 ± 0.01 |
| Refractariedad | agnóstica, ventana 5 ms | warm start | 3 | 5.66 ± 0.03 | 22.40 ± 0.16 | 6.93 ± 0.65 | 0.74 ± 0.03 | 4.82 ± 0.01 |
| Refractariedad | agnóstica, ventana 20 ms | desde cero | 3 | 5.51 ± 0.05 | 23.99 ± 0.88 | 18.02 ± 1.47 | 0.02 ± 0.10 | 4.64 ± 0.00 |
| Refractariedad | agnóstica, ventana 20 ms | warm start | 3 | 5.48 ± 0.05 | 24.03 ± 1.16 | 4.89 ± 1.13 | 0.75 ± 0.03 | 4.61 ± 0.01 |
| Refractariedad | forma exacta | warm start | 3 | 5.70 ± 0.04 | 22.67 ± 0.73 | 7.60 ± 1.52 | 0.94 ± 0.01 | 4.87 ± 0.01 |
| Actuador | white-box | desde cero | 3 | 7.93 ± 0.05 | 29.54 ± 0.14 | 20.77 ± 0.83 | -0.00 ± 0.00 | 6.99 ± 0.03 |
| Actuador | agnóstica | desde cero | 1 | 8.17 | 29.76 | 24.20 | -0.36 | 7.09 |
| Actuador | agnóstica | warm start | 1 | 7.96 | 29.16 | 25.59 | -0.05 | 6.86 |
| Actuador | historia, 5 ms | desde cero | 3 | 6.62 ± 0.16 | 29.12 ± 0.25 | 25.32 ± 0.41 | 0.02 ± 0.12 | 5.46 ± 0.11 |
| Actuador | historia, 5 ms | warm start | 3 | 6.50 ± 0.08 | 28.81 ± 0.18 | 29.40 ± 0.37 | 0.39 ± 0.07 | 5.30 ± 0.05 |
| Actuador | historia, 20 ms | desde cero | 1 | 7.08 | 27.41 | 18.20 | 0.22 | 5.69 |
| Actuador | historia, 20 ms | warm start | 1 | 6.83 | 28.34 | 37.22 | 0.28 | 5.47 |
| Actuador | filtro, 4 canales | desde cero | 1 | 7.11 | 27.95 | 26.58 | -0.14 | 5.85 |
| Actuador | filtro, 4 canales | warm start | 1 | 6.70 | 28.74 | 36.84 | 0.15 | 5.49 |
| Actuador | filtro, 8 canales | desde cero | 3 | 6.88 ± 0.18 | 28.08 ± 0.58 | 25.38 ± 0.89 | -0.27 ± 0.13 | 5.65 ± 0.11 |
| Actuador | filtro, 8 canales | warm start | 3 | 6.81 ± 0.14 | 27.91 ± 0.22 | 33.24 ± 0.39 | 0.23 ± 0.08 | 5.48 ± 0.13 |
| Actuador | filtro, 16 canales | desde cero | 1 | 6.81 | 27.76 | 23.05 | -0.22 | 5.66 |
| Actuador | filtro, 16 canales | warm start | 1 | 6.63 | 27.82 | 33.71 | 0.03 | 5.37 |
| Actuador | estado de filtro | desde cero | 3 | 6.25 ± 0.04 | 28.00 ± 0.11 | 12.56 ± 0.06 | 0.97 ± 0.00 | 5.23 ± 0.05 |
| Actuador | estado de filtro | warm start | 3 | 6.09 ± 0.03 | 27.98 ± 0.19 | 14.47 ± 0.94 | 0.96 ± 0.00 | 5.10 ± 0.04 |

