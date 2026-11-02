Simulador sin perturbación: RMSE 0,0382 en la theta-gamma y 0,0324 en la gamma.

**Con refractariedad**

| planta | σ | RMSE theta-gamma | RMSE gamma |
|---|---|---|---|
| simulador | | 0,0386 | 0,0339 |
| white-box | 0,01 | 0,0401 ± 0,0000 | 0,0348 ± 0,0001 |
|  | 0,05 | 0,0388 ± 0,0001 | 0,0315 ± 0,0002 |
| agnóstica | 0,01 | 0,0388 ± 0,0000 | 0,0335 ± 0,0001 |
|  | 0,05 | 0,0384 ± 0,0001 | 0,0327 ± 0,0001 |
| forma exacta | 0,01 | 0,0388 ± 0,0000 | 0,0341 ± 0,0001 |
|  | 0,05 | 0,0383 ± 0,0006 | 0,0313 ± 0,0008 |

**Con actuador**

| planta | σ | RMSE theta-gamma | RMSE gamma |
|---|---|---|---|
| simulador | | 0,0599 | 0,0692 |
| white-box | 0,01 | 0,0450 ± 0,0001 | 0,0387 ± 0,0003 |
|  | 0,05 | 0,0430 ± 0,0001 | 0,0324 ± 0,0004 |
| historia 5 ms | 0,01 | 0,1223 ± 0,0120 | 0,0536 ± 0,0014 |
|  | 0,05 | 0,1336 ± 0,0158 | 0,0488 ± 0,0017 |
| estado de filtro | 0,01 | 0,0556 ± 0,0003 | 0,0599 ± 0,0010 |
|  | 0,05 | 0,0545 ± 0,0001 | 0,0607 ± 0,0008 |

**Rango del comando**

| perturbación | planta | $P$, theta-gamma | $Q$, theta-gamma | $P$, gamma | $Q$, gamma |
|---|---|---|---|---|---|
| ninguna | simulador | −12,9 a 12,4 | −15,5 a 15,5 | 0,6 a 12,4 | −13,7 a 6,2 |
| refractariedad | simulador | −12,9 a 12,4 | −15,5 a 15,5 | 0,8 a 12,4 | −13,7 a 6,2 |
|  | white-box | −12,9 a 12,4 | −15,5 a 15,5 | 0,7 a 12,4 | −13,8 a 6,2 |
|  | agnóstica | −12,9 a 12,4 | −15,5 a 15,5 | 0,7 a 12,4 | −13,7 a 6,2 |
|  | forma exacta | −12,9 a 12,4 | −15,5 a 15,5 | 0,7 a 12,4 | −13,7 a 6,2 |
| actuador | simulador | −13,7 a 13,1 | −16,4 a 15,5 | −13,4 a 12,9 | −16,4 a 15,5 |
|  | white-box | −13,0 a 12,4 | −15,6 a 15,5 | −0,2 a 12,4 | −14,1 a 6,2 |
|  | historia 5 ms | −15,1 a 13,2 | −16,9 a 16,1 | −13,5 a 12,4 | −15,0 a 6,2 |
|  | estado de filtro | −14,7 a 13,1 | −16,4 a 15,5 | −13,6 a 12,9 | −16,0 a 15,5 |
