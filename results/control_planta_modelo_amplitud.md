**Refractariedad, theta-gamma**

| conectado al controlador | σ | RMSE de seguimiento | distancia al lazo con el simulador |
|---|---|---|---|
| simulador | | 0,0386 | |
| white-box | 0,01 | 0,0391 ± 0,0000 | 0,0068 ± 0,0000 |
|  | 0,05 | 0,0390 ± 0,0000 | 0,0068 ± 0,0000 |
| agnóstica | 0,01 | 0,0388 ± 0,0000 | 0,0007 ± 0,0000 |
|  | 0,05 | 0,0387 ± 0,0000 | 0,0007 ± 0,0000 |
| backbone expandido | 0,01 | 0,0385 ± 0,0000 | 0,0002 ± 0,0000 |
|  | 0,05 | 0,0385 ± 0,0000 | 0,0003 ± 0,0000 |

**Refractariedad, gamma continua**

| conectado al controlador | σ | RMSE de seguimiento | distancia al lazo con el simulador |
|---|---|---|---|
| simulador | | 0,0339 | |
| white-box | 0,01 | 0,0348 ± 0,0000 | 0,0026 ± 0,0000 |
|  | 0,05 | 0,0347 ± 0,0000 | 0,0027 ± 0,0000 |
| agnóstica | 0,01 | 0,0344 ± 0,0000 | 0,0011 ± 0,0000 |
|  | 0,05 | 0,0342 ± 0,0001 | 0,0008 ± 0,0001 |
| backbone expandido | 0,01 | 0,0337 ± 0,0000 | 0,0002 ± 0,0000 |
|  | 0,05 | 0,0337 ± 0,0000 | 0,0003 ± 0,0000 |

**Actuador con recorte, theta-gamma**

| conectado al controlador | σ | RMSE de seguimiento | distancia al lazo con el simulador |
|---|---|---|---|
| simulador | | 0,0570 | |
| white-box | 0,01 | 0,0535 ± 0,0000 | 0,0365 ± 0,0000 |
|  | 0,05 | 0,0534 ± 0,0000 | 0,0365 ± 0,0000 |
| historia con comando actual | 0,01 | 0,0613 ± 0,0006 | 0,0412 ± 0,0018 |
|  | 0,05 | 0,0643 ± 0,0038 | 0,0439 ± 0,0048 |
| backbone expandido | 0,01 | 0,0574 ± 0,0000 | 0,0123 ± 0,0001 |
|  | 0,05 | 0,0572 ± 0,0001 | 0,0131 ± 0,0003 |

**Actuador con recorte, gamma continua**

| conectado al controlador | σ | RMSE de seguimiento | distancia al lazo con el simulador |
|---|---|---|---|
| simulador | | 0,0676 | |
| white-box | 0,01 | 0,0563 ± 0,0000 | 0,0676 ± 0,0000 |
|  | 0,05 | 0,0558 ± 0,0000 | 0,0674 ± 0,0000 |
| historia con comando actual | 0,01 | 0,0594 ± 0,0017 | 0,0728 ± 0,0034 |
|  | 0,05 | 0,0571 ± 0,0027 | 0,0680 ± 0,0024 |
| backbone expandido | 0,01 | 0,0668 ± 0,0001 | 0,0085 ± 0,0003 |
|  | 0,05 | 0,0653 ± 0,0002 | 0,0141 ± 0,0011 |

**Rango del comando** (percentiles 5 y 95; mínimo del 5 y máximo del 95 sobre los dos σ y las tres semillas)

| perturbación | conectado al controlador | $P$, theta-gamma | $Q$, theta-gamma | $P$, gamma continua | $Q$, gamma continua |
|---|---|---|---|---|---|
| refractariedad | simulador | −9,5 a 1,5 | −12,4 a 0,6 | 0,9 a 1,5 | −13,2 a 0,8 |
|  | white-box | −9,2 a 1,5 | −12,4 a 0,8 | 0,8 a 1,6 | −13,1 a 0,9 |
|  | agnóstica | −9,5 a 1,4 | −12,4 a 0,6 | 0,9 a 1,5 | −13,3 a 0,8 |
|  | backbone expandido | −9,5 a 1,5 | −12,4 a 0,6 | 0,9 a 1,5 | −13,2 a 0,8 |
| actuador con recorte | simulador | −9,5 a 5,2 | −12,7 a 2,7 | −11,7 a 11,0 | −13,7 a 11,3 |
|  | white-box | −9,4 a 3,1 | −12,6 a 1,7 | −2,0 a 9,0 | −13,6 a 2,4 |
|  | historia con comando actual | −10,1 a 5,0 | −12,7 a 3,9 | −10,2 a 9,7 | −14,0 a 11,8 |
|  | backbone expandido | −9,5 a 5,1 | −12,7 a 2,7 | −11,8 a 10,9 | −13,7 a 11,3 |
