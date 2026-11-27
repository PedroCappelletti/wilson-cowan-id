### Referencia theta-gamma de 2 s

Ideal (Wilson-Cowan sin perturbación, β verdaderos): RMSE 0,0382 (I 0,0331, E 0,0426).

**Planta con refractariedad**

| modelo | uso en el controlador | σ = 0,01: error de β̂ | σ = 0,01: RMSE | σ = 0,05: error de β̂ | σ = 0,05: RMSE |
|---|---|---|---|---|---|
| β verdaderos | sin modelo de la perturbación | 0 % | 0,0386 | 0 % | 0,0386 |
| white-box | solo β̂ | 8,8 % | 0,0386 ± 0,0000 | 8,9 % | 0,0386 ± 0,0000 |
| con red | solo β̂ | 1,7 % | 0,0386 ± 0,0000 | 2,0 % | 0,0386 ± 0,0000 |
| con red | β̂ + g | 1,7 % | 0,0392 ± 0,0000 | 2,0 % | 0,0392 ± 0,0000 |
| forma exacta | β̂ + forma | 0,2 % | 0,0384 ± 0,0000 | 0,5 % | 0,0384 ± 0,0000 |

**Planta con actuador**

| modelo | uso en el controlador | σ = 0,01: error de β̂ | σ = 0,01: RMSE | σ = 0,05: error de β̂ | σ = 0,05: RMSE |
|---|---|---|---|---|---|
| β verdaderos | sin modelo de la perturbación | 0 % | 0,0570 | 0 % | 0,0570 |
| white-box | solo β̂ | 121,9 % | 0,0738 ± 0,0001 | 121,2 % | 0,0726 ± 0,0001 |
| con red | solo β̂ | 117,9 % | 0,0798 ± 0,0046 | 114,4 % | 0,0828 ± 0,0051 |
| forma exacta | solo β̂ | 0,1 % | 0,0583 ± 0,0000 | 0,7 % | 0,0584 ± 0,0020 |

### Referencia gamma continua

Ideal (Wilson-Cowan sin perturbación, β verdaderos): RMSE 0,0324 (I 0,0335, E 0,0313).

**Planta con refractariedad**

| modelo | uso en el controlador | σ = 0,01: error de β̂ | σ = 0,01: RMSE | σ = 0,05: error de β̂ | σ = 0,05: RMSE |
|---|---|---|---|---|---|
| β verdaderos | sin modelo de la perturbación | 0 % | 0,0339 | 0 % | 0,0339 |
| white-box | solo β̂ | 8,8 % | 0,0324 ± 0,0000 | 8,9 % | 0,0323 ± 0,0000 |
| con red | solo β̂ | 1,7 % | 0,0340 ± 0,0000 | 2,0 % | 0,0339 ± 0,0000 |
| con red | β̂ + g | 1,7 % | 0,0336 ± 0,0000 | 2,0 % | 0,0338 ± 0,0001 |
| forma exacta | β̂ + forma | 0,2 % | 0,0324 ± 0,0000 | 0,5 % | 0,0324 ± 0,0000 |

**Planta con actuador**

| modelo | uso en el controlador | σ = 0,01: error de β̂ | σ = 0,01: RMSE | σ = 0,05: error de β̂ | σ = 0,05: RMSE |
|---|---|---|---|---|---|
| β verdaderos | sin modelo de la perturbación | 0 % | 0,0676 | 0 % | 0,0676 |
| white-box | solo β̂ | 121,9 % | 0,0961 ± 0,0000 | 121,2 % | 0,0959 ± 0,0001 |
| con red | solo β̂ | 117,9 % | 0,0784 ± 0,0165 | 114,4 % | 0,0730 ± 0,0059 |
| forma exacta | solo β̂ | 0,1 % | 0,0665 ± 0,0012 | 0,7 % | 0,0619 ± 0,0065 |

