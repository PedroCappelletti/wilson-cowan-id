### Referencia theta-gamma de 2 s

Ideal (Wilson-Cowan sin perturbación, θ verdaderos): RMSE 0,0382 (I 0,0331, E 0,0426).

**Planta con refractariedad**

| modelo | uso en el controlador | σ = 0,01: error de θ̂ | σ = 0,01: RMSE | σ = 0,05: error de θ̂ | σ = 0,05: RMSE |
|---|---|---|---|---|---|
| θ verdaderos | sin modelo de la perturbación | 0 % | 0,0386 | 0 % | 0,0386 |
| white-box | solo θ̂ | 10,5 % | 0,0388 ± 0,0000 | 5,4 % | 0,0386 ± 0,0000 |
| con red | solo θ̂ | 3,5 % | 0,0387 ± 0,0000 | 4,9 % | 0,0386 ± 0,0000 |
| con red | θ̂ + g | 3,5 % | 0,0395 ± 0,0000 | 4,9 % | 0,0393 ± 0,0001 |
| forma exacta | θ̂ + forma | 2,8 % | 0,0384 ± 0,0000 | 7,6 % | 0,0384 ± 0,0001 |

**Planta con actuador**

| modelo | uso en el controlador | σ = 0,01: error de θ̂ | σ = 0,01: RMSE | σ = 0,05: error de θ̂ | σ = 0,05: RMSE |
|---|---|---|---|---|---|
| θ verdaderos | sin modelo de la perturbación | 0 % | 0,0599 | 0 % | 0,0599 |
| white-box | solo θ̂ | 25,2 % | 0,0643 ± 0,0001 | 20,8 % | 0,0651 ± 0,0035 |
| con red | solo θ̂ | 24,3 % | 0,0680 ± 0,0045 | 29,4 % | 0,0713 ± 0,0003 |
| forma exacta | solo θ̂ | 6,7 % | 0,0711 ± 0,0002 | 12,6 % | 0,0711 ± 0,0007 |

### Referencia gamma de 50 ms

Ideal (Wilson-Cowan sin perturbación, θ verdaderos): RMSE 0,0324 (I 0,0335, E 0,0313).

**Planta con refractariedad**

| modelo | uso en el controlador | σ = 0,01: error de θ̂ | σ = 0,01: RMSE | σ = 0,05: error de θ̂ | σ = 0,05: RMSE |
|---|---|---|---|---|---|
| θ verdaderos | sin modelo de la perturbación | 0 % | 0,0339 | 0 % | 0,0339 |
| white-box | solo θ̂ | 10,5 % | 0,0339 ± 0,0000 | 5,4 % | 0,0332 ± 0,0001 |
| con red | solo θ̂ | 3,5 % | 0,0342 ± 0,0000 | 4,9 % | 0,0339 ± 0,0001 |
| con red | θ̂ + g | 3,5 % | 0,0340 ± 0,0001 | 4,9 % | 0,0336 ± 0,0004 |
| forma exacta | θ̂ + forma | 2,8 % | 0,0326 ± 0,0000 | 7,6 % | 0,0325 ± 0,0002 |

**Planta con actuador**

| modelo | uso en el controlador | σ = 0,01: error de θ̂ | σ = 0,01: RMSE | σ = 0,05: error de θ̂ | σ = 0,05: RMSE |
|---|---|---|---|---|---|
| θ verdaderos | sin modelo de la perturbación | 0 % | 0,0692 | 0 % | 0,0692 |
| white-box | solo θ̂ | 25,2 % | 0,0901 ± 0,0001 | 20,8 % | 0,0878 ± 0,0012 |
| con red | solo θ̂ | 24,3 % | 0,0828 ± 0,0068 | 29,4 % | 0,0771 ± 0,0088 |
| forma exacta | solo θ̂ | 6,7 % | 0,0788 ± 0,0038 | 12,6 % | 0,0762 ± 0,0005 |

