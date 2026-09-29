| planta | configuración | póster | sin ruido | σ = 0,01 | σ = 0,05 |
|---|---|---|---|---|---|
| `refrac1` | *modelo perfecto (piso del NRMSE)* |  | *0,00* | *2,37* | *6,19* |
| `refrac1` | white-box | sí | 5,93 / -0,27 / 11,5 | 7,21 / -0,27 / 10,7 | 10,09 / -0,27 / 17,2 |
| `refrac1` | corrección agnóstica, ventana 400 | sí | 2,84 / -0,49 / 25,6 | 4,80 / -0,46 / 26,4 | 9,40 / -0,66 / 26,7 |
| `refrac1` | forma exacta | sí | 1,70 / +0,96 / 2,4 | 4,68 / +0,96 / 2,0 | 9,87 / +0,87 / 17,7 |
| `refrac1` | forma exacta y red |  | 1,82 / +0,95 / 3,6 | 5,20 / +0,97 / 1,0 | 10,00 / -1,77 / 17,9 |
| `act1` | *modelo perfecto (piso del NRMSE)* |  | *0,00* | *2,64* | *6,41* |
| `act1` | white-box | sí | 15,23 / -0,01 / 30,7 | 15,66 / -0,01 / 29,3 | 14,08 / -0,01 / 27,4 |
| `act1` | corrección agnóstica | sí | 15,40 / -1,87 / 34,4 | 15,15 / -0,73 / 28,1 | 14,76 / -1,05 / 34,3 |
| `act1` | estado de filtro | sí | 2,96 / +0,94 / 5,1 | 5,10 / +0,94 / 3,2 | 11,90 / +0,71 / 22,1 |
| `act1` | comando actual (A) |  | 15,18 / -0,71 / 32,9 | no se repite | no se repite |
| `act1` | historia cruda, 100 |  | 12,92 / +0,25 / 26,3 | 7,39 / +0,23 / 26,5 | 12,57 / -0,02 / 26,3 |
| `act1` | historia cruda, 400 |  | 11,48 / +0,15 / 22,7 | 10,78 / +0,13 / 22,1 | 14,13 / -0,11 / 25,8 |
| `act1` | FIR, 4 canales |  | 10,16 / -0,64 / 16,7 | 11,01 / -0,64 / 17,1 | 12,69 / -1,44 / 21,0 |
| `act1` | FIR, 8 canales |  | 9,12 / +0,13 / 21,4 | 9,81 / +0,10 / 26,6 | 12,36 / +0,11 / 23,6 |
| `act1` | FIR, 16 canales |  | 11,72 / -0,15 / 22,7 | 11,29 / -0,14 / 22,5 | 15,36 / -0,37 / 26,7 |
| `act1` | FIR, 800 retardos |  | 12,48 / -0,40 / 16,4 | no se repite | no se repite |
| `act1` | FIR, red de ancho 64 |  | 10,98 / -0,86 / 25,7 | no se repite | no se repite |
| `act1` | FIR, L2 1e-4 |  | 12,40 / +0,15 / 22,9 † | 22,24 / -4,20 / 114,9 † | 12,94 / -1,39 / 22,5 |
| `act1` | FIR, L2 1e-3 |  | 16,88 / -0,40 / 38,6 † | 6,45 / -0,90 / 28,3 | 11,16 / -1,20 / 24,7 |
| `act1` | latent ODE, ventana 100 |  | 14,92 / -0,51 / 38,9 | no se repite | no se repite |
| `act1` | latent ODE, ventana 400 |  | 13,28 / -3,55 / 52,6 | no se repite | no se repite |

*Cada celda: NRMSE (%) / R² de la corrección / error de parámetros (%). El piso es el NRMSE que sacaría la trayectoria verdadera contra los datos con ruido suavizados: lo mejor posible en ese nivel. Con ruido, media móvil de 7 muestras sobre I y E. El NRMSE se compara solo dentro de una columna; el error de parámetros también entre columnas. Una semilla por celda. † El entrenamiento divergió al final: la pérdida terminó más de 1,5 veces por encima de su mínimo y el resultado es el de un modelo ya degradado, así que no mide la configuración.*
