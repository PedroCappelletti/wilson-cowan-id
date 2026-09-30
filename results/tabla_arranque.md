| planta | corrección | arranque | σ = 0,01 | σ = 0,05 |
|---|---|---|---|---|
| `refrac1` | agnóstica, ventana 20 ms | desde cero | 4,80 / 26,4 / -0,46 | 9,40 / 26,7 / -0,66 |
|  |  | desde el white-box | 4,29 / 5,5 / +0,79 | 9,29 / 9,5 / +0,46 |
|  |  | congelada | 4,49 / 10,7 / +0,26 | 9,19 / 17,2 / -0,01 |
| `refrac1` | agnóstica, ventana 5 ms | desde cero | … | … |
|  |  | desde el white-box | … | … |
| `act1` | agnóstica g(I, E) | desde cero | 15,15 / 28,1 / -0,73 | 14,76 / 34,3 / -1,05 |
|  |  | desde el white-box | … | … |
| `act1` | historia del comando, 100 | desde cero | 7,39 / 26,5 / +0,23 | 12,57 / 26,3 / -0,02 |
|  |  | desde el white-box | … | … |
| `act1` | historia del comando, 400 | desde cero | 10,78 / 22,1 / +0,13 | 14,13 / 25,8 / -0,11 |
|  |  | desde el white-box | … | … |
| `act1` | filtro entrenable, 4 canales | desde cero | 11,01 / 17,1 / -0,64 | 12,69 / 21,0 / -1,44 |
|  |  | desde el white-box | … | … |
| `act1` | filtro entrenable, 8 canales | desde cero | 9,81 / 26,6 / +0,10 | 12,36 / 23,6 / +0,11 |
|  |  | desde el white-box | 7,05 / 30,4 / -0,15 | 13,80 / 30,6 / -0,41 |
|  |  | congelada | 14,00 / 29,3 / +0,14 | 14,32 / 27,4 / -0,01 † |
| `act1` | filtro entrenable, 16 canales | desde cero | 11,29 / 22,5 / -0,14 | 15,36 / 26,7 / -0,37 |
|  |  | desde el white-box | … | … |
| `act1` | filtro, 4 canales, L2 1e-4 | desde cero | 22,24 / 114,9 / -4,20 † | 12,94 / 22,5 / -1,39 |
|  |  | desde el white-box | … | … |
| `act1` | filtro, 4 canales, L2 1e-3 | desde cero | 6,45 / 28,3 / -0,90 | 11,16 / 24,7 / -1,20 |
|  |  | desde el white-box | … | … |
| | *referencias sin red* | | | |
| `refrac1` | white-box | | 7,21 / 10,7 / — | 10,09 / 17,2 / — |
| `refrac1` | forma exacta (backbone expandido) | | 4,68 / 2,0 / +0,96 | 9,87 / 17,7 / +0,87 |
| `act1` | white-box | | 15,66 / 29,3 / — | 14,08 / 27,4 / — |
| `act1` | estado de filtro (backbone expandido) | | 5,10 / 3,2 / +0,94 | 11,90 / 22,1 / +0,71 |
