| planta | corrección | arranque | sin ruido | σ = 0,01 | σ = 0,05 |
|---|---|---|---|---|---|
| `refrac1` | agnóstica, ventana 20 ms | desde cero | 2,84 / 25,6 / -0,49 | 4,80 / 26,4 / -0,46 | 9,40 / 26,7 / -0,66 |
|  |  | desde el white-box | 2,18 / 5,1 / +0,82 † | 4,29 / 5,5 / +0,79 | 9,29 / 9,5 / +0,46 |
|  |  | congelada | 3,51 / 11,5 / +0,22 | 4,49 / 10,7 / +0,26 | 9,19 / 17,2 / -0,01 |
| `refrac1` | agnóstica, ventana 5 ms | desde cero | 2,99 / 21,3 / +0,02 † | 5,73 / 23,7 / +0,07 | 9,88 / 33,3 / -1,06 |
|  |  | desde el white-box | no se corrió | 5,41 / 4,2 / +0,76 | 10,00 / 18,8 / -1,79 |
| `act1` | agnóstica g(I, E) | desde cero | 15,40 / 34,4 / -1,87 | 15,15 / 28,1 / -0,73 | 14,76 / 34,3 / -1,05 |
|  |  | desde el white-box | no se corrió | 15,57 / 35,6 / -0,24 | 15,16 / 27,9 / -0,99 |
| `act1` | historia del comando, 100 | desde cero | 12,92 / 26,3 / +0,25 | 7,39 / 26,5 / +0,23 | 12,57 / 26,3 / -0,02 |
|  |  | desde el white-box | no se corrió | 11,13 / 36,0 / -0,06 | 12,71 / 30,1 / -0,13 |
| `act1` | historia del comando, 400 | desde cero | 11,48 / 22,7 / +0,15 | 10,78 / 22,1 / +0,13 | 14,13 / 25,8 / -0,11 |
|  |  | desde el white-box | no se corrió | 11,41 / 27,3 / -0,04 | 12,95 / 31,7 / -0,17 |
| `act1` | filtro entrenable, 4 canales | desde cero | 10,16 / 16,7 / -0,64 | 11,01 / 17,1 / -0,64 | 12,69 / 21,0 / -1,44 |
|  |  | desde el white-box | no se corrió | 10,12 / 31,4 / -0,27 | 14,05 / 30,4 / -0,51 |
| `act1` | filtro entrenable, 8 canales | desde cero | 9,12 / 21,4 / +0,13 | 9,81 / 26,6 / +0,10 | 12,36 / 23,6 / +0,11 |
|  |  | desde el white-box | 11,33 / 32,0 / -0,12 | 7,05 / 30,4 / -0,15 | 13,80 / 30,6 / -0,41 |
|  |  | congelada | 13,58 / 30,7 / +0,13 | 14,00 / 29,3 / +0,14 | 14,32 / 27,4 / -0,01 † |
| `act1` | filtro entrenable, 16 canales | desde cero | 11,72 / 22,7 / -0,15 | 11,29 / 22,5 / -0,14 | 15,36 / 26,7 / -0,37 |
|  |  | desde el white-box | no se corrió | 11,61 / 28,8 / -0,05 | 12,37 / 27,8 / -0,27 |
| `act1` | filtro, 4 canales, L2 1e-4 | desde cero | 12,40 / 22,9 / +0,15 † | 22,24 / 114,9 / -4,20 † | 12,94 / 22,5 / -1,39 |
|  |  | desde el white-box | no se corrió | 6,92 / 36,7 / -0,42 | 14,27 / 32,3 / -0,56 |
| `act1` | filtro, 4 canales, L2 1e-3 | desde cero | 16,88 / 38,6 / -0,40 † | 6,45 / 28,3 / -0,90 | 11,16 / 24,7 / -1,20 |
|  |  | desde el white-box | no se corrió | 17,02 / 54,2 / -1,69 † | 16,90 / 187,2 / -11,46 † |
| | *referencias sin red* | | | | |
| `refrac1` | white-box | | 5,93 / 11,5 / — | 7,21 / 10,7 / — | 10,09 / 17,2 / — |
| `refrac1` | forma exacta (backbone expandido) | | 1,70 / 2,4 / +0,96 | 4,68 / 2,0 / +0,96 | 9,87 / 17,7 / +0,87 |
| `act1` | white-box | | 15,23 / 30,7 / — | 15,66 / 29,3 / — | 14,08 / 27,4 / — |
| `act1` | estado de filtro (backbone expandido) | | 2,96 / 5,1 / +0,94 | 5,10 / 3,2 / +0,94 | 11,90 / 22,1 / +0,71 |
