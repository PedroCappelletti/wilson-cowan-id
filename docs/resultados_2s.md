---
type: note
status: inbox
project: "[[Investigación Neurociencia]]"
fecha: 2026-10-27
tags: [wilson-cowan, gray-box, resultados, semillas, warm-start]
---

# Resultados con datos de 2 s

Tanda `e16`: las correcciones de la tabla del póster entrenadas con trayectorias de 2 s muestreadas a 2 kHz, desde cero y desde el white-box (warm start), con ruido σ = 0,01 y 0,05.
- **Configuraciones clave, con tres semillas:** white-box, las agnósticas de 5 y 20 ms y la forma exacta en refractariedad; white-box, historia de 5 ms, filtro de 8 canales y estado de filtro en el actuador.
- **El resto, con la semilla 0:** la agnóstica del actuador, la historia de 20 ms y los filtros de 4 y 16 canales.

En total, 94 modelos. Cada semilla cambia la realización del ruido y la inicialización de la red.

Relacionado: [[Dataset ampliado - figuras]] (los mismos modelos con datos de 200 ms) · [[Controladores - resultados]] (el control con estos modelos) · [[Especificaciones técnicas - pipeline gray-box]].

---

## Lo principal

1. **La variación entre semillas es chica,** así que las diferencias que veníamos viendo con una semilla son reales. El desvío es como mucho 0,24 puntos de NRMSE, 2,3 puntos de error de β y 0,13 de R². La excepción es el escalón de extrapolación, que varía hasta 13 puntos.
2. **En refractariedad, el warm start hace que la corrección agnóstica identifique β:** de 18–22 % a 3,5–4 % con σ = 0,01, y de 17–18 % a 5–7 % con σ = 0,05. También sube su R² de entre −0,04 y 0,42 a entre 0,74 y 0,78. Es el resultado del póster de la Jornada, ahora con tres semillas.
3. **La forma exacta sigue recuperando mejor el término faltante** (R² de 0,97), pero en β la agnóstica con warm start queda a la par.
4. **En el actuador, el estado de filtro es el mejor en todo:** reproducción, identificación (6,7 a 14,5 %) y física recuperada (R² de 0,96).
5. **Las correcciones entrenables del actuador reproducen mejor con warm start, pero no identifican.** La mejora de reproducción, de 0,6 a 1,4 puntos con σ = 0,01, supera con margen la variación entre semillas. El R² pasa de negativo a levemente positivo. El error de β queda entre 23 y 37 %, y con σ = 0,05 el warm start lo empeora.
6. **Las elecciones de arquitectura no cambian:** los filtros de 4, 8 y 16 canales dan lo mismo, la historia de 5 ms le gana a la de 20 ms, y las ventanas de 5 y 20 ms en refractariedad dan lo mismo.
7. **La agnóstica g(I, E) en el actuador no aporta:** queda al nivel del white-box.

---

## σ = 0,01

![[dos-s-n01.png]]

| planta | configuración | arranque | NRMSE | error de β | R² |
|---|---|---|---|---|---|
| refractariedad | white-box | — | 3,29 ± 0,01 | 10,5 ± 0,1 | — |
| | agnóstica 5 ms | desde cero | 2,28 ± 0,02 | 18,3 ± 0,3 | 0,42 ± 0,02 |
| | | warm start | **2,13 ± 0,00** | **3,5 ± 0,3** | 0,78 ± 0,01 |
| | agnóstica 20 ms | desde cero | 2,38 ± 0,07 | 21,7 ± 2,0 | −0,04 ± 0,10 |
| | | warm start | 2,14 ± 0,00 | 4,0 ± 0,3 | 0,74 ± 0,01 |
| | forma exacta | warm start | 2,14 ± 0,00 | **2,8 ± 0,1** | **0,97 ± 0,00** |
| actuador | white-box | — | 7,26 ± 0,04 | 25,2 ± 0,3 | — |
| | agnóstica | desde cero | 7,15 | 26,8 | −0,46 |
| | | warm start | 7,04 | 20,2 | 0,03 |
| | historia 5 ms | desde cero | 4,52 ± 0,21 | 25,2 ± 2,3 | −0,02 ± 0,02 |
| | | warm start | 3,91 ± 0,03 | 24,3 ± 1,2 | 0,23 ± 0,05 |
| | historia 20 ms | desde cero | 5,63 | 31,9 | 0,11 |
| | | warm start | 4,51 | 26,6 | 0,26 |
| | filtro, 4 canales | desde cero | 5,60 | 26,4 | −0,26 |
| | | warm start | 4,18 | 28,1 | 0,08 |
| | filtro, 8 canales | desde cero | 4,93 ± 0,24 | 26,9 ± 0,5 | −0,37 ± 0,09 |
| | | warm start | 4,18 ± 0,03 | 26,7 ± 1,4 | 0,12 ± 0,05 |
| | filtro, 16 canales | desde cero | 4,87 | 25,6 | −0,37 |
| | | warm start | 4,16 | 24,2 | −0,07 |
| | estado de filtro | desde cero | **2,51 ± 0,00** | **6,7 ± 0,3** | **0,96 ± 0,00** |
| | | warm start | 2,52 ± 0,01 | 8,7 ± 1,3 | 0,96 ± 0,00 |

## σ = 0,05

![[dos-s-n05.png]]

| planta | configuración | arranque | NRMSE | error de β | R² |
|---|---|---|---|---|---|
| refractariedad | white-box | — | 5,91 ± 0,05 | 5,4 ± 0,4 | — |
| | agnóstica 5 ms | desde cero | 5,70 ± 0,02 | 17,0 ± 1,7 | 0,23 ± 0,09 |
| | | warm start | 5,66 ± 0,03 | 6,9 ± 0,7 | 0,74 ± 0,03 |
| | agnóstica 20 ms | desde cero | 5,51 ± 0,05 | 18,0 ± 1,5 | 0,02 ± 0,10 |
| | | warm start | **5,48 ± 0,05** | **4,9 ± 1,1** | 0,75 ± 0,03 |
| | forma exacta | warm start | 5,70 ± 0,04 | 7,6 ± 1,5 | **0,94 ± 0,01** |
| actuador | white-box | — | 7,93 ± 0,05 | 20,8 ± 0,8 | — |
| | agnóstica | desde cero | 8,17 | 24,2 | −0,36 |
| | | warm start | 7,96 | 25,6 | −0,05 |
| | historia 5 ms | desde cero | 6,62 ± 0,16 | 25,3 ± 0,4 | 0,02 ± 0,12 |
| | | warm start | 6,50 ± 0,08 | 29,4 ± 0,4 | 0,39 ± 0,07 |
| | historia 20 ms | desde cero | 7,08 | 18,2 | 0,22 |
| | | warm start | 6,83 | 37,2 | 0,28 |
| | filtro, 4 canales | desde cero | 7,11 | 26,6 | −0,14 |
| | | warm start | 6,70 | 36,8 | 0,15 |
| | filtro, 8 canales | desde cero | 6,88 ± 0,18 | 25,4 ± 0,9 | −0,27 ± 0,13 |
| | | warm start | 6,81 ± 0,14 | 33,2 ± 0,4 | 0,23 ± 0,08 |
| | filtro, 16 canales | desde cero | 6,81 | 23,1 | −0,22 |
| | | warm start | 6,63 | 33,7 | 0,03 |
| | estado de filtro | desde cero | 6,25 ± 0,04 | **12,6 ± 0,1** | **0,97 ± 0,00** |
| | | warm start | **6,09 ± 0,03** | 14,5 ± 0,9 | 0,96 ± 0,00 |

Valores en %, salvo el R². Media ± desvío sobre tres semillas; sin ±, una sola semilla. NRMSE sobre los 13 escenarios de test que interpolan.

---

## Contra los datos de 200 ms

El NRMSE no se puede comparar entre las dos tandas: las trayectorias tienen otra duración y otro contenido. El error de β y el R² sí, y dan las mismas conclusiones con valores parecidos:

| σ = 0,01 | 200 ms (`e13`) | 2 s (`e16`) |
|---|---|---|
| agnóstica 5 ms, warm start: β / R² | 5,6 % / 0,81 | 3,5 % / 0,78 |
| agnóstica 20 ms, warm start: β / R² | 3,9 % / 0,72 | 4,0 % / 0,74 |
| forma exacta: β / R² | 3,5 % / 0,97 | 2,8 % / 0,97 |
| estado de filtro, desde cero: β / R² | 7,5 % / 0,93 | 6,7 % / 0,96 |

---

## Detalles a tener en cuenta

- **El escalón de extrapolación sigue siendo el peor escenario para todos.** `box_a1.2` da entre 20 y 35 % de NRMSE, y es el que más varía entre semillas.
- **El estado de filtro es el más caro de entrenar:** entre 3,5 y 7,5 horas por corrida, contra 30 a 60 minutos de las demás.
- **El ruido es el equivalente al de antes:** σ por muestra de 0,0032 y 0,016, escalados para que la potencia de ruido dentro de la banda de la señal sea la misma que con 20 kHz.

Tabla completa, incluida la extrapolación y la validación: `wilson-cowan-id/results/resumen_2s.md`. Figuras: `results/figures/ampliado/dos_s_*` (`scripts/figura_2s.py`).
