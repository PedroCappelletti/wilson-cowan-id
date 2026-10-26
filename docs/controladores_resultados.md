---
type: note
status: inbox
project: "[[Investigación Neurociencia]]"
fecha: 2026-10-26
tags: [wilson-cowan, control, imc, resultados]
---

# Controladores: resultados

Controlador del paper (Martínez et al., 2022) contra el extendido, sobre todos los modelos, con dos referencias y tres modos del comando: 1848 simulaciones. Cómo está armado, en [[Controladores - explicación a fondo]].

Valores: RMSE de seguimiento, mediana sobre ruidos y semillas, modelos de 2 s con warm start salvo que se indique. Tope: P en [0, 2,0], Q en [0, 1,5].

---

## Lo principal

1. **El único modelo que mejora mucho el control es el estado de filtro, con el controlador extendido:** −52 % contra el oráculo con la gamma de 50 ms, y −28 % con la theta-gamma de 2 s.
2. **Ningún controlador pide comandos realizables.** Incluso el oráculo pide P entre −13 y +13, contra el rango de entrenamiento de 0 a 2. El extendido con estado de filtro llega a +73.
3. **Con el tope literal, todos se rompen en la planta con actuador** (0,40), por el windup del integrador. Con anti-windup vuelven a niveles normales.
4. **Con tope y anti-windup, la mejora del estado de filtro sobrevive en la gamma de 50 ms** (−44 % contra el oráculo), **pero se pierde en la theta-gamma de 2 s** (−4 %): el comando pasa la mayor parte del tiempo contra el tope y la autoridad del actuador manda.
5. **Historia y filtro entrenable no mejoran el control de forma consistente:** el extendido no encuentra un comando mejor en la mitad de los pasos.
6. **Cancelar la agnóstica empeora,** entre 50 y 70 % en el actuador sin tope.
7. **En refractariedad, todos quedan al nivel del oráculo:** la acción integral compensa sola.

![[control-actuador.png]]

---

## Planta con actuador

**Gamma de 50 ms**

| modelo | controlador | sin tope | tope literal | tope y anti-windup |
|---|---|---|---|---|
| oráculo | — | 0,069 | 0,398 | 0,073 |
| white-box | original | 0,090 | 0,397 | 0,065 |
| agnóstica | original, solo β̂ | 0,083 | 0,397 | 0,066 |
| agnóstica | original, cancelando g | 0,128 | 0,394 | 0,065 |
| historia | original, solo β̂ | 0,085 | 0,397 | 0,064 |
| historia | **extendido** | 0,055 | 0,398 | 0,077 |
| filtro | original, solo β̂ | 0,074 | 0,397 | 0,065 |
| filtro | **extendido** | 0,078 | 0,395 | 0,065 |
| estado de filtro | original, solo β̂ | 0,077 | 0,398 | 0,067 |
| estado de filtro | **extendido** | **0,033** | 0,396 | **0,041** |

**Theta-gamma de 2 s**

| modelo | controlador | sin tope | tope literal | tope y anti-windup |
|---|---|---|---|---|
| oráculo | — | 0,060 | 0,200 | 0,082 |
| white-box | original | 0,064 | 0,199 | 0,082 |
| agnóstica | original, solo β̂ | 0,067 | 0,200 | 0,082 |
| agnóstica | original, cancelando g | 0,080 | 0,194 | 0,082 |
| historia | original, solo β̂ | 0,071 | 0,199 | 0,082 |
| historia | **extendido** | 0,064 | 0,198 | 0,083 |
| filtro | original, solo β̂ | 0,067 | 0,199 | 0,082 |
| filtro | **extendido** | 0,066 | 0,199 | 0,082 |
| estado de filtro | original, solo β̂ | 0,071 | 0,200 | 0,082 |
| estado de filtro | **extendido** | **0,043** | 0,194 | 0,079 |

- **Desde cero da lo mismo que con warm start** en historia, filtro y estado de filtro. Con la agnóstica, cancelar g desde cero empeora más (0,136 contra 0,128 en la gamma).
- **Los modelos de 200 ms dan los mismos resultados**, con una diferencia: la historia con el extendido sale 0,064 en vez de 0,055 en la gamma.
- **Las fallas de inversión del extendido con historia y filtro** van del 40 al 60 % de los pasos.

---

## Planta con refractariedad

**Gamma de 50 ms**

| modelo | controlador | sin tope | tope literal | tope y anti-windup |
|---|---|---|---|---|
| oráculo | — | 0,0339 | 0,0422 | 0,0364 |
| oráculo con la forma | — | 0,0324 | 0,0408 | 0,0351 |
| white-box | original | 0,0336 | 0,0418 | 0,0361 |
| agnósticas | original, solo β̂ | 0,034 | 0,042 | 0,036 |
| agnósticas, desde cero | original, cancelando g | 0,044 a 0,045 | 0,045 | 0,042 |
| agnósticas, warm start | original, cancelando g | 0,034 | 0,041 | 0,0355 |
| forma exacta | original, con la forma | 0,0326 | 0,0408 | 0,0351 |

**Theta-gamma de 2 s:** todos los controladores dan alrededor de 0,039 sin tope, 0,059 con tope literal y 0,051 con anti-windup. La excepción es cancelar g de la agnóstica desde cero, con 0,043, 0,060 y 0,053.

- **El tope molesta poco en refractariedad.** En la gamma de 50 ms el comando nunca baja de 0,8, así que solo muerde el techo.
- **La forma exacta con warm start iguala al oráculo que conoce la forma**, y cancelar g con warm start queda al nivel del oráculo. Desde cero, cancelar g empeora un 30 %.

---

## Rango del comando sin tope

| controlador y modelo, planta con actuador | P, gamma 50 ms | P, theta-gamma 2 s |
|---|---|---|
| oráculo | −13 a +13 | −14 a +13 |
| original, β̂ de cualquier modelo | −15 a +17 | −16 a +18 |
| extendido, historia y filtro | −19 a +17 | −19 a +18 |
| extendido, estado de filtro | **−29 a +73** | **−37 a +73** |
| estímulos de entrenamiento | 0 a 2 | 0 a 2 |

En refractariedad, P va de 0,8 a 12 con la gamma de 50 ms y de −13 a +12 con la theta-gamma: ahí el comando negativo aparece al apagar cada ráfaga.

**Tiempo contra el tope, con anti-windup.** En el actuador, entre 61 y 94 % con la gamma de 50 ms, y entre 38 y 96 % con la theta-gamma. El estado de filtro con el extendido es el que menos tiempo pasa contra el tope: 61 y 38 %.

---

## Qué significa para el trabajo

- **Representar el actuador con su forma física sí sirve para controlar mejor:** es lo único que le gana claramente al oráculo. Las correcciones agnósticas ajustan las trayectorias, pero no dan un modelo invertible.
- **Con un comando realizable, la ventaja depende de la tarea.** Se mantiene en una gamma continua, y se pierde cuando la referencia pide apagar y encender rápido, porque sin luz negativa no se puede frenar la actividad.
- **Para seguir:**
  - rediseñar las ganancias del PI para la autoridad real del actuador;
  - probar un controlador que no invierta el modelo (MPC, IMC clásica);
  - medir con ruido y un estimador de estado, como en el paper.

Tablas completas (200 ms y 2 s, desde cero y warm start): `wilson-cowan-id/results/control_resumen.md`.
