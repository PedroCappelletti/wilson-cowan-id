# Lo troncal del gray-box · Wilson–Cowan

Los archivos esenciales del proyecto, numerados en el orden en que conviene
leerlos. Entre corchetes va el nombre original, para ubicarlos en el repo.

> **Esta carpeta es para leer, no para correr.** No trae los `__init__.py` ni las
> dependencias, así que los `import` no van a resolver. La idea es entender el
> camino, no ejecutarlo.

## Qué se hizo, en un párrafo

El proyecto identifica los 10 parámetros de Wilson–Cowan con una Neural ODE, y
lo hacía con un error del 1 %. Pero el simulador que generaba los datos y el
modelo que se entrenaba **eran exactamente las mismas ecuaciones**: el problema
que se resolvía era *"dado que el modelo es perfecto, encontrale los
parámetros"*. Le rompimos esa igualdad a propósito —agregándole al **simulador**
física que el modelo no contempla— y encendimos un término de corrección
neuronal para que tape el hueco. Todo lo que sigue es esa cadena.

## El camino de los datos

```
una función P(t)                      ← archivo 1
        ↓
20 escenarios de (P, Q)               ← archivo 3
        ↓
integrar y muestrear → .npz           ← archivos 4 y 5
        ↓   (acá entra la perturbación del archivo 2)
ventanas + RK4 diferenciable          ← archivos 7 y 8
        ↓
10 parámetros + la corrección         ← archivo 6
```

## Los archivos

**`1 - simulador y estimulos [model.py].py`**
El simulador de Wilson-Cowan y la libreria de 9 estimulos. Un estimulo no es un array: es una FUNCION del tiempo P(t).

**`2 - las perturbaciones [uncertainty.py].py`**
La fisica que se le agrega al SIMULADOR y el modelo no contempla. Nueve familias; se usan refractariedad y actuador optogenetico.

**`3 - catalogo de 20 escenarios [gen_multi_dataset.py].py`**
Elige las 20 combinaciones concretas de (P, Q) y marca cuales quedan como test. Aca estan las amplitudes y frecuencias.

**`4 - de funcion a trayectoria [generate.py].py`**
El motor: integra el sistema, muestrea en una grilla y empaqueta el .npz. Aca se separa el estimulo COMANDADO del que realmente llega.

**`5 - datasets con perturbacion [gen_uncertain_dataset.py].py`**
Genera el barrido de epsilon (cuanta fisica falta) y guarda el Delta f verdadero, que es lo que la red deberia aprender.

**`6 - el modelo gray-box [dynamics.py].py`**
El modelo que se entrena: backbone de Wilson-Cowan (10 parametros interpretables) + la correccion neuronal g_phi.

**`7 - el integrador diferenciable [integrate.py].py`**
RK4 de paso fijo escrito en torch, para que el gradiente atraviese la integracion. El estimulo entra como zero-order hold.

**`8 - el entrenamiento [graybox_train.py].py`**
Multiple shooting, las 5 variantes de correccion y las metricas. El corazon del trabajo esta aca.

**`9 - correr un entrenamiento [exp_f3_graybox.py].py`**
El punto de entrada: elige la variante, entrena y guarda el resultado. Sirve como ejemplo de como se usa todo lo anterior.

## Las tres ideas que hay que sacar de la lectura

**1 · Un estímulo es una función, no un vector.** El integrador lo evalúa en
instantes arbitrarios y desordenados, así que `P(t)` tiene que ser una función
pura del tiempo. Por eso los estímulos aleatorios (APRBS, PRBS, Poisson)
pre-calculan toda su agenda al construirse: si no, el mismo `t` daría valores
distintos y el resultado no sería reproducible. → archivo 1

**2 · La perturbación va en el simulador, nunca en el modelo.** El simulador
pasa a representar "el cerebro real" con toda la física; el modelo es la
hipótesis incompleta. La red **nunca ve** la perturbación: sólo ve la actividad
(I, E) y el estímulo que se comandó (P, Q). Todo lo que no cierre con
Wilson–Cowan puro lo tiene que descubrir sola. → archivos 2 y 4

**3 · El estímulo comandado y el que llega no son el mismo.** Con el actuador
optogenético, lo que la neurona recibe llegó filtrado y saturado. El dataset
guarda los dos por separado, y el entrenamiento **sólo puede usar el
comandado** — usar el efectivo sería regalarle la respuesta. → archivo 4

## Si querés ver los resultados

No están en esta carpeta. En el repo:

- `docs/graybox_manual_completo.md` — el manual completo, con todo lo medido
- `docs/resumen graybox para compañeros.pdf` — el resumen con figuras
- `docs/presentacion_graybox.html` — la presentación

*Regenerar esta carpeta:* `python scripts/armar_paquete.py`
