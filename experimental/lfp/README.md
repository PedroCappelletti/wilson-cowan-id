# Piloto: observar S en vez de I y E

> **EN EVALUACIÓN.** Nada de esta carpeta forma parte del pipeline del escalado
> ni de ningún resultado reportado. Está por verse si se incluye. Los datos van
> a `data/processed/lfp_piloto/` y los resultados a `results/lfp_piloto/`, los
> dos separados de los del escalado.

## Qué pregunta

El escalado entrena y evalúa viendo $I$ y $E$ por separado, que en un registro
real no se miden. Lo que se mide es un potencial de campo local, una sola señal
que combina las dos poblaciones. El piloto deja el modelo exactamente igual
(mismas variantes, mismo arranque ignorante, misma selección por validación) y
cambia solo la observación: la pérdida compara $S = h(I, E, P, Q)$ contra el $S$
medido.

Lo que se quiere saber:

1. Cuánto se pierde de identificabilidad de $\beta$ al mirar una combinación en
   vez de dos variables.
2. Si el modelo reconstruye $I$ y $E$ por separado sin haberlos visto
   (`nrmse_IE_oculto_test`). En el simulador se puede medir; en datos reales, no.
3. Si la corrección sigue recuperando el término faltante ($R^2$ contra
   $\Delta f$) cuando solo ve $S$.

## Archivos

| archivo | qué hace |
|---|---|
| `observacion.py` | los proxies de $S$ (`PROXIES`), que funcionan con numpy y con torch |
| `ruido.py` | ruido $1/f^\chi$ con SNR fijada por trayectoria |
| `gen_lfp.py` | toma un dataset limpio del escalado, le agrega $S$ y ruido sobre $S$ |
| `entrenar_lfp.py` | entrena con $S$ solamente y evalúa |

## Los proxies de S

La literatura no tiene un observable único. Los que hay se dividen en dos
hipótesis físicas que difieren en el signo de la inhibición, y esa diferencia
no se puede esconder: es la decisión de modelado que el piloto tiene que poder
comparar.

| proxy | fórmula | hipótesis | fuente |
|---|---|---|---|
| `E_menos_I` | $E - I$ | potencial de membrana, con signo | decisión D3 de `train_real_output.py` |
| `entrada_E` | $w_{EE}E - w_{EI}I + P$ | despolarización neta de las piramidales | análogo de Jansen y Rit 1995 y Moran et al. 2007 |
| `suma_corrientes` | $\lvert w_{EE}E + P\rvert + \lvert w_{EI}I\rvert$ | suma de magnitudes de corrientes | Mazzoni et al. 2008 |
| `menos_E_mas_I` | $-(E + I)$ | suma de magnitudes, pesos iguales | Krishnakumaran, Raees y Ray 2022 |

Queda afuera el proxy de Mazzoni et al. 2015, con retardo de 6 ms y factor 1,65
sobre la corriente inhibitoria. Esos valores salen de una red de piramidales con
morfología fija, y en este simulador $\tau_e = 1$ ms: el retardo sería más
largo que una ventana de entrenamiento entera.

Referencias:
- Jansen BH, Rit VG (1995). Biol Cybern 73:357-366. Sin leer en el original.
- Moran RJ et al. (2007). NeuroImage 37:706-720. Salida $v_6 = v_2 - v_3$, verificada.
- Mazzoni A, Panzeri S, Logothetis NK, Brunel N (2008). PLoS Comput Biol 4:e1000239.
- Mazzoni A, Lindén H, Cuntz H, Lansner A, Panzeri S, Einevoll GT (2015). PLoS Comput Biol 11:e1004584.
- Krishnakumaran R, Raees M, Ray S (2022). PLoS Comput Biol 18:e1009886.
- Cowan JD, Neuman J, van Drongelen W (2016). J Math Neurosci 6:1. Identifican
  $E$ e $I$ con el LFP filtrado, sin dar una combinación.

No se encontró ningún trabajo que analice qué se pierde en identificabilidad al
observar una sola combinación de $E$ e $I$. El piloto lo mide directamente.

## Cómo se entrena sin ver el estado

Sin $I$ y $E$ observados, el estado al arranque de cada ventana no se conoce.
Pasa a ser una incógnita por ventana (`X0`), con una penalización de continuidad
contra el final de la ventana anterior (`--lam-cont`). Es el mismo esquema de
`scripts/train_real_output.py`, que ya lo usa para los datos reales. La primera
ventana de cada escenario arranca del reposo, que se conoce porque lo fija el
experimento. Los `X0` arrancan de la corrida libre del modelo recién
inicializado.

## Limitaciones conocidas

- Solo las variantes de `build_model` (whitebox, B, H, K, S). El estado de
  filtro y la latent ODE se entrenan con `fit_aug` y no están conectados.
- Sin L-BFGS: con los `X0` adentro son miles de incógnitas más.
- El proxy es fijo y conocido, con ganancia 1. Con datos reales la ganancia
  sería otra incógnita (`c_out` en `train_real_output.py`).

## Uso

```bash
python experimental/lfp/gen_lfp.py refrac1_amp E_menos_I --snr 20 --chi 1
python experimental/lfp/entrenar_lfp.py --data refrac1_amp__E_menos_I__snr20_chi1 --variant whitebox --tag piloto_wb
```

Probado de punta a punta el 3-10 con 4 épocas, solo para verificar que corre.
No hay ninguna corrida completa todavía.
