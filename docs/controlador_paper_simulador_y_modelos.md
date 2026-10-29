---
type: note
status: inbox
project: "[[Investigación Neurociencia]]"
fecha: 2026-10-28
tags: [wilson-cowan, control, imc, resultados]
pdf: "[[controlador-paper-simulador-y-modelos.pdf]]"
---

# Controlador de Martínez et al. (2022): pruebas sobre el simulador y los modelos gray-box

El controlador se usa tal cual está en el paper: IMC con linealización por realimentación, sin cota sobre el comando y sin anti-windup. Entre filas cambia una sola cosa, los parámetros con los que se construye y, cuando el despeje cerrado lo admite, la corrección que cancela. La arquitectura de los modelos está en [[Arquitectura de los modelos gray-box]].

## 1. Controlador

La planta nominal es

$$\tau_e \dot E = -E + S_e(w_{EE}E - w_{EI}I + P), \qquad \tau_i \dot I = -I + S_i(w_{IE}E - w_{II}I + Q),$$

con $S_x(v) = 1/(1+e^{-a_x(v-\theta_x)}) - 1/(1+e^{a_x\theta_x})$. El controlador impone a cada población la dinámica lineal $\tau_x \dot x = -x + U_x$, con

$$U_x = k_{p,x}(r_x - x) + k_{i,x} Z_x, \qquad \dot Z_x = r_x - x .$$

Para eso pide a la sigmoidea la salida $U_x + k_x$, con $k_x = 1/(1+e^{a_x\theta_x})$, la invierte y resta el acoplamiento:

$$Q = \theta_i - \frac{1}{a_i}\ln\!\left(\frac{1}{U_I + k_i} - 1\right) - (w_{IE}E - w_{II}I),$$

y lo mismo para $P$ con los parámetros de E. Las ganancias son las del paper (ec. 12) y del código original, $k_p = \tau/\lambda$ y $k_i = 1/\lambda$ con $\lambda = 0{,}2$ ms: $k_{p,I} = 10$, $k_{i,I} = 5$, $k_{p,E} = 5$, $k_{i,E} = 5$. Quedan fijas en todas las pruebas. $U_x$ se recorta al dominio de la sigmoidea inversa, calculado siempre con los parámetros verdaderos para que todos los controladores tengan el mismo rango de actuación.

**Qué toma el controlador de cada modelo.** Todo modelo identificado tiene dos partes: los diez parámetros físicos $\hat\beta$ del backbone de Wilson-Cowan y una corrección para lo que el backbone no explica. La corrección depende del modelo:

- en el white-box no hay corrección;
- en la agnóstica y en la historia es una red neuronal ($g(I, E)$ en la agnóstica; una red que además ve los últimos 5 ms del comando en la historia);
- en la forma exacta y en el estado de filtro son términos físicos agregados al backbone.

La columna «uso» de las tablas indica qué partes entran al controlador:

- **solo $\hat\beta$.** Los $a_x$, $\theta_x$, $w$ y $k_x$ de las ecuaciones de arriba son los del modelo, en lugar de los verdaderos. La corrección se descarta aunque el modelo la tenga: en la agnóstica y en la historia, la red no interviene en el control.
- **$\hat\beta$ y cancela $g$.** Además de los $\hat\beta$, el controlador usa la red agnóstica. Como $g(I, E)$ no depende del comando, se resta del objetivo y la sigmoidea pasa a pedir $U_x + k_x - \tau_x\, g_x(I, E)$.
- **$\hat\beta$ e invierte la forma.** El modelo de forma exacta tiene el factor $(1 - r_x x)$ sobre la sigmoidea y una saturación $\tanh(\alpha u)/\alpha$ en la entrada. El objetivo se divide por $(1 - \hat r_x x)$ y al final se invierte la saturación.

Las dos últimas solo existen en refractariedad. En el actuador, la red de la historia depende del comando y el estado de filtro agrega dos estados; en los dos casos el despeje del paper deja de ser cerrado, así que esos modelos se usan solo con sus $\hat\beta$.

**Simulación.** RK4 con paso de 0,005 ms; el controlador se recalcula en cada paso. Estado inicial $E = I = 0$ e integradores en cero. La medición no tiene ruido; el ruido está en los datos con que se identificó cada modelo.

## 2. Referencias y error

- **Gamma de 50 ms**, la de las ecuaciones 13 y 14 del paper con $T_w = 1$, como en su figura 6: $r_E = 0{,}3\sin(2\pi\, 120\, t) + 0{,}45$ y $r_I = 0{,}2\sin(2\pi\, 120\, t - 0{,}94) + 0{,}25$, con $t$ en segundos.
- **Theta-gamma de 2 s**, la de la figura 7. Es la misma gamma, encendida solo cuando la fase de una theta está a menos de $\pi/4$ de su valle; fuera de esa ventana, $r_E = r_I = 0$. La theta es ruido blanco filtrado entre 4 y 12 Hz con semilla fija, para que no sea una senoidal pura.

El error que se reporta es el RMSE de seguimiento de cada población, sin el primer quinto de la simulación (10 ms en la gamma, 400 ms en la theta-gamma):

$$\mathrm{RMSE}_x = \sqrt{\frac{1}{N - n_0}\sum_{k=n_0}^{N-1}(x_k - r_{x,k})^2}, \qquad n_0 = N/5, \qquad \mathrm{RMSE} = \sqrt{\frac{\mathrm{RMSE}_E^2 + \mathrm{RMSE}_I^2}{2}} .$$

Las tablas dan ese RMSE combinado, en las mismas unidades de actividad que E e I. La columna «error de $\hat\beta$» es el error relativo medio de los diez parámetros físicos ($w_{EE}$, $w_{EI}$, $w_{IE}$, $w_{II}$, $\tau_e$, $\tau_i$, $a_e$, $a_i$, $\theta_e$, $\theta_i$) respecto de los verdaderos.

## 3. Casos

| caso | planta | parámetros del controlador |
|------------|--------------------|------------------------------------|
| ideal | Wilson-Cowan sin perturbación | verdaderos |
| β verdaderos | perturbada | verdaderos; no sabe de la perturbación |
| white-box | perturbada | $\hat\beta$ del white-box |
| con red | perturbada | $\hat\beta$ del mejor modelo con red por validación; la red se usa o no según la columna «uso» |
| forma exacta | perturbada | $\hat\beta$ del backbone expandido |

Las perturbaciones son las del resto del trabajo: refractariedad con $r = 0{,}1$, y un actuador con filtro de primer orden ($\tau = 1$ ms) seguido de saturación $3\tanh(u/3)$. Los modelos son los de la tanda `e16` (datos de 2 s a 2 kHz), con $\sigma$ = 0,01 y 0,05 y tres semillas cada uno. Esos σ son el equivalente a 20 kHz; por muestra, el ruido de los datos a 2 kHz es de 0,0032 y 0,016. Por la regla de validación, la mejor corrección con red es:

- en refractariedad, la agnóstica $g(I, E)$ con warm start, ventana de 5 ms con $\sigma$ = 0,01 y de 20 ms con $\sigma$ = 0,05;
- en el actuador, la historia de 5 ms con warm start.

En la forma exacta, la refractariedad usa el factor $(1 - r x)$ con warm start; el actuador usa el estado de filtro, también con warm start.

Son 90 simulaciones: 6 sin modelo identificado y 84 con modelo.

## 4. Resultados

Ideal: RMSE de 0,0382 en la theta-gamma (E 0,0426, I 0,0331) y de 0,0324 en la gamma (E 0,0313, I 0,0335). Con modelo, los valores son media ± desvío sobre tres semillas.

**Planta con refractariedad**

| modelo | uso | σ | error de $\hat\beta$ | RMSE theta-gamma | RMSE gamma |
|----------------|--------------------|------|----------|--------------------|--------------------|
| ideal | | | 0 % | 0,0382 | 0,0324 |
| β verdaderos | | | 0 % | 0,0386 | 0,0339 |
| white-box | solo $\hat\beta$ | 0,01 | 10,5 % | 0,0388 ± 0,0000 | 0,0339 ± 0,0000 |
| | | 0,05 | 5,4 % | 0,0386 ± 0,0000 | 0,0332 ± 0,0001 |
| agnóstica (red) | solo $\hat\beta$ | 0,01 | 3,5 % | 0,0387 ± 0,0000 | 0,0342 ± 0,0000 |
| | | 0,05 | 4,9 % | 0,0386 ± 0,0000 | 0,0339 ± 0,0001 |
| agnóstica (red) | $\hat\beta$ y cancela $g$ | 0,01 | 3,5 % | 0,0395 ± 0,0000 | 0,0340 ± 0,0001 |
| | | 0,05 | 4,9 % | 0,0393 ± 0,0001 | 0,0336 ± 0,0004 |
| forma exacta | $\hat\beta$ e invierte la forma | 0,01 | 2,8 % | 0,0384 ± 0,0000 | **0,0326 ± 0,0000** |
| | | 0,05 | 7,6 % | 0,0384 ± 0,0001 | **0,0325 ± 0,0002** |

**Planta con actuador**

| modelo | uso | σ | error de $\hat\beta$ | RMSE theta-gamma | RMSE gamma |
|----------------|--------------------|------|----------|--------------------|--------------------|
| ideal | | | 0 % | 0,0382 | 0,0324 |
| β verdaderos | | | 0 % | **0,0599** | **0,0692** |
| white-box | solo $\hat\beta$ | 0,01 | 25,2 % | 0,0643 ± 0,0001 | 0,0901 ± 0,0001 |
| | | 0,05 | 20,8 % | 0,0651 ± 0,0035 | 0,0878 ± 0,0012 |
| historia 5 ms (red) | solo $\hat\beta$ | 0,01 | 24,3 % | 0,0680 ± 0,0045 | 0,0828 ± 0,0068 |
| | | 0,05 | 29,4 % | 0,0713 ± 0,0003 | 0,0771 ± 0,0088 |
| estado de filtro | solo $\hat\beta$ | 0,01 | 8,7 % | 0,0717 ± 0,0002 | 0,0765 ± 0,0023 |
| | | 0,05 | 14,5 % | 0,0707 ± 0,0002 | 0,0779 ± 0,0028 |

![[control-paper-trazas-actuador.png]]

*Figura 1.* Actividad excitatoria en lazo cerrado (azul) y referencia $r_E$ (gris, discontinua) durante una ráfaga gamma de la theta-gamma de 2 s. Controlador de Martínez et al. (2022) sin cota ni anti-windup. Fila 1: planta sin perturbación con parámetros verdaderos. Filas 2 a 4: planta con actuador (filtro de 1 ms y saturación $3\tanh(u/3)$) con parámetros verdaderos, con los $\hat\beta$ del modelo de historia de 5 ms y con los $\hat\beta$ del estado de filtro, los dos con warm start. Modelos con $\sigma$ = 0,01, semilla 0.

![[control-paper-trazas-refractariedad.png]]

*Figura 2.* Igual que la figura 1, para la planta con refractariedad ($r = 0{,}1$) y tres ráfagas. Filas: parámetros verdaderos; $\hat\beta$ de la agnóstica cancelando $g$; $\hat\beta$ de la forma exacta con la inversión del factor $(1 - r x)$ y de la saturación.

## 5. Observaciones

**Refractariedad.** La perturbación casi no degrada al controlador: con parámetros verdaderos el RMSE sube de 0,0382 a 0,0386 en la theta-gamma y de 0,0324 a 0,0339 en la gamma. Los modelos quedan cerca de ese caso: entre 0,0384 y 0,0395 en la theta-gamma (−0,5 a +2,3 %) y entre 0,0325 y 0,0342 en la gamma (−4,1 a +0,9 %). La forma exacta con su inversión es el único caso que recupera el nivel ideal en la gamma (0,0326 contra 0,0324). Cancelar $g$ mejora la gamma respecto de usar solo $\hat\beta$ y empeora la theta-gamma en un 2 %: en la theta-gamma, el error de E sube de 0,0434 a 0,0443.

**Actuador.** Es la perturbación que importa para el control. Con parámetros verdaderos, el RMSE sube un 57 % en la theta-gamma (0,0382 a 0,0599) y se duplica en la gamma (0,0324 a 0,0692). La figura 1 muestra el origen: el filtro de 1 ms atrasa el comando y la oscilación sale deformada, con picos secundarios. Ningún modelo identificado, usado así, baja del controlador con parámetros verdaderos:

- En la gamma, el estado de filtro queda a +11 a +13 % del controlador con β verdaderos, la historia a +11 a +20 % y el white-box a +27 a +30 %. El orden no sigue del todo al error de $\hat\beta$: el estado de filtro tiene el menor (8,7 a 14,5 %), pero la historia controla mejor que el white-box con más error de parámetros (24,3 a 29,4 % contra 20,8 a 25,2 %).
- En la theta-gamma el orden se invierte: el white-box es el mejor (+7 a +9 %) la historia queda a +14 a +19 % y el estado de filtro a +18 a +20 %. Una explicación posible es que los $\hat\beta$ del white-box absorben parte del actuador, y ese sesgo, que empeora la identificación, compensa algo del atraso en las transiciones entre ráfaga y reposo. No está verificada.

La dispersión entre semillas de la historia (hasta 0,0088) es del mismo orden que las diferencias entre modelos.

**Comando.** Sin cota, el controlador pide en todos los casos comandos muy fuera del rango de los estímulos de entrenamiento (de 0 a 2). Con la planta ideal, $P$ va de −13 a +12 en la theta-gamma; en el actuador llega a −17 y +18 según el modelo. En la planta ideal y en refractariedad, $P$ solo es negativo en la theta-gamma, al apagar cada ráfaga. Esto también ocurre en el paper: su saturación acota $\tilde u_x$ a $[-k_x, 1 - k_x]$ para que exista la sigmoidea inversa, pero no acota $P$ ni $Q$. Su figura 8d muestra $P$ entre −15 y +8 en la misma tarea.

**Limitaciones de esta prueba.** El controlador del paper solo aprovecha de un modelo con red lo que entra en su despeje cerrado: los $\hat\beta$ y, en refractariedad, una $g$ que depende solo del estado. Las variantes que invierten la red o el estado de filtro, y las versiones con cota y anti-windup, están en [[Controladores - resultados]]. La medición es sin ruido y el controlador recibe $E$ e $I$ directamente; en el paper los estima un filtro de Kalman extendido a partir del LFP con ruido. Fuera de eso, la estructura es la misma: ganancias, saturación de $\tilde u_x$, inversión de la sigmoidea y cancelación del acoplamiento.
