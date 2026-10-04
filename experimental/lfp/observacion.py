"""Proxies de S, el potencial de campo local, como función del estado de
Wilson-Cowan. EN EVALUACIÓN, ver README.md.

Cada proxy recibe I, E, P, Q del mismo tamaño y los pesos w = (wEE, wEI, wIE,
wII), y funciona igual con arrays de numpy que con tensores de torch. Los que
dependen de los pesos usan los verdaderos al generar el dataset y los del
modelo al entrenar, así que también entran en lo que hay que identificar.

La literatura no tiene un observable único, y los que hay se dividen en dos
hipótesis físicas que difieren en el signo de la inhibición:

  - potencial de membrana de las piramidales, excitación menos inhibición
    (Jansen y Rit 1995; Moran et al. 2007, NeuroImage 37:706, salida v2 − v3);
  - suma de magnitudes de las corrientes sobre las piramidales, porque las
    excitatorias y las inhibitorias generan dipolos del mismo signo (Mazzoni
    et al. 2008, PLoS Comput Biol 4:e1000239; Krishnakumaran, Raees y Ray 2022,
    PLoS Comput Biol 18:e1009886, que usan −(rE + rI) sobre un Wilson-Cowan).

Queda afuera el proxy de Mazzoni et al. 2015 (PLoS Comput Biol 11:e1004584),
AMPA(t − 6 ms) − 1,65·GABA(t). Sus 6 ms y su 1,65 salen de una red de
piramidales con morfología fija, y en este simulador τe = 1 ms: el retardo
sería seis constantes de tiempo y además más largo que una ventana de
entrenamiento de 5 ms.
"""

from __future__ import annotations


def E_menos_I(I, E, P, Q, w):
    """S = E − I, con signo, en la línea del potencial de membrana. Es la
    decisión D3 de scripts/train_real_output.py para los datos reales, ahí con
    una ganancia c_out entrenable. Acá la ganancia es 1: el piloto pregunta
    primero qué se pierde por mirar una sola combinación, y una ganancia libre
    agrega otra incógnita."""
    return E - I


def entrada_E(I, E, P, Q, w):
    """S = wEE·E − wEI·I + P, la entrada sináptica neta a la población
    excitatoria. Es el análogo más cercano a la salida de Jansen-Rit y del DCM,
    con una diferencia: Wilson-Cowan no tiene núcleos sinápticos, así que acá
    es instantánea. Depende de dos pesos, así que S cambia si el modelo los
    estima mal."""
    return w[0] * E - w[1] * I + P


def suma_corrientes(I, E, P, Q, w):
    """S = |wEE·E + P| + |wEI·I|, la suma de magnitudes de las corrientes
    excitatorias e inhibitorias sobre la población excitatoria (Mazzoni et al.
    2008, sin retardos ni pesos relativos). Supone que la entrada externa P
    llega como corriente excitatoria a las piramidales, cosa que no está
    verificada en la fuente."""
    return abs(w[0] * E + P) + abs(w[1] * I)


def menos_E_mas_I(I, E, P, Q, w):
    """S = −(E + I), el proxy de Krishnakumaran, Raees y Ray (2022) sobre un
    Wilson-Cowan. Pesa las dos poblaciones igual y con el mismo signo."""
    return -(E + I)


PROXIES = {"E_menos_I": E_menos_I, "entrada_E": entrada_E,
           "suma_corrientes": suma_corrientes, "menos_E_mas_I": menos_E_mas_I}
