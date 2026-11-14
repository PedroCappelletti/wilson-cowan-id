#!/usr/bin/env python3
# Datasets de 2 s a 2 kHz con comandos de amplitud grande (desde el 10-11-2026).
#
# Los datasets anteriores cubren P en [0, 2], y el controlador del paper pide
# comandos entre -13 y 13, negativos más de la mitad del tiempo. Este dataset
# entrena con la misma amplitud que usa el controlador.
#
#   Familias: square, APRBS, theta-gamma, Poisson, chirp y ruido blanco. Box y
#   PRBS salen del entrenamiento.
#   Entrenamiento: una amplitud por escenario y familia, de 1 a 12 en pasos de 1.
#   La amplitud A es bipolar: el comando P va de -A a +A. Q va de -A_Q a +A_Q
#   con A_Q = 14/12 · A, porque el controlador pide más en Q que en P.
#   Validación y test: dos amplitudes cada uno, fuera de la grilla (1,5 a 11,5),
#   distintas en cada familia, para medir interpolación entre amplitudes.
#   Test del dataset anterior: los 14 escenarios de test de
#   build_scenarios_ampliado, sin cambios, para comparar con lo hecho. Cuatro
#   son de box y PRBS, familias sin entrenamiento.
#
# Total 72 + 12 + 12 + 14 = 110 escenarios. El estímulo va de 100 a 1900 ms y se
# retiene entre muestras, como en esc_gen_largo.py.
#
# Ruido blanco: gaussiano de media 0, filtrado con un Butterworth de orden 4 sin
# desfase a 50, 150 o 400 Hz, y escalado a desvío A/3 dentro de la ventana. No
# se recorta: el 99,7 % de las muestras queda en [-A, A].
#
# Archivos: refrac_amp_2s, actc_amp_2s, wc0_amp_2s. Ruido con
#   python scripts/esc_gen_ruido.py <nombre> 0.01 0.05 --ref-fs 20000 [--semilla k]
#
# USO:  python scripts/esc_gen_amplitud.py [nombre ...]

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import numpy as np
from scipy.signal import butter, filtfilt

from src.wilson_cowan import Refractoriness, Actuator, NoPerturbation

from gen_multi_dataset import build_scenarios_ampliado
from gen_uncertain_dataset import generar_con, OUT_DIR

T_SPAN = (0.0, 2000.0)      # ms
DT = 0.5                    # ms
N_EVAL = int(round((T_SPAN[1] - T_SPAN[0]) / DT)) + 1
TON, TOFF = 100.0, 1900.0   # ms
AMPLITUDES = list(range(1, 13))
FUERA = [1.5 + k for k in range(11)]          # 1,5 ... 11,5
FACTOR_Q = 14.0 / 12.0
FAMILIAS = ["square", "aprbs", "thetagamma", "poisson", "chirp", "ruido"]


def hz(f):
    return f / 1000.0


def ventana(f):
    """El estímulo vale 0 fuera de [TON, TOFF)."""
    return lambda t: f(t) if TON <= t < TOFF else 0.0


# --- familias bipolares, amplitud A ------------------------------------------------

def square(A, f_hz, duty):
    return ventana(lambda t: A if (hz(f_hz) * (t - TON)) % 1.0 < duty else -A)


def aprbs(A, dmin, dmax, semilla):
    rng = np.random.default_rng(semilla)
    bordes, valores, t = [TON], [], TON
    while t < TOFF:
        t += rng.uniform(dmin, dmax)
        bordes.append(min(t, TOFF))
        valores.append(rng.uniform(-A, A))
    b, v = np.asarray(bordes), np.asarray(valores)

    def f(t):
        i = int(np.searchsorted(b, t, side="right")) - 1
        return float(v[min(max(i, 0), len(v) - 1)])
    return ventana(f)


def thetagamma(A, fg, ft, duty=0.5):
    """Ráfagas gamma de signo alternado (+A, -A) bajo una envolvente theta en [0, 1]."""
    def f(t):
        tau = t - TON
        env = 0.5 * (1.0 + np.sin(2 * np.pi * hz(ft) * tau))
        return A * env * (1.0 if (hz(fg) * tau) % 1.0 < duty else -1.0)
    return ventana(f)


def poisson(A, tasa, ancho, semilla):
    """Pulsos de ancho fijo en tiempos de Poisson, con signo al azar."""
    rng = np.random.default_rng(semilla)
    tiempos, t = [], TON
    while True:
        t += rng.exponential(1.0 / tasa)
        if t >= TOFF:
            break
        tiempos.append(t)
    tt = np.asarray(tiempos)
    signo = rng.choice([-1.0, 1.0], size=len(tt))

    def f(t):
        i = int(np.searchsorted(tt, t, side="right")) - 1
        return A * signo[i] if i >= 0 and t - tt[i] < ancho else 0.0
    return ventana(f)


def chirp(A, f0, f1, fase=0.0):
    k = (f1 - f0) / (TOFF - TON)

    def f(t):
        tau = t - TON
        return A * np.sin(2 * np.pi * (hz(f0) * tau + 0.5 * hz(k) * tau * tau) + fase)
    return ventana(f)


def ruido(A, banda_hz, semilla):
    n = N_EVAL
    t = np.arange(n) * DT
    fs = 1000.0 / DT
    x = np.random.default_rng(semilla).standard_normal(n)
    b, a = butter(4, banda_hz / (fs / 2))
    x = filtfilt(b, a, x)
    dentro = (t >= TON) & (t < TOFF)
    x = x / x[dentro].std() * (A / 3.0)

    def f(tt):
        return float(x[min(int(round(tt / DT)), n - 1)])
    return ventana(f)


# --- parámetros de cada familia, rotando con el índice del escenario --------------

def par(fam, A, k, AQ, semilla):
    """(P, Q) de la familia `fam` con amplitudes A y AQ. k rota frecuencias y
    formas; la semilla separa los sorteos de P y de Q."""
    if fam == "square":
        f = (50, 80, 100, 130)[k % 4]
        return square(A, f, 0.4), square(AQ, 0.8 * f, 0.5)
    if fam == "aprbs":
        dmin, dmax = ((2, 7), (2.5, 9), (1.5, 6), (3, 10))[k % 4]
        return aprbs(A, dmin, dmax, semilla), aprbs(AQ, 1.3 * dmin, 1.2 * dmax, semilla + 500)
    if fam == "thetagamma":
        fg, ft = ((45, 9), (55, 11), (40, 8), (60, 12))[k % 4]
        return thetagamma(A, fg, ft), thetagamma(AQ, 0.9 * fg, ft)
    if fam == "poisson":
        tasa, ancho = ((0.08, 6.0), (0.11, 4.5), (0.10, 5.0), (0.09, 5.5))[k % 4]
        return poisson(A, tasa, ancho, semilla), poisson(AQ, 0.8 * tasa, ancho, semilla + 500)
    if fam == "chirp":
        f0, f1 = ((10, 80), (60, 150), (20, 120), (15, 140))[k % 4]
        return chirp(A, f0, f1), chirp(AQ, 1.2 * f0, 0.85 * f1, fase=1.0)
    if fam == "ruido":
        banda = (50, 150, 400)[k % 3]
        return ruido(A, banda, semilla), ruido(AQ, banda, semilla + 500)
    raise ValueError(fam)


def escenarios():
    S = []
    for i, fam in enumerate(FAMILIAS):
        base = 1000 * (i + 1)
        for k, A in enumerate(AMPLITUDES):
            P, Q = par(fam, A, k, FACTOR_Q * A, base + k)
            S.append((f"{fam}_A{A}", P, Q, "train"))
        # amplitudes fuera de la grilla, distintas en cada familia
        val = [FUERA[i % 11], FUERA[(i + 6) % 11]]
        test = [FUERA[(i + 3) % 11], FUERA[(i + 9) % 11]]
        for rol, amps in (("val", val), ("test", test)):
            for j, A in enumerate(amps):
                P, Q = par(fam, A, 7 + j + (rol == "test") * 2, FACTOR_Q * A, base + 100 + j + (rol == "test") * 10)
                S.append((f"{fam}_A{A}_{rol}", P, Q, rol))
    viejos = [(lab, P, Q, rol) for lab, P, Q, rol in build_scenarios_ampliado(TON, TOFF) if rol == "test"]
    assert len(viejos) == 14
    return S + viejos


def main():
    kw = dict(escenarios=escenarios, t_span=T_SPAN, n_eval=N_EVAL, retenido=True, n_sub=10)
    plantas = [("wc0_amp_2s", NoPerturbation, {"eps": 0.0}),
               ("refrac_amp_2s", lambda: Refractoriness(r=0.10), {"eps": 1.0}),
               ("actc_amp_2s", lambda: Actuator(sat=3.0, tau_act=1.0, recorte=True), {"eps": 1.0})]
    pedidas = sys.argv[1:]
    for nombre, fab, meta in plantas:
        if pedidas and nombre not in pedidas:
            continue
        generar_con(fab, OUT_DIR / f"{nombre}.npz", meta, **kw)
    print("\nListo.")


if __name__ == "__main__":
    main()
