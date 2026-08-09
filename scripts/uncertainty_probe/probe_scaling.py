#!/usr/bin/env python3
"""
DONDE SE ROMPE LA MEMORYLESSNESS.

En el barrido anterior el retardo y el actuador dieron R2 alto, contra lo que
dice la intuicion (ambos tienen estado propio). Hipotesis: para constantes de
tiempo CHICAS frente a la dinamica del sistema, el efecto es de primer orden

    E(t-tau) ~ E(t) - tau*Edot(t)   y   Edot = f_WC(I,E,P,Q)

o sea el termino faltante SIGUE SIENDO una funcion del estado -> g_phi lo captura.
La memoria recien aparece cuando la constante de tiempo es comparable a te,ti.

Se barre la constante de tiempo de cada mecanismo con la INTENSIDAD FIJA.
"""
from __future__ import annotations

import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_P(__file__).resolve().parent))


import numpy as np
from scipy.special import expit

import probe_uncertainty as PU
from probe_uncertainty import make_delay, make_adapt, P0, KE, KI, f_wc
from probe_learnable import collect, fit_r2, STIM_TR, STIM_TE


def make_actuator_tau(tau_act, sat):
    """Actuador con lag y saturacion SEPARABLES, para aislar cual manda."""
    def maker(eps):
        def rhs(t, s, P, Q, hist, k):
            I, E, Pl, Ql = s[0], s[1], s[2], s[3]
            Pe = sat * np.tanh(Pl / sat) if sat is not None else Pl
            Qe = sat * np.tanh(Ql / sat) if sat is not None else Ql
            u_i = P0.wIE * E - P0.wII * I + Qe - P0.thetai
            u_e = P0.wEE * E - P0.wEI * I + Pe - P0.thetae
            dI = (1.0 / P0.ti) * (-I + expit(P0.ai * u_i) - KI)
            dE = (1.0 / P0.te) * (-E + expit(P0.ae * u_e) - KE)
            return np.array([dI, dE, (P - Pl) / tau_act, (Q - Ql) / tau_act])
        return 2, rhs
    return maker


def make_adapt_tau(tau_a):
    def maker(eps):
        return make_adapt(eps, tau_a=tau_a)
    return maker


def run(label, maker, eps):
    Xtr, Ytr = collect(maker, eps, STIM_TR)
    Xte, Yte = collect(maker, eps, STIM_TE)
    _, r2te = fit_r2(Xtr, Ytr, Xte, Yte, epochs=1200)
    # tamano del termino faltante
    mag = 100 * np.sqrt((Ytr ** 2).sum(1)).mean() / 0.026
    print(f"  {label:44} |Df|~{mag:6.1f}%   R2_test = {r2te:6.3f}")
    return r2te


def main():
    print("Referencia: te=1 ms, ti=2 ms (las constantes propias del sistema)\n")

    print("A) RETARDO AXONAL — se barre tau [ms] (te=1, ti=2)")
    for tau in (0.2, 0.5, 1.0, 2.0, 4.0, 8.0):
        run(f"tau = {tau:4.1f} ms   (tau/te = {tau/P0.te:4.1f})", make_delay, tau)

    print("\nB) ACTUADOR — lag PURO (sin saturacion), se barre tau_act [ms]")
    for tau in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
        run(f"tau_act = {tau:4.1f} ms (lag puro)", make_actuator_tau(tau, None), 1.0)

    print("\nC) ACTUADOR — saturacion PURA (sin lag: tau_act muy chico)")
    for sat in (1.5, 0.8, 0.4):
        run(f"saturacion A = {sat:.1f}, tau_act = 0.05 ms",
            make_actuator_tau(0.05, sat), 1.0)

    print("\nD) ADAPTACION — se barre tau_a [ms] con b = 0.5 fijo")
    for tau in (1.0, 3.0, 10.0, 30.0, 100.0):
        run(f"tau_a = {tau:5.1f} ms (tau_a/te = {tau:5.1f})",
            make_adapt_tau(tau), 0.5)


if __name__ == "__main__":
    main()
