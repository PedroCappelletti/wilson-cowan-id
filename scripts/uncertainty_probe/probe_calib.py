#!/usr/bin/env python3
"""
CALIBRACION del par recomendado: refractariedad (entrada 1) + actuador (entrada 2).

Busca la intensidad que deja el efecto BIEN VISIBLE por encima del ruido de
observacion que ya se usa en el proyecto (sigma = 0.01 .. 0.10) pero SIN destruir
el regimen oscilatorio ni el rango de operacion.

Criterio: D_abs (deformacion absoluta de la trayectoria) debe ser >> sigma para
que la perturbacion sea detectable, y el sistema debe seguir oscilando con un
rango de E comparable al nominal.
"""
from __future__ import annotations

import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_P(__file__).resolve().parent))


import numpy as np
from scipy.special import expit

from probe_uncertainty import simulate, make_nominal, make_refract, P0, KE, KI, HZ
from probe_learnable import collect, fit_r2, STIM_TR, STIM_TE
from src.wilson_cowan import aprbs_pulse, theta_gamma_pulse

STIM = {
    "aprbs": (aprbs_pulse(1.0, 10, 190, 2, 8, seed=71, amp_min=0.2),
              aprbs_pulse(0.8, 10, 190, 2.6, 9.6, seed=81, amp_min=0.1)),
    "thetagamma": (theta_gamma_pulse(1.0, HZ(40), HZ(10), 10, 190, 0.5),
                   theta_gamma_pulse(0.7, HZ(36), HZ(10), 10, 190, 0.5)),
}


def make_combo(r, sat, tau_act):
    """Las DOS entradas simultaneas: refractariedad (estado) + actuador (entrada)."""
    def maker(eps):
        def rhs(t, s, P, Q, hist, k):
            I, E, Pl, Ql = s[0], s[1], s[2], s[3]
            Pe = sat * np.tanh(Pl / sat)
            Qe = sat * np.tanh(Ql / sat)
            u_i = P0.wIE * E - P0.wII * I + Qe - P0.thetai
            u_e = P0.wEE * E - P0.wEI * I + Pe - P0.thetae
            dI = (1.0 / P0.ti) * (-I + (1.0 - r * I) * expit(P0.ai * u_i) - KI)
            dE = (1.0 / P0.te) * (-E + (1.0 - r * E) * expit(P0.ae * u_e) - KE)
            return np.array([dI, dE, (P - Pl) / tau_act, (Q - Ql) / tau_act])
        return 2, rhs
    return maker


def make_act(sat, tau_act):
    def maker(eps):
        def rhs(t, s, P, Q, hist, k):
            I, E, Pl, Ql = s[0], s[1], s[2], s[3]
            Pe = sat * np.tanh(Pl / sat); Qe = sat * np.tanh(Ql / sat)
            u_i = P0.wIE * E - P0.wII * I + Qe - P0.thetai
            u_e = P0.wEE * E - P0.wEI * I + Pe - P0.thetae
            dI = (1.0 / P0.ti) * (-I + expit(P0.ai * u_i) - KI)
            dE = (1.0 / P0.te) * (-E + expit(P0.ae * u_e) - KE)
            return np.array([dI, dE, (P - Pl) / tau_act, (Q - Ql) / tau_act])
        return 2, rhs
    return maker


def metrics(maker, eps=1.0):
    Dt, Da, Emax, osc = [], [], [], []
    for name, (Pf, Qf) in STIM.items():
        xn, _, _, _, _ = simulate(make_nominal(0), Pf, Qf)
        xp, dfp, fwp, _, _ = simulate(maker(eps), Pf, Qf)
        d = xp - xn; sig = xn - xn.mean(0)
        Dt.append(100 * np.sqrt((d ** 2).mean()) / np.sqrt((sig ** 2).mean()))
        Da.append(np.sqrt((d ** 2).mean()))
        Emax.append(xp[:, 1].max())
        e = xp[:, 1] - xp[:, 1].mean()
        osc.append(int(np.sum(np.diff(np.sign(e)) != 0)))
    return np.mean(Dt), np.mean(Da), np.mean(Emax), int(np.mean(osc))


def main():
    _, _, En, on = metrics(make_nominal)
    print(f"NOMINAL: E_max = {En:.3f}, cruces = {on}\n")

    print("ENTRADA 1 — refractariedad (1 - r*x) sobre la sigmoidea")
    print(f"  {'r':>6} {'D_traj%':>8} {'D_abs':>8} {'E_max':>7} {'osc':>5}  vs sigma_ruido")
    for r in (0.05, 0.1, 0.2, 0.35, 0.5, 0.8):
        dt, da, em, os_ = metrics(make_refract, r)
        print(f"  {r:6.2f} {dt:8.2f} {da:8.4f} {em:7.3f} {os_:5d}   "
              f"{da/0.02:5.1f}x sigma=0.02")

    print("\nENTRADA 2 — actuador optogenetico (lag tau_act + saturacion A)")
    print(f"  {'A':>5} {'tau':>5} {'D_traj%':>8} {'D_abs':>8} {'E_max':>7} {'osc':>5}")
    for sat in (3.0, 2.0, 1.5):
        for tau in (0.5, 1.0, 2.0):
            dt, da, em, os_ = metrics(make_act(sat, tau))
            print(f"  {sat:5.1f} {tau:5.1f} {dt:8.2f} {da:8.4f} {em:7.3f} {os_:5d}")

    print("\nLAS DOS JUNTAS (lo que iria al simulador)")
    print(f"  {'r':>5} {'A':>5} {'tau':>5} {'D_traj%':>8} {'D_abs':>8} {'E_max':>7} {'osc':>5} {'R2_g':>7}")
    for (r, sat, tau) in [(0.1, 3.0, 1.0), (0.2, 2.0, 1.0), (0.2, 1.5, 2.0), (0.35, 1.5, 1.0)]:
        mk = make_combo(r, sat, tau)
        dt, da, em, os_ = metrics(mk)
        Xtr, Ytr = collect(mk, 1.0, STIM_TR)
        Xte, Yte = collect(mk, 1.0, STIM_TE)
        _, r2 = fit_r2(Xtr, Ytr, Xte, Yte, epochs=1200)
        print(f"  {r:5.2f} {sat:5.1f} {tau:5.1f} {dt:8.2f} {da:8.4f} {em:7.3f} {os_:5d} {r2:7.3f}")


if __name__ == "__main__":
    main()
