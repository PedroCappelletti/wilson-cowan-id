#!/usr/bin/env python3
"""
PRUEBA DECISIVA: cuanto del termino faltante puede capturar g_phi(I,E,P,Q)?

Para cada familia de incertidumbre se calcula el Delta f VERDADERO (conocemos el
simulador) y se entrena EXACTAMENTE la arquitectura de la correccion neuronal de
src/neural_ode/dynamics.py  (Linear(4,32)-Tanh-Linear(32,32)-Tanh-Linear(32,2))
para predecirlo desde (I,E,P,Q).

Split: se entrena con unos estimulos y se evalua en otros NUNCA VISTOS (mismo
criterio que is_test en gen_multi_dataset.py).

R2_test ~ 1  -> la perturbacion es una funcion sin memoria del estado y la
                entrada: g_phi PUEDE representarla. Gray-box viable.
R2_test << 1 -> hay estado oculto / memoria / dependencia explicita del tiempo:
                g_phi(I,E,P,Q) NO puede representarla por mas que entrene.
"""
from __future__ import annotations

import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_P(__file__).resolve().parent))


import numpy as np
import torch
import torch.nn as nn

from probe_uncertainty import (
    simulate, make_nominal, FAMILIES, HZ, DT,
)
from src.wilson_cowan import (
    aprbs_pulse, theta_gamma_pulse, prbs_pulse, chirp_pulse, box_pulse,
)

# Estimulos de ENTRENAMIENTO (los que el modelo ve)
STIM_TR = {
    "aprbs1": (aprbs_pulse(1.0, 10, 190, 2, 8, seed=71, amp_min=0.2),
               aprbs_pulse(0.8, 10, 190, 2.6, 9.6, seed=81, amp_min=0.1)),
    "aprbs2": (aprbs_pulse(1.4, 10, 190, 1.5, 6, seed=72, amp_min=0.28),
               aprbs_pulse(1.12, 10, 190, 1.95, 7.2, seed=82, amp_min=0.14)),
    "thetagamma": (theta_gamma_pulse(1.0, HZ(40), HZ(10), 10, 190, 0.5),
                   theta_gamma_pulse(0.7, HZ(36), HZ(10), 10, 190, 0.5)),
    "prbs": (prbs_pulse(1.0, 10, 190, 4, seed=91),
             prbs_pulse(0.8, 10, 190, 4.8, seed=101)),
    "box": (box_pulse(0.8, 10, 190), box_pulse(0.56, 15, 185)),
}

# Estimulos de TEST (held-out: familias enteras que no se entrenaron)
STIM_TE = {
    "chirp": (chirp_pulse(0.8, HZ(10), HZ(150), 10, 190),
              chirp_pulse(0.6, HZ(15), HZ(120), 10, 190)),
    "aprbs_new": (aprbs_pulse(0.8, 10, 190, 3, 10, seed=73, amp_min=0.16),
                  aprbs_pulse(0.64, 10, 190, 3.9, 12, seed=83, amp_min=0.08)),
}


def collect(maker, eps, stims):
    """Devuelve X=(I,E,P,Q) y Y=Delta f sobre las trayectorias PERTURBADAS."""
    Xs, Ys = [], []
    for (Pf, Qf) in stims.values():
        x, df, fw, Ps, Qs = simulate(maker(eps), Pf, Qf)
        Xs.append(np.column_stack([x[:, 0], x[:, 1], Ps, Qs]))
        Ys.append(df)
    return np.vstack(Xs).astype(np.float32), np.vstack(Ys).astype(np.float32)


def make_g(hidden=32):
    """Copia exacta de la correccion g_phi de src/neural_ode/dynamics.py."""
    g = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(),
                      nn.Linear(hidden, hidden), nn.Tanh(),
                      nn.Linear(hidden, 2))
    for m in g:
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias)
    return g


def fit_r2(Xtr, Ytr, Xte, Yte, epochs=1500, seed=0):
    torch.manual_seed(seed)
    g = make_g()
    # escalado de entrada (el forward real no lo tiene, pero sin esto la
    # comparacion entre familias mediria el acondicionamiento, no la capacidad)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    xt = torch.tensor((Xtr - mu) / sd); yt = torch.tensor(Ytr)
    xv = torch.tensor((Xte - mu) / sd); yv = torch.tensor(Yte)
    sy = float(np.sqrt((Ytr ** 2).mean())) + 1e-12
    yt_s, yv_s = yt / sy, yv / sy

    opt = torch.optim.Adam(g.parameters(), lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = xt.shape[0]
    for ep in range(epochs):
        idx = torch.randint(0, n, (4096,))
        opt.zero_grad()
        loss = ((g(xt[idx]) - yt_s[idx]) ** 2).mean()
        loss.backward(); opt.step(); sch.step()

    def r2(x, y):
        with torch.no_grad():
            p = g(x)
        ss_res = float(((p - y) ** 2).sum())
        ss_tot = float(((y - y.mean(0)) ** 2).sum())
        return 1.0 - ss_res / max(ss_tot, 1e-30)
    return r2(xt, yt_s), r2(xv, yv_s)


def main():
    print("R2 de g_phi(I,E,P,Q) ajustando el Delta f VERDADERO")
    print("(train: 5 estimulos | test: chirp + APRBS nuevos, nunca vistos)\n")
    hdr = f"{'familia':36} {'eps':>6} {'R2_train':>9} {'R2_test':>9}  veredicto"
    print(hdr); print("-" * (len(hdr) + 22))

    for fam, (maker, eps_list) in FAMILIES.items():
        eps = eps_list[1]                      # intensidad media de cada familia
        Xtr, Ytr = collect(maker, eps, STIM_TR)
        Xte, Yte = collect(maker, eps, STIM_TE)
        r2tr, r2te = fit_r2(Xtr, Ytr, Xte, Yte)
        if r2te > 0.95:
            v = "CAPTURABLE     (sin memoria)"
        elif r2te > 0.6:
            v = "parcial        (memoria leve)"
        elif r2te > 0.1:
            v = "POBRE          (estado oculto)"
        else:
            v = "IMPOSIBLE      (irreducible)"
        print(f"{fam:36} {eps:6.3g} {r2tr:9.3f} {r2te:9.3f}  {v}")


if __name__ == "__main__":
    main()
