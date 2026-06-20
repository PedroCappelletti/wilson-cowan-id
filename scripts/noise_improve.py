#!/usr/bin/env python3
# =============================================================================
#  DIAGNÓSTICO — qué mejora la identificación bajo ruido alto (σ=0.10)
# =============================================================================
#
#  El experimento de ruido (Iter 12) rompía θ̂ a ruido alto (excitación moderada).
#  Acá se prueban 4 palancas a σ=0.10 para ver cuál ayuda, ANTES del barrido full:
#    A) baseline      : escenarios moderados, sin suavizar.
#    B) +suavizado    : promedio móvil de I,E (quita ruido de alta frecuencia).
#    C) +estímulo fuerte: escenarios con Q GRANDE (excita I -> mejor SNR de wII).
#    D) C + suavizado.
#
#  Reporta el error de θ̂ por config. USO: python -u scripts/noise_improve.py
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.wilson_cowan import (
    WilsonCowanParams, zero_input,
    aprbs_pulse, theta_gamma_pulse, square_wave_pulse, prbs_pulse, chirp_pulse,
)
from src.data import generate_dataset
from src.neural_ode import GrayBoxWC, rollout
from scripts.train_neural_ode import make_windows, WINDOW, IDENT

T_SPAN = (0.0, 200.0)
N_EVAL = 4000
SEED = 42
W_TRUE = {"wEE": 6.4, "wEI": 4.8, "wIE": 6.0, "wII": 1.2}
EPOCHS, LBFGS_STEPS, LR_W = 1500, 30, 5e-2
hz = lambda f: f / 1000.0
ton, toff = 10.0, 190.0


def baseline_scenarios():
    import scripts.gen_multi_dataset as G
    return [(lbl, Pf, Qf, t) for lbl, Pf, Qf, t in G.build_scenarios()]


def strong_scenarios():
    """Excitación fuerte y diversa, con Q GRANDE en varias (excita I -> wII)."""
    ap = lambda a, amin, s: aprbs_pulse(a, ton, toff, 2, 8, seed=s, amp_min=amin)
    sq = lambda a, f, d: square_wave_pulse(a, hz(f), ton, toff, d)
    return [
        ("E_fuerte_Q0",  ap(1.4, 0.4, 301), zero_input, False),
        ("E_fuerte_Qb",  ap(1.2, 0.3, 302), ap(0.5, 0.1, 312), False),
        ("E_sq",         sq(1.2, 100, 0.5), sq(0.4, 80, 0.5), True),
        ("Qgrande_a",    ap(1.0, 0.2, 303), ap(4.0, 1.5, 313), False),
        ("Qgrande_b",    ap(0.8, 0.2, 304), ap(5.0, 2.0, 314), True),
        ("P0_Qmuygde",   zero_input,        ap(5.0, 2.0, 315), False),
        ("Qgde_sq",      sq(1.0, 100, 0.5), sq(4.0, 80, 0.5), False),
        ("Qgde_inter",   ap(1.2, 0.3, 305), ap(3.0, 1.0, 316), True),
        ("bal_aprbs",    ap(0.9, 0.2, 306), ap(0.9, 0.2, 317), False),
        ("bal_prbs",     prbs_pulse(1.0, ton, toff, 4, seed=307),
                         prbs_pulse(1.2, ton, toff, 5, seed=318), False),
        ("bal_tg",       theta_gamma_pulse(1.0, hz(40), hz(10), ton, toff, 0.5),
                         theta_gamma_pulse(1.2, hz(50), hz(12), ton, toff, 0.5), True),
        ("bal_chirp",    chirp_pulse(0.8, hz(10), hz(150), ton, toff),
                         chirp_pulse(1.0, hz(15), hz(120), ton, toff), False),
    ]


def smooth(x, k):
    if k <= 1:
        return x
    ker = np.ones(k) / k
    return np.stack([np.convolve(row, ker, mode="same") for row in x])


def generate(scenarios, noise):
    p = WilsonCowanParams()
    I, E, P, Q, is_test = [], [], [], [], []
    dt = None
    for lbl, Pf, Qf, test in scenarios:
        ds = generate_dataset(params=p, P=Pf, Q=Qf, I0=0.0, E0=0.0,
                              t_span=T_SPAN, n_eval=N_EVAL, noise_std=noise, seed=SEED)
        I.append(ds["I"]); E.append(ds["E"]); P.append(ds["P"]); Q.append(ds["Q"])
        is_test.append(test); dt = float(ds["t"][1] - ds["t"][0])
    return (np.stack(I), np.stack(E), np.stack(P), np.stack(Q),
            np.asarray(is_test, dtype=bool), dt)


def identify(scenarios, noise, smooth_k):
    p = WilsonCowanParams()
    fixed = {"te": p.te, "ti": p.ti, "ae": p.ae, "ai": p.ai,
             "thetae": p.thetae, "thetai": p.thetai, "ke": p.ke, "ki": p.ki}
    I, E, P, Q, is_test, dt = generate(scenarios, noise)
    if smooth_k > 1:
        I, E = smooth(I, smooth_k), smooth(E, smooth_k)
    Itr, Etr, Ptr, Qtr = I[~is_test], E[~is_test], P[~is_test], Q[~is_test]
    model = GrayBoxWC(fixed, {k: 1.0 for k in IDENT}, learnable_weights=True, use_correction=False)
    x0, Pw, Qw, tgt = make_windows(Itr, Etr, Ptr, Qtr, WINDOW)
    opt = torch.optim.Adam([{"params": [model.raw_w], "lr": LR_W}])
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = ((rollout(model, x0, Pw, Qw, dt) - tgt) ** 2).mean()
        loss.backward(); opt.step()
    opt2 = torch.optim.LBFGS([model.raw_w], lr=1.0, max_iter=20, line_search_fn="strong_wolfe")
    def closure():
        opt2.zero_grad(); l = ((rollout(model, x0, Pw, Qw, dt) - tgt) ** 2).mean()
        l.backward(); return l
    for _ in range(LBFGS_STEPS):
        opt2.step(closure)
    w = model.weights_dict()
    err = {k: 100.0 * abs(w[k] - W_TRUE[k]) / W_TRUE[k] for k in W_TRUE}
    return w, err


def main():
    NOISE = 0.10
    base = baseline_scenarios()
    strong = strong_scenarios()
    configs = [
        ("A baseline",            base,   1),
        ("B +suavizado",          base,   7),
        ("C estimulo fuerte",     strong, 1),
        ("D fuerte+suavizado",    strong, 7),
    ]
    print(f"=== Diagnóstico a σ={NOISE} (error de θ̂, menor = mejor) ===")
    for name, sc, sk in configs:
        torch.manual_seed(0); np.random.seed(0)
        w, err = identify(sc, NOISE, sk)
        print(f"  {name:22} | max={max(err.values()):6.1f}% | "
              + " ".join(f"{k}={err[k]:5.1f}%" for k in W_TRUE)
              + " | " + " ".join(f"{k}={w[k]:.2f}" for k in W_TRUE))


if __name__ == "__main__":
    main()
