#!/usr/bin/env python3
# =============================================================================
#  HERRAMIENTA COMPARTIDA — identificar los 10 params fijando / regularizando
#  un subconjunto (parameter subset selection / profile-likelihood / MAP ridge)
# =============================================================================
#
#  Generaliza identify_full (noise_full_sweep.py) para poder:
#    - FIJAR un subconjunto de parametros en su valor verdadero (no se optimizan),
#      = "parameter subset selection" (Chu & Hahn 2007; Weijers & Kok 1997): dejar
#      fuera del ajuste los parametros mal condicionados y estimar solo el resto.
#    - REGULARIZAR parametros hacia un prior con penalizacion L2 (estimacion MAP /
#      ridge; profile-likelihood suave): en el limite lambda->inf equivale a fijar.
#
#  Reutiliza: generate/smooth/strong_scenarios (noise_improve), make_windows/WINDOW
#  (train_neural_ode), GrayBoxWC/rollout (neural_ode). NO duplica el nucleo.
#
#  Uso como libreria:  from scripts.ident_subset import identify_subset, true_params
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.wilson_cowan import WilsonCowanParams
from src.neural_ode import GrayBoxWC, rollout
from scripts.train_neural_ode import make_windows, WINDOW
from scripts.noise_improve import generate, smooth

WEIGHTS = ("wEE", "wEI", "wIE", "wII")
PHYS    = ("te", "ti", "ae", "ai", "thetae", "thetai")
ALL_P   = WEIGHTS + PHYS

EPOCHS, LBFGS_STEPS = 1500, 40
LR_W, LR_PHYS = 5e-2, 2e-2


def true_params() -> dict:
    p = WilsonCowanParams()
    return {"wEE": p.wEE, "wEI": p.wEI, "wIE": p.wIE, "wII": p.wII,
            "te": p.te, "ti": p.ti, "ae": p.ae, "ai": p.ai,
            "thetae": p.thetae, "thetai": p.thetai}


def _fixed_init(init: dict) -> dict:
    """Dict fixed (con ke,ki derivados) para GrayBoxWC desde valores iniciales."""
    ae, ai, the, thi = init["ae"], init["ai"], init["thetae"], init["thetai"]
    return {"te": init["te"], "ti": init["ti"], "ae": ae, "ai": ai,
            "thetae": the, "thetai": thi,
            "ke": 1.0 / (1.0 + np.exp(ae * the)),
            "ki": 1.0 / (1.0 + np.exp(ai * thi))}


def identify_subset(scenarios, noise, smooth_k, fix=(), reg_lambda=0.0, reg_params=(),
                    seed=0, epochs=EPOCHS, lbfgs=LBFGS_STEPS):
    """Identifica los 10 params de WC desde arranque ignorante (1.0), salvo:
        fix         : nombres que se MANTIENEN en su valor verdadero (no se optimizan).
        reg_params  : nombres penalizados L2 hacia su valor verdadero (prior) con reg_lambda.
    Devuelve (p_hat, err) donde err[k] = |p_hat-true|/true * 100.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    true = true_params()
    fix = set(fix); reg_params = tuple(reg_params)

    # Arranque: ignorante (1.0) para libres; verdadero para fijos.
    init = {k: 1.0 for k in ALL_P}
    for k in fix:
        init[k] = true[k]

    model = GrayBoxWC(_fixed_init(init), {k: init[k] for k in WEIGHTS},
                      learnable_weights=True, use_correction=False, learnable_params=True)

    # Datos.
    I, E, P, Q, is_test, dt = generate(scenarios, noise)
    if smooth_k > 1:
        I, E = smooth(I, smooth_k), smooth(E, smooth_k)
    Itr, Etr, Ptr, Qtr = I[~is_test], E[~is_test], P[~is_test], Q[~is_test]
    x0, Pw, Qw, tgt = make_windows(Itr, Etr, Ptr, Qtr, WINDOW)

    # Que se optimiza: pesos libres (via mascara de gradiente) + fisicos libres.
    fixed_w_idx = [i for i, k in enumerate(WEIGHTS) if k in fix]
    free_phys = [k for k in PHYS if k not in fix]
    groups = [{"params": [model.raw_w], "lr": LR_W}]
    if free_phys:
        groups.append({"params": [getattr(model, f"raw_{k}") for k in free_phys], "lr": LR_PHYS})
    opt = torch.optim.Adam(groups)

    def value_of(k):
        return model.weights()[WEIGHTS.index(k)] if k in WEIGHTS else model._extra(k)

    def reg_term():
        if reg_lambda <= 0.0 or not reg_params:
            return torch.zeros((), dtype=torch.float32)
        r = torch.zeros((), dtype=torch.float32)
        for k in reg_params:
            r = r + (value_of(k) - true[k]) ** 2
        return reg_lambda * r

    def mask_fixed():
        if fixed_w_idx and model.raw_w.grad is not None:
            model.raw_w.grad[fixed_w_idx] = 0.0

    for _ in range(epochs):
        opt.zero_grad()
        loss = ((rollout(model, x0, Pw, Qw, dt) - tgt) ** 2).mean() + reg_term()
        loss.backward()
        mask_fixed()
        opt.step()

    if lbfgs > 0:
        params = [model.raw_w] + [getattr(model, f"raw_{k}") for k in free_phys]
        opt2 = torch.optim.LBFGS(params, lr=1.0, max_iter=20, line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            l = ((rollout(model, x0, Pw, Qw, dt) - tgt) ** 2).mean() + reg_term()
            l.backward()
            mask_fixed()
            return l
        for _ in range(lbfgs):
            opt2.step(closure)

    p = model.params_dict()
    err = {k: 100.0 * abs(p[k] - true[k]) / abs(true[k]) for k in ALL_P}
    return p, err


if __name__ == "__main__":
    # Smoke test: identificar 10 libres vs fijar wII, en limpio.
    from scripts.noise_improve import strong_scenarios
    sc = strong_scenarios()
    print("=== smoke test (sigma=0, limpio) ===")
    _, e_free = identify_subset(sc, 0.0, 1)
    print(f"10 libres      : wII err={e_free['wII']:.2f}%  max={max(e_free.values()):.2f}%")
    _, e_fix = identify_subset(sc, 0.0, 1, fix={"wII"})
    print(f"wII fijo       : max(resto)={max(v for k,v in e_fix.items() if k!='wII'):.2f}%")
