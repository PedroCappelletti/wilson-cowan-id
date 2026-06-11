#!/usr/bin/env python3
# =============================================================================
#  ENTRENAR EL NEURAL ODE (planta controlable / identificacion por Neural ODE)
# =============================================================================
#
#  Ajusta el modelo de estados f_θ(x,P,Q) para que, integrado, reproduzca las
#  trayectorias del dataset multi-variante. Como los pesos son aprendibles, esto
#  ES el brazo "Neural ODE" del OE2 (identifica θ a traves del integrador) y a la
#  vez produce la PLANTA controlable (OE3).
#
#  ENTRENAMIENTO POR MULTIPLE SHOOTING: en vez de integrar los 100 ms de un tiron
#  (gradientes que explotan/desvanecen), se parte cada trayectoria en VENTANAS
#  cortas, cada una integrada desde el estado OBSERVADO al inicio de la ventana.
#  Es el truco estandar para entrenar Neural ODEs de forma estable.
#
#  Variantes (flags abajo):
#    - use_correction=False : identificacion pura (aprende los 4 pesos). Comparable a la PINN.
#    - use_correction=True  : gray-box, agrega g_φ para capturar mismatch (paso siguiente).
#
#  USO:  python scripts/train_neural_ode.py
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from src.neural_ode import GrayBoxWC, rollout

# #############################################################################
# ##   ZONA EDITABLE                                                         ##
# #############################################################################

DATA_PATH = Path("data/processed/control/multi_dataset.npz")

LEARN_WEIGHTS  = True    # aprender los 4 pesos (identificacion Neural ODE)
USE_CORRECTION = False   # V_id: sin correccion. True -> gray-box V1 (paso siguiente)
W_INIT         = 1.0     # arranque ignorante (como el experimento 4.2 de la PINN)

WINDOW   = 100           # largo de ventana de multiple shooting (pasos) -> 5 ms con dt=0.05
EPOCHS   = 3000
LR_W     = 5e-2          # lr de los pesos (viajan lejos)
LR_NET   = 1e-3          # lr de la correccion g_φ (si USE_CORRECTION)
LOG_EVERY = 200
LBFGS_STEPS = 50

OUT_DIR = Path("results")

# #############################################################################
# ##   FIN ZONA EDITABLE                                                     ##
# #############################################################################

W_TRUE = {"wEE": 6.4, "wEI": 4.8, "wIE": 6.0, "wII": 1.2}
IDENT  = ("wEE", "wEI", "wIE", "wII")


def load():
    d = np.load(DATA_PATH, allow_pickle=True)
    fixed = {
        "te": float(d["te"]), "ti": float(d["ti"]),
        "ae": float(d["ae"]), "ai": float(d["ai"]),
        "thetae": float(d["thetae"]), "thetai": float(d["thetai"]),
        "ke": 1.0 / (1.0 + np.exp(float(d["ae"]) * float(d["thetae"]))),
        "ki": 1.0 / (1.0 + np.exp(float(d["ai"]) * float(d["thetai"]))),
    }
    return d, fixed, float(d["dt"])


def make_windows(I, E, P, Q, W):
    """Parte trayectorias (n,T) en ventanas. Devuelve tensores apilados:
       x0 (Nw,2), Pw (W,Nw,1), Qw (W,Nw,1), target (W+1,Nw,2)."""
    n, T = I.shape
    nwin = (T - 1) // W
    x0, Pw, Qw, tgt = [], [], [], []
    for s in range(n):
        for w in range(nwin):
            a = w * W
            x0.append([I[s, a], E[s, a]])
            Pw.append(P[s, a:a + W])
            Qw.append(Q[s, a:a + W])
            tgt.append(np.stack([I[s, a:a + W + 1], E[s, a:a + W + 1]], axis=1))
    x0 = torch.tensor(np.asarray(x0), dtype=torch.float32)                       # (Nw,2)
    Pw = torch.tensor(np.asarray(Pw), dtype=torch.float32).T.unsqueeze(-1)        # (W,Nw,1)
    Qw = torch.tensor(np.asarray(Qw), dtype=torch.float32).T.unsqueeze(-1)        # (W,Nw,1)
    tgt = torch.tensor(np.asarray(tgt), dtype=torch.float32).permute(1, 0, 2)     # (W+1,Nw,2)
    return x0, Pw, Qw, tgt


@torch.no_grad()
def open_loop_mse(model, I, E, P, Q, dt):
    """Rollout COMPLETO (sin resets) de cada trayectoria -> MSE. Test de generalizacion."""
    n, T = I.shape
    errs = []
    for s in range(n):
        x0 = torch.tensor([[I[s, 0], E[s, 0]]], dtype=torch.float32)
        Ps = torch.tensor(P[s], dtype=torch.float32).reshape(T, 1, 1)
        Qs = torch.tensor(Q[s], dtype=torch.float32).reshape(T, 1, 1)
        traj = rollout(model, x0, Ps[:-1], Qs[:-1], dt)[:, 0, :]   # (T,2)
        tgt = torch.tensor(np.stack([I[s], E[s]], axis=1), dtype=torch.float32)
        errs.append(float(((traj - tgt) ** 2).mean()))
    return float(np.mean(errs))


def main():
    d, fixed, dt = load()
    is_test = d["is_test"].astype(bool)
    I, E, P, Q = d["I"], d["E"], d["P"], d["Q"]

    Itr, Etr, Ptr, Qtr = I[~is_test], E[~is_test], P[~is_test], Q[~is_test]
    Ite, Ete, Pte, Qte = I[is_test], E[is_test], P[is_test], Q[is_test]
    print(f"=== Neural ODE — train {(~is_test).sum()} / test {is_test.sum()} trayectorias ===")
    print(f"    correccion g_phi: {'ON (gray-box)' if USE_CORRECTION else 'OFF (identificacion pura)'}")

    model = GrayBoxWC(fixed, {k: W_INIT for k in IDENT},
                      learnable_weights=LEARN_WEIGHTS, use_correction=USE_CORRECTION)

    x0, Pw, Qw, tgt = make_windows(Itr, Etr, Ptr, Qtr, WINDOW)
    print(f"    multiple shooting: {x0.shape[0]} ventanas de {WINDOW} pasos")

    # Optimizador: pesos con lr alto, correccion con lr normal.
    groups = []
    if LEARN_WEIGHTS:
        groups.append({"params": [model.raw_w], "lr": LR_W})
    if USE_CORRECTION:
        groups.append({"params": model.g.parameters(), "lr": LR_NET})
    opt = torch.optim.Adam(groups)

    for ep in range(EPOCHS):
        opt.zero_grad()
        pred = rollout(model, x0, Pw, Qw, dt)        # (W+1,Nw,2)
        loss = ((pred - tgt) ** 2).mean()
        loss.backward()
        opt.step()
        if ep % LOG_EVERY == 0 or ep == EPOCHS - 1:
            w = model.weights_dict()
            extra = " ".join(f"{k}={w[k]:.3f}" for k in IDENT)
            print(f"  {ep:5d} | loss={loss.item():.3e} | {extra}")

    # --- Refinamiento L-BFGS ---
    if LBFGS_STEPS > 0:
        params = ([model.raw_w] if LEARN_WEIGHTS else []) + \
                 (list(model.g.parameters()) if USE_CORRECTION else [])
        opt2 = torch.optim.LBFGS(params, lr=1.0, max_iter=20, line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            loss = ((rollout(model, x0, Pw, Qw, dt) - tgt) ** 2).mean()
            loss.backward()
            return loss
        for _ in range(LBFGS_STEPS):
            opt2.step(closure)
        print("  --- L-BFGS aplicado ---")

    # --- Resultados ---
    w = model.weights_dict()
    print("\n=== Pesos identificados (Neural ODE) vs verdaderos ===")
    print(f"  {'peso':6} {'verdadero':>10} {'estimado':>10} {'error %':>10}")
    max_err = 0.0
    for k in IDENT:
        err = 100.0 * abs(w[k] - W_TRUE[k]) / abs(W_TRUE[k])
        max_err = max(max_err, err)
        print(f"  {k:6} {W_TRUE[k]:10.4f} {w[k]:10.4f} {err:9.2f}%")

    mse_tr = open_loop_mse(model, Itr, Etr, Ptr, Qtr, dt)
    mse_te = open_loop_mse(model, Ite, Ete, Pte, Qte, dt)
    print(f"\n  MSE open-loop (rollout completo):  train={mse_tr:.2e}   test(held-out)={mse_te:.2e}")
    print(f"  max error parametrico = {max_err:.2f}%")

    ckpt = OUT_DIR / "models" / "neural_ode.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "fixed": fixed,
                "use_correction": USE_CORRECTION, "learn_weights": LEARN_WEIGHTS}, ckpt)
    print(f"\n  Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
