#!/usr/bin/env python3
# =============================================================================
#  ENTRENAR EL NEURAL ODE — IDENTIFICACION COMPLETA (caso "real")
# =============================================================================
#
#  Igual que train_neural_ode.py, pero NO se le regalan los parametros fisicos
#  del dataset: ademas de los 4 pesos sinapticos, aprende tambien
#      te, ti, ae, ai, thetae, thetai   (10 parametros en total).
#  ke,ki NO se entrenan: se derivan de ae,thetae / ai,thetai dentro del modelo
#  para conservar el equilibrio en reposo (E=I=0).
#
#  ARRANQUE IGNORANTE: todo (pesos y parametros fisicos) parte de 1.0. La red NO
#  ve los valores verdaderos; solo ve las trayectorias (I,E) y los estimulos P,Q.
#  Asi se ve como responde cuando no cuenta con los parametros del dataset, que es
#  lo que pasaria en un caso real (solo medis la actividad y el estimulo aplicado).
#
#  Es mas dificil que identificar solo los pesos: 10 parametros, mas no-convexo y
#  mal condicionado (la ganancia ae multiplica adentro a los pesos -> acoplamiento;
#  lo que rompe la ambiguedad de escala es que P,Q son conocidos).
#
#  USO:  python scripts/train_neural_ode_full.py
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.neural_ode import GrayBoxWC, rollout

# #############################################################################
# ##   ZONA EDITABLE                                                         ##
# #############################################################################

DATA_PATH = Path("data/processed/control/multi_dataset.npz")

INIT_VALUE = 1.0     # arranque ignorante: TODO (pesos y fisicos) parte de aca

WINDOW   = 100       # ventana de multiple shooting (pasos) -> ~5 ms con dt=0.05
EPOCHS   = 2000      # converge ~epoca 1250 (ver log); 2000 deja margen + L-BFGS
LR_W     = 5e-2      # lr de los pesos
LR_PHYS  = 2e-2      # lr de los parametros fisicos (te,ti,ae,ai,thetae,thetai)
LOG_EVERY = 250
LBFGS_STEPS = 80

OUT_DIR = Path("results")

# #############################################################################
# ##   FIN ZONA EDITABLE                                                     ##
# #############################################################################

WEIGHTS = ("wEE", "wEI", "wIE", "wII")
PHYS    = ("te", "ti", "ae", "ai", "thetae", "thetai")
ALL_P   = WEIGHTS + PHYS


def load():
    d = np.load(DATA_PATH, allow_pickle=True)
    # Valores VERDADEROS del dataset: SOLO para reportar el error al final,
    # nunca para inicializar ni entrenar.
    true = {k: float(d[k]) for k in ALL_P}
    return d, true, float(d["dt"])


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
    d, true, dt = load()
    is_test = d["is_test"].astype(bool)
    I, E, P, Q = d["I"], d["E"], d["P"], d["Q"]

    Itr, Etr, Ptr, Qtr = I[~is_test], E[~is_test], P[~is_test], Q[~is_test]
    Ite, Ete, Pte, Qte = I[is_test], E[is_test], P[is_test], Q[is_test]
    print(f"=== Neural ODE — IDENTIFICACION COMPLETA (10 parametros) ===")
    print(f"    train {(~is_test).sum()} / test {is_test.sum()} trayectorias")
    print(f"    arranque ignorante: TODO inicia en {INIT_VALUE} (no se usan los valores verdaderos)")

    # Init ignorante: ni pesos ni fisicos usan el valor del dataset.
    init = {k: INIT_VALUE for k in ALL_P}
    model = GrayBoxWC(
        init, {k: INIT_VALUE for k in WEIGHTS},
        learnable_weights=True, use_correction=False, learnable_params=True,
    )

    x0, Pw, Qw, tgt = make_windows(Itr, Etr, Ptr, Qtr, WINDOW)
    print(f"    multiple shooting: {x0.shape[0]} ventanas de {WINDOW} pasos")

    phys_raw = [getattr(model, f"raw_{k}") for k in PHYS]
    opt = torch.optim.Adam([
        {"params": [model.raw_w], "lr": LR_W},
        {"params": phys_raw, "lr": LR_PHYS},
    ])

    for ep in range(EPOCHS):
        opt.zero_grad()
        pred = rollout(model, x0, Pw, Qw, dt)        # (W+1,Nw,2)
        loss = ((pred - tgt) ** 2).mean()
        loss.backward()
        opt.step()
        if ep % LOG_EVERY == 0 or ep == EPOCHS - 1:
            p = model.params_dict()
            extra = " ".join(f"{k}={p[k]:.2f}" for k in ALL_P)
            print(f"  {ep:5d} | loss={loss.item():.3e} | {extra}")

    # --- Refinamiento L-BFGS sobre todos los parametros ---
    if LBFGS_STEPS > 0:
        params = [model.raw_w] + phys_raw
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
    p = model.params_dict()
    print("\n=== Parametros identificados (Neural ODE, sin ver los verdaderos) ===")
    print(f"  {'param':8} {'verdadero':>10} {'estimado':>10} {'error %':>10}")
    max_err = 0.0
    for k in ALL_P:
        err = 100.0 * abs(p[k] - true[k]) / abs(true[k])
        max_err = max(max_err, err)
        marca = "  <-- peso" if k in WEIGHTS else ""
        print(f"  {k:8} {true[k]:10.4f} {p[k]:10.4f} {err:9.2f}%{marca}")

    mse_tr = open_loop_mse(model, Itr, Etr, Ptr, Qtr, dt)
    mse_te = open_loop_mse(model, Ite, Ete, Pte, Qte, dt)
    print(f"\n  MSE open-loop (rollout completo):  train={mse_tr:.2e}   test(held-out)={mse_te:.2e}")
    print(f"  max error parametrico = {max_err:.2f}%")

    ckpt = OUT_DIR / "models" / "neural_ode_full.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "init": init,
                "learn_params": True, "learn_weights": True}, ckpt)
    print(f"\n  Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
