#!/usr/bin/env python3
# =============================================================================
#  EVALUACION DEL MODELO IDENTIFICADO COMPLETO (10 parametros)
#  -> Lazo ABIERTO (rollout vs. real) + lazo CERRADO (controlador IMC)
# =============================================================================
#
#  Toma el checkpoint de la identificacion completa (results/models/neural_ode_full.pt,
#  los 10 parametros aprendidos desde arranque ignorante) y lo evalua de dos formas:
#
#   1) LAZO ABIERTO: integra el modelo aprendido sobre cada trayectoria del dataset
#      (sin resets) y lo compara contra la real -> mide si el modelo reproduce la
#      dinamica. Se grafica el chirp held-out (estimulo nunca visto).
#
#   2) LAZO CERRADO: enchufa el modelo como PLANTA del controlador IMC y, ademas,
#      construye el controlador con los parametros IDENTIFICADOS (θ̂, fisicos + pesos)
#      corriendo sobre la planta VERDADERA -> mide cuanto degrada el control el error
#      de identificacion (validacion orientada al control, OE3).
#
#  USO:  python scripts/eval_full_identified.py
# =============================================================================

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.wilson_cowan import WilsonCowanParams
from src.neural_ode import (
    GrayBoxWC, rollout,
    IMCController, make_true_plant, make_neural_plant, simulate_closed_loop,
    theta_gamma_refs,
)

DATA_PATH   = Path("data/processed/control/multi_dataset.npz")
NEURAL_CKPT = Path("results/models/neural_ode_full.pt")
OUT_DIR     = Path("results")

T_SPAN = (0.0, 50.0)   # ms (igual que el MATLAB del controlador)
DT_CL  = 0.005         # paso del lazo cerrado
FREQ   = 120.0         # Hz de las referencias theta-gamma


# -----------------------------------------------------------------------------
#  Carga del modelo identificado completo.
# -----------------------------------------------------------------------------
def load_full_model():
    ck = torch.load(NEURAL_CKPT, weights_only=False)
    model = GrayBoxWC(
        ck["init"], {k: 1.0 for k in GrayBoxWC.WEIGHTS},
        learnable_weights=True, use_correction=False, learnable_params=True,
    )
    model.load_state_dict(ck["state"])
    model.eval()
    return model


def fixed_from_params(p: dict) -> dict:
    """Arma el dict 'fixed' (con ke,ki derivados) a partir de un params_dict."""
    ae, ai, the, thi = p["ae"], p["ai"], p["thetae"], p["thetai"]
    return {
        "te": p["te"], "ti": p["ti"], "ae": ae, "ai": ai,
        "thetae": the, "thetai": thi,
        "ke": 1.0 / (1.0 + math.exp(ae * the)),
        "ki": 1.0 / (1.0 + math.exp(ai * thi)),
    }


# =============================================================================
#  1) LAZO ABIERTO
# =============================================================================
@torch.no_grad()
def open_loop(model):
    d = np.load(DATA_PATH, allow_pickle=True)
    dt = float(d["dt"])
    I, E, P, Q = d["I"], d["E"], d["P"], d["Q"]
    labels = [str(x) for x in d["labels"]]
    is_test = d["is_test"].astype(bool)

    rows = []
    trajs = {}
    for s in range(I.shape[0]):
        T = I.shape[1]
        x0 = torch.tensor([[I[s, 0], E[s, 0]]], dtype=torch.float32)
        Ps = torch.tensor(P[s], dtype=torch.float32).reshape(T, 1, 1)
        Qs = torch.tensor(Q[s], dtype=torch.float32).reshape(T, 1, 1)
        pred = rollout(model, x0, Ps[:-1], Qs[:-1], dt)[:, 0, :].numpy()   # (T,2)
        real = np.stack([I[s], E[s]], axis=1)
        mse = float(((pred - real) ** 2).mean())
        rows.append((labels[s], bool(is_test[s]), mse))
        trajs[labels[s]] = (d["t"] if "t" in d else np.arange(T) * dt, real, pred)

    mse_tr = np.mean([m for _, t, m in rows if not t])
    mse_te = np.mean([m for _, t, m in rows if t])
    return rows, mse_tr, mse_te, trajs


def plot_open_loop(trajs, label, path):
    import matplotlib.pyplot as plt
    t, real, pred = trajs[label]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    for ax, j, name in zip(axes, (0, 1), ("I (inhibitoria)", "E (excitatoria)")):
        ax.plot(t, real[:, j], lw=1.4, label=f"{name} — real")
        ax.plot(t, pred[:, j], lw=1.1, ls=":", color="#d62728", label=f"{name} — modelo")
        ax.set_ylabel(name); ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.3)
    axes[0].set_title(f"Lazo abierto — modelo identificado vs. real  ({label}, held-out)")
    axes[-1].set_xlabel("tiempo (s)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)


# =============================================================================
#  2) LAZO CERRADO
# =============================================================================
def _rmse(sol):
    n0 = len(sol["t"]) // 5   # descarta transitorio inicial
    rI = float(np.sqrt(np.mean((sol["I"][n0:] - sol["rI"][n0:]) ** 2)))
    rE = float(np.sqrt(np.mean((sol["E"][n0:] - sol["rE"][n0:]) ** 2)))
    return rI, rE


def closed_loop(model):
    p = WilsonCowanParams()
    fixed_true = {
        "te": p.te, "ti": p.ti, "ae": p.ae, "ai": p.ai,
        "thetae": p.thetae, "thetai": p.thetai, "ke": p.ke, "ki": p.ki,
    }
    w_true = {"wEE": p.wEE, "wEI": p.wEI, "wIE": p.wIE, "wII": p.wII}

    phat = model.params_dict()
    fixed_hat = fixed_from_params(phat)
    w_hat = {k: phat[k] for k in GrayBoxWC.WEIGHTS}

    refs = theta_gamma_refs(freq_hz=FREQ, time_in_ms=True)

    # Plantas.
    plant_true = make_true_plant(fixed_true, w_true)
    plant_neural = make_neural_plant(model)

    # Controladores.
    ctrl_true = IMCController(fixed_true, w_true)   # nominal (todo verdadero)
    ctrl_hat = IMCController(fixed_hat, w_hat)      # construido con θ̂ (todo identificado)

    kw = dict(t_span=T_SPAN, dt=DT_CL)
    sol_nom = simulate_closed_loop(plant_true, ctrl_true, refs, **kw)         # referencia
    sol_neural = simulate_closed_loop(plant_neural, ctrl_true, refs, **kw)    # planta aprendida
    sol_hat = simulate_closed_loop(plant_true, ctrl_hat, refs, **kw)          # controlador θ̂

    res = {
        "nominal (ctrl verdadero / planta verdadera)": (sol_nom, _rmse(sol_nom)),
        "planta APRENDIDA (ctrl verdadero)":           (sol_neural, _rmse(sol_neural)),
        "controlador θ̂ / planta verdadera":           (sol_hat, _rmse(sol_hat)),
    }
    return res, w_hat, phat


def plot_closed_loop(res, path):
    import matplotlib.pyplot as plt
    sol_nom = res["nominal (ctrl verdadero / planta verdadera)"][0]
    sol_neu = res["planta APRENDIDA (ctrl verdadero)"][0]
    sol_hat = res["controlador θ̂ / planta verdadera"][0]
    t = sol_nom["t"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axes[0].plot(t, sol_nom["rI"], "k--", lw=1.0, label="ref rI")
    axes[0].plot(t, sol_nom["I"], lw=1.4, label="nominal")
    axes[0].plot(sol_neu["t"], sol_neu["I"], lw=1.0, ls=":", color="#d62728", label="planta aprendida")
    axes[0].plot(sol_hat["t"], sol_hat["I"], lw=1.0, ls="-.", color="#2ca02c", label="controlador θ̂")
    axes[0].set_ylabel("I"); axes[0].legend(loc="upper right", fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Lazo cerrado — seguimiento theta-gamma (modelo identificado completo)")

    axes[1].plot(t, sol_nom["rE"], "k--", lw=1.0, label="ref rE")
    axes[1].plot(t, sol_nom["E"], lw=1.4, label="nominal")
    axes[1].plot(sol_neu["t"], sol_neu["E"], lw=1.0, ls=":", color="#d62728", label="planta aprendida")
    axes[1].plot(sol_hat["t"], sol_hat["E"], lw=1.0, ls="-.", color="#2ca02c", label="controlador θ̂")
    axes[1].set_ylabel("E"); axes[1].legend(loc="upper right", fontsize=8); axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("tiempo (ms)")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)


# =============================================================================
def main():
    model = load_full_model()

    # --- Lazo abierto ---
    print("=== LAZO ABIERTO (modelo identificado completo) ===")
    rows, mse_tr, mse_te, trajs = open_loop(model)
    print(f"  {'estimulo':22} {'split':6} {'MSE':>12}")
    for lab, tst, mse in rows:
        print(f"  {lab:22} {'TEST' if tst else 'train':6} {mse:12.3e}")
    print(f"  --> MSE medio  train={mse_tr:.3e}   test(held-out)={mse_te:.3e}")
    ol_path = OUT_DIR / "figures" / "open_loop_full.png"
    if "chirp" in trajs:
        plot_open_loop(trajs, "chirp", ol_path)
        print(f"  Figura: {ol_path}")

    # --- Lazo cerrado ---
    print("\n=== LAZO CERRADO (controlador IMC) ===")
    res, w_hat, phat = closed_loop(model)
    print(f"  {'caso':45} {'RMSE I':>10} {'RMSE E':>10}")
    for name, (_, (rI, rE)) in res.items():
        print(f"  {name:45} {rI:10.3e} {rE:10.3e}")
    cl_path = OUT_DIR / "figures" / "closed_loop_full.png"
    plot_closed_loop(res, cl_path)
    print(f"  Figura: {cl_path}")

    return rows, mse_tr, mse_te, res


if __name__ == "__main__":
    main()
