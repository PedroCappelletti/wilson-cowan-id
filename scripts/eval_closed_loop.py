#!/usr/bin/env python3
# =============================================================================
#  LAZO CERRADO — validacion del port del controlador (V0: planta verdadera)
# =============================================================================
#
#  Corre el controlador IMC (port de simulador_wilson_cowan_con_control.m) contra
#  la planta Wilson-Cowan VERDADERA y grafica el seguimiento de las referencias
#  theta-gamma. Sirve para verificar que el port a Python reproduce el MATLAB.
#
#  Mas adelante: cambiar make_true_plant por make_neural_plant(modelo) para correr
#  el lazo con el modelo APRENDIDO como planta (la extension del OE3), y/o construir
#  el controlador con los pesos IDENTIFICADOS para medir la degradacion.
#
#  USO:  python scripts/eval_closed_loop.py
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.wilson_cowan import WilsonCowanParams
from src.neural_ode import (
    IMCController, make_true_plant, make_neural_plant, simulate_closed_loop,
    theta_gamma_refs, GrayBoxWC,
)

# #############################################################################
# ##   ZONA EDITABLE                                                         ##
# #############################################################################

PARAMS = WilsonCowanParams()   # parametros verdaderos

# Pesos que USA el controlador en su cancelacion.
#   - verdaderos        -> caso nominal (deberia seguir perfecto).
#   - identificados θ̂   -> para medir la degradacion por error de identificacion.
W_CTRL = {"wEE": PARAMS.wEE, "wEI": PARAMS.wEI, "wIE": PARAMS.wIE, "wII": PARAMS.wII}

T_SPAN = (0.0, 50.0)   # ms (igual que el MATLAB)
DT     = 0.005         # paso de integracion
FREQ   = 120.0         # Hz de las referencias (gamma)

NEURAL_CKPT = Path("results/models/neural_ode.pt")  # planta aprendida (si existe)

OUT_DIR = Path("results")

# #############################################################################
# ##   FIN ZONA EDITABLE                                                     ##
# #############################################################################


def _rmse(sol):
    n0 = len(sol["t"]) // 5   # descarta transitorio inicial
    rI = float(np.sqrt(np.mean((sol["I"][n0:] - sol["rI"][n0:]) ** 2)))
    rE = float(np.sqrt(np.mean((sol["E"][n0:] - sol["rE"][n0:]) ** 2)))
    return rI, rE


def _load_neural():
    """Devuelve (planta_callable, pesos_aprendidos) o (None, None) si no hay checkpoint."""
    if not NEURAL_CKPT.exists():
        return None, None
    ck = torch.load(NEURAL_CKPT, weights_only=False)
    model = GrayBoxWC(ck["fixed"], {k: 1.0 for k in GrayBoxWC.WEIGHTS},
                      learnable_weights=ck["learn_weights"], use_correction=ck["use_correction"])
    model.load_state_dict(ck["state"])
    model.eval()
    return make_neural_plant(model), model.weights_dict()


def main():
    p = PARAMS
    fixed = {
        "te": p.te, "ti": p.ti, "ae": p.ae, "ai": p.ai,
        "thetae": p.thetae, "thetai": p.thetai, "ke": p.ke, "ki": p.ki,
    }
    ctrl = IMCController(fixed, W_CTRL)
    refs = theta_gamma_refs(freq_hz=FREQ, time_in_ms=True)

    # Planta verdadera (referencia) y planta aprendida (Neural ODE), MISMO controlador.
    plant_true = make_true_plant(fixed, {"wEE": p.wEE, "wEI": p.wEI, "wIE": p.wIE, "wII": p.wII})
    sol_true = simulate_closed_loop(plant_true, ctrl, refs, t_span=T_SPAN, dt=DT)
    rIt, rEt = _rmse(sol_true)
    print(f"Planta VERDADERA  — RMSE seguimiento:  I={rIt:.3e}  E={rEt:.3e}")

    sol_neural = None
    plant_neural, w_hat = _load_neural()
    if plant_neural is not None:
        sol_neural = simulate_closed_loop(plant_neural, ctrl, refs, t_span=T_SPAN, dt=DT)
        rIn, rEn = _rmse(sol_neural)
        print(f"Planta APRENDIDA  — RMSE seguimiento:  I={rIn:.3e}  E={rEn:.3e}")
        print("  -> Si ~ la verdadera, el modelo aprendido es una planta controlable valida.")

        # OE3 honesto: controlador construido con los pesos APRENDIDOS (θ̂) corriendo
        # sobre la planta VERDADERA -> cuanto degrada el control el error de identificacion.
        ctrl_hat = IMCController(fixed, w_hat)
        sol_hat = simulate_closed_loop(plant_true, ctrl_hat, refs, t_span=T_SPAN, dt=DT)
        rIh, rEh = _rmse(sol_hat)
        print(f"Controlador θ̂ s/ planta VERDADERA — RMSE: I={rIh:.3e}  E={rEh:.3e}")
        print("  (pesos θ̂ del controlador: " + " ".join(f"{k}={w_hat[k]:.3f}" for k in w_hat) + ")")
    else:
        print("(No hay checkpoint del Neural ODE; corro solo la planta verdadera.)")

    path = OUT_DIR / "figures" / "closed_loop_compare.png"
    _plot(sol_true, sol_neural, path)
    print(f"Figura: {path}")


def _plot(sol_t, sol_n, path):
    import matplotlib.pyplot as plt
    t = sol_t["t"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)

    axes[0].plot(t, sol_t["rI"], "k--", lw=1.0, label="ref rI")
    axes[0].plot(t, sol_t["I"], lw=1.3, label="I — planta verdadera")
    if sol_n is not None:
        axes[0].plot(sol_n["t"], sol_n["I"], lw=1.1, ls=":", color="#d62728",
                     label="I — planta aprendida")
    axes[0].set_ylabel("I"); axes[0].legend(loc="upper right", fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Lazo cerrado: controlador sobre planta verdadera vs. aprendida")

    axes[1].plot(t, sol_t["rE"], "k--", lw=1.0, label="ref rE")
    axes[1].plot(t, sol_t["E"], lw=1.3, label="E — planta verdadera")
    if sol_n is not None:
        axes[1].plot(sol_n["t"], sol_n["E"], lw=1.1, ls=":", color="#d62728",
                     label="E — planta aprendida")
    axes[1].set_ylabel("E"); axes[1].legend(loc="upper right", fontsize=8); axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, sol_t["y"], lw=1.3, label="y — verdadera")
    if sol_n is not None:
        axes[2].plot(sol_n["t"], sol_n["y"], lw=1.1, ls=":", color="#d62728", label="y — aprendida")
    axes[2].set_ylabel("y = E - I"); axes[2].set_xlabel("tiempo (ms)")
    axes[2].legend(loc="upper right", fontsize=8); axes[2].grid(True, alpha=0.3)
    axes[2].set_title("Salida (potencial)")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
