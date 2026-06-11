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

from src.wilson_cowan import WilsonCowanParams
from src.neural_ode import (
    IMCController, make_true_plant, simulate_closed_loop, theta_gamma_refs,
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

OUT_DIR = Path("results")

# #############################################################################
# ##   FIN ZONA EDITABLE                                                     ##
# #############################################################################


def main():
    p = PARAMS
    fixed = {
        "te": p.te, "ti": p.ti, "ae": p.ae, "ai": p.ai,
        "thetae": p.thetae, "thetai": p.thetai, "ke": p.ke, "ki": p.ki,
    }

    plant = make_true_plant(fixed, {"wEE": p.wEE, "wEI": p.wEI, "wIE": p.wIE, "wII": p.wII})
    ctrl  = IMCController(fixed, W_CTRL)
    refs  = theta_gamma_refs(freq_hz=FREQ, time_in_ms=True)

    sol = simulate_closed_loop(plant, ctrl, refs, t_span=T_SPAN, dt=DT)

    # --- Error de seguimiento (RMSE) en regimen (descartando el transitorio inicial).
    n0 = len(sol["t"]) // 5
    rmse_I = float(np.sqrt(np.mean((sol["I"][n0:] - sol["rI"][n0:]) ** 2)))
    rmse_E = float(np.sqrt(np.mean((sol["E"][n0:] - sol["rE"][n0:]) ** 2)))
    print(f"Seguimiento (RMSE en regimen):  I = {rmse_I:.3e}   E = {rmse_E:.3e}")

    _plot(sol, OUT_DIR / "figures" / "closed_loop_v0.png")
    print(f"Figura: {OUT_DIR / 'figures' / 'closed_loop_v0.png'}")


def _plot(sol, path):
    import matplotlib.pyplot as plt
    t = sol["t"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t, sol["rI"], "k--", lw=1.0, label="ref rI")
    axes[0].plot(t, sol["I"], lw=1.3, label="I (lazo cerrado)")
    axes[0].set_ylabel("I"); axes[0].legend(loc="upper right"); axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Seguimiento en lazo cerrado — planta verdadera (V0)")

    axes[1].plot(t, sol["rE"], "k--", lw=1.0, label="ref rE")
    axes[1].plot(t, sol["E"], lw=1.3, label="E (lazo cerrado)")
    axes[1].set_ylabel("E"); axes[1].legend(loc="upper right"); axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, sol["y"], lw=1.3, color="#5b2a9e")
    axes[2].set_ylabel("y = E - I"); axes[2].set_xlabel("tiempo (ms)")
    axes[2].grid(True, alpha=0.3); axes[2].set_title("Salida (potencial)")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
