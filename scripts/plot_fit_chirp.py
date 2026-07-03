#!/usr/bin/env python3
# =============================================================================
#  FIGURA — ajuste predicho (θ̂ identificados) vs real, trayectoria de test (chirp)
#  θ̂ de docs/identificacion_completa_neural_ode.md (identificación limpia, 10 params).
#  "Predicho" = WC con los parámetros identificados; "real" = WC con los verdaderos.
# =============================================================================
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import matplotlib.pyplot as plt
from src.wilson_cowan import WilsonCowan, WilsonCowanParams, chirp_pulse

FIG = Path("results/figures/fit_chirp_test.png")

true = WilsonCowanParams()
hat = WilsonCowanParams(wEE=6.374, wEI=4.775, wIE=6.048, wII=1.187,
                        te=0.994, ti=1.995, ae=1.204, ai=0.994,
                        thetae=2.798, thetai=4.046)

hz = lambda f: f / 1000.0
P = chirp_pulse(0.8, hz(10), hz(150), 10.0, 190.0)
Q = chirp_pulse(1.0, hz(15), hz(120), 10.0, 190.0)

t = np.linspace(0.0, 200.0, 4000)
st = WilsonCowan(true, P, Q).simulate(t_span=(0.0, 200.0), t_eval=t)
sh = WilsonCowan(hat, P, Q).simulate(t_span=(0.0, 200.0), t_eval=t)

mse_I = float(np.mean((st["I"] - sh["I"]) ** 2))
mse_E = float(np.mean((st["E"] - sh["E"]) ** 2))

fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax[0].plot(t, st["I"], color="#1f4e79", lw=2.2, label="real (θ verdaderos)")
ax[0].plot(t, sh["I"], color="#d62728", lw=1.4, ls="--", label="predicho (θ̂ identificados)")
ax[0].set_ylabel("I (inhibitoria)")
ax[0].set_title(f"Ajuste predicho (θ̂) vs real — trayectoria de test: chirp   "
                f"(MSE_I={mse_I:.1e}, MSE_E={mse_E:.1e})")
ax[0].legend(loc="upper right", fontsize=9); ax[0].grid(True, alpha=0.3)

ax[1].plot(t, st["E"], color="#1f4e79", lw=2.2, label="real")
ax[1].plot(t, sh["E"], color="#d62728", lw=1.4, ls="--", label="predicho")
ax[1].set_ylabel("E (excitatoria)"); ax[1].set_xlabel("tiempo (ms)")
ax[1].legend(loc="upper right", fontsize=9); ax[1].grid(True, alpha=0.3)

fig.tight_layout()
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=140, bbox_inches="tight"); plt.close(fig)
print(f"Figura: {FIG}   MSE_I={mse_I:.2e}  MSE_E={mse_E:.2e}")
