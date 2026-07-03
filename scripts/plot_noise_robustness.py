#!/usr/bin/env python3
# =============================================================================
#  FIGURA — Robustez al ruido: error por parámetro (10 params) vs σ
#  Datos de docs/robustez_ruido_identificacion_completa.md (noise_full_sweep).
#  (regenera la figura que corrió otra persona; no reentrena)
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

FIG = Path("results/figures/noise_param_error.png")
SIG = [0.0, 0.01, 0.05, 0.10]

# Error θ̂ (%) por parámetro y σ (tabla del informe de robustez).
ERR = {
    "wEE": [0.05, 0.78, 4.78, 3.33],
    "wEI": [0.08, 0.92, 1.23, 4.25],
    "wIE": [0.14, 0.19, 0.57, 1.18],
    "wII": [0.83, 4.53, 19.32, 41.34],
    "te":  [0.16, 0.02, 2.92, 5.44],   # 0.00 -> 0.02 (piso para log)
    "ti":  [0.68, 1.44, 4.55, 15.88],
    "ae":  [0.03, 0.72, 7.54, 9.05],
    "ai":  [0.35, 0.92, 4.76, 11.53],
    "thetae": [0.03, 0.35, 3.53, 4.05],
    "thetai": [0.03, 0.48, 2.18, 5.21],
}
MAX4 = [0.14, 0.65, 1.19, 8.92]   # máx identificando SOLO los 4 pesos

fig, ax = plt.subplots(figsize=(8.5, 5.4))
x = np.array(SIG)

# Resto de params: líneas finas grises.
for k, v in ERR.items():
    if k in ("wII", "ti"):
        continue
    ax.plot(x, np.maximum(v, 0.02), "-", color="#bbbbbb", lw=1.2, zorder=1)
ax.plot([], [], "-", color="#bbbbbb", lw=1.2, label="otros params (10)")

# ti: segundo peor.
ax.plot(x, ERR["ti"], "s-", color="#ff9800", lw=2, label="ti (2º peor)", zorder=3)
# wII = máx de los 10.
ax.plot(x, ERR["wII"], "o-", color="#d62728", lw=3, label="wII  (= máx 10 params)", zorder=5)
# máx solo 4 pesos.
ax.plot(x, MAX4, "^--", color="#1f4e79", lw=2.5, label="máx si se identifican SOLO 4 pesos", zorder=4)

ax.annotate(f"41.3 %", (0.10, 41.34), color="#d62728", fontsize=11, fontweight="bold",
            xytext=(-4, 8), textcoords="offset points", ha="right")
ax.annotate(f"8.9 %", (0.10, 8.92), color="#1f4e79", fontsize=10, fontweight="bold",
            xytext=(-4, -14), textcoords="offset points", ha="right")

ax.set_yscale("log")
ax.set_xlabel("nivel de ruido  σ")
ax.set_ylabel("error de θ̂  (%)  —  escala log")
ax.set_title("Robustez al ruido de la identificación completa (10 parámetros)\n"
             "la FIM predijo que wII es el cuello de botella → el ruido lo confirma")
ax.set_xticks(SIG)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=140, bbox_inches="tight"); plt.close(fig)
print(f"Figura: {FIG}")
