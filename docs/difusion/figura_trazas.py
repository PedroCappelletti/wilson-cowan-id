#!/usr/bin/env python3
# Figura de difusión con datos: lazo cerrado theta-gamma sobre la planta con
# actuador, la planta contra el modelo con el actuador en su estructura y
# contra una red con historia del comando. Pocas palabras, para el público
# general. Sale de results/trazas_lazo_referencias_amplitud.npz.

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
z = np.load(RAIZ / "results" / "trazas_lazo_referencias_amplitud.npz", allow_pickle=True)
nombres = list(z["nombres"])
i_sim, i_lag, i_red = (nombres.index(n) for n in ("simulador", "estado de filtro", "historia con comando actual"))
t = z[f"0_{i_sim}_t"]
m = (t >= 1150) & (t <= 1500)

plt.rcParams.update({"font.family": "sans-serif", "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False})
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(t[m], z[f"0_{i_sim}_rE"][m], color="#b9b6ae", lw=2.2, label="patrón buscado")
ax.plot(t[m], z[f"0_{i_sim}_E"][m], color="#1f1f1e", lw=1.6, label="planta")
ax.plot(t[m], z[f"0_{i_lag}_E"][m], color="#C8590E", lw=1.4, ls=(0, (4, 2)),
        label="modelo con el actuador")
ax.plot(t[m], z[f"0_{i_red}_E"][m], color="#1F5F99", lw=1.2, alpha=0.85,
        label="modelo con red neuronal")
ax.set_yticks([])
ax.set_xlabel("tiempo (ms)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False, fontsize=11.5)
fig.tight_layout()
out = Path(__file__).with_name("figura_trazas")
fig.savefig(out.with_suffix(".pdf"))
fig.savefig(out.with_suffix(".png"), dpi=300)
print(out.with_suffix(".png"))
