# Figura del rizo para el informe final (rizo_100_120): un lazo por panel
# (simulador a la izquierda, backbone expandido a la derecha), referencia punteada,
# actividad llena, la gamma continua completa (50 ms). La versión de
# figura_rizo_100hz.py es la del README; esta es la del informe.
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
os.chdir(REPO)
import figura_rizo_100hz as R  # noqa: E402

OUT = REPO / "results" / "figures" / "informe"
TINTA, TINTA2 = "#1f1f1e", "#5c5c58"
COL = {"sim": TINTA2, "lag": "#2a78d6"}
plt.rcParams.update({
    "font.family": "Arial", "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": TINTA2,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
})


def coma(v, _=None):
    return f"{v:.10g}".replace(".", ",").replace("-", "−")


OUT.mkdir(parents=True, exist_ok=True)
fig, axs = plt.subplots(2, 2, figsize=(8, 4.6), sharey=True, sharex=True)
for fila, fg in zip(axs, ("120", "100")):
    tr = R.trazas(fg)
    (fs_, fm_), deriva = R.rizo(tr["sim"], tr["lag"])
    print(f"{fg} Hz: rizo simulador {fs_:.1f} Hz, backbone {fm_:.1f} Hz, deriva {deriva:.2f} ciclos")
    for ax, k, titulo in zip(fila, ("sim", "lag"), ("lazo con el simulador", "lazo con el backbone expandido")):
        t = tr[k]["t"]
        ax.plot(t, tr[k]["E"], color=COL[k], lw=1.1)
        ax.plot(t, tr[k]["rE"], color=TINTA, lw=1.0, ls=(0, (3, 2)))
        ax.set_xlim(0, 50)
        ax.set_title(f"{titulo}, gamma de {fg} Hz", loc="left", pad=3)
        ax.xaxis.set_major_formatter(FuncFormatter(coma))
        ax.yaxis.set_major_formatter(FuncFormatter(coma))
    fila[0].set_ylabel("E")
for ax in axs[-1]:
    ax.set_xlabel("tiempo (ms)")
fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=0.6, w_pad=2.0)
fig.legend([Line2D([], [], color=TINTA, lw=1.0, ls=(0, (3, 2))), Line2D([], [], color=TINTA2, lw=1.1)],
           ["referencia", "actividad E del lazo del panel"], loc="upper center", ncol=2,
           bbox_to_anchor=(0.5, 1.0))
fig.savefig(OUT / "rizo_100_120.pdf", bbox_inches="tight")
fig.savefig(OUT / "rizo_100_120.png", dpi=110, bbox_inches="tight")
print("ok")
