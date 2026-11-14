#!/usr/bin/env python3
# Figuras de seguimiento en lazo cerrado: referencia, lazo con el simulador y
# lazo con cada modelo, para las cuatro referencias. Lee
# results/trazas_lazo_referencias.npz y escribe PDF en docs/figuras_lazo/.
#
#   python scripts/figuras_lazo_referencias.py

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))
OUT = RAIZ / "docs" / "figuras_lazo"
OUT.mkdir(parents=True, exist_ok=True)

TINTA, TINTA2, REF = "#1f1f1e", "#5c5c58", "#b9b8b2"
COLORES = {"white-box": "#9a9a96", "agnóstica": "#1baf7a", "historia con comando actual": "#eb6834",
           "historia sin comando actual": "#e87ba4", "filtro 8 canales": "#6250d6",
           "estado de filtro, recorte en ±10": "#eda100", "estado de filtro, recorte en ±3": "#2a78d6",
           "estado de filtro x0, recorte en ±3": "#2a78d6", "estado de filtro": "#2a78d6",
           "black-box": "#e34948", "black-box con historia": "#b5651d"}
AMPLITUD = "--amplitud" in sys.argv
if AMPLITUD:
    OUT = RAIZ / "docs" / "figuras_amplitud"
# ventana que se muestra de cada referencia, en ms
VENTANA = {"theta2s": (1150, 1500), "gamma50": (0, 50), "theta_rango": (0, 500), "gamma_rango": (0, 250)}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 13,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": TINTA2,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
})


def coma(v, _=None):
    return f"{v:.10g}".replace(".", ",").replace("-", "−")


def figura(z, i, ref, var):
    nombres = list(z["nombres"])
    sim = nombres.index("simulador")
    t = z[f"{i}_{sim}_t"]
    a, b = np.searchsorted(t, VENTANA[ref][0]), np.searchsorted(t, VENTANA[ref][1])
    modelos = [n for n in nombres if n != "simulador"]
    fig, axs = plt.subplots(4, 2, figsize=(11, 9.6), sharex=True, sharey=True)
    for ax, nom in zip(axs.T.flat, modelos):
        j = nombres.index(nom)
        ax.plot(t[a:b], z[f"{i}_{sim}_r{var}"][a:b], color=REF, lw=3.2)
        ax.plot(t[a:b], z[f"{i}_{sim}_{var}"][a:b], color=TINTA, lw=1.2)
        ax.plot(t[a:b], z[f"{i}_{j}_{var}"][a:b], color=COLORES[nom], lw=1.4, ls="--")
        r = z[f"{i}_{j}_rmse"][2 if var == "E" else 1]
        ax.set_title(f"{nom}   RMSE {var} {coma(round(float(r), 3))}", loc="left", fontsize=12.5, pad=3)
        ax.xaxis.set_major_formatter(FuncFormatter(coma))
        ax.yaxis.set_major_formatter(FuncFormatter(coma))
    for ax in axs[:, 0]:
        ax.set_ylabel(var)
    for ax in axs[-1]:
        ax.set_xlabel("tiempo (ms)")
    rs = z[f"{i}_{sim}_rmse"][2 if var == "E" else 1]
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.5)
    fig.legend([Line2D([], [], color=REF, lw=3.2), Line2D([], [], color=TINTA, lw=1.2),
                Line2D([], [], color=TINTA2, lw=1.4, ls="--")],
               ["referencia", f"lazo con el simulador (RMSE {var} {coma(round(float(rs), 3))})",
                "lazo con el modelo del panel"], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(OUT / f"{ref}_{var}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{ref}_{var}.png", dpi=90, bbox_inches="tight")
    plt.close(fig)


def figura_theta():
    """La referencia theta-gamma entera: theta, ventana y rE."""
    from src.neural_ode.closed_loop_ext import theta_gamma_ventana_refs
    refs, (t, theta, ventana) = theta_gamma_ventana_refs()
    rE = np.array([refs(tt)[1] for tt in t])
    fig, axs = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True)
    axs[0].plot(t, theta, color=TINTA2, lw=1.2)
    axs[0].fill_between(t, -1.05, 1.05, where=ventana, color="#cfe0f6", lw=0)
    axs[0].set_ylabel("theta")
    axs[1].plot(t, rE, color=TINTA, lw=0.8)
    axs[1].axvspan(*VENTANA["theta2s"], color="#ecebe7", zorder=0)
    axs[1].set_ylabel("$r_E$")
    axs[1].set_xlabel("tiempo (ms)")
    for ax in axs:
        ax.xaxis.set_major_formatter(FuncFormatter(coma))
        ax.yaxis.set_major_formatter(FuncFormatter(coma))
    fig.tight_layout()
    fig.savefig(OUT / "theta2s_referencia.pdf", bbox_inches="tight")
    fig.savefig(OUT / "theta2s_referencia.png", dpi=90, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    z = np.load(RAIZ / "results" / ("trazas_lazo_referencias_amplitud.npz" if AMPLITUD
                                     else "trazas_lazo_referencias.npz"))
    for i, ref in enumerate(z["refs"]):
        for var in ("E", "I"):
            figura(z, i, str(ref), var)
    if not AMPLITUD:
        figura_theta()
    print(f"-> {OUT}")
