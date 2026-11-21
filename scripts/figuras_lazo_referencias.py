#!/usr/bin/env python3
# Figuras de seguimiento en lazo cerrado para las cuatro referencias. Lee
# results/trazas_lazo_referencias.npz y escribe PDF en docs/figuras_lazo/.
#
# Cada panel muestra la referencia punteada y la respuesta de un solo lazo en
# línea llena. El lazo con el simulador va en su propia figura y no se
# superpone a los modelos (pedido de Ralf, 17-11). En theta-gamma se muestran,
# lado a lado, el principio de la corrida y el tramo de 1150 a 1500 ms.
#
#   python scripts/figuras_lazo_referencias.py

from __future__ import annotations

import sys
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from nombres_modelos import mostrar  # noqa: E402
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
# tramos que se muestran de cada referencia, en ms. Con dos tramos van lado a lado.
TRAMOS = {"theta2s": [(0, 350), (1150, 1500)], "gamma50": [(0, 50)], "theta_rango": [(0, 500)],
          "gamma_rango": [(0, 250)]}
# modelos por figura cuando cada modelo ocupa una fila
POR_FIGURA = 4

plt.rcParams.update({
    "font.family": "Arial", "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 13,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": TINTA2,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
})


def coma(v, _=None):
    return f"{v:.10g}".replace(".", ",").replace("-", "−")


def _rmse(z, i, j, var):
    return coma(round(float(z[f"{i}_{j}_rmse"][2 if var == "E" else 1]), 3))


def _panel(ax, z, i, j, var, tramo, color):
    """Respuesta del lazo j en línea llena y la referencia punteada encima, para que
    se vea también donde las dos coinciden."""
    t = z[f"{i}_{j}_t"]
    a, b = np.searchsorted(t, tramo[0]), np.searchsorted(t, tramo[1])
    ax.plot(t[a:b], z[f"{i}_{j}_{var}"][a:b], color=color, lw=1.5)
    ax.plot(t[a:b], z[f"{i}_{j}_r{var}"][a:b], color=TINTA, lw=1.1, ls=(0, (3, 2)))
    ax.set_xlim(*tramo)
    ax.xaxis.set_major_formatter(FuncFormatter(coma))
    ax.yaxis.set_major_formatter(FuncFormatter(coma))


def _leyenda(fig, lazo, ref):
    # Las referencias en rango son la respuesta en lazo abierto del Wilson-Cowan;
    # sin la aclaración se leen como un lazo cerrado más.
    etiqueta = "referencia (construida en lazo abierto)" if ref.endswith("_rango") else "referencia"
    fig.legend([Line2D([], [], color=TINTA, lw=1.1, ls=(0, (3, 2))), Line2D([], [], color=TINTA2, lw=1.5)],
               [etiqueta, lazo], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))


def _guardar(fig, nombre):
    fig.savefig(OUT / f"{nombre}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{nombre}.png", dpi=90, bbox_inches="tight")
    plt.close(fig)


def figura_modelos(z, i, ref, var):
    """Un panel por modelo, sin el lazo con el simulador."""
    nombres = list(z["nombres"])
    modelos = [n for n in nombres if n != "simulador"]
    tramos = TRAMOS[ref]
    if len(tramos) == 1:
        filas = (len(modelos) + 1) // 2
        fig, axs = plt.subplots(filas, 2, figsize=(11, 2.4 * filas), sharex=True, sharey=True)
        for ax, nom in zip(axs.T.flat, modelos):
            j = nombres.index(nom)
            _panel(ax, z, i, j, var, tramos[0], COLORES[nom])
            ax.set_title(f"{mostrar(nom)}   RMSE {var} {_rmse(z, i, j, var)}", loc="left", fontsize=12.5, pad=3)
        for ax in axs.T.flat[len(modelos):]:
            ax.set_visible(False)
        for ax in axs[:, 0]:
            ax.set_ylabel(var)
        for ax in axs[-1]:
            ax.set_xlabel("tiempo (ms)")
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / (2.4 * filas)), h_pad=0.5)
        _leyenda(fig, "lazo cerrado con el modelo del panel", ref)
        _guardar(fig, f"{ref}_{var}")
        return
    # Con dos tramos cada modelo es una fila, y los modelos se reparten en varias figuras.
    for parte, k in enumerate(range(0, len(modelos), POR_FIGURA), start=1):
        grupo = modelos[k:k + POR_FIGURA]
        fig, axs = plt.subplots(len(grupo), 2, figsize=(11, 2.4 * len(grupo)), sharey=True,
                                squeeze=False)
        for fila, nom in zip(axs, grupo):
            j = nombres.index(nom)
            for ax, tramo in zip(fila, tramos):
                _panel(ax, z, i, j, var, tramo, COLORES[nom])
            fila[0].set_title(f"{mostrar(nom)}   RMSE {var} {_rmse(z, i, j, var)}", loc="left",
                              fontsize=12.5, pad=3)
            fila[0].set_ylabel(var)
        for ax in axs[-1]:
            ax.set_xlabel("tiempo (ms)")
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / (2.4 * len(grupo))), h_pad=0.6)
        _leyenda(fig, "lazo cerrado con el modelo del panel", ref)
        _guardar(fig, f"{ref}_{var}_{parte}")


def figura_simulador(z, i, ref):
    """El lazo con el simulador solo, E arriba e I abajo."""
    nombres = list(z["nombres"])
    sim = nombres.index("simulador")
    tramos = TRAMOS[ref]
    fig, axs = plt.subplots(2, len(tramos), figsize=(11, 4.8), sharey="row", squeeze=False)
    for fila, var in zip(axs, ("E", "I")):
        for ax, tramo in zip(fila, tramos):
            _panel(ax, z, i, sim, var, tramo, TINTA2)
        fila[0].set_ylabel(var)
        fila[0].set_title(f"RMSE {var} {_rmse(z, i, sim, var)}", loc="left", fontsize=12.5, pad=3)
    for ax in axs[-1]:
        ax.set_xlabel("tiempo (ms)")
    fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=0.6)
    _leyenda(fig, "lazo cerrado con el simulador", ref)
    _guardar(fig, f"{ref}_simulador")


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
    for tramo in TRAMOS["theta2s"]:
        axs[1].axvspan(*tramo, color="#ecebe7", zorder=0)
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
        figura_simulador(z, i, str(ref))
        for var in ("E", "I"):
            figura_modelos(z, i, str(ref), var)
    if not AMPLITUD:
        figura_theta()
    print(f"-> {OUT}")
