# Figuras del informe final que no salen de otro script del repo:
#  - trazas de theta-gamma (theta2s_E_1, theta2s_E_2): referencia punteada,
#    respuesta llena, un lazo por panel, el simulador en su propia fila;
#    principio (0-350 ms) y tramo del medio (1150-1500 ms).
#  - RMSE de seguimiento contra distancia al lazo con el simulador (rmse_vs_distancia).
#  - recorte en ±3 sin la curva de la tanh ni la franja del rango preliminar (saturacion).
# Salen a results/figures/informe/; al informe se copian a mano.
import inspect
import json
import statistics as st
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
from nombres_modelos import mostrar  # noqa: E402

OUT = REPO / "results" / "figures" / "informe"
OUT.mkdir(exist_ok=True)
TINTA, TINTA2 = "#1f1f1e", "#5c5c58"
COLORES = {"simulador": TINTA2, "white-box": "#9a9a96", "agnóstica": "#1baf7a",
           "historia con comando actual": "#eb6834", "filtro 8 canales": "#6250d6",
           "estado de filtro": "#2a78d6", "black-box": "#e34948", "black-box con historia": "#b5651d"}
TRAMOS = [(0, 350), (1150, 1500)]
plt.rcParams.update({
    "font.family": "Arial", "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": TINTA2,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
})


def coma(v, _=None):
    return f"{v:.10g}".replace(".", ",").replace("-", "−")


def trazas():
    z = np.load(REPO / "results" / "trazas_lazo_referencias_amplitud.npz")
    nombres = [str(n) for n in z["nombres"]]
    i = [str(r) for r in z["refs"]].index("theta2s")
    grupos = [["simulador", "white-box", "agnóstica", "black-box"],
              ["simulador", "historia con comando actual", "filtro 8 canales", "estado de filtro",
               "black-box con historia"]]
    for parte, grupo in enumerate(grupos, start=1):
        fig, axs = plt.subplots(len(grupo), 2, figsize=(8, 1.75 * len(grupo) + 0.5), sharey=True,
                                squeeze=False, gridspec_kw={"width_ratios": [1, 1]})
        for fila, nom in zip(axs, grupo):
            j = nombres.index(nom)
            t = z[f"{i}_{j}_t"]
            for ax, (a0, b0) in zip(fila, TRAMOS):
                a, b = np.searchsorted(t, a0), np.searchsorted(t, b0)
                ax.plot(t[a:b], z[f"{i}_{j}_E"][a:b], color=COLORES[nom], lw=1.4)
                ax.plot(t[a:b], z[f"{i}_{j}_rE"][a:b], color=TINTA, lw=1.0, ls=(0, (3, 2)))
                ax.set_xlim(a0, b0)
                ax.set_xticks(range(a0 if a0 == 0 else a0 + 50, b0, 100))
                ax.xaxis.set_major_formatter(FuncFormatter(coma))
                ax.yaxis.set_major_formatter(FuncFormatter(coma))
            rm = f"{float(z[f'{i}_{j}_rmse'][2]):.3f}".replace(".", ",")
            titulo = "lazo con el simulador" if nom == "simulador" else mostrar(nom)
            fila[0].set_title(f"{titulo}   RMSE E {rm}", loc="left", pad=3)
            fila[0].set_ylabel("E")
        for ax in axs[-1]:
            ax.set_xlabel("tiempo (ms)")
        for fila in axs[:-1]:
            for ax in fila:
                ax.tick_params(labelbottom=False)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / (1.75 * len(grupo) + 0.5)), h_pad=0.4, w_pad=2.0)
        fig.legend([Line2D([], [], color=TINTA, lw=1.0, ls=(0, (3, 2))),
                    Line2D([], [], color=TINTA2, lw=1.4)],
                   ["referencia", "actividad E del lazo del panel"], loc="upper center", ncol=2,
                   bbox_to_anchor=(0.5, 1.0))
        fig.savefig(OUT / f"theta2s_E_{parte}.pdf", bbox_inches="tight")
        fig.savefig(OUT / f"theta2s_E_{parte}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)


def rmse_vs_distancia():
    d = json.load(open(REPO / "results" / "control_amplitud.json", encoding="utf-8"))
    sim = [r["rmse"] for r in d if r["perturbacion"] == "actc" and r["referencia"] == "theta2s"
           and r["planta"] == "simulador"][0]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.axhline(sim, color=TINTA2, lw=1.0, ls=(0, (3, 2)))
    ax.text(0.0005, sim - 0.00035, "lazo con el\nsimulador", color=TINTA2, fontsize=10, va="top")
    desplaz = {"white-box": (6, -10), "agnóstica": (6, 2), "historia con comando actual": (-6, 6),
               "filtro 8 canales": (6, -12), "estado de filtro": (6, -12), "black-box": (6, -4),
               "black-box con historia": (-6, 4)}
    for nom in desplaz:
        rs = [r for r in d if r["perturbacion"] == "actc" and r["referencia"] == "theta2s"
              and r["planta"] == nom and r["ruido"] == "_n01"]
        x, y = st.median(r["dist_x"] for r in rs), st.median(r["rmse"] for r in rs)
        ax.plot(x, y, "o", ms=8, color=COLORES[nom], mec="white", mew=1.2)
        dx, dy = desplaz[nom]
        ax.annotate(mostrar(nom), (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=10,
                    ha="left" if dx > 0 else "right", color=TINTA)
    ax.set_xlim(0, 0.05)
    ax.set_ylim(0.046, 0.065)
    ax.set_yticks([0.05, 0.055, 0.06, 0.065])
    ax.set_xlabel("distancia al lazo con el simulador")
    ax.set_ylabel("RMSE de seguimiento")
    ax.xaxis.set_major_formatter(FuncFormatter(coma))
    ax.yaxis.set_major_formatter(FuncFormatter(coma))
    fig.tight_layout()
    fig.savefig(OUT / "rmse_vs_distancia.pdf", bbox_inches="tight")
    fig.savefig(OUT / "rmse_vs_distancia.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def saturacion():
    import figuras_avances_recorte as f
    f.OUT = OUT
    src = inspect.getsource(f.fig_saturacion)
    quitar = [
        '    ax.axvspan(0, 2, color="#ecebe7", zorder=0)\n',
        '    ax.text(1.0, -5.3, "rango de\\nentrenamiento", ha="center", va="bottom", fontsize=14.5, color=TINTA2)\n',
        '    ax.plot(x, 3 * np.tanh(x / 3), color=AMARILLO, lw=2, label=r"$3\\,\\tanh(P_f/3)$ (antes)")\n',
        '    ax.plot([2], [3 * np.tanh(2 / 3)], "o", color=AMARILLO, ms=7, mec="white", mew=1.5)\n',
        '    ax.annotate("2 → 1,75", (2, 3 * np.tanh(2 / 3)), xytext=(2.6, 0.4), fontsize=14.5,\n'
        '                arrowprops=dict(arrowstyle="-", color=TINTA2, lw=0.8))\n',
    ]
    for q in quitar:
        assert src.count(q) == 1, q
        src = src.replace(q, "")
    src = src.replace(" (ahora)", "")
    src = src.replace('label="sin saturación"', 'label="sin recorte"')
    exec(src, f.__dict__)
    f.fig_saturacion()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    trazas()
    rmse_vs_distancia()
    saturacion()
    print("ok")
