#!/usr/bin/env python3
# Cuánto se degrada cada configuración con el ruido, en dos métricas.
#
# Izquierda, NRMSE de corrida libre, directo. Con ruido ni la trayectoria
# verdadera baja de un piso (2,4-2,6 % con sigma 0,01 y 6,2-6,4 % con 0,05,
# ver la fila de piso en tabla_ruido.py), así que entre niveles de ruido se
# compara con ese piso en mente. Derecha, error de parámetros, que se mide
# contra el beta verdadero y se compara directo entre niveles.
#
#   python scripts/figura_ruido.py

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tabla_ruido import FILAS, RES

FIG = Path(__file__).resolve().parents[1] / "results/figures"
NIVELES = [("sin ruido", "#86b6ef", "o"), ("σ = 0,01", "#2a78d6", "s"),
           ("σ = 0,05", "#104281", "^")]
TINTA, TINTA_2, GRILLA = "#1f1f1e", "#6b6a66", "#e4e3df"


def leer(tag):
    if tag is None or not (RES / f"{tag}.json").exists():
        return None
    return json.loads((RES / f"{tag}.json").read_text())


def main():
    # Todas las variantes. Las que salieron negativas sin ruido no se repitieron
    # con ruido y muestran un solo punto.
    filas = list(FILAS)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 0.42 * len(filas) + 1.6),
                               sharey=True, layout="constrained")
    etiquetas = []
    for i, (planta, conf, *tags, _) in enumerate(filas):
        y = len(filas) - 1 - i
        etiquetas.append((y, f"{conf}  ({planta})"))
        ds = [leer(t) for t in tags]
        nr = [d["nrmse_test"] if d else None for d in ds]
        err = [d["mean_param_error"] if d else None for d in ds]
        for ax, vals in ((a, nr), (b, err)):
            v = [x for x in vals if x is not None]
            if len(v) > 1:
                ax.plot([min(v), max(v)], [y, y], color=GRILLA, lw=2, zorder=1)
            for (nombre, color, m), x in zip(NIVELES, vals):
                if x is not None:
                    ax.scatter(x, y, s=46, color=color, marker=m, zorder=3,
                               edgecolor="#fcfcfb", linewidth=1.5)

    # separador entre plantas
    corte = [i for i, f in enumerate(filas) if f[0] == "act1"][0]
    for ax in (a, b):
        ax.axhline(len(filas) - corte - 0.5, color=TINTA_2, lw=0.6)

    a.set_yticks([y for y, _ in etiquetas], [e for _, e in etiquetas])
    a.set_xlabel("NRMSE de corrida libre [%]")
    b.set_xlabel("error de parámetros [%]")
    a.set_xlim(left=0)
    b.set_xlim(left=0)
    a.set_title("reproducción", loc="left", color=TINTA)
    b.set_title("identificación", loc="left", color=TINTA)
    for ax in (a, b):
        ax.grid(axis="x", color=GRILLA, lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(colors=TINTA_2, labelcolor=TINTA)
    manijas = [plt.Line2D([], [], ls="", marker=m, color=c, markersize=7,
                          markeredgecolor="#fcfcfb") for _, c, m in NIVELES]
    fig.legend(manijas, [n for n, _, _ in NIVELES], loc="outside upper center",
               ncol=3, frameon=False, fontsize=9)

    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"ruido_comparacion.{ext}", dpi=150, facecolor="#fcfcfb")
    print(f"-> {FIG / 'ruido_comparacion.png'}")


if __name__ == "__main__":
    main()
