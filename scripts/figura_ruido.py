#!/usr/bin/env python3
# Cuánto se degrada cada configuración con el ruido, en dos métricas.
#
# Izquierda, error de parámetros: se mide contra el beta verdadero y no depende
# de la escala de la señal, así que se compara directo entre niveles de ruido.
# Derecha, cuánto NRMSE tiene cada variante por encima del piso (lo que sacaría
# un modelo perfecto con ese ruido), dividido por lo mismo del white-box de la
# misma planta. El NRMSE crudo no se compara entre niveles: el ruido infla el
# rango con que se normaliza y además pone un piso de 6 % con sigma = 0.05.
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
from tabla_ruido import FILAS, RES, piso

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
    wb = {f[0]: [leer(t) for t in f[2:5]] for f in filas if f[1] == "white-box"}
    pisos = {p: [piso(p, n) for n in (None, "01", "05")] for p in wb}

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 0.42 * len(filas) + 1.6),
                               sharey=True, layout="constrained")
    etiquetas = []
    for i, (planta, conf, *tags, _) in enumerate(filas):
        y = len(filas) - 1 - i
        etiquetas.append((y, f"{conf}  ({planta})"))
        err, rel = [], []
        for j, t in enumerate(tags):
            d, w = leer(t), wb[planta][j]
            err.append(d["mean_param_error"] if d else None)
            # el white-box vale 1 por definicion en el cociente: se omite ahi
            ref = conf == "white-box"
            # exceso sobre el piso, relativo al del white-box
            if d and w and not ref:
                p0 = pisos[planta][j]
                rel.append((d["nrmse_test"] - p0) / (w["nrmse_test"] - p0))
            else:
                rel.append(None)
        for ax, vals in ((a, err), (b, rel)):
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
    b.axvline(1.0, color=TINTA_2, lw=1, zorder=0)
    b.text(0.99, len(filas) - 0.55, "igual que el white-box ", color=TINTA_2,
           fontsize=8, ha="right", va="bottom")

    a.set_yticks([y for y, _ in etiquetas], [e for _, e in etiquetas])
    a.set_xlabel("error de parámetros [%]")
    b.set_xlabel("exceso de NRMSE sobre el piso, relativo al del white-box")
    b.set_xlim(left=0)
    a.set_title("identificación", loc="left", color=TINTA)
    b.set_title("reproducción", loc="left", color=TINTA)
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
