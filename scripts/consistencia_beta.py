#!/usr/bin/env python3
# Análisis de la tanda e14 (cola_familias.py): cuánto se dispersa β entre
# familias de estímulos, en el control sin perturbación y en las dos plantas.
#
# Por familia y parámetro, el desvío relativo (β_fam − β_verdadero)/β_verdadero.
# Dos números por planta y nivel de ruido:
#   error medio     el promedio de |desvío| sobre parámetros y familias;
#   dispersión      el desvío estándar entre familias de cada parámetro,
#                   promediado sobre parámetros.
# El exceso de dispersión de una planta sobre el control es lo que aporta la
# perturbación: el control solo se dispersa por falta de información.
#
# Escribe results/consistencia_beta.md y las figuras
# results/figures/ampliado/consistencia_beta_<nivel>.{pdf,png}.
#
#   python scripts/consistencia_beta.py

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "results" / "escalado"
OUT = RAIZ / "results" / "figures" / "ampliado"

PARAMS = ("wEE", "wEI", "wIE", "wII", "te", "ti", "ae", "ai", "thetae", "thetai")
ROTULOS = {"wEE": "$w_{EE}$", "wEI": "$w_{EI}$", "wIE": "$w_{IE}$", "wII": "$w_{II}$",
           "te": r"$\tau_e$", "ti": r"$\tau_i$", "ae": "$a_e$", "ai": "$a_i$",
           "thetae": r"$\theta_e$", "thetai": r"$\theta_i$"}
FAMILIAS = ["box", "square", "aprbs", "prbs", "thetagamma", "poisson", "chirp"]
# Orden categórico fijo, el mismo en todas las figuras.
COLORES = ["#2a78d6", "#e8743b", "#19a979", "#945ecf", "#d64f8b", "#13a4b4", "#7a6f2c"]
PLANTAS = [("wc0", "sin perturbación (control)", "e14_wc0_todas"),
           ("refrac1", "refractariedad", "e13_e1wb"),
           ("act1", "actuador", "e13_e2wb")]
NIVELES = [("_n01", "σ = 0,01"), ("_n05", "σ = 0,05"), ("", "sin ruido")]


def verdaderos():
    d = np.load(RAIZ / "data/processed/uncertain/wc0_amp.npz", allow_pickle=True)
    return {k: float(d[k]) for k in PARAMS}


def leer(tag):
    f = RES / f"{tag}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def desvios(d, true):
    return np.array([(d["params"][k] - true[k]) / true[k] for k in PARAMS]) * 100


def main():
    true = verdaderos()
    md = ["# Consistencia de β entre familias de estímulos (tanda e14)", "",
          "Un white-box por familia, entrenado y validado solo con esa familia. "
          "«error» = promedio de |desvío relativo| sobre parámetros y familias; "
          "«dispersión» = desvío estándar entre familias, promediado sobre parámetros; "
          "«todas» = el white-box con las siete familias juntas. En %.", "",
          "| planta | ruido | familias | error | dispersión | error de «todas» |",
          "|---|---|---|---|---|---|"]
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic"})
    for suf, nivel in NIVELES:
        fig, axs = plt.subplots(1, 3, figsize=(26 / 2.54, 12 / 2.54), sharey=True,
                                layout="constrained")
        for ax, (p, nombre, tag_todas) in zip(axs, PLANTAS):
            filas = {f: leer(f"e14_{p}_{f}{suf}") for f in FAMILIAS}
            hay = {f: desvios(d, true) for f, d in filas.items() if d}
            todas = leer(tag_todas + suf)
            ys = np.arange(len(PARAMS))[::-1]
            for j, f in enumerate(FAMILIAS):
                if f in hay:
                    ax.scatter(np.clip(hay[f], -100, 100), ys + (j - 3) * 0.08, s=16,
                               color=COLORES[j], label=f, zorder=3)
            if todas:
                ax.scatter(np.clip(desvios(todas, true), -100, 100), ys, marker="|",
                           s=220, color="#0b0b0b", linewidth=2, label="todas", zorder=4)
            ax.axvline(0, color="#52514e", lw=1)
            ax.set_xlim(-100, 100)
            ax.set_title(f"{nombre}\n{len(hay)} de 7 familias", fontsize=10, loc="left",
                         fontweight="bold")
            ax.set_xlabel("desvío relativo del parámetro [%]")
            ax.grid(axis="x", color="#e6e5e0")
            if hay:
                M = np.array(list(hay.values()))
                err = float(np.abs(M).mean())
                disp = float(M.std(axis=0).mean()) if len(M) > 1 else float("nan")
                et = f"{np.abs(desvios(todas, true)).mean():.1f}" if todas else "—"
                md.append(f"| {nombre} | {nivel} | {len(hay)} | {err:.1f} | {disp:.1f} | {et} |")
        axs[0].set_yticks(np.arange(len(PARAMS))[::-1], [ROTULOS[k] for k in PARAMS])
        h, l = axs[0].get_legend_handles_labels()
        if h:
            fig.legend(h, l, loc="outside lower center", ncol=8, frameon=False)
        fig.suptitle(f"β por familia de estímulos, white-box, {nivel}", x=0.01, ha="left",
                     fontweight="bold", fontsize=12)
        OUT.mkdir(parents=True, exist_ok=True)
        nombre_fig = f"consistencia_beta{suf or '_sin_ruido'}"
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{nombre_fig}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"escrito: {nombre_fig}")
    (RAIZ / "results" / "consistencia_beta.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[6:]))


if __name__ == "__main__":
    main()
