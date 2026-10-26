#!/usr/bin/env python3
# Tablas y figura de las comparaciones de controladores (control_comparado.py).
#
# Mediana del RMSE de seguimiento sobre ruidos y semillas, por planta, modelo,
# arranque, controlador, referencia y modo del comando. También el rango del
# comando sin tope y la fracción del tiempo contra el tope con anti-windup.
#
# Escribe results/control_resumen.md y results/figures/control/control_*.{pdf,png}.
#
#   python scripts/resumen_control.py

from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "results" / "figures" / "control"
MODOS = [("libre", "sin tope"), ("tope", "tope literal"), ("tope_aw", "tope y anti-windup")]
REFS = [("gamma50", "gamma 50 ms"), ("theta2s", "theta-gamma 2 s")]
ORDEN = {"act": ["oráculo", "white-box", "agnóstica", "historia", "filtro", "estado de filtro"],
         "refrac": ["oráculo", "oráculo + forma", "white-box", "agnóstica 5 ms",
                    "agnóstica 20 ms", "forma exacta"]}


def cargar():
    filas = []
    for f in glob.glob(str(RAIZ / "results" / "control_comparado_e1*_*_*.json")):
        tanda = "e13" if "_e13_" in f else "e16"
        for r in json.load(open(f, encoding="utf-8")):
            m = r["modelo"]
            r["tanda"] = tanda
            r["base"] = m.split(",")[0]
            r["arranque"] = "warm start" if "warm" in m else ("desde cero" if "cero" in m else "—")
            filas.append(r)
    return filas


def agrupar(filas):
    g = collections.defaultdict(list)
    for r in filas:
        g[(r["tanda"], r["referencia"], r["modo"], r["planta"], r["base"],
           r["arranque"], r["controlador"])].append(r)
    return g


def med(rs, k):
    return float(np.median([r[k] for r in rs])) if rs else float("nan")


def tablas(g):
    md = ["# Comparación de controladores: tablas", "",
          "Mediana del RMSE de seguimiento sobre niveles de ruido y semillas. «solo β̂» = "
          "controlador original con los parámetros del modelo; «con modelo» = original "
          "cancelando g o con la forma exacta; «extendido» = despeje con la red o con la "
          "copia del actuador. Tope: P en [0, 2,0], Q en [0, 1,5].", ""]
    for tanda, nt in (("e16", "modelos de 2 s"), ("e13", "modelos de 200 ms")):
        for ref, nr in REFS:
            md += [f"## {nt}, referencia {nr}", ""]
            for planta, npl in (("act", "actuador"), ("refrac", "refractariedad")):
                md += [f"**{npl}**", "",
                       "| modelo | arranque | controlador | sin tope | tope literal | "
                       "tope y anti-windup | P sin tope | contra el tope (aw) |",
                       "|---|---|---|---|---|---|---|---|"]
                claves = sorted({(k[4], k[5], k[6]) for k in g
                                 if k[0] == tanda and k[1] == ref and k[3] == planta},
                                key=lambda c: (ORDEN[planta].index(c[0]) if c[0] in ORDEN[planta] else 99,
                                               c[1], c[2]))
                for base, arr, ctrl in claves:
                    v = [med(g.get((tanda, ref, m, planta, base, arr, ctrl), []), "rmse")
                         for m, _ in MODOS]
                    rl = g[(tanda, ref, "libre", planta, base, arr, ctrl)]
                    pmin = np.median([r["rango_P_sin_tope"][0] for r in rl])
                    pmax = np.median([r["rango_P_sin_tope"][1] for r in rl])
                    ft = med(g.get((tanda, ref, "tope_aw", planta, base, arr, ctrl), []), "frac_tope")
                    md.append(f"| {base} | {arr} | {ctrl} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} | "
                              f"[{pmin:.1f}, {pmax:.1f}] | {100 * ft:.0f} % |")
                md.append("")
    return md


def figura(g, tanda="e16"):
    """Actuador: original (solo β̂) contra el mejor uso del modelo, warm start."""
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    filas = ["white-box", "agnóstica", "historia", "filtro", "estado de filtro"]
    usa = {"agnóstica": "original, con modelo", "historia": "extendido",
           "filtro": "extendido", "estado de filtro": "extendido"}
    fig, axs = plt.subplots(1, 4, figsize=(27 / 2.54, 8 / 2.54), sharey=True, layout="constrained")
    combos = [(ref, nr, m, nm) for ref, nr in REFS for m, nm in (MODOS[0], MODOS[2])]
    for ax, (ref, nr, modo, nm) in zip(axs, combos):
        ora = med(g.get((tanda, ref, modo, "act", "oráculo", "—", "oráculo"), []), "rmse")
        ax.axvline(ora, color="#52514e", lw=1, ls="--")
        tope_x = [ora]
        for i, base in enumerate(filas):
            y = len(filas) - 1 - i
            arr = "—" if base == "white-box" else "warm start"
            a = med(g.get((tanda, ref, modo, "act", base, arr, "original, solo β̂"), []), "rmse")
            b = (med(g.get((tanda, ref, modo, "act", base, arr, usa[base]), []), "rmse")
                 if base in usa else np.nan)
            if np.isfinite(b):
                ax.plot([a, b], [y, y], color="#e6e5e0", lw=3, zorder=1)
                ax.scatter(b, y, marker="D", s=55, color="#104281", zorder=3)
            ax.scatter(a, y, s=55, color="#77766f", zorder=3)
            tope_x += [a, b]
        ax.set_title(f"{nr}\n{nm}", fontsize=9.5, loc="left", fontweight="bold")
        ax.set_xlabel("RMSE de seguimiento")
        ax.grid(axis="x", color="#e6e5e0")
        ax.set_xlim(0, np.nanmax(tope_x) * 1.12)
    axs[0].set_yticks(range(len(filas))[::-1], filas)
    h = [plt.Line2D([], [], ls="", marker="o", color="#77766f", markersize=7),
         plt.Line2D([], [], ls="", marker="D", color="#104281", markersize=7),
         plt.Line2D([], [], color="#52514e", ls="--")]
    fig.legend(h, ["original, solo β̂", "usando el modelo (extendido, o cancelando g)",
                   "oráculo (θ verdaderos, sin saber del actuador)"],
               loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Planta con actuador, modelos de 2 s con warm start", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"control_actuador.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    g = agrupar(cargar())
    (RAIZ / "results" / "control_resumen.md").write_text("\n".join(tablas(g)) + "\n", encoding="utf-8")
    figura(g)
    print("escrito: results/control_resumen.md y results/figures/control/control_actuador")


if __name__ == "__main__":
    main()
