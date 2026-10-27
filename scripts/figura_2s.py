#!/usr/bin/env python3
# Figuras y tabla de la tanda e16: los datos de 2 s a 2 kHz, con tres semillas.
#
# Una figura por nivel de ruido, con los tres paneles de siempre (reproducción,
# identificación, física recuperada). Cada fila es una configuración con sus dos
# arranques; el punto es la media sobre semillas y la barra, el desvío. Las
# variantes que solo tienen la semilla 0 van sin barra.
#
# Escribe results/figures/ampliado/dos_s_<nivel>.{pdf,png} y results/resumen_2s.md.
#
#   python scripts/figura_2s.py

from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "results" / "figures" / "ampliado"
C = dict(ink="#0b0b0b", ink2="#52514e", grid="#e6e5e0", cero="#77766f", wb="#2a78d6")
NIVELES = [("_n01", "σ = 0,01"), ("_n05", "σ = 0,05")]
PLANTAS = [
    ("Refractariedad", [("white-box", "e1wb"), ("agnóstica, ventana 5 ms", "B100"),
                        ("agnóstica, ventana 20 ms", "B400"), ("forma exacta", "e1S2")]),
    ("Actuador", [("white-box", "e2wb"), ("agnóstica", "e2B"), ("historia, 5 ms", "H100"),
                  ("historia, 20 ms", "H400"), ("filtro, 4 canales", "K400"),
                  ("filtro, 8 canales", "K400f8"), ("filtro, 16 canales", "K400f16"),
                  ("estado de filtro", "lag")]),
]
METRICAS = [("nrmse_test_interp", "NRMSE de corrida libre, test [%]", "reproducción"),
            ("mean_param_error", "error medio de parámetros [%]", "identificación"),
            ("r2_delta_test", "$R^2$ de la corrección contra $\\Delta f$", "física recuperada")]


def cargar():
    R = defaultdict(list)
    for f in glob.glob(str(RAIZ / "results" / "escalado" / "e16_*.json")):
        j = json.load(open(f))
        m = re.match(r"e16_(.+?)(_wrm)?(_n0[15])_s(\d)$", j["tag"])
        base, wrm, suf, _ = m.groups()
        # la forma exacta solo existe desde el white-box, con tag e1S2_wrm
        R[(base, "wrm" if wrm else "cero", suf)].append(j)
    return R


def ms(js, k):
    v = np.array([j[k] for j in js])
    return v.mean(), (v.std() if len(v) > 1 else np.nan), len(v)


def figura(R, suf, titulo):
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "ytick.major.size": 0,
                         "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic"})
    filas = []
    for planta, confs in PLANTAS:
        filas.append(planta)
        filas += confs
    n = len(filas)
    fig, axs = plt.subplots(1, 3, figsize=(26 / 2.54, (0.62 * n + 3) / 2.54), sharey=True,
                            layout="constrained")
    for k, (met, xl, tt) in enumerate(METRICAS):
        ax = axs[k]
        for i, fila in enumerate(filas):
            y = n - 1 - i
            if isinstance(fila, str):
                continue
            rot, base = fila
            puntos = []
            for arr, col, lleno in (("cero", C["cero"], True), ("wrm", C["wb"], False)):
                js = R.get((base, arr, suf), [])
                if not js or (k == 2 and "wb" in base):
                    continue
                m, s, cnt = ms(js, met)
                puntos.append(m)
                ax.errorbar(m, y, xerr=None if np.isnan(s) else s, fmt="o", ms=7, mew=1.8,
                            color=col, mfc=col if lleno else "white", ecolor=col,
                            elinewidth=1.5, capsize=3, zorder=3)
            if len(puntos) > 1:
                ax.plot(puntos, [y, y], color=C["grid"], lw=3.5, zorder=1)
            if k == 2 and "wb" in base:
                ax.text(0.02, y, "sin corrección", transform=ax.get_yaxis_transform(),
                        va="center", fontsize=8, color="#9a9a94")
        ax.set_xlabel(xl)
        ax.set_title(tt, loc="left", fontweight="bold", fontsize=10)
        ax.grid(axis="x", color=C["grid"])
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, n - 0.4)
    axs[0].set_xlim(left=0)
    axs[1].set_xlim(left=0)
    axs[2].axvline(0, color=C["ink2"], lw=1)
    axs[2].set_xlim(-0.6, 1.05)
    for i, fila in enumerate(filas):
        if isinstance(fila, str):
            axs[0].text(-0.02, n - 1 - i, fila, transform=axs[0].get_yaxis_transform(),
                        ha="right", va="center", fontweight="bold", color=C["ink2"])
            if i:
                for ax in axs:
                    ax.axhline(n - 1 - i + 0.5, color="#9a9a94", lw=0.8)
    axs[0].set_yticks([n - 1 - i for i, f in enumerate(filas) if not isinstance(f, str)],
                      [f[0] for f in filas if not isinstance(f, str)])
    h = [plt.Line2D([], [], ls="", marker="o", ms=7, color=C["cero"]),
         plt.Line2D([], [], ls="", marker="o", ms=7, mew=1.8, color=C["wb"], mfc="white")]
    fig.legend(h, ["desde cero", "desde el white-box"], loc="outside upper right", ncol=2,
               frameon=False)
    fig.suptitle(f"Datos de 2 s, {titulo}: media y desvío sobre tres semillas", x=0.01,
                 ha="left", fontweight="bold", fontsize=11)
    fig.text(0.01, -0.01, "Historia de 20 ms, filtros de 4 y 16 canales y agnóstica del actuador: "
             "solo la semilla 0, sin barra. NRMSE sobre los 13 escenarios de test que interpolan.",
             fontsize=8, color=C["ink2"], va="top")
    OUT.mkdir(parents=True, exist_ok=True)
    nombre = f"dos_s{suf}"
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{nombre}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return nombre


def tabla(R):
    md = ["# Tanda e16 (datos de 2 s): resultados", "",
          "Media ± desvío sobre semillas (n = 3; n = 1 donde no hay ±). NRMSE sobre los 13 "
          "escenarios de test que interpolan; extrapolación = box_a1.2.", ""]
    for suf, titulo in NIVELES:
        md += [f"## {titulo}", "", "| planta | configuración | arranque | n | NRMSE | extrap. | "
               "error de β | R² | validación |", "|---|---|---|---|---|---|---|---|---|"]
        for planta, confs in PLANTAS:
            for rot, base in confs:
                for arr in ("cero", "wrm"):
                    js = R.get((base, arr, suf), [])
                    if not js:
                        continue
                    def f(k):
                        m, s, _ = ms(js, k)
                        return f"{m:.2f}" + ("" if np.isnan(s) else f" ± {s:.2f}")
                    md.append(f"| {planta} | {rot} | {'warm start' if arr == 'wrm' else 'desde cero'} | "
                              f"{len(js)} | {f('nrmse_test_interp')} | {f('nrmse_extrap')} | "
                              f"{f('mean_param_error')} | {f('r2_delta_test')} | {f('nrmse_val_mejor')} |")
        md.append("")
    (RAIZ / "results" / "resumen_2s.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    R = cargar()
    for suf, titulo in NIVELES:
        print("escrito:", figura(R, suf, titulo))
    tabla(R)
    print("escrito: results/resumen_2s.md")
