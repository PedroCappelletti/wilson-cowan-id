#!/usr/bin/env python3
# Figuras del informe de amplitud grande: lazo abierto (resumen_amplitud.json) y
# lazo cerrado (control_amplitud.json frente a control_distancia_rango.json).
#
#   python scripts/figuras_amplitud.py      -> docs/figuras_amplitud/

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "docs" / "figuras_amplitud"
OUT.mkdir(parents=True, exist_ok=True)
AZUL, NARANJA, AQUA, AMARILLO, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#9a9a96"
TINTA, TINTA2 = "#1f1f1e", "#5c5c58"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": TINTA2,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
})


def coma(v, _=None):
    return f"{v:.10g}".replace(".", ",").replace("-", "−")


def guardar(fig, nombre):
    for ax in fig.axes:
        ax.xaxis.set_major_formatter(FuncFormatter(coma))
    fig.savefig(OUT / f"{nombre}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{nombre}.png", dpi=90, bbox_inches="tight")
    plt.close(fig)


def barras(ax, etiquetas, series, xmax, fmt="{:.1f}", log=False):
    n, k = len(etiquetas), len(series)
    alto = 0.8 / k
    y = np.arange(n)
    for j, (nombre, color, vals) in enumerate(series):
        pos = y - 0.4 + alto * (j + 0.5)
        v = [x if x is not None else np.nan for x in vals]
        ax.barh(pos, v, height=alto * 0.86, color=color, label=nombre)
        for p, x in zip(pos, v):
            if np.isfinite(x):
                ax.text(x * (1.08 if log else 1) + (0 if log else xmax * 0.012), p,
                        coma(float(fmt.format(x))), va="center", fontsize=11.5, color=TINTA2)
    ax.set_yticks(y, etiquetas)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    if log:
        ax.set_xscale("log")
    ax.set_xlim(0.05 if log else 0, xmax)


def leyenda_arriba(fig, ax):
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.0))


def lazo_abierto(res, planta, modelos, nombre, xmax_beta):
    filas = {(f["modelo"], f["ruido"]): f for f in res[planta]}
    fig, axs = plt.subplots(1, 2, figsize=(12, 0.62 * len(modelos) + 2.2), sharey=True)
    for ax, (suf, titulo) in zip(axs, (("_n01", "σ = 0,01"), ("_n05", "σ = 0,05"))):
        nuevo = [filas[(m, suf)]["amplitud"]["beta"] for m in modelos]
        viejo = [(filas[(m, suf)]["anterior"] or {}).get("beta") for m in modelos]
        barras(ax, modelos, [("entrenado con [0, 2]", GRIS, viejo), ("entrenado con amplitud grande", AZUL, nuevo)],
               xmax_beta, log=True)
        ax.set_title(titulo)
        ax.set_xlabel(r"error medio de $\hat{\beta}$ (%, escala log)")
    leyenda_arriba(fig, axs[0])
    guardar(fig, nombre)


def nrmse_nuevo(res, planta, modelos, nombre):
    filas = {(f["modelo"], f["ruido"]): f for f in res[planta]}
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(modelos) + 1.8))
    series = [(t, c, [filas[(m, s)]["amplitud"]["nrmse_nuevo"] for m in modelos])
              for s, t, c in (("_n01", "σ = 0,01", AZUL), ("_n05", "σ = 0,05", NARANJA))]
    barras(ax, modelos, series, max(max(v for v in s[2] if v) for s in series) * 1.18, fmt="{:.2f}")
    ax.set_xlabel("NRMSE en el test de amplitud grande (%)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    guardar(fig, nombre)


def mediana_lazo(archivo):
    filas = json.loads((RAIZ / "results" / archivo).read_text(encoding="utf-8"))
    g = defaultdict(list)
    for r in filas:
        if r["finito"]:
            g[(r["perturbacion"], r["referencia"], r["planta"], r["ruido"])].append(r)
    return lambda pert, ref, pl, suf, k: (st.median(x[k] for x in g[(pert, ref, pl, suf)])
                                          if g.get((pert, ref, pl, suf)) else np.nan)


def distancia(viejo, nuevo, nombre):
    # (etiqueta, nombre en control_distancia_rango, nombre en control_amplitud)
    pares = [("white-box", "white-box", "white-box"), ("agnóstica", "agnóstica", "agnóstica"),
             ("historia con comando actual", "con red", "historia con comando actual"),
             ("historia sin comando actual", "historia pasada", "historia sin comando actual"),
             ("filtro 8 canales", "filtro entrenable", "filtro 8 canales"),
             ("estado de filtro", "forma exacta", "estado de filtro"),
             ("estado de filtro x0", "forma exacta x0", "estado de filtro x0"),
             ("black-box", None, "black-box"), ("black-box con historia", None, "black-box con historia")]
    fig, axs = plt.subplots(1, 2, figsize=(12, 6.2), sharey=True)
    for ax, (ref, titulo) in zip(axs, (("theta2s", "theta-gamma, 2 s"), ("gamma50", "gamma, 50 ms"))):
        v = [viejo("actc", ref, a, "_n01", "dist_x") if a else np.nan for _, a, _ in pares]
        n = [nuevo("actc", ref, b, "_n01", "dist_x") for _, _, b in pares]
        xmax = np.nanmax(v + n) * 1.25
        barras(ax, [p[0] for p in pares], [("entrenado con [0, 2]", GRIS, v), ("entrenado con amplitud grande", AZUL, n)],
               xmax, fmt="{:.3f}")
        ax.set_title(titulo)
        ax.set_xlabel("distancia al lazo con el simulador")
    leyenda_arriba(fig, axs[0])
    guardar(fig, nombre)


if __name__ == "__main__":
    res = json.loads((RAIZ / "results" / "resumen_amplitud.json").read_text(encoding="utf-8"))
    act = ["white-box", "agnóstica, warm start", "historia con comando actual, warm start",
           "historia con comando actual, desde cero", "historia sin comando actual, warm start",
           "filtro 8 canales, warm start", "estado de filtro, warm start", "estado de filtro, x0"]
    ref = ["white-box", "agnóstica 5 ms, warm start", "agnóstica 5 ms, desde cero",
           "agnóstica 20 ms, warm start", "forma exacta, warm start", "forma exacta, x0"]
    lazo_abierto(res, "actuador", act, "beta_actuador", 900)
    lazo_abierto(res, "refractaria", ref, "beta_refractaria", 60)
    nrmse_nuevo(res, "actuador", act + ["black-box", "black-box con historia"], "nrmse_nuevo_actuador")
    nrmse_nuevo(res, "refractaria", ref + ["black-box", "black-box con historia"], "nrmse_nuevo_refractaria")
    if (RAIZ / "results" / "control_amplitud.json").exists():
        distancia(mediana_lazo("control_distancia_rango.json"), mediana_lazo("control_amplitud.json"),
                  "distancia_lazos_amplitud")
    print(f"-> {OUT}")
