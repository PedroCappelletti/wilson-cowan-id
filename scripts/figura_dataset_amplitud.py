#!/usr/bin/env python3
# Ejemplos del dataset de amplitud grande: comando P y respuesta E del
# simulador con actuador con recorte, una familia por fila, amplitudes 2 y 10.
#
#   python scripts/figura_dataset_amplitud.py

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "docs" / "figuras_dataset"
OUT.mkdir(parents=True, exist_ok=True)
coma = FuncFormatter(lambda v, _: f"{v:.10g}".replace(".", ",").replace("-", "−"))
plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                     "pdf.fonttype": 42})

d = np.load(RAIZ / "data/processed/uncertain/actc_amp_2s.npz", allow_pickle=True)
lab = list(d["labels"]); t = d["t"]
fams = ["square", "aprbs", "thetagamma", "poisson", "chirp", "ruido"]
a, b = np.searchsorted(t, 400), np.searchsorted(t, 520)
fig, axs = plt.subplots(len(fams), 2, figsize=(11, 12), sharex=True)
for i, fam in enumerate(fams):
    for j, A in enumerate((2, 10)):
        k = lab.index(f"{fam}_A{A}")
        ax = axs[i, j]
        ax.plot(t[a:b], d["P"][k, a:b], color="#2a78d6", lw=1.1, label="comando P")
        ax.set_ylim(-14, 14)
        ax2 = ax.twinx()
        ax2.plot(t[a:b], d["E"][k, a:b], color="#1f1f1e", lw=1.3, label="E")
        ax2.set_ylim(-0.1, 1.0)
        ax2.spines["top"].set_visible(False)
        ax.set_title(f"{fam}, A = {A}", loc="left", fontsize=11)
        ax.yaxis.set_major_formatter(coma); ax2.yaxis.set_major_formatter(coma)
        if j == 0: ax.set_ylabel("P")
        if j == 1: ax2.set_ylabel("E")
for ax in axs[-1]:
    ax.set_xlabel("tiempo (ms)"); ax.xaxis.set_major_formatter(coma)
fig.tight_layout()
fig.savefig(OUT / "ejemplos_amplitud.png", dpi=90)
fig.savefig(OUT / "ejemplos_amplitud.pdf")
print(OUT)
