#!/usr/bin/env python3
# Un chirp entero del régimen de amplitud grande: el barrido lineal de 10 a
# 80 Hz durante los 1,8 s del estímulo, y los primeros y los últimos 150 ms
# para que se vea el cambio de frecuencia. La figura de ejemplos por familia
# muestra solo 120 ms, donde el chirp parece una senoide.
#
#   python scripts/figura_chirp.py   -> docs/informe_final/figuras/chirp_barrido.pdf

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "docs" / "informe_final" / "figuras" / "chirp_barrido.pdf"
ESCENARIO = "chirp_A5"          # amplitud 5, banda de 10 a 80 Hz
coma = FuncFormatter(lambda v, _: f"{v:.10g}".replace(".", ",").replace("-", "−"))
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})

d = np.load(RAIZ / "data/processed/uncertain/actc_amp_2s.npz", allow_pickle=True)
k = list(d["labels"]).index(ESCENARIO)
t, P, E = d["t"], d["P"][k], d["E"][k]

fig = plt.figure(figsize=(10, 5.6))
arriba = fig.add_subplot(2, 1, 1)
arriba.plot(t, P, color="#2a78d6", lw=0.5)
arriba.set_xlim(0, 2000)
arriba.set_ylabel("P")
arriba.set_xlabel("tiempo (ms)")
arriba.set_title("barrido completo, de 10 Hz (100 ms) a 80 Hz (1900 ms)", loc="left", fontsize=11)
for x0, x1 in ((100, 250), (1750, 1900)):
    arriba.axvspan(x0, x1, color="#e4e3df", zorder=0)
for j, (x0, x1, titulo) in enumerate(((100, 250, "inicio del estímulo, alrededor de 10 Hz"),
                                      (1750, 1900, "final del estímulo, alrededor de 80 Hz"))):
    ax = fig.add_subplot(2, 2, 3 + j)
    m = (t >= x0) & (t <= x1)
    ax.plot(t[m], P[m], color="#2a78d6", lw=1.1)
    ax.set_ylim(-6, 6)
    ax2 = ax.twinx()
    ax2.plot(t[m], E[m], color="#1f1f1e", lw=1.1)
    ax2.set_ylim(-0.05, 1.0)
    ax2.spines["top"].set_visible(False)
    ax.set_title(titulo, loc="left", fontsize=11)
    ax.set_xlabel("tiempo (ms)")
    if j == 0:
        ax.set_ylabel("P")
    else:
        ax2.set_ylabel("E")
    ax.xaxis.set_major_formatter(coma)
    ax2.yaxis.set_major_formatter(coma)
fig.tight_layout()
fig.savefig(OUT)
fig.savefig(OUT.with_suffix(".png"), dpi=120)
print(OUT)
