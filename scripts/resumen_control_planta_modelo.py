#!/usr/bin/env python3
# Tabla y figuras de control_paper_planta_modelo.py.
#
# Tabla: RMSE de seguimiento por perturbación, planta y nivel de ruido (media ±
# desvío sobre las tres semillas), y rango del comando. Figuras: E y su
# referencia en la theta-gamma para cada planta (σ = 0,01, semilla 0).
#
#   python scripts/resumen_control_planta_modelo.py

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "results" / "figures" / "control"
PLANTAS = ("white-box", "con red", "forma exacta")
NOMBRE = {"refrac": {"con red": "agnóstica", "forma exacta": "forma exacta", "white-box": "white-box"},
          "act": {"con red": "historia 5 ms", "forma exacta": "estado de filtro", "white-box": "white-box"}}


def coma(x, d=4):
    return "diverge" if not math.isfinite(x) else f"{x:.{d}f}".replace(".", ",")


def rango(rs, k):
    lo = min(r[k][0] for r in rs)
    hi = max(r[k][1] for r in rs)
    return f"{lo:.1f} a {hi:.1f}".replace(".", ",").replace("-", "−")


def tabla(filas):
    g = defaultdict(list)
    for r in filas:
        g[(r["referencia"], r["perturbacion"], r["planta"], r["ruido"])].append(r)

    def celda(ref, pert, pl, suf):
        rs = g[(ref, pert, pl, suf)]
        v = np.array([r["rmse"] for r in rs])
        if len(v) == 1:
            return coma(v[0])
        return f"{coma(v.mean())} ± {coma(v.std())}"

    md = [f"Simulador sin perturbación: RMSE {celda('theta2s', 'ideal', 'simulador', '')} en la theta-gamma "
          f"y {celda('gamma50', 'ideal', 'simulador', '')} en la gamma.", ""]
    for pert, npert in (("refrac", "refractariedad"), ("act", "actuador")):
        md += [f"**Con {npert}**", "",
               "| planta | σ | RMSE theta-gamma | RMSE gamma |", "|---|---|---|---|",
               f"| simulador | | {celda('theta2s', pert, 'simulador', '')} | {celda('gamma50', pert, 'simulador', '')} |"]
        for pl in PLANTAS:
            for i, (suf, s) in enumerate((("_n01", "0,01"), ("_n05", "0,05"))):
                nom = NOMBRE[pert][pl] if i == 0 else ""
                md.append(f"| {nom} | {s} | {celda('theta2s', pert, pl, suf)} | {celda('gamma50', pert, pl, suf)} |")
        md.append("")
    md += ["**Rango del comando**", "",
           "| perturbación | planta | $P$, theta-gamma | $Q$, theta-gamma | $P$, gamma | $Q$, gamma |",
           "|---|---|---|---|---|---|"]
    for pert, npert in (("ideal", "ninguna"), ("refrac", "refractariedad"), ("act", "actuador")):
        for i, pl in enumerate(("simulador",) + (PLANTAS if pert != "ideal" else ())):
            a = [r for (ref, p, q, _), rs in g.items() if ref == "theta2s" and p == pert and q == pl for r in rs]
            b = [r for (ref, p, q, _), rs in g.items() if ref == "gamma50" and p == pert and q == pl for r in rs]
            nom = pl if pl == "simulador" else NOMBRE[pert][pl]
            md.append(f"| {npert if i == 0 else ''} | {nom} | {rango(a, 'P_rango')} | {rango(a, 'Q_rango')} "
                      f"| {rango(b, 'P_rango')} | {rango(b, 'Q_rango')} |")
    return "\n".join(md)


def figura(trazas, nombre, casos, ventana):
    plt.rcParams.update({"font.family": "Arial", "font.size": 9.5, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(len(casos), 1, figsize=(17 / 2.54, 2.6 * len(casos) / 2.54 + 1),
                            sharex=True, layout="constrained")
    for ax, (clave, rot) in zip(axs, casos):
        t = trazas[f"theta2s|{clave}|t"]
        m = (t >= ventana[0]) & (t <= ventana[1])
        ax.plot(t[m], trazas[f"theta2s|{clave}|rE"][m], color="#9a9a94", lw=1.2, ls="--")
        ax.plot(t[m], trazas[f"theta2s|{clave}|E"][m], color="#104281", lw=1.1)
        ax.set_ylabel("E")
        ax.set_title(rot, loc="left", fontsize=9.5)
        ax.set_ylim(min(-0.1, float(trazas[f"theta2s|{clave}|E"][m].min()) - 0.05), 0.9)
    axs[-1].set_xlabel("tiempo [ms]")
    h = [plt.Line2D([], [], color="#104281"), plt.Line2D([], [], color="#9a9a94", ls="--")]
    fig.legend(h, ["E en lazo cerrado", "referencia $r_E$"], loc="outside lower center",
               ncol=2, frameon=False)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{nombre}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    filas = json.loads((RAIZ / "results" / "control_planta_modelo.json").read_text(encoding="utf-8"))
    (RAIZ / "results" / "control_planta_modelo.md").write_text(tabla(filas) + "\n", encoding="utf-8")
    tr = dict(np.load(RAIZ / "results" / "control_planta_modelo_trazas.npz"))
    for pert, nombre, ventana in (("act", "planta_modelo_actuador", (720, 760)),
                                  ("refrac", "planta_modelo_refractariedad", (600, 900))):
        npert = "actuador" if pert == "act" else "refractariedad"
        casos = [(f"{pert}|simulador", f"Simulador con {npert}")]
        casos += [(f"{pert}|{pl}", f"Modelo {NOMBRE[pert][pl]}") for pl in PLANTAS]
        figura(tr, nombre, casos, ventana)
    print("escrito: results/control_planta_modelo.md y figuras planta_modelo_*")


if __name__ == "__main__":
    main()
