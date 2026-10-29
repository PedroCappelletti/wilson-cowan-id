#!/usr/bin/env python3
# Tabla y figuras de control_paper_modelos.py.
#
# Tabla: RMSE de seguimiento (media ± desvío sobre las tres semillas) por
# referencia, planta, modelo y nivel de ruido. Figuras: trazas de E y su
# referencia en la theta-gamma, una ráfaga para la planta con actuador y tres
# para la de refractariedad (σ = 0,01, semilla 0).
#
#   python scripts/resumen_control_paper_modelos.py

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "results" / "figures" / "control"
ORDEN = [("β verdaderos", "β verdaderos"), ("white-box", "solo β̂"), ("con red", "solo β̂"),
         ("con red", "β̂ + g"), ("forma exacta", "β̂ + forma"), ("forma exacta", "solo β̂")]


def coma(x, d=4):
    return f"{x:.{d}f}".replace(".", ",")


def tabla(filas):
    g = defaultdict(list)
    for r in filas:
        g[(r["referencia"], r["planta"], r["modelo"], r["uso"], r["ruido"])].append(r)
    md = []
    for ref, nr in (("theta2s", "theta-gamma de 2 s"), ("gamma50", "gamma de 50 ms")):
        ideal = g[(ref, "ideal", "ideal", "β verdaderos", "")][0]
        md += [f"### Referencia {nr}", "",
               f"Ideal (Wilson-Cowan sin perturbación, β verdaderos): RMSE {coma(ideal['rmse'])} "
               f"(I {coma(ideal['rmse_I'])}, E {coma(ideal['rmse_E'])}).", ""]
        for planta, npl in (("refrac", "refractariedad"), ("act", "actuador")):
            md += [f"**Planta con {npl}**", "",
                   "| modelo | uso en el controlador | σ = 0,01: error de β̂ | σ = 0,01: RMSE "
                   "| σ = 0,05: error de β̂ | σ = 0,05: RMSE |",
                   "|---|---|---|---|---|---|"]
            for modelo, uso in ORDEN:
                if modelo == "β verdaderos":
                    rs = g.get((ref, planta, modelo, uso, ""), [])
                    if rs:
                        v = coma(rs[0]["rmse"])
                        md.append(f"| β verdaderos | sin modelo de la perturbación | 0 % | {v} | 0 % | {v} |")
                    continue
                celdas = []
                for suf in ("_n01", "_n05"):
                    rs = g.get((ref, planta, modelo, uso, suf), [])
                    if not rs:
                        celdas += ["", ""]
                        continue
                    v = np.array([r["rmse"] for r in rs])
                    eb = np.array([r["error_beta"] for r in rs])
                    celdas += [f"{coma(eb.mean(), 1)} %", f"{coma(v.mean())} ± {coma(v.std())}"]
                if not any(celdas):
                    continue
                md.append(f"| {modelo} | {uso} | " + " | ".join(celdas) + " |")
            md.append("")
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
        ax.set_title(rot.replace("β̂", r"$\hat{\beta}$"), loc="left", fontsize=9.5)
        ax.set_ylim(-0.1, 0.9)
    axs[-1].set_xlabel("tiempo [ms]")
    h = [plt.Line2D([], [], color="#104281"), plt.Line2D([], [], color="#9a9a94", ls="--")]
    fig.legend(h, ["E en lazo cerrado", "referencia $r_E$"], loc="outside lower center",
               ncol=2, frameon=False)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{nombre}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    filas = json.loads((RAIZ / "results" / "control_paper_modelos.json").read_text(encoding="utf-8"))
    (RAIZ / "results" / "control_paper_modelos.md").write_text(tabla(filas) + "\n", encoding="utf-8")
    tr = dict(np.load(RAIZ / "results" / "control_paper_modelos_trazas.npz"))
    figura(tr, "paper_modelos_actuador", [
        ("ideal|ideal|β verdaderos", "Ideal: planta sin perturbación, β verdaderos"),
        ("act|β verdaderos|β verdaderos", "Planta con actuador, β verdaderos"),
        ("act|con red|solo β̂", "Planta con actuador, historia de 5 ms (con red), β̂"),
        ("act|forma exacta|solo β̂", "Planta con actuador, estado de filtro, β̂")], (720, 760))
    figura(tr, "paper_modelos_refractariedad", [
        ("refrac|β verdaderos|β verdaderos", "Planta con refractariedad, β verdaderos"),
        ("refrac|con red|β̂ + g", "Planta con refractariedad, agnóstica (con red), β̂ + g"),
        ("refrac|forma exacta|β̂ + forma", "Planta con refractariedad, forma exacta, β̂ + forma")], (600, 900))
    print("escrito: results/control_paper_modelos.md y figuras paper_modelos_*")


if __name__ == "__main__":
    main()
