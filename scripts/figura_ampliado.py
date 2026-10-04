#!/usr/bin/env python3
# Figuras de la tanda e13 sobre el dataset ampliado. No reemplazan la figura
# del póster (poster-csan2026/figuras_poster.py), que sigue con sus corridas.
#
# Dos figuras:
#   ampliado_arranque_<nivel>   una por nivel de ruido. Cada fila es una
#       configuración con sus dos arranques, desde cero y desde el white-box,
#       unidos por un segmento. Se lee si el arranque mejora, y cuánto.
#   ampliado_vs_poster          desde cero, el dataset del póster contra el
#       ampliado, medido sobre los 7 escenarios de test que tienen en común.
#
# El NRMSE es sobre los escenarios de test que interpolan; box_a1.2, el único
# que extrapola en amplitud, se informa aparte en la consola. Cada corrida se
# evalúa en su estado de mejor validación, así que una divergencia al final
# del entrenamiento no afecta lo que se grafica. Lo que todavía no terminó
# queda marcado como pendiente; volver a correr el script lo completa.
#
#   python scripts/figura_ampliado.py

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "results" / "escalado"
OUT = RAIZ / "results" / "figures" / "ampliado"

CUERPO = 10
R2_MIN = -2.1
C = dict(ink="#0b0b0b", ink2="#52514e", grid="#e6e5e0", tenue="#9a9a94",
         pend="#c2185b", cero="#77766f", wb="#2a78d6")

# El criterio de los pósters: marcador hueco = arranque desde el white-box.
ARRANQUES = [("desde cero", C["cero"], True, "o"),
             ("desde el white-box", C["wb"], False, "o")]

NIVELES = [("_n01", "σ = 0,01"), ("_n05", "σ = 0,05"), ("", "sin ruido")]

# (rótulo, tag desde cero, tag desde el white-box); el sufijo de ruido se agrega.
PLANTAS = [
    ("Refractariedad", [
        ("white-box", "e13_e1wb", None),
        ("agnóstica, ventana 5 ms", "e13_B100", "e13_B100_wrm"),
        ("agnóstica, ventana 20 ms", "e13_B400", "e13_B400_wrm"),
        ("forma exacta", "e13_e1S_cero", "e13_e1S2"),
    ]),
    ("Actuador", [
        ("white-box", "e13_e2wb", None),
        ("agnóstica", "e13_e2B", "e13_e2B_wrm"),
        ("historia del comando, 100", "e13_H100", "e13_H100_wrm"),
        ("historia del comando, 400", "e13_H400", "e13_H400_wrm"),
        ("filtro entrenable, 4 canales", "e13_K400", "e13_K400_wrm"),
        ("filtro entrenable, 8 canales", "e13_K400f8", "e13_K400f8_wrm"),
        ("filtro entrenable, 16 canales", "e13_K400f16", "e13_K400f16_wrm"),
        ("estado de filtro", "e13_lag", "e13_lag_wrm"),
    ]),
]

# Las mismas configuraciones en las tablas del póster (dataset de 20 escenarios).
POSTER = {
    "e13_e1wb": ("e1_wb", "e7_e1wb_n01", "e7_e1wb_n05"),
    "e13_B100": ("e1_B_w100", "e11_B100_cero_n01", "e11_B100_cero_n05"),
    "e13_B400": ("e1_B_w400", "e7_e1B400_n01", "e7_e1B400_n05"),
    "e13_e1S2": ("e1_S2", "e7_e1S2_n01", "e7_e1S2_n05"),
    "e13_e2wb": ("e2_wb", "e7_e2wb_n01", "e6_wb_n05_s7"),
    "e13_e2B": ("e2_B", "e7_e2B_n01", "e7_e2B_n05"),
    "e13_H100": ("e2_H100", "e8_H100_n01", "e8_H100_n05"),
    "e13_H400": ("e2_H400", "e8_H400_n01", "e6_H400_n05_s7"),
    "e13_K400": ("e3_K400", "e8_K400_n01", "e6_K400_n05_s7"),
    "e13_K400f8": ("e3_K400_f8", "e8_K400f8_n01", "e8_K400f8_n05"),
    "e13_K400f16": ("e9_K400f16", "e9_K400f16_n01", "e9_K400f16_n05"),
    "e13_lag": ("e2_lag2", "e7_e2lag_n01", "e7_e2lag_n05"),
}
TEST_VIEJO = {"box_a1.2", "square_a1.0_f130", "aprbs_2", "prbs_1",
              "thetagamma_2", "poisson_1", "chirp"}


def leer(tag):
    f = RES / f"{tag}.json"
    return json.loads(f.read_text(encoding="utf-8")) if tag and f.exists() else None


def nrmse_interp(d):
    return d.get("nrmse_test_interp", d["nrmse_test"])


def nrmse_viejo(d):
    """NRMSE sobre los 7 escenarios del test del póster."""
    v = [f["nrmse"] for f in d["por_escenario"] if f["label"] in TEST_VIEJO]
    return sum(v) / len(v)


def estilo():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "Arial", "font.size": CUERPO,
        "axes.labelsize": CUERPO, "xtick.labelsize": CUERPO - 1,
        "ytick.labelsize": CUERPO,
        "text.color": C["ink"], "axes.labelcolor": C["ink2"],
        "xtick.color": C["ink2"], "ytick.color": C["ink"],
        "axes.edgecolor": C["grid"], "grid.color": C["grid"],
        "axes.grid": True, "grid.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "ytick.major.size": 0,
        "legend.frameon": False, "pdf.fonttype": 42,
        "mathtext.fontset": "custom", "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    })


def tres_paneles(filas, titulo, series, nombre, nota):
    """filas: lista de (rótulo, [(nrmse, err, r2) o None por serie]) o None para
    un encabezado de planta. series: [(nombre, color, lleno, marcador)]."""
    n = len(filas)
    fig, axs = plt.subplots(1, 3, figsize=(26 / 2.54, (0.62 * n + 3.2) / 2.54),
                            sharey=True, layout="constrained")
    pendientes = 0
    for k, ax in enumerate(axs):
        for i, fila in enumerate(filas):
            y = n - 1 - i
            if isinstance(fila, str):
                continue
            rot, puntos = fila
            vals = [p[k] if p else None for p in puntos]
            eje = [max(v, R2_MIN) if (k == 2 and v is not None) else v for v in vals]
            v = [x for x in eje if x is not None]
            if len(v) > 1:
                ax.plot([min(v), max(v)], [y, y], color=C["grid"], lw=3.5, zorder=1)
            for (_, col, lleno, marca), x, xe in zip(series, vals, eje):
                if x is None:
                    continue
                kw = dict(s=70, zorder=3, linewidth=1.8,
                          color=col if lleno else "white", edgecolor=col)
                if xe != x:
                    ax.scatter(xe + 0.06, y, marker="<", **kw)
                    ax.text(xe + 0.17, y + 0.33, f"{x:.1f}".replace("-", "−"),
                            fontsize=CUERPO - 2, color=col, va="center")
                else:
                    ax.scatter(x, y, marker=marca, **kw)
            if k == 2 and any(p is None for p in puntos[:len(series)]) and \
                    not (rot == "white-box" and puntos[0] is not None):
                pendientes += 1
                ax.text(1.02, y, "pendiente", transform=ax.get_yaxis_transform(),
                        va="center", fontsize=CUERPO - 2, color=C["pend"])
            if k == 2 and rot == "white-box":
                ax.text(R2_MIN + 0.1, y, "sin corrección", va="center",
                        fontsize=CUERPO - 2, color=C["tenue"])
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, n - 0.4)
    for i, fila in enumerate(filas):
        if isinstance(fila, str):
            axs[0].text(-0.02, n - 1 - i, fila, transform=axs[0].get_yaxis_transform(),
                        ha="right", va="center", fontweight="bold", color=C["ink2"])
            if i > 0:
                for ax in axs:
                    ax.axhline(n - 1 - i + 0.5, color=C["tenue"], lw=0.8)
    axs[0].set_yticks([n - 1 - i for i, f in enumerate(filas) if not isinstance(f, str)],
                      [f[0] for f in filas if not isinstance(f, str)])
    axs[0].set_xlim(left=0)
    axs[0].set_xlabel("NRMSE de corrida libre, test [%]")
    axs[1].set_xlim(left=0)
    axs[1].set_xlabel("error medio de parámetros [%]")
    axs[2].axvline(0, color=C["ink2"], lw=1, zorder=0)
    axs[2].set_xlim(R2_MIN, 1.1)
    axs[2].set_xticks([-2, -1, 0, 1])
    axs[2].set_xlabel("$R^2$ de la corrección contra $\\Delta f$")
    for ax, t in zip(axs, ("reproducción", "identificación", "física recuperada")):
        ax.set_title(t, loc="left", fontweight="bold", fontsize=CUERPO)
    manijas = [plt.Line2D([], [], ls="", marker=m, markersize=8, markeredgewidth=1.8,
                          color=c, markerfacecolor=c if lleno else "white")
               for _, c, lleno, m in series]
    fig.legend(manijas, [s[0] for s in series], loc="outside upper right", ncol=len(series))
    fig.suptitle(titulo, x=0.01, ha="left", fontweight="bold", fontsize=CUERPO + 2)
    fig.text(0.01, -0.01, nota, fontsize=CUERPO - 2, color=C["ink2"], va="top", wrap=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{nombre}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"escrito: {nombre}  ({pendientes} filas con corridas pendientes)")


def figura_arranque(suf, titulo):
    filas = []
    for planta, confs in PLANTAS:
        filas.append(planta)
        for rot, cero, wrm in confs:
            puntos = []
            for tag in (cero, wrm):
                d = leer(tag + suf) if tag else None
                if d is not None:
                    r2 = None if rot == "white-box" else d["r2_delta_test"]
                    puntos.append((nrmse_interp(d), d["mean_param_error"], r2))
                    if d.get("nrmse_extrap") is not None:
                        print(f"  {tag + suf:24} extrapolación box_a1.2: {d['nrmse_extrap']:5.1f} %")
                else:
                    puntos.append(None)
            if rot == "white-box":
                puntos[1] = puntos[0] and None
            filas.append((rot, puntos))
    tres_paneles(filas, f"Dataset ampliado, {titulo}: desde cero contra desde el white-box",
                 ARRANQUES, f"ampliado_arranque{suf or '_sin_ruido'}",
                 "49 escenarios: 28 de entrenamiento, 7 de validación, 14 de test. NRMSE sobre los 13 "
                 "de test que interpolan; box_a1.2 aparte. Cada corrida en su época de mejor "
                 "validación. Una semilla. La forma exacta desde cero cambia solo el β inicial: "
                 "r_i y r_e arrancan en 0,05 en las dos.")


def figura_vs_poster():
    # Rombo lleno y no círculo hueco: el hueco ya quiere decir warm start.
    series = [("dataset del póster (20 escenarios)", C["cero"], True, "o"),
              ("dataset ampliado (49 escenarios)", "#104281", True, "D")]
    for k_ruido, (suf, titulo) in enumerate(NIVELES):
        filas = []
        for planta, confs in PLANTAS:
            filas.append(planta)
            for rot, cero, _ in confs:
                clave = "e13_e1S2" if cero == "e13_e1S_cero" else cero
                tag_nuevo = (clave if clave == "e13_e1S2" else cero) + suf
                viejo = leer(POSTER[clave][[1, 2, 0][k_ruido]])
                nuevo = leer(tag_nuevo)
                puntos = []
                for d in (viejo, nuevo):
                    if d is None:
                        puntos.append(None)
                        continue
                    r2 = None if rot == "white-box" else d["r2_delta_test"]
                    puntos.append((nrmse_viejo(d), d["mean_param_error"], r2))
                filas.append((rot + (" (desde el white-box)" if clave == "e13_e1S2" else ""),
                              puntos))
        tres_paneles(filas, f"Desde cero, {titulo}: el dataset del póster contra el ampliado",
                     series, f"ampliado_vs_poster{suf or '_sin_ruido'}",
                     "NRMSE sobre los 7 escenarios de test que comparten los dos datasets, incluido "
                     "box_a1.2. El póster evalúa la última época; el ampliado, la de mejor "
                     "validación. El R² se mide sobre el test de cada dataset. Una semilla. "
                     "La forma exacta va desde el white-box en los dos, como en el póster.")


if __name__ == "__main__":
    estilo()
    for suf, titulo in NIVELES:
        figura_arranque(suf, titulo)
    figura_vs_poster()
