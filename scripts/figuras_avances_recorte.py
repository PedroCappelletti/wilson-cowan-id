#!/usr/bin/env python3
# Figuras de la presentación de avances: recorte contra tanh, x0 entrenable,
# lazo cerrado y referencia black-box. Lee los json de results/ y escribe PDF
# en docs/figuras_avances/.
#
#   python scripts/figuras_avances_recorte.py

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
RES = RAIZ / "results" / "escalado"
OUT = RAIZ / "docs" / "figuras_avances"
OUT.mkdir(parents=True, exist_ok=True)

# Paleta categórica validada (dataviz): azul, naranja, aqua, amarillo.
AZUL, NARANJA, AQUA, AMARILLO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRIS, TINTA, TINTA2 = "#9a9a96", "#1f1f1e", "#5c5c58"
SIGMA = [("_n01", "σ = 0,01", AZUL), ("_n05", "σ = 0,05", NARANJA)]

plt.rcParams.update({
    "font.family": "Arial", "font.size": 17, "axes.titlesize": 18,
    "axes.labelsize": 17, "xtick.labelsize": 16, "ytick.labelsize": 16,
    "legend.fontsize": 16.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA, "xtick.color": TINTA2,
    "ytick.color": TINTA2, "text.color": TINTA, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42,
})


def media(prefijo, suf, clave="mean_param_error"):
    vals = []
    for f in sorted(RES.glob(f"{prefijo}{suf}_s*.json")):
        v = json.loads(f.read_text())[clave]
        if v is not None:
            vals.append(v)
    return st.mean(vals) if vals else float("nan")


def media_x0(prefijo, suf, clave="mean_param_error"):
    """Segunda etapa con el modelo de partida como candidato. Si ninguna época
    valida mejor que el checkpoint de partida, vale el de partida. Las corridas
    e19 a e23 se hicieron antes de que MejorVal evaluara el punto de partida."""
    vals = []
    for f in sorted(RES.glob(f"{prefijo}{suf}_s*.json")):
        d = json.loads(f.read_text())
        s = json.loads((RES / f"{Path(d['init_model']).stem}.json").read_text())
        vals.append(s[clave] if d["nrmse_val"] >= s["nrmse_val"] else d[clave])
    return st.mean(vals)


def coma(v, _=None):
    s = f"{v:.10g}".replace(".", ",")
    return s.replace("-", "−")


def guardar(fig, nombre, y_numerico=False):
    # coma decimal; el eje y solo cuando es numérico (en las barras son categorías)
    for ax in fig.axes:
        ax.xaxis.set_major_formatter(FuncFormatter(coma))
        if y_numerico:
            ax.yaxis.set_major_formatter(FuncFormatter(coma))
    fig.savefig(OUT / f"{nombre}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{nombre}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def leyenda_arriba(fig, ax):
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.0))


def barras_h(ax, etiquetas, series, fmt="{:.1f}", xmax=None):
    """Barras horizontales agrupadas. series: [(nombre, color, valores)]."""
    n, k = len(etiquetas), len(series)
    alto = 0.8 / k
    y = np.arange(n)
    for j, (nombre, color, vals) in enumerate(series):
        pos = y - 0.4 + alto * (j + 0.5)
        ax.barh(pos, vals, height=alto * 0.86, color=color, label=nombre)
        for p, v in zip(pos, vals):
            if np.isfinite(v):
                ax.text(v + (xmax or max(max(s[2]) for s in series)) * 0.012, p,
                        fmt.format(v).replace(".", ","), va="center", fontsize=14.5, color=TINTA2)
    ax.set_yticks(y, etiquetas)
    if not ax.yaxis_inverted():          # con sharey el eje es el mismo para los dos paneles
        ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    if xmax:
        ax.set_xlim(0, xmax)


# --- 1. Forma de la saturación -------------------------------------------------
def fig_saturacion():
    x = np.linspace(-6, 6, 600)
    fig, ax = plt.subplots(figsize=(6.2, 4.42))
    ax.axvspan(0, 2, color="#ecebe7", zorder=0)
    ax.text(1.0, -5.3, "rango de\nentrenamiento", ha="center", va="bottom", fontsize=14.5, color=TINTA2)
    ax.plot(x, x, color=GRIS, lw=1.2, ls="--", label="sin saturación")
    ax.plot(x, 3 * np.tanh(x / 3), color=AMARILLO, lw=2, label=r"$3\,\tanh(P_f/3)$ (antes)")
    ax.plot(x, np.clip(x, -3, 3), color=AQUA, lw=2, label="recorte en ±3 (ahora)")
    ax.plot([2], [3 * np.tanh(2 / 3)], "o", color=AMARILLO, ms=7, mec="white", mew=1.5)
    ax.annotate("2 → 1,75", (2, 3 * np.tanh(2 / 3)), xytext=(2.6, 0.4), fontsize=14.5,
                arrowprops=dict(arrowstyle="-", color=TINTA2, lw=0.8))
    ax.set_xlabel(r"estado del filtro $P_f$")
    ax.set_ylabel(r"entrada efectiva $P_{ef}$")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-5.5, 5.5)
    ax.legend(loc="upper left")
    guardar(fig, "saturacion", y_numerico=True)


# --- 2. Lazo abierto, tanh contra recorte ----------------------------------------
MODELOS_ACT = [("white-box", "e2wb"), ("agnóstica", "e2B_wrm"), ("historia con comando actual", "H100_wrm"),
               ("filtro 8 canales", "K400f8_wrm"), ("historia sin comando actual", "Hp100_wrm"),
               ("estado de filtro", "lag_wrm")]


def fig_tanh_vs_recorte(clave, nombre, xlabel, xmax):
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.8), sharey=True)
    for ax, (suf, titulo, _) in zip(axs, SIGMA):
        et = [m for m, _ in MODELOS_ACT]
        tanh = [media(f"e16_{c}", suf, clave) for _, c in MODELOS_ACT]
        rec = [media(f"e21_{c}", suf, clave) for _, c in MODELOS_ACT]
        barras_h(ax, et, [("tanh (e16)", AMARILLO, tanh), ("recorte (e21)", AQUA, rec)], xmax=xmax)
        ax.set_title(titulo)
        ax.set_xlabel(xlabel)
    leyenda_arriba(fig, axs[0])
    guardar(fig, nombre)


# --- 3. β verdadero fijo ------------------------------------------------------------
def fig_e20():
    redes = [("agnóstica, refractariedad", "B100"), ("historia con comando actual", "H100"),
             ("filtro 8 canales", "K400f8"), ("historia sin comando actual", "Hp100"),
             ("agnóstica, actuador", "e2B")]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    series = [(t, c, [media(f"e20_{r}_vrd", s, "r2_residuo_test") for _, r in redes]) for s, t, c in SIGMA]
    y = np.arange(len(redes))
    alto = 0.4
    for j, (t, c, vals) in enumerate(series):
        pos = y - 0.2 + alto * j
        ax.barh(pos, vals, height=alto * 0.86, color=c, label=t)
        for p, v in zip(pos, vals):
            ax.text(v + (0.02 if v >= 0 else -0.02), p, f"{v:.2f}".replace(".", ",").replace("-", "−"), va="center",
                    ha="left" if v >= 0 else "right", fontsize=14.5, color=TINTA2)
    ax.axvline(0, color=TINTA2, lw=0.8)
    ax.set_yticks(y, [r for r, _ in redes])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.45, 1.12)
    ax.set_xlabel("R² contra el término faltante exacto")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    guardar(fig, "beta_verdadero_r2")


# --- 4. x0 entrenable: error de β̂ antes y después -----------------------------------
def fig_x0():
    filas = [("white-box", "e21_e2wb", "e22_e2wb_x0"), ("historia con comando actual", "e21_H100_wrm", "e22_H100_x0"),
             ("historia sin comando actual", "e21_Hp100_wrm", "e22_Hp100_x0"),
             ("filtro 8 canales", "e21_K400f8_wrm", "e22_K400f8_x0"),
             ("estado de filtro", "e21_lag_wrm", "e23_lag_x0")]
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.6), sharey=True)
    for ax, (suf, titulo, _) in zip(axs, SIGMA):
        for i, (n, a, b) in enumerate(filas):
            va, vb = media(a, suf), media_x0(b, suf)
            ax.annotate("", (vb, i), (va, i), arrowprops=dict(arrowstyle="-|>", color=TINTA2, lw=1.1,
                                                                 shrinkA=5, shrinkB=5))
            ax.plot(va, i, "o", color=GRIS, ms=8, label="sin x0 entrenable" if i == 0 else None)
            ax.plot(vb, i, "o", color=AZUL, ms=8, label="con x0 entrenable" if i == 0 else None)
            ax.text(vb, i - 0.28, coma(round(vb, 1)), ha="center", va="bottom", fontsize=14.5, color=AZUL)
        ax.set_yticks(range(len(filas)), [f[0] for f in filas])
        ax.set_ylim(len(filas) - 0.4, -0.7)
        ax.set_xlim(-3, 40)
        ax.grid(axis="y", visible=False)
        ax.set_title(titulo)
        ax.set_xlabel(r"error medio de $\hat{\beta}$ (%)")
    leyenda_arriba(fig, axs[0])
    guardar(fig, "x0_entrenable_beta")


# --- 5. Lazo cerrado --------------------------------------------------------------
def lazo():
    filas = json.loads((RAIZ / "results" / "control_distancia_rango.json").read_text(encoding="utf-8"))
    g = defaultdict(list)
    for r in filas:
        if r["finito"]:
            g[(r["perturbacion"], r["referencia"], r["planta"], r["ruido"])].append(r)
    return lambda pert, ref, pl, suf, k: st.median(x[k] for x in g[(pert, ref, pl, suf)])


PLANTAS_LAZO = [("white-box", "white-box"), ("agnóstica", "agnóstica"), ("historia con comando actual", "con red"),
                ("filtro 8 canales", "filtro entrenable"), ("historia sin comando actual", "historia pasada"),
                ("estado de filtro, recorte en ±10", "forma exacta"),
                ("estado de filtro, recorte en ±3", "forma exacta, sat 3"),
                ("estado de filtro x0, recorte en ±3", "forma exacta x0, sat 3")]


def fig_distancia(L):
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 6.2), sharey=True)
    for ax, (ref, titulo, xmax) in zip(axs, (("theta2s", "theta-gamma, 2 s", 0.19),
                                              ("gamma50", "gamma, 50 ms", 0.11))):
        series = [(t, c, [L("actc", ref, pl, s, "dist_x") for _, pl in PLANTAS_LAZO]) for s, t, c in SIGMA]
        barras_h(ax, [e for e, _ in PLANTAS_LAZO], series, fmt="{:.3f}", xmax=xmax)
        ax.set_title(titulo)
        ax.set_xlabel("distancia al lazo con el simulador")
    leyenda_arriba(fig, axs[0])
    guardar(fig, "distancia_lazos")


def fig_rmse_vs_distancia(L):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sim = L("actc", "theta2s", "simulador", "", "rmse")
    ax.axhline(sim, color=GRIS, lw=1, ls="--")
    ax.text(0.181, sim + 0.003, "simulador", ha="right", fontsize=14.5, color=TINTA2)
    for e, pl in PLANTAS_LAZO:
        x, y = L("actc", "theta2s", pl, "_n01", "dist_x"), L("actc", "theta2s", pl, "_n01", "rmse")
        ax.plot(x, y, "o", color=AZUL, ms=8, mec="white", mew=1.2)
    p = lambda pl, k: L("actc", "theta2s", pl, "_n01", k)
    etiquetas = [("white-box y agnóstica", "white-box", (0.055, 0.036)),
                 ("estado de filtro, recorte en ±10", "forma exacta", (0.06, 0.047)),
                 ("historia sin comando actual", "historia pasada", (0.058, 0.08)),
                 ("estado de filtro, recorte en ±3,\ncon y sin x0 entrenable", "forma exacta, sat 3", (0.004, 0.105)),
                 ("historia con comando actual", "con red", (0.07, 0.145)),
                 ("filtro 8 canales", "filtro entrenable", (0.125, 0.178))]
    for txt, pl, (tx, ty) in etiquetas:
        ax.annotate(txt, (p(pl, "dist_x"), p(pl, "rmse")), xytext=(tx, ty), fontsize=14.5, color=TINTA,
                    va="center", arrowprops=dict(arrowstyle="-", color=GRIS, lw=0.8, shrinkB=5))
    ax.set_xlim(0, 0.19)
    ax.set_ylim(0.03, 0.19)
    ax.set_xlabel("distancia al lazo con el simulador")
    ax.set_ylabel("RMSE de seguimiento")
    guardar(fig, "rmse_vs_distancia", y_numerico=True)


def fig_saturacion_lazo(L):
    casos = [("recorte en ±10", "actc", "forma exacta"), ("recorte en ±3", "actc", "forma exacta, sat 3"),
             ("x0 entrenable, recorte en ±3", "actc", "forma exacta x0, sat 3"),
             ("sin recorte*", "actf", "forma exacta, sin saturación"),
             ("x0 entrenable, sin recorte*", "actf", "forma exacta x0, sin saturación")]
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True)
    for ax, (ref, titulo, xmax) in zip(axs, (("theta2s", "theta-gamma, 2 s", 0.05),
                                              ("gamma50", "gamma, 50 ms", 0.11))):
        series = [(t, c, [L(p, ref, pl, s, "dist_x") for _, p, pl in casos]) for s, t, c in SIGMA]
        barras_h(ax, [c[0] for c in casos], series, fmt="{:.3f}", xmax=xmax)
        ax.set_title(titulo)
        ax.set_xlabel("distancia al lazo con el simulador")
    leyenda_arriba(fig, axs[0])
    guardar(fig, "estado_filtro_en_lazo")


def fig_trazas():
    """Lazo con cada modelo contra el lazo con el simulador, un panel por modelo.
    La ventana de 1250 a 1350 ms muestra a la historia con comando actual
    siguiendo la referencia y después trabada en un punto fijo."""
    from matplotlib.lines import Line2D
    z = np.load(RAIZ / "results" / "trazas_lazo_ejemplo.npz")
    nombres = list(z["nombres"])
    dt = 0.05                                   # ms entre muestras guardadas
    t0 = 400.0                                  # la traza arranca en el 20 % de 2 s
    t = t0 + dt * np.arange(len(z["0_E"]))
    a, b = np.searchsorted(t, 1250), np.searchsorted(t, 1350)
    sim = nombres.index("simulador")
    grupos = {"redes": [("white-box", GRIS), ("agnóstica", AQUA), ("historia 5 ms con comando actual", NARANJA),
                        ("historia 5 ms sin comando actual", "#e87ba4")],
              "filtro": [("filtro 8 canales", "#6250d6"), ("estado de filtro, recorte en ±10", AMARILLO),
                         ("estado de filtro, recorte en ±3", AZUL), ("estado de filtro x0, recorte en ±3", AZUL)]}
    for var, ylab, banda in (("E", "E", False), ("P", "comando P", True)):
        for grupo, modelos in grupos.items():
            fig, axs = plt.subplots(2, 2, figsize=(12.4, 5.8), sharex=True, sharey=True)
            for ax, (clave, color) in zip(axs.flat, modelos):
                k = nombres.index(clave)
                if banda:
                    ax.axhspan(-3, 3, color="#ecebe7", zorder=0)
                ax.plot(t[a:b], z[f"{sim}_{var}"][a:b], color=TINTA, lw=2.0)
                ax.plot(t[a:b], z[f"{k}_{var}"][a:b], color=color, lw=1.8, ls="--")
                ax.set_title(clave, loc="left", fontsize=15, pad=3)
            for ax in axs[:, 0]:
                ax.set_ylabel(ylab)
            for ax in axs[1]:
                ax.set_xlabel("tiempo (ms)")
            fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=0.6, w_pad=1.0)
            fig.legend([Line2D([], [], color=TINTA, lw=2.0), Line2D([], [], color=TINTA2, lw=1.8, ls="--")],
                       ["simulador", "modelo del título"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
            guardar(fig, f"trazas_{var}_{grupo}", y_numerico=True)


# --- 6. Black-box ---------------------------------------------------------------
def fig_blackbox():
    plantas = [("actuador con recorte", "act", "e21_e2wb", "e21_H100_wrm", "historia con comando actual", "e21_lag_wrm",
                "estado de filtro"),
               ("refractariedad", "ref", "e16_e1wb", "e16_B100_wrm", "agnóstica", "e16_e1S2_wrm",
                "forma exacta")]
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.6))
    for ax, (titulo, corto, wb, red, nred, exacta, nex) in zip(axs, plantas):
        et = ["white-box", "black-box", "black-box con historia", f"gray-box, {nred}", f"gray-box, {nex}"]
        series = [(t, c, [media(wb, s, "nrmse_test_interp"), media(f"e24_BB64_{corto}", s, "nrmse_test_interp"),
                          media(f"e24_BBH64_{corto}", s, "nrmse_test_interp"),
                          media(red, s, "nrmse_test_interp"), media(exacta, s, "nrmse_test_interp")])
                  for s, t, c in SIGMA]
        barras_h(ax, et, series, fmt="{:.2f}", xmax=9.5)
        ax.set_title(titulo)
        ax.set_xlabel("NRMSE de test (%)")
    leyenda_arriba(fig, axs[0])
    guardar(fig, "black_box")


if __name__ == "__main__":
    fig_saturacion()
    fig_tanh_vs_recorte("mean_param_error", "beta_tanh_vs_recorte", r"error medio de $\hat{\beta}$ (%)", 40)
    fig_tanh_vs_recorte("nrmse_test_interp", "nrmse_tanh_vs_recorte", "NRMSE de test (%)", 9.5)
    fig_e20()
    fig_x0()
    L = lazo()
    fig_distancia(L)
    fig_rmse_vs_distancia(L)
    fig_saturacion_lazo(L)
    fig_blackbox()
    if (RAIZ / "results" / "trazas_lazo_ejemplo.npz").exists():
        fig_trazas()
    print(f"-> {OUT}")
