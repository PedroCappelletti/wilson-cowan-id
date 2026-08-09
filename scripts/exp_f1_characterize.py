#!/usr/bin/env python3
# =============================================================================
#  F1 — CARACTERIZAR LA PERTURBACION ELEGIDA
# =============================================================================
#
#  Antes de identificar nada hay que entender QUE le hace la incertidumbre al
#  sistema. Este script contesta tres cosas:
#
#   1. ¿Cuanto deforma la trayectoria cada nivel de eps, y sigue el sistema en
#      un regimen sano (oscilando, con rango de E parecido)?
#      Criterio: la deformacion tiene que quedar POR ENCIMA del ruido de
#      observacion que el proyecto estudia (sigma = 0.01 - 0.10), si no la
#      perturbacion es indistinguible del ruido y no hay nada que aprender.
#
#   2. ¿Que tamano tiene el termino faltante Delta f respecto del campo de
#      Wilson-Cowan? Es la carga de trabajo de g_φ.
#
#   3. ¿Como se ve? Figuras: trayectorias, retrato de fase, el estimulo
#      comandado vs el que realmente llega, y el mapa de Delta f.
#
#  USO:  python scripts/exp_f1_characterize.py
# =============================================================================

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np

OUT = Path("results/uncertainty/f1_caracterizacion.json")
FIGDIR = Path("results/figures")
DATA_DIR = Path("data/processed/uncertain")
EPS_GRID = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
SIGMA_REF = 0.02          # ruido de observacion tipico de los barridos del proyecto


def carga(eps):
    p = DATA_DIR / f"eps{eps:g}.npz"
    return np.load(p, allow_pickle=True) if p.exists() else None


def main():
    base = carga(0.0)
    if base is None:
        print("Falta data/processed/uncertain/eps0.npz — corre gen_uncertain_dataset.py")
        return

    filas = []
    print("=== F1 · caracterizacion del par refractariedad + actuador ===")
    print(f"    referencia de ruido de observacion: sigma = {SIGMA_REF}\n")
    hdr = (f"{'eps':>5} {'D_traj%':>8} {'D_abs':>8} {'xSigma':>7} "
           f"{'|Df|/|f|%':>10} {'E_max':>7} {'cruces':>7}")
    print(hdr); print("-" * len(hdr))

    for eps in EPS_GRID:
        d = carga(eps)
        if d is None:
            continue
        dE = d["E"] - base["E"]
        dI = d["I"] - base["I"]
        sig = base["E"] - base["E"].mean(axis=1, keepdims=True)
        d_abs = float(np.sqrt(np.mean(dE ** 2 + dI ** 2) / 2))
        d_rel = 100.0 * d_abs / float(np.sqrt(np.mean(sig ** 2)))

        # tamano del termino faltante respecto del campo nominal
        df = np.sqrt(d["dfI"] ** 2 + d["dfE"] ** 2).mean()
        # |f_WC| aproximado por la derivada numerica de la trayectoria base
        dt = float(base["dt"])
        f_nom = np.sqrt((np.diff(base["I"], axis=1) / dt) ** 2 +
                        (np.diff(base["E"], axis=1) / dt) ** 2).mean()
        df_rel = 100.0 * df / f_nom

        e = d["E"] - d["E"].mean(axis=1, keepdims=True)
        cruces = int(np.mean([(np.diff(np.sign(row)) != 0).sum() for row in e]))
        print(f"{eps:5.2f} {d_rel:8.2f} {d_abs:8.4f} {d_abs/SIGMA_REF:7.1f} "
              f"{df_rel:10.2f} {float(d['E'].max()):7.3f} {cruces:7d}")
        filas.append({"eps": eps, "D_traj_pct": d_rel, "D_abs": d_abs,
                      "veces_sigma": d_abs / SIGMA_REF, "df_rel_pct": df_rel,
                      "E_max": float(d["E"].max()), "cruces": cruces})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(filas, indent=2))
    _figuras(base)
    print(f"\n  -> {OUT}")
    print(f"  -> {FIGDIR}/f1_caracterizacion.png")


def _figuras(base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d1 = carga(1.0)
    if d1 is None:
        return
    labels = [str(x) for x in base["labels"]]
    s = labels.index("aprbs_0") if "aprbs_0" in labels else 0
    t = base["t"]
    n = len(t) // 2      # primera mitad, para que se vea el detalle

    fig, ax = plt.subplots(2, 2, figsize=(13, 7.5))

    # (1) trayectorias: nominal vs perturbada
    ax[0, 0].plot(t[:n], base["E"][s, :n], lw=1.4, label="E — WC puro", color="#1f4e79")
    ax[0, 0].plot(t[:n], d1["E"][s, :n], lw=1.4, label="E — planta real (eps=1)",
                  color="#d62728")
    ax[0, 0].set_title("La perturbacion esta SIEMPRE activa, no es un golpe")
    ax[0, 0].set_xlabel("t [ms]"); ax[0, 0].set_ylabel("E")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)

    # (2) retrato de fase
    ax[0, 1].plot(base["I"][s], base["E"][s], lw=0.8, color="#1f4e79", label="WC puro")
    ax[0, 1].plot(d1["I"][s], d1["E"][s], lw=0.8, color="#d62728", label="planta real")
    ax[0, 1].set_title("Retrato de fase: el ciclo se deforma, no desaparece")
    ax[0, 1].set_xlabel("I"); ax[0, 1].set_ylabel("E")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3)

    # (3) estimulo comandado vs efectivo (la segunda entrada de incertidumbre)
    ax[1, 0].plot(t[:n], d1["P"][s, :n], lw=1.2, label="P comandado (lo que ve el modelo)",
                  color="#1f4e79")
    ax[1, 0].plot(t[:n], d1["P_eff"][s, :n], lw=1.2,
                  label="P efectivo (lo que llega a la neurona)", color="#ff7f0e")
    ax[1, 0].set_title("Canal de actuacion: comandado != efectivo")
    ax[1, 0].set_xlabel("t [ms]"); ax[1, 0].set_ylabel("P")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

    # (4) el termino faltante que g_φ tiene que aprender
    sc = ax[1, 1].scatter(d1["I"][s, ::5], d1["E"][s, ::5],
                          c=d1["dfE"][s, ::5], s=4, cmap="RdBu_r")
    ax[1, 1].set_title("Delta f (componente dE): lo que le falta al modelo")
    ax[1, 1].set_xlabel("I"); ax[1, 1].set_ylabel("E")
    fig.colorbar(sc, ax=ax[1, 1], fraction=0.046)

    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "f1_caracterizacion.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
