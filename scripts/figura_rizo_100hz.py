#!/usr/bin/env python3
# Actividad E con la gamma continua a 120 y a 100 Hz, lazo cerrado con el
# simulador con actuador y recorte y con el estado de filtro (e25_lag_wrm,
# σ = 0,01, semilla 0). A 120 Hz el rizo del ciclo límite queda enganchado en
# fase en los dos lazos; a 100 Hz cada lazo oscila a su propia frecuencia y el
# rizo se desfasa. También imprime la frecuencia del rizo y su deriva de fase.
#
#   python scripts/figura_rizo_100hz.py  -> docs/informe_final/figuras/rizo_100_120.pdf

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))
OUT = RAIZ / "docs" / "informe_final" / "figuras" / "rizo_100_120.pdf"
TAG = "e25_lag_wrm_n01_s0"


def trazas(fg):
    os.environ["WC_FGAMMA"] = fg
    import control_distancia_y_rango as C
    C = importlib.reload(C)
    from src.wilson_cowan import WilsonCowanParams, Actuator
    from src.neural_ode import IMCController, make_perturbed_plant
    from src.neural_ode.closed_loop_ext import simulate_closed_loop_ext
    from src.neural_ode.graybox_train import ALL_P
    from esc_eval import cargar
    from exp_f6_closed_loop import fixed_de
    p = WilsonCowanParams()
    true = {k: getattr(p, k) for k in ALL_P}
    refs, ts = C.referencia("gamma50")
    n = int(round((ts[1] - ts[0]) / C.DT))
    out = {}
    for nombre, pl in (("sim", make_perturbed_plant(p, Actuator(sat=3.0, tau_act=1.0, recorte=True))),
                       ("lag", C.planta_de(cargar(C.RES / "models" / f"{TAG}.pt")[0], n))):
        ctrl = IMCController(fixed_de(true), {k: true[k] for k in C.W}, sat_ref=fixed_de(true))
        out[nombre] = simulate_closed_loop_ext(pl, ctrl, refs, t_span=ts, dt=C.DT, dt_ctrl=C.DT)
    return out


def rizo(s, m):
    from scipy.signal import butter, sosfiltfilt, hilbert
    t = s["t"]
    fs = 1000.0 / (t[1] - t[0])
    bp = butter(4, [350, 650], btype="band", fs=fs, output="sos")
    n0 = len(t) // 5
    fa, fb = (np.unwrap(np.angle(hilbert(sosfiltfilt(bp, x["E"])))) for x in (s, m))
    f = [np.diff(v[n0:]).mean() * fs / (2 * np.pi) for v in (fa, fb)]
    return f, np.ptp((fb - fa)[n0:]) / (2 * np.pi)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    for a, fg in zip(ax, ("120", "100")):
        tr = trazas(fg)
        (fs_, fm_), deriva = rizo(tr["sim"], tr["lag"])
        print(f"{fg} Hz: rizo simulador {fs_:.1f} Hz, estado de filtro {fm_:.1f} Hz, deriva {deriva:.2f} ciclos")
        t = tr["sim"]["t"]
        m = t >= 30
        a.plot(t[m], tr["sim"]["rE"][m], ":", c="k", lw=1.3, label="referencia")
        a.plot(t[m], tr["sim"]["E"][m], c="#1f77b4", lw=1.2, label="lazo cerrado con la planta")
        a.plot(t[m], tr["lag"]["E"][m], c="#d95f02", lw=1.2, label="lazo cerrado con el backbone expandido")
        a.set_ylabel("E")
        a.set_title(f"gamma continua de {fg} Hz", fontsize=11, loc="left")
    ax[-1].set_xlabel("t (ms)")
    ax[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=9.5)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=130)
    print("->", OUT)
