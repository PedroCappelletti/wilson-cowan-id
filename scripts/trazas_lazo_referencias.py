#!/usr/bin/env python3
# Trazas completas de lazo cerrado con la referencia, para las cuatro referencias
# de control_distancia_y_rango.py y los ocho modelos del actuador con recorte
# (semilla 0, σ 0,01). A diferencia de trazas_lazo_ejemplo.py guarda la corrida
# entera y la referencia rI, rE, para graficar el seguimiento.
#
#   python scripts/trazas_lazo_referencias.py

from __future__ import annotations

import math
import sys
from multiprocessing import Pool
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import numpy as np

OUT = RAIZ / "results" / "trazas_lazo_referencias.npz"
REFS = ("theta2s", "gamma50", "theta_rango", "gamma_rango")
CASOS = [
    ("simulador", None),
    ("white-box", "e21_e2wb_n01_s0"),
    ("agnóstica", "e21_e2B_wrm_n01_s0"),
    ("historia con comando actual", "e21_H100_wrm_n01_s0"),
    ("historia sin comando actual", "e21_Hp100_wrm_n01_s0"),
    ("filtro 8 canales", "e21_K400f8_wrm_n01_s0"),
    ("estado de filtro, recorte en ±10", "e21_lag_wrm_n01_s0"),
    ("estado de filtro, recorte en ±3", "e21_lag_wrm_n01_s0@sat3"),
    ("estado de filtro x0, recorte en ±3", "e23_lag_x0_n01_s0@sat3"),
]
# --amplitud: los mismos lazos con los modelos entrenados con amplitud grande
CASOS_AMPLITUD = [
    ("simulador", None),
    ("white-box", "e25_e2wb_n01_s0"),
    ("agnóstica", "e25_e2B_wrm_n01_s0"),
    ("historia con comando actual", "e25_H100_wrm_n01_s0"),
    ("historia sin comando actual", "e25_Hp100_wrm_n01_s0"),
    ("filtro 8 canales", "e25_K400f8_wrm_n01_s0"),
    ("estado de filtro", "e25_lag_wrm_n01_s0"),
    ("black-box", "e27_BB64_act_n01_s0"),
    ("black-box con historia", "e27_BBH64_act_n01_s0"),
]
if "--amplitud" in sys.argv:
    CASOS = CASOS_AMPLITUD
    OUT = RAIZ / "results" / "trazas_lazo_referencias_amplitud.npz"
SUB = 4          # se guarda cada 0,02 ms


def una(tarea):
    ref, nombre, tag = tarea
    import torch
    torch.set_num_threads(1)
    from src.wilson_cowan import WilsonCowanParams, Actuator
    from src.neural_ode import IMCController, make_perturbed_plant
    from src.neural_ode.closed_loop_ext import simulate_closed_loop_ext
    from src.neural_ode.graybox_train import ALL_P
    from esc_eval import cargar
    from control_paper_modelos import RES, W, DT
    from exp_f6_closed_loop import fixed_de, rmse
    from control_paper_planta_modelo import planta_de
    from control_distancia_y_rango import referencia

    p = WilsonCowanParams()
    true = {k: getattr(p, k) for k in ALL_P}
    refs, t_span = referencia(ref)
    n_pasos = int(round((t_span[1] - t_span[0]) / DT))
    if tag is None:
        planta = make_perturbed_plant(p, Actuator(sat=3.0, tau_act=1.0, recorte=True))
    else:
        base, _, sat = tag.partition("@sat")
        planta = planta_de(cargar(RES / "models" / f"{base}.pt")[0], n_pasos)
        if sat:
            planta.alpha = 1e-9 if sat == "inf" else 1.0 / float(sat)
    ctrl = IMCController(fixed_de(true), {k: true[k] for k in W}, sat_ref=fixed_de(true))
    with np.errstate(over="ignore", invalid="ignore"):
        sol = simulate_closed_loop_ext(planta, ctrl, refs, t_span=t_span, dt=DT, dt_ctrl=DT)
    eI, eE = rmse(sol)
    traza = {k: sol[k][::SUB].astype(np.float32) for k in ("t", "E", "I", "P", "Q", "rE", "rI")}
    return ref, nombre, math.sqrt((eI ** 2 + eE ** 2) / 2), eI, eE, traza


if __name__ == "__main__":
    tareas = [(r, n, t) for r in REFS for n, t in CASOS]
    with Pool(3) as pool:
        res = pool.map(una, tareas)
    out = {"refs": np.array(REFS), "nombres": np.array([n for n, _ in CASOS])}
    for ref, nombre, r, eI, eE, traza in res:
        i, j = REFS.index(ref), [n for n, _ in CASOS].index(nombre)
        for k, v in traza.items():
            out[f"{i}_{j}_{k}"] = v
        out[f"{i}_{j}_rmse"] = np.array([r, eI, eE])
        print(f"{ref:12} {nombre:36} RMSE {r:.4f}  (I {eI:.4f}, E {eE:.4f})", flush=True)
    np.savez_compressed(OUT, **out)
    print(f"-> {OUT}")
