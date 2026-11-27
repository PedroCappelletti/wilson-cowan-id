#!/usr/bin/env python3
# Trazas de la theta-gamma de 2 s para docs/controlador_planta_modelo.md con los
# modelos de amplitud grande (e25, σ 0,01, semilla 0): el lazo con el simulador
# y, en su lugar, white-box, la mejor red del documento y el backbone expandido,
# para la planta refractaria y el actuador con recorte en ±3.
#
# control_amplitud.py no guarda trazas, y trazas_lazo_referencias.py solo tiene
# el actuador. Las claves siguen el formato de control_planta_modelo_trazas.npz.
#
#   python scripts/trazas_planta_modelo_amplitud.py
#       -> results/control_planta_modelo_amplitud_trazas.npz

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import numpy as np

OUT = RAIZ / "results" / "control_planta_modelo_amplitud_trazas.npz"
CASOS = {
    "actc": [("simulador", None), ("white-box", "e25_e2wb_n01_s0"),
             ("con red", "e25_H100_wrm_n01_s0"), ("forma exacta", "e25_lag_wrm_n01_s0")],
    "refrac": [("simulador", None), ("white-box", "e25_e1wb_n01_s0"),
               ("con red", "e25_B100_wrm_n01_s0"), ("forma exacta", "e25_e1S2_wrm_n01_s0")],
}
SUB = 4          # se guarda cada 0,02 ms


def una(tarea):
    pert, nombre, tag = tarea
    import torch
    torch.set_num_threads(1)
    from src.wilson_cowan import WilsonCowanParams, Actuator, Refractoriness
    from src.neural_ode import IMCController, make_perturbed_plant
    from src.neural_ode.closed_loop_ext import simulate_closed_loop_ext
    from src.neural_ode.graybox_train import ALL_P
    from esc_eval import cargar
    from control_paper_modelos import RES, W, DT
    from exp_f6_closed_loop import fixed_de
    from control_paper_planta_modelo import planta_de
    from control_distancia_y_rango import referencia

    p = WilsonCowanParams()
    true = {k: getattr(p, k) for k in ALL_P}
    refs, t_span = referencia("theta2s")
    n_pasos = int(round((t_span[1] - t_span[0]) / DT))
    if tag is None:
        planta = make_perturbed_plant(p, {"actc": Actuator(sat=3.0, tau_act=1.0, recorte=True),
                                          "refrac": Refractoriness(r=0.10)}[pert])
    else:
        planta = planta_de(cargar(RES / "models" / f"{tag}.pt")[0], n_pasos)
    ctrl = IMCController(fixed_de(true), {k: true[k] for k in W}, sat_ref=fixed_de(true))
    with np.errstate(over="ignore", invalid="ignore"):
        sol = simulate_closed_loop_ext(planta, ctrl, refs, t_span=t_span, dt=DT, dt_ctrl=DT)
    return pert, nombre, {k: sol[k][::SUB].astype(np.float32) for k in ("t", "E", "I", "P", "Q", "rE", "rI")}


if __name__ == "__main__":
    tareas = [(pert, n, t) for pert, casos in CASOS.items() for n, t in casos]
    with Pool(3) as pool:
        res = pool.map(una, tareas)
    out = {}
    for pert, nombre, traza in res:
        for k, v in traza.items():
            out[f"theta2s|{pert}|{nombre}|{k}"] = v
        print(f"{pert:7} {nombre:14} listo", flush=True)
    np.savez_compressed(OUT, **out)
    print(f"-> {OUT}")
