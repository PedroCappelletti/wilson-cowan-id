#!/usr/bin/env python3
# Pruebas del controlador de Martínez et al. (2022), tal cual, sobre el
# simulador y los modelos gray-box.
#
# Controlador: IMCController (src/neural_ode/closed_loop.py), sin tope ni
# anti-windup, como en el paper. Lo único que cambia entre filas es con qué
# parámetros (y con qué corrección, cuando el despeje cerrado la admite) se
# construye:
#
#   ideal         planta = Wilson-Cowan sin perturbación, θ verdaderos
#   θ verdaderos  planta perturbada, θ verdaderos (no sabe de la perturbación)
#   white-box     θ̂ del white-box
#   con red       la mejor corrección con red por validación:
#                   refractariedad: agnóstica g(I, E) con warm start, con su θ̂
#                     sola y cancelando g (el despeje cerrado lo admite)
#                   actuador: historia de 5 ms con warm start, solo θ̂ (su red
#                     depende del comando y el despeje cerrado no la admite)
#   forma exacta  backbone expandido: refractariedad con (1 − r x) y saturación;
#                 en el actuador, el estado de filtro, solo con su θ̂
#
# Modelos de la tanda e16 (datos de 2 s), σ = 0,01 y 0,05, semillas 0, 1 y 2.
# Referencias: theta-gamma de 2 s (fig. 7 del paper) y gamma de 50 ms.
# Error: RMSE entre E, I y sus referencias, promedio de las dos poblaciones,
# sin el primer quinto de la simulación.
#
#   python scripts/control_paper_modelos.py [procesos]

from __future__ import annotations

import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import numpy as np

RES = RAIZ / "results" / "escalado"
OUT = RAIZ / "results" / "control_paper_modelos.json"
TRAZAS = RAIZ / "results" / "control_paper_modelos_trazas.npz"
W = ("wEE", "wEI", "wIE", "wII")
DT = 0.005
REFS = ("theta2s", "gamma50")
NIVELES = ("_n01", "_n05")
SEMILLAS = (0, 1, 2)
# La mejor corrección con red por NRMSE de validación (results/resumen_2s.md).
CON_RED = {("refrac", "_n01"): "B100_wrm", ("refrac", "_n05"): "B400_wrm",
           ("act", "_n01"): "H100_wrm", ("act", "_n05"): "H100_wrm"}
EXACTA = {"refrac": "e1S2_wrm", "act": "lag"}
WB = {"refrac": "e1wb", "act": "e2wb"}


def tareas():
    out = []
    for ref in REFS:
        out.append((ref, "ideal", "ideal", None, "θ verdaderos", "", -1))
        for planta in ("refrac", "act"):
            out.append((ref, planta, "θ verdaderos", None, "θ verdaderos", "", -1))
            for suf in NIVELES:
                for k in SEMILLAS:
                    out.append((ref, planta, "white-box", f"e16_{WB[planta]}{suf}_s{k}", "solo θ̂", suf, k))
                    red = f"e16_{CON_RED[(planta, suf)]}{suf}_s{k}"
                    out.append((ref, planta, "con red", red, "solo θ̂", suf, k))
                    if planta == "refrac":
                        out.append((ref, planta, "con red", red, "θ̂ + g", suf, k))
                    ex = f"e16_{EXACTA[planta]}{suf}_s{k}"
                    out.append((ref, planta, "forma exacta", ex,
                                "θ̂ + forma" if planta == "refrac" else "solo θ̂", suf, k))
    return out


def una(t):
    import torch
    torch.set_num_threads(1)
    from src.wilson_cowan import WilsonCowanParams, Refractoriness, Actuator, NoPerturbation
    from src.neural_ode import (IMCController, theta_gamma_refs, make_perturbed_plant,
                                make_graybox_correction)
    from src.neural_ode.closed_loop_ext import simulate_closed_loop_ext, theta_gamma_ventana_refs
    from src.neural_ode.graybox_train import ALL_P
    from esc_eval import cargar
    from exp_f6_closed_loop import fixed_de, rmse

    ref, planta_n, modelo, tag, uso, suf, k = t
    p = WilsonCowanParams()
    true = {kk: getattr(p, kk) for kk in ALL_P}
    pert = {"ideal": NoPerturbation(), "refrac": Refractoriness(r=0.10),
            "act": Actuator(sat=3.0, tau_act=1.0)}[planta_n]
    planta = make_perturbed_plant(p, pert)
    SAT = fixed_de(true)
    if tag is None:
        ph, ctrl_kw = true, {}
    else:
        m, _ = cargar(RES / "models" / f"{tag}.pt")
        ph = m.params_dict()
        ctrl_kw = {}
        if uso == "θ̂ + g":
            ctrl_kw["g_hat"] = make_graybox_correction(m)
        elif uso == "θ̂ + forma":
            ctrl_kw["structured"] = m.structured_dict()
    ctrl = IMCController(fixed_de(ph), {kk: ph[kk] for kk in W}, sat_ref=SAT, **ctrl_kw)
    if ref == "theta2s":
        refs, t_span = theta_gamma_ventana_refs()[0], (0.0, 2000.0)
    else:
        refs, t_span = theta_gamma_refs(120.0, True), (0.0, 50.0)
    sol = simulate_closed_loop_ext(planta, ctrl, refs, t_span=t_span, dt=DT, dt_ctrl=DT)
    eI, eE = rmse(sol)
    eb = float(np.mean([100 * abs(ph[kk] - true[kk]) / abs(true[kk]) for kk in ALL_P]))
    fila = {"referencia": ref, "planta": planta_n, "modelo": modelo, "uso": uso, "tag": tag,
            "ruido": suf, "semilla": k, "rmse": math.sqrt((eI ** 2 + eE ** 2) / 2),
            "rmse_I": eI, "rmse_E": eE, "error_beta": eb,
            "P_rango": [float(sol["P"].min()), float(sol["P"].max())],
            "Q_rango": [float(sol["Q"].min()), float(sol["Q"].max())]}
    # Trazas para la figura: semilla 0, σ = 0,01, y los casos sin modelo.
    traza = None
    if k in (0, -1) and suf in ("", "_n01"):
        paso = 20 if ref == "theta2s" else 2
        traza = {kk: sol[kk][::paso].astype(np.float32) for kk in ("t", "E", "I", "rE", "rI", "P")}
    return fila, traza


def main():
    procesos = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ts = tareas()
    print(f"{len(ts)} simulaciones, {procesos} procesos", flush=True)
    filas, trazas = [], {}
    with Pool(procesos) as pool:
        for fila, traza in pool.imap_unordered(una, ts):
            filas.append(fila)
            print(f"  {fila['referencia']:8} {fila['planta']:7} {fila['modelo']:13} {fila['uso']:12} "
                  f"{fila['ruido']:5} s{fila['semilla']:>2}  RMSE {fila['rmse']:.4f}", flush=True)
            if traza is not None:
                clave = f"{fila['referencia']}|{fila['planta']}|{fila['modelo']}|{fila['uso']}"
                for kk, v in traza.items():
                    trazas[f"{clave}|{kk}"] = v
    OUT.write_text(json.dumps(filas, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(TRAZAS, **trazas)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
