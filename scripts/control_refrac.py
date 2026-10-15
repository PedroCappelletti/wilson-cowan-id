#!/usr/bin/env python3
# Lazo cerrado sobre la planta con refractariedad, con los modelos de la tanda e13.
#
# La pregunta: si la corrección agnóstica con warm start identifica β mucho
# mejor que desde cero (4 a 6 % contra 18 %), ¿eso se traduce en controlar
# mejor? Todos los controladores enfrentan la misma planta, Wilson-Cowan con
# refractariedad r = 0,10, y los mismos topes de actuación. Cambia solo con qué
# modelo se construye cada uno:
#
#   oráculo                 θ verdaderos, sin saber de la refractariedad
#   oráculo + forma         θ y r verdaderos, cancelación exacta
#   white-box               θ̂ del white-box
#   agnóstica, solo θ̂       θ̂ de la corrección, sin cancelar g
#   agnóstica, θ̂ + g        θ̂ y la corrección cancelada en el controlador
#   forma exacta            θ̂, r̂ y saturación del modelo
#
# para los tres niveles de ruido y los dos arranques. Mismo controlador IMC,
# referencias theta-gamma de 120 Hz, 50 ms, dt = 0,005 ms (como F6).
#
#   python scripts/control_refrac.py

from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import numpy as np
import torch

torch.set_num_threads(1)

from src.wilson_cowan import WilsonCowanParams, Refractoriness
from src.neural_ode import (IMCController, simulate_closed_loop, theta_gamma_refs,
                            make_perturbed_plant, make_graybox_correction)
from src.neural_ode.graybox_train import ALL_P
from esc_eval import cargar
from exp_f6_closed_loop import fixed_de, rmse

RES = RAIZ / "results" / "escalado"
OUT = RAIZ / "results" / "control_refrac.json"
T_SPAN, DT = (0.0, 50.0), 0.005
W = ("wEE", "wEI", "wIE", "wII")
NIVELES = [("", "sin ruido"), ("_n01", "σ = 0,01"), ("_n05", "σ = 0,05")]


def error_beta(ph, true):
    return float(np.mean([100 * abs(ph[k] - true[k]) / abs(true[k]) for k in ALL_P]))


def corre(fixed, weights, planta, sat_ref, g_hat=None, structured=None):
    ctrl = IMCController(fixed, weights, g_hat=g_hat, sat_ref=sat_ref, structured=structured)
    sol = simulate_closed_loop(planta, ctrl, theta_gamma_refs(120.0, True), t_span=T_SPAN, dt=DT)
    eI, eE = rmse(sol)
    sat = max(ctrl.n_sat_i, ctrl.n_sat_e) / max(ctrl.n_pasos, 1)
    ok = bool(np.isfinite(eI) and np.isfinite(eE) and sat < 0.5)
    return {"rmse_I": eI, "rmse_E": eE, "rmse": float(np.sqrt((eI**2 + eE**2) / 2)),
            "frac_saturado": float(sat), "ok": ok}


def main():
    p_true = WilsonCowanParams()
    true = {k: getattr(p_true, k) for k in ALL_P}
    planta = make_perturbed_plant(p_true, Refractoriness(r=0.10))
    SAT = fixed_de(true)
    filas = []

    def fila(nivel, modelo, arranque, res, eb):
        res.update(nivel=nivel, modelo=modelo, arranque=arranque, error_beta=eb)
        filas.append(res)
        marca = "" if res["ok"] else "   <-- divergió o saturó"
        print(f"  {nivel:9} {modelo:24} {arranque:12} β {eb:5.1f} %  "
              f"RMSE {res['rmse']:.4f} (I {res['rmse_I']:.4f}, E {res['rmse_E']:.4f}){marca}",
              flush=True)

    print("=== lazo cerrado, planta con refractariedad r = 0,10 ===")
    fila("—", "oráculo", "—", corre(fixed_de(true), {k: true[k] for k in W}, planta, SAT), 0.0)
    fila("—", "oráculo + forma", "—",
         corre(fixed_de(true), {k: true[k] for k in W}, planta, SAT,
               structured={"r_i": 0.10, "r_e": 0.10, "sat": 1e6}), 0.0)

    for suf, nivel in NIVELES:
        wb = json.loads((RES / f"e13_e1wb{suf}.json").read_text())["params"]
        fila(nivel, "white-box", "—",
             corre(fixed_de(wb), {k: wb[k] for k in W}, planta, SAT), error_beta(wb, true))
        for nombre, tags in [("agnóstica 5 ms", ("e13_B100", "e13_B100_wrm")),
                             ("agnóstica 20 ms", ("e13_B400", "e13_B400_wrm"))]:
            for arranque, tag in zip(("desde cero", "warm start"), tags):
                m, _ = cargar(RES / "models" / f"{tag}{suf}.pt")
                ph = m.params_dict()
                fx, wh, eb = fixed_de(ph), {k: ph[k] for k in W}, error_beta(ph, true)
                fila(nivel, nombre + ", solo θ̂", arranque, corre(fx, wh, planta, SAT), eb)
                fila(nivel, nombre + ", θ̂ + g", arranque,
                     corre(fx, wh, planta, SAT, g_hat=make_graybox_correction(m)), eb)
        for arranque, tag in (("desde cero", "e13_e1S_cero"), ("warm start", "e13_e1S2")):
            m, _ = cargar(RES / "models" / f"{tag}{suf}.pt")
            ph = m.params_dict()
            fila(nivel, "forma exacta", arranque,
                 corre(fixed_de(ph), {k: ph[k] for k in W}, planta, SAT,
                       structured=m.structured_dict()), error_beta(ph, true))

    OUT.write_text(json.dumps(filas, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  -> {OUT}")


if __name__ == "__main__":
    main()
