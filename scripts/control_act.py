#!/usr/bin/env python3
# Lazo cerrado sobre la planta con actuador, con los modelos de la tanda e13.
#
# Mismo protocolo que control_refrac.py, contra Wilson-Cowan con el actuador de
# τ = 1 ms y saturación 3. Solo entran los modelos que el IMC acepta sin código
# nuevo (ver la nota «Conectar el controlador - opciones» del vault):
#
#   oráculo              θ verdaderos, sin saber del actuador
#   white-box            θ̂ del white-box
#   agnóstica            θ̂ solo, y θ̂ + g(I, E) cancelada
#   historia, filtro,    θ̂ solo: su corrección depende del comando actual y
#   estado de filtro     el IMC no la puede cancelar todavía
#
#   python scripts/control_act.py

from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import torch

torch.set_num_threads(1)

from src.wilson_cowan import WilsonCowanParams, Actuator
from src.neural_ode import make_perturbed_plant, make_graybox_correction
from src.neural_ode.graybox_train import ALL_P
from esc_eval import cargar
from exp_f6_closed_loop import fixed_de
from control_refrac import corre, error_beta, NIVELES, RES, W

OUT = RAIZ / "results" / "control_act.json"
SOLO_BETA = [("historia, 100", "e13_H100"), ("historia, 400", "e13_H400"),
             ("filtro, 4 canales", "e13_K400"), ("filtro, 8 canales", "e13_K400f8"),
             ("filtro, 16 canales", "e13_K400f16"), ("estado de filtro", "e13_lag")]


def main():
    p_true = WilsonCowanParams()
    true = {k: getattr(p_true, k) for k in ALL_P}
    planta = make_perturbed_plant(p_true, Actuator(sat=3.0, tau_act=1.0))
    SAT = fixed_de(true)
    filas = []

    def fila(nivel, modelo, arranque, res, eb):
        res.update(nivel=nivel, modelo=modelo, arranque=arranque, error_beta=eb)
        filas.append(res)
        marca = "" if res["ok"] else "   <-- divergió o saturó"
        print(f"  {nivel:9} {modelo:28} {arranque:12} β {eb:5.1f} %  "
              f"RMSE {res['rmse']:.4f} (I {res['rmse_I']:.4f}, E {res['rmse_E']:.4f}){marca}",
              flush=True)

    print("=== lazo cerrado, planta con actuador (τ = 1 ms, sat = 3) ===")
    fila("—", "oráculo", "—", corre(fixed_de(true), {k: true[k] for k in W}, planta, SAT), 0.0)

    for suf, nivel in NIVELES:
        wb = json.loads((RES / f"e13_e2wb{suf}.json").read_text())["params"]
        fila(nivel, "white-box", "—",
             corre(fixed_de(wb), {k: wb[k] for k in W}, planta, SAT), error_beta(wb, true))
        for arranque, tag in (("desde cero", "e13_e2B"), ("warm start", "e13_e2B_wrm")):
            m, _ = cargar(RES / "models" / f"{tag}{suf}.pt")
            ph = m.params_dict()
            fx, wh, eb = fixed_de(ph), {k: ph[k] for k in W}, error_beta(ph, true)
            fila(nivel, "agnóstica, solo θ̂", arranque, corre(fx, wh, planta, SAT), eb)
            fila(nivel, "agnóstica, θ̂ + g", arranque,
                 corre(fx, wh, planta, SAT, g_hat=make_graybox_correction(m)), eb)
        for nombre, base in SOLO_BETA:
            for arranque, tag in (("desde cero", base), ("warm start", base + "_wrm")):
                ph = json.loads((RES / f"{tag}{suf}.json").read_text())["params"]
                fila(nivel, nombre + ", solo θ̂", arranque,
                     corre(fixed_de(ph), {k: ph[k] for k in W}, planta, SAT),
                     error_beta(ph, true))

    OUT.write_text(json.dumps(filas, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  -> {OUT}")


if __name__ == "__main__":
    main()
