#!/usr/bin/env python3
# Dos lotes en una cola, para una noche.
#
#   e23  estado de filtro (lag) con x0 entrenable sobre el actuador con recorte.
#        Completa e22 con la forma exacta: arranca del e21 lag_wrm entero
#        (--init-model) y libera solo los x0 de (I, E). El estado del filtro sale
#        de filtrar el comando, así que no hace falta liberarlo.
#   e24  referencia black-box: ẋ = g(I, E, P, Q) sin Wilson-Cowan (BB) y la misma
#        con la historia del comando de H100 (BBH). Las dos plantas, refractariedad
#        y recorte. Sin warm start, porque no hay backbone de donde arrancar.
#        Red de 64 y 3000 épocas: el doble que las correcciones, porque la red
#        tiene que aprender el campo entero y no un término chico.
#
# Prioridad: e23 entero (seis corridas), después e24 por semilla, σ 0,05 antes
# que 0,01. Espera a que termine la prueba de lazo cerrado sin saturación, para
# no pasar de tres procesos.
#
#   python scripts/cola_blackbox_y_lag_x0.py

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import RES, correr
from cola_semillas_2s import datos

V5 = ["--window", "10"]
NIVELES = ["_n05", "_n01"]
BB = {
    "BB64":    ["--variant", "BB", "--hidden", "64", "--epochs", "3000"],
    "BBH64":   ["--variant", "BBH", "--hist", "10", "--hidden", "64", "--epochs", "3000"],
}


if __name__ == "__main__":
    tareas = []
    for k in (0, 1, 2):
        for suf in NIVELES:
            pt = RES / "models" / f"e21_lag_wrm{suf}_s{k}.pt"
            assert pt.exists(), pt
            tareas.append((f"e23_lag_x0{suf}_s{k}",
                           [*datos("actc1", suf, k), "--variant", "lag", *V5,
                            "--x0-libre", "1", "--init-model", str(pt)], None))
    for k in (0, 1, 2):
        for suf in NIVELES:
            for planta, corto in (("actc1", "act"), ("refrac1", "ref")):
                for n, a in BB.items():
                    tareas.append((f"e24_{n}_{corto}{suf}_s{k}",
                                   [*datos(planta, suf, k), *a, *V5], None))
    correr(tareas, marca="DONE_blackbox_y_lag_x0", lugares=3,
           esperar="DONE_control_sin_sat")
