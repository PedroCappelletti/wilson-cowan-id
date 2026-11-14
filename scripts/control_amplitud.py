#!/usr/bin/env python3
# Lazo cerrado con los modelos entrenados con amplitud grande (e25, e26, e27).
#
# Mismo controlador, mismas cuatro referencias y misma distancia entre lazos que
# control_distancia_y_rango.py, que tiene los resultados de los modelos
# entrenados con [0, 2]. Plantas: actuador con recorte (actc) y refractaria.
# En el actuador, el estado de filtro ya trae el nivel del recorte identificado
# (1/α̂ ≈ 3), así que no hace falta fijarlo.
#
#   python scripts/control_amplitud.py [procesos]   -> results/control_amplitud.json

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

from control_distancia_y_rango import una, distancias, REFS  # noqa: E402
from control_paper_modelos import NIVELES, SEMILLAS, RES  # noqa: E402

OUT = RAIZ / "results" / "control_amplitud.json"
MODELOS = {
    "actc": [("white-box", "e25_e2wb"), ("agnóstica", "e25_e2B_wrm"),
             ("historia con comando actual", "e25_H100_wrm"),
             ("historia con comando actual, desde cero", "e25_H100"),
             ("historia sin comando actual", "e25_Hp100_wrm"),
             ("filtro 8 canales", "e25_K400f8_wrm"),
             ("estado de filtro", "e25_lag_wrm"), ("estado de filtro x0", "e26_lag_x0"),
             ("black-box", "e27_BB64_act"), ("black-box con historia", "e27_BBH64_act")],
    "refrac": [("white-box", "e25_e1wb"), ("agnóstica", "e25_B100_wrm"),
               ("forma exacta", "e25_e1S2_wrm"), ("forma exacta x0", "e26_e1S2_x0"),
               ("black-box", "e27_BB64_ref"), ("black-box con historia", "e27_BBH64_ref")],
}


def tareas():
    out = []
    for ref in REFS:
        for planta, mods in MODELOS.items():
            out.append((ref, planta, "simulador", None, "", -1))
            for suf in NIVELES:
                for k in SEMILLAS:
                    for nombre, pre in mods:
                        tag = f"{pre}{suf}_s{k}"
                        if (RES / "models" / f"{tag}.pt").exists():
                            out.append((ref, planta, nombre, tag, suf, k))
    return out


if __name__ == "__main__":
    procesos = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    ts = tareas()
    print(f"{len(ts)} simulaciones, {procesos} procesos", flush=True)
    filas, trazas = [], []
    with Pool(procesos) as pool:
        for fila, traza in pool.imap_unordered(una, ts):
            filas.append(fila)
            trazas.append(traza)
            print(f"  {fila['referencia']:11} {fila['perturbacion']:6} {fila['planta']:40} "
                  f"{fila['ruido']:5} s{fila['semilla']:>2}  RMSE {fila['rmse']:.4f}", flush=True)
    distancias(filas, trazas)
    OUT.write_text(json.dumps(filas, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {OUT}")
