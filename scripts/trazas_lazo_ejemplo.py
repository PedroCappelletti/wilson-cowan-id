#!/usr/bin/env python3
# Trazas de lazo cerrado para las figuras de avances: theta-gamma de 2 s sobre
# el actuador con recorte, semilla 0 y σ 0,01. control_distancia_y_rango.py no
# guarda las trazas, así que acá se vuelven a correr las cinco plantas.
#
#   python scripts/trazas_lazo_ejemplo.py

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import numpy as np

from control_distancia_y_rango import una

OUT = RAIZ / "results" / "trazas_lazo_ejemplo.npz"
CASOS = [
    ("simulador", None),
    ("white-box", "e21_e2wb_n01_s0"),
    ("historia 5 ms", "e21_H100_wrm_n01_s0"),
    ("estado de filtro, nivel 10", "e21_lag_wrm_n01_s0"),
    ("estado de filtro x0, nivel 3", "e23_lag_x0_n01_s0@sat3"),
]


def correr(caso):
    nombre, tag = caso
    fila, traza = una(("theta2s", "actc", nombre, tag, "_n01", 0))
    return nombre, fila["rmse"], traza


if __name__ == "__main__":
    from control_distancia_y_rango import referencia
    with Pool(3) as pool:
        res = pool.map(correr, CASOS)
    refs, _ = referencia("theta2s")
    out = {}
    for k, (nombre, rmse, traza) in enumerate(res):
        for v in ("E", "I", "P", "Q"):
            out[f"{k}_{v}"] = traza[v]
        print(f"{nombre:30} RMSE {rmse:.4f}")
    out["nombres"] = np.array([r[0] for r in res])
    np.savez(OUT, **out)
    print(f"-> {OUT}")
