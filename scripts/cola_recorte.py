#!/usr/bin/env python3
# Actuador con recorte duro en ±3 (actc1_2s) en lugar de 3·tanh(·/3) (e21).
#
# La tanh comprime por debajo de 3 (2 sale como 1,75), y ese efecto no lineal
# extra no se quiere en la planta. Con el recorte, y comandos de entrenamiento
# entre 0 y 2, el actuador es un filtro de 1 ms puro en lazo abierto. La línea
# con tanh queda como punto de comparación.
#
# Las mismas configuraciones del actuador que e16, tres semillas, los dos
# ruidos. Primero la semilla 0 entera; dentro de cada semilla, white-box,
# después _wrm (el estado de filtro primero, que es la más larga) y después
# desde cero. Datos:
#
#   python scripts/esc_gen_largo.py actc1_2s
#   python scripts/esc_gen_ruido.py actc1_2s 0.01 0.05 --ref-fs 20000 [--semilla k]
#
#   python scripts/cola_recorte.py

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import RES, correr
from cola_semillas_2s import datos

V5 = ["--window", "10"]
CONFIGS = [
    ("lag",    ["--variant", "lag"]),
    ("K400f8", ["--variant", "K", "--hist", "40", "--n-fir", "8"]),
    ("H100",   ["--variant", "H", "--hist", "10"]),
    ("Hp100",  ["--variant", "Hp", "--hist", "11"]),
    ("e2B",    ["--variant", "B"]),
]
NIVELES = ["_n05", "_n01"]


if __name__ == "__main__":
    tareas = []
    for k in (0, 1, 2):
        wbs, wrm, cero = [], [], []
        for suf in NIVELES:
            wb = f"e21_e2wb{suf}_s{k}"
            wbs.append((wb, [*datos("actc1", suf, k), "--variant", "whitebox", *V5], None))
            for n, a in CONFIGS:
                wrm.append((f"e21_{n}_wrm{suf}_s{k}",
                            [*datos("actc1", suf, k), *a, *V5, "--init-params", str(RES / f"{wb}.json")], wb))
                cero.append((f"e21_{n}{suf}_s{k}", [*datos("actc1", suf, k), *a, *V5], None))
        tareas += wbs + wrm + cero
    correr(tareas, marca="DONE_recorte", lugares=3)
