#!/usr/bin/env python3
# Sesgo por ventanas con el estado inicial entrenable.
#
# Mismo diseño que cola_sesgo_ventanas.py (white-box sobre wc0_2s, cinco
# realizaciones, ventanas de 5 ms), con x0 entrenable y continuidad entre
# ventanas (--x0-libre, Giampiccolo et al. 2024). Con λ = 1 la continuidad pesa
# por punto lo mismo que el término de datos; λ = 10 se corre solo con σ = 0,05,
# que es donde el sesgo es grande, para ver si el peso cambia el resultado.
#
#   python scripts/cola_x0_libre.py
#   python scripts/resumen_sesgo_ventanas.py

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import correr
from cola_sesgo_ventanas import SEMILLAS, datos


def args(lam):
    return ["--variant", "whitebox", "--window", "10", "--x0-libre", str(lam)]


if __name__ == "__main__":
    tareas = [("e17_wb_W10L1_limpio", [*datos("", 0), *args(1)], None)]
    for suf in ("_n05", "_n01"):
        for k in SEMILLAS:
            tareas.append((f"e17_wb_W10L1{suf}_s{k}", [*datos(suf, k), *args(1)], None))
    for k in SEMILLAS:
        tareas.append((f"e17_wb_W10L10_n05_s{k}", [*datos("_n05", k), *args(10)], None))
    correr(tareas, marca="DONE_x0_libre", lugares=3, esperar="DONE_x0_suave")
