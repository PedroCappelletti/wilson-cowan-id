#!/usr/bin/env python3
# Semillas 1 y 2 de las configuraciones del actuador que en e16 tienen una sola.
#
# Primero la agnóstica g(I, E) (e2B), que va en la tabla de arquitecturas del
# paper. Después las ablaciones de la historia del comando: 20 ms (H400), el
# filtro entrenable sobre 20 ms (K400) y con 16 coeficientes (K400f16). Mismos
# argumentos que la semilla 0 de cola_semillas_2s.py; el _wrm primero.
#
#   python scripts/cola_semillas_actuador.py

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import RES, correr
from cola_semillas_2s import datos, V5

CONFIGS = [
    ("e2B",     ["--variant", "B", *V5]),
    ("H400",    ["--variant", "H", "--hist", "40", *V5]),
    ("K400",    ["--variant", "K", "--hist", "40", *V5]),
    ("K400f16", ["--variant", "K", "--hist", "40", "--n-fir", "16", *V5]),
]
NIVELES = ["_n05", "_n01"]


if __name__ == "__main__":
    tareas = []
    for n, a in CONFIGS:
        for k in (1, 2):
            for suf in NIVELES:
                wb = RES / f"e16_e2wb{suf}_s{k}.json"
                tareas.append((f"e16_{n}_wrm{suf}_s{k}", [*datos("act1", suf, k), *a, "--init-params", str(wb)], None))
                tareas.append((f"e16_{n}{suf}_s{k}", [*datos("act1", suf, k), *a], None))
    correr(tareas, marca="DONE_semillas_actuador", lugares=3)
