#!/usr/bin/env python3
# Sesgo por ventanas con el estado inicial suavizado.
#
# Mismo diseño que cola_sesgo_ventanas.py (white-box sobre wc0_2s, cinco
# realizaciones por nivel, ventanas de 5 ms), con el x0 de cada ventana tomado
# como la media centrada de 5 muestras (--x0-suave 5). Si el sesgo entra por el
# ruido del estado inicial, debería bajar a más o menos un quinto. La corrida
# sin ruido mide lo que la media le cuesta a la señal.
#
#   python scripts/cola_x0_suave.py
#   python scripts/resumen_sesgo_ventanas.py

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import correr
from cola_sesgo_ventanas import SEMILLAS, datos

ARGS = ["--variant", "whitebox", "--window", "10", "--x0-suave", "5"]


if __name__ == "__main__":
    tareas = [("e17_wb_W10s5_limpio", [*datos("", 0), *ARGS], None)]
    for suf in ("_n01", "_n05"):
        for k in SEMILLAS:
            tareas.append((f"e17_wb_W10s5{suf}_s{k}", [*datos(suf, k), *ARGS], None))
    correr(tareas, marca="DONE_x0_suave", lugares=3, esperar="DONE_sesgo_ventanas")
