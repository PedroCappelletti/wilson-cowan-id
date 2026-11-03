#!/usr/bin/env python3
# x0 entrenable desde el white-box de ventanas fijas.
#
# Con el arranque ignorante (β = 1), cola_x0_libre.py dio un sesgo de 2 a 4 %
# que es el mismo sin ruido y con ruido: viene de la optimización, no del ruido.
# La hipótesis es que al principio, con β lejos, los x0 se mueven para tapar el
# error del modelo y quedan ahí. Acá β arranca del white-box e17_wb_W10 de la
# misma realización, así que los x0 solo tienen que corregir el ruido.
#
#   python scripts/cola_x0_libre_wrm.py
#   python scripts/resumen_sesgo_ventanas.py

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import RES, correr
from cola_sesgo_ventanas import SEMILLAS, datos

ARGS = ["--variant", "whitebox", "--window", "10", "--x0-libre", "1"]


def desde(tag):
    return ["--init-params", str(RES / f"{tag}.json")]


if __name__ == "__main__":
    tareas = [("e17_wb_W10L1w_limpio", [*datos("", 0), *ARGS, *desde("e17_wb_W10_limpio")], None)]
    for suf in ("_n05", "_n01"):
        for k in SEMILLAS:
            tareas.append((f"e17_wb_W10L1w{suf}_s{k}",
                           [*datos(suf, k), *ARGS, *desde(f"e17_wb_W10{suf}_s{k}")], None))
    correr(tareas, marca="DONE_x0_libre_wrm", lugares=3)
