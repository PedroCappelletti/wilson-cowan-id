#!/usr/bin/env python3
# Historia pasada del comando (variante Hp) sobre los datos de 2 s (e16).
#
# Igual que H100 en cola_semillas_2s.py, pero la red ve solo los 10 retardos
# (5 ms) y no el comando actual, que entra únicamente al backbone. Así el
# modelo queda afín en u_k y el despeje cerrado del controlador del paper
# sigue valiendo. --hist 11 apila el actual y 10 retardos; la red descarta el
# actual.
#
# Solo la planta con actuador: en la de refractariedad el término faltante
# depende solo del estado y la variante tendría que dar lo mismo que la
# agnóstica. Con ruido, tres semillas, desde el white-box (primero, usa los
# e16_e2wb que ya existen) y desde cero.
#
#   python scripts/cola_historia_pasada.py      # junto con scripts/despierto.py

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import RES, correr
from cola_semillas_2s import NIVELES, SEMILLAS, datos

ARGS = ["--variant", "Hp", "--hist", "11", "--window", "10"]

if __name__ == "__main__":
    wrm, cero = [], []
    for k in SEMILLAS:
        for suf in NIVELES:
            wb = f"e16_e2wb{suf}_s{k}"
            wrm.append((f"e16_Hp100_wrm{suf}_s{k}",
                        [*datos("act1", suf, k), *ARGS, "--init-params", str(RES / f"{wb}.json")], wb))
            cero.append((f"e16_Hp100{suf}_s{k}", [*datos("act1", suf, k), *ARGS], None))
    correr(wrm + cero, marca="DONE_historia_pasada", lugares=3)
