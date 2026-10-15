#!/usr/bin/env python3
# Semillas de las configuraciones clave del dataset ampliado (tanda e15).
#
# Las e13 son de una semilla, y varias comparaciones dependen de diferencias
# chicas. La semilla k cambia dos cosas a la vez: la realización del ruido
# (dataset <planta>_amp_n0x_s<k>, de esc_gen_ruido.py --semilla k) y la
# inicialización de la red (--seed k). La primera es la única que mueve a los
# modelos deterministas: white-box, forma exacta y estado de filtro.
#
# La semilla 0 son las e13 que ya existen. Acá van la 1 y la 2, con ruido, desde
# cero y desde el white-box de la misma planta, ruido y semilla. El white-box
# primero, después el arranque desde el white-box, después desde cero.
#
# Tags: e15_<nombre>[_wrm]<ruido>_s<k>.

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import RES, correr

K8 = ["--variant", "K", "--hist", "400", "--n-fir", "8"]
CONFIGS = [   # (nombre, planta, argumentos, se corre desde cero)
    ("B400",   "refrac1", ["--variant", "B", "--window", "400"], True),
    ("K400f8", "act1",    K8, True),
    ("lag",    "act1",    ["--variant", "lag"], True),
    ("H100",   "act1",    ["--variant", "H", "--hist", "100"], True),
    ("B100",   "refrac1", ["--variant", "B", "--window", "100"], True),
    # La forma exacta solo desde el white-box: desde cero colapsa a él.
    ("e1S2",   "refrac1", ["--variant", "S", "--r-init", "0.05"], False),
]
WB = {"refrac1": "e1wb", "act1": "e2wb"}
NIVELES = ["_n01", "_n05"]
SEMILLAS = [1, 2]


def datos(p, suf, k):
    return ["--data", f"{p}_amp{suf}_s{k}", "--smooth", "7", "--seed", str(k)]


if __name__ == "__main__":
    wbs, wrm, cero = [], [], []
    for k in SEMILLAS:
        for suf in NIVELES:
            for p, n in WB.items():
                wbs.append((f"e15_{n}{suf}_s{k}", [*datos(p, suf, k), "--variant", "whitebox"], None))
            for n, p, a, con_cero in CONFIGS:
                wb = f"e15_{WB[p]}{suf}_s{k}"
                wrm.append((f"e15_{n}_wrm{suf}_s{k}",
                            [*datos(p, suf, k), *a, "--init-params", str(RES / f"{wb}.json")], wb))
                if con_cero:
                    cero.append((f"e15_{n}{suf}_s{k}", [*datos(p, suf, k), *a], None))
    correr(wbs + wrm + cero, marca="DONE_semillas", lugares=3)
