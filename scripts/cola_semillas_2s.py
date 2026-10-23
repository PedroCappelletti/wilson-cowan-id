#!/usr/bin/env python3
# Semillas de las configuraciones clave sobre los datos de 2 s a 2 kHz (e16).
#
# Reemplaza a cola_semillas.py (e15, sobre los datos de 200 ms), que se
# descartó el 23-10 al pasar a los datos nuevos; sus resultados quedaron en
# results/escalado/_descartado_e15/.
#
# La semilla k cambia la realización del ruido (dataset <planta>_2s_n0x_s<k>,
# de esc_gen_ruido.py --semilla k --ref-fs 20000) y la inicialización de la red
# (--seed k). La semilla 0 usa el ruido sin sufijo. Con ruido, desde cero y
# desde el white-box de la misma planta, ruido y semilla; el white-box primero.
#
# Con dt = 0,5 ms, ventanas e historia se cuentan en muestras de 0,5 ms:
# 5 ms = 10, 20 ms = 40. Sin media móvil: el ruido ya está escalado a la banda.
#
# Tags: e16_<nombre>[_wrm]<ruido>_s<k>.

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import RES, correr

V5, V20 = ["--window", "10"], ["--window", "40"]
CONFIGS = [   # (nombre, planta, argumentos, se corre desde cero)
    ("B400",   "refrac1", ["--variant", "B", *V20], True),
    ("K400f8", "act1",    ["--variant", "K", "--hist", "40", "--n-fir", "8", *V5], True),
    ("lag",    "act1",    ["--variant", "lag", *V5], True),
    ("H100",   "act1",    ["--variant", "H", "--hist", "10", *V5], True),
    ("B100",   "refrac1", ["--variant", "B", *V5], True),
    # La forma exacta solo desde el white-box: desde cero colapsa a él.
    ("e1S2",   "refrac1", ["--variant", "S", "--r-init", "0.05", *V5], False),
]
WB = {"refrac1": "e1wb", "act1": "e2wb"}
NIVELES = ["_n01", "_n05"]
SEMILLAS = [0, 1, 2]


def datos(p, suf, k):
    nombre = f"{p}_2s{suf}" + (f"_s{k}" if k else "")
    return ["--data", nombre, "--seed", str(k)]


if __name__ == "__main__":
    wbs, wrm, cero = [], [], []
    for k in SEMILLAS:
        for suf in NIVELES:
            for p, n in WB.items():
                wbs.append((f"e16_{n}{suf}_s{k}",
                            [*datos(p, suf, k), "--variant", "whitebox", *V5], None))
            for n, p, a, con_cero in CONFIGS:
                wb = f"e16_{WB[p]}{suf}_s{k}"
                wrm.append((f"e16_{n}_wrm{suf}_s{k}",
                            [*datos(p, suf, k), *a, "--init-params", str(RES / f"{wb}.json")], wb))
                if con_cero:
                    cero.append((f"e16_{n}{suf}_s{k}", [*datos(p, suf, k), *a], None))
    correr(wbs + wrm + cero, marca="DONE_semillas_2s", lugares=3)
