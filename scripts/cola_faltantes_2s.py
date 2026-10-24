#!/usr/bin/env python3
# Las variantes de la tabla del póster que la tanda e16 (2 s) no incluye, para
# poder probar los controladores sobre todas sin mezclar datos de 200 ms y 2 s.
#
# Agnóstica del actuador, historia de 20 ms y filtros de 4 y 16 canales; con
# ruido, desde cero y desde el white-box de e16 (semilla 0). Espera a que
# termine cola_semillas_2s.py (marca DONE_semillas_2s) para no pasar de 3
# corridas a la vez.
#
# Tags: e16_<nombre>[_wrm]<ruido>_s0, la misma familia que cola_semillas_2s.py.

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import RES, correr

V5 = ["--window", "10"]
K = ["--variant", "K", "--hist", "40"]
CONFIGS = [   # (nombre, argumentos), de la más lenta a la más rápida
    ("K400f16", [*K, "--n-fir", "16", *V5]),
    ("K400",    [*K, "--n-fir", "4", *V5]),
    ("H400",    ["--variant", "H", "--hist", "40", *V5]),
    ("e2B",     ["--variant", "B", *V5]),
]

if __name__ == "__main__":
    wrm, cero = [], []
    for suf in ("_n01", "_n05"):
        datos = ["--data", f"act1_2s{suf}", "--seed", "0"]
        wb = f"e16_e2wb{suf}_s0"
        for n, a in CONFIGS:
            wrm.append((f"e16_{n}_wrm{suf}_s0",
                        [*datos, *a, "--init-params", str(RES / f"{wb}.json")], wb))
            cero.append((f"e16_{n}{suf}_s0", [*datos, *a], None))
    correr(wrm + cero, marca="DONE_faltantes_2s", lugares=3,
           esperar="DONE_semillas_2s")
