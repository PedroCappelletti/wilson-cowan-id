#!/usr/bin/env python3
# Consistencia de β entre familias de estímulos (tema 5 de la agenda del 4-10).
#
# Un white-box por familia, entrenado y validado solo con esa familia, sobre el
# dataset ampliado. Si el backbone alcanzara, las siete familias darían el mismo
# β; si dan β distintos, algo no modelado actúa distinto según el estímulo, y la
# dispersión mide cuánto contamina a β sin saber qué es.
#
# wc0_amp es el control: Wilson-Cowan puro, sin perturbación. Ahí β por familia
# solo puede diferir por falta de información en cada familia, así que lo que
# refractariedad o actuador se dispersen de más es lo que aporta la perturbación.
#
# Tags: e14_<planta>_<familia><ruido>, y e14_wc0_todas<ruido> para el white-box
# del control con todas las familias (el de las otras plantas es e13_e*wb).
# Primero con ruido. Análisis: scripts/consistencia_beta.py.

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import correr

PLANTAS = ["wc0", "refrac1", "act1"]
# de la que tiene más escenarios de entrenamiento a la que tiene menos
FAMILIAS = ["square", "box", "aprbs", "thetagamma", "prbs", "poisson", "chirp"]


def corridas(suf):
    suave = ["--smooth", "7"] if suf else []
    out = []
    for fam in FAMILIAS:
        for p in PLANTAS:
            out.append((f"e14_{p}_{fam}{suf}",
                        ["--data", f"{p}_amp{suf}", "--variant", "whitebox",
                         "--familia", fam, *suave], None))
    out.append((f"e14_wc0_todas{suf}",
                ["--data", f"wc0_amp{suf}", "--variant", "whitebox", *suave], None))
    return out


if __name__ == "__main__":
    correr(corridas("_n01") + corridas("_n05") + corridas(""),
           marca="DONE_familias", lugares=3)
