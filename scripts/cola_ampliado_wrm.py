#!/usr/bin/env python3
# La tabla del póster sobre el dataset ampliado, desde cero y desde el white-box.
#
# Reemplaza el orden de cola_ampliado.py. Cada configuración se corre dos veces,
# arrancando ignorante (β = 1,0) y arrancando del β del white-box de la misma
# planta y el mismo ruido, para poder compararlas. El arranque desde el
# white-box va primero porque es el que viene dando mejor. Tags: e13_<nombre>
# desde cero, e13_<nombre>_wrm desde el white-box.
#
# La forma exacta siempre arrancó del white-box (e13_e1S2). Su versión desde
# cero es e13_e1S_cero, y va al final.
#
# Orden: los white-box de σ = 0,05 (los necesita el arranque de ese nivel), el
# arranque desde el white-box de σ = 0,01 y de 0,05, lo que falta desde cero
# de 0,05, y después todo el nivel sin ruido en el mismo orden.

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import RES, correr
from cola_ampliado import CONFIGS, corridas

WB = {"refrac1": "e1wb", "act1": "e2wb"}
SIN_ARRANQUE = {"e1wb", "e2wb"}

# Las de cola_ampliado.py que estaban andando al cambiar de cola: ocupan lugar
# hasta que terminan y no se vuelven a lanzar.
AJENAS = ["e13_K400f16_n05", "e13_K400f8_n05", "e13_K400_n05", "e13_H400_n05",
          # al relanzar la cola el 4-10 para bajarla a 3 lugares
          "e13_B400_wrm_n05", "e13_H100_wrm_n05", "e13_e2B_wrm_n05",
          "e13_B100_wrm_n05"]


def desde_wb(suf):
    suave = ["--smooth", "7"] if suf else []
    out = []
    for n, p, a in CONFIGS:
        if n in SIN_ARRANQUE:
            continue
        wb = f"e13_{WB[p]}{suf}"
        out.append((f"e13_{n}_wrm{suf}",
                    ["--data", f"{p}_amp{suf}", *a, *suave,
                     "--init-params", str(RES / f"{wb}.json")], wb))
    return out


def s_desde_cero(suf):
    suave = ["--smooth", "7"] if suf else []
    return [(f"e13_e1S_cero{suf}",
             ["--data", f"refrac1_amp{suf}", "--variant", "S", "--r-init", "0.05",
              *suave], None)]


def nivel(suf):
    cero = corridas(suf)
    wbs = [c for c in cero if c[0] in (f"e13_e1wb{suf}", f"e13_e2wb{suf}")]
    resto = [c for c in cero if c not in wbs]
    return wbs, desde_wb(suf), resto


if __name__ == "__main__":
    wb05, wrm05, cero05 = nivel("_n05")
    _, wrm01, _ = nivel("_n01")
    wb00, wrm00, cero00 = nivel("")
    orden = (wb05 + wrm01 + wrm05 + cero05 + wb00 + wrm00 + cero00
             + s_desde_cero("_n01") + s_desde_cero("_n05") + s_desde_cero(""))
    orden = [c for c in orden if c[0] not in AJENAS]
    correr(orden, marca="DONE_ampliado_wrm", ajenas=AJENAS)
