#!/usr/bin/env python3
# β verdadero fijo sobre los datasets de amplitud grande (e28), el diagnóstico
# de e20 repetido con comandos de hasta ±12.
#
# Se entrena solo la red, con β fijo en los parámetros verdaderos
# (--init-params parametros_verdaderos.json --freeze-phys), y r2_residuo_test
# mide contra el término faltante exacto (dfI, dfE del dataset). La pregunta es
# si cada red puede representar el filtro con el recorte en ±3 cuando β es el
# correcto, o si con β libre la deformación de β (42 a 204 % en e25) es la única
# salida que encuentra. Sirve como diagnóstico, no como planta para el control.
#
# e20 se hizo con [0, 2] y sobre el actuador con tanh (act1); con el recorte
# duro (actc) no había ninguna corrida con β verdadero. Mismas arquitecturas
# que e25: agnóstica, las dos historias y el filtro entrenable en el actuador,
# y la agnóstica en la refractaria. Dos ruidos, tres semillas: 30 corridas.
#
#   python scripts/cola_amplitud_beta_verdadero.py [lugares]

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import RES, correr
from cola_amplitud import NIVELES, SEMILLAS, args_de, datos

VERDADERO = RES / "parametros_verdaderos.json"
CONFS = [("actc", "e2B"), ("actc", "H100"), ("actc", "Hp100"), ("actc", "K400f8"),
         ("refrac", "B100")]


def tareas():
    return [(f"e28_{n}_vrd{suf}_s{k}",
             [*datos(planta, suf, k), *args_de(planta, n),
              "--init-params", str(VERDADERO), "--freeze-phys"], None)
            for k in SEMILLAS for suf in NIVELES for planta, n in CONFS]


if __name__ == "__main__":
    assert VERDADERO.exists(), VERDADERO
    lugares = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    correr(tareas(), marca="DONE_amplitud_beta_verdadero", lugares=lugares)
