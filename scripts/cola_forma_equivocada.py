#!/usr/bin/env python3
# El backbone expandido con la forma equivocada del actuador (e29).
#
# En e25 el estado de filtro recibe la forma correcta de la planta (filtro de
# primer orden y recorte en ±1/α) y recupera β con 0,1 a 0,7 % de error. Acá se
# le cambia una pieza por vez, sobre los mismos datasets de amplitud grande del
# actuador con recorte y con el mismo arranque desde el white-box (e25_e2wb):
#
#   lagtanh   saturación tanh(α P̂)/α en lugar del recorte (--forma tanh)
#   lagtau2   τ̂ fijo en 2 ms, el doble del verdadero (--tau-fijo 2)
#   lagtau05  τ̂ fijo en 0,5 ms, la mitad (--tau-fijo 0.5)
#
# La pregunta es cuánto de la ventaja del backbone expandido depende de
# conocer la forma exacta del mecanismo omitido. Dos ruidos, tres semillas:
# 18 corridas.
#
#   python scripts/cola_forma_equivocada.py [lugares]

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import RES, correr
from cola_amplitud import NIVELES, SEMILLAS, V5, datos

CONFS = [("lagtanh", ["--variant", "lag", "--forma", "tanh"]),
         ("lagtau2", ["--variant", "lag", "--tau-fijo", "2.0"]),
         ("lagtau05", ["--variant", "lag", "--tau-fijo", "0.5"])]


def tareas():
    return [(f"e29_{n}_wrm{suf}_s{k}",
             [*datos("actc", suf, k), *a, *V5, "--init-params", str(RES / f"e25_e2wb{suf}_s{k}.json")],
             None)
            for k in SEMILLAS for suf in NIVELES for n, a in CONFS]


if __name__ == "__main__":
    faltan = [t for t, a, _ in tareas() if not Path(a[a.index("--init-params") + 1]).exists()]
    assert not faltan, f"faltan los white-box de partida de {faltan}"
    lugares = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    correr(tareas(), marca="DONE_forma_equivocada", lugares=lugares)
