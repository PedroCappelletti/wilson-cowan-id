#!/usr/bin/env python3
# Segunda etapa con x0 entrenable sobre el actuador con recorte en ±3 (e22).
#
# Es e19_<conf>_x0 de cola_x0_libre_y_beta_verdadero.py con la planta nueva:
# arranca del modelo e21 entero (--init-model: β y red) y entrena los x0 de cada
# ventana con λ = 1. Las mismas cuatro configuraciones del actuador que e19, así
# las dos líneas se comparan fila por fila. El estado de filtro (lag) sigue
# afuera, porque x0 libre no está implementado para el estado aumentado.
#
# Prioridad: semilla 0 entera, σ 0,05 antes que 0,01.
#
#   python scripts/cola_x0_recorte.py

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import RES, correr
from cola_semillas_2s import datos

V5 = ["--window", "10"]
CONFS = {
    "e2wb":   ["--variant", "whitebox"],
    "H100":   ["--variant", "H", "--hist", "10"],
    "Hp100":  ["--variant", "Hp", "--hist", "11"],
    "K400f8": ["--variant", "K", "--hist", "40", "--n-fir", "8"],
}
NIVELES = ["_n05", "_n01"]


def partida(n, suf, k):
    # el white-box no lleva _wrm en el tag de e21
    return f"e21_{n}{suf}_s{k}" if n == "e2wb" else f"e21_{n}_wrm{suf}_s{k}"


if __name__ == "__main__":
    tareas = []
    for k in (0, 1, 2):
        for suf in NIVELES:
            for n, a in CONFS.items():
                pt = RES / "models" / f"{partida(n, suf, k)}.pt"
                assert pt.exists(), pt
                tareas.append((f"e22_{n}_x0{suf}_s{k}",
                               [*datos("actc1", suf, k), *a, *V5, "--x0-libre", "1", "--init-model", str(pt)],
                               None))
    correr(tareas, marca="DONE_x0_recorte", lugares=3)
