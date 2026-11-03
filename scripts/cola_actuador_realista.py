#!/usr/bin/env python3
# Línea realista del actuador (e18): las mismas variantes que e16 sobre plantas
# con cinética de ChR2 en lugar del filtro simétrico de 1 ms.
#
#   act_asim  activación 1,5 ms, desactivación 12 ms
#   act_t43   simétrico de 4,3 ms (ancho de banda de H134R, 37 Hz)
#   act_t23   simétrico de 2,3 ms (ancho de banda de ChR2, 69 Hz)
#
# La saturación es la de la línea de diseño (3 tanh(·/3)), así que contra
# act1_2s cambia solo la cinética. Se reporta aparte de e16, nunca mezclado.
#
# Primera pasada con la semilla 0. La asimétrica va primero y con todas las
# variantes; en las simétricas, las de memoria corta y el estado de filtro.
# H400 (20 ms de historia) entra porque la desactivación de 12 ms excede los
# 5 ms de H100. Mismo esquema de ventanas que e16 (5 ms, x0 medido) para que la
# comparación con la línea de diseño sea directa.
#
#   python scripts/cola_actuador_realista.py

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import RES, correr
from cola_semillas_2s import NIVELES, V5, datos

SEMILLAS = [0]
VARIANTES = {
    "H100":  ["--variant", "H", "--hist", "10", *V5],
    "e2B":   ["--variant", "B", *V5],
    "H400":  ["--variant", "H", "--hist", "40", *V5],
    "Hp100": ["--variant", "Hp", "--hist", "11", *V5],
    "lag":   ["--variant", "lag", *V5],
}
PLANTAS = {   # planta: variantes, en orden de prioridad
    "act_asim": ["H100", "e2B", "H400", "Hp100", "lag"],
    "act_t43":  ["H100", "e2B", "lag"],
    "act_t23":  ["H100", "e2B", "lag"],
}


def corto(p):
    return p.removeprefix("act_")


if __name__ == "__main__":
    wbs, wrm, cero = [], [], []
    for k in SEMILLAS:
        for p, vs in PLANTAS.items():
            for suf in NIVELES:
                wb = f"e18_{corto(p)}_wb{suf}_s{k}"
                wbs.append((wb, [*datos(p, suf, k), "--variant", "whitebox", *V5], None))
                for v in vs:
                    wrm.append((f"e18_{corto(p)}_{v}_wrm{suf}_s{k}",
                                [*datos(p, suf, k), *VARIANTES[v],
                                 "--init-params", str(RES / f"{wb}.json")], wb))
                    cero.append((f"e18_{corto(p)}_{v}{suf}_s{k}",
                                 [*datos(p, suf, k), *VARIANTES[v]], None))
    correr(wbs + wrm + cero, marca="DONE_actuador_realista", lugares=3, esperar=["DONE_x0_libre_wrm", "DONE_gen_actuador_realista"])
