#!/usr/bin/env python3
# β verdadero fijo para la historia pasada (e20_Hp100_vrd), que quedó fuera de
# cola_x0_libre_y_beta_verdadero.py. Mismo esquema: solo se entrena la red y
# r2_residuo_test mide contra el término faltante exacto. Completa la
# comparación con la historia de 5 ms (H100), que lleva el comando actual.
#
#   python scripts/cola_hp_beta_verdadero.py

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import correr
from cola_semillas_2s import datos
from cola_x0_libre_y_beta_verdadero import CONFS, NIVELES, V5, VERDADERO

if __name__ == "__main__":
    tareas = [(f"e20_Hp100_vrd{suf}_s{k}",
               [*datos("act1", suf, k), *CONFS["Hp100"][1], *V5,
                "--init-params", str(VERDADERO), "--freeze-phys"], None)
              for k in (0, 1, 2) for suf in NIVELES]
    correr(tareas, marca="DONE_hp_beta_verdadero", lugares=3)
