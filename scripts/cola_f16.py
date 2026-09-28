#!/usr/bin/env python3
# El filtro de 16 canales, sin ruido y con los dos niveles.
#
# Con 8 canales la red ya usa unas 4 direcciones independientes de la historia,
# el mínimo teórico (P y Q filtrados, P y Q actuales). Si esa lectura es
# correcta, 16 canales no deberían mejorar mucho. Arranca cuando termina el lote
# del informe con ruido.

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import correr

K16 = ["--variant", "K", "--hist", "400", "--n-fir", "16"]

if __name__ == "__main__":
    correr([
        ("e9_K400f16_n05", ["--data", "act1_n05", *K16, "--smooth", "7"], None),
        ("e9_K400f16_n01", ["--data", "act1_n01", *K16, "--smooth", "7"], None),
        ("e9_K400f16", ["--data", "act1", *K16], None),
    ], marca="DONE_f16", esperar="DONE_informe_ruido")
