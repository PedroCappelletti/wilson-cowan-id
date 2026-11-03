#!/usr/bin/env python3
# Sesgo del entrenamiento por ventanas que arrancan del estado observado.
#
# Cada ventana arranca del (I, E) medido, con ruido, y no hay estados iniciales
# libres ni penalización de la discontinuidad: es el error de predicción a K
# pasos de Ribeiro et al. 2020 (Automatica 121, sec. 6), que con ruido puede
# converger a parámetros sesgados. La prueba aísla ese efecto: white-box sobre
# la planta ideal (wc0_2s), donde el modelo está bien especificado, sin ruido
# y con cinco realizaciones de cada nivel. Si la media de β̂ sobre las
# realizaciones se aleja de la verdad más que su error estándar, el sesgo es
# del esquema y no del ruido de una realización. Con ventanas de 5 ms (10
# muestras) y de 20 ms (40), porque el sesgo debería bajar con ventanas largas.
#
# Arranca cuando termina cola_historia_pasada.py, para no pasar de 3 corridas.
#
#   python scripts/cola_sesgo_ventanas.py
#   python scripts/resumen_sesgo_ventanas.py

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import correr

SEMILLAS = [0, 1, 2, 3, 4]
VENTANAS = {"W10": "10", "W40": "40"}


def datos(suf, k):
    return ["--data", f"wc0_2s{suf}" + (f"_s{k}" if k else ""), "--seed", str(k)]


if __name__ == "__main__":
    tareas = []
    for w, n in VENTANAS.items():
        tareas.append((f"e17_wb_{w}_limpio", [*datos("", 0), "--variant", "whitebox", "--window", n], None))
        for suf in ("_n01", "_n05"):
            for k in SEMILLAS:
                tareas.append((f"e17_wb_{w}{suf}_s{k}",
                               [*datos(suf, k), "--variant", "whitebox", "--window", n], None))
    correr(tareas, marca="DONE_sesgo_ventanas", lugares=3, esperar="DONE_historia_pasada")
