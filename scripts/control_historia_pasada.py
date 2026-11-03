#!/usr/bin/env python3
# Controlador del paper sobre la historia pasada del comando como planta.
#
# Mismo lazo que control_paper_planta_modelo.py (controlador con los
# parámetros del .m, sin cambios), con los e16_Hp100_wrm de la planta con
# actuador en lugar de los H100_wrm. El simulador, el white-box y las otras
# variantes ya están en results/control_planta_modelo.json.
#
#   python scripts/control_historia_pasada.py [procesos]

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import numpy as np

from control_paper_planta_modelo import REFS, NIVELES, SEMILLAS, RES, una, verificar

OUT = RAIZ / "results" / "control_historia_pasada.json"
TRAZAS = RAIZ / "results" / "control_historia_pasada_trazas.npz"


def tareas():
    return [(ref, "act", "historia pasada", f"e16_Hp100_wrm{suf}_s{k}", suf, k)
            for ref in REFS for suf in NIVELES for k in SEMILLAS]


def main():
    from esc_eval import cargar
    for t in tareas():
        print(f"  {t[3]:24} {verificar(cargar(RES / 'models' / f'{t[3]}.pt')[0]):.2e}", flush=True)
    procesos = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    filas, trazas = [], {}
    with Pool(procesos) as pool:
        for fila, traza in pool.imap_unordered(una, tareas()):
            filas.append(fila)
            print(f"  {fila['referencia']:8} {fila['ruido']:5} s{fila['semilla']}  "
                  f"RMSE {fila['rmse']:.4f}", flush=True)
            if traza is not None:
                for kk, v in traza.items():
                    trazas[f"{fila['referencia']}|act|historia pasada|{kk}"] = v
    OUT.write_text(json.dumps(filas, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(TRAZAS, **trazas)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
