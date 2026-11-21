#!/usr/bin/env python3
# Lazo cerrado con los e28: backbone con los parámetros verdaderos congelados y
# solo la red entrenada. Mismas referencias, plantas y distancia entre lazos que
# control_amplitud.py, para comparar contra los e25 con warm start.
#
#   python scripts/control_e28.py [procesos]   -> results/control_e28.json
#   WC_FGAMMA=100 python scripts/control_e28.py -> results/control_e28_100hz.json

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

from control_distancia_y_rango import una, distancias, REFS, SUFIJO  # noqa: E402
from control_paper_modelos import NIVELES, SEMILLAS, RES  # noqa: E402

OUT = RAIZ / "results" / f"control_e28{SUFIJO}.json"
MODELOS = {
    "actc": [("agnóstica, β verdadero", "e28_e2B_vrd"),
             ("historia con comando actual, β verdadero", "e28_H100_vrd"),
             ("historia sin comando actual, β verdadero", "e28_Hp100_vrd"),
             ("filtro 8 canales, β verdadero", "e28_K400f8_vrd")],
    "refrac": [("agnóstica, β verdadero", "e28_B100_vrd")],
}


def tareas():
    out = []
    for ref in REFS:
        for planta, mods in MODELOS.items():
            out.append((ref, planta, "simulador", None, "", -1))
            for suf in NIVELES:
                for k in SEMILLAS:
                    for nombre, pre in mods:
                        tag = f"{pre}{suf}_s{k}"
                        if (RES / "models" / f"{tag}.pt").exists():
                            out.append((ref, planta, nombre, tag, suf, k))
    return out


if __name__ == "__main__":
    if "--probar" in sys.argv:
        fila, _ = una(("gamma50", "actc", "agnóstica, β verdadero", "e28_e2B_vrd_n01_s0", "_n01", 0))
        print(fila)
        sys.exit()
    procesos = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    ts = tareas()
    print(f"{len(ts)} simulaciones, {procesos} procesos", flush=True)
    filas, trazas = [], []
    with Pool(procesos) as pool:
        for fila, traza in pool.imap_unordered(una, ts):
            filas.append(fila)
            trazas.append(traza)
            print(f"  {fila['referencia']:11} {fila['perturbacion']:6} {fila['planta']:42} "
                  f"{fila['ruido']:5} s{fila['semilla']:>2}  RMSE {fila['rmse']:.4f}", flush=True)
    distancias(filas, trazas)
    OUT.write_text(json.dumps(filas, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {OUT}")
