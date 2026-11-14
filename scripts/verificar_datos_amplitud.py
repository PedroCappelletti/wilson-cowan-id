#!/usr/bin/env python3
# Huellas de los datasets de amplitud grande, para comprobar que una copia o una
# regeneración en otra máquina coincide con la original.
#
# La huella es el sha256 de los arreglos I, E, P y Q, no del archivo: el .npz
# comprimido puede cambiar de bytes con otra versión de zlib aunque los datos
# sean los mismos.
#
#   python scripts/verificar_datos_amplitud.py --escribir   guarda las huellas (máquina original)
#   python scripts/verificar_datos_amplitud.py              compara contra las guardadas

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
DIR = RAIZ / "data" / "processed" / "uncertain"
HUELLAS = RAIZ / "data" / "huellas_amplitud.json"
PLANTAS = ("wc0", "refrac", "actc")
VARIANTES = ("", "_n01", "_n05", "_n01_s1", "_n05_s1", "_n01_s2", "_n05_s2")


def huella(p: Path) -> str:
    d = np.load(p, allow_pickle=True)
    h = hashlib.sha256()
    for k in ("I", "E", "P", "Q"):
        h.update(np.ascontiguousarray(d[k]).tobytes())
    return h.hexdigest()


def nombres():
    return [f"{p}_amp_2s{v}" for p in PLANTAS for v in VARIANTES]


if __name__ == "__main__":
    if "--escribir" in sys.argv:
        HUELLAS.write_text(json.dumps({n: huella(DIR / f"{n}.npz") for n in nombres()}, indent=2))
        print(f"-> {HUELLAS}")
        sys.exit(0)
    ref = json.loads(HUELLAS.read_text())
    mal = []
    for n in nombres():
        p = DIR / f"{n}.npz"
        estado = "falta" if not p.exists() else ("ok" if huella(p) == ref[n] else "DISTINTO")
        print(f"{n:24} {estado}")
        if estado != "ok":
            mal.append(n)
    print("TODO OK" if not mal else f"PROBLEMAS: {mal}")
    sys.exit(1 if mal else 0)
