#!/usr/bin/env python3
# Dos diagnósticos del filtro entrenable (variante K) que cita el complemento
# del póster y que hasta el 29/9 se calcularon sueltos en consola.
#
# Correlación entre muestras vecinas del comando: mide qué tan mal condicionado
# está el problema de estimar el núcleo. Con correlación cerca de 1, núcleos
# muy distintos dan casi la misma salida.
#
# Dimensión efectiva de lo que la red ve de la historia: el filtro y la primera
# capa son lineales y seguidos, así que sobre la historia actúa W_z @ A, con A
# el FIR (n x 2K) y W_z la parte de la primera capa que ve sus salidas. La
# dimensión efectiva es (sum s^2)^2 / sum s^4 sobre sus valores singulares.
#
#   python scripts/diagnostico_fir.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "results" / "escalado" / "models"
DATOS = RAIZ / "data" / "processed" / "uncertain"


def correlacion_vecinas(nombre="act1"):
    d = np.load(DATOS / f"{nombre}.npz", allow_pickle=True)
    tren = ~d["is_test"].astype(bool)
    a, b = [], []
    for u in list(d["P"][tren]) + list(d["Q"][tren]):
        a.append(u[1:])
        b.append(u[:-1])
    return float(np.corrcoef(np.concatenate(a), np.concatenate(b))[0, 1])


def dimension_efectiva(tag):
    st = torch.load(RES / f"{tag}.pt", map_location="cpu", weights_only=False)["state"]
    A = st["g.fir.weight"].numpy()                 # n x 2K
    W = st["g.mlp.0.weight"].numpy()[:, 2:]        # 32 x n, sin las columnas de (I, E)
    s = np.linalg.svd(W @ A, compute_uv=False)
    return float((s ** 2).sum() ** 2 / (s ** 4).sum()), A.shape[0]


if __name__ == "__main__":
    print(f"correlación entre muestras vecinas del comando (act1, entrenamiento): "
          f"{correlacion_vecinas():.3f}")
    for tag in ("e3_K400", "e3_K400_f8", "e9_K400f16"):
        if (RES / f"{tag}.pt").exists():
            de, n = dimension_efectiva(tag)
            print(f"{tag:12} {n:2d} canales  dimensión efectiva {de:.2f}")
