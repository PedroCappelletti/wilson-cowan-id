#!/usr/bin/env python3
# Versiones con ruido de observacion de un dataset del escalado (eje 6).
#
# El ruido se suma al dataset YA GENERADO en vez de regenerar las trayectorias.
# Dos razones: la trayectoria queda identica a la limpia, asi que las corridas
# son comparables muestra a muestra, y dfI/dfE siguen calculados sobre la
# trayectoria exacta, que es lo unico que hace valido el R2 de la correccion.
#
# USO:  python scripts/esc_gen_ruido.py act1 0.01 0.05

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

DIR = Path("data/processed/uncertain")
SEED = 7


def agregar_ruido(nombre: str, sigma: float) -> Path:
    d = dict(np.load(DIR / f"{nombre}.npz", allow_pickle=True))
    if float(d["noise_std"]) != 0.0:
        raise SystemExit(f"{nombre} ya tiene ruido {d['noise_std']}")

    rng = np.random.default_rng(SEED)
    for k in ("I", "E"):
        d[k] = d[k] + rng.normal(0.0, sigma, size=d[k].shape).astype(d[k].dtype)
    d["noise_std"] = np.asarray(sigma)
    d["noise_seed"] = np.asarray(SEED)
    d["base_dataset"] = np.asarray(nombre)

    out = DIR / f"{nombre}_n{str(sigma).split('.')[-1]}.npz"
    np.savez_compressed(out, **d)
    rel = sigma / float(np.std(d["E"]))
    print(f"  {out.name}: sigma={sigma}  ({rel:.1%} del desvio de E)")
    return out


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    nombre = sys.argv[1]
    for s in sys.argv[2:]:
        agregar_ruido(nombre, float(s))


if __name__ == "__main__":
    main()
