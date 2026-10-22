#!/usr/bin/env python3
# Versiones con ruido de observacion de un dataset del escalado (eje 6).
#
# El ruido se suma al dataset YA GENERADO en vez de regenerar las trayectorias.
# Dos razones: la trayectoria queda identica a la limpia, asi que las corridas
# son comparables muestra a muestra, y dfI/dfE siguen calculados sobre la
# trayectoria exacta, que es lo unico que hace valido el R2 de la correccion.
#
# USO:  python scripts/esc_gen_ruido.py act1 0.01 0.05
#       python scripts/esc_gen_ruido.py act1_amp 0.01 0.05 --semilla 1
#
# --semilla k usa la semilla 7 + k y agrega _s<k> al nombre: otra realización
# del ruido sobre la misma trayectoria, para medir la variación entre corridas
# de los modelos deterministas (white-box, forma exacta, estado de filtro).
#
#       python scripts/esc_gen_ruido.py act1_2s 0.01 0.05 --ref-fs 20000
#
# --ref-fs f da σ como equivalente a la de un dataset muestreado a f Hz: se
# escala por sqrt(fs/f) para que la densidad espectral del ruido dentro de
# banda sea la misma. Con el mismo σ por muestra, bajar de 20 a 2 kHz metería
# diez veces más potencia de ruido en la banda de la señal. El nombre del
# archivo conserva el σ equivalente (_n01, _n05).
#
# El ruido de I y el de E son dos sorteos distintos del mismo generador, así
# que son independientes: no se cancelan al combinar E e I (medido: correlación
# −0,0002, y var(ruido E − ruido I) = 2σ²).

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

DIR = Path("data/processed/uncertain")
SEED = 7


def agregar_ruido(nombre: str, sigma: float, semilla: int = 0,
                  ref_fs: float | None = None) -> Path:
    d = dict(np.load(DIR / f"{nombre}.npz", allow_pickle=True))
    if float(d["noise_std"]) != 0.0:
        raise SystemExit(f"{nombre} ya tiene ruido {d['noise_std']}")

    sigma_eq = sigma
    if ref_fs:
        fs = 1000.0 / float(d["dt"])      # dt en ms
        sigma = sigma_eq * np.sqrt(fs / ref_fs)
    rng = np.random.default_rng(SEED + semilla)
    for k in ("I", "E"):
        d[k] = d[k] + rng.normal(0.0, sigma, size=d[k].shape).astype(d[k].dtype)
    d["noise_std"] = np.asarray(sigma)
    d["noise_std_equiv"] = np.asarray(sigma_eq)
    d["noise_ref_fs"] = np.asarray(ref_fs or 0.0)
    d["noise_seed"] = np.asarray(SEED + semilla)
    d["base_dataset"] = np.asarray(nombre)

    sufijo = f"_s{semilla}" if semilla else ""
    out = DIR / f"{nombre}_n{str(sigma_eq).split('.')[-1]}{sufijo}.npz"
    np.savez_compressed(out, **d)
    rel = sigma / float(np.std(d["E"]))
    print(f"  {out.name}: sigma={sigma}  ({rel:.1%} del desvio de E)")
    return out


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    args = sys.argv[1:]
    semilla, ref_fs = 0, None
    if "--semilla" in args:
        i = args.index("--semilla")
        semilla = int(args[i + 1])
        del args[i:i + 2]
    if "--ref-fs" in args:
        i = args.index("--ref-fs")
        ref_fs = float(args[i + 1])
        del args[i:i + 2]
    nombre = args[0]
    for s in args[1:]:
        agregar_ruido(nombre, float(s), semilla, ref_fs)


if __name__ == "__main__":
    main()
