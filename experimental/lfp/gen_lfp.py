#!/usr/bin/env python3
"""Dataset del piloto de S. EN EVALUACIÓN, ver README.md.

Toma un dataset del escalado ya generado (por ejemplo act1_amp) y le agrega S,
una observación escalar que combina I y E según uno de los proxies de
observacion.py, más ruido 1/f^chi sobre S. I y E quedan limpios en el archivo,
solo para el diagnóstico: el entrenamiento del piloto no los mira.

Se parte del dataset limpio y no del que ya tiene ruido en I y E, porque acá el
ruido de medición va sobre S, que es lo que se mide.

USO
  python experimental/lfp/gen_lfp.py refrac1_amp E_menos_I --snr 20 --chi 1
  -> data/processed/lfp_piloto/refrac1_amp__E_menos_I__snr20_chi1.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from observacion import PROXIES
from ruido import agregar

ORIGEN = RAIZ / "data/processed/uncertain"
DESTINO = RAIZ / "data/processed/lfp_piloto"
PESOS = ("wEE", "wEI", "wIE", "wII")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="dataset limpio del escalado, sin extensión")
    ap.add_argument("proxy", choices=sorted(PROXIES))
    ap.add_argument("--snr", type=float, default=20.0, help="dB, potencia total")
    ap.add_argument("--chi", type=float, default=1.0, help="exponente del 1/f^chi")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    d = dict(np.load(ORIGEN / f"{a.base}.npz", allow_pickle=True))
    if float(d["noise_std"]) != 0.0:
        raise SystemExit(f"{a.base} ya tiene ruido en I y E: usar el limpio")

    w = np.array([float(d[k]) for k in PESOS])
    S = PROXIES[a.proxy](d["I"], d["E"], d["P"], d["Q"], w).astype(np.float32)
    dt_ms = float(d["dt"])
    d["S_limpio"] = S
    d["S"] = agregar(S, a.snr, a.chi, dt_ms, seed=a.seed).astype(np.float32)
    d.update(proxy=np.asarray(a.proxy), snr_db=np.asarray(a.snr),
             chi=np.asarray(a.chi), noise_seed=np.asarray(a.seed),
             base_dataset=np.asarray(a.base))

    DESTINO.mkdir(parents=True, exist_ok=True)
    snr = f"{a.snr:g}".replace(".", "p")
    chi = f"{a.chi:g}".replace(".", "p")
    out = DESTINO / f"{a.base}__{a.proxy}__snr{snr}_chi{chi}.npz"
    np.savez_compressed(out, **d)
    print(f"{out.name}: S en [{S.min():.3f}, {S.max():.3f}], desvío {S.std():.3f}")


if __name__ == "__main__":
    main()
