#!/usr/bin/env python3
# =============================================================================
#  CONVERSION DE DATOS REALES  (data8_filtered.mat -> .npz)
# =============================================================================
#
#  Que nos dieron (info del .mat):
#    "Tres estimulaciones sucesivas tipo chirp [1 10] Hz, senales u1,u2,u3 y sus
#     respuestas, filtradas [0.5 19] Hz, senales s1,s2,s3. fs = 1250 Hz."
#
#    - u_i : estimulo aplicado (chirp 1->10 Hz, NO negativo, en [0, 0.2],
#            offset ~0.1 -> u = 0.1 + 0.1*sin(fase_chirp), estilo optogenetico).
#    - s_i : respuesta MEDIDA, escalar, filtrada pasa-banda [0.5, 19] Hz
#            (por eso es de media ~0). Es el analogo de la salida y = E - I.
#    - fs = 1250 Hz. Duraciones ~16-20 s por grabacion.
#
#  DIFERENCIA CLAVE con los datos sinteticos: aca solo observamos UN escalar s
#  (no I y E por separado). => identificacion por SALIDA (estado latente).
#
#  Este script NO identifica nada: solo empaqueta a .npz en un formato comodo,
#  con opcion de submuestreo (decimacion) para acelerar el rollout despues.
#
#  USO:
#    python scripts/load_real_data.py                 # fs plena (1250 Hz)
#    python scripts/load_real_data.py --decimate 5    # -> 250 Hz (recomendado)
# =============================================================================

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import decimate

DEFAULT_MAT = Path.home() / "Downloads" / "data8_filtered.mat"
OUT_DIR = Path("data/processed/real")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", type=Path, default=DEFAULT_MAT)
    ap.add_argument("--decimate", type=int, default=1,
                    help="factor de decimacion (5 -> 1250/5 = 250 Hz). 1 = sin decimar.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    m = sio.loadmat(args.mat)
    fs = float(m["fsd"].ravel()[0])
    info = str(m["info"][0])
    q = max(int(args.decimate), 1)
    fs_out = fs / q

    recs = []
    for i in (1, 2, 3):
        u = m[f"u{i}"].ravel().astype(np.float64)
        s = m[f"s{i}"].ravel().astype(np.float64)
        if q > 1:
            # decimate aplica un anti-alias (Chebyshev) antes de submuestrear.
            # La banda util es [0.5,19] Hz; con fs_out>=250 Hz sobra margen.
            u = decimate(u, q, ftype="iir", zero_phase=True)
            s = decimate(s, q, ftype="iir", zero_phase=True)
        t = np.arange(len(s)) / fs_out
        recs.append({"u": u, "s": s, "t": t})
        print(f"  rec{i}: n={len(s):6d}  dur={t[-1]:.2f}s  "
              f"u[{u.min():.3f},{u.max():.3f}]  s std={s.std():.4f}")

    out = args.out or (OUT_DIR / f"data8_fs{int(round(fs_out))}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Guardamos como arrays object (largos distintos por grabacion).
    np.savez(
        out,
        fs=fs_out,
        fs_original=fs,
        info=info,
        n_rec=3,
        u=np.array([r["u"] for r in recs], dtype=object),
        s=np.array([r["s"] for r in recs], dtype=object),
        t=np.array([r["t"] for r in recs], dtype=object),
    )
    print(f"\n  fs_original={fs:.0f} Hz  ->  fs_out={fs_out:.0f} Hz  (decimate x{q})")
    print(f"  guardado: {out}")


if __name__ == "__main__":
    main()
