#!/usr/bin/env python3
# Una linea por corrida terminada, para seguir un lote sin abrir los json.
#
# USO:  python scripts/esc_watch.py e3_K400 e3_K800 ...

from __future__ import annotations

import json
import sys
from pathlib import Path

DIR = Path("results/escalado")


def linea(tag: str) -> str:
    d = json.loads((DIR / f"{tag}.json").read_text())
    campos = [
        f"{tag:14}",
        f"hist={d.get('hist', 0):<4}",
        f"nrmse={d['nrmse_test']:6.2f}",
        f"r2={d['r2_delta_test']:+6.3f}",
        f"err_par={d['mean_param_error']:5.1f}",
        f"{d['minutos']:.0f} min",
    ]
    for k, fmt in (("g_rms", "{:.4f}"), ("frac_redundante", "{:.3f}")):
        if k in d:
            campos.append(f"{k}=" + fmt.format(d[k]))
    return "  ".join(campos)


if __name__ == "__main__":
    for t in sys.argv[1:]:
        if (DIR / f"{t}.json").exists():
            print(linea(t))
