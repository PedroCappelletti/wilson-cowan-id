#!/usr/bin/env python3
# Detecta corridas cuya pérdida de entrenamiento terminó bastante por encima de
# su mínimo. El pipeline evalúa el modelo de la última época, así que en esos
# casos el resultado es el de un modelo que ya se degradó, y no mide la
# configuración.
#
# Pasó con las tres corridas de penalización L2 (e5_K400_wd1e-4, e5_K400_wd1e-3
# y e8_K400wd4_n01) y con ninguna otra.
#
#   python scripts/divergencia.py            # revisa todos los logs

from __future__ import annotations

import re
import sys
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "logs" / "escalado"
UMBRAL = 1.5          # pérdida final sobre la mínima


def perdidas(tag: str) -> list[float]:
    f = LOG / f"{tag}.log"
    if not f.exists():
        return []
    return [float(b) for b in re.findall(r"ep\s+\d+ \| data=([\d.e+-]+)",
                                          f.read_text(errors="ignore"))]


def divergio(tag: str) -> float | None:
    """Cociente entre la pérdida final y la mínima si supera el umbral."""
    v = perdidas(tag)
    if len(v) < 3:
        return None
    r = v[-1] / min(v)
    return r if r > UMBRAL else None


if __name__ == "__main__":
    for f in sorted(LOG.glob("e*.log")):
        r = divergio(f.stem)
        if r:
            print(f"{f.stem:20} la pérdida terminó x{r:.1f} sobre su mínimo")
