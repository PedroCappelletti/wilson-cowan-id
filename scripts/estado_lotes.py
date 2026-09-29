#!/usr/bin/env python3
# Estado de todas las corridas encoladas: terminada, andando, en cola o caída.
#
#   python scripts/estado_lotes.py

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import LOG, RES

LOTES = {
    "póster con ruido": [f"e7_{c}_n{n}" for n in ("05", "01")
                         for c in ("e1wb", "e1B400", "e1S2", "e2B", "e2lag")]
                        + ["e7_e2wb_n01"],
    "informe con ruido": [f"e8_{c}_n{n}" for c in
                          ("K400f8", "K400wd4", "K400wd3", "Sg", "H100")
                          for n in ("05", "01")] + ["e8_K400_n01", "e8_H400_n01"],
    "16 canales": ["e9_K400f16_n05", "e9_K400f16_n01", "e9_K400f16"],
}


def estado(tag: str) -> str:
    log = LOG / f"{tag}.log"
    if (RES / f"{tag}.json").exists():
        return "terminada"
    if log.exists() and "Traceback" in log.read_text(errors="ignore"):
        return "CAÍDA"
    if log.exists():
        ultima = [l for l in log.read_text(errors="ignore").splitlines() if " ep " in l]
        ep = ultima[-1].split()[1] if ultima else "0"
        return f"andando (época {ep} de 1500)"
    return "en cola"


if __name__ == "__main__":
    for lote, tags in LOTES.items():
        estados = [estado(t) for t in tags]
        print(f"\n{lote}: {sum(e == 'terminada' for e in estados)} de {len(tags)} terminadas")
        for t, e in zip(tags, estados):
            if e != "terminada":
                print(f"   {t:18} {e}")
