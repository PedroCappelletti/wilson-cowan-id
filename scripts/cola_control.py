#!/usr/bin/env python3
# Todas las comparaciones de controladores, en orden (control_comparado.py).
#
# Tanda × referencia × modo del comando:
#   e13 (200 ms) y e16 (2 s);  gamma50 y theta2s;  libre, tope y tope_aw.
# Mientras se entrenan los modelos de 2 s (cola_faltantes_2s.py) corre con un
# proceso; cuando aparece la marca DONE_faltantes_2s, con cuatro. Las de e16
# esperan esa marca. Saltea las que ya tienen su JSON.
#
#   python scripts/cola_control.py

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
LOG = RAIZ / "logs" / "escalado"
MARCA = LOG / "DONE_faltantes_2s"

TRABAJOS = [(t, r, m) for t in ("e13", "e16") for r in ("gamma50", "theta2s")
            for m in ("libre", "tope", "tope_aw")]

if __name__ == "__main__":
    for tanda, ref, modo in TRABAJOS:
        salida = RAIZ / "results" / f"control_comparado_{tanda}_{ref}_{modo}.json"
        if salida.exists():
            continue
        while tanda == "e16" and not MARCA.exists():
            time.sleep(120)
        procesos = 4 if MARCA.exists() else 1
        print(f"{time.strftime('%H:%M')} {tanda} {ref} {modo} con {procesos} procesos", flush=True)
        with open(LOG / f"control_{tanda}_{ref}_{modo}.log", "w", encoding="utf-8") as f:
            subprocess.run([sys.executable, "scripts/control_comparado.py", tanda,
                            str(procesos), ref, modo], cwd=RAIZ, stdout=f,
                           stderr=subprocess.STDOUT, env={**__import__("os").environ,
                                                          "PYTHONIOENCODING": "utf-8"})
    (LOG / "DONE_control").write_text("completo\n")
    print(f"{time.strftime('%H:%M')} completo", flush=True)
