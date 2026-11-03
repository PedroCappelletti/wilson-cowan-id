#!/usr/bin/env python3
# Tabla de cola_sesgo_ventanas.py.
#
# Por parámetro, ventana y nivel de ruido: sesgo relativo (media de β̂ sobre
# las cinco realizaciones menos el verdadero, en % del verdadero), su error
# estándar y el cociente |sesgo| / EE. Un cociente mayor que 3 es un
# desplazamiento que no explica la dispersión entre realizaciones. La corrida
# sin ruido es la referencia de lo que el entrenamiento recupera solo. W10s5
# es la misma prueba con el estado inicial suavizado (cola_x0_suave.py), y
# W10L1 y W10L10 con el estado inicial entrenable (cola_x0_libre.py).
#
#   python scripts/resumen_sesgo_ventanas.py

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.wilson_cowan.model import WilsonCowanParams

RES = RAIZ / "results" / "escalado"
OUT = RAIZ / "results" / "sesgo_ventanas.md"
VERDAD = asdict(WilsonCowanParams())
NOMBRES = ("wEE", "wEI", "wIE", "wII", "te", "ti", "ae", "ai", "thetae", "thetai")
SEMILLAS = range(5)


def coma(x, d=2):
    return f"{x:.{d}f}".replace(".", ",").replace("-", "−")


def leer(tag):
    f = RES / f"{tag}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def main():
    lineas = []
    for w, titulo in (("W10", "Ventanas de 5 ms (W10)"), ("W40", "Ventanas de 20 ms (W40)"),
                      ("W10s5", "Ventanas de 5 ms, x0 con media de 5 muestras (W10s5)"),
                      ("W10L1", "Ventanas de 5 ms, x0 entrenable, λ = 1 (W10L1)"),
                      ("W10L10", "Ventanas de 5 ms, x0 entrenable, λ = 10 (W10L10)"),
                      ("W10L1w", "Ventanas de 5 ms, x0 entrenable, λ = 1, desde el white-box W10 (W10L1w)")):
        limpio = leer(f"e17_wb_{w}_limpio")
        lineas += [f"**{titulo}**", "",
                   "| parámetro | sin ruido (%) | σ 0,01: sesgo ± EE (%) | \\|sesgo\\|/EE "
                   "| σ 0,05: sesgo ± EE (%) | \\|sesgo\\|/EE |", "|---|---|---|---|---|---|"]
        filas = {n: [coma(100 * (limpio["params"][n] / VERDAD[n] - 1)) if limpio else "falta"] for n in NOMBRES}
        for suf in ("_n01", "_n05"):
            corr = [leer(f"e17_wb_{w}{suf}_s{k}") for k in SEMILLAS]
            corr = [c for c in corr if c]
            for n in NOMBRES:
                if len(corr) < 2:
                    filas[n] += ["falta", ""]
                    continue
                rel = np.array([100 * (c["params"][n] / VERDAD[n] - 1) for c in corr])
                ee = rel.std(ddof=1) / np.sqrt(len(rel))
                filas[n] += [f"{coma(rel.mean())} ± {coma(ee)}", coma(abs(rel.mean()) / ee, 1)]
        lineas += [f"| {n} | " + " | ".join(v) + " |" for n, v in filas.items()] + [""]
    OUT.write_text("\n".join(lineas), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
