#!/usr/bin/env python3
# Tabla de resultados sin ruido y con los dos niveles de ruido, por configuración.
#
# Con ruido todas las corridas usan media móvil de 7 muestras. El NRMSE se
# compara solo dentro de una misma columna: la normalización por rango pico a
# pico se infla con el ruido. El error de parámetros sí se compara entre columnas.
#
#   python scripts/tabla_ruido.py            # escribe results/tabla_ruido.md

from __future__ import annotations

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "results/escalado"

# (planta, configuración, sin ruido, sigma 0.01, sigma 0.05, en el póster)
FILAS = [
    ("refrac1", "white-box", "e1_wb", "e7_e1wb_n01", "e7_e1wb_n05", True),
    ("refrac1", "corrección agnóstica, ventana 400", "e1_B_w400", "e7_e1B400_n01", "e7_e1B400_n05", True),
    ("refrac1", "forma exacta", "e1_S2", "e7_e1S2_n01", "e7_e1S2_n05", True),
    ("refrac1", "forma exacta y red", "e4_Sg", "e8_Sg_n01", "e8_Sg_n05", False),
    ("act1", "white-box", "e2_wb", "e7_e2wb_n01", "e6_wb_n05_s7", True),
    ("act1", "corrección agnóstica", "e2_B", "e7_e2B_n01", "e7_e2B_n05", True),
    ("act1", "estado de filtro", "e2_lag2", "e7_e2lag_n01", "e7_e2lag_n05", True),
    ("act1", "comando actual (A)", "e2_A", None, None, False),
    ("act1", "historia cruda, 100", "e2_H100", "e8_H100_n01", "e8_H100_n05", False),
    ("act1", "historia cruda, 400", "e2_H400", "e8_H400_n01", "e6_H400_n05_s7", False),
    ("act1", "FIR, 4 canales", "e3_K400", "e8_K400_n01", "e6_K400_n05_s7", False),
    ("act1", "FIR, 8 canales", "e3_K400_f8", "e8_K400f8_n01", "e8_K400f8_n05", False),
    ("act1", "FIR, 16 canales", "e9_K400f16", "e9_K400f16_n01", "e9_K400f16_n05", False),
    ("act1", "FIR, 800 retardos", "e3_K800", None, None, False),
    ("act1", "FIR, red de ancho 64", "e3_K400_h64", None, None, False),
    ("act1", "FIR, L2 1e-4", "e5_K400_wd1e-4", "e8_K400wd4_n01", "e8_K400wd4_n05", False),
    ("act1", "FIR, L2 1e-3", "e5_K400_wd1e-3", "e8_K400wd3_n01", "e8_K400wd3_n05", False),
    ("act1", "latent ODE, ventana 100", "e2_lat_w100", None, None, False),
    ("act1", "latent ODE, ventana 400", "e2_lat_w400", None, None, False),
]


def piso(planta: str, nivel: str | None) -> float:
    """NRMSE de un modelo perfecto: la trayectoria verdadera contra los datos con
    ruido suavizados, que es contra lo que se evalúa. Ningún modelo baja de acá.
    Sin ruido vale 0."""
    import numpy as np
    if nivel is None:
        return 0.0
    base = RAIZ / "data/processed/uncertain"
    c = np.load(base / f"{planta}.npz", allow_pickle=True)
    r = np.load(base / f"{planta}_n{nivel}.npz", allow_pickle=True)
    ker = np.ones(7) / 7
    vals = []
    for s in np.where(c["is_test"].astype(bool))[0]:
        obs = np.stack([np.convolve(r[k][s], ker, mode="same") for k in ("I", "E")], 1)
        verdad = np.stack([c["I"][s], c["E"][s]], 1)
        rango = obs.max(0) - obs.min(0)
        vals.append((100 * np.sqrt(((verdad - obs) ** 2).mean(0)) / rango).mean())
    return float(np.mean(vals))


def celda(tag):
    if tag is None:
        return "no se repite"
    f = RES / f"{tag}.json"
    if not f.exists():
        return "pendiente"
    d = json.loads(f.read_text())
    return (f"{d['nrmse_test']:.2f} / {d['r2_delta_test']:+.2f} / "
            f"{d['mean_param_error']:.1f}").replace(".", ",")


def main():
    lineas = [
        "| planta | configuración | póster | sin ruido | σ = 0,01 | σ = 0,05 |",
        "|---|---|---|---|---|---|",
    ]
    for i, (planta, conf, *tags, poster) in enumerate(FILAS):
        if i == 0 or FILAS[i - 1][0] != planta:
            pisos = [f"{piso(planta, n):.2f}".replace(".", ",") for n in (None, "01", "05")]
            lineas.append(f"| `{planta}` | *modelo perfecto (piso del NRMSE)* |  | "
                          + " | ".join(f"*{x}*" for x in pisos) + " |")
        lineas.append(f"| `{planta}` | {conf} | {'sí' if poster else ''} | "
                      + " | ".join(celda(t) for t in tags) + " |")
    pie = ("\n*Cada celda: NRMSE (%) / R² de la corrección / error de parámetros (%). "
           "El piso es el NRMSE que sacaría la trayectoria verdadera contra los datos con "
           "ruido suavizados: lo mejor posible en ese nivel. "
           "Con ruido, media móvil de 7 muestras sobre I y E. El NRMSE se compara "
           "solo dentro de una columna; el error de parámetros también entre columnas. "
           "Una semilla por celda.*\n")
    texto = "\n".join(lineas) + "\n" + pie
    (RAIZ / "results/tabla_ruido.md").write_text(texto, encoding="utf-8")
    print(texto)
    faltan = sum(celda(t) == "pendiente" for _, _, *ts, _ in FILAS for t in ts)
    print(f"{faltan} celdas pendientes")


if __name__ == "__main__":
    main()
