#!/usr/bin/env python3
# Arranque desde el white-box para todas las correcciones con red.
#
# En refractariedad, arrancar el entrenamiento conjunto desde el β del white-box
# (sin congelar) hizo que la corrección agnóstica identificara los parámetros y
# recuperara buena parte del término faltante (cola_dos_etapas.py). Acá se
# prueba lo mismo en todas las variantes que tienen versión desde cero con
# ruido, para compararlas en las mismas condiciones.
#
# Quedan afuera el comando actual, K800, la red de ancho 64 y la latent ODE,
# que dieron negativo sin ruido y no tienen versión con ruido contra la cual
# comparar; y el estado de filtro, que se entrena con fit_aug, que todavía no
# acepta --init-params. La forma exacta ya arrancaba del white-box.
#
# La agnóstica de refractariedad con ventana de 100 muestras no tenía versión
# desde cero con ruido: se corre también, para tener contra qué comparar.

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import RES, correr

WB = {("refrac1", "_n01"): "e7_e1wb_n01", ("refrac1", "_n05"): "e7_e1wb_n05",
      ("act1", "_n01"): "e7_e2wb_n01", ("act1", "_n05"): "e6_wb_n05_s7"}

K = ["--variant", "K", "--hist", "400"]
VARIANTES = {          # nombre: (planta, argumentos), de la más lenta a la más rápida
    "K400f16": ("act1", K + ["--n-fir", "16"]),
    "H400":    ("act1", ["--variant", "H", "--hist", "400"]),
    "K400":    ("act1", K + ["--n-fir", "4"]),
    "K400wd4": ("act1", K + ["--n-fir", "4", "--wd-fir", "1e-4"]),
    "K400wd3": ("act1", K + ["--n-fir", "4", "--wd-fir", "1e-3"]),
    "H100":    ("act1", ["--variant", "H", "--hist", "100"]),
    "e2B":     ("act1", ["--variant", "B"]),
    "B100":    ("refrac1", ["--variant", "B", "--window", "100"]),
}

# Las corridas sin ruido de cola_dos_etapas.py, que siguen andando: ocupan lugar.
AJENAS = ["e10_B400_frz", "e10_B400_wrm", "e10_K400f8_frz", "e10_K400f8_wrm"]


def corrida(nombre, suf, desde_wb=True):
    planta, args = VARIANTES[nombre]
    a = ["--data", planta + suf, *args, "--smooth", "7"]
    if desde_wb:
        a += ["--init-params", str(RES / f"{WB[(planta, suf)]}.json")]
    return f"e11_{nombre}_{'wrm' if desde_wb else 'cero'}{suf}", a, None


if __name__ == "__main__":
    todas = [corrida(n, s) for n in VARIANTES for s in ("_n05", "_n01")]
    todas += [corrida("B100", s, desde_wb=False) for s in ("_n05", "_n01")]
    correr(todas, marca="DONE_arranque_wb", ajenas=AJENAS)
