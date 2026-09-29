#!/usr/bin/env python3
# Entrenamiento en dos etapas: primero el backbone, después la corrección.
#
# En todas las corridas anteriores β y la corrección se entrenan juntos desde el
# arranque ignorante (los diez parámetros en 1,0). Acá β sale del white-box ya
# ajustado sobre los mismos datos y queda fijo (--freeze-phys), y se entrena solo
# la corrección. El brazo de control arranca del mismo white-box pero no congela,
# para separar el efecto de arrancar bien del efecto de congelar.
#
# Dos correcciones: la agnóstica de refractariedad con ventana de 400 muestras,
# la del póster, y el filtro entrenable de 8 canales en el actuador. La agnóstica
# del actuador no entra: sin el pasado del comando no puede representar el
# término, esté bien β o no.
#
# Con β congelado el error de parámetros es el del white-box por construcción, y
# el R2 contra Δf no mide lo que la corrección tiene que aprender. Para eso está
# r2_residuo_test, contra f_planta − f_WC(β̂).

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import RES, correr

# white-box de cada planta y nivel, del que sale β
WB = {("refrac1", ""): "e1_wb", ("refrac1", "_n01"): "e7_e1wb_n01",
      ("refrac1", "_n05"): "e7_e1wb_n05", ("act1", ""): "e2_wb",
      ("act1", "_n01"): "e7_e2wb_n01", ("act1", "_n05"): "e6_wb_n05_s7"}

CORR = {
    "B400": ("refrac1", ["--variant", "B", "--window", "400"]),
    "K400f8": ("act1", ["--variant", "K", "--hist", "400", "--n-fir", "8"]),
}


def corrida(nombre, suf, congelar):
    planta, args = CORR[nombre]
    tag = f"e10_{nombre}_{'frz' if congelar else 'wrm'}{suf}"
    a = ["--data", planta + suf, *args,
         "--init-params", str(RES / f"{WB[(planta, suf)]}.json")]
    if suf:
        a += ["--smooth", "7"]
    if congelar:
        a.append("--freeze-phys")
    return tag, a, None


# Orden de prioridad. La primera versión de la cola corría primero todas las de
# refractariedad; se reordenó a las 14:20 del 29/9 porque la máquina se apaga a
# las 19:15 y así las del filtro con ruido entran antes que las sin ruido.
ORDEN = [("B400", "_n01", True), ("B400", "_n01", False),
         ("B400", "_n05", True), ("B400", "_n05", False),
         ("K400f8", "_n01", True), ("K400f8", "_n05", True),
         ("K400f8", "_n01", False), ("K400f8", "_n05", False),
         ("B400", "", True), ("K400f8", "", True),
         ("K400f8", "", False), ("B400", "", False)]

# Las que ya estaban andando cuando se relanzó la cola: no se vuelven a lanzar,
# y mientras sigan vivas ocupan lugar.
EN_CURSO = ["e10_B400_frz_n01", "e10_B400_wrm_n01",
            "e10_B400_frz_n05", "e10_B400_wrm_n05"]

if __name__ == "__main__":
    todas = [c for c in (corrida(*o) for o in ORDEN) if c[0] not in EN_CURSO]
    correr(todas, marca="DONE_dos_etapas", ajenas=EN_CURSO)
