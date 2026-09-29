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


if __name__ == "__main__":
    # La agnóstica de refractariedad con ventana de 400 tarda unas 7 horas y el
    # filtro unas 2: van primero las largas, y dentro de cada una, con ruido
    # primero, que es lo que decide si entra al póster.
    todas = [corrida(n, s, c)
             for n in ("B400", "K400f8")
             for s in ("_n01", "_n05", "")
             for c in (True, False)]
    correr(todas, marca="DONE_dos_etapas")
