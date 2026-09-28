#!/usr/bin/env python3
# El informe del 11-09, otra vez con ruido.
#
# Todo lo que el informe mide sobre act1 y refrac1 corrió sin ruido. Se repite
# con sigma = 0.05 y 0.01 y media móvil de 7 muestras, el mismo protocolo que
# las corridas del póster. Arranca solo cuando esas terminan, porque la variante
# Sg arranca del white-box con ruido que calcula ese lote.
#
# Ya existen con el protocolo y no se repiten: e6_H400_n05_s7, e6_K400_n05_s7 y
# e6_wb_n05_s7. La verificación del dataset (e4_S2ver) no se repite: la forma
# exacta con ruido ya está en el lote del póster.

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import RES, correr

K = ["--variant", "K", "--hist", "400"]


def lote(n: str):
    a = ["--data", f"act1_n{n}"]
    return {
        "K800":   (a + ["--variant", "K", "--hist", "800", "--n-fir", "4"], None),
        "K400h64": (a + K + ["--n-fir", "4", "--hidden", "64"], None),
        "K400f8": (a + K + ["--n-fir", "8"], None),
        "K400wd4": (a + K + ["--n-fir", "4", "--wd-fir", "1e-4"], None),
        "K400wd3": (a + K + ["--n-fir", "4", "--wd-fir", "1e-3"], None),
        "K400":   (a + K + ["--n-fir", "4"], None),
        "H400":   (a + ["--variant", "H", "--hist", "400"], None),
        "Sg":     (["--data", f"refrac1_n{n}", "--variant", "Sg", "--r-init", "0.05",
                    "--init-params", str(RES / f"e7_e1wb_n{n}.json")], f"e7_e1wb_n{n}"),
        "H100":   (a + ["--variant", "H", "--hist", "100"], None),
        "A":      (a + ["--variant", "A"], None),
    }


YA_HECHAS = {("H400", "05"), ("K400", "05")}

# De la más larga a la más corta, intercalando los dos niveles de ruido.
ORDEN = ["K800", "K400h64", "K400f8", "K400wd4", "K400wd3", "K400", "H400",
         "Sg", "H100", "A"]

if __name__ == "__main__":
    pendientes = []
    for nombre in ORDEN:
        for n in ("05", "01"):
            if (nombre, n) in YA_HECHAS:
                continue
            args, dep = lote(n)[nombre]
            pendientes.append((f"e8_{nombre}_n{n}", args, dep))
    correr(pendientes, marca="DONE_informe_ruido", esperar="DONE_poster_ruido",
           extra=("--smooth", "7"))
