#!/usr/bin/env python3
# La tabla del póster sobre el dataset ampliado (esc_gen_ampliado.py).
#
# Las doce configuraciones de tabla_ruido.py, con los mismos argumentos que en
# el póster, sobre refrac1_amp y act1_amp. Ahora el modelo que se evalúa es el
# de mejor validación (esc_run.py lo hace solo cuando el dataset trae is_val), y
# las elecciones entre variantes se hacen por nrmse_val_mejor.
#
# Orden: σ = 0,01 entero, después 0,05, después sin ruido, y dentro de cada
# nivel de la más larga a la más corta. Con σ = 0,01 ya se ve si el óptimo de
# canales del filtro y el de la ventana se mueven al duplicar los escenarios.
#
# Los tags llevan el prefijo e13_.

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import RES, correr

K = ["--variant", "K", "--hist", "400"]
CONFIGS = [   # (nombre, planta, argumentos), de la más lenta a la más rápida
    ("lag",     "act1",    ["--variant", "lag"]),
    ("K400f16", "act1",    K + ["--n-fir", "16"]),
    ("K400f8",  "act1",    K + ["--n-fir", "8"]),
    ("K400",    "act1",    K + ["--n-fir", "4"]),
    ("H400",    "act1",    ["--variant", "H", "--hist", "400"]),
    ("B400",    "refrac1", ["--variant", "B", "--window", "400"]),
    ("H100",    "act1",    ["--variant", "H", "--hist", "100"]),
    ("e2wb",    "act1",    ["--variant", "whitebox"]),
    ("e1wb",    "refrac1", ["--variant", "whitebox"]),
    ("e2B",     "act1",    ["--variant", "B"]),
    ("B100",    "refrac1", ["--variant", "B", "--window", "100"]),
]


def corridas(suf):
    """Las doce de un nivel de ruido. suf: "_n01", "_n05" o "" (sin ruido)."""
    suave = ["--smooth", "7"] if suf else []
    out = [(f"e13_{n}{suf}", ["--data", f"{p}_amp{suf}", *a, *suave], None)
           for n, p, a in CONFIGS]
    # La forma exacta arranca del β de su white-box, como en el póster.
    wb = f"e13_e1wb{suf}"
    out.append((f"e13_e1S2{suf}",
                ["--data", f"refrac1_amp{suf}", "--variant", "S", "--r-init", "0.05",
                 "--init-params", str(RES / f"{wb}.json"), *suave], wb))
    return out


if __name__ == "__main__":
    correr(corridas("_n01") + corridas("_n05") + corridas(""),
           marca="DONE_ampliado")
