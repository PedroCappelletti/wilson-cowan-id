#!/usr/bin/env python3
# Arranque desde el white-box para el estado de filtro.
#
# Era la unica correccion sin version con arranque: fit_aug no aceptaba
# --init-params y por eso quedo afuera de cola_arranque_wb.py. Los diez
# parametros fisicos del modelo de estado aumentado viven en model.core, que es
# el mismo GrayBoxWC del graybox suelto, asi que es la misma copia de crudos.
# El soporte se agrego en esc_run.py (cargar_params).
#
# Congelar beta no aplica: fit_aug siempre lo entrena.
#
# Los tres niveles, para que la fila quede completa en la Fig. 2 del poster.
# Sin ruido tarda ~28 min y cada una con ruido ~120 min.

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import RES, correr

WB = {"": "e2_wb", "_n01": "e7_e2wb_n01", "_n05": "e6_wb_n05_s7"}


def corrida(suf):
    a = ["--data", "act1" + suf, "--variant", "lag",
         "--init-params", str(RES / f"{WB[suf]}.json")]
    if suf:
        a += ["--smooth", "7"]
    return f"e12_lag_wrm{suf}", a, None


if __name__ == "__main__":
    # Las de ruido primero: son las que la figura necesita, porque el resto de
    # las filas con arranque tampoco tiene nivel sin ruido.
    correr([corrida(s) for s in ("_n01", "_n05", "")],
           marca="DONE_lag_wb", lugares=2)
