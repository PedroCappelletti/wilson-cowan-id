#!/usr/bin/env python3
# Cola para el resto del lote del poster con ruido.
#
# Reemplaza a los carriles fijos de run_poster_ruido.sh, que dejaban tres de las
# corridas largas del actuador en un mismo carril y dos nucleos ociosos durante
# dos horas. Aca hay 4 lugares y cada corrida pendiente arranca, la mas larga
# primero, apenas se libera uno. Las cuatro corridas que ya estaban andando
# cuando se armo la cola siguen vivas y ocupan su lugar hasta que terminan.

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "results/escalado"
LOG = RAIZ / "logs/escalado"
LUGARES = 4

# Ya andando antes de la cola: ocupan lugar hasta que aparece su json.
EN_CURSO = ["e7_e1wb_n05", "e7_e1wb_n01", "e7_e2B_n05", "e7_e2wb_n01"]

# Pendientes, de la mas larga a la mas corta. (tag, argumentos, depende de)
PENDIENTES = [
    ("e7_e2lag_n05", ["--data", "act1_n05", "--variant", "lag"], None),
    ("e7_e2lag_n01", ["--data", "act1_n01", "--variant", "lag"], None),
    ("e7_e2B_n01", ["--data", "act1_n01", "--variant", "B"], None),
    ("e7_e1B400_n05", ["--data", "refrac1_n05", "--variant", "B", "--window", "400"], None),
    ("e7_e1B400_n01", ["--data", "refrac1_n01", "--variant", "B", "--window", "400"], None),
    ("e7_e1S2_n05", ["--data", "refrac1_n05", "--variant", "S", "--r-init", "0.05",
                     "--init-params", str(RES / "e7_e1wb_n05.json")], "e7_e1wb_n05"),
    ("e7_e1S2_n01", ["--data", "refrac1_n01", "--variant", "S", "--r-init", "0.05",
                     "--init-params", str(RES / "e7_e1wb_n01.json")], "e7_e1wb_n01"),
]


def terminada(tag: str) -> bool:
    """Tiene json, o su log muestra que se cayo. Una caida tambien libera lugar."""
    if (RES / f"{tag}.json").exists():
        return True
    log = LOG / f"{tag}.log"
    return log.exists() and "Traceback" in log.read_text(errors="ignore")


def main():
    env = {**os.environ, "WC_THREADS": "1"}
    propias: dict[str, subprocess.Popen] = {}
    pendientes = list(PENDIENTES)

    while pendientes or propias:
        for t in [t for t, p in propias.items() if p.poll() is not None]:
            del propias[t]
        ocupados = len(propias) + sum(not terminada(t) for t in EN_CURSO)

        while ocupados < LUGARES:
            lista = [j for j in pendientes if j[2] is None or terminada(j[2])]
            if not lista:
                break
            tag, args, _ = lista[0]
            pendientes.remove(lista[0])
            with open(LOG / f"{tag}.log", "w") as f:
                propias[tag] = subprocess.Popen(
                    [sys.executable, "scripts/esc_run.py", *args, "--smooth", "7",
                     "--tag", tag], cwd=RAIZ, env=env, stdout=f,
                    stderr=subprocess.STDOUT)
            print(f"{time.strftime('%H:%M')} arranca {tag}", flush=True)
            ocupados += 1
        time.sleep(30)

    while not all(terminada(t) for t in EN_CURSO):
        time.sleep(30)
    (LOG / "DONE_poster_ruido").write_text("POSTER CON RUIDO COMPLETO\n")
    print(f"{time.strftime('%H:%M')} lote completo", flush=True)


if __name__ == "__main__":
    main()
