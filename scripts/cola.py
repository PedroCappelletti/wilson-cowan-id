#!/usr/bin/env python3
# Cola de corridas con lugares fijos, para lotes largos.
#
# Cada corrida pendiente arranca apenas se libera un lugar, en el orden de la
# lista, así que conviene ponerlas de la más larga a la más corta. Una corrida
# puede esperar el json de otra (la forma exacta, a su white-box).
#
# 4 lugares de 1 hilo es lo que más rinde en esta máquina (scripts/bench_hilos.py).

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "results/escalado"
LOG = RAIZ / "logs/escalado"


def terminada(tag: str) -> bool:
    """Tiene json, o su log muestra que se cayó. Una caída también libera lugar."""
    if (RES / f"{tag}.json").exists():
        return True
    log = LOG / f"{tag}.log"
    return log.exists() and "Traceback" in log.read_text(errors="ignore")


def ocupando(tag: str) -> bool:
    """Una corrida de otro lote que arrancó (tiene log) y todavía no terminó."""
    return (LOG / f"{tag}.log").exists() and not terminada(tag)


def correr(pendientes, marca: str, lugares: int = 4, esperar=(), extra=(),
           ajenas=()):
    """pendientes: lista de (tag, argumentos de esc_run, tag del que depende o None).

    esperar: archivos en logs/escalado que tienen que existir antes de arrancar,
    para encadenar lotes. extra: argumentos que llevan todas. ajenas: tags de
    otro lote que corre en paralelo; mientras estén andando ocupan lugar, así
    las dos colas juntas no pasan de `lugares` corridas.
    """
    LOG.mkdir(parents=True, exist_ok=True)
    esperar = [esperar] if isinstance(esperar, str) else list(esperar)
    while not all((LOG / e).exists() for e in esperar):
        time.sleep(60)

    env = {**os.environ, "WC_THREADS": "1"}
    propias: dict[str, subprocess.Popen] = {}
    pendientes = [j for j in pendientes if not terminada(j[0])]
    print(f"{time.strftime('%H:%M')} {len(pendientes)} corridas en cola", flush=True)

    while pendientes or propias:
        for t in [t for t, p in propias.items() if p.poll() is not None]:
            del propias[t]
        while len(propias) + sum(map(ocupando, ajenas)) < lugares:
            listas = [j for j in pendientes if j[2] is None or terminada(j[2])]
            if not listas:
                break
            tag, args, _ = listas[0]
            pendientes.remove(listas[0])
            with open(LOG / f"{tag}.log", "w") as f:
                propias[tag] = subprocess.Popen(
                    [sys.executable, "scripts/esc_run.py", *args, *extra, "--tag", tag],
                    cwd=RAIZ, env=env, stdout=f, stderr=subprocess.STDOUT)
            print(f"{time.strftime('%H:%M')} arranca {tag}", flush=True)
        time.sleep(30)

    (LOG / marca).write_text("completo\n")
    print(f"{time.strftime('%H:%M')} lote completo", flush=True)
