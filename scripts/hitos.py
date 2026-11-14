#!/usr/bin/env python3
# Hitos de la cola de amplitud (cola_amplitud.py) y paquetes de resultados.
#
#   python scripts/hitos.py              estado: hitos cumplidos, corridas hechas, caídas
#   python scripts/hitos.py --empaquetar arma el zip de cada hito cumplido que no tenga zip
#   python scripts/hitos.py --esperar    se bloquea hasta que pase algo que haya que
#                                        atender (hito nuevo, corrida caída, cola muerta)
#                                        y lo imprime. Pensado para correrlo en segundo
#                                        plano y despertarse cuando termina.
#
# Los zips quedan en salida_hitos/, con todo lo terminado hasta ese hito: json,
# checkpoints y logs de e25, e26 y e27, y el log de la cola.

from __future__ import annotations

import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import RES, LOG
from cola_amplitud import tareas

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "salida_hitos"
ESTADO = SALIDA / "estado.json"

HITOS = [
    ("hito_1_e25_semilla0", lambda t: t.startswith("e25_") and t.endswith("_s0")),
    ("hito_2_e25_semilla1", lambda t: t.startswith("e25_") and t.endswith(("_s0", "_s1"))),
    ("hito_3_e25_completo", lambda t: t.startswith("e25_")),
    ("hito_4_e26_completo", lambda t: t.startswith(("e25_", "e26_"))),
    ("hito_5_todo", lambda t: True),
]


def hecha(tag):
    return (RES / f"{tag}.json").exists()


def caida(tag):
    log = LOG / f"{tag}.log"
    return (not hecha(tag)) and log.exists() and "Traceback" in log.read_text(errors="ignore")


def cola_viva():
    cmd = ("(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
           "Where-Object { $_.CommandLine -match 'cola_amplitud' }).Count")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
    try:
        return int(r.stdout.strip() or 0) > 0
    except ValueError:
        return True


def estado():
    tags = [t for t, _, _ in tareas()]
    hechas = {t for t in tags if hecha(t)}
    caidas = sorted(t for t in tags if caida(t))
    cumplidos = [n for n, f in HITOS if all(t in hechas for t in tags if f(t))]
    return {"total": len(tags), "hechas": len(hechas), "caidas": caidas, "hitos": cumplidos,
            "terminada": (LOG / "DONE_amplitud").exists()}


def empaquetar():
    SALIDA.mkdir(exist_ok=True)
    e = estado()
    nuevos = []
    for nombre in e["hitos"]:
        z = SALIDA / f"{nombre}.zip"
        if z.exists():
            continue
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
            for pre in ("e25_", "e26_", "e27_"):
                for p in RES.glob(f"{pre}*.json"):
                    f.write(p, f"results/escalado/{p.name}")
                for p in (RES / "models").glob(f"{pre}*.pt"):
                    f.write(p, f"results/escalado/models/{p.name}")
                for p in LOG.glob(f"{pre}*.log"):
                    f.write(p, f"logs/escalado/{p.name}")
            for p in LOG.glob("cola_amplitud*.out"):
                f.write(p, f"logs/escalado/{p.name}")
            f.writestr("estado.json", json.dumps(e, indent=2))
        nuevos.append(z.name)
    return nuevos


def esperar(cada=120):
    antes = estado()
    while True:
        time.sleep(cada)
        e = estado()
        if set(e["hitos"]) - set(antes["hitos"]):
            print("HITO", ",".join(sorted(set(e["hitos"]) - set(antes["hitos"]))), flush=True)
            return
        if set(e["caidas"]) - set(antes["caidas"]):
            print("CAIDA", ",".join(sorted(set(e["caidas"]) - set(antes["caidas"]))), flush=True)
            return
        if e["terminada"] or not cola_viva():
            print("COLA_TERMINADA" if e["terminada"] else "COLA_MUERTA", flush=True)
            return


if __name__ == "__main__":
    if "--esperar" in sys.argv:
        esperar()
    elif "--empaquetar" in sys.argv:
        print("nuevos:", empaquetar())
    print(json.dumps(estado(), indent=2))
