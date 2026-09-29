#!/usr/bin/env python3
# Actualiza la tabla y la figura de resultados de la carpeta "Validación de
# arquitecturas" del vault con lo que haya en results/escalado.
#
#   python scripts/vault_validacion.py

from __future__ import annotations

import io
import re
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
VAULT = Path(r"C:\Users\User\Desktop\Vault\01-Projects\Investigación Neurociencia"
             r"\Validación de arquitecturas")
NOTA = VAULT / "Validación de arquitecturas - variantes y resultados.md"


def main():
    for s in ("tabla_ruido.py", "figura_ruido.py"):
        subprocess.run([sys.executable, str(RAIZ / "scripts" / s)], cwd=RAIZ,
                       check=True, capture_output=True)
    shutil.copy(RAIZ / "results/figures/ruido_comparacion.png",
                VAULT / "resultados-con-y-sin-ruido.png")

    tabla = io.open(RAIZ / "results/tabla_ruido.md", encoding="utf-8").read()
    filas = [l for l in tabla.splitlines() if l.startswith("|")]
    pendientes = sum(l.count("pendiente") for l in filas)
    nueva = "\n".join(filas).replace("-", "−").replace("|−−−", "|---")
    nueva = re.sub(r"\|(−)+", lambda m: "|" + "-" * (len(m.group(0)) - 1), nueva)

    s = io.open(NOTA, encoding="utf-8").read()
    ini = s.index("| planta | configuración |")
    fin = s.index("\n\n", ini)
    s = s[:ini] + nueva + s[fin:]
    s = re.sub(r"faltan \d+ celdas", f"faltan {pendientes} celdas", s)
    io.open(NOTA, "w", encoding="utf-8").write(s)
    print(f"nota y figura actualizadas, {pendientes} celdas pendientes")


if __name__ == "__main__":
    main()
