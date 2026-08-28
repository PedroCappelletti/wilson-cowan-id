"""Procedencia de artefactos: con que script y en que commit se genero cada cosa.

Los datasets (`data/processed/`) y los checkpoints (`results/models/`) estan en
`.gitignore` a proposito, porque se regeneran. El costo de eso es que un archivo
suelto no dice de donde salio: si alguien lo pisa con otra configuracion, el
resultado que documentaba deja de ser recomputable. Estampar la procedencia
adentro del propio artefacto es lo que evita esa arqueologia.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def git_commit() -> dict[str, Any]:
    """Commit actual y si el arbol tiene cambios sin commitear.

    Devuelve `{"commit": None, "dirty": None}` si git no esta disponible o el
    directorio no es un repo, para que estampar procedencia nunca rompa una
    corrida.
    """
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=10,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}

    if commit.returncode != 0:
        return {"commit": None, "dirty": None}
    return {
        "commit": commit.stdout.strip(),
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def provenance(script: str, **extra: Any) -> dict[str, Any]:
    """Bloque de procedencia para guardar junto a un artefacto.

    `script` es el que produjo el artefacto (usar `__file__`). Todo lo que se
    pase en `extra` (semilla, dataset de entrada, hiperparametros) se agrega tal
    cual: es lo que hace falta para reconstruir el comando.
    """
    return {
        "script": Path(script).name,
        "generado": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        **git_commit(),
        **extra,
    }


def provenance_str(script: str, **extra: Any) -> str:
    """Igual que `provenance`, serializado, para meter en un `.npz`.

    `np.savez` solo guarda arrays, asi que la procedencia entra como string.
    Se lee con `json.loads(str(d["provenance"]))`.
    """
    return json.dumps(provenance(script, **extra))
