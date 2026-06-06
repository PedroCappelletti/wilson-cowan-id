"""Carga de archivos de configuración YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Lee un archivo YAML de configuración y lo devuelve como dict."""
    raise NotImplementedError
