"""Reproducibilidad: fija semillas de numpy y torch."""

from __future__ import annotations

import random


def set_seed(seed: int) -> None:
    """Fija las semillas de random, numpy y torch.

    Se llama al principio de cualquier script cuyo resultado se reporte. Ojo:
    varios entrenamientos del proyecto hoy ya son deterministas sin esto (los
    parametros arrancan en un valor fijo y no hay barajado ni dropout), asi que
    fijar la semilla no cambia sus numeros. Es defensivo: el dia que alguien
    agregue inicializacion aleatoria, barajado de ventanas o un split al azar,
    el script dejaria de ser reproducible en silencio.
    """
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    try:
        import torch
    except ImportError:          # los scripts de figuras no necesitan torch
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_plot_style() -> None:
    """Estilo común para las figuras (matplotlib)."""
    raise NotImplementedError
