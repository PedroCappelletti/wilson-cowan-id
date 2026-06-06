"""Generación de datos sintéticos a partir del modelo de Wilson-Cowan.

Simula trayectorias del modelo (potencialmente con distintas condiciones
iniciales y/o parámetros) y las guarda en disco para entrenar la PINN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.wilson_cowan import WilsonCowan, WilsonCowanParams


def generate_dataset(
    params: WilsonCowanParams,
    initial_conditions: list[tuple[float, float]],
    t_span: tuple[float, float],
    dt: float,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Genera un dataset de trayectorias del modelo de Wilson-Cowan.

    Args:
        params: parámetros del modelo.
        initial_conditions: lista de tuplas (E0, I0).
        t_span: (t_inicial, t_final).
        dt: paso de integración.
        noise_std: desvío del ruido gaussiano de observación (0 = sin ruido).
        seed: semilla para reproducibilidad.

    Returns:
        dict con claves como 't', 'E', 'I' (y metadatos).
    """
    raise NotImplementedError


def save_dataset(dataset: dict[str, np.ndarray], path: str | Path) -> None:
    """Guarda el dataset en disco (.npz)."""
    raise NotImplementedError


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Carga un dataset desde disco (.npz)."""
    raise NotImplementedError
