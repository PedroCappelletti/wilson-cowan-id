"""Reproducibilidad: fija semillas de numpy y torch."""

from __future__ import annotations


def set_seed(seed: int) -> None:
    """Fija las semillas de random, numpy y torch."""
    raise NotImplementedError


def set_plot_style() -> None:
    """Estilo común para las figuras (matplotlib)."""
    raise NotImplementedError
