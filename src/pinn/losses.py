"""Funciones de pérdida de la PINN.

La pérdida total combina típicamente:
    L = w_data * L_data + w_phys * L_physics + w_ic * L_ic

- L_data: ajuste a las observaciones (E, I) medidas.
- L_physics: residuo de las ODEs de Wilson-Cowan evaluado por autograd.
- L_ic: cumplimiento de las condiciones iniciales.
"""

from __future__ import annotations

import torch

from src.wilson_cowan import WilsonCowanParams


def data_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE entre la predicción de la red y las observaciones."""
    raise NotImplementedError


def physics_loss(
    model: torch.nn.Module,
    t_collocation: torch.Tensor,
    params: WilsonCowanParams,
) -> torch.Tensor:
    """Residuo de las ODEs de Wilson-Cowan en los puntos de colocación.

    Calcula dE/dt y dI/dt con autograd y mide cuánto se aleja la red de
    satisfacer el sistema de ecuaciones.
    """
    raise NotImplementedError


def initial_condition_loss(
    model: torch.nn.Module,
    t0: torch.Tensor,
    ic: torch.Tensor,
) -> torch.Tensor:
    """Penaliza la desviación respecto de la condición inicial [E0, I0]."""
    raise NotImplementedError
