"""Arquitectura de la PINN.

Red feed-forward que mapea el tiempo t -> [E(t), I(t)]. Para problemas de
estimación de parámetros (inverse problem), los parámetros del modelo de
Wilson-Cowan pueden declararse como nn.Parameter entrenables.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PINN(nn.Module):
    """Red neuronal informada por la física.

    Args:
        in_dim: dimensión de entrada (1 para t).
        out_dim: dimensión de salida (2 para [E, I]).
        hidden_dim: neuronas por capa oculta.
        n_layers: cantidad de capas ocultas.
        activation: función de activación (tanh suele funcionar bien en PINNs).
    """

    def __init__(
        self,
        in_dim: int = 1,
        out_dim: int = 2,
        hidden_dim: int = 64,
        n_layers: int = 4,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: shape (N, 1) -> salida shape (N, 2) con [E, I]."""
        raise NotImplementedError
