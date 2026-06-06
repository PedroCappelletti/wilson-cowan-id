"""Bucle de entrenamiento de la PINN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TrainConfig:
    """Hiperparámetros de entrenamiento."""

    epochs: int = 10_000
    lr: float = 1e-3
    w_data: float = 1.0
    w_physics: float = 1.0
    w_ic: float = 1.0
    n_collocation: int = 1000
    device: str = "cpu"
    log_every: int = 100
    checkpoint_dir: str = "results/models"


class Trainer:
    """Orquesta el entrenamiento de la PINN."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainConfig,
    ) -> None:
        self.model = model
        self.config = config
        raise NotImplementedError

    def train(self, dataset) -> dict[str, list[float]]:
        """Entrena la red y devuelve el historial de pérdidas."""
        raise NotImplementedError

    def save_checkpoint(self, path: str | Path) -> None:
        raise NotImplementedError

    def load_checkpoint(self, path: str | Path) -> None:
        raise NotImplementedError
