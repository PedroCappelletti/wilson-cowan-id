"""Dataset de PyTorch para alimentar la PINN."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class WilsonCowanDataset(Dataset):
    """Envuelve un dataset generado para usarlo con DataLoader.

    Cada muestra suele ser (t, [E, I]); la PINN además evalúa el residuo
    físico en puntos de colocación que pueden no tener etiqueta.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError
