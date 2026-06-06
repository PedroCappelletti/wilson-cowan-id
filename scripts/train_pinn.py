#!/usr/bin/env python3
"""Script de línea de comandos para entrenar la PINN.

Uso:
    python scripts/train_pinn.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenar la PINN de Wilson-Cowan")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default="data/processed/dataset.npz")
    args = parser.parse_args()

    # TODO: cargar config y datos, construir PINN, instanciar Trainer y entrenar.
    raise NotImplementedError


if __name__ == "__main__":
    main()
