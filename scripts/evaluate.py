#!/usr/bin/env python3
"""Evaluación / comparación entre la PINN y la solución numérica de referencia.

Uso:
    python scripts/evaluate.py --checkpoint results/models/pinn.pt
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluar la PINN entrenada")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/processed/dataset.npz")
    args = parser.parse_args()

    # TODO: cargar checkpoint, predecir, comparar contra referencia y graficar.
    raise NotImplementedError


if __name__ == "__main__":
    main()
