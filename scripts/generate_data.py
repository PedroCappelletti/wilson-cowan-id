#!/usr/bin/env python3
"""Script de línea de comandos para generar los datos de Wilson-Cowan.

Uso:
    python scripts/generate_data.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Generar dataset de Wilson-Cowan")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="data/processed/dataset.npz")
    args = parser.parse_args()

    # TODO: cargar config, instanciar WilsonCowan, generar y guardar dataset.
    raise NotImplementedError


if __name__ == "__main__":
    main()
