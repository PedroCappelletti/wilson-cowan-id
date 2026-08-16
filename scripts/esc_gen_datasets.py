#!/usr/bin/env python3
# =============================================================================
#  DATASETS PARA EL ESCALADO PROGRESIVO  (plan: docs/plan_escalado_progresivo.md)
# =============================================================================
#
#  Genera los datasets con UNA SOLA perturbacion activa, para poder atacar la
#  complejidad de a una por vez:
#
#    refrac1.npz -> solo refractariedad (r=0.10). Sin memoria: es el caso que el
#                   gray-box g(I,E) DEBERIA poder capturar (techo R2 = 0.97).
#    act1.npz    -> solo actuador (tau=1.0 ms, sat=3.0). Con memoria: g(I,E) no
#                   puede por diseno; requiere estados aumentados.
#
#  Los valores son los mismos que usa default_uncertainty(eps=1), asi los
#  resultados son comparables con los experimentos F que usaron las dos juntas.
#
#  El eps0.npz de referencia (etapa 0) ya existe y no se regenera.
#
#  USO:  python scripts/esc_gen_datasets.py
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from src.wilson_cowan import Refractoriness, Actuator

from gen_uncertain_dataset import generar_con, OUT_DIR


def main():
    generar_con(lambda: Refractoriness(r=0.10),
                OUT_DIR / "refrac1.npz", {"eps": 1.0})
    generar_con(lambda: Actuator(sat=3.0, tau_act=1.0),
                OUT_DIR / "act1.npz", {"eps": 1.0})
    print("\nListo.")


if __name__ == "__main__":
    main()
