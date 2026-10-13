#!/usr/bin/env python3
# Datasets ampliados del escalado, con particion de validacion.
#
# Las mismas dos plantas que esc_gen_datasets.py, sobre los 49 escenarios de
# build_scenarios_ampliado (28 de entrenamiento, 7 de validacion, 14 de test).
# Van a archivos nuevos, refrac1_amp y act1_amp: regenerar refrac1 o act1
# cambiaria su data_sha256 y dejaria sin respaldo las corridas hechas sobre ellos.
#
# Las versiones con ruido se arman despues con esc_gen_ruido.py, igual que las
# de los datasets originales:
#     python scripts/esc_gen_ruido.py act1_amp 0.01 0.05
#
# wc0_amp es el control sin perturbación: Wilson-Cowan puro sobre los mismos
# escenarios, para separar lo que aporta la perturbación de lo que falta de
# información en cada familia de estímulos.
#
# USO:  python scripts/esc_gen_ampliado.py          las dos plantas
#       python scripts/esc_gen_ampliado.py wc0      solo el control

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from src.wilson_cowan import Refractoriness, Actuator, NoPerturbation

from gen_multi_dataset import build_scenarios_ampliado
from gen_uncertain_dataset import generar_con, OUT_DIR


def main():
    if "wc0" in sys.argv[1:]:
        generar_con(lambda: NoPerturbation(), OUT_DIR / "wc0_amp.npz",
                    {"eps": 0.0}, escenarios=build_scenarios_ampliado)
        return
    generar_con(lambda: Refractoriness(r=0.10), OUT_DIR / "refrac1_amp.npz",
                {"eps": 1.0}, escenarios=build_scenarios_ampliado)
    generar_con(lambda: Actuator(sat=3.0, tau_act=1.0), OUT_DIR / "act1_amp.npz",
                {"eps": 1.0}, escenarios=build_scenarios_ampliado)
    print("\nListo.")


if __name__ == "__main__":
    main()
