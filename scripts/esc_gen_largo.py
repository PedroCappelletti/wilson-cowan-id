#!/usr/bin/env python3
# Datasets de 2 s muestreados a 2 kHz (desde el 22-10).
#
# Los mismos 49 escenarios y las mismas tres plantas que esc_gen_ampliado.py,
# con dos cambios:
#
#   - 2 s en vez de 200 ms, para que las frecuencias de los estímulos (chirp,
#     envolvente theta, trenes de pulsos) tengan varios ciclos. El estímulo va de
#     100 a 1900 ms.
#   - dt = 0,5 ms (2 kHz) en vez de 0,05 ms (20 kHz). El criterio, sin correr
#     nada: sobre las trayectorias de 20 kHz, el 99,99 % de la potencia de E e I
#     está por debajo de 420 Hz, y por encima la potencia cae más de 40 dB bajo
#     el pico. Nyquist pide más de 840 Hz; 2 kHz deja un margen de 2,4. La
#     dinámica más rápida del modelo tiene |λ| = 0,9 / ms, así que con dt = 0,5 ms
#     el RK4 queda en |λ|·dt = 0,45, lejos de su límite de estabilidad (2,8).
#
# El estímulo se mantiene constante entre muestras (`retenido`), como lo haría
# un láser comandado a 2 kHz y como lo integra el modelo. La planta integra con
# 10 subpasos por muestra (0,05 ms), la misma resolución de antes.
#
# Archivos: refrac1_2s, act1_2s, wc0_2s. Ruido con esc_gen_ruido.py --ref-fs.
#
# USO:  python scripts/esc_gen_largo.py

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from src.wilson_cowan import Refractoriness, Actuator, NoPerturbation

from gen_multi_dataset import build_scenarios_ampliado
from gen_uncertain_dataset import generar_con, OUT_DIR

T_SPAN = (0.0, 2000.0)      # ms
DT = 0.5                    # ms
N_EVAL = int(round((T_SPAN[1] - T_SPAN[0]) / DT)) + 1
TON, TOFF = 100.0, 1900.0   # ms


def escenarios():
    return build_scenarios_ampliado(TON, TOFF)


def main():
    kw = dict(escenarios=escenarios, t_span=T_SPAN, n_eval=N_EVAL,
              retenido=True, n_sub=10)
    plantas = [("wc0_2s", NoPerturbation, {"eps": 0.0}),
               ("refrac1_2s", lambda: Refractoriness(r=0.10), {"eps": 1.0}),
               ("act1_2s", lambda: Actuator(sat=3.0, tau_act=1.0), {"eps": 1.0})]
    pedidas = sys.argv[1:]
    for nombre, fab, meta in plantas:
        if pedidas and nombre not in pedidas:
            continue
        generar_con(fab, OUT_DIR / f"{nombre}.npz", meta, **kw)
    print("\nListo.")


if __name__ == "__main__":
    main()
