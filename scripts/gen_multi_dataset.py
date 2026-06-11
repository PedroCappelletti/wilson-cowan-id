#!/usr/bin/env python3
# =============================================================================
#  DATASET MULTI-VARIANTE para el Neural ODE (planta controlable) — régimen ms
# =============================================================================
#
#  Genera trayectorias de Wilson-Cowan bajo MUCHOS tipos de estimulo (escalones,
#  senoides, multisenos, chirps) variando amplitud y frecuencia. Con esa
#  diversidad el modelo de estados f_θ(x,P,Q) aprende a RESPONDER a entradas
#  variadas -> condicion para que despues responda a un controlador.
#
#  Convencion (decision tomada): TIEMPO EN ms (regimen del control). Las
#  frecuencias se dan en Hz y se convierten a ciclos/ms con hz(). Estado COMPLETO
#  [I,E] (sin EKF, como el controlador que tenemos).
#
#  SPLIT train/test en el ESPACIO DE ENTRADAS: algunos escenarios completos se
#  reservan como test (el modelo no los ve en entrenamiento) -> mide si generaliza
#  a estimulos nuevos, el analogo de "responder a un controlador con senal nueva".
#
#  Guarda un unico .npz con todas las trayectorias apiladas + la mascara de test.
#
#  USO:  python scripts/gen_multi_dataset.py
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.wilson_cowan import (
    WilsonCowanParams, box_pulse, sine_pulse, multisine_pulse, chirp_pulse,
)
from src.data import generate_dataset

# #############################################################################
# ##   ZONA EDITABLE                                                         ##
# #############################################################################

PARAMS = WilsonCowanParams()    # parametros verdaderos (genera los datos)

T_SPAN   = (0.0, 100.0)   # ms  (varias decenas -> captura transitorios + ciclos)
N_EVAL   = 2000           # dt = 0.05 ms
I0, E0   = 0.0, 0.0
SEED     = 42
NOISE    = 0.0            # plant dataset limpio (el ruido es otra dimension, luego)

OUT_PATH = Path("data/processed/control/multi_dataset.npz")

def hz(f_hz: float) -> float:
    """Hz -> ciclos por ms (porque t esta en ms)."""
    return f_hz / 1000.0

# --- Definicion de escenarios: (label, P_func, Q_func, es_test) ---------------
# Nota sobre amplitudes: el controlador puede comandar P,Q grandes/picudos en los
# transitorios (feedback linearization). Esta grilla cubre un rango moderado para
# arrancar; si el test en lazo cerrado muestra que falta el regimen de P,Q altos,
# se amplia (o se agregan trayectorias generadas por el propio lazo cerrado).
def build_scenarios():
    S = []
    ton, toff = 10.0, 90.0

    # --- Escalones (box) ---
    for amp in (0.4, 0.8, 1.2, 1.6):
        S.append((f"step_a{amp}", box_pulse(amp, ton, toff),
                  box_pulse(0.7 * amp, ton + 5, toff - 5), amp == 1.2))  # a=1.2 -> test

    # --- Senoides (amp x frecuencia) ---
    for amp in (0.4, 0.8):
        for fhz in (20, 50, 100, 150):
            es_test = (fhz == 150)  # la mas alta -> test (extrapolacion en frecuencia)
            S.append((f"sine_a{amp}_f{fhz}",
                      sine_pulse(amp, hz(fhz), ton, toff),
                      sine_pulse(0.7 * amp, hz(0.8 * fhz), ton, toff), es_test))

    # --- Multisenos (combinaciones de frecuencias) ---
    combos = [[20, 50, 90], [30, 70, 120], [15, 45, 100], [25, 60, 110], [40, 80, 130]]
    for i, c in enumerate(combos):
        es_test = (i == len(combos) - 1)  # el ultimo combo -> test
        S.append((f"multisine_{i}",
                  multisine_pulse(0.8, [hz(x) for x in c], ton, toff),
                  multisine_pulse(0.6, [hz(0.9 * x) for x in c], ton, toff), es_test))

    # --- Chirps (barrido de frecuencia) ---
    for amp in (0.4, 0.8):
        es_test = (amp == 0.8)  # uno a test
        S.append((f"chirp_a{amp}",
                  chirp_pulse(amp, hz(10), hz(150), ton, toff),
                  chirp_pulse(0.7 * amp, hz(15), hz(120), ton, toff), es_test))

    return S

# #############################################################################
# ##   FIN ZONA EDITABLE                                                     ##
# #############################################################################


def main():
    scenarios = build_scenarios()
    n = len(scenarios)
    n_test = sum(1 for *_, t in scenarios if t)
    print(f"=== Dataset multi-variante (regimen ms) — {n} escenarios "
          f"({n - n_test} train, {n_test} test) ===")

    I_all, E_all, P_all, Q_all, labels, is_test = [], [], [], [], [], []
    t_ref = None
    for label, Pf, Qf, test in scenarios:
        ds = generate_dataset(params=PARAMS, P=Pf, Q=Qf, I0=I0, E0=E0,
                              t_span=T_SPAN, n_eval=N_EVAL, noise_std=NOISE, seed=SEED)
        t_ref = ds["t"]
        I_all.append(ds["I"]); E_all.append(ds["E"])
        P_all.append(ds["P"]); Q_all.append(ds["Q"])
        labels.append(label); is_test.append(test)
        flag = "TEST " if test else "train"
        print(f"  [{flag}] {label:18}  P=[{ds['P'].min():.2f},{ds['P'].max():.2f}] "
              f"E=[{ds['E'].min():.2f},{ds['E'].max():.2f}]")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        t=t_ref,
        I=np.stack(I_all), E=np.stack(E_all),
        P=np.stack(P_all), Q=np.stack(Q_all),
        is_test=np.asarray(is_test),
        labels=np.asarray(labels),
        # metadatos
        dt=float(t_ref[1] - t_ref[0]), t_span=np.asarray(T_SPAN),
        noise_std=np.asarray(NOISE), seed=np.asarray(SEED),
        **{k: np.asarray(getattr(PARAMS, k)) for k in
           ("te", "ti", "wEE", "wEI", "wIE", "wII", "ae", "ai", "thetae", "thetai")},
    )
    n_samples = n * N_EVAL
    print(f"\nGuardado: {OUT_PATH}")
    print(f"Total: {n} trayectorias x {N_EVAL} puntos = {n_samples:,} muestras (t,I,E,P,Q)")


if __name__ == "__main__":
    main()
