#!/usr/bin/env python3
# =============================================================================
#  DATASET MULTI-VARIANTE para el Neural ODE (planta controlable) — régimen ms
# =============================================================================
#
#  Genera trayectorias de Wilson-Cowan bajo MUCHOS tipos de estimulo tipo PULSO
#  (escalon, onda cuadrada/tren de pulsos, APRBS, PRBS, theta-gamma, Poisson) +
#  chirp, variando amplitud y frecuencia. Sin senoides puras (decision del
#  proyecto: estimulos realizables on/off). Con esa diversidad el modelo de estados
#  f_θ(x,P,Q) aprende a RESPONDER a entradas variadas -> condicion para el control.
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
    WilsonCowanParams, box_pulse, chirp_pulse,
    aprbs_pulse, theta_gamma_pulse, square_wave_pulse, prbs_pulse, poisson_pulse,
)
from src.data import generate_dataset
from src.utils import provenance_str, set_seed

# #############################################################################
# ##   ZONA EDITABLE                                                         ##
# #############################################################################

PARAMS = WilsonCowanParams()    # parametros verdaderos (genera los datos)

T_SPAN   = (0.0, 200.0)   # ms  (mas largo -> entran mas ciclos, incl. envolvente theta)
N_EVAL   = 4000           # dt = 0.05 ms
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
    """Escenarios con estimulos tipo PULSO (la libreria nueva). Sin senoides puras;
    se conserva el chirp por su cobertura espectral. P y Q decorrelacionados
    (distinta amplitud/timing/semilla). Cada familia reserva 1 escenario como test."""
    S = []
    ton, toff = 10.0, 190.0

    # --- Escalon (box): baseline ---
    for amp in (0.4, 0.8, 1.2):
        S.append((f"box_a{amp}", box_pulse(amp, ton, toff),
                  box_pulse(0.7 * amp, ton + 5, toff - 5), amp == 1.2))

    # --- Onda cuadrada / tren de pulsos (DBS, optogenetica): amp x frecuencia ---
    for amp in (0.6, 1.0):
        for fhz in (50, 100, 130):
            es_test = (fhz == 130 and amp == 1.0)
            S.append((f"square_a{amp}_f{fhz}",
                      square_wave_pulse(amp, hz(fhz), ton, toff, 0.4),
                      square_wave_pulse(0.7 * amp, hz(0.8 * fhz), ton, toff, 0.5), es_test))

    # --- APRBS (amplitud x frecuencia; el mas rico para no lineal) ---
    for i, (amp, dmin, dmax, s) in enumerate([(1.0, 2, 8, 71), (1.4, 1.5, 6, 72), (0.8, 3, 10, 73)]):
        S.append((f"aprbs_{i}",
                  aprbs_pulse(amp, ton, toff, dmin, dmax, seed=s, amp_min=0.2 * amp),
                  aprbs_pulse(0.8 * amp, ton, toff, dmin * 1.3, dmax * 1.2, seed=s + 10, amp_min=0.1 * amp),
                  i == 2))

    # --- PRBS (banda ancha, binario) ---
    for i, (amp, bp, s) in enumerate([(1.0, 4, 81), (1.3, 6, 82)]):
        S.append((f"prbs_{i}", prbs_pulse(amp, ton, toff, bp, seed=s),
                  prbs_pulse(0.8 * amp, ton, toff, bp * 1.2, seed=s + 10), i == 1))

    # --- Theta-gamma (regimen del proyecto): rafagas gamma bajo envolvente theta ---
    for i, (amp, fg, ft) in enumerate([(1.0, 40, 10), (1.2, 60, 12), (0.8, 50, 8)]):
        S.append((f"thetagamma_{i}",
                  theta_gamma_pulse(amp, hz(fg), hz(ft), ton, toff, 0.5),
                  theta_gamma_pulse(0.7 * amp, hz(0.9 * fg), hz(ft), ton, toff, 0.5), i == 2))

    # --- Tren de Poisson (naturalista). Pulsos ANCHOS (~4-6 ms): los angostos se
    #     promedian a ~0 en un modelo de tasas y no excitan (verificado). ---
    for i, (amp, rate, pw, s) in enumerate([(1.2, 0.10, 4.0, 91), (1.4, 0.12, 5.0, 92)]):
        S.append((f"poisson_{i}", poisson_pulse(amp, rate, ton, toff, pw, seed=s),
                  poisson_pulse(0.8 * amp, rate * 0.8, ton, toff, pw, seed=s + 10), i == 1))

    # --- Chirp (unica senoidal conservada: cobertura espectral) ---
    S.append(("chirp", chirp_pulse(0.8, hz(10), hz(150), ton, toff),
              chirp_pulse(0.6, hz(15), hz(120), ton, toff), True))

    return S

# #############################################################################
# ##   FIN ZONA EDITABLE                                                     ##
# #############################################################################


def main():
    set_seed(SEED)
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
        # Con que se genero: sin esto, un .npz pisado no se puede identificar.
        provenance=np.asarray(provenance_str(
            __file__, seed=SEED, noise=NOISE, n_eval=N_EVAL,
            t_span=list(T_SPAN), n_escenarios=n,
        )),
        **{k: np.asarray(getattr(PARAMS, k)) for k in
           ("te", "ti", "wEE", "wEI", "wIE", "wII", "ae", "ai", "thetae", "thetai")},
    )
    n_samples = n * N_EVAL
    print(f"\nGuardado: {OUT_PATH}")
    print(f"Total: {n} trayectorias x {N_EVAL} puntos = {n_samples:,} muestras (t,I,E,P,Q)")


if __name__ == "__main__":
    main()
