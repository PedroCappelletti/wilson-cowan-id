#!/usr/bin/env python3
# =============================================================================
#  IDENTIFICACION NEURAL ODE (10 params) POR FAMILIA DE ESTIMULO
# =============================================================================
#
#  El hueco: la comparacion por familia entrenada solo se hizo con la PINN
#  (4 pesos, compare_estimulos.py) y con FIM sin entrenar (Exp C / OED). Este
#  script cierra el hueco: ENTRENA la Neural ODE (10 params, GrayBoxWC) por
#  familia de estimulo y mide el error de θ̂ (sobre todo wII). Luego CONTRASTA
#  el ranking empirico con lo que la CRB del Exp C predijo SIN entrenar.
#
#  Si el ranking empirico coincide con el de la CRB -> confirmacion empirica (NODE)
#  de la prediccion OED. Es la union de: PINN-por-familia + NODE-sobre-mezcla + OED.
#
#  REUSA (no duplica el core): identify_subset (ident_subset), los generadores de
#  estimulo de src.wilson_cowan, T_SPAN/N_EVAL/generate via ident_subset->noise_improve.
#
#  MEDICION DE TIEMPO: calibra (2 corridas cortas) y estima el ETA ANTES de lanzar
#  todo; luego cronometra cada familia y refina el ETA. Para SOLO calibrar y ver el
#  ETA sin correr el experimento completo:   CALIBRATE=1 python -u scripts/exp_family_neural_ode.py
#
#  USO:  python -u scripts/exp_family_neural_ode.py
# =============================================================================

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from src.wilson_cowan import (
    zero_input, box_pulse, chirp_pulse, aprbs_pulse, prbs_pulse,
    poisson_pulse, square_wave_pulse, theta_gamma_pulse,
)
from scripts.ident_subset import identify_subset, ALL_P, EPOCHS, LBFGS_STEPS

# --- Configuracion -----------------------------------------------------------
NOISE = 0.05                # ruido de observacion (donde el ranking de familias se nota)
N_TRAJ_FAM = 4              # trayectorias por familia (P,Q decorrelados)
SEED = 0
TON, TOFF = 10.0, 190.0
hz = lambda f: f / 1000.0

# Familias a correr (5): chirp/poisson/aprbs (buenas) + square (mala) + thetagamma.
FAMILY_LIST = ["chirp", "poisson", "aprbs", "square", "thetagamma"]

# --- Lever de costo: dt grande (WC no es rigido, ver exp_compute_cost) ---------
#  Submuestrear los datos a dt≈0.2 (N_EVAL 4000->1000) = 4x menos pasos por rollout.
#  WINDOW baja en proporcion (100->25) para mantener la ventana de multiple shooting
#  en la MISMA duracion temporal (25*0.2 = 100*0.05 = 5 unidades) -> mismo nº de
#  ventanas, solo pasos mas gruesos. None = usar el original.
N_EVAL_OVERRIDE = None      # None = dt fino original (0.05). 1000 = dt≈0.2 (lever).
WINDOW_OVERRIDE = None      # None = WINDOW original (100). 25 = proporcional a dt≈0.2.

# Salidas etiquetadas por dt para no pisar la corrida a dt grande (ya documentada).
_TAG = "dtgrande" if N_EVAL_OVERRIDE else "dtfino"
OUT_JSON = Path(f"results/ident_family_neural_ode_{_TAG}.json")
FIG = Path(f"results/figures/family_neural_ode_{_TAG}.png")

# CRB(wII) del Exp C (OED, sin entrenar) — PREDICCION a contrastar. Menor = mejor.
# Fuente: docs/subset_selection_input_design.md / vault Resultado OED.
CRB_WII_PRED = {"chirp": 1.81, "poisson": 2.25, "aprbs": 2.43, "prbs": 2.82,
                "box": 5.41, "thetagamma": 6.86, "square": 35.5}


# -----------------------------------------------------------------------------
#  Familias: cada una devuelve N_TRAJ_FAM escenarios (lbl, P, Q, is_test=False),
#  con P y Q DECORRELADOS (distintos seeds/frecuencias) para romper wIE/wII.
# -----------------------------------------------------------------------------
def _scen(name, mkP, mkQ):
    return [(f"{name}_{i}", mkP(i), mkQ(i), False) for i in range(N_TRAJ_FAM)]


def families():
    return {
        "chirp":      _scen("chirp",
                            lambda i: chirp_pulse(0.8, hz(10 + 5 * i), hz(150 - 10 * i), TON, TOFF),
                            lambda i: chirp_pulse(1.0, hz(15 + 5 * i), hz(120 - 10 * i), TON, TOFF)),
        "poisson":    _scen("poisson",
                            lambda i: poisson_pulse(1.0, hz(60), TON, TOFF, 1.0, seed=600 + i),
                            lambda i: poisson_pulse(1.2, hz(70), TON, TOFF, 1.0, seed=700 + i)),
        "aprbs":      _scen("aprbs",
                            lambda i: aprbs_pulse(1.0, TON, TOFF, 2, 8, seed=400 + i, amp_min=0.2),
                            lambda i: aprbs_pulse(1.2, TON, TOFF, 2, 8, seed=500 + i, amp_min=0.2)),
        "prbs":       _scen("prbs",
                            lambda i: prbs_pulse(1.0, TON, TOFF, 4, seed=300 + i),
                            lambda i: prbs_pulse(1.2, TON, TOFF, 5, seed=350 + i)),
        "square":     _scen("square",
                            lambda i: square_wave_pulse(1.0, hz(100 + 10 * i), TON, TOFF, 0.5),
                            lambda i: square_wave_pulse(1.2, hz(80 + 10 * i), TON, TOFF, 0.5)),
        "thetagamma": _scen("thetagamma",
                            lambda i: theta_gamma_pulse(1.0, hz(40 + 5 * i), hz(10 + i), TON, TOFF, 0.5),
                            lambda i: theta_gamma_pulse(1.2, hz(50 + 5 * i), hz(12 + i), TON, TOFF, 0.5)),
        "box":        _scen("box",
                            lambda i: box_pulse(0.6 + 0.2 * i, TON, TOFF),
                            lambda i: box_pulse(0.5 + 0.2 * i, TON + 20 * i, TOFF)),
    }


def smooth_k_for(noise):
    if noise <= 0.0:
        return 1
    return 7 if noise < 0.10 else 11


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def fmt(sec):
    return f"{sec:.1f} s" if sec < 90 else f"{sec/60:.1f} min"


# -----------------------------------------------------------------------------
#  Calibracion: time = a + b*epochs (2 puntos). Separa costo fijo (generate+setup)
#  del costo por epoca -> ETA por familia (incluye estimacion gruesa de L-BFGS).
# -----------------------------------------------------------------------------
def calibrate(fam0, noise, sk, n_fam):
    e1, e2 = 40, 120
    _, t1 = timed(lambda: identify_subset(fam0, noise, sk, epochs=e1, lbfgs=0, seed=SEED))
    _, t2 = timed(lambda: identify_subset(fam0, noise, sk, epochs=e2, lbfgs=0, seed=SEED))
    b = (t2 - t1) / (e2 - e1)                 # s por epoca
    a = max(t1 - b * e1, 0.0)                  # s fijos (generate + build_windows)
    lbfgs_est = b * LBFGS_STEPS * 8            # L-BFGS: ~8 evals tipo-epoca por step (grueso)
    per_fam = a + b * EPOCHS + lbfgs_est
    total = per_fam * n_fam
    print(f"    calibracion: fijo≈{a:.1f}s  por-epoca≈{b*1e3:.1f}ms  "
          f"L-BFGS≈{lbfgs_est:.1f}s")
    print(f"    -> por familia ≈ {fmt(per_fam)}  |  TOTAL {n_fam} familias ≈ {fmt(total)}")
    return per_fam


def main():
    # Aplicar el lever de dt grande (monkeypatch de N_EVAL y WINDOW, leidos al vuelo).
    import scripts.noise_improve as NI
    import scripts.ident_subset as IS
    if N_EVAL_OVERRIDE:
        NI.N_EVAL = N_EVAL_OVERRIDE
    if WINDOW_OVERRIDE:
        IS.WINDOW = WINDOW_OVERRIDE
    dt_eff = 200.0 / (NI.N_EVAL - 1)      # T_SPAN=(0,200)

    fams = families()
    names = [n for n in FAMILY_LIST if n in fams]
    sk = smooth_k_for(NOISE)
    print("=" * 74)
    print(f" NEURAL ODE (10 params) POR FAMILIA — σ={NOISE}, {N_TRAJ_FAM} tray/familia")
    print("=" * 74)
    print(f" familias: {', '.join(names)}")
    print(f" dt≈{dt_eff:.3f} (N_EVAL={NI.N_EVAL})  WINDOW={IS.WINDOW}  "
          f"EPOCHS={EPOCHS}  LBFGS={LBFGS_STEPS}  smooth_k={sk}\n")

    print("[calibracion de tiempo] (2 corridas cortas sobre la 1ra familia)")
    per_fam = calibrate(fams[names[0]], NOISE, sk, len(names))

    if os.environ.get("CALIBRATE"):
        print("\nCALIBRATE=1 -> solo ETA, no corro el experimento completo.")
        return

    print("\n[corrida completa]")
    results = {}
    t_start = time.perf_counter()
    for j, name in enumerate(names):
        (p, err), t = timed(lambda: identify_subset(fams[name], NOISE, sk, seed=SEED))
        results[name] = {"err": err, "err_wII": err["wII"],
                         "err_max": max(err.values()), "secs": t}
        done, left = j + 1, len(names) - (j + 1)
        eta = (time.perf_counter() - t_start) / done * left
        print(f"  {name:11} wII={err['wII']:6.2f}%  max={max(err.values()):6.2f}%  "
              f"({fmt(t)})   ETA restante: {fmt(eta)}")

    # Ranking empirico (por error de wII) vs prediccion CRB del Exp C.
    emp = sorted(names, key=lambda n: results[n]["err_wII"])
    pred = sorted([n for n in names if n in CRB_WII_PRED], key=lambda n: CRB_WII_PRED[n])
    print("\n--- Ranking wII: EMPIRICO (NODE) vs PREDICHO (CRB Exp C) ---")
    print(f"  empirico  (mejor→peor): {'  '.join(emp)}")
    print(f"  CRB Exp C (mejor→peor): {'  '.join(pred)}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"noise": NOISE, "n_traj": N_TRAJ_FAM, "results": results,
         "crb_wII_pred": CRB_WII_PRED, "rank_emp": emp, "rank_pred": pred},
        indent=2), encoding="utf-8")
    print(f"\nJSON: {OUT_JSON}")
    _plot(results, names)
    print(f"Figura: {FIG}")


def _plot(results, names):
    import matplotlib.pyplot as plt
    order = sorted(names, key=lambda n: results[n]["err_wII"])
    wII = [results[n]["err_wII"] for n in order]
    crb = [CRB_WII_PRED.get(n, np.nan) for n in order]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].bar(range(len(order)), wII, color="#1f4e79")
    ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels(order, rotation=45, ha="right")
    ax[0].set_ylabel("error de wII (%)")
    ax[0].set_title(f"Identificación NODE de wII por familia (σ={NOISE})")
    ax[0].grid(True, axis="y", alpha=0.3)

    ax[1].scatter(crb, wII, color="#d62728")
    for n, x, y in zip(order, crb, wII):
        ax[1].annotate(n, (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax[1].set_xlabel("CRB(wII) predicha — Exp C (menor = mejor)")
    ax[1].set_ylabel("error empírico de wII (%) — NODE")
    ax[1].set_title("¿Confirma la NODE la predicción de la CRB?")
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
