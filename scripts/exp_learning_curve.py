#!/usr/bin/env python3
# =============================================================================
#  CURVA DE APRENDIZAJE — ¿cuántos datos hacen falta? (costo de datos)
# =============================================================================
#
#  Pregunta: los datasets actuales (12 trayectorias x 4000 pasos) parecen HOLGADOS
#  para identificar ~10 parametros. Como Fisher mostro que el cuello de botella es
#  la IDENTIFICABILIDAD (direccion plana de wII), no la cantidad de datos, deberiamos
#  poder identificar igual con MENOS datos -> palanca de costo barata.
#
#  Dos barridos, midiendo el error de θ̂ por parametro:
#    (A) Nº DE TRAYECTORIAS: 2, 4, 6, 8 (subconjunto de los escenarios de train).
#        -> ¿donde esta el "codo"? ¿cuantas trayectorias alcanzan?
#    (B) DENSIDAD TEMPORAL:  n_eval 4000, 2000, 1000, 500 (submuestreo del tiempo).
#        -> dt de observacion crece; atado al barrido de dt del costo de integracion.
#
#  Metrica extra: NFE de entrenamiento (nº ventanas x WINDOW x 4[RK4] x EPOCHS),
#  para ver el AHORRO de costo de cada recorte junto con la (no) perdida de calidad.
#
#  REUSA (no duplica el core): identify_subset/true_params (ident_subset),
#  strong_scenarios (noise_improve), WINDOW (train_neural_ode). El barrido de densidad
#  monkeypatchea noise_improve.N_EVAL (leido por generate() en tiempo de llamada).
#
#  ⚠️ ESTE SI ENTRENA (a diferencia de exp_compute_cost). Cada punto = una
#  identificacion completa (~1500 epochs + L-BFGS). Con los grids por defecto son
#  4+4 = 8 corridas por nivel de ruido. Ajustar los grids/NOISE_LEVELS abajo.
#
#  USO:  python -u scripts/exp_learning_curve.py
# =============================================================================

from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

import scripts.noise_improve as NI
from scripts.noise_improve import strong_scenarios
from scripts.ident_subset import identify_subset, true_params, ALL_P
from scripts.train_neural_ode import WINDOW
from scripts.ident_subset import EPOCHS, LBFGS_STEPS

# --- Configuracion de los barridos (achicar/agrandar aca) --------------------
NOISE_LEVELS = [0.0]              # agregar 0.05 / 0.10 para ver el efecto bajo ruido
N_TRAJ_GRID  = [2, 4, 6, 8]       # (A) nº de trayectorias de train
N_EVAL_GRID  = [4000, 2000, 1000, 500]   # (B) densidad temporal (pasos por trayectoria)
SEED = 0

OUT_JSON = Path("results/ident_learning_curve.json")
FIG = Path("results/figures/learning_curve.png")


def smooth_k_for(noise):
    """Ventana de suavizado adaptativa (misma logica que los scripts de ruido)."""
    if noise <= 0.0:
        return 1
    return 7 if noise < 0.10 else 11


def train_scenarios():
    """Escenarios de train (is_test=False) de strong_scenarios, en orden fijo."""
    return [s for s in strong_scenarios() if not s[3]]


def train_nfe(n_traj, n_eval):
    """NFE de entrenamiento = nº ventanas x WINDOW x 4 (RK4) x EPOCHS (proxy de costo)."""
    nwin_per_traj = (n_eval - 1) // WINDOW
    nwin = n_traj * nwin_per_traj
    return nwin * WINDOW * 4 * EPOCHS


# -----------------------------------------------------------------------------
#  (A) Barrido de nº de trayectorias (n_eval fijo en el valor original 4000).
# -----------------------------------------------------------------------------
def sweep_trajectories(noise):
    NI.N_EVAL = 4000                      # densidad original
    train = train_scenarios()
    rows = []
    for n in N_TRAJ_GRID:
        subset = train[:n]
        p_hat, err = identify_subset(subset, noise, smooth_k_for(noise), seed=SEED)
        emax = max(err.values())
        rows.append({"n_traj": n, "n_eval": 4000,
                     "err_max": emax, "err_wII": err["wII"],
                     "err": err, "nfe_train": train_nfe(n, 4000)})
        print(f"    n_traj={n:2d}  err_max={emax:6.2f}%  wII={err['wII']:6.2f}%  "
              f"NFE_train={train_nfe(n,4000):,}")
    return rows


# -----------------------------------------------------------------------------
#  (B) Barrido de densidad temporal (nº de trayectorias fijo = todas las de train).
# -----------------------------------------------------------------------------
def sweep_density(noise):
    train = train_scenarios()
    n_traj = len(train)
    rows = []
    for ne in N_EVAL_GRID:
        NI.N_EVAL = ne                    # monkeypatch: generate() lo lee al vuelo
        p_hat, err = identify_subset(train, noise, smooth_k_for(noise), seed=SEED)
        emax = max(err.values())
        rows.append({"n_traj": n_traj, "n_eval": ne,
                     "err_max": emax, "err_wII": err["wII"],
                     "err": err, "nfe_train": train_nfe(n_traj, ne)})
        print(f"    n_eval={ne:5d}  err_max={emax:6.2f}%  wII={err['wII']:6.2f}%  "
              f"NFE_train={train_nfe(n_traj,ne):,}")
    NI.N_EVAL = 4000                      # restaurar
    return rows


def main():
    print("=" * 74)
    print(" CURVA DE APRENDIZAJE — costo de datos (nº trayectorias + densidad)")
    print("=" * 74)
    print(f" params identificados: {len(ALL_P)} ({', '.join(ALL_P)})")
    print(f" WINDOW={WINDOW}  EPOCHS={EPOCHS}  LBFGS={LBFGS_STEPS}\n")

    out = {"config": {"n_traj_grid": N_TRAJ_GRID, "n_eval_grid": N_EVAL_GRID,
                      "noise_levels": NOISE_LEVELS, "window": WINDOW,
                      "epochs": EPOCHS, "true": true_params()},
           "trajectories": {}, "density": {}}

    for noise in NOISE_LEVELS:
        print(f"[A] Nº de trayectorias (σ={noise})")
        out["trajectories"][str(noise)] = sweep_trajectories(noise)
        print(f"\n[B] Densidad temporal (σ={noise})")
        out["density"][str(noise)] = sweep_density(noise)
        print()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"JSON: {OUT_JSON}")
    _plot(out)
    print(f"Figura: {FIG}")


def _plot(out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(NOISE_LEVELS)))

    for c, noise in zip(colors, NOISE_LEVELS):
        A = out["trajectories"][str(noise)]
        ax[0].plot([r["n_traj"] for r in A], [r["err_max"] for r in A],
                   "o-", color=c, label=f"máx θ̂ (σ={noise})")
        ax[0].plot([r["n_traj"] for r in A], [r["err_wII"] for r in A],
                   "s--", color=c, alpha=0.5, label=f"wII (σ={noise})")
        B = out["density"][str(noise)]
        ax[1].plot([r["n_eval"] for r in B], [r["err_max"] for r in B],
                   "o-", color=c, label=f"máx θ̂ (σ={noise})")
        ax[1].plot([r["n_eval"] for r in B], [r["err_wII"] for r in B],
                   "s--", color=c, alpha=0.5, label=f"wII (σ={noise})")

    ax[0].set_xlabel("nº de trayectorias de train"); ax[0].set_ylabel("error θ̂ (%)")
    ax[0].set_title("(A) Curva de aprendizaje — nº trayectorias")
    ax[0].grid(True, alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("nº de pasos por trayectoria (densidad)"); ax[1].set_ylabel("error θ̂ (%)")
    ax[1].set_title("(B) Densidad temporal"); ax[1].invert_xaxis()
    ax[1].grid(True, alpha=0.3); ax[1].legend(fontsize=8)

    fig.suptitle("Costo de datos: ¿estamos pasados del codo?", fontsize=11)
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
