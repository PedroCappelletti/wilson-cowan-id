#!/usr/bin/env python3
# =============================================================================
#  OED PONDERADO AL CUELLO DE BOTELLA — prueba de concepto (componentes suaves)
# =============================================================================
#
#  El Exp C (OED) mostro que la mezcla ayuda a ruido moderado pero que wII necesita
#  excitacion Q-ALTA concentrada a ruido extremo: "el OED debe ponderar hacia la
#  direccion cuello de botella, no solo repartir cobertura". Este experimento lo
#  testea: barre la PROPORCION entre
#     - trayectorias de DECORRELACION (chirp P,Q en bandas distintas -> separan
#       las direcciones acopladas, buena identificabilidad conjunta), y
#     - trayectorias de Q-ALTA (chirp de gran amplitud en Q -> excita fuerte a la
#       poblacion I -> mejora el SNR de wII, el parametro cuello de botella).
#  y mide el error de θ̂ (foco wII) bajo ruido, buscando la proporcion optima.
#
#  CLAVE (por que componentes SUAVES): ambos componentes son chirp (suaves, banda
#  limitada) -> el sistema NO es rigido con ellos -> se puede usar dt grande (4x mas
#  barato de entrenar) SIN perder precision. Si usaramos componentes conmutados
#  (aprbs/poisson) para la parte Q-alta, perderiamos ese lever (ver
#  Resultado - Identificacion NODE por familia y el limite del dt grande).
#
#  LIMITACION (documentar): los estimulos suaves (chirp) son MENOS realizables con
#  optogenetica (que es on/off). Esta prueba de concepto valida la IDEA de OED
#  ponderado con datos sinteticos; la version optogenetica realizable exige
#  estimulos conmutados y dt fino (experimento propuesto aparte).
#
#  REUSA (no duplica): identify_subset (ident_subset), chirp_pulse (wilson_cowan).
#  Multi-seed para barras de error (la identificacion tiene varianza de optimizacion).
#
#  USO:  python -u scripts/exp_oed_weighted.py
#        CALIBRATE=1 python -u scripts/exp_oed_weighted.py   (solo ETA)
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

from src.wilson_cowan import chirp_pulse
from scripts.ident_subset import identify_subset, ALL_P, EPOCHS, LBFGS_STEPS

# --- Configuracion -----------------------------------------------------------
NOISE = 0.10                 # ruido alto: donde wII es cuello de botella
N_TRAJ = 8                   # trayectorias totales por dataset (constante en el barrido)
PROPORCIONES = [0.0, 0.25, 0.5, 0.75, 1.0]   # fraccion de trayectorias Q-alta
SEEDS = [0, 1, 2]            # multi-seed -> barras de error (la ident tiene varianza)
TON, TOFF = 10.0, 190.0
hz = lambda f: f / 1000.0

# Lever de dt grande (todo suave -> vale). Ver exp_compute_cost / exp_family.
N_EVAL_OVERRIDE = 1000       # dt≈0.2 (4x menos pasos)
WINDOW_OVERRIDE = 25         # ventana ~5 unidades (igual que 100@0.05)

OUT_JSON = Path("results/ident_oed_weighted.json")
FIG = Path("results/figures/oed_weighted.png")


# -----------------------------------------------------------------------------
#  Componentes suaves (chirp). i = indice de trayectoria (varia bandas -> decorrela).
# -----------------------------------------------------------------------------
def decorr_traj(i):
    """Decorrelacion: chirp P,Q en bandas distintas, amplitud moderada."""
    P = chirp_pulse(0.8, hz(10 + 5 * i), hz(150 - 10 * i), TON, TOFF)
    Q = chirp_pulse(0.8, hz(15 + 5 * i), hz(120 - 10 * i), TON, TOFF)
    return (f"decorr_{i}", P, Q, False)


def qhigh_traj(i):
    """Q-alta: chirp de gran amplitud en Q (excita I -> SNR de wII)."""
    P = chirp_pulse(0.5, hz(12 + 5 * i), hz(140 - 10 * i), TON, TOFF)
    Q = chirp_pulse(4.0, hz(20 + 5 * i), hz(130 - 10 * i), TON, TOFF)
    return (f"qhigh_{i}", P, Q, False)


def build_dataset(rho):
    """rho = fraccion de trayectorias Q-alta (resto, decorrelacion)."""
    n_q = int(round(rho * N_TRAJ))
    n_d = N_TRAJ - n_q
    return [decorr_traj(i) for i in range(n_d)] + [qhigh_traj(i) for i in range(n_q)]


def smooth_k_for(noise):
    if noise <= 0.0:
        return 1
    return 7 if noise < 0.10 else 11


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def fmt(s):
    return f"{s:.1f} s" if s < 90 else f"{s/60:.1f} min"


def calibrate(sc, noise, sk, n_runs):
    e1, e2 = 40, 120
    _, t1 = timed(lambda: identify_subset(sc, noise, sk, epochs=e1, lbfgs=0, seed=0))
    _, t2 = timed(lambda: identify_subset(sc, noise, sk, epochs=e2, lbfgs=0, seed=0))
    b = (t2 - t1) / (e2 - e1)
    a = max(t1 - b * e1, 0.0)
    per_run = a + b * EPOCHS + b * LBFGS_STEPS * 8
    print(f"    por corrida ≈ {fmt(per_run)}  |  {n_runs} corridas ≈ {fmt(per_run * n_runs)}")
    return per_run


def main():
    import scripts.noise_improve as NI
    import scripts.ident_subset as IS
    if N_EVAL_OVERRIDE:
        NI.N_EVAL = N_EVAL_OVERRIDE
    if WINDOW_OVERRIDE:
        IS.WINDOW = WINDOW_OVERRIDE
    dt_eff = 200.0 / (NI.N_EVAL - 1)
    sk = smooth_k_for(NOISE)

    print("=" * 74)
    print(f" OED PONDERADO (componentes suaves) — σ={NOISE}, {N_TRAJ} tray, "
          f"{len(SEEDS)} seeds")
    print("=" * 74)
    print(f" proporciones Q-alta: {PROPORCIONES}")
    print(f" dt≈{dt_eff:.3f} (N_EVAL={NI.N_EVAL})  WINDOW={IS.WINDOW}  "
          f"EPOCHS={EPOCHS}  smooth_k={sk}\n")

    n_runs = len(PROPORCIONES) * len(SEEDS)
    print("[calibracion de tiempo]")
    calibrate(build_dataset(0.5), NOISE, sk, n_runs)
    if os.environ.get("CALIBRATE"):
        print("\nCALIBRATE=1 -> solo ETA.")
        return

    print("\n[corrida completa]")
    results = {}
    t_start = time.perf_counter()
    done = 0
    for rho in PROPORCIONES:
        sc = build_dataset(rho)
        errs_wII, errs_max, per_param = [], [], []
        for seed in SEEDS:
            (p, err), t = timed(lambda: identify_subset(sc, NOISE, sk, seed=seed))
            errs_wII.append(err["wII"]); errs_max.append(max(err.values()))
            per_param.append(err)
            done += 1
            eta = (time.perf_counter() - t_start) / done * (n_runs - done)
            print(f"  ρ={rho:.2f} seed={seed}  wII={err['wII']:6.2f}%  "
                  f"max={max(err.values()):6.2f}%  ({fmt(t)})  ETA {fmt(eta)}")
        results[f"{rho:.2f}"] = {
            "rho": rho,
            "n_qhigh": int(round(rho * N_TRAJ)),
            "wII_mean": float(np.mean(errs_wII)), "wII_std": float(np.std(errs_wII)),
            "max_mean": float(np.mean(errs_max)), "max_std": float(np.std(errs_max)),
            "wII_all": errs_wII, "per_param": per_param,
        }
        print(f"    -> ρ={rho:.2f}: wII {np.mean(errs_wII):.1f}±{np.std(errs_wII):.1f}%  "
              f"max {np.mean(errs_max):.1f}±{np.std(errs_max):.1f}%")

    # Optimo.
    best = min(results.values(), key=lambda r: r["wII_mean"])
    print(f"\n  ÓPTIMO (min error wII): ρ={best['rho']:.2f} "
          f"({best['n_qhigh']}/{N_TRAJ} Q-alta) -> wII {best['wII_mean']:.1f}%")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"noise": NOISE, "n_traj": N_TRAJ, "seeds": SEEDS,
         "proporciones": PROPORCIONES, "dt": dt_eff, "results": results,
         "best_rho": best["rho"]}, indent=2), encoding="utf-8")
    print(f"\nJSON: {OUT_JSON}")
    _plot(results)
    print(f"Figura: {FIG}")


def _plot(results):
    import matplotlib.pyplot as plt
    rs = sorted(results.values(), key=lambda r: r["rho"])
    rho = [r["rho"] for r in rs]
    wm = np.array([r["wII_mean"] for r in rs]); ws = np.array([r["wII_std"] for r in rs])
    mm = np.array([r["max_mean"] for r in rs]); ms = np.array([r["max_std"] for r in rs])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(rho, wm, "o-", color="#d62728", label="error wII")
    ax.fill_between(rho, wm - ws, wm + ws, color="#d62728", alpha=0.2)
    ax.plot(rho, mm, "s--", color="#1f4e79", label="error máx (10 params)")
    ax.fill_between(rho, mm - ms, mm + ms, color="#1f4e79", alpha=0.15)
    imin = int(np.argmin(wm))
    ax.axvline(rho[imin], ls=":", color="#2ca02c", label=f"óptimo ρ={rho[imin]:.2f}")
    ax.set_xlabel("proporción de trayectorias Q-alta  (ρ=0: solo decorrelación, ρ=1: solo Q-alta)")
    ax.set_ylabel("error de θ̂ (%)")
    ax.set_title(f"OED ponderado al cuello de botella (σ={NOISE}, componentes suaves)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
