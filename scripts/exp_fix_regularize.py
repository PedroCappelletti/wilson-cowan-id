#!/usr/bin/env python3
# =============================================================================
#  EXPERIMENTO A — Fijar / regularizar el parametro mal condicionado (wII)
# =============================================================================
#
#  El diagnostico FIM+SVD (fisher_identifiability.py) y el barrido de ruido
#  (noise_full_sweep.py) mostraron que wII domina la direccion singular mas debil
#  y se rompe primero bajo ruido (41% a σ=0.10). Aca se prueba si SACARLO del
#  ajuste ayuda al resto — dos formas establecidas:
#
#    A1. FIJAR wII en su valor verdadero y re-identificar los otros 9
#        (parameter subset selection; Chu & Hahn 2007). ¿Baja el error de ti,ai,ae?
#    A2. REGULARIZAR wII hacia un prior con penalizacion L2 de fuerza λ
#        (estimacion MAP / ridge; profile-likelihood suave). λ→∞ ≈ fijar (A1).
#        Se grafica error-del-resto vs λ y se reporta el λ optimo.
#
#  Reutiliza identify_subset (ident_subset.py), strong_scenarios (noise_improve),
#  smooth_k_for (noise_final) y el baseline "10 libres" (results/noise_full_sweep.json).
#
#  USO:  python -u scripts/exp_fix_regularize.py
# =============================================================================

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from scripts.ident_subset import identify_subset, ALL_P, WEIGHTS, PHYS
from scripts.noise_improve import strong_scenarios
from scripts.noise_final import smooth_k_for

NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10]
SIGMA_A2 = 0.10                                   # nivel donde wII se rompe
LAMBDAS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1, 1.0]

BASELINE_JSON = Path("results/noise_full_sweep.json")   # err10 (10 libres)
OUT_JSON = Path("results/exp_fix_regularize.json")
FIG_A1 = Path("results/figures/expA1_fix_wII.png")
FIG_A2 = Path("results/figures/expA2_reg_lambda.png")


def load_baseline():
    """err10 (10 libres) por nivel de ruido, desde noise_full_sweep.json."""
    d = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    return {lvl["noise"]: lvl["err10"] for lvl in d["levels"]}


def rest_error(err, exclude=("wII",)):
    """Error del RESTO (sin los excluidos): (media, max) sobre los demas params."""
    vals = [v for k, v in err.items() if k not in exclude]
    return float(np.mean(vals)), float(np.max(vals))


def main():
    scen = strong_scenarios()
    base = load_baseline()

    # ----------------------------- A1 -------------------------------------
    print("=== A1 — fijar wII vs 10 libres (error por parametro) ===")
    a1 = []
    for noise in NOISE_LEVELS:
        sk = smooth_k_for(noise)
        _, err_fix = identify_subset(scen, noise, sk, fix={"wII"})
        err_free = base.get(noise, {})
        mean_fix, max_fix = rest_error(err_fix)
        mean_free, max_free = rest_error(err_free) if err_free else (float("nan"), float("nan"))
        a1.append({"noise": noise, "err_fix": err_fix, "err_free": err_free,
                   "rest_fix": [mean_fix, max_fix], "rest_free": [mean_free, max_free]})
        print(f"\nσ={noise:.2f} (k={sk}) — resto: fijo(máx)={max_fix:.2f}%  libre(máx)={max_free:.2f}%")
        for k in ("ti", "ai", "ae", "te", "thetai", "thetae", "wEE", "wEI", "wIE"):
            ef = err_fix.get(k, float("nan")); el = err_free.get(k, float("nan"))
            flag = " <=" if ef < el - 1e-9 else ""
            print(f"    {k:7} fijo-wII={ef:6.2f}%   10libres={el:6.2f}%{flag}")

    # ----------------------------- A2 -------------------------------------
    print(f"\n=== A2 — regularizar wII (σ={SIGMA_A2}); error-del-resto vs λ ===")
    sk = smooth_k_for(SIGMA_A2)
    a2 = []
    for lam in LAMBDAS:
        _, err = identify_subset(scen, SIGMA_A2, sk, reg_params={"wII"}, reg_lambda=lam)
        mean_r, max_r = rest_error(err)
        a2.append({"lambda": lam, "err": err, "wII": err["wII"],
                   "rest_mean": mean_r, "rest_max": max_r})
        print(f"  λ={lam:8.1e} | wII err={err['wII']:6.2f}% | resto media={mean_r:5.2f}% máx={max_r:6.2f}%")
    best = min(a2, key=lambda r: r["rest_max"])
    print(f"\n  λ óptimo (min resto-máx): λ={best['lambda']:.1e} -> resto máx={best['rest_max']:.2f}% "
          f"(wII err={best['wII']:.2f}%)")

    _plot_a1(a1)
    _plot_a2(a2, best)
    OUT_JSON.write_text(json.dumps({"A1": a1, "A2": a2, "best_lambda": best["lambda"]},
                                   indent=2), encoding="utf-8")
    print(f"\nFiguras: {FIG_A1} | {FIG_A2}\nResultados: {OUT_JSON}")


def _plot_a1(a1):
    import matplotlib.pyplot as plt
    ns = [r["noise"] for r in a1]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(ns, [r["rest_free"][1] for r in a1], "s--", color="#d62728", label="10 libres (resto máx)")
    ax.plot(ns, [r["rest_fix"][1] for r in a1], "o-", color="#1f4e79", label="wII fijo (resto máx)")
    ax.set_xlabel("noise_std σ"); ax.set_ylabel("error del resto — máx (%)")
    ax.set_title("A1: fijar wII vs identificar los 10 libres")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); FIG_A1.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_A1, dpi=120); plt.close(fig)


def _plot_a2(a2, best):
    import matplotlib.pyplot as plt
    lams = [max(r["lambda"], 1e-5) for r in a2]   # 0 -> 1e-5 para el eje log
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(lams, [r["rest_max"] for r in a2], "o-", color="#1f4e79", label="resto máx")
    ax.plot(lams, [r["rest_mean"] for r in a2], "s-", color="#1b7f4b", label="resto media")
    ax.plot(lams, [r["wII"] for r in a2], "^:", color="#d62728", label="error wII")
    ax.axvline(max(best["lambda"], 1e-5), ls=":", color="#888", label=f"λ óptimo={best['lambda']:.0e}")
    ax.set_xscale("log"); ax.set_xlabel("λ (fuerza de regularización de wII)")
    ax.set_ylabel("error θ̂ (%)"); ax.set_title(f"A2: regularizar wII (σ={SIGMA_A2})")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_A2, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
