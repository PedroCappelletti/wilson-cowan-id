#!/usr/bin/env python3
# =============================================================================
#  EXPERIMENTO C3 — Mezcla de estimulos que cubre direcciones complementarias
# =============================================================================
#
#  Optimal input design en la practica: ningun estimulo unico excita bien TODAS
#  las direcciones del espacio de parametros. La propuesta (guiada por C1/C2,
#  exp_input_design.py) es MEZCLAR familias que exciten direcciones complementarias
#  -> maximizar la identificabilidad conjunta. Se testea bajo ruido comparando el
#  error de θ̂ de la mezcla contra el baseline strong_scenarios.
#
#  La mezcla se arma tomando, de cada familia (family_scenarios), las trayectorias
#  que mejor cubren: wII (Q grande), constantes de tiempo (chirp/broadband) y el
#  regimen de aplicacion (theta-gamma). Ajustable segun el ranking de C1.
#
#  USO:  python -u scripts/exp_mix_test.py
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

from src.wilson_cowan import (
    aprbs_pulse, square_wave_pulse, prbs_pulse, theta_gamma_pulse, chirp_pulse, zero_input,
)
from scripts.ident_subset import identify_subset, ALL_P
from scripts.noise_improve import strong_scenarios
from scripts.noise_final import smooth_k_for

NOISE_LEVELS = [0.05, 0.10]
OUT_JSON = Path("results/exp_mix_test.json")
FIG = Path("results/figures/expC3_mezcla.png")

hz = lambda f: f / 1000.0
ton, toff = 10.0, 190.0
F, T = False, True


def complementary_mix():
    """Mezcla complementaria (ajustada segun C1/C2):
       - Q GRANDE (aprbs/square): excita I -> identifica wII.
       - chirp broadband: barre frecuencias -> constantes de tiempo te,ti.
       - theta-gamma: regimen de aplicacion.
       - prbs/aprbs balanceados: cobertura general.
    """
    ap = lambda a, amin, s: aprbs_pulse(a, ton, toff, 2, 8, seed=s, amp_min=amin)
    sq = lambda a, f, d: square_wave_pulse(a, hz(f), ton, toff, d)
    return [
        # --- direccion wII: Q grande ---
        ("mix_Qgde_a", ap(1.0, 0.2, 501), ap(5.0, 2.0, 511), F),
        ("mix_Qgde_b", zero_input,        ap(5.0, 2.0, 512), F),
        ("mix_Qgde_sq", sq(1.0, 100, .5), sq(4.0, 80, .5),  T),
        # --- constantes de tiempo: chirp broadband en P y Q ---
        ("mix_chirp_a", chirp_pulse(1.0, hz(10), hz(150), ton, toff),
                        chirp_pulse(0.8, hz(15), hz(120), ton, toff), F),
        ("mix_chirp_b", chirp_pulse(0.8, hz(20), hz(140), ton, toff),
                        chirp_pulse(1.2, hz(10), hz(100), ton, toff), T),
        # --- regimen de aplicacion: theta-gamma ---
        ("mix_tg_a", theta_gamma_pulse(1.0, hz(40), hz(10), ton, toff, .5),
                     theta_gamma_pulse(1.2, hz(50), hz(12), ton, toff, .5), F),
        # --- cobertura general ---
        ("mix_gen_a", ap(1.2, 0.3, 502), ap(1.0, 0.3, 513), F),
        ("mix_gen_b", prbs_pulse(1.0, ton, toff, 4, seed=503),
                      prbs_pulse(1.2, ton, toff, 5, seed=514), T),
    ]


def main():
    baseline = strong_scenarios()
    mix = complementary_mix()
    print(f"=== C3 — mezcla complementaria vs baseline (strong_scenarios) ===")
    print(f"    baseline: {len(baseline)} tray.  |  mezcla: {len(mix)} tray.\n")

    rows = []
    for noise in NOISE_LEVELS:
        sk = smooth_k_for(noise)
        _, err_base = identify_subset(baseline, noise, sk)
        _, err_mix = identify_subset(mix, noise, sk)
        mb, mm = max(err_base.values()), max(err_mix.values())
        rows.append({"noise": noise, "err_base": err_base, "err_mix": err_mix,
                     "max_base": mb, "max_mix": mm})
        print(f"σ={noise:.2f} (k={sk}) | máx: baseline={mb:6.2f}%  mezcla={mm:6.2f}%")
        for k in ("wII", "ti", "ai", "ae", "te"):
            print(f"    {k:5} baseline={err_base[k]:6.2f}%   mezcla={err_mix[k]:6.2f}%")

    _plot(rows)
    OUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nFigura: {FIG}\nResultados: {OUT_JSON}")


def _plot(rows):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(rows), figsize=(6 * len(rows), 4.6), squeeze=False)
    for c, r in enumerate(rows):
        a = ax[0][c]
        x = np.arange(len(ALL_P)); w = 0.4
        a.bar(x - w / 2, [r["err_base"][k] for k in ALL_P], w, label="baseline", color="#d62728")
        a.bar(x + w / 2, [r["err_mix"][k] for k in ALL_P], w, label="mezcla", color="#1f4e79")
        a.set_xticks(x); a.set_xticklabels(ALL_P, rotation=45, ha="right", fontsize=8)
        a.set_ylabel("error θ̂ (%)"); a.set_title(f"σ={r['noise']:.2f}")
        a.grid(True, axis="y", alpha=0.3); a.legend()
    fig.suptitle("C3: mezcla complementaria vs baseline (error por parámetro)")
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
