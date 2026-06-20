#!/usr/bin/env python3
# =============================================================================
#  REFINAMIENTO σ=0.10 — bajar el residual (sobre todo wEI) variando el estímulo
# =============================================================================
#  Prueba a σ=0.10: suavizado k=7 vs k=11, y set de escenarios "strong" vs "XL"
#  (más trayectorias, más variantes Q-grande). Reporta error de θ̂.
#  USO: python -u scripts/noise_refine.py
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.wilson_cowan import (
    aprbs_pulse, square_wave_pulse, prbs_pulse, theta_gamma_pulse, chirp_pulse, zero_input,
)
from scripts.noise_improve import strong_scenarios, identify, ton, toff, hz, W_TRUE


def strong_xl():
    """strong_scenarios + más trayectorias diversas con Q grande (≈20 en total)."""
    ap = lambda a, amin, s: aprbs_pulse(a, ton, toff, 2, 8, seed=s, amp_min=amin)
    sq = lambda a, f, d: square_wave_pulse(a, hz(f), ton, toff, d)
    extra = [
        ("Qgde_c",      ap(0.6, 0.1, 401), ap(4.5, 1.8, 411), False),
        ("Qgde_d",      ap(1.0, 0.2, 402), ap(3.5, 1.2, 412), True),
        ("P0_Qgde2",    zero_input,        ap(4.0, 1.5, 413), False),
        ("Qgde_prbs",   prbs_pulse(0.8, ton, toff, 4, seed=403),
                        prbs_pulse(4.0, ton, toff, 5, seed=414), False),
        ("Qgde_sq2",    sq(0.8, 130, 0.4), sq(5.0, 100, 0.5), True),
        ("bal2",        ap(1.1, 0.3, 404), ap(1.5, 0.3, 415), False),
        ("E_chirp_Qgde",chirp_pulse(1.0, hz(20), hz(120), ton, toff), ap(4.0, 1.5, 416), False),
        ("Qgde_tg",     theta_gamma_pulse(0.8, hz(40), hz(10), ton, toff, 0.5),
                        theta_gamma_pulse(4.0, hz(50), hz(12), ton, toff, 0.5), False),
    ]
    return strong_scenarios() + extra


def main():
    NOISE = 0.10
    configs = [
        ("strong  k=7  (Iter 14)", strong_scenarios(), 7),
        ("strong  k=11",           strong_scenarios(), 11),
        ("XL      k=7",            strong_xl(),        7),
        ("XL      k=11",           strong_xl(),        11),
    ]
    print(f"=== Refinamiento a σ={NOISE} (error de θ̂, menor = mejor) ===")
    for name, sc, sk in configs:
        torch.manual_seed(0); np.random.seed(0)
        w, err = identify(sc, NOISE, sk)
        print(f"  {name:22} ({len(sc)} tray) | max={max(err.values()):5.1f}% | "
              + " ".join(f"{k}={err[k]:5.1f}%" for k in W_TRUE))


if __name__ == "__main__":
    main()
