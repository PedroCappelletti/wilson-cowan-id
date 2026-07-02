#!/usr/bin/env python3
# =============================================================================
#  EXPERIMENTO C — Identificabilidad DEPENDIENTE DEL ESTIMULO (optimal input design)
# =============================================================================
#
#  Optimal experiment design (OED): la matriz de Fisher depende del estimulo, asi
#  que distintas entradas hacen identificables distintas direcciones del espacio de
#  parametros (Franceschini & Macchietto 2008; Ljung 1999, cap. 13). Aca, SIN
#  entrenar, se calcula la FIM+SVD (misma maquinaria que fisher_identifiability.py)
#  por CADA familia de estimulo y se reporta:
#    C1: valores singulares, numero de condicion, y el parametro/combinacion mas debil.
#    C2: que estimulo hace mas identificable a wII y a las constantes de tiempo te,ti.
#    C3: (en exp_mix_test.py) mezcla que cubre direcciones complementarias.
#
#  Reutiliza predict/make_rhs/true_theta/PNAMES/SUBSAMPLE de fisher_identifiability
#  y generate/generadores. NO duplica la maquinaria FIM.
#
#  USO:  python -u scripts/exp_input_design.py
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")   # consola Windows: permitir σ, θ, →
except Exception:
    pass

import numpy as np
import torch
from torch.func import jacfwd

from src.wilson_cowan import (
    box_pulse, square_wave_pulse, aprbs_pulse, prbs_pulse,
    theta_gamma_pulse, poisson_pulse, chirp_pulse,
)
from scripts.noise_improve import generate
from scripts.fisher_identifiability import predict, true_theta, PNAMES

torch.set_default_dtype(torch.float64)

FIG = Path("results/figures/oed_cond_por_estimulo.png")
hz = lambda f: f / 1000.0
ton, toff = 10.0, 190.0
F = False


def family_scenarios() -> dict:
    """3 trayectorias por familia (P y Q ambos excitados, con variedad)."""
    ap = lambda a, amin, s: aprbs_pulse(a, ton, toff, 2, 8, seed=s, amp_min=amin)
    sq = lambda a, f, d: square_wave_pulse(a, hz(f), ton, toff, d)
    tg = lambda a, fg, ft: theta_gamma_pulse(a, hz(fg), hz(ft), ton, toff, 0.5)
    pr = lambda a, bp, s: prbs_pulse(a, ton, toff, bp, seed=s)
    po = lambda a, rate, pw, s: poisson_pulse(a, rate, ton, toff, pw, seed=s)
    ch = lambda a, f0, f1: chirp_pulse(a, hz(f0), hz(f1), ton, toff)
    bx = lambda a: box_pulse(a, ton, toff)
    return {
        "box":        [("b1", bx(0.8), bx(0.6), F), ("b2", bx(1.2), bx(1.0), F), ("b3", bx(0.5), bx(1.5), F)],
        "square":     [("s1", sq(1.0, 100, .5), sq(0.6, 80, .5), F), ("s2", sq(0.8, 50, .5), sq(1.2, 60, .5), F), ("s3", sq(1.2, 130, .4), sq(0.5, 100, .5), F)],
        "aprbs":      [("a1", ap(1.2, 0.3, 401), ap(0.5, 0.1, 411), F), ("a2", ap(1.0, 0.2, 402), ap(3.0, 1.0, 412), F), ("a3", ap(0.8, 0.2, 403), ap(5.0, 2.0, 413), F)],
        "prbs":       [("p1", pr(1.0, 4, 421), pr(1.2, 5, 431), F), ("p2", pr(0.8, 6, 422), pr(1.0, 4, 432), F), ("p3", pr(1.3, 5, 423), pr(0.6, 6, 433), F)],
        "thetagamma": [("t1", tg(1.0, 40, 10), tg(0.8, 50, 12), F), ("t2", tg(1.2, 60, 12), tg(0.6, 40, 8), F), ("t3", tg(0.8, 50, 8), tg(1.2, 60, 10), F)],
        "poisson":    [("q1", po(1.2, 0.10, 4, 441), po(1.0, 0.10, 4, 451), F), ("q2", po(1.4, 0.12, 5, 442), po(0.8, 0.08, 5, 452), F), ("q3", po(1.0, 0.10, 4, 443), po(1.4, 0.12, 5, 453), F)],
        "chirp":      [("c1", ch(0.8, 10, 150), ch(0.6, 15, 120), F), ("c2", ch(1.0, 20, 140), ch(0.8, 10, 100), F), ("c3", ch(0.6, 5, 150), ch(1.0, 20, 130), F)],
    }


def inputs_from(scenarios):
    I, E, P, Q, _, dt = generate(scenarios, 0.0)
    inp = []
    for s in range(I.shape[0]):
        T = I.shape[1]
        x0 = torch.tensor([[I[s, 0], E[s, 0]]])
        Ps = torch.tensor(P[s]).reshape(T, 1, 1)
        Qs = torch.tensor(Q[s]).reshape(T, 1, 1)
        inp.append((x0, Ps, Qs))
    return inp, float(dt)


def fim_svd(scenarios, theta):
    """FIM+SVD para una lista de escenarios. Devuelve S, V (der.), y la cota de
    Cramer-Rao RELATIVA por parametro (crb[j] = std relativa minima de θ̂_j).

    La CRB es la metrica CORRECTA de identificabilidad marginal: (FIM^-1)_jj tiene
    en cuenta las CORRELACIONES con los demas parametros (a diferencia de ||columna||,
    que mide sensibilidad marginal e ignora el acople). Menor CRB = mas identificable.
    Es la base de los criterios A/D-optimal de optimal experiment design.
    """
    inp, dt = inputs_from(scenarios)
    J = jacfwd(lambda th: predict(th, inp, dt))(theta)     # (N_out, 10)
    Jr = (J * theta.unsqueeze(0)).numpy()                  # sensibilidad relativa
    U, S, Vt = np.linalg.svd(Jr, full_matrices=False)
    fim = Jr.T @ Jr                                        # FIM relativa (x σ²)
    crb = np.sqrt(np.diag(np.linalg.inv(fim)))             # std relativa minima por param
    return S, Vt, crb


def main():
    theta = true_theta()
    fams = family_scenarios()
    print(f"=== Exp C — FIM+SVD por familia de estimulo ({len(fams)} familias) ===\n")

    res = {}
    for fam, sc in fams.items():
        S, V, crb = fim_svd(sc, theta)
        cond = S[0] / S[-1]
        weak = V[-1]
        j = int(np.argmax(np.abs(weak)))
        res[fam] = {"S": S, "V": V, "crb": crb, "cond": cond,
                    "weak_param": PNAMES[j], "weak_w": weak}
        print(f"[{fam:11}] cond(σ1/σ10)={cond:9.2e} | dir. mas debil dominada por "
              f"{PNAMES[j]} ({weak[j]:+.2f})")

    # --- C2: identificabilidad marginal (CRB) de wII y las constantes de tiempo ---
    #     CRB baja = ese estimulo identifica MEJOR ese parametro (menor incertidumbre).
    def rank_for(param):
        idx = PNAMES.index(param)
        order = sorted(res.items(), key=lambda kv: kv[1]["crb"][idx])   # menor primero
        return [(f, r["crb"][idx]) for f, r in order]

    print("\n--- C2: cota de Cramér-Rao relativa por parámetro (MENOR = más identificable) ---")
    for prm in ("wII", "te", "ti"):
        rk = rank_for(prm)
        print(f"  {prm:4}: " + "  ".join(f"{f}={v:.2e}" for f, v in rk))
        print(f"        -> mejor estímulo para {prm}: {rk[0][0]}  (CRB más baja)")

    _plot(res)
    print(f"\nFigura: {FIG}")
    return res


def _plot(res):
    import matplotlib.pyplot as plt
    fams = list(res.keys())
    conds = [res[f]["cond"] for f in fams]
    # CRB relativa (identificabilidad marginal) de wII, te, ti por familia
    iw, ie, it = PNAMES.index("wII"), PNAMES.index("te"), PNAMES.index("ti")
    cw = [res[f]["crb"][iw] for f in fams]
    ce = [res[f]["crb"][ie] for f in fams]
    ct = [res[f]["crb"][it] for f in fams]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].bar(fams, conds, color="#1f4e79")
    ax[0].set_yscale("log"); ax[0].set_ylabel("número de condición σ1/σ10")
    ax[0].set_title("Condicionamiento de la FIM por estímulo\n(menor = más identificable en conjunto)")
    ax[0].tick_params(axis="x", rotation=45); ax[0].grid(True, axis="y", alpha=0.3)

    x = np.arange(len(fams)); w = 0.27
    ax[1].bar(x - w, cw, w, label="wII", color="#d62728")
    ax[1].bar(x, ce, w, label="te", color="#1b7f4b")
    ax[1].bar(x + w, ct, w, label="ti", color="#b06a00")
    ax[1].set_yscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels(fams, rotation=45, ha="right")
    ax[1].set_ylabel("cota de Cramér-Rao relativa (std mín.)")
    ax[1].set_title("Identificabilidad marginal de wII, te, ti por estímulo\n(MENOR = ese estímulo lo identifica mejor)")
    ax[1].legend(); ax[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
