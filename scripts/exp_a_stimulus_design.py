#!/usr/bin/env python3
# =============================================================================
#  A — DISENAR EL ESTIMULO QUE ROMPE LA DEGENERACION
# =============================================================================
#
#  De donde sale. F4b midio que el 67% de la fisica que le falta al modelo es
#  IMITABLE moviendo los 10 parametros. Esa degeneracion es la causa de todo lo
#  demas: hace que el reparto entre θ y g no sea unico, y por eso la correccion
#  aprende algo del tamano correcto pero de forma equivocada.
#
#  La observacion que abre la puerta: esa fraccion NO es una constante del
#  problema. Es geometria de la TRAYECTORIA, y la trayectoria la fija el
#  estimulo. Medido sobre los estimulos de libreria, va del 71% (APRBS) al 94%
#  (pulso casi constante). O sea que se puede ELEGIR.
#
#  Lo que hace este script: buscar el estimulo que MAXIMIZA la parte del
#  mismatch que ningun cambio de parametros puede imitar:
#
#      Delta f  =  S·δθ*  +  residuo        ->  maximizar ‖residuo‖
#
#  Por que maximizar el residuo y no minimizar la fraccion. Minimizar la
#  fraccion se puede hacer trampeando: basta con volver el Delta f chiquito y la
#  fraccion baja sin que haya nada que aprender. Maximizar el residuo en valor
#  ABSOLUTO pide las dos cosas a la vez: que el hueco sea grande Y que no se
#  parezca a mover parametros.
#
#  POTENCIA IGUALADA. Un estimulo mas fuerte excita mas no linealidad y sube el
#  residuo por la via facil. Para que la comparacion mida FORMA y no amplitud,
#  todos los candidatos se renormalizan a la misma amplitud media que el mejor
#  estimulo de libreria (APRBS). Asi la mejora, si aparece, es del diseno.
#
#  USO:  python scripts/exp_a_stimulus_design.py
# =============================================================================

#  DOS OBJETIVOS, y hay que correr los dos porque miden cosas distintas:
#
#    --objetivo residuo  : maximizar ‖residuo‖ (el default). Puede lograrse
#        agrandando TODO el hueco, aunque la fraccion imitable no baje. Es util
#        —da mas senal para que g aprenda— pero NO demuestra que se rompio la
#        degeneracion.
#    --objetivo fraccion : minimizar la fraccion imitable, exigiendo que el
#        |Delta f| no caiga por debajo del de la referencia. ESE es el test
#        directo de "romper la degeneracion": el mismatch tiene que dejar de
#        parecerse a un cambio de parametros sin volverse mas chico.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import numpy as np
import torch

torch.set_num_threads(2)

from scipy.special import expit

from src.wilson_cowan import WilsonCowan, WilsonCowanParams, default_uncertainty, aprbs_pulse
from src.neural_ode import GrayBoxWC
from src.neural_ode.graybox_train import projection_operator, projected_fraction, ALL_P

OUT_BASE = Path("results/uncertainty/a_stimulus_design")
FIG_BASE = Path("results/figures/a_stimulus_design")

# --- Parametrizacion del estimulo -------------------------------------------
T_ON, T_OFF = 10.0, 190.0
N_SEG = 24                       # tramos de ~7.5 ms (realizable con optogenetica)
T_SPAN = (0.0, 200.0)
N_EVAL = 2000                    # dt = 0.1 ms durante la busqueda
N_SUB = 2
EPS = 1.0                        # nivel de mismatch (el nominal del roadmap)
N_PUNTOS = 2500                  # puntos para la proyeccion

# --- Busqueda ---------------------------------------------------------------
N_ITER = 60
POB = 8                          # (mu + lambda) chico: cada evaluacion es cara
SIGMA0 = 0.35
SEED = 0


def escalon(vals: np.ndarray):
    """Senal constante a tramos, gateada a [T_ON, T_OFF). Funcion PURA de t."""
    bordes = np.linspace(T_ON, T_OFF, len(vals) + 1)

    def f(t: float) -> float:
        if T_ON <= t < T_OFF:
            i = int(np.searchsorted(bordes, t, side="right")) - 1
            return float(vals[min(max(i, 0), len(vals) - 1)])
        return 0.0
    return f


def normaliza(v: np.ndarray, media_obj: float, amp_max: float) -> np.ndarray:
    """Lleva el vector a [0, amp_max] y despues ajusta su media al objetivo.
    Asi todos los candidatos gastan la misma 'energia' y solo compiten en FORMA."""
    v = np.clip(v, 0.0, amp_max)
    m = v.mean()
    if m > 1e-9:
        v = v * (media_obj / m)
    return np.clip(v, 0.0, amp_max)


# --- Evaluacion de un candidato ---------------------------------------------
_P0 = WilsonCowanParams()
_TRUE = {k: getattr(_P0, k) for k in ALL_P}
_MODEL = GrayBoxWC(_TRUE, {k: _TRUE[k] for k in ("wEE", "wEI", "wIE", "wII")},
                   learnable_weights=True, learnable_params=True)


def campo_wc(I, E, P, Q):
    p = _P0
    u_i = p.wIE * E - p.wII * I + Q - p.thetai
    u_e = p.wEE * E - p.wEI * I + P - p.thetae
    return ((1 / p.ti) * (-I + expit(p.ai * u_i) - p.ki),
            (1 / p.te) * (-E + expit(p.ae * u_e) - p.ke))


def evalua(Pf, Qf, n_eval=N_EVAL, n_sub=N_SUB, seed=0) -> dict:
    """Simula con la planta perturbada y devuelve la descomposicion del Delta f."""
    pert = default_uncertainty(EPS)
    m = WilsonCowan(params=_P0, P=Pf, Q=Qf, perturbation=pert)
    t_eval = np.linspace(T_SPAN[0], T_SPAN[1], n_eval)
    sol = m.simulate(I0=0.0, E0=0.0, t_span=T_SPAN, t_eval=t_eval, n_sub=n_sub)

    I, E, Pc, Qc = sol["I"], sol["E"], sol["P"], sol["Q"]
    extra = sol["extra"]
    nomI, nomE = campo_wc(I, E, Pc, Qc)
    dfI = np.zeros(len(t_eval)); dfE = np.zeros(len(t_eval))
    for k in range(len(t_eval)):
        s = np.concatenate(([I[k], E[k]], extra[k]))
        d = m.rhs_aug(float(t_eval[k]), s)
        dfI[k] = d[0] - nomI[k]; dfE[k] = d[1] - nomE[k]

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(t_eval), min(N_PUNTOS, len(t_eval)), replace=False)
    X = torch.tensor(np.stack([I[idx], E[idx]], 1), dtype=torch.float32)
    Pt = torch.tensor(Pc[idx].reshape(-1, 1), dtype=torch.float32)
    Qt = torch.tensor(Qc[idx].reshape(-1, 1), dtype=torch.float32)
    DF = torch.tensor(np.stack([dfI[idx], dfE[idx]], 1), dtype=torch.float32)

    S = _MODEL.backbone_sensitivities(X, Pt, Qt)
    A, Sf = projection_operator(S)
    _, frac = projected_fraction(DF, A, Sf)
    c = A @ DF.reshape(-1)
    res = DF - (Sf @ c).reshape(-1, 2)

    return {
        "residuo": float(res.pow(2).mean().sqrt()),   # <- lo que se maximiza
        "df_rms": float(DF.pow(2).mean().sqrt()),
        "frac_imitable": float(frac),
        "E_max": float(E.max()), "I_max": float(I.max()),
        "P_media": float(Pc[Pc > 0].mean()) if (Pc > 0).any() else 0.0,
    }


def puntaje(r: dict, objetivo: str, df_min: float) -> float:
    """Funcion a MAXIMIZAR, segun el objetivo elegido."""
    if objetivo == "residuo":
        return r["residuo"]
    # objetivo == "fraccion": minimizar la fraccion imitable, pero sin permitir
    # que el hueco se achique (si no, la fraccion baja por la via facil).
    if r["df_rms"] < df_min:
        # penalizacion proporcional a cuanto se achico: mantiene el gradiente
        # de busqueda apuntando hacia adentro de la region valida.
        return -1.0 - (df_min - r["df_rms"]) / max(df_min, 1e-9)
    return -r["frac_imitable"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objetivo", choices=["residuo", "fraccion"], default="residuo")
    args = ap.parse_args()
    objetivo = args.objetivo
    rng = np.random.default_rng(SEED)

    # --- Referencia: el mejor estimulo de libreria (APRBS) -------------------
    Pf_ref = aprbs_pulse(1.4, T_ON, T_OFF, 1.5, 6, seed=72, amp_min=0.28)
    Qf_ref = aprbs_pulse(1.12, T_ON, T_OFF, 1.95, 7.2, seed=82, amp_min=0.14)
    ref = evalua(Pf_ref, Qf_ref)
    print("=== A · disenar el estimulo que rompe la degeneracion ===\n")
    print(f"  Referencia (APRBS de libreria):")
    print(f"    |Delta f| = {ref['df_rms']:.5f}   imitable = {100*ref['frac_imitable']:.1f}%"
          f"   RESIDUO = {ref['residuo']:.5f}")

    # Amplitud media de la referencia: el presupuesto que todos deben respetar.
    ts = np.linspace(T_ON, T_OFF, 400)
    media_P = float(np.mean([Pf_ref(t) for t in ts]))
    media_Q = float(np.mean([Qf_ref(t) for t in ts]))
    AMP_MAX = 1.4
    print(f"    amplitud media: P={media_P:.3f}  Q={media_Q:.3f}  "
          f"(presupuesto igualado para todos los candidatos)\n")

    def arma(v):
        p = normaliza(v[:N_SEG], media_P, AMP_MAX)
        q = normaliza(v[N_SEG:], media_Q, AMP_MAX)
        return escalon(p), escalon(q), p, q

    # --- Poblacion inicial: aleatoria + la referencia muestreada -------------
    def muestrea_ref():
        b = np.linspace(T_ON, T_OFF, N_SEG + 1)
        return np.array([np.mean([Pf_ref(t) for t in np.linspace(b[i], b[i+1], 5)])
                         for i in range(N_SEG)] +
                        [np.mean([Qf_ref(t) for t in np.linspace(b[i], b[i+1], 5)])
                         for i in range(N_SEG)])

    df_min = ref["df_rms"]
    mejor_v = muestrea_ref()
    Pf, Qf, _, _ = arma(mejor_v)
    mejor = evalua(Pf, Qf)
    mejor_p = puntaje(mejor, objetivo, df_min)
    print(f"  Objetivo: {objetivo.upper()}"
          + (f" (con |Delta f| >= {df_min:.5f})" if objetivo == "fraccion" else ""))
    print(f"  Punto de partida (APRBS discretizado en {N_SEG} tramos): "
          f"residuo = {mejor['residuo']:.5f}, imitable = "
          f"{100*mejor['frac_imitable']:.1f}%")
    print(f"\n  {'iter':>5} {'residuo':>10} {'|Df|':>10} {'imitable':>10} {'E_max':>8}")

    hist = []
    sigma = SIGMA0
    sin_mejora = 0
    for it in range(N_ITER):
        cands = [np.clip(mejor_v + rng.normal(0, sigma, mejor_v.shape), 0, AMP_MAX)
                 for _ in range(POB)]
        res = []
        for v in cands:
            Pf, Qf, _, _ = arma(v)
            res.append(evalua(Pf, Qf))
        pts = [puntaje(r, objetivo, df_min) for r in res]
        j = int(np.argmax(pts))
        if pts[j] > mejor_p:
            mejor, mejor_v, mejor_p = res[j], cands[j], pts[j]
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= 5:      # se estanco: afinar la busqueda
                sigma *= 0.6
                sin_mejora = 0
        hist.append(mejor_p)
        if it % 5 == 0 or it == N_ITER - 1:
            print(f"  {it:5d} {mejor['residuo']:10.5f} {mejor['df_rms']:10.5f} "
                  f"{100*mejor['frac_imitable']:9.1f}% {mejor['E_max']:8.3f}",
                  flush=True)

    # --- Validacion del ganador a resolucion completa -----------------------
    Pf, Qf, pv, qv = arma(mejor_v)
    fino = evalua(Pf, Qf, n_eval=4000, n_sub=4)

    print("\n=== RESULTADO ===")
    print(f"  {'':22} {'|Delta f|':>11} {'imitable':>11} {'RESIDUO':>11}")
    print(f"  {'APRBS de libreria':22} {ref['df_rms']:11.5f} "
          f"{100*ref['frac_imitable']:10.1f}% {ref['residuo']:11.5f}")
    print(f"  {'estimulo DISENADO':22} {fino['df_rms']:11.5f} "
          f"{100*fino['frac_imitable']:10.1f}% {fino['residuo']:11.5f}")
    mejora = 100 * (fino["residuo"] / ref["residuo"] - 1)
    print(f"\n  Mejora del residuo (a IGUAL amplitud media): {mejora:+.1f}%")
    print(f"  Fraccion imitable: {100*ref['frac_imitable']:.1f}% -> "
          f"{100*fino['frac_imitable']:.1f}%")

    OUT = OUT_BASE.with_name(OUT_BASE.name + f"_{objetivo}.json")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "referencia": ref, "disenado": fino, "mejora_pct": mejora,
        "P_seg": pv.tolist(), "Q_seg": qv.tolist(),
        "n_seg": N_SEG, "t_on": T_ON, "t_off": T_OFF,
        "amp_media_P": media_P, "amp_media_Q": media_Q,
        "historia": hist, "objetivo": objetivo,
    }, indent=2))
    _figura(pv, qv, hist, Pf_ref, Qf_ref, objetivo)
    print(f"\n  -> {OUT}")


def _figura(pv, qv, hist, Pf_ref, Qf_ref, objetivo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = np.linspace(T_ON, T_OFF, len(pv) + 1)
    t = np.linspace(0, 200, 1500)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].step(b[:-1], pv, where="post", label="P disenado", color="#1f4e79")
    ax[0].plot(t, [Pf_ref(x) for x in t], lw=0.8, alpha=0.6,
               label="P de libreria (APRBS)", color="#999999")
    ax[0].set_title("Estimulo a E"); ax[0].set_xlabel("t [ms]")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].step(b[:-1], qv, where="post", label="Q disenado", color="#d62728")
    ax[1].plot(t, [Qf_ref(x) for x in t], lw=0.8, alpha=0.6,
               label="Q de libreria", color="#999999")
    ax[1].set_title("Estimulo a I"); ax[1].set_xlabel("t [ms]")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    ax[2].plot(hist, color="#1f4e79")
    ax[2].set_title(f"objetivo: {objetivo}")
    ax[2].set_xlabel("iteracion"); ax[2].grid(alpha=0.3)
    fig.tight_layout()
    FIG_BASE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_BASE.with_name(FIG_BASE.name + f"_{objetivo}.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
