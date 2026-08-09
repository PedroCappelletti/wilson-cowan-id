#!/usr/bin/env python3
# =============================================================================
#  F4b — LA GEOMETRIA DEL MISMATCH
# =============================================================================
#
#  Esta es la medicion que explica todos los demas resultados, y solo se puede
#  hacer porque conocemos el simulador y por lo tanto el Delta f VERDADERO.
#
#  La pregunta: de la fisica que al modelo le falta, ¿cuanta es "parecida a mover
#  parametros" y cuanta es genuinamente nueva?
#
#  Formalmente: se proyecta el Delta f verdadero sobre el espacio que generan las
#  sensibilidades ∂f/∂θ evaluadas en los parametros VERDADEROS.
#
#      Delta f  =  S·δθ*        +        residuo
#                  \_______/             \_____/
#              imitable moviendo θ      fisica que NINGUN
#              (el white-box la va a     ajuste de los 10
#               absorber sesgando θ)     parametros puede dar
#
#  Por que importa tanto:
#
#  1. Explica el sesgo del white-box. Si una fraccion grande del mismatch es
#     imitable, entonces sesgar los parametros es la MEJOR respuesta disponible
#     para un modelo rigido. El error parametrico no es un fracaso del
#     optimizador: es el precio estructural de la rigidez.
#
#  2. Pone un techo al gray-box. Sobre la parte imitable, el reparto entre θ y g
#     NO ES UNICO: cualquier division ajusta los datos igual de bien. Por eso la
#     correccion aprendida puede tener la magnitud correcta y aun asi no parecerse
#     al Delta f verdadero.
#
#  3. Dice cuando conviene regularizar. Si la parte genuinamente nueva es chica,
#     casi todo lo que g puede hacer es competir con θ, y hay que restringirla.
#
#  USO:  python scripts/exp_f4b_geometria_mismatch.py
# =============================================================================

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

torch.set_num_threads(2)

from src.wilson_cowan import WilsonCowanParams
from src.neural_ode import GrayBoxWC
from src.neural_ode.graybox_train import projection_operator, projected_fraction, ALL_P

DATA_DIR = Path("data/processed/uncertain")
EPS_GRID = (0.25, 0.5, 1.0, 1.5, 2.0)
OUT = Path("results/uncertainty/f4b_geometria.json")
FIG = Path("results/figures/f4b_geometria.png")
N_PUNTOS = 4000


def analiza(path: Path, seed: int = 0) -> dict | None:
    d = np.load(path, allow_pickle=True)
    if "dfI" not in d:
        return None
    it = d["is_test"].astype(bool)
    I, E, P, Q = (d[k][~it].ravel() for k in ("I", "E", "P", "Q"))
    dfI, dfE = d["dfI"][~it].ravel(), d["dfE"][~it].ravel()
    if np.allclose(dfI, 0) and np.allclose(dfE, 0):
        return None                       # eps = 0: no hay mismatch

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(I), min(N_PUNTOS, len(I)), replace=False)
    X = torch.tensor(np.stack([I[idx], E[idx]], 1), dtype=torch.float32)
    Pt = torch.tensor(P[idx].reshape(-1, 1), dtype=torch.float32)
    Qt = torch.tensor(Q[idx].reshape(-1, 1), dtype=torch.float32)
    DF = torch.tensor(np.stack([dfI[idx], dfE[idx]], 1), dtype=torch.float32)

    # Sensibilidades evaluadas en los parametros VERDADEROS: es la geometria del
    # problema, no la de una estimacion particular.
    p = WilsonCowanParams()
    true = {k: getattr(p, k) for k in ALL_P}
    m = GrayBoxWC(true, {k: true[k] for k in ("wEE", "wEI", "wIE", "wII")},
                  learnable_weights=True, learnable_params=True)
    S = m.backbone_sensitivities(X, Pt, Qt)
    A, Sf = projection_operator(S)
    _, frac = projected_fraction(DF, A, Sf)
    c = A @ DF.reshape(-1)
    proj = (Sf @ c).reshape(-1, 2)
    res = DF - proj

    return {
        "eps": float(d["eps"]) if "eps" in d else float("nan"),
        "df_rms": float(DF.pow(2).mean().sqrt()),
        "frac_imitable": float(frac),
        "rms_imitable": float(proj.pow(2).mean().sqrt()),
        "rms_nuevo": float(res.pow(2).mean().sqrt()),
        "delta_theta": {k: float(c[j]) for j, k in enumerate(GrayBoxWC.PARAM_ORDER)},
    }


def por_estimulo(path: Path) -> list[dict]:
    """La fraccion imitable NO es una constante: es geometria de la TRAYECTORIA,
    y la trayectoria la fija el estimulo. Si varia mucho entre familias de
    estimulo, entonces se puede ELEGIR el estimulo que la minimice -> el mismatch
    dejaria de parecerse a un cambio de parametros y g_phi estaria obligada a
    aprender fisica real. Es el primer paso del diseno de entradas para el
    hibrido (ver seccion 20-A del manual)."""
    d = np.load(path, allow_pickle=True)
    if "dfI" not in d:
        return []
    p = WilsonCowanParams()
    true = {k: getattr(p, k) for k in ALL_P}
    m = GrayBoxWC(true, {k: true[k] for k in ("wEE", "wEI", "wIE", "wII")},
                  learnable_weights=True, learnable_params=True)
    labels = [str(x) for x in d["labels"]]
    out = []
    rng = np.random.default_rng(0)
    for s, lab in enumerate(labels):
        I, E, P, Q = d["I"][s], d["E"][s], d["P"][s], d["Q"][s]
        DFn = np.stack([d["dfI"][s], d["dfE"][s]], 1)
        if np.allclose(DFn, 0):
            continue
        idx = rng.choice(len(I), min(2000, len(I)), replace=False)
        X = torch.tensor(np.stack([I[idx], E[idx]], 1), dtype=torch.float32)
        Pt = torch.tensor(P[idx].reshape(-1, 1), dtype=torch.float32)
        Qt = torch.tensor(Q[idx].reshape(-1, 1), dtype=torch.float32)
        DF = torch.tensor(DFn[idx], dtype=torch.float32)
        S = m.backbone_sensitivities(X, Pt, Qt)
        A, Sf = projection_operator(S)
        _, frac = projected_fraction(DF, A, Sf)
        out.append({"label": lab, "frac_imitable": float(frac),
                    "df_rms": float(DF.pow(2).mean().sqrt())})
    return out


def main():
    filas = []
    print("=== F4b · geometria del mismatch ===")
    print("    ¿cuanta de la fisica que falta es indistinguible de mover parametros?\n")
    hdr = (f"  {'eps':>5} {'|Df|':>9} {'imitable por θ':>15} "
           f"{'|imitable|':>11} {'|nuevo|':>10}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for eps in EPS_GRID:
        p = DATA_DIR / f"eps{eps:g}.npz"
        if not p.exists():
            continue
        r = analiza(p)
        if r is None:
            continue
        print(f"  {r['eps']:5.2f} {r['df_rms']:9.5f} {100*r['frac_imitable']:14.1f}% "
              f"{r['rms_imitable']:11.5f} {r['rms_nuevo']:10.5f}")
        filas.append(r)

    if filas:
        m = float(np.mean([f["frac_imitable"] for f in filas]))
        print(f"\n  Promedio: {100*m:.1f}% del mismatch es imitable moviendo los 10 "
              f"parametros.")
        print(f"  => El sesgo parametrico del white-box no es un fallo del ajuste:")
        print(f"     es la mejor respuesta que un modelo rigido puede dar.")
        print(f"  => Y sobre esa fraccion, el reparto entre θ y g NO es unico, asi que")
        print(f"     la correccion puede tener el tamano correcto sin parecerse al Df real.")

        print(f"\n  Direccion del cambio de parametros equivalente (eps=1):")
        uno = [f for f in filas if abs(f["eps"] - 1.0) < 1e-9]
        if uno:
            dt = uno[0]["delta_theta"]
            for k in sorted(dt, key=lambda k: -abs(dt[k]))[:5]:
                print(f"     {k:8} {dt[k]:+.3f}")

    # --- Desglose por familia de estimulo (primer paso del diseno de entradas) ---
    porest = por_estimulo(DATA_DIR / "eps1.npz")
    if porest:
        print("\n=== Fraccion imitable POR ESTIMULO (eps=1) ===")
        print("    Si varia mucho, se puede DISENAR el estimulo que la minimice,")
        print("    y asi obligar a la correccion a aprender fisica de verdad.\n")
        print(f"  {'estimulo':20} {'|Df|':>9} {'imitable por θ':>16}")
        for r in sorted(porest, key=lambda r: r["frac_imitable"]):
            print(f"  {r['label']:20} {r['df_rms']:9.5f} "
                  f"{100*r['frac_imitable']:15.1f}%")
        fr = [r["frac_imitable"] for r in porest]
        print(f"\n  rango: {100*min(fr):.1f}% a {100*max(fr):.1f}%  "
              f"(spread de {100*(max(fr)-min(fr)):.1f} puntos)")
        mejor = min(porest, key=lambda r: r["frac_imitable"])
        peor = max(porest, key=lambda r: r["frac_imitable"])
        print(f"  El estimulo '{mejor['label']}' deja el mismatch mucho menos")
        print(f"  confundible con parametros que '{peor['label']}'.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"por_eps": filas, "por_estimulo": porest}, indent=2))
    _figura(filas)
    print(f"\n  -> {OUT}")


def _figura(filas):
    if not filas:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eps = [f["eps"] for f in filas]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(eps, [100 * f["frac_imitable"] for f in filas], "o-", color="#d62728")
    ax[0].set_xlabel("eps"); ax[0].set_ylabel("% del mismatch imitable por θ")
    ax[0].set_ylim(0, 100)
    ax[0].set_title("Cuanto del hueco es 'parecido a mover parametros'")
    ax[0].grid(alpha=0.3)

    ax[1].plot(eps, [f["rms_imitable"] for f in filas], "o-",
               color="#d62728", label="imitable por θ")
    ax[1].plot(eps, [f["rms_nuevo"] for f in filas], "s-",
               color="#1f4e79", label="fisica genuinamente nueva")
    ax[1].set_xlabel("eps"); ax[1].set_ylabel("RMS")
    ax[1].set_title("Descomposicion del Delta f verdadero")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
