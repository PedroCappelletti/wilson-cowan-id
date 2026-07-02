#!/usr/bin/env python3
# =============================================================================
#  EXPERIMENTO B — Selección de subconjunto identificable (subset selection por FIM)
# =============================================================================
#
#  ¿Qué conviene fijar para mejorar la identificación del resto? Metodo establecido:
#  PARAMETER SUBSET SELECTION guiado por la matriz de sensibilidad / FIM.
#
#    B1. Ranking de identificabilidad por QR con pivoteo de columnas sobre la matriz
#        de sensibilidad relativa (Golub column pivoting; usado por Chu & Hahn 2007,
#        Quaiser & Monnigmann 2009). El pivoteo ordena los parametros de MAS a MENOS
#        identificable; la cola (pivotes chicos) = candidatos a fijar. Se contrasta
#        con las direcciones singulares debiles de la SVD (mismas de fisher_*).
#    B2. Fijar un representante de cada acople debil (wII; luego ae, ai) y medir la
#        mejora del resto bajo ruido.
#    B3. Tabla/heatmap empirico "fijar X -> error de Y", contrastado con B1.
#
#  Reutiliza predict/true_theta/PNAMES (fisher), inputs_from (exp_input_design),
#  identify_subset (ident_subset) y el baseline 10-libres (noise_full_sweep.json).
#
#  USO:  python -u scripts/exp_subset_selection.py
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
import scipy.linalg as sla
import torch
from torch.func import jacfwd

from scripts.fisher_identifiability import predict, true_theta, PNAMES
from scripts.exp_input_design import inputs_from
from scripts.ident_subset import identify_subset, ALL_P
from scripts.noise_improve import strong_scenarios
from scripts.noise_final import smooth_k_for

torch.set_default_dtype(torch.float64)

SIGMA_B = 0.10
FIX_CONFIGS = [
    ("10 libres",   ()),
    ("fijo wII",    ("wII",)),
    ("fijo ae",     ("ae",)),
    ("fijo ai",     ("ai",)),
    ("fijo ti",     ("ti",)),
    ("fijo wII+ae", ("wII", "ae")),
    ("fijo wII+ai", ("wII", "ai")),
]
OUT_JSON = Path("results/exp_subset_selection.json")
FIG_B = Path("results/figures/expB_heatmap_fijar.png")


# ---------------------------------------------------------------------------
#  B1 — ranking por QR con pivoteo + direcciones singulares debiles.
# ---------------------------------------------------------------------------
def b1_subset_selection():
    theta = true_theta()
    inp, dt = inputs_from(strong_scenarios()[:6])
    J = jacfwd(lambda th: predict(th, inp, dt))(theta)
    Jr = (J * theta.unsqueeze(0)).numpy()          # sensibilidad relativa

    # QR con pivoteo de columnas: el orden de pivoteo = de MAS a MENOS identificable.
    _, _, piv = sla.qr(Jr, mode="economic", pivoting=True)
    ranking = [PNAMES[i] for i in piv]

    # SVD: direccion mas debil y su combinacion dominante.
    U, S, Vt = np.linalg.svd(Jr, full_matrices=False)
    weak = Vt[-1]; order = np.argsort(-np.abs(weak))
    cond = S[0] / S[-1]

    print("=== B1 — subset selection (QR con pivoteo sobre sensibilidad relativa) ===")
    print("  Identificabilidad (MAS -> MENOS):")
    print("    " + "  >  ".join(ranking))
    print(f"  Candidatos a FIJAR (cola del pivoteo): {ranking[-3:]}")
    print(f"\n  Numero de condicion σ1/σ10 = {cond:.2e}")
    print("  Direccion singular mas debil (σ10) dominada por: "
          + "  ".join(f"{PNAMES[j]}={weak[j]:+.2f}" for j in order[:4]))
    return {"ranking_qr": ranking, "cond": float(cond),
            "weak_dir": {PNAMES[j]: float(weak[j]) for j in order[:4]}}


# ---------------------------------------------------------------------------
#  B2/B3 — fijar candidatos y medir el error del resto (heatmap).
# ---------------------------------------------------------------------------
def b2_b3():
    sk = smooth_k_for(SIGMA_B)
    scen = strong_scenarios()
    print(f"\n=== B2/B3 — fijar candidatos y medir el resto (σ={SIGMA_B}, k={sk}) ===")
    rows = []
    for name, fix in FIX_CONFIGS:
        _, err = identify_subset(scen, SIGMA_B, sk, fix=set(fix))
        free = [k for k in ALL_P if k not in fix]
        rest_max = max(err[k] for k in free)
        rows.append({"name": name, "fix": list(fix), "err": err, "rest_max": rest_max})
        print(f"  {name:12} | resto máx={rest_max:6.2f}% | "
              + " ".join(f"{k}={err[k]:.1f}" for k in ("wII", "ti", "ai", "ae", "wEE")))
    return rows


def _plot_heatmap(rows):
    import matplotlib.pyplot as plt
    names = [r["name"] for r in rows]
    # matriz error%: filas = config, col = parametro. Fijados -> NaN (no estimado).
    M = np.full((len(rows), len(ALL_P)), np.nan)
    for i, r in enumerate(rows):
        for j, k in enumerate(ALL_P):
            if k not in r["fix"]:
                M[i, j] = r["err"][k]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=40)
    ax.set_xticks(range(len(ALL_P))); ax.set_xticklabels(ALL_P, rotation=45, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_title(f"B3: fijar X (fila) → error de cada parámetro (σ={SIGMA_B}). "
                 "Gris = fijado; verde = mejor")
    for i in range(len(rows)):
        for j, k in enumerate(ALL_P):
            if np.isnan(M[i, j]):
                ax.text(j, i, "fix", ha="center", va="center", fontsize=7, color="#555")
            else:
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="black")
    fig.colorbar(im, ax=ax, fraction=0.03, label="error θ̂ (%)")
    fig.tight_layout(); FIG_B.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_B, dpi=120); plt.close(fig)


def main():
    b1 = b1_subset_selection()
    rows = b2_b3()
    _plot_heatmap(rows)
    OUT_JSON.write_text(json.dumps({"B1": b1, "B2_B3": rows}, indent=2), encoding="utf-8")
    print(f"\nFigura: {FIG_B}\nResultados: {OUT_JSON}")


if __name__ == "__main__":
    main()
