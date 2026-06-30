#!/usr/bin/env python3
# =============================================================================
#  IDENTIFICABILIDAD DE LOS 10 PARAMETROS — Matriz de Fisher + SVD
# =============================================================================
#
#  Pregunta: bajo ruido de observacion, ¿que se puede recuperar de los 10 params
#  de Wilson-Cowan, y que COMBINACIONES son poco identificables? Es la generalizacion
#  del proxy s2/s1 (nube E-I) que ya usa el proyecto: en vez de 2 direcciones (wIE,wII),
#  miramos las 10 a la vez via la Matriz de Informacion de Fisher (FIM) y su SVD
#  (metodo de Plate et al. 2024, OED para UDEs).
#
#  Como: para un estimulo dado se arma la trayectoria predicha y(θ) = [I(t),E(t)]
#  apilada. La sensibilidad es el jacobiano  J = ∂y/∂θ  evaluado en θ_verdadero
#  (por modo-forward, jacfwd: 10 evaluaciones). Con ruido i.i.d. de varianza σ²,
#       FIM = Jᵀ J / σ²   →   sus autovalores = (valores singulares de J)² / σ².
#  Trabajamos en sensibilidad RELATIVA (columnas escaladas por θ_j) para que la SVD
#  compare cambios FRACCIONALES de cada parametro (adimensional, log-parametrizacion).
#
#  Reporta:
#    (a) espectro de valores singulares (direcciones bien vs mal condicionadas),
#    (b) numero de condicion,
#    (c) vectores singulares de los valores singulares chicos -> que combinaciones
#        de parametros son poco identificables (esperar acoples ae-wEE, wIE-wII).
#
#  NO toca el nucleo de la dinamica (src/neural_ode/dynamics.py): el rhs de abajo es
#  un ESPEJO diferenciable de ese forward, solo para el analisis de sensibilidad.
#
#  USO:  python -u scripts/fisher_identifiability.py
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.func import jacfwd

from src.wilson_cowan import WilsonCowanParams
from src.neural_ode import rollout
from scripts.noise_improve import strong_scenarios, generate

torch.set_default_dtype(torch.float64)

# Orden canonico de los 10 parametros.
PNAMES = ["wEE", "wEI", "wIE", "wII", "te", "ti", "ae", "ai", "thetae", "thetai"]
SUBSAMPLE = 5     # submuestreo temporal del jacobiano (condicionamiento es robusto)
N_SCEN = 4        # cuantos escenarios fuertes usar (diversidad de excitacion)
FIG = Path("results/figures/fisher_svd.png")


def true_theta() -> torch.Tensor:
    p = WilsonCowanParams()
    return torch.tensor([p.wEE, p.wEI, p.wIE, p.wII, p.te, p.ti,
                         p.ae, p.ai, p.thetae, p.thetai])


def make_rhs(theta):
    """Espejo diferenciable del forward de GrayBoxWC (NO modifica el nucleo)."""
    wEE, wEI, wIE, wII, te, ti, ae, ai, thetae, thetai = theta
    ke = torch.sigmoid(-ae * thetae)
    ki = torch.sigmoid(-ai * thetai)

    def f(x, P, Q):
        I = x[..., 0:1]
        E = x[..., 1:2]
        u_i = wIE * E - wII * I + Q - thetai
        u_e = wEE * E - wEI * I + P - thetae
        dI = (1.0 / ti) * (-I + torch.sigmoid(ai * u_i) - ki)
        dE = (1.0 / te) * (-E + torch.sigmoid(ae * u_e) - ke)
        return torch.cat([dI, dE], dim=-1)
    return f


def build_inputs():
    """Trayectorias (x0, P, Q) de varios escenarios fuertes, en limpio."""
    I, E, P, Q, is_test, dt = generate(strong_scenarios()[:N_SCEN], noise=0.0)
    inputs = []
    for s in range(I.shape[0]):
        T = I.shape[1]
        x0 = torch.tensor([[I[s, 0], E[s, 0]]])
        Ps = torch.tensor(P[s]).reshape(T, 1, 1)
        Qs = torch.tensor(Q[s]).reshape(T, 1, 1)
        inputs.append((x0, Ps, Qs))
    return inputs, float(dt)


def predict(theta, inputs, dt):
    """Apila las trayectorias predichas (submuestreadas) en un vector y(θ)."""
    outs = []
    for x0, Ps, Qs in inputs:
        traj = rollout(make_rhs(theta), x0, Ps[:-1], Qs[:-1], dt)[:, 0, :]  # (T,2)
        outs.append(traj[::SUBSAMPLE].reshape(-1))
    return torch.cat(outs)


def main():
    theta = true_theta()
    inputs, dt = build_inputs()
    print(f"=== Identificabilidad FIM+SVD — {len(inputs)} escenarios, dt={dt:.4f} ms ===")

    # Jacobiano J = ∂y/∂θ por modo-forward (10 columnas).
    J = jacfwd(lambda th: predict(th, inputs, dt))(theta)   # (N_out, 10)
    # Sensibilidad RELATIVA: columna j escalada por θ_j (cambios fraccionales).
    Jr = J * theta.unsqueeze(0)
    print(f"    salidas (N_out) = {Jr.shape[0]}")

    # SVD de la sensibilidad relativa.
    U, S, Vt = torch.linalg.svd(Jr, full_matrices=False)
    S = S.numpy()
    V = Vt.numpy()          # filas = vectores singulares derechos (en espacio de params)
    Snorm = S / S[0]
    cond = S[0] / S[-1]

    print("\n--- Valores singulares (normalizados al mayor) ---")
    for i, (s, sn) in enumerate(zip(S, Snorm)):
        bien = "bien cond." if sn > 1e-2 else ("debil" if sn > 1e-3 else "MUY debil")
        print(f"  σ{i+1:2d} = {s:11.4e}   (rel {sn:9.2e})   {bien}")
    print(f"\n  Numero de condicion  σ1/σ10 = {cond:.3e}")

    print("\n--- Direcciones MAL condicionadas (σ chicos) -> combinaciones poco identificables ---")
    for i in range(len(S) - 1, max(len(S) - 4, -1), -1):
        vec = V[i]
        order = np.argsort(-np.abs(vec))
        comps = "  ".join(f"{PNAMES[j]}={vec[j]:+.3f}" for j in order[:4])
        print(f"  σ{i+1} (rel {Snorm[i]:.2e}): {comps}")

    print("\n--- Sensibilidad relativa total por parametro (||columna||) ---")
    col = np.linalg.norm(Jr.numpy(), axis=0)
    for j in np.argsort(col):
        print(f"  {PNAMES[j]:7} {col[j]:.3e}")

    _plot(S, Snorm, V)
    print(f"\nFigura: {FIG}")
    return S, V


def _plot(S, Snorm, V):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].semilogy(range(1, len(S) + 1), Snorm, "o-", color="#1f4e79")
    ax[0].axhline(1e-2, ls=":", color="#aaaaaa")
    ax[0].axhline(1e-3, ls=":", color="#d62728")
    ax[0].set_xlabel("indice"); ax[0].set_ylabel("valor singular (normalizado)")
    ax[0].set_title("Espectro FIM (sensibilidad relativa)"); ax[0].grid(True, alpha=0.3)

    im = ax[1].imshow(np.abs(V), aspect="auto", cmap="viridis")
    ax[1].set_xticks(range(len(PNAMES))); ax[1].set_xticklabels(PNAMES, rotation=45, ha="right", fontsize=8)
    ax[1].set_yticks(range(len(S))); ax[1].set_yticklabels([f"σ{i+1}" for i in range(len(S))], fontsize=8)
    ax[1].set_title("|vectores singulares| (fila = modo, col = parametro)")
    fig.colorbar(im, ax=ax[1], fraction=0.046)

    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
