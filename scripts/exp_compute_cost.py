#!/usr/bin/env python3
# =============================================================================
#  COSTO COMPUTACIONAL DEL NEURAL ODE — experimentos baratos (1 + 2 + 3)
# =============================================================================
#
#  Responde, sin barrer entrenamientos completos, tres preguntas de costo:
#
#   (1) BASELINE.  Con el integrador actual (RK4 de paso fijo), ¿cuanto cuesta
#       integrar la dinamica? Metrica = NFE (nº de evaluaciones de f) y error de
#       la trayectoria vs una referencia de alta precision. Se compara ademas
#       contra el solver ADAPTATIVO (scipy RK45, sol.nfev) como anticipo del exp 5.
#
#   (2) RIGIDEZ (stiffness).  El estimador A-PRIORI del costo, analogo al numero de
#       condicion del FIM: autovalores del Jacobiano de ESTADO  ∂f/∂x  (matriz 2x2)
#       a lo largo de la trayectoria. El ratio |λmax|/|λmin| y |λmax| predicen el dt
#       maximo estable -> predicen el NFE, ANTES de barrer nada.
#
#   (3) BARRIDO DE dt x ORDEN DE SOLVER.  Euler (1 eval), RK2/midpoint (2), RK4 (4):
#       error vs referencia y NFE para varios dt. Da el dt mas grande (y el orden)
#       que mantiene la calidad -> el "win" mas barato para bajar el costo.
#
#  REUSA (no duplica el nucleo): make_rhs/true_theta (fisher_identifiability),
#  rollout/rk4_step (src.neural_ode), WilsonCowan + chirp_pulse (src.wilson_cowan).
#
#  USO:  python -u scripts/exp_compute_cost.py
# =============================================================================

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
from torch.func import jacrev
from scipy.integrate import solve_ivp

from src.wilson_cowan import WilsonCowan, WilsonCowanParams
from scripts.fisher_identifiability import make_rhs, true_theta

torch.set_default_dtype(torch.float64)

FIG = Path("results/figures/cost_dt_sweep.png")
T_END = 300.0                     # ventana temporal (unidades del modelo)
N_CMP = 3001                      # grilla de comparacion (referencia densa)
ERR_TOL = 1e-3                    # umbral de error de trayectoria "aceptable"
DT_GRID = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]   # pasos fijos a barrer
# Estimulo chirp SUAVE (sin gate: continuo en [0,T] para no inyectar discontinuidades
# que arruinen la convergencia de alto orden). Banda ~0.005-0.05 (donde WC reacciona),
# reescalado a [0, amplitude] para ser >= 0 (estimulo optogenetico no negativo).
def _smooth_chirp(amp, f0, f1):
    k = (f1 - f0) / T_END
    def f(t):
        phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
        return amp * (0.5 + 0.5 * np.sin(phase))
    return f

P_FN = _smooth_chirp(0.8, 0.005, 0.05)
Q_FN = _smooth_chirp(0.6, 0.005, 0.05)


# -----------------------------------------------------------------------------
#  Integradores de paso fijo. El estimulo se evalua EN LOS SUB-PASOS (igual que
#  la referencia adaptativa) para aislar el error del INTEGRADOR del error de
#  muestreo del estimulo. NOTA: el lazo cerrado del proyecto usa ZOH (P,Q constante
#  por paso, control muestreado) -> alli el orden alto no ayuda mas que O(dt); aca
#  medimos la exactitud de integracion pura (estimulo continuo conocido).
# -----------------------------------------------------------------------------
def euler_step_t(f, x, t, dt):
    return x + dt * f(x, P_FN(t), Q_FN(t))


def rk2_step_t(f, x, t, dt):           # midpoint (2 evaluaciones)
    k1 = f(x, P_FN(t), Q_FN(t))
    k2 = f(x + 0.5 * dt * k1, P_FN(t + 0.5 * dt), Q_FN(t + 0.5 * dt))
    return x + dt * k2


def rk4_step_t(f, x, t, dt):           # RK4 clasico (4 evaluaciones)
    k1 = f(x, P_FN(t), Q_FN(t))
    k2 = f(x + 0.5 * dt * k1, P_FN(t + 0.5 * dt), Q_FN(t + 0.5 * dt))
    k3 = f(x + 0.5 * dt * k2, P_FN(t + 0.5 * dt), Q_FN(t + 0.5 * dt))
    k4 = f(x + dt * k3, P_FN(t + dt), Q_FN(t + dt))
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


SOLVERS = {"euler": (euler_step_t, 1), "rk2": (rk2_step_t, 2), "rk4": (rk4_step_t, 4)}


# -----------------------------------------------------------------------------
#  Referencia de alta precision (scipy RK45 adaptativo, tolerancias apretadas).
# -----------------------------------------------------------------------------
def reference():
    wc = WilsonCowan(params=WilsonCowanParams(), P=P_FN, Q=Q_FN)
    t_eval = np.linspace(0.0, T_END, N_CMP)
    t0 = time.perf_counter()
    sol = solve_ivp(wc.rhs, (0.0, T_END), [0.0, 0.0], method="RK45",
                    t_eval=t_eval, rtol=1e-11, atol=1e-13, dense_output=True)
    wall = time.perf_counter() - t0
    # sol.sol(t) = solucion densa -> se evalua EXACTO en la grilla de cada solver
    # (sin error de interpolacion, que contaminaria los dt grandes).
    return t_eval, sol.y.T, int(sol.nfev), wall, wc, sol.sol


def integrate_fixed(step_fn, dt):
    """Integra con paso fijo dt; devuelve (t_grid, estados (n+1)x2, nº pasos)."""
    n = int(round(T_END / dt))
    f = make_rhs(THETA)
    x = torch.zeros(1, 2)
    xs = [x]
    for k in range(n):
        x = step_fn(f, x, k * dt, dt)
        xs.append(x)
    traj = torch.stack(xs, dim=0)[:, 0, :].numpy()
    return np.arange(n + 1) * dt, traj, n


def traj_error(tk, traj, dense):
    """Error max y RMS: se evalua la referencia densa EN la grilla tk del solver."""
    ref = dense(tk).T                       # (len(tk), 2), sin interpolacion
    err = np.linalg.norm(traj - ref, axis=1)
    return float(err.max()), float(np.sqrt(np.mean(err ** 2)))


# -----------------------------------------------------------------------------
#  (2) Rigidez: autovalores de ∂f/∂x a lo largo de la trayectoria de referencia.
# -----------------------------------------------------------------------------
def stiffness(t_ref, y_ref, n_samples=200):
    f = make_rhs(THETA)

    def fx(x2, P, Q):                      # f: R^2 -> R^2 (P,Q escalares fijos)
        return f(x2.reshape(1, 2), torch.tensor([[P]]), torch.tensor([[Q]])).reshape(2)

    idx = np.linspace(0, len(t_ref) - 1, n_samples).astype(int)
    lam_max, lam_min = [], []
    for i in idx:
        x2 = torch.tensor(y_ref[i])
        P, Q = float(P_FN(float(t_ref[i]))), float(Q_FN(float(t_ref[i])))
        Jx = jacrev(lambda x: fx(x, P, Q))(x2)          # 2x2
        ev = torch.linalg.eigvals(Jx).abs().numpy()
        lam_max.append(ev.max()); lam_min.append(ev.min())
    lam_max, lam_min = np.array(lam_max), np.array(lam_min)
    ratio = lam_max / np.maximum(lam_min, 1e-30)
    return lam_max, lam_min, ratio


THETA = true_theta()


def main():
    print("=" * 74)
    print(" COSTO COMPUTACIONAL DEL NEURAL ODE — baseline + rigidez + barrido dt")
    print("=" * 74)

    # --- Referencia de alta precision + baseline adaptativo (exp 1 / anticipo 5) --
    t_ref, y_ref, nfev_ref, wall_ref, wc, dense = reference()
    print(f"\n[ref] scipy RK45 adaptativo rtol=1e-11 (verdad): "
          f"nfev={nfev_ref}  wall={wall_ref*1e3:.1f} ms")
    # Baseline adaptativo a tolerancia realista (el NFE a batir; anticipo del exp 5).
    sol_a = solve_ivp(wc.rhs, (0.0, T_END), [0.0, 0.0], method="RK45",
                      t_eval=t_ref, rtol=1e-6, atol=1e-9)
    emax_a, _ = traj_error(t_ref, sol_a.y.T, dense)
    print(f"[1] ADAPTATIVO rtol=1e-6 (a batir): nfev={int(sol_a.nfev)}  "
          f"err_max={emax_a:.2e}")

    # --- (2) Rigidez -----------------------------------------------------------
    lam_max, lam_min, ratio = stiffness(t_ref, y_ref)
    Lmax = float(lam_max.max())
    dt_stable_rk4 = 2.78 / Lmax        # limite de estabilidad RK4 sobre eje real
    print("\n[2] RIGIDEZ (autovalores de ∂f/∂x a lo largo de la trayectoria)")
    print(f"    |λ| max = {Lmax:.4f}   |λ| min = {float(lam_min.min()):.4f}")
    print(f"    ratio de rigidez |λmax|/|λmin| : mediana={np.median(ratio):.2f}  "
          f"max={ratio.max():.2f}")
    print(f"    -> dt estable (RK4, ~2.78/|λmax|) ≈ {dt_stable_rk4:.3f}   "
          f"[WC {'NO es rigido' if ratio.max() < 20 else 'ES rigido'}]")

    # --- (1)+(3) Barrido dt x orden de solver ---------------------------------
    print("\n[1+3] BARRIDO dt x ORDEN DE SOLVER (error vs referencia, NFE, tiempo)")
    print(f"    {'solver':6} {'dt':>6} {'pasos':>7} {'NFE':>9} "
          f"{'err_max':>10} {'err_rms':>10} {'wall_ms':>9}")
    results = {name: {"dt": [], "nfe": [], "emax": []} for name in SOLVERS}
    for name, (step_fn, evals) in SOLVERS.items():
        for dt in DT_GRID:
            t0 = time.perf_counter()
            tk, traj, n = integrate_fixed(step_fn, dt)
            wall = (time.perf_counter() - t0) * 1e3
            emax, erms = traj_error(tk, traj, dense)
            nfe = n * evals
            ok = "" if emax < ERR_TOL else "  (> tol)"
            print(f"    {name:6} {dt:6.3f} {n:7d} {nfe:9d} "
                  f"{emax:10.2e} {erms:10.2e} {wall:9.1f}{ok}")
            results[name]["dt"].append(dt)
            results[name]["nfe"].append(nfe)
            results[name]["emax"].append(emax)

    # dt mas grande que cumple el umbral, por solver.
    print(f"\n    dt maximo con err_max < {ERR_TOL:g} (menos NFE = mas barato):")
    for name in SOLVERS:
        dts = [d for d, e in zip(results[name]["dt"], results[name]["emax"]) if e < ERR_TOL]
        best = max(dts) if dts else None
        print(f"      {name:6}: {'dt=%.3f' % best if best else 'ninguno en la grilla'}")

    _plot(results, Lmax, dt_stable_rk4)
    print(f"\nFigura: {FIG}")


def _plot(results, Lmax, dt_stable):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {"euler": "#d62728", "rk2": "#ff7f0e", "rk4": "#1f4e79"}

    for name in results:
        ax[0].loglog(results[name]["dt"], results[name]["emax"], "o-",
                     color=colors[name], label=name.upper())
    ax[0].axhline(ERR_TOL, ls=":", color="#888", label=f"tol {ERR_TOL:g}")
    ax[0].axvline(dt_stable, ls="--", color="#2ca02c",
                  label=f"dt estable≈{dt_stable:.2f}")
    ax[0].set_xlabel("paso dt"); ax[0].set_ylabel("error max de trayectoria")
    ax[0].set_title("Error vs dt (por orden de solver)")
    ax[0].grid(True, which="both", alpha=0.3); ax[0].legend(fontsize=8)

    for name in results:
        ax[1].loglog(results[name]["nfe"], results[name]["emax"], "o-",
                     color=colors[name], label=name.upper())
    ax[1].axhline(ERR_TOL, ls=":", color="#888")
    ax[1].set_xlabel("NFE (nº evaluaciones de f)"); ax[1].set_ylabel("error max")
    ax[1].set_title("Frontera costo-precision (Pareto)")
    ax[1].grid(True, which="both", alpha=0.3); ax[1].legend(fontsize=8)

    fig.suptitle(f"Costo computacional — WC |λmax|≈{Lmax:.2f} (rigidez baja)", fontsize=11)
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
