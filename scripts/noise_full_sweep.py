#!/usr/bin/env python3
# =============================================================================
#  ROBUSTEZ AL RUIDO — IDENTIFICACION COMPLETA (10 params) + PROPAGACION AL CONTROL
# =============================================================================
#
#  Lleva la identificacion de los 10 parametros de Wilson-Cowan al CASO REALISTA:
#  barrido de ruido de observacion σ = [0, 0.01, 0.05, 0.10], reutilizando el
#  suavizado ADAPTATIVO ya existente (ventana k que crece con σ; noise_final) y la
#  excitacion fuerte (strong_scenarios; noise_improve). Para cada σ:
#
#    1) IDENTIFICACION: arranque ignorante (todo 1.0), aprende los 10 params
#       (learnable_params=True, use_correction=False). Reporta error POR parametro
#       y lo compara contra identificar solo los 4 pesos (mismo pipeline).
#
#    2) PROPAGACION AL CONTROL: arma el controlador IMC ENTERAMENTE con el θ̂ ruidoso
#       (los 10 params; ke,ki derivados) y mide el RMSE de seguimiento en lazo cerrado
#       sobre la planta verdadera. Pregunta: ¿la accion integral absorbe el error de
#       identificacion aun con θ̂ malo?
#
#  Predice el diagnostico FIM+SVD (fisher_identifiability.py): la direccion mas debil
#  esta dominada por wII, y los acoples ae-wEE / ai-ti-thetai son los mas flojos ->
#  esos parametros deberian degradarse antes.
#
#  USO:  python -u scripts/noise_full_sweep.py
# =============================================================================

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.wilson_cowan import WilsonCowanParams
from src.neural_ode import (
    GrayBoxWC, rollout,
    IMCController, make_true_plant, simulate_closed_loop, theta_gamma_refs,
)
from scripts.train_neural_ode import make_windows, WINDOW
from scripts.noise_improve import strong_scenarios, smooth, generate, identify
from scripts.noise_final import smooth_k_for

NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10]
EPOCHS, LBFGS_STEPS = 1500, 40
LR_W, LR_PHYS = 5e-2, 2e-2

WEIGHTS = ("wEE", "wEI", "wIE", "wII")
PHYS    = ("te", "ti", "ae", "ai", "thetae", "thetai")
ALL_P   = WEIGHTS + PHYS

T_SPAN_CL, DT_CL = (0.0, 50.0), 0.005   # lazo cerrado (ms, ver convencion de unidades)

OUT_FIG_ERR = Path("results/figures/noise_param_error.png")
OUT_FIG_CTRL = Path("results/figures/noise_control_rmse.png")
OUT_JSON = Path("results/noise_full_sweep.json")


def _true():
    p = WilsonCowanParams()
    return {"wEE": p.wEE, "wEI": p.wEI, "wIE": p.wIE, "wII": p.wII,
            "te": p.te, "ti": p.ti, "ae": p.ae, "ai": p.ai,
            "thetae": p.thetae, "thetai": p.thetai}


def _fixed_true():
    p = WilsonCowanParams()
    return {"te": p.te, "ti": p.ti, "ae": p.ae, "ai": p.ai,
            "thetae": p.thetae, "thetai": p.thetai, "ke": p.ke, "ki": p.ki}


def fixed_from_params(p: dict) -> dict:
    """fixed (con ke,ki derivados) desde un params_dict de 10 valores."""
    ae, ai, the, thi = p["ae"], p["ai"], p["thetae"], p["thetai"]
    return {"te": p["te"], "ti": p["ti"], "ae": ae, "ai": ai,
            "thetae": the, "thetai": thi,
            "ke": 1.0 / (1.0 + np.exp(ae * the)),
            "ki": 1.0 / (1.0 + np.exp(ai * thi))}


def identify_full(scenarios, noise, smooth_k):
    """Identifica los 10 params desde arranque ignorante (1.0). Devuelve (p, err)."""
    true = _true()
    I, E, P, Q, is_test, dt = generate(scenarios, noise)
    if smooth_k > 1:
        I, E = smooth(I, smooth_k), smooth(E, smooth_k)
    Itr, Etr, Ptr, Qtr = I[~is_test], E[~is_test], P[~is_test], Q[~is_test]

    init = {k: 1.0 for k in ALL_P}
    model = GrayBoxWC(init, {k: 1.0 for k in WEIGHTS},
                      learnable_weights=True, use_correction=False, learnable_params=True)
    x0, Pw, Qw, tgt = make_windows(Itr, Etr, Ptr, Qtr, WINDOW)

    phys_raw = [getattr(model, f"raw_{k}") for k in PHYS]
    opt = torch.optim.Adam([{"params": [model.raw_w], "lr": LR_W},
                            {"params": phys_raw, "lr": LR_PHYS}])
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = ((rollout(model, x0, Pw, Qw, dt) - tgt) ** 2).mean()
        loss.backward(); opt.step()

    opt2 = torch.optim.LBFGS([model.raw_w] + phys_raw, lr=1.0, max_iter=20,
                             line_search_fn="strong_wolfe")
    def closure():
        opt2.zero_grad(); l = ((rollout(model, x0, Pw, Qw, dt) - tgt) ** 2).mean()
        l.backward(); return l
    for _ in range(LBFGS_STEPS):
        opt2.step(closure)

    p = model.params_dict()
    err = {k: 100.0 * abs(p[k] - true[k]) / abs(true[k]) for k in ALL_P}
    return p, err


def _rmse(sol):
    n0 = len(sol["t"]) // 5
    rI = float(np.sqrt(np.mean((sol["I"][n0:] - sol["rI"][n0:]) ** 2)))
    rE = float(np.sqrt(np.mean((sol["E"][n0:] - sol["rE"][n0:]) ** 2)))
    return rI, rE


def main():
    true = _true()
    fixed_true = _fixed_true()
    refs = theta_gamma_refs(freq_hz=120.0, time_in_ms=True)
    plant_true = make_true_plant(fixed_true, {k: true[k] for k in WEIGHTS})

    # Baseline ideal del control (parametros reales).
    rI0, rE0 = _rmse(simulate_closed_loop(plant_true, IMCController(fixed_true, {k: true[k] for k in WEIGHTS}),
                                          refs, t_span=T_SPAN_CL, dt=DT_CL))
    print(f"baseline ideal (params reales): control RMSE I={rI0:.3e} E={rE0:.3e}\n")

    scen = strong_scenarios()
    rows = []
    for noise in NOISE_LEVELS:
        torch.manual_seed(0); np.random.seed(0)
        sk = smooth_k_for(noise)

        # (1) Identificacion 10 params.
        p_hat, err10 = identify_full(scen, noise, sk)
        # 4 pesos (mismo pipeline, solo-pesos) para comparar.
        torch.manual_seed(0); np.random.seed(0)
        w4, err4 = identify(scen, noise, sk)

        # (2) Propagacion al control: IMC con θ̂ ruidoso (10 params).
        fixed_hat = fixed_from_params(p_hat)
        w_hat = {k: p_hat[k] for k in WEIGHTS}
        ctrl_hat = IMCController(fixed_hat, w_hat)
        rI, rE = _rmse(simulate_closed_loop(plant_true, ctrl_hat, refs, t_span=T_SPAN_CL, dt=DT_CL))

        rows.append({"noise": noise, "k": sk, "p_hat": p_hat, "err10": err10,
                     "max10": max(err10.values()), "err4": err4, "max4": max(err4.values()),
                     "rmse": (rI, rE)})
        print(f"σ={noise:.2f} (k={sk}) | θ̂(10) máx={max(err10.values()):6.2f}%  "
              f"θ̂(4 pesos) máx={max(err4.values()):6.2f}% | ctrl RMSE I={rI:.3e} E={rE:.3e}")
        print("   err10 por param: " + "  ".join(f"{k}={err10[k]:.2f}%" for k in ALL_P))

    _plot_err(rows)
    _plot_ctrl(rows, rI0, rE0)
    _save_json(rows, rI0, rE0, true)
    print(f"\nFiguras: {OUT_FIG_ERR}  |  {OUT_FIG_CTRL}")
    print(f"Resultados: {OUT_JSON}")
    return rows


def _plot_err(rows):
    import matplotlib.pyplot as plt
    ns = [r["noise"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    # (a) error por cada uno de los 10 parametros.
    cmap = plt.get_cmap("tab10")
    for j, k in enumerate(ALL_P):
        ax[0].plot(ns, [max(r["err10"][k], 1e-3) for r in rows], "o-",
                   color=cmap(j % 10), label=k, lw=1.4, ms=4)
    ax[0].set_yscale("log"); ax[0].set_xlabel("noise_std σ"); ax[0].set_ylabel("error θ̂ (%)")
    ax[0].set_title("Error por parámetro vs ruido (10 params)")
    ax[0].grid(True, alpha=0.3); ax[0].legend(fontsize=7, ncol=2)
    # (b) 10 params (máx) vs 4 pesos (máx).
    ax[1].plot(ns, [r["max10"] for r in rows], "o-", color="#1f4e79", label="10 params (máx)")
    ax[1].plot(ns, [r["max4"] for r in rows], "s--", color="#d62728", label="solo 4 pesos (máx)")
    ax[1].set_xlabel("noise_std σ"); ax[1].set_ylabel("error θ̂ máx (%)")
    ax[1].set_title("Identificación completa vs solo-pesos"); ax[1].grid(True, alpha=0.3); ax[1].legend(fontsize=8)
    fig.tight_layout(); OUT_FIG_ERR.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG_ERR, dpi=120); plt.close(fig)


def _plot_ctrl(rows, rI0, rE0):
    import matplotlib.pyplot as plt
    ns = [r["noise"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(ns, [r["rmse"][0] for r in rows], "o-", color="#1f4e79", label="RMSE I (θ̂ 10 params)")
    ax.plot(ns, [r["rmse"][1] for r in rows], "s-", color="#d62728", label="RMSE E (θ̂ 10 params)")
    ax.axhline(rI0, ls=":", color="#1f4e79", lw=1, label="RMSE I ideal")
    ax.axhline(rE0, ls=":", color="#d62728", lw=1, label="RMSE E ideal")
    ax.set_xlabel("noise_std σ"); ax.set_ylabel("RMSE seguimiento (lazo cerrado)")
    ax.set_title("Propagación al control del θ̂ ruidoso"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT_FIG_CTRL, dpi=120); plt.close(fig)


def _save_json(rows, rI0, rE0, true):
    out = {"true": true, "ideal_rmse": [rI0, rE0], "levels": []}
    for r in rows:
        out["levels"].append({
            "noise": r["noise"], "k": r["k"], "p_hat": r["p_hat"],
            "err10": r["err10"], "max10": r["max10"],
            "err4": r["err4"], "max4": r["max4"], "rmse": list(r["rmse"]),
        })
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
