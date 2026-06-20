#!/usr/bin/env python3
# =============================================================================
#  ROBUSTEZ BAJO RUIDO (OE3) — ¿el ruido en la identificación degrada el control?
# =============================================================================
#
#  Para cada nivel de ruido de observación:
#    1. Genera el dataset de control (estímulos nuevos) CON ruido.
#    2. Identifica el Neural ODE sobre esos datos ruidosos -> θ̂(ruido).
#    3. Construye el controlador IMC con θ̂(ruido).
#    4. Cierra el lazo sobre la planta VERDADERA (sistema real) y mide el RMSE de
#       seguimiento de la referencia theta-gamma.
#
#  Compara contra el baseline ideal (controlador con los pesos reales). Mide cómo
#  el error de identificación inducido por el ruido se propaga al control.
#
#  Niveles calibrados a SNR de LFP: 0 / 0.01 (~30 dB) / 0.05 (~20 dB) / 0.10 (~14 dB).
#  Genera docs/informe_ruido.html.  USO: python -u scripts/noise_robustness.py
# =============================================================================

from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.wilson_cowan import WilsonCowanParams
from src.data import generate_dataset
from src.neural_ode import (
    GrayBoxWC, rollout, IMCController, make_true_plant, simulate_closed_loop,
    theta_gamma_refs,
)
import scripts.gen_multi_dataset as G          # build_scenarios, PARAMS, T_SPAN, N_EVAL, I0, E0, SEED
from scripts.train_neural_ode import make_windows, WINDOW, IDENT, W_INIT

NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10]
EPOCHS = 2000
LBFGS_STEPS = 40
LR_W = 5e-2

T_SPAN_CL = (0.0, 50.0)
DT_CL = 0.005
FREQ_CL = 120.0

W_TRUE = {"wEE": 6.4, "wEI": 4.8, "wIE": 6.0, "wII": 1.2}
OUT_HTML = Path("docs/informe_ruido.html")
FIG = Path("results/figures/ruido.png")


def _fixed():
    p = WilsonCowanParams()
    return {"te": p.te, "ti": p.ti, "ae": p.ae, "ai": p.ai,
            "thetae": p.thetae, "thetai": p.thetai, "ke": p.ke, "ki": p.ki}


def gen_noisy(noise):
    """Dataset de control con ruido. Devuelve arrays + dt y máscara de test."""
    I, E, P, Q, is_test = [], [], [], [], []
    dt = None
    for label, Pf, Qf, test in G.build_scenarios():
        ds = generate_dataset(params=G.PARAMS, P=Pf, Q=Qf, I0=G.I0, E0=G.E0,
                              t_span=G.T_SPAN, n_eval=G.N_EVAL, noise_std=noise, seed=G.SEED)
        I.append(ds["I"]); E.append(ds["E"]); P.append(ds["P"]); Q.append(ds["Q"])
        is_test.append(test); dt = float(ds["t"][1] - ds["t"][0])
    return (np.stack(I), np.stack(E), np.stack(P), np.stack(Q),
            np.asarray(is_test, dtype=bool), dt)


def identify(I, E, P, Q, is_test, dt, fixed):
    """Identifica θ con el Neural ODE (multiple shooting) sobre datos (posiblemente ruidosos)."""
    Itr, Etr, Ptr, Qtr = I[~is_test], E[~is_test], P[~is_test], Q[~is_test]
    model = GrayBoxWC(fixed, {k: W_INIT for k in IDENT}, learnable_weights=True, use_correction=False)
    x0, Pw, Qw, tgt = make_windows(Itr, Etr, Ptr, Qtr, WINDOW)

    opt = torch.optim.Adam([{"params": [model.raw_w], "lr": LR_W}])
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = ((rollout(model, x0, Pw, Qw, dt) - tgt) ** 2).mean()
        loss.backward(); opt.step()

    opt2 = torch.optim.LBFGS([model.raw_w], lr=1.0, max_iter=20, line_search_fn="strong_wolfe")
    def closure():
        opt2.zero_grad()
        l = ((rollout(model, x0, Pw, Qw, dt) - tgt) ** 2).mean()
        l.backward(); return l
    for _ in range(LBFGS_STEPS):
        opt2.step(closure)
    return model.weights_dict()


def _rmse(sol):
    n0 = len(sol["t"]) // 5
    rI = float(np.sqrt(np.mean((sol["I"][n0:] - sol["rI"][n0:]) ** 2)))
    rE = float(np.sqrt(np.mean((sol["E"][n0:] - sol["rE"][n0:]) ** 2)))
    return rI, rE


def main():
    fixed = _fixed()
    plant_true = make_true_plant(fixed, W_TRUE)
    refs = theta_gamma_refs(freq_hz=FREQ_CL, time_in_ms=True)

    # Baseline ideal: controlador con los pesos REALES sobre la planta verdadera.
    ctrl_ideal = IMCController(fixed, W_TRUE)
    rI0, rE0 = _rmse(simulate_closed_loop(plant_true, ctrl_ideal, refs, t_span=T_SPAN_CL, dt=DT_CL))
    print(f"baseline ideal (pesos reales): RMSE I={rI0:.3e} E={rE0:.3e}")

    rows = []
    for noise in NOISE_LEVELS:
        torch.manual_seed(0); np.random.seed(0)
        data = gen_noisy(noise)
        w_hat = identify(*data, fixed)
        errs = {k: 100.0 * abs(w_hat[k] - W_TRUE[k]) / W_TRUE[k] for k in W_TRUE}
        ctrl = IMCController(fixed, w_hat)
        rI, rE = _rmse(simulate_closed_loop(plant_true, ctrl, refs, t_span=T_SPAN_CL, dt=DT_CL))
        rows.append({"noise": noise, "w_hat": w_hat, "errs": errs,
                     "max_err": max(errs.values()), "rmse": (rI, rE)})
        print(f"ruido={noise:.2f} | θ̂ max err={max(errs.values()):5.1f}% | "
              f"control RMSE I={rI:.3e} E={rE:.3e} | "
              + " ".join(f"{k}={w_hat[k]:.3f}" for k in W_TRUE))

    _plot(rows, rI0, rE0)
    _html(rows, rI0, rE0)
    print(f"\nInforme: {OUT_HTML}")


def _plot(rows, rI0, rE0):
    import matplotlib.pyplot as plt
    ns = [r["noise"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ns, [r["max_err"] for r in rows], "o-", color="#1f4e79")
    ax[0].set_xlabel("noise_std"); ax[0].set_ylabel("error θ̂ máx (%)")
    ax[0].set_title("Error de identificación vs ruido"); ax[0].grid(True, alpha=0.3)
    ax[1].plot(ns, [r["rmse"][0] for r in rows], "o-", label="RMSE I (θ̂)", color="#1f4e79")
    ax[1].plot(ns, [r["rmse"][1] for r in rows], "s-", label="RMSE E (θ̂)", color="#d62728")
    ax[1].axhline(rI0, ls=":", color="#1f4e79", lw=1, label="RMSE I ideal")
    ax[1].axhline(rE0, ls=":", color="#d62728", lw=1, label="RMSE E ideal")
    ax[1].set_xlabel("noise_std"); ax[1].set_ylabel("RMSE seguimiento")
    ax[1].set_title("Control en lazo cerrado vs ruido"); ax[1].grid(True, alpha=0.3); ax[1].legend(fontsize=8)
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=120); plt.close(fig)


def _html(rows, rI0, rE0):
    img = base64.b64encode(FIG.read_bytes()).decode()
    filas = ""
    for r in rows:
        we = " ".join(f"{k}={r['errs'][k]:.1f}%" for k in W_TRUE)
        filas += (f"<tr><td class='num'>{r['noise']:.2f}</td>"
                  f"<td class='num'>{r['max_err']:.1f}%</td><td>{we}</td>"
                  f"<td class='num'>{r['rmse'][0]:.3e}</td><td class='num'>{r['rmse'][1]:.3e}</td></tr>")
    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
<title>Robustez bajo ruido (OE3)</title>
<style>
 body{{font-family:-apple-system,"Segoe UI",Arial,sans-serif;color:#1c2733;line-height:1.6;margin:0;background:#f4f6f8;padding:2rem 1rem}}
 .wrap{{max-width:1000px;margin:0 auto}}
 header{{background:#1f4e79;color:#fff;border-radius:12px 12px 0 0;padding:1.5rem 1.9rem}}
 header h1{{margin:0;font-size:1.3rem}}
 main{{background:#fff;border:1px solid #d7dee5;border-top:none;border-radius:0 0 12px 12px;padding:1.5rem 1.9rem 2rem}}
 h2{{color:#1f4e79;font-size:1.1rem;border-bottom:2px solid #e8f0f8;padding-bottom:.3rem}}
 img{{width:100%;border:1px solid #d7dee5;border-radius:8px;margin:.5rem 0}}
 table{{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:.85rem}}
 th,td{{text-align:left;padding:.45rem .6rem;border-bottom:1px solid #d7dee5}}
 th{{background:#e8f0f8;color:#1f4e79}} td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
 .nota{{background:#e8f0f8;border-left:4px solid #1f4e79;padding:.7rem 1rem;border-radius:0 6px 6px 0;margin:1rem 0;font-size:.92rem}}
</style></head><body><div class="wrap">
<header><h1>Robustez bajo ruido — validación orientada al control (OE3)</h1></header>
<main>
 <div class="nota">Para cada nivel de ruido de observación se identifica el Neural ODE con datos
 ruidosos (θ̂), se arma el controlador con esos θ̂ y se cierra el lazo sobre la planta <b>verdadera</b>.
 Mide cómo el error de identificación inducido por el ruido degrada el seguimiento. Baseline ideal
 (pesos reales): RMSE I={rI0:.3e}, E={rE0:.3e}.</div>
 <h2>Resultados por nivel de ruido</h2>
 <table><tr><th class="num">noise_std</th><th class="num">θ̂ máx err</th><th>error por peso</th>
 <th class="num">RMSE I</th><th class="num">RMSE E</th></tr>{filas}</table>
 <img src="data:image/png;base64,{img}">
</main></div></body></html>"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
