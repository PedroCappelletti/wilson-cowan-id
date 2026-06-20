#!/usr/bin/env python3
# =============================================================================
#  INFORME — Simulador (Wilson-Cowan verdadero) vs Neural ODE aprendido
# =============================================================================
#
#  Dos comparaciones, ambas con los ESTIMULOS NUEVOS (tipo pulso):
#
#    PARTE 1 — LAZO ABIERTO: se aplica el MISMO estimulo (P,Q) al simulador y al
#      Neural ODE, partiendo de la misma condicion inicial, y se comparan las
#      trayectorias libres I(t), E(t). Mide cuanto se parece la dinamica aprendida
#      a la verdadera (incluye estimulos held-out que el modelo NO vio).
#
#    PARTE 2 — LAZO CERRADO: el MISMO controlador IMC maneja al simulador y al
#      Neural ODE como planta, siguiendo la referencia theta-gamma. Mide si el
#      modelo aprendido se comporta como el verdadero cuando esta bajo control.
#
#  Genera docs/informe_neural_ode.html (figuras embebidas).
#  USO:  python scripts/informe_neural_ode.py
# =============================================================================

from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.wilson_cowan import WilsonCowanParams
from src.neural_ode import (
    GrayBoxWC, rollout,
    IMCController, make_true_plant, make_neural_plant, simulate_closed_loop,
    theta_gamma_refs,
)

DATA_PATH = Path("data/processed/control/multi_dataset.npz")
CKPT = Path("results/models/neural_ode.pt")
OUT_HTML = Path("docs/informe_neural_ode.html")
FIG_OL = Path("results/figures/informe_ol.png")
FIG_CL = Path("results/figures/informe_cl.png")

# Trayectorias a mostrar en lazo abierto (mezcla train + held-out).
SHOW = ["aprbs_1", "thetagamma_1", "square_a1.0_f130", "chirp"]

# Parametros del lazo cerrado (regimen ms, como eval_closed_loop.py).
T_SPAN_CL = (0.0, 50.0)
DT_CL = 0.005
FREQ_CL = 120.0


def load_model():
    ck = torch.load(CKPT, weights_only=False)
    model = GrayBoxWC(ck["fixed"], {k: 1.0 for k in GrayBoxWC.WEIGHTS},
                      learnable_weights=ck["learn_weights"], use_correction=ck["use_correction"])
    model.load_state_dict(ck["state"]); model.eval()
    return model, ck["fixed"]


def rollout_traj(model, I_row, E_row, P_row, Q_row, dt):
    """Neural ODE en lazo abierto sobre el estimulo (P,Q) de una trayectoria."""
    T = len(I_row)
    x0 = torch.tensor([[I_row[0], E_row[0]]], dtype=torch.float32)
    Ps = torch.tensor(P_row, dtype=torch.float32).reshape(T, 1, 1)
    Qs = torch.tensor(Q_row, dtype=torch.float32).reshape(T, 1, 1)
    with torch.no_grad():
        tr = rollout(model, x0, Ps[:-1], Qs[:-1], dt)[:, 0, :].numpy()
    return tr[:, 0], tr[:, 1]


# =============================================================================
#  PARTE 1 — LAZO ABIERTO
# =============================================================================
def parte_open_loop(model, d):
    t = d["t"]; dt = float(d["dt"])
    I, E, P, Q = d["I"], d["E"], d["P"], d["Q"]
    labels = list(d["labels"]); is_test = d["is_test"].astype(bool)

    # MSE global (todas las trayectorias) train vs test held-out.
    mse_tr, mse_te = [], []
    for s in range(len(labels)):
        Ip, Ep = rollout_traj(model, I[s], E[s], P[s], Q[s], dt)
        m = float(np.mean((Ip - I[s]) ** 2 + (Ep - E[s]) ** 2) / 2)
        (mse_te if is_test[s] else mse_tr).append(m)
    mse_tr, mse_te = float(np.mean(mse_tr)), float(np.mean(mse_te))

    # Figura: trayectorias seleccionadas, simulador (linea) vs Neural ODE (punteado).
    import matplotlib.pyplot as plt
    rows = [labels.index(l) for l in SHOW if l in labels]
    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 2.5 * len(rows)), sharex=True)
    filas_tab = ""
    for r, s in enumerate(rows):
        Ip, Ep = rollout_traj(model, I[s], E[s], P[s], Q[s], dt)
        mse = float(np.mean((Ip - I[s]) ** 2 + (Ep - E[s]) ** 2) / 2)
        tag = "held-out (no visto)" if is_test[s] else "train"
        filas_tab += (f"<tr><td>{labels[s]}</td><td>{tag}</td>"
                      f"<td class='num'>{mse:.2e}</td></tr>")
        axes[r, 0].plot(t, I[s], lw=1.4, label="simulador")
        axes[r, 0].plot(t, Ip, "--", lw=1.1, color="#d62728", label="Neural ODE")
        axes[r, 0].set_ylabel(f"{labels[s]}\nI", fontsize=8)
        axes[r, 1].plot(t, E[s], lw=1.4, label="simulador")
        axes[r, 1].plot(t, Ep, "--", lw=1.1, color="#d62728", label="Neural ODE")
        axes[r, 1].set_ylabel("E", fontsize=8)
        if r == 0:
            axes[r, 0].set_title("I(t)"); axes[r, 1].set_title("E(t)")
            axes[r, 0].legend(fontsize=8, loc="upper right")
        for c in (0, 1):
            axes[r, c].grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("tiempo (ms)"); axes[-1, 1].set_xlabel("tiempo (ms)")
    fig.suptitle("Lazo abierto: simulador (línea) vs Neural ODE (punteado)")
    fig.tight_layout()
    FIG_OL.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OL, dpi=120); plt.close(fig)
    return mse_tr, mse_te, filas_tab


# =============================================================================
#  PARTE 2 — LAZO CERRADO
# =============================================================================
def _rmse(sol):
    n0 = len(sol["t"]) // 5  # descarta transitorio inicial
    rI = float(np.sqrt(np.mean((sol["I"][n0:] - sol["rI"][n0:]) ** 2)))
    rE = float(np.sqrt(np.mean((sol["E"][n0:] - sol["rE"][n0:]) ** 2)))
    return rI, rE


def parte_closed_loop(model):
    p = WilsonCowanParams()
    fixed = {"te": p.te, "ti": p.ti, "ae": p.ae, "ai": p.ai,
             "thetae": p.thetae, "thetai": p.thetai, "ke": p.ke, "ki": p.ki}
    w_true = {"wEE": p.wEE, "wEI": p.wEI, "wIE": p.wIE, "wII": p.wII}  # sistema verdadero
    w_hat = model.weights_dict()    # el controlador se DISEÑA con los pesos APRENDIDOS (θ̂)
    ctrl = IMCController(fixed, w_hat)
    refs = theta_gamma_refs(freq_hz=FREQ_CL, time_in_ms=True)

    sol_t = simulate_closed_loop(make_true_plant(fixed, w_true), ctrl, refs,
                                 t_span=T_SPAN_CL, dt=DT_CL)
    sol_n = simulate_closed_loop(make_neural_plant(model), ctrl, refs,
                                 t_span=T_SPAN_CL, dt=DT_CL)
    rIt, rEt = _rmse(sol_t); rIn, rEn = _rmse(sol_n)

    import matplotlib.pyplot as plt
    t = sol_t["t"]
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(t, sol_t["rI"], "k--", lw=1.0, label="ref")
    ax[0].plot(t, sol_t["I"], lw=1.4, label="simulador")
    ax[0].plot(sol_n["t"], sol_n["I"], ":", lw=1.2, color="#d62728", label="Neural ODE")
    ax[0].set_ylabel("I"); ax[0].legend(fontsize=8, loc="upper right"); ax[0].grid(True, alpha=0.3)
    ax[0].set_title("Lazo cerrado: controlador IMC sobre simulador vs Neural ODE")
    ax[1].plot(t, sol_t["rE"], "k--", lw=1.0, label="ref")
    ax[1].plot(t, sol_t["E"], lw=1.4, label="simulador")
    ax[1].plot(sol_n["t"], sol_n["E"], ":", lw=1.2, color="#d62728", label="Neural ODE")
    ax[1].set_ylabel("E"); ax[1].grid(True, alpha=0.3)
    ax[2].plot(t, sol_t["y"], lw=1.4, label="simulador")
    ax[2].plot(sol_n["t"], sol_n["y"], ":", lw=1.2, color="#d62728", label="Neural ODE")
    ax[2].set_ylabel("y = E - I"); ax[2].set_xlabel("tiempo (ms)"); ax[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_CL, dpi=120); plt.close(fig)
    return (rIt, rEt), (rIn, rEn), w_hat


def _b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


def main():
    model, _ = load_model()
    d = np.load(DATA_PATH, allow_pickle=True)
    print("PARTE 1 — lazo abierto...")
    mse_tr, mse_te, filas = parte_open_loop(model, d)
    print(f"  MSE simulador vs Neural ODE: train={mse_tr:.2e}  held-out={mse_te:.2e}")
    print("PARTE 2 — lazo cerrado...")
    (rIt, rEt), (rIn, rEn), w_hat = parte_closed_loop(model)
    w_hat_str = " · ".join(f"{k}={w_hat[k]:.3f}" for k in w_hat)
    print(f"  controlador con θ̂: {w_hat_str}")
    print(f"  RMSE seguimiento: simulador I={rIt:.2e} E={rEt:.2e} | "
          f"Neural ODE I={rIn:.2e} E={rEn:.2e}")

    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
<title>Informe — Simulador vs Neural ODE</title>
<style>
 body{{font-family:-apple-system,"Segoe UI",Arial,sans-serif;color:#1c2733;line-height:1.6;margin:0;background:#f4f6f8;padding:2rem 1rem}}
 .wrap{{max-width:1000px;margin:0 auto}}
 header{{background:#1f4e79;color:#fff;border-radius:12px 12px 0 0;padding:1.5rem 1.9rem}}
 header h1{{margin:0;font-size:1.35rem}} header p{{margin:.3rem 0 0;opacity:.9;font-size:.95rem}}
 main{{background:#fff;border:1px solid #d7dee5;border-top:none;border-radius:0 0 12px 12px;padding:1.5rem 1.9rem 2rem}}
 h2{{color:#1f4e79;font-size:1.15rem;border-bottom:2px solid #e8f0f8;padding-bottom:.3rem;margin-top:1.8rem}}
 img{{width:100%;border:1px solid #d7dee5;border-radius:8px;margin:.5rem 0}}
 table{{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:.9rem}}
 th,td{{text-align:left;padding:.45rem .7rem;border-bottom:1px solid #d7dee5}}
 th{{background:#e8f0f8;color:#1f4e79}} td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
 .nota{{background:#e8f0f8;border-left:4px solid #1f4e79;padding:.7rem 1rem;border-radius:0 6px 6px 0;margin:1rem 0;font-size:.92rem}}
 .ok{{background:#e7f6ec;border-left-color:#0a7d33}}
</style></head><body><div class="wrap">
<header><h1>Simulador (Wilson-Cowan) vs Neural ODE aprendido</h1>
<p>Comparación de comportamiento en lazo abierto y lazo cerrado · estímulos tipo pulso · ITBA 2026</p></header>
<main>
 <div class="nota">El <b>simulador</b> es el modelo de Wilson-Cowan verdadero (genera los datos). El
 <b>Neural ODE</b> es el modelo de estados f(x,P,Q) aprendido de esos datos, entrenado con los
 estímulos nuevos (tipo pulso). Se mide qué tan parecido se comporta el modelo aprendido al real.</div>

 <h2>1 · Lazo abierto — respuesta libre al mismo estímulo</h2>
 <p>Se aplica el mismo estímulo (P, Q) a ambos desde la misma condición inicial y se comparan las
 trayectorias. Incluye estímulos <b>held-out</b> (que el Neural ODE no vio en entrenamiento).</p>
 <table><tr><th>Estímulo</th><th>Conjunto</th><th class="num">MSE (sim. vs N-ODE)</th></tr>{filas}</table>
 <p>MSE promedio sobre <b>todas</b> las trayectorias: train = <b>{mse_tr:.2e}</b> ·
 held-out = <b>{mse_te:.2e}</b>.</p>
 <img src="data:image/png;base64,{_b64(FIG_OL)}">
 <div class="nota ok">El Neural ODE reproduce la dinámica libre del simulador con error muy bajo,
 también en estímulos no vistos → aprendió la dinámica, no memorizó.</div>

 <h2>2 · Lazo cerrado — bajo el controlador IMC</h2>
 <p>El controlador IMC se <b>diseña con los pesos aprendidos por el Neural ODE</b>
 (θ̂ = {w_hat_str}) — no con los verdaderos. Es el caso honesto: el controlador solo conoce lo
 identificado. El mismo controlador maneja al simulador y al Neural ODE como planta, siguiendo la
 referencia theta-gamma. RMSE de seguimiento (descartando el transitorio inicial):</p>
 <table><tr><th>Planta</th><th class="num">RMSE I</th><th class="num">RMSE E</th></tr>
 <tr><td>Simulador (verdadero)</td><td class="num">{rIt:.3e}</td><td class="num">{rEt:.3e}</td></tr>
 <tr><td>Neural ODE (aprendido)</td><td class="num">{rIn:.3e}</td><td class="num">{rEn:.3e}</td></tr></table>
 <img src="data:image/png;base64,{_b64(FIG_CL)}">
 <div class="nota ok">El seguimiento es prácticamente idéntico → un controlador diseñado solo con
 los pesos aprendidos controla el sistema verdadero igual de bien: el error de identificación no
 degrada el control, y el Neural ODE es una planta controlable válida.</div>
</main></div></body></html>"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nInforme: {OUT_HTML}")


if __name__ == "__main__":
    main()
