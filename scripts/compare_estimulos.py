#!/usr/bin/env python3
# =============================================================================
#  COMPARAR ESTIMULOS POR CALIDAD DE IDENTIFICACION
# =============================================================================
#
#  Idea: el estimulo NO se juzga por que tan lindo queda el retrato de fase, sino
#  por CUAN BIEN deja identificar los parametros. Este script, para cada familia
#  de estimulo, arma 4 trayectorias (P,Q decorrelacionados para romper la
#  degeneracion wIE/wII), corre la identificacion PINN multi-trayectoria y mide
#  el ERROR PARAMETRICO de los 4 pesos (sobre todo wII, que es el fragil).
#
#  Es la prueba que realmente importa para elegir con que estimulos generar los
#  datasets. Complementa a demo_estimulos.py (que solo mira la cobertura visual).
#
#  USO:  python scripts/compare_estimulos.py
#  OJO: corre una identificacion por familia -> tarda varios minutos cada una.
#       Elegi cuales correr en FAMILIAS (abajo). QUICK=1 para una corrida rapida
#       de humo (pocos epochs, resultados NO validos, solo para probar que anda).
# =============================================================================

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.wilson_cowan import (  # noqa: E402
    WilsonCowanParams,
    box_pulse, chirp_pulse,
    aprbs_pulse, theta_gamma_pulse, square_wave_pulse, prbs_pulse, poisson_pulse,
)
from src.data import generate_dataset  # noqa: E402
from src.pinn.multitraj import MultiTrajPINN, MultiTrajectoryTrainer  # noqa: E402
from src.pinn.train import TrainConfig  # noqa: E402

# #############################################################################
# ##   ZONA EDITABLE                                                         ##
# #############################################################################

PARAMS = WilsonCowanParams()           # parametros VERDADEROS (lo que hay que recuperar)
T_SPAN = (0.0, 600.0)
N_PUNTOS = 24001                        # dt = 0.025 (mas grueso = mas rapido para iterar)
I0, E0 = 0.0, 0.0
T_ON, T_OFF = 50.0, 550.0               # ventana de excitacion (comun a todas)

# Banda de frecuencias util (en unidades de la simulacion, NO en Hz biologicos:
# ver nota en model.py). El sistema reacciona alrededor de ~0.005-0.05.
QUICK = os.environ.get("QUICK") == "1"  # corrida de humo (resultados no validos)

# --- Cada familia define 4 trayectorias (P,Q). Variar amplitud/timing/semilla
#     entre ellas (y P vs Q con semillas distintas) es lo que decorrelaciona E e I
#     y hace identificable a wII. Comentar las que no quieras correr.
def _box():
    return [
        (box_pulse(0.8, 100, 400), box_pulse(0.6, 200, 500)),
        (box_pulse(1.2, 100, 300), box_pulse(0.3, 150, 450)),
        (box_pulse(0.5, 100, 500), box_pulse(1.0, 100, 400)),
        (box_pulse(0.9, 150, 350), box_pulse(0.5, 100, 300)),
    ]

def _aprbs():
    # mk(amplitud_max, piso, dwell_min, dwell_max, seed). Estrategia: asimetria
    # fuerte P vs Q entre trayectorias (incl. una con Q FUERTE, clave para wII),
    # amplitudes altas y piso > 0 para no apagar la excitacion. P y Q con semillas
    # distintas para decorrelacionar E e I.
    mk = lambda a, amin, dmin, dmax, s: aprbs_pulse(a, T_ON, T_OFF, dmin, dmax, seed=s, amp_min=amin)
    return [
        (mk(0.9, 0.2, 20, 60, 10), mk(0.7, 0.1, 25, 70, 20)),   # balanceado
        (mk(1.3, 0.3, 15, 50, 11), mk(0.4, 0.0, 30, 80, 21)),   # P fuerte, Q debil
        (mk(0.5, 0.0, 25, 70, 12), mk(1.3, 0.4, 18, 55, 22)),   # P debil, Q FUERTE (clave wII)
        (mk(1.0, 0.2, 20, 60, 13), mk(0.6, 0.1, 22, 65, 23)),   # otra combinacion
    ]

def _theta_gamma():
    # Misma estrategia de asimetria que el box/APRBS, con amplitudes mas altas y
    # frecuencias distintas en P vs Q (asi no quedan en fase = decorrelacionadas).
    mk = lambda a, fg, ft, duty: theta_gamma_pulse(a, fg, ft, T_ON, T_OFF, duty)
    return [
        (mk(1.0, 0.05, 0.008, 0.5), mk(0.8, 0.04, 0.007, 0.5)),   # balanceado
        (mk(1.4, 0.06, 0.010, 0.4), mk(0.4, 0.05, 0.008, 0.5)),   # P fuerte, Q debil
        (mk(0.5, 0.04, 0.007, 0.6), mk(1.4, 0.06, 0.009, 0.4)),   # P debil, Q FUERTE (clave wII)
        (mk(1.1, 0.05, 0.009, 0.5), mk(0.7, 0.045, 0.008, 0.5)),  # otra combinacion
    ]

def _square():
    mk = lambda a, f, duty: square_wave_pulse(a, f, T_ON, T_OFF, duty)
    return [
        (mk(0.8, 0.03, 0.5), mk(0.6, 0.02, 0.5)),
        (mk(1.2, 0.04, 0.4), mk(0.3, 0.015, 0.5)),
        (mk(0.5, 0.02, 0.6), mk(1.0, 0.035, 0.4)),
        (mk(0.9, 0.025, 0.5), mk(0.5, 0.03, 0.5)),
    ]

def _prbs():
    mk = lambda a, bp, s: prbs_pulse(a, T_ON, T_OFF, bp, seed=s)
    return [
        (mk(0.8, 20, 10), mk(0.6, 25, 20)),
        (mk(1.2, 15, 11), mk(0.3, 30, 21)),
        (mk(0.5, 25, 12), mk(1.0, 18, 22)),
        (mk(0.9, 20, 13), mk(0.5, 22, 23)),
    ]

def _poisson():
    mk = lambda a, r, pw, s: poisson_pulse(a, r, T_ON, T_OFF, pw, seed=s)
    return [
        (mk(0.8, 0.05, 8, 10), mk(0.6, 0.04, 8, 20)),
        (mk(1.2, 0.06, 6, 11), mk(0.3, 0.03, 10, 21)),
        (mk(0.5, 0.04, 10, 12), mk(1.0, 0.06, 6, 22)),
        (mk(0.9, 0.05, 8, 13), mk(0.5, 0.045, 8, 23)),
    ]

def _chirp():
    mk = lambda a, f0, f1: chirp_pulse(a, f0, f1, T_ON, T_OFF)
    return [
        (mk(0.4, 0.005, 0.05), mk(0.3, 0.005, 0.04)),
        (mk(0.6, 0.004, 0.06), mk(0.2, 0.006, 0.03)),
        (mk(0.3, 0.006, 0.04), mk(0.5, 0.005, 0.05)),
        (mk(0.5, 0.005, 0.05), mk(0.3, 0.004, 0.04)),
    ]

# Las familias a comparar (nombre -> builder). Comenta las que no quieras correr.
FAMILIAS = {
    "Box (baseline)": _box,
    "APRBS ⭐": _aprbs,
    "Theta-gamma ⭐": _theta_gamma,
    # --- descomentar para comparar tambien estas (mas lento) ---
    # "Onda cuadrada": _square,
    # "PRBS": _prbs,
    # "Tren de Poisson": _poisson,
    # "Chirp": _chirp,
}

HIDDEN_DIM, N_LAYERS = 64, 4
N_FOURIER, FOURIER_SCALE = 128, 6.0

CONFIG = TrainConfig(
    epochs=300 if QUICK else 15_000, lr=1e-3, lr_weights=0.05,
    w_data=1.0, w_physics=1.0, w_ic=1.0,
    batch_size=4_000, n_collocation=2_000,
    val_fraction=0.0, physics_warmup=100 if QUICK else 2000,
    freeze_net_after_warmup=True,
    device="cpu", log_every=500, early_stop=True,
)

# #############################################################################
# ##   FIN ZONA EDITABLE                                                     ##
# #############################################################################

OUT = Path("results")
FIG_PATH = OUT / "figures" / "compare_estimulos.png"
HTML_PATH = OUT / "reporte_compare_estimulos.html"
PESOS = ("wEE", "wEI", "wIE", "wII")


def _identificar(trajs) -> dict:
    """Genera las 4 trayectorias de una familia e identifica los 4 pesos."""
    if QUICK:
        n_eval = 4001
    else:
        n_eval = N_PUNTOS
    datasets = []
    for P, Q in trajs:
        ds = generate_dataset(params=PARAMS, P=P, Q=Q, I0=I0, E0=E0,
                              t_span=T_SPAN, n_eval=n_eval, noise_std=0.0, seed=0)
        datasets.append(ds)

    fixed = {
        "te": PARAMS.te, "ti": PARAMS.ti, "ae": PARAMS.ae, "ai": PARAMS.ai,
        "thetae": PARAMS.thetae, "thetai": PARAMS.thetai, "ke": PARAMS.ke, "ki": PARAMS.ki,
    }
    model = MultiTrajPINN(
        n_traj=len(datasets), t_min=T_SPAN[0], t_max=T_SPAN[1],
        hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
        n_fourier=N_FOURIER, fourier_scale=FOURIER_SCALE,
        w_init={k: 1.0 for k in PESOS},   # arranque ignorante
    )
    trainer = MultiTrajectoryTrainer(model, CONFIG, fixed)
    hist = trainer.train(datasets)
    w_pred = model.weights_dict()
    return {"w_pred": w_pred, "hist": hist}


def main() -> None:
    w_true = {k: getattr(PARAMS, k) for k in PESOS}
    resultados = {}

    for nombre, builder in FAMILIAS.items():
        print(f"\n{'='*70}\n  IDENTIFICANDO con: {nombre}\n{'='*70}")
        torch.manual_seed(0); np.random.seed(0)
        out = _identificar(builder())
        w_pred = out["w_pred"]
        err = {k: 100.0 * abs(w_pred[k] - w_true[k]) / abs(w_true[k]) for k in PESOS}
        resultados[nombre] = {"w_pred": w_pred, "err": err, "max_err": max(err.values())}
        print(f"  -> {nombre}: " + " ".join(f"{k}={w_pred[k]:.3f}({err[k]:.1f}%)" for k in PESOS)
              + f" | max_err={max(err.values()):.1f}%")

    # --- Tabla por consola, ordenada por max_err (mejor estimulo primero).
    orden = sorted(resultados, key=lambda n: resultados[n]["max_err"])
    print(f"\n{'='*70}\n  RANKING (menor error parametrico = mejor estimulo)\n{'='*70}")
    print(f"{'familia':20} " + " ".join(f"{k+' err%':>11}" for k in PESOS) + f"{'max err%':>11}")
    for n in orden:
        r = resultados[n]
        print(f"{n:20} " + " ".join(f"{r['err'][k]:10.2f}%" for k in PESOS)
              + f"{r['max_err']:10.2f}%")

    _plot(resultados, orden, w_true)
    _build_html(resultados, orden, w_true)
    print(f"\nFigura: {FIG_PATH}\nReporte HTML: {HTML_PATH}")


def _plot(resultados, orden, w_true):
    import matplotlib.pyplot as plt
    x = np.arange(len(orden))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(orden)), 4.5))
    for j, k in enumerate(PESOS):
        ax.bar(x + (j - 1.5) * width, [resultados[n]["err"][k] for n in orden],
               width, label=k)
    ax.axhline(10, color="red", ls=":", lw=1, label="10% (criterio)")
    ax.set_xticks(x); ax.set_xticklabels(orden, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("error parametrico (%)")
    ax.set_title("Error de identificacion por familia de estimulo (menor = mejor)")
    ax.legend(ncol=5, fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=120); plt.close(fig)


def _build_html(resultados, orden, w_true):
    img = base64.b64encode(FIG_PATH.read_bytes()).decode()
    filas = ""
    for n in orden:
        r = resultados[n]
        celdas = "".join(
            f"<td class='num'>{r['w_pred'][k]:.3f}</td>"
            f"<td class='num {'bad' if r['err'][k] >= 10 else 'good'}'>{r['err'][k]:.1f}%</td>"
            for k in PESOS)
        filas += f"<tr><td>{n}</td>{celdas}<td class='num'><b>{r['max_err']:.1f}%</b></td></tr>"
    th = "".join(f"<th class='num'>{k}</th><th class='num'>err%</th>" for k in PESOS)
    quick = "<div class='nota' style='border-color:#b00'>CORRIDA QUICK: pocos epochs, resultados NO validos.</div>" if QUICK else ""
    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
<title>Comparacion de estimulos por identificacion</title>
<style>
 body{{font-family:-apple-system,"Segoe UI",Arial,sans-serif;color:#1c2733;line-height:1.6;margin:0;background:#f4f6f8;padding:2rem 1rem}}
 .wrap{{max-width:1000px;margin:0 auto}}
 header{{background:#1f4e79;color:#fff;border-radius:12px 12px 0 0;padding:1.5rem 1.9rem}}
 header h1{{margin:0;font-size:1.3rem}}
 main{{background:#fff;border:1px solid #d7dee5;border-top:none;border-radius:0 0 12px 12px;padding:1.5rem 1.9rem 2rem}}
 h2{{color:#1f4e79;font-size:1.1rem;border-bottom:2px solid #e8f0f8;padding-bottom:.3rem}}
 img{{width:100%;border:1px solid #d7dee5;border-radius:8px;margin:.5rem 0}}
 table{{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:.85rem}}
 th,td{{text-align:left;padding:.4rem .6rem;border-bottom:1px solid #d7dee5}}
 th{{background:#e8f0f8;color:#1f4e79}} td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
 td.good{{color:#0a7d33}} td.bad{{color:#b00020}}
 .nota{{background:#e8f0f8;border-left:4px solid #1f4e79;padding:.7rem 1rem;border-radius:0 6px 6px 0;margin:1rem 0;font-size:.92rem}}
</style></head><body><div class="wrap">
<header><h1>Comparacion de estimulos por calidad de identificacion</h1></header>
<main>
 {quick}
 <div class="nota">El estimulo se juzga por cuan bien deja identificar los parametros (no por el
 retrato de fase). Verdaderos: wEE={w_true['wEE']}, wEI={w_true['wEI']}, wIE={w_true['wIE']}, wII={w_true['wII']}.
 Identificacion PINN multi-trayectoria (4 trayectorias, theta compartido, arranque ignorante en 1.0).
 Ordenado de menor a mayor error maximo.</div>
 <h2>Ranking (menor error = mejor estimulo)</h2>
 <table><tr><th>Familia</th>{th}<th class="num">max err%</th></tr>{filas}</table>
 <h2>Error parametrico por familia</h2>
 <img src="data:image/png;base64,{img}">
</main></div></body></html>"""
    HTML_PATH.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
