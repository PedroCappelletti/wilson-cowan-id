#!/usr/bin/env python3
# =============================================================================
#  EXPERIMENTO DEDICADO A wII  (estimulos realizables tipo pulso)
# =============================================================================
#
#  wII es el peso mas dificil de identificar. Aparece SOLO en la entrada
#  inhibitoria  u_i = wIE*E - wII*I + Q - thetai, y por dos motivos cuesta:
#    1) su "huella" en los datos es proporcional a I (sensibilidad ~ -I): si I es
#       chico, los datos no dicen casi nada sobre wII.
#    2) compite con wIE*E en el mismo termino: si E e I van correlacionados, no se
#       pueden separar (degeneracion).
#
#  RECETA: trayectorias que bombeen I FUERTE y por SEPARADO de E, para que la nube
#  de puntos (E,I) llene el plano 2D (si es una recta, wIE y wII no se separan).
#  CLAVE de ESTE sistema: el umbral inhibitorio es alto (thetai=4), asi que Q tiene
#  que ser GRANDE (~3-5, no ~1) para vencerlo y mover I por su cuenta; con Q chico,
#  I queda manejado solo por E y wII no se identifica. Se mezclan trayectorias con Q
#  de 0 a muy grande (distintos ratios E/I) + un theta compartido que las explica todas.
#  Diseno validado con un proxy barato: s2/s1 de la nube (E,I) ~= 0.64 (vs ~0.13 del box).
#
#  Config "buena" (no la liviana de la comparacion, que perjudicaba a wII).
#
#  USO:  python -u scripts/train_wii.py
# =============================================================================

from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.wilson_cowan import (  # noqa: E402
    WilsonCowanParams, zero_input,
    aprbs_pulse, square_wave_pulse, prbs_pulse,
)
from src.data import generate_dataset  # noqa: E402
from src.pinn.multitraj import MultiTrajPINN, MultiTrajectoryTrainer  # noqa: E402
from src.pinn.train import TrainConfig  # noqa: E402

# #############################################################################
# ##   ZONA EDITABLE                                                         ##
# #############################################################################

PARAMS = WilsonCowanParams()           # parametros VERDADEROS (lo que hay que recuperar)
T_SPAN = (0.0, 600.0)
N_PUNTOS = 40001                        # dt = 0.015 (densidad buena para wII)
I0, E0 = 0.0, 0.0
T_ON, T_OFF = 50.0, 550.0

# Las 6 trayectorias (etiqueta, P->E, Q->I). P apagado en las I-dominantes.
def _trayectorias():
    ap = lambda a, amin, s: aprbs_pulse(a, T_ON, T_OFF, 18, 55, seed=s, amp_min=amin)
    sq = lambda a, f, duty: square_wave_pulse(a, f, T_ON, T_OFF, duty)
    return [
        # P fuerte, Q=0 -> ratio E/I "baseline" (I manejado solo por E)
        ("P fuerte Q=0",  ap(1.4, 0.4, 33),   zero_input),
        # Q GRANDE (vence el umbral thetai=4) -> I se mueve independiente de E
        ("Q grande",      ap(1.0, 0.2, 31),   ap(4.0, 1.5, 41)),
        ("Q muy grande",  ap(0.8, 0.2, 34),   ap(5.0, 2.0, 42)),
        ("pulsos Q gde",  sq(1.2, 0.05, 0.5), sq(4.0, 0.03, 0.5)),
        # P=0 + Q muy grande -> AHORA si: I grande con E chico (decorrelacion pura)
        ("P=0 Q muy gde", zero_input,         ap(5.0, 2.0, 45)),
        ("Q intermedio",  ap(1.2, 0.3, 36),   ap(3.0, 1.0, 44)),
    ]

HIDDEN_DIM, N_LAYERS = 64, 4
N_FOURIER, FOURIER_SCALE = 128, 6.0

CONFIG = TrainConfig(
    epochs=25_000, lr=1e-3, lr_weights=0.05,
    w_data=1.0, w_physics=1.0, w_ic=1.0,
    batch_size=6_000, n_collocation=3_000,
    val_fraction=0.0, physics_warmup=3000, freeze_net_after_warmup=True,
    device="cpu", log_every=500, early_stop=True,
)

# #############################################################################
# ##   FIN ZONA EDITABLE                                                     ##
# #############################################################################

OUT = Path("results")
FIG_W = OUT / "figures" / "wii_pesos.png"
HTML_PATH = OUT / "reporte_wii.html"
PESOS = ("wEE", "wEI", "wIE", "wII")


def main() -> None:
    torch.manual_seed(0); np.random.seed(0)
    trays = _trayectorias()
    w_true = {k: getattr(PARAMS, k) for k in PESOS}

    # --- 1) Generar las 6 trayectorias.
    datasets, info = [], []
    for label, P, Q in trays:
        ds = generate_dataset(params=PARAMS, P=P, Q=Q, I0=I0, E0=E0,
                              t_span=T_SPAN, n_eval=N_PUNTOS, noise_std=0.0, seed=0)
        datasets.append(ds)
        # cuanto excita cada poblacion (rango de I y E) -> para ver la decorrelacion
        info.append((label, float(ds["I"].max()), float(ds["E"].max())))
        print(f"  tray '{label:13}': I_max={ds['I'].max():.3f}  E_max={ds['E'].max():.3f}")

    fixed = {
        "te": PARAMS.te, "ti": PARAMS.ti, "ae": PARAMS.ae, "ai": PARAMS.ai,
        "thetae": PARAMS.thetae, "thetai": PARAMS.thetai, "ke": PARAMS.ke, "ki": PARAMS.ki,
    }

    # --- 2) Identificar (theta compartido, arranque ignorante en 1.0).
    model = MultiTrajPINN(
        n_traj=len(datasets), t_min=T_SPAN[0], t_max=T_SPAN[1],
        hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
        n_fourier=N_FOURIER, fourier_scale=FOURIER_SCALE,
        w_init={k: 1.0 for k in PESOS},
    )
    print(f"\n=== Identificacion dedicada a wII: {len(datasets)} trayectorias ===")
    trainer = MultiTrajectoryTrainer(model, CONFIG, fixed)
    hist = trainer.train(datasets)

    # --- 3) Reporte.
    w_pred = model.weights_dict()
    print("\n=== Pesos identificados vs verdaderos ===")
    print(f"{'peso':6} {'verdadero':>10} {'estimado':>10} {'error %':>10}")
    err = {}
    for k in PESOS:
        err[k] = 100.0 * abs(w_pred[k] - w_true[k]) / abs(w_true[k])
        print(f"{k:6} {w_true[k]:10.4f} {w_pred[k]:10.4f} {err[k]:9.2f}%")
    print(f"\n>>> wII error = {err['wII']:.2f}%   (max global = {max(err.values()):.2f}%)")

    _plot_weights(hist, w_true)
    _build_html(w_true, w_pred, err, info, hist)
    print(f"\nReporte HTML: {HTML_PATH}")


def _plot_weights(hist, w_true):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    for k in PESOS:
        line, = ax.plot(hist[k], label=k, lw=1.3)
        ax.axhline(w_true[k], color=line.get_color(), ls=":", lw=1)
    if hist.get("warmup"):
        ax.axvline(hist["warmup"], color="purple", ls=":", lw=1.2, label="fin warmup")
    ax.set_xlabel("epoch"); ax.set_ylabel("valor del peso")
    ax.set_title("Convergencia de los pesos (punteado = verdadero)")
    ax.legend(ncol=5); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    FIG_W.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_W, dpi=120); plt.close(fig)


def _build_html(w_true, w_pred, err, info, hist):
    img = base64.b64encode(FIG_W.read_bytes()).decode()
    filas_w = "".join(
        f"<tr><td>{k}</td><td class='num'>{w_true[k]:.3f}</td>"
        f"<td class='num'>{w_pred[k]:.3f}</td>"
        f"<td class='num {'bad' if err[k] >= 10 else 'good'}'>{err[k]:.2f}%</td></tr>"
        for k in PESOS)
    filas_t = "".join(
        f"<tr><td>{lbl}</td><td class='num'>{imax:.3f}</td><td class='num'>{emax:.3f}</td></tr>"
        for lbl, imax, emax in info)
    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
<title>Experimento dedicado a wII</title>
<style>
 body{{font-family:-apple-system,"Segoe UI",Arial,sans-serif;color:#1c2733;line-height:1.6;margin:0;background:#f4f6f8;padding:2rem 1rem}}
 .wrap{{max-width:980px;margin:0 auto}}
 header{{background:#1f4e79;color:#fff;border-radius:12px 12px 0 0;padding:1.5rem 1.9rem}}
 header h1{{margin:0;font-size:1.3rem}}
 main{{background:#fff;border:1px solid #d7dee5;border-top:none;border-radius:0 0 12px 12px;padding:1.5rem 1.9rem 2rem}}
 h2{{color:#1f4e79;font-size:1.1rem;border-bottom:2px solid #e8f0f8;padding-bottom:.3rem}}
 img{{width:100%;border:1px solid #d7dee5;border-radius:8px;margin:.5rem 0}}
 table{{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:.9rem}}
 th,td{{text-align:left;padding:.45rem .7rem;border-bottom:1px solid #d7dee5}}
 th{{background:#e8f0f8;color:#1f4e79}} td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
 td.good{{color:#0a7d33}} td.bad{{color:#b00020}}
 .nota{{background:#e8f0f8;border-left:4px solid #1f4e79;padding:.7rem 1rem;border-radius:0 6px 6px 0;margin:1rem 0;font-size:.92rem}}
</style></head><body><div class="wrap">
<header><h1>Identificacion dedicada a wII (estimulos tipo pulso)</h1></header>
<main>
 <div class="nota">Estrategia: trayectorias I-dominantes (P apagado/debil, Q fuerte) para que I sea
 grande y decorrelado de E, que es lo que carga informacion sobre wII. 6 trayectorias, theta compartido,
 arranque ignorante en 1.0.</div>
 <h2>Pesos identificados</h2>
 <table><tr><th>Peso</th><th class="num">Verdadero</th><th class="num">Estimado</th><th class="num">Error</th></tr>
 {filas_w}</table>
 <h2>Trayectorias (excitacion de cada poblacion)</h2>
 <table><tr><th>Trayectoria</th><th class="num">I max</th><th class="num">E max</th></tr>{filas_t}</table>
 <h2>Convergencia de los pesos</h2>
 <img src="data:image/png;base64,{img}">
</main></div></body></html>"""
    HTML_PATH.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
