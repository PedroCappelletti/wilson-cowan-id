#!/usr/bin/env python3
# =============================================================================
#  EXPERIMENTO 2 — IDENTIFICACION CON MULTIPLES TRAYECTORIAS
# =============================================================================
#
#  Objetivo: recuperar los 4 pesos de Wilson-Cowan incluido wII, que en una
#  sola trayectoria no se podia (se degenera con wIE en u_i = wIE*E - wII*I).
#
#  Como: se generan VARIAS trayectorias del MISMO sistema (mismo theta) pero
#  con estimulos P/Q distintos -> excitan E e I de formas diferentes. Un solo
#  theta COMPARTIDO tiene que explicarlas a todas, lo que rompe la degeneracion.
#
#  USO:  python scripts/train_multi.py
# =============================================================================

from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.wilson_cowan import WilsonCowanParams, box_pulse  # noqa: E402
from src.data import generate_dataset  # noqa: E402
from src.pinn.multitraj import MultiTrajPINN, MultiTrajectoryTrainer  # noqa: E402
from src.pinn.train import TrainConfig  # noqa: E402


# #############################################################################
# ##   ZONA EDITABLE                                                         ##
# #############################################################################

TRAIN_SEED = 0

# --- Las TRAYECTORIAS: mismo theta (parametros), distintos estimulos P y Q.
#     Cada (amplitud, t_on, t_off). Variar esto es lo que rompe la degeneracion:
#     conviene que entre ellas cambie cuanto se excita E vs I.
TRAYECTORIAS = [
    {"P": (0.8, 100, 400), "Q": (0.6, 200, 500)},   # la original
    {"P": (1.2, 100, 300), "Q": (0.3, 150, 450)},   # P fuerte, Q debil/temprano
    {"P": (0.5, 100, 500), "Q": (1.0, 100, 400)},   # P debil, Q fuerte (excita mas a I)
    {"P": (0.9, 150, 350), "Q": (0.5, 100, 300)},   # otra combinacion de tiempos
]

# Tiempo y densidad (igual para todas).
T_SPAN = (0.0, 600.0)
N_PUNTOS = 60001          # dt = 0.01
I0, E0 = 0.0, 0.0

# Parametros VERDADEROS del modelo (los del MATLAB). Es lo que hay que recuperar.
PARAMS = WilsonCowanParams()

# Arquitectura.
HIDDEN_DIM, N_LAYERS = 64, 4
N_FOURIER, FOURIER_SCALE = 128, 6.0

# Entrenamiento (misma receta que el 4.2: warmup + congelar + lr alto para theta).
CONFIG = TrainConfig(
    epochs=30_000, lr=1e-3, lr_weights=0.05,
    w_data=1.0, w_physics=1.0, w_ic=1.0,
    batch_size=8_000, n_collocation=4_000,
    val_fraction=0.0, physics_warmup=3000, freeze_net_after_warmup=True,
    device="cpu", log_every=200, early_stop=True,
)

# #############################################################################
# ##   FIN ZONA EDITABLE                                                     ##
# #############################################################################

OUT = Path("results")
FIG_W = OUT / "figures" / "multi_pesos.png"
FIG_FIT = OUT / "figures" / "multi_ajuste.png"
HTML_PATH = OUT / "reporte_multitrayectoria.html"


def main() -> None:
    torch.manual_seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)

    # --- 1) Generar las trayectorias (mismo theta, distintos estimulos).
    datasets = []
    for cfg_t in TRAYECTORIAS:
        ds = generate_dataset(
            params=PARAMS,
            P=box_pulse(*cfg_t["P"]), Q=box_pulse(*cfg_t["Q"]),
            I0=I0, E0=E0, t_span=T_SPAN, n_eval=N_PUNTOS, noise_std=0.0, seed=TRAIN_SEED,
        )
        datasets.append(ds)
    print(f"{len(datasets)} trayectorias generadas (mismo theta, distintos estimulos).")

    w_true = {k: getattr(PARAMS, k) for k in ("wEE", "wEI", "wIE", "wII")}

    # --- 2) Parametros fijos (los que no se identifican) + offsets.
    fixed = {
        "te": PARAMS.te, "ti": PARAMS.ti, "ae": PARAMS.ae, "ai": PARAMS.ai,
        "thetae": PARAMS.thetae, "thetai": PARAMS.thetai, "ke": PARAMS.ke, "ki": PARAMS.ki,
    }

    # --- 3) Modelo multi-trayectoria (theta arranca ignorante en 1.0).
    model = MultiTrajPINN(
        n_traj=len(datasets), t_min=T_SPAN[0], t_max=T_SPAN[1],
        hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
        n_fourier=N_FOURIER, fourier_scale=FOURIER_SCALE,
        w_init={k: 1.0 for k in ("wEE", "wEI", "wIE", "wII")},
    )

    print("=== Experimento 2: identificacion multi-trayectoria (theta compartido) ===")
    trainer = MultiTrajectoryTrainer(model, CONFIG, fixed)
    hist = trainer.train(datasets)

    # --- 4) Reporte de pesos.
    w_pred = model.weights_dict()
    print("\n=== Pesos identificados vs verdaderos ===")
    print(f"{'peso':6} {'verdadero':>10} {'estimado':>10} {'error %':>10}")
    for k in ("wEE", "wEI", "wIE", "wII"):
        err = 100.0 * abs(w_pred[k] - w_true[k]) / abs(w_true[k])
        print(f"{k:6} {w_true[k]:10.4f} {w_pred[k]:10.4f} {err:9.2f}%")

    _plot_weights(hist, w_true, FIG_W)
    _plot_fits(trainer, datasets, FIG_FIT)
    _build_html(w_true, w_pred, hist, datasets)
    print(f"\nReporte HTML: {HTML_PATH}")


def _plot_weights(hist, w_true, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    for k in ("wEE", "wEI", "wIE", "wII"):
        line, = ax.plot(hist[k], label=k, lw=1.3)
        ax.axhline(w_true[k], color=line.get_color(), ls=":", lw=1)
    if hist.get("warmup"):
        ax.axvline(hist["warmup"], color="purple", ls=":", lw=1.2, label="fin warmup")
    ax.set_xlabel("epoch"); ax.set_ylabel("valor del peso")
    ax.set_title("Convergencia de los pesos (punteado = valor verdadero)")
    ax.legend(ncol=5); ax.grid(True, alpha=0.3)
    fig.tight_layout(); _save(fig, path)


def _plot_fits(trainer, datasets, path):
    import matplotlib.pyplot as plt
    n = len(datasets)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for i, (ax, ds) in enumerate(zip(axes, datasets)):
        t = ds["t"]; pred = trainer.predict(i, t)
        ax.plot(t, ds["E"], lw=1.2, label="E real")
        ax.plot(t, pred[:, 1], "--", lw=1.0, label="E PINN")
        ax.plot(t, ds["I"], lw=1.2, label="I real")
        ax.plot(t, pred[:, 0], "--", lw=1.0, label="I PINN")
        ax.set_ylabel(f"tray. {i}"); ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("tiempo (s)")
    fig.suptitle("Ajuste por trayectoria (PINN vs real)")
    fig.tight_layout(); _save(fig, path)


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)


def _b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def _build_html(w_true, w_pred, hist, datasets):
    max_err = max(100.0 * abs(w_pred[k] - w_true[k]) / abs(w_true[k])
                  for k in ("wEE", "wEI", "wIE", "wII"))
    paso_ok = max_err < 10
    veredicto = ("ok", "PASA — los 4 pesos recuperados") if paso_ok else ("bad", "NO PASA")

    filas = ""
    for k in ("wEE", "wEI", "wIE", "wII"):
        err = 100.0 * abs(w_pred[k] - w_true[k]) / abs(w_true[k])
        cls = "ok" if err < 5 else ("warn" if err < 15 else "bad")
        filas += (f"<tr><td><code>{k}</code></td><td class='num'>{w_true[k]:.3f}</td>"
                  f"<td class='num'>{w_pred[k]:.3f}</td>"
                  f"<td class='num'><span class='pill {cls}'>{err:.2f}%</span></td></tr>")

    estim = "".join(
        f"<tr><td>tray. {i}</td><td>P={t['P']}</td><td>Q={t['Q']}</td></tr>"
        for i, t in enumerate(TRAYECTORIAS))

    img_w = f"<img src='data:image/png;base64,{_b64(FIG_W)}'>"
    img_fit = f"<img src='data:image/png;base64,{_b64(FIG_FIT)}'>"

    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Experimento 2 — Multi-trayectoria</title>
<style>
  body{{font-family:-apple-system,"Segoe UI",Roboto,Arial,sans-serif;color:#1c2733;line-height:1.55;margin:0;background:#f4f6f8;padding:2rem 1rem}}
  .wrap{{max-width:880px;margin:0 auto}}
  header{{background:#1f4e79;color:#fff;border-radius:12px 12px 0 0;padding:1.5rem 1.8rem}}
  header h1{{margin:0;font-size:1.35rem}} header p{{margin:.35rem 0 0;opacity:.9;font-size:.93rem}}
  main{{background:#fff;border:1px solid #d7dee5;border-top:none;border-radius:0 0 12px 12px;padding:1.4rem 1.8rem 2rem}}
  h2{{color:#1f4e79;font-size:1.12rem;margin:1.6rem 0 .5rem;padding-bottom:.3rem;border-bottom:2px solid #e8f0f8}}
  table{{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:.92rem}}
  th,td{{text-align:left;padding:.45rem .7rem;border-bottom:1px solid #d7dee5}}
  th{{background:#e8f0f8;color:#1f4e79}} td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
  code{{background:#f0f3f6;padding:.1rem .35rem;border-radius:4px;font-family:Consolas,monospace;font-size:.88em}}
  img{{width:100%;border:1px solid #d7dee5;border-radius:8px;margin:.4rem 0}}
  .pill{{display:inline-block;padding:.1rem .5rem;border-radius:999px;font-size:.78rem;font-weight:700}}
  .ok{{background:#e3f5ea;color:#1b7f4b}} .warn{{background:#fbf0db;color:#b06a00}} .bad{{background:#fbe3e3;color:#b02a2a}}
  .veredicto{{font-size:1.05rem;font-weight:700;padding:.7rem 1rem;border-radius:8px;margin:.6rem 0}}
  .v-ok{{background:#e3f5ea;color:#1b7f4b;border:1px solid #b6e2c6}} .v-bad{{background:#fbe3e3;color:#b02a2a;border:1px solid #eebcbc}}
  .nota{{background:#e8f0f8;border-left:4px solid #1f4e79;padding:.7rem 1rem;border-radius:0 6px 6px 0;margin:1rem 0;font-size:.92rem}}
  footer{{color:#5b6770;font-size:.82rem;text-align:center;margin-top:1.3rem}}
</style></head><body><div class="wrap">
<header><h1>Experimento 2 — Identificacion con multiples trayectorias</h1>
<p>Wilson-Cowan + PINN · theta compartido entre {len(datasets)} trayectorias</p></header>
<main>
  <div class="veredicto {'v-ok' if veredicto[0]=='ok' else 'v-bad'}">{veredicto[1]}</div>

  <h2>Objetivo</h2>
  <p>Recuperar los 4 pesos —incluido <code>wII</code>, que en una sola trayectoria no se podia—
  usando varias trayectorias del mismo sistema con estimulos distintos y un <b>theta compartido</b>.
  Esto rompe la degeneracion <code>wIE</code>/<code>wII</code> de la entrada inhibitoria
  <code>u_i = wIE·E − wII·I</code>.</p>

  <h2>Trayectorias usadas (mismo theta, distintos estimulos)</h2>
  <table><tr><th>Trayectoria</th><th>Estimulo P</th><th>Estimulo Q</th></tr>{estim}</table>

  <h2>Pesos identificados (arranque ignorante: todos en 1.0)</h2>
  <table><tr><th>Peso</th><th class='num'>Verdadero</th><th class='num'>Estimado</th><th class='num'>Error</th></tr>{filas}</table>

  <h2>Convergencia de los pesos</h2>
  <p>Con varias trayectorias, <code>wII</code> ya no puede esconderse compensando a <code>wIE</code>.</p>
  {img_w}

  <h2>Ajuste por trayectoria</h2>
  {img_fit}

  <div class="nota">Generado por <code>scripts/train_multi.py</code>. Receta: warmup 'datos primero'
  + congelar las redes + lr propio para theta (heredada del paso 4.2), ahora sumando la fisica
  sobre todas las trayectorias.</div>
</main>
<footer>Proyecto Wilson-Cowan + PINN · Concurso I+D ITBA 2026 · Experimento 2</footer>
</div></body></html>"""
    HTML_PATH.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
