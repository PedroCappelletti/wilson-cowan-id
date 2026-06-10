#!/usr/bin/env python3
# =============================================================================
#  BARRIDO DE RUIDO  (Experimento 4.2 — multi-trayectoria + FD)
# =============================================================================
#
#  Niveles de ruido (calibrados a SNR de senales LFP):
#      noise_std = 0.00  ->  sin ruido   (referencia)
#      noise_std = 0.01  ->  bajo        (SNR ~ 30 dB)
#      noise_std = 0.05  ->  medio       (SNR ~ 20 dB)
#      noise_std = 0.10  ->  alto        (SNR ~ 14 dB)
#
#  Estrategia de dos etapas con MULTIPLES TRAYECTORIAS:
#
#    Por que multiples trayectorias:
#      Con una sola trayectoria (Q fijo), E(t) e I(t) oscilan correlacionados.
#      El optimizador puede satisfacer la fisica con wIE*E - wII*I casi igual
#      para distintos pares (wIE, wII), dejando wII sin identificar (error ~97%).
#      Con Q = 0.2, 0.6, 1.0 la relacion E(t)/I(t) varia entre trayectorias,
#      rompiendo la correlacion y haciendo que los 4 pesos sean identificables.
#
#    Etapa 1 — FORWARD (sin fisica, por trayectoria):
#      Para cada amplitud de Q, una MLP separada ajusta esa trayectoria.
#      Los pesos fisicos quedan fijos en 1.0 (no se usan).
#
#    Etapa 2 — IDENTIFICACION (MLPs congelados, raw_w compartido):
#      Las derivadas dI/dt, dE/dt se computan por FD sobre cada MLP congelado
#      y se guardan como constantes antes del loop.
#      Un unico raw_w (4 parametros) minimiza la suma de residuos fisicos
#      sobre las 3 trayectorias simultaneamente.
#
#  USO:  python scripts/noise_sweep.py
# =============================================================================

from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from src.wilson_cowan import WilsonCowanParams, box_pulse
from src.data import generate_dataset, save_dataset
from src.pinn import PINN
from src.pinn.train import Trainer, TrainConfig


# #############################################################################
# ##   ZONA EDITABLE                                                         ##
# #############################################################################

NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10]  # referencia + bajo/medio/alto (SNR LFP)

# Configuraciones de estimulos para las 4 trayectorias.
# Variamos TANTO P como Q (amplitud y timing) para maximizar la diversidad
# de las relaciones E(t)/I(t) — condicion necesaria para identificar wII.
# Formato: (P_amp, P_t_on, P_t_off, Q_amp, Q_t_on, Q_t_off)
TRAJ_STIMULI = [
    (0.8, 100.0, 400.0, 0.6, 200.0, 500.0),
    (1.2, 100.0, 300.0, 0.3, 150.0, 450.0),
    (0.5, 100.0, 500.0, 1.0, 100.0, 400.0),
    (0.9, 150.0, 350.0, 1.8, 100.0, 400.0),  # Q ALTO: inyecta amplitud en I -> gradiente fuerte hacia wII
]

PARAMS = WilsonCowanParams(
    te=1.0, ti=2.0,
    wEE=6.4, wEI=4.8, wIE=6.0, wII=1.2,
    ae=1.2, ai=1.0, thetae=2.8, thetai=4.0,
)

T_SPAN   = (0.0, 600.0)
N_PUNTOS = 6001   # 1 punto cada 0.1s — coincide con FD_DT y es 10x mas rapido que 60001
I0, E0   = 0.0, 0.0
SEED     = 42

HIDDEN_DIM    = 64
N_LAYERS      = 4
N_FOURIER     = 128   # restaurado: capacidad completa para FD precisas (fidelidad sin ruido)
FOURIER_SCALE = 6.0   # restaurado

# --- Etapa 1: ajuste de trayectoria (Adam, sin fisica) ---
CONFIG_S1 = TrainConfig(
    epochs=15_000,
    lr=1e-3,
    weight_decay=0.0,   # sin regularizacion: maxima fidelidad (sin ruido). Se reactiva para el sweep con ruido.
    w_data=1.0,
    w_physics=0.0,
    w_ic=1.0,
    batch_size=8_000,
    n_collocation=1,    # w_physics=0 => fisica no se usa; 1 punto evita el overhead de create_graph
    val_fraction=0.0,   # entrena en TODOS los puntos (FD precisas en todo el dominio)
    device="cpu",
    log_every=200,      # cada 200 epochs para que early_stop funcione (antes era 2000 -> nunca paraba)
    early_stop=True,
    plateau_patience=10,  # para despues de 10x200=2000 epochs sin mejora
    plateau_tol=1e-2,   # 1% de mejora requerida (1e-3 era tan chico que nunca disparaba)
)

# --- Etapa 2: identificacion con FD multi-trayectoria ---
S2_EPOCHS    = 30_000
S2_LR        = 1e-2
S2_N_COLLOC  = 5_000   # puntos por trayectoria (fijos, pre-computados)
S2_FD_DT     = 0.1     # paso de diferencias finitas [s]
S2_LOG_EVERY = 2_000

OUT_DIR = Path("results")

# #############################################################################
# ##   FIN ZONA EDITABLE                                                     ##
# #############################################################################

W_TRUE   = {"wEE": 6.4, "wEI": 4.8, "wIE": 6.0, "wII": 1.2}
IDENTIFY = ("wEE", "wEI", "wIE", "wII")


# =============================================================================
#  ETAPA 2 — identificacion multi-trayectoria con raw_w compartido
# =============================================================================
def stage2_identify_multi(models: list, datasets: list, fixed: dict) -> tuple:
    """
    Optimiza UN SOLO conjunto de pesos fisicos (raw_w) minimizando la suma
    de residuos fisicos sobre todas las trayectorias simultaneamente.

    Cada trayectoria tiene su propia MLP congelada con FD pre-computadas.
    El raw_w compartido es una nn.Parameter independiente.
    """
    dev = "cpu"

    def col(x):
        return torch.tensor(np.asarray(x), dtype=torch.float32, device=dev).reshape(-1, 1)

    # Congelar todos los MLPs
    for model in models:
        for param in model.net.parameters():
            param.requires_grad_(False)

    # Pre-computar FD para cada trayectoria (UNA SOLA VEZ)
    dt = S2_FD_DT
    traj_list = []
    for model, ds in zip(models, datasets):
        t_all = col(ds["t"])
        P_all = col(ds["P"])
        Q_all = col(ds["Q"])
        n = t_all.shape[0]
        torch.manual_seed(0)
        ci  = torch.randperm(n)[:S2_N_COLLOC]
        t_c = t_all[ci]; P_c = P_all[ci]; Q_c = Q_all[ci]
        t_lo = float(t_all.min()); t_hi = float(t_all.max())
        with torch.no_grad():
            t_plus  = torch.clamp(t_c + dt, t_lo, t_hi)
            t_minus = torch.clamp(t_c - dt, t_lo, t_hi)
            out_c      = model(t_c)
            out_plus   = model(t_plus)
            out_minus  = model(t_minus)
            I_c  = out_c[:, 0:1]; E_c  = out_c[:, 1:2]
            gap  = t_plus - t_minus
            dI_dt = (out_plus[:, 0:1] - out_minus[:, 0:1]) / gap
            dE_dt = (out_plus[:, 1:2] - out_minus[:, 1:2]) / gap
        traj_list.append((I_c, E_c, dI_dt, dE_dt, P_c, Q_c))

    # raw_w compartido: inicializado desde el primer modelo
    raw_w = torch.nn.Parameter(models[0].raw_w.detach().clone())
    opt   = torch.optim.Adam([raw_w], lr=S2_LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)

    p    = fixed
    hist = {"loss": [], **{k: [] for k in IDENTIFY}}

    for epoch in range(S2_EPOCHS):
        opt.zero_grad()

        w_vals = F.softplus(raw_w)
        w = {k: w_vals[i] for i, k in enumerate(IDENTIFY)}

        # Suma de residuos sobre todas las trayectorias
        losses = []
        for (I_c, E_c, dI_dt, dE_dt, P_c, Q_c) in traj_list:
            u_i = w["wIE"] * E_c - w["wII"] * I_c + Q_c - p["thetai"]
            u_e = w["wEE"] * E_c - w["wEI"] * I_c + P_c - p["thetae"]
            rhs_I = (1.0 / p["ti"]) * (-I_c + torch.sigmoid(p["ai"] * u_i) - p["ki"])
            rhs_E = (1.0 / p["te"]) * (-E_c + torch.sigmoid(p["ae"] * u_e) - p["ke"])
            losses.append((( dI_dt - rhs_I) ** 2 + (dE_dt - rhs_E) ** 2).mean())
        total_L = sum(losses)

        total_L.backward()
        opt.step()

        loss_f = total_L.item()
        wdict  = {k: float(F.softplus(raw_w.detach())[i]) for i, k in enumerate(IDENTIFY)}
        hist["loss"].append(loss_f)
        for k in IDENTIFY:
            hist[k].append(wdict[k])
        sched.step(loss_f)

        if epoch % S2_LOG_EVERY == 0 or epoch == S2_EPOCHS - 1:
            lr_now = opt.param_groups[0]["lr"]
            print(
                f"  S2 {epoch:5d} | L={loss_f:.3e} | lr={lr_now:.1e} | "
                + " ".join(f"{k}={wdict[k]:.3f}" for k in IDENTIFY)
            )

    # Descongelar
    for model in models:
        for param in model.net.parameters():
            param.requires_grad_(True)

    hist["stop_epoch"] = S2_EPOCHS - 1
    return hist, raw_w.detach()


# =============================================================================
#  EXPERIMENTO POR NIVEL DE RUIDO
# =============================================================================
def run_one(noise_std: float) -> dict:
    tag = f"noise_{noise_std:.2f}".replace(".", "_")
    print(f"\n{'='*60}")
    print(f"  noise_std = {noise_std:.2f}")
    print(f"{'='*60}")

    ae, ai = PARAMS.ae, PARAMS.ai
    te, ti = PARAMS.te, PARAMS.ti
    thetae, thetai = PARAMS.thetae, PARAMS.thetai
    fixed = {
        "te": te, "ti": ti, "ae": ae, "ai": ai,
        "thetae": thetae, "thetai": thetai,
        "ke": 1.0 / (1.0 + np.exp(ae * thetae)),
        "ki": 1.0 / (1.0 + np.exp(ai * thetai)),
    }
    fixed_s1 = {**fixed, **{k: 1.0 for k in IDENTIFY}}

    models   = []
    datasets = []

    # =========================================================================
    #  ETAPA 1: una MLP por trayectoria (distintos estimulos P y Q)
    # =========================================================================
    for traj_idx, (p_amp, p_ton, p_toff, q_amp, q_ton, q_toff) in enumerate(TRAJ_STIMULI):
        P_func = box_pulse(amplitude=p_amp, t_on=p_ton, t_off=p_toff)
        Q_func = box_pulse(amplitude=q_amp, t_on=q_ton, t_off=q_toff)
        ds = generate_dataset(
            params=PARAMS, P=P_func, Q=Q_func,
            I0=I0, E0=E0, t_span=T_SPAN, n_eval=N_PUNTOS,
            noise_std=noise_std, seed=SEED,
        )
        save_dataset(ds, OUT_DIR / "datasets" / f"dataset_{tag}_t{traj_idx}.npz")
        print(f"\n  [Etapa 1] tray {traj_idx} — P=({p_amp},{p_ton},{p_toff})  Q=({q_amp},{q_ton},{q_toff})  {ds['t'].shape[0]} puntos")

        t_min = float(ds["t"].min()); t_max = float(ds["t"].max())
        model = PINN(
            hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
            t_min=t_min, t_max=t_max,
            identify=IDENTIFY, w_init={k: 1.0 for k in IDENTIFY},
            n_fourier=N_FOURIER, fourier_scale=FOURIER_SCALE,
        )
        trainer = Trainer(model, CONFIG_S1, fixed_s1)
        trainer.train(ds)

        # --- DIAGNOSTICO: que tan bien ajusto Stage 1 ESTA trayectoria.
        #     Si una traj tiene MSE mucho peor que las otras, sus FD estan sucias
        #     y contaminan el raw_w compartido de Stage 2.
        with torch.no_grad():
            t_t = torch.tensor(ds["t"], dtype=torch.float32).reshape(-1, 1)
            pred_s1 = model(t_t).numpy()
        tgt_s1 = np.stack([ds["I"], ds["E"]], axis=1)
        mse_s1 = float(((pred_s1 - tgt_s1) ** 2).mean())
        print(f"  [Etapa 1] tray {traj_idx} ajuste: MSE = {mse_s1:.2e}")

        models.append(model)
        datasets.append(ds)

    # =========================================================================
    #  ETAPA 2: identificacion multi-trayectoria con raw_w compartido
    # =========================================================================
    print(f"\n  [Etapa 2] Identificacion — {len(models)} trayectorias, raw_w compartido")
    hist2, final_raw_w = stage2_identify_multi(models, datasets, fixed)

    # --- Pesos finales (del raw_w compartido)
    w_vals = F.softplus(final_raw_w)
    w_pred = {k: float(w_vals[i]) for i, k in enumerate(IDENTIFY)}
    errors  = {k: 100.0 * abs(w_pred[k] - W_TRUE[k]) / abs(W_TRUE[k]) for k in IDENTIFY}
    max_err = max(errors.values())

    # --- MSE sobre la trayectoria de referencia (indice 0, estimulos base)
    ref_idx   = 0
    ref_model = models[ref_idx]
    ref_ds    = datasets[ref_idx]
    with torch.no_grad():
        ref_model.raw_w.data.copy_(final_raw_w)
        t_tensor = torch.tensor(ref_ds["t"], dtype=torch.float32).reshape(-1, 1)
        pred = ref_model(t_tensor).numpy()
    target    = np.stack([ref_ds["I"], ref_ds["E"]], axis=1)
    n_train   = int(round(0.8 * ref_ds["t"].shape[0]))
    train_mse = float(((pred[:n_train] - target[:n_train]) ** 2).mean())
    val_mse   = float(((pred[n_train:] - target[n_train:]) ** 2).mean())

    print(f"\n  Pesos identificados vs verdaderos:")
    print(f"  {'peso':6} {'verdadero':>10} {'estimado':>10} {'error %':>10}")
    for k in IDENTIFY:
        print(f"  {k:6} {W_TRUE[k]:10.4f} {w_pred[k]:10.4f} {errors[k]:9.2f}%")
    print(f"  MSE train={train_mse:.2e}  val={val_mse:.2e}  max_err={max_err:.1f}%")

    # --- Checkpoint
    ckpt_path = OUT_DIR / "models" / f"pinn_{tag}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"raw_w": final_raw_w, "fixed": fixed}, ckpt_path)

    fig_path = OUT_DIR / "figures" / f"fit_{tag}.png"
    ref_stim = TRAJ_STIMULI[ref_idx]
    _plot_fit(ref_model, ref_ds, noise_std,
              f"P=({ref_stim[0]},{ref_stim[1]},{ref_stim[2]})  Q=({ref_stim[3]},{ref_stim[4]},{ref_stim[5]})",
              fig_path)

    conv_path = OUT_DIR / "figures" / f"convergencia_s2_{tag}.png"
    _plot_stage2_convergence(hist2, noise_std, conv_path)

    return {
        "noise_std": noise_std, "tag": tag,
        "w_pred": w_pred, "errors": errors, "max_err": max_err,
        "train_mse": train_mse, "val_mse": val_mse,
        "stop_epoch": hist2["stop_epoch"],
        "fig_path": fig_path, "conv_path": conv_path, "hist": hist2,
    }


# =============================================================================
#  GRAFICOS
# =============================================================================
def _plot_fit(model, ds, noise_std, stim_label, path):
    import matplotlib.pyplot as plt
    t = ds["t"]
    with torch.no_grad():
        t_tensor = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)
        pred = model(t_tensor).numpy()
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, j, nombre in zip(axes, (0, 1), ("I", "E")):
        ax.plot(t, ds[nombre], label=f"{nombre} real", lw=1.2, alpha=0.7)
        ax.plot(t, pred[:, j], "--", label=f"{nombre} PINN", lw=1.1)
        ax.set_ylabel(nombre); ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    axes[0].set_title(f"PINN vs real  |  noise_std={noise_std:.2f}  |  {stim_label}")
    axes[1].set_xlabel("tiempo (s)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)


def _plot_stage2_convergence(hist, noise_std, path):
    """Trayectoria de cada peso durante Stage 2 + loss. Permite ver si wII converge."""
    import matplotlib.pyplot as plt
    epochs = list(range(len(hist["loss"])))
    colores = {"wEE": "#1f77b4", "wEI": "#ff7f0e", "wIE": "#2ca02c", "wII": "#d62728"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Izquierda: pesos vs epoca
    ax = axes[0]
    for k in IDENTIFY:
        ax.plot(epochs, hist[k], label=k, color=colores[k], lw=1.3)
    for k, v in W_TRUE.items():
        ax.axhline(v, color=colores[k], ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("epoca S2"); ax.set_ylabel("valor estimado")
    ax.set_title(f"Convergencia de pesos — noise={noise_std:.2f}\n(lineas punteadas = valores verdaderos)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- Derecha: loss vs epoca (escala log)
    ax2 = axes[1]
    ax2.semilogy(epochs, hist["loss"], color="#555", lw=1.2)
    ax2.set_xlabel("epoca S2"); ax2.set_ylabel("loss (log)")
    ax2.set_title("Loss Stage 2 (FD residual)")
    ax2.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)


def _plot_comparison(results, path):
    import matplotlib.pyplot as plt
    noise_vals = [r["noise_std"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    colores = {"wEE": "#1f77b4", "wEI": "#ff7f0e", "wIE": "#2ca02c", "wII": "#d62728"}
    for k in IDENTIFY:
        errs = [r["errors"][k] for r in results]
        ax.plot(noise_vals, errs, "o-", label=k, color=colores[k], lw=1.8, ms=7)
    ax.axhline(10, color="gray", ls="--", lw=1, label="umbral 10%")
    ax.set_xlabel("noise_std"); ax.set_ylabel("error parametrico relativo (%)")
    ax.set_title(f"Error de identificacion vs ruido ({len(TRAJ_STIMULI)} trayectorias, FD)")
    ax.legend(ncol=5); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)


# =============================================================================
#  REPORTE HTML
# =============================================================================
def _b64(p):
    return base64.b64encode(p.read_bytes()).decode()


def _build_html(results, fig_cmp_path, html_path):
    filas = ""
    for r in results:
        cls_max = "ok" if r["max_err"] < 10 else ("warn" if r["max_err"] < 25 else "bad")
        w_cells = "".join(
            f"<td class='num'><span class='pill "
            f"{'ok' if r['errors'][k]<10 else ('warn' if r['errors'][k]<25 else 'bad')}'>"
            f"{r['errors'][k]:.1f}%</span></td>"
            for k in IDENTIFY
        )
        filas += (
            f"<tr><td class='num'>{r['noise_std']:.2f}</td>{w_cells}"
            f"<td class='num'>{r['train_mse']:.2e}</td>"
            f"<td class='num'>{r['val_mse']:.2e}</td>"
            f"<td class='num'>{r['stop_epoch']}</td>"
            f"<td class='num'><span class='pill {cls_max}'>{r['max_err']:.1f}%</span></td></tr>"
        )

    imgs_fit = "".join(
        f"<h3>noise_std = {r['noise_std']:.2f}</h3>"
        f"<img src='data:image/png;base64,{_b64(r['fig_path'])}'>"
        f"<img src='data:image/png;base64,{_b64(r['conv_path'])}'>"
        for r in results
    )
    img_cmp = f"<img src='data:image/png;base64,{_b64(fig_cmp_path)}'>"

    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Barrido de ruido — Wilson-Cowan PINN (multi-trayectoria)</title>
<style>
  body{{font-family:-apple-system,"Segoe UI",Roboto,Arial,sans-serif;color:#1c2733;
       line-height:1.55;margin:0;background:#f4f6f8;padding:2rem 1rem}}
  .wrap{{max-width:960px;margin:0 auto}}
  header{{background:#1f4e79;color:#fff;border-radius:12px 12px 0 0;padding:1.5rem 1.8rem}}
  header h1{{margin:0;font-size:1.35rem}}
  header p{{margin:.35rem 0 0;opacity:.9;font-size:.93rem}}
  main{{background:#fff;border:1px solid #d7dee5;border-top:none;
        border-radius:0 0 12px 12px;padding:1.4rem 1.8rem 2rem}}
  h2{{color:#1f4e79;font-size:1.12rem;margin:1.6rem 0 .5rem;
      padding-bottom:.3rem;border-bottom:2px solid #e8f0f8}}
  h3{{color:#1f4e79;font-size:1rem;margin:1.2rem 0 .3rem}}
  table{{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:.9rem}}
  th,td{{text-align:left;padding:.4rem .6rem;border-bottom:1px solid #d7dee5}}
  th{{background:#e8f0f8;color:#1f4e79}}
  td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
  img{{width:100%;border:1px solid #d7dee5;border-radius:8px;margin:.4rem 0}}
  .pill{{display:inline-block;padding:.1rem .5rem;border-radius:999px;
         font-size:.75rem;font-weight:700;margin-right:.3rem}}
  .ok{{background:#e3f5ea;color:#1b7f4b}}
  .warn{{background:#fbf0db;color:#b06a00}}
  .bad{{background:#fbe3e3;color:#b02a2a}}
  footer{{color:#5b6770;font-size:.82rem;text-align:center;margin-top:1.3rem}}
</style></head><body><div class="wrap">
<header>
  <h1>Barrido de ruido — Experimento 4.2 (PINN multi-trayectoria)</h1>
  <p>Identificacion parametrica de Wilson-Cowan bajo ruido gaussiano aditivo · ITBA 2026</p>
</header>
<main>
  <h2>Estrategia de entrenamiento</h2>
  <table>
    <tr><th>Trayectorias</th>
        <td>{len(TRAJ_STIMULI)} trayectorias — P y Q distintos (amplitud + timing) para romper correlacion E/I y hacer wII identificable</td></tr>
    <tr><th>Etapa 1</th>
        <td>Adam ajusta una MLP separada por trayectoria (w_fisica=0). {CONFIG_S1.epochs} epochs max.</td></tr>
    <tr><th>Etapa 2</th>
        <td>MLPs congelados. FD pre-computadas. Adam optimiza raw_w compartido sobre las {len(TRAJ_STIMULI)} trayectorias. {S2_EPOCHS} epochs.</td></tr>
    <tr><th>Verdaderos</th><td>wEE=6.4 · wEI=4.8 · wIE=6.0 · wII=1.2</td></tr>
    <tr><th>Red</th>
        <td>MLP {HIDDEN_DIM}x{N_LAYERS}, tanh · Fourier features={N_FOURIER} (escala {FOURIER_SCALE})</td></tr>
  </table>

  <h2>Resultados por nivel de ruido</h2>
  <table>
    <tr>
      <th class="num">noise_std</th>
      <th class="num">wEE err%</th><th class="num">wEI err%</th>
      <th class="num">wIE err%</th><th class="num">wII err%</th>
      <th class="num">MSE train</th><th class="num">MSE val</th>
      <th class="num">epoch S2</th><th class="num">max err%</th>
    </tr>
    {filas}
  </table>

  <h2>Error parametrico vs nivel de ruido</h2>
  {img_cmp}

  <h2>Ajuste PINN + convergencia de pesos Stage 2 (por nivel de ruido)</h2>
  {imgs_fit}
</main>
<footer>Proyecto Wilson-Cowan + PINN · Concurso I+D ITBA 2026</footer>
</div></body></html>"""

    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")
    print(f"\nReporte HTML: {html_path}")


# =============================================================================
#  MAIN
# =============================================================================
def main():
    print(f"=== Barrido de ruido ({len(TRAJ_STIMULI)} trayectorias, P+Q variados): {len(NOISE_LEVELS)} niveles ===")
    results = []
    for noise_std in NOISE_LEVELS:
        results.append(run_one(noise_std))

    fig_cmp = OUT_DIR / "figures" / "noise_sweep_comparison.png"
    _plot_comparison(results, fig_cmp)
    _build_html(results, fig_cmp, OUT_DIR / "reporte_noise_sweep.html")

    print("\n=== RESUMEN FINAL ===")
    print(f"{'noise_std':>10} {'wEE%':>7} {'wEI%':>7} {'wIE%':>7} {'wII%':>7} {'max%':>7}")
    for r in results:
        print(
            f"{r['noise_std']:>10.2f} "
            + " ".join(f"{r['errors'][k]:>7.2f}" for k in IDENTIFY)
            + f" {r['max_err']:>7.2f}"
        )


if __name__ == "__main__":
    main()
