#!/usr/bin/env python3
# =============================================================================
#  PINN CANONICA — autograd + entrenamiento CONJUNTO + multi-trayectoria
# =============================================================================
#
#  Este es el metodo "PINN sin asteriscos", como complemento al metodo de
#  dos-etapas + diferencias finitas de scripts/noise_sweep.py.
#
#  Diferencias clave con noise_sweep.py:
#
#    noise_sweep.py (surrogate + matching, NO es PINN canonica):
#      - Etapa 1: cada MLP ajusta su trayectoria SIN fisica (solo suaviza).
#      - Etapa 2: MLPs CONGELADAS, derivadas por DIFERENCIAS FINITAS,
#                 se ajustan los pesos aparte.
#      - La red nunca "aprende la fisica".
#
#    pinn_joint_sweep.py (PINN canonica):
#      - UNA sola fase: se optimizan a la vez los pesos de TODAS las MLPs
#        y los 4 pesos fisicos compartidos (raw_w).
#      - Las derivadas dI/dt, dE/dt salen por AUTOGRAD a traves de la red.
#      - La fisica entra como termino de perdida DURANTE el entrenamiento.
#        Asi la red esta obligada a ser fisicamente consistente, y la fisica
#        actua como regularizador (la red no puede ajustar ruido de alta
#        frecuencia sin violar la ecuacion).
#
#  POR QUE MULTI-TRAYECTORIA: con una sola trayectoria, E(t) e I(t) oscilan
#  correlacionados y wIE*E - wII*I queda degenerado (wII no identificable).
#  Variando P y Q entre trayectorias se rompe esa correlacion. (Lo mismo que
#  hizo funcionar al metodo FD; aca lo aplicamos a la PINN conjunta.)
#
#  ESTRUCTURA: una MLP por trayectoria (cada trayectoria es una solucion
#  distinta t->[I,E]) + UN raw_w compartido (los 4 pesos son los mismos para
#  todas las trayectorias, porque el sistema fisico es uno solo).
#
#  USO:  python scripts/pinn_joint_sweep.py
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
from src.pinn.network import inv_softplus
from src.pinn.losses import data_loss, physics_loss, initial_condition_loss


# #############################################################################
# ##   ZONA EDITABLE                                                         ##
# #############################################################################

NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10]  # referencia + bajo/medio/alto (SNR LFP)

# Mismos estimulos que noise_sweep.py (incluida la traj 4 con Q alto), para que
# las dos curvas sean comparables. (P_amp, P_t_on, P_t_off, Q_amp, Q_t_on, Q_t_off)
TRAJ_STIMULI = [
    (0.8, 100.0, 400.0, 0.6, 200.0, 500.0),
    (1.2, 100.0, 300.0, 0.3, 150.0, 450.0),
    (0.5, 100.0, 500.0, 1.0, 100.0, 400.0),
    (0.9, 150.0, 350.0, 1.8, 100.0, 400.0),  # Q ALTO: rompe la degeneracion de wII
]

PARAMS = WilsonCowanParams(
    te=1.0, ti=2.0,
    wEE=6.4, wEI=4.8, wIE=6.0, wII=1.2,
    ae=1.2, ai=1.0, thetae=2.8, thetai=4.0,
)

T_SPAN   = (0.0, 600.0)
N_PUNTOS = 6001
I0, E0   = 0.0, 0.0
SEED     = 42

HIDDEN_DIM    = 64
N_LAYERS      = 4
N_FOURIER     = 128
FOURIER_SCALE = 6.0

# --- Entrenamiento conjunto (Adam) ---
EPOCHS       = 20_000
LR           = 1e-3   # lr de las MLPs (ajuste de trayectoria)
LR_W         = 1e-2   # lr de los pesos fisicos raw_w: 10x mas alto porque deben
                      # viajar lejos (1.0 -> 6.4) y reciben gradiente debil.
                      # Ataca la causa del stall de wII (Intentos 1-2 del log).
# --- Politica de learning rate (cambiar esta UNA linea) ---
#   True  = lr FIJO durante todo Adam. Mas rapido, no se ralentiza ni congela.
#           La precision fina la da L-BFGS al final. Recomendado para el barrido
#           con ruido (varios niveles = mucho computo).
#   False = ReduceLROnPlateau (baja el lr en mesetas). Mas lento al final pero
#           afina solo; util si se corre sin L-BFGS.
FIXED_LR     = True
W_DATA       = 20.0   # alto a proposito: ancla las redes a la trayectoria VERDADERA
                      # para que no se co-adapten a pesos incorrectos (minimo degenerado).
                      # No frena la identificacion: raw_w solo recibe gradiente de la fisica.
W_PHYSICS    = 1.0    # el "lambda" de la fisica (la que mueve raw_w).
W_IC         = 1.0
BATCH_DATA   = 2_000  # puntos de DATOS por trayectoria por paso (del 80% inicial)
N_COLLOC     = 2_000  # puntos de FISICA por trayectoria por paso (todo el dominio)
VAL_FRACTION = 0.2    # 20% final reservado para test temporal (extrapolacion)
LOG_EVERY    = 500

# --- Refinamiento final con L-BFGS (segundo orden: cierra fino lo que Adam dejo) ---
LBFGS_STEPS  = 50     # pasos externos de L-BFGS (0 = desactivar). Cada uno hace
                      # hasta max_iter iteraciones internas con line search.
LBFGS_LR     = 1.0    # lr de L-BFGS (con strong_wolfe, 1.0 es lo habitual)

# --- Checkpoints (para no perder runs largos ante un corte) ---
CHECKPOINT_EVERY = 2_000  # guardar estado cada N epochs de Adam
RESUME           = True   # True = retoma desde el ultimo checkpoint de cada nivel;
                          # False = ignora checkpoints viejos y arranca limpio (igual los guarda)

OUT_DIR = Path("results")

# #############################################################################
# ##   FIN ZONA EDITABLE                                                     ##
# #############################################################################

W_TRUE   = {"wEE": 6.4, "wEI": 4.8, "wIE": 6.0, "wII": 1.2}
IDENTIFY = ("wEE", "wEI", "wIE", "wII")


# =============================================================================
#  ENTRENAMIENTO CONJUNTO MULTI-TRAYECTORIA
# =============================================================================
def train_joint(models: list, datasets: list, fixed: dict, ckpt_path=None) -> tuple:
    """
    Optimiza SIMULTANEAMENTE:
      - los pesos de cada MLP (una por trayectoria), y
      - un unico raw_w compartido (los 4 pesos fisicos),
    con un solo optimizador. Las derivadas son por autograd (physics_loss).

    Si ckpt_path se da, guarda el estado cada CHECKPOINT_EVERY epochs y (si
    RESUME) retoma desde ahi en vez de arrancar de cero.
    """
    dev = "cpu"

    def col(x):
        return torch.tensor(np.asarray(x), dtype=torch.float32, device=dev).reshape(-1, 1)

    # --- Tensores por trayectoria + split temporal train/test.
    trajs = []
    for ds in datasets:
        t      = col(ds["t"])
        target = torch.cat([col(ds["I"]), col(ds["E"])], dim=1)
        P      = col(ds["P"])
        Q      = col(ds["Q"])
        n       = t.shape[0]
        n_train = int(round((1.0 - VAL_FRACTION) * n))
        trajs.append({
            "t": t, "target": target, "P": P, "Q": Q,
            "n": n, "n_train": n_train,
            "t0": t[0:1], "ic": target[0:1],
        })

    # --- raw_w compartido: los 4 pesos fisicos, arrancando "ignorantes" en 1.0.
    raw_w = torch.nn.Parameter(torch.stack([inv_softplus(1.0) for _ in IDENTIFY]))

    # --- Un optimizador sobre TODO, pero con DOS grupos: las MLPs a lr normal y
    #     el raw_w a lr mas alto. Los pesos fisicos deben recorrer mucho (1.0->6.4)
    #     con gradiente debil; sin esto se quedan atras (stall de wII).
    net_params = [p for m in models for p in m.parameters()]
    opt = torch.optim.Adam([
        {"params": net_params, "lr": LR},
        {"params": [raw_w],    "lr": LR_W},
    ])
    # Scheduler: solo si NO usamos lr fijo. Cuando se usa, el scheduler pisa sobre
    # una EMA de la loss (no la cruda del minibatch, ruidosa) con paciencia alta y
    # min_lr para que el lr no colapse a ~0 y congele los pesos antes de tiempo.
    sched = None if FIXED_LR else torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=80, min_lr=1e-5)

    def build_params():
        w = F.softplus(raw_w)
        return {**fixed, **{k: w[i] for i, k in enumerate(IDENTIFY)}}

    hist = {"loss": [], "data": [], "physics": [], "ic": [], **{k: [] for k in IDENTIFY}}
    ema = None  # media movil de la loss para el scheduler (ignora ruido del minibatch)

    # -------------------------------------------------------------------------
    #  CHECKPOINTING: guardar el estado completo (redes + raw_w + optimizador +
    #  historial) cada CHECKPOINT_EVERY epochs, y reanudar desde ahi si RESUME.
    # -------------------------------------------------------------------------
    def save_ckpt(epoch, phase):
        if ckpt_path is None:
            return
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch, "phase": phase,
            "models": [m.state_dict() for m in models],
            "raw_w": raw_w.detach().clone(),
            "opt": opt.state_dict(),
            "hist": hist,
        }, ckpt_path)

    start_epoch = 0
    if RESUME and ckpt_path is not None and ckpt_path.exists():
        ck = torch.load(ckpt_path)
        for m, sd in zip(models, ck["models"]):
            m.load_state_dict(sd)
        raw_w.data.copy_(ck["raw_w"])
        hist = ck["hist"]
        if ck["phase"] == "done":
            print(f"  [checkpoint] nivel ya completo (epoch {ck['epoch']}) -> salteo el entrenamiento")
            return hist, raw_w.detach()
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        print(f"  [checkpoint] reanudando Adam desde epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        opt.zero_grad()
        params = build_params()

        L_d = L_f = L_ic = 0.0
        for m, tr in zip(models, trajs):
            # Datos: minibatch del tramo de entrenamiento (80% inicial).
            ntr = tr["n_train"]
            bi  = torch.randperm(ntr, device=dev)[:BATCH_DATA]
            L_d = L_d + data_loss(m(tr["t"][bi]), tr["target"][bi])

            # Fisica: minibatch de TODO el dominio (permite extrapolar al 20% final).
            ci  = torch.randperm(tr["n"], device=dev)[:N_COLLOC]
            L_f = L_f + physics_loss(m, tr["t"][ci], tr["P"][ci], tr["Q"][ci], params)

            # Condicion inicial.
            L_ic = L_ic + initial_condition_loss(m, tr["t0"], tr["ic"])

        L = W_DATA * L_d + W_PHYSICS * L_f + W_IC * L_ic
        L.backward()
        opt.step()
        if sched is not None:
            ema = L.item() if ema is None else 0.99 * ema + 0.01 * L.item()
            sched.step(ema)

        wdict = {k: float(F.softplus(raw_w.detach())[i]) for i, k in enumerate(IDENTIFY)}
        ld, lf, lic = L_d.item(), L_f.item(), L_ic.item()
        hist["loss"].append(L.item())
        hist["data"].append(ld)
        hist["physics"].append(lf)
        hist["ic"].append(lic)
        for k in IDENTIFY:
            hist[k].append(wdict[k])

        if epoch % LOG_EVERY == 0 or epoch == EPOCHS - 1:
            lr_now = opt.param_groups[0]["lr"]
            print(
                f"  {epoch:5d} | L={L.item():.3e} "
                f"(datos={ld:.2e} fis={lf:.2e} ic={lic:.2e}) "
                f"| lr={lr_now:.1e} | "
                + " ".join(f"{k}={wdict[k]:.3f}" for k in IDENTIFY)
            )

        # Checkpoint periodico (cada CHECKPOINT_EVERY epochs de Adam).
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            save_ckpt(epoch, "adam")

    # Guardar al terminar Adam: si un corte ocurre durante L-BFGS, al reanudar
    # se saltea Adam (start_epoch = EPOCHS) y se rehace solo L-BFGS.
    save_ckpt(EPOCHS - 1, "adam")

    # =========================================================================
    #  REFINAMIENTO FINAL CON L-BFGS (segundo orden, cierra lo que Adam dejo).
    #  Sigue siendo PINN: misma perdida (datos+fisica+ic), misma red, autograd.
    #  Solo cambia el optimizador (Adam -> L-BFGS), como recomienda la literatura.
    # =========================================================================
    if LBFGS_STEPS > 0:
        print(f"\n  --- Refinamiento L-BFGS ({LBFGS_STEPS} pasos) ---")
        # L-BFGS necesita una superficie de perdida DETERMINISTA -> subconjuntos
        # FIJOS de datos y colocacion (no se re-muestrean en cada closure).
        torch.manual_seed(0)
        fixed_batches = []
        for tr in trajs:
            di = torch.randperm(tr["n_train"], device=dev)[:BATCH_DATA]
            ci = torch.randperm(tr["n"], device=dev)[:N_COLLOC]
            fixed_batches.append((di, ci))

        all_params = net_params + [raw_w]
        opt_lbfgs = torch.optim.LBFGS(
            all_params, lr=LBFGS_LR, max_iter=20, line_search_fn="strong_wolfe",
        )

        last = {}

        def closure():
            opt_lbfgs.zero_grad()
            params = build_params()
            L_d = L_f = L_ic = 0.0
            for (m, tr), (di, ci) in zip(zip(models, trajs), fixed_batches):
                L_d  = L_d  + data_loss(m(tr["t"][di]), tr["target"][di])
                L_f  = L_f  + physics_loss(m, tr["t"][ci], tr["P"][ci], tr["Q"][ci], params)
                L_ic = L_ic + initial_condition_loss(m, tr["t0"], tr["ic"])
            L = W_DATA * L_d + W_PHYSICS * L_f + W_IC * L_ic
            L.backward()
            last["d"], last["f"], last["ic"] = L_d.item(), L_f.item(), L_ic.item()
            return L

        for step in range(LBFGS_STEPS):
            L = opt_lbfgs.step(closure)
            wdict = {k: float(F.softplus(raw_w.detach())[i]) for i, k in enumerate(IDENTIFY)}
            hist["loss"].append(float(L))
            hist["data"].append(last.get("d", 0.0))
            hist["physics"].append(last.get("f", 0.0))
            hist["ic"].append(last.get("ic", 0.0))
            for k in IDENTIFY:
                hist[k].append(wdict[k])
            if step % max(1, LBFGS_STEPS // 10) == 0 or step == LBFGS_STEPS - 1:
                print(
                    f"  LBFGS {step:4d} | L={float(L):.3e} | "
                    + " ".join(f"{k}={wdict[k]:.3f}" for k in IDENTIFY)
                )

    # Checkpoint final: marca el nivel como completo (al reanudar se saltea).
    save_ckpt(EPOCHS - 1, "done")

    hist["stop_epoch"] = len(hist["loss"]) - 1
    return hist, raw_w.detach()


# =============================================================================
#  EXPERIMENTO POR NIVEL DE RUIDO
# =============================================================================
def run_one(noise_std: float) -> dict:
    tag = f"joint_noise_{noise_std:.2f}".replace(".", "_")
    print(f"\n{'='*60}")
    print(f"  PINN CONJUNTA — noise_std = {noise_std:.2f}")
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

    # --- Una MLP por trayectoria (identify=() -> MLP pura, sin pesos propios;
    #     los pesos fisicos viven en el raw_w compartido de train_joint).
    models   = []
    datasets = []
    for traj_idx, (p_amp, p_ton, p_toff, q_amp, q_ton, q_toff) in enumerate(TRAJ_STIMULI):
        P_func = box_pulse(amplitude=p_amp, t_on=p_ton, t_off=p_toff)
        Q_func = box_pulse(amplitude=q_amp, t_on=q_ton, t_off=q_toff)
        ds = generate_dataset(
            params=PARAMS, P=P_func, Q=Q_func,
            I0=I0, E0=E0, t_span=T_SPAN, n_eval=N_PUNTOS,
            noise_std=noise_std, seed=SEED,
        )
        save_dataset(ds, OUT_DIR / "datasets" / f"dataset_{tag}_t{traj_idx}.npz")
        print(f"  tray {traj_idx} — P=({p_amp},{p_ton},{p_toff})  Q=({q_amp},{q_ton},{q_toff})")

        t_min = float(ds["t"].min()); t_max = float(ds["t"].max())
        model = PINN(
            hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
            t_min=t_min, t_max=t_max,
            identify=(),  # MLP pura: los pesos fisicos son el raw_w compartido
            n_fourier=N_FOURIER, fourier_scale=FOURIER_SCALE,
        )
        models.append(model)
        datasets.append(ds)

    # --- Entrenamiento conjunto (con checkpoint por nivel de ruido).
    print(f"\n  Entrenamiento conjunto — {len(models)} trayectorias, raw_w compartido, autograd")
    ckpt_path = OUT_DIR / "checkpoints" / f"ckpt_{tag}.pt"
    hist, final_raw_w = train_joint(models, datasets, fixed, ckpt_path)

    # --- Pesos finales.
    w_vals = F.softplus(final_raw_w)
    w_pred = {k: float(w_vals[i]) for i, k in enumerate(IDENTIFY)}
    errors = {k: 100.0 * abs(w_pred[k] - W_TRUE[k]) / abs(W_TRUE[k]) for k in IDENTIFY}
    max_err = max(errors.values())

    # --- MSE sobre la trayectoria de referencia (indice 0), train (80%) / test (20%).
    ref_model = models[0]
    ref_ds    = datasets[0]
    with torch.no_grad():
        t_tensor = torch.tensor(ref_ds["t"], dtype=torch.float32).reshape(-1, 1)
        pred = ref_model(t_tensor).numpy()
    target    = np.stack([ref_ds["I"], ref_ds["E"]], axis=1)
    n_train   = int(round((1.0 - VAL_FRACTION) * ref_ds["t"].shape[0]))
    train_mse = float(((pred[:n_train] - target[:n_train]) ** 2).mean())
    val_mse   = float(((pred[n_train:] - target[n_train:]) ** 2).mean())

    print(f"\n  Pesos identificados vs verdaderos:")
    print(f"  {'peso':6} {'verdadero':>10} {'estimado':>10} {'error %':>10}")
    for k in IDENTIFY:
        print(f"  {k:6} {W_TRUE[k]:10.4f} {w_pred[k]:10.4f} {errors[k]:9.2f}%")
    print(f"  MSE train={train_mse:.2e}  val={val_mse:.2e}  max_err={max_err:.1f}%")

    # --- Checkpoint.
    ckpt_path = OUT_DIR / "models" / f"pinn_{tag}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"raw_w": final_raw_w, "fixed": fixed}, ckpt_path)

    # --- Graficos.
    ref_stim = TRAJ_STIMULI[0]
    fig_path = OUT_DIR / "figures" / f"fit_{tag}.png"
    _plot_fit(ref_model, ref_ds, noise_std, VAL_FRACTION,
              f"P=({ref_stim[0]},{ref_stim[1]},{ref_stim[2]})  Q=({ref_stim[3]},{ref_stim[4]},{ref_stim[5]})",
              fig_path)
    conv_path = OUT_DIR / "figures" / f"convergencia_{tag}.png"
    _plot_convergence(hist, noise_std, conv_path)

    return {
        "noise_std": noise_std, "tag": tag,
        "w_pred": w_pred, "errors": errors, "max_err": max_err,
        "train_mse": train_mse, "val_mse": val_mse,
        "stop_epoch": hist["stop_epoch"],
        "fig_path": fig_path, "conv_path": conv_path, "hist": hist,
    }


# =============================================================================
#  GRAFICOS
# =============================================================================
def _plot_fit(model, ds, noise_std, val_fraction, stim_label, path):
    import matplotlib.pyplot as plt
    t = ds["t"]
    with torch.no_grad():
        t_tensor = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)
        pred = model(t_tensor).numpy()
    t_split = t.min() + (1.0 - val_fraction) * (t.max() - t.min())
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, j, nombre in zip(axes, (0, 1), ("I", "E")):
        ax.plot(t, ds[nombre], label=f"{nombre} real", lw=1.2, alpha=0.7)
        ax.plot(t, pred[:, j], "--", label=f"{nombre} PINN", lw=1.1)
        ax.axvline(t_split, color="gray", ls=":", lw=1)
        ax.set_ylabel(nombre); ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    axes[0].set_title(f"PINN conjunta vs real  |  noise_std={noise_std:.2f}  |  {stim_label}\n"
                      "(punteado gris = corte train | test)")
    axes[1].set_xlabel("tiempo (s)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)


def _plot_convergence(hist, noise_std, path):
    import matplotlib.pyplot as plt
    epochs = list(range(len(hist["loss"])))
    colores = {"wEE": "#1f77b4", "wEI": "#ff7f0e", "wIE": "#2ca02c", "wII": "#d62728"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for k in IDENTIFY:
        ax.plot(epochs, hist[k], label=k, color=colores[k], lw=1.3)
    for k, v in W_TRUE.items():
        ax.axhline(v, color=colores[k], ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("epoca"); ax.set_ylabel("valor estimado")
    ax.set_title(f"Convergencia de pesos (conjunta) — noise={noise_std:.2f}\n"
                 "(punteadas = valores verdaderos)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.semilogy(epochs, hist["loss"], label="total", color="#555", lw=1.0, alpha=0.6)
    ax2.semilogy(epochs, hist["data"], label="datos", lw=1.0)
    ax2.semilogy(epochs, hist["physics"], label="fisica", lw=1.0)
    ax2.set_xlabel("epoca"); ax2.set_ylabel("perdida (log)")
    ax2.set_title("Perdidas (conjunta)")
    ax2.legend(); ax2.grid(True, alpha=0.3, which="both")

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
    ax.set_title("PINN conjunta (autograd) — error de identificacion vs ruido")
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
            f"<td class='num'><span class='pill {cls_max}'>{r['max_err']:.1f}%</span></td></tr>"
        )

    imgs = "".join(
        f"<h3>noise_std = {r['noise_std']:.2f}</h3>"
        f"<img src='data:image/png;base64,{_b64(r['fig_path'])}'>"
        f"<img src='data:image/png;base64,{_b64(r['conv_path'])}'>"
        for r in results
    )
    img_cmp = f"<img src='data:image/png;base64,{_b64(fig_cmp_path)}'>"

    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>PINN canonica (autograd, conjunta) — Wilson-Cowan</title>
<style>
  body{{font-family:-apple-system,"Segoe UI",Roboto,Arial,sans-serif;color:#1c2733;
       line-height:1.55;margin:0;background:#f4f6f8;padding:2rem 1rem}}
  .wrap{{max-width:960px;margin:0 auto}}
  header{{background:#1f4e79;color:#fff;border-radius:12px 12px 0 0;padding:1.5rem 1.8rem}}
  header h1{{margin:0;font-size:1.35rem}} header p{{margin:.35rem 0 0;opacity:.9;font-size:.93rem}}
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
  .nota{{background:#e8f0f8;border-left:4px solid #1f4e79;padding:.7rem 1rem;
         border-radius:0 6px 6px 0;margin:1rem 0;font-size:.92rem}}
  footer{{color:#5b6770;font-size:.82rem;text-align:center;margin-top:1.3rem}}
</style></head><body><div class="wrap">
<header>
  <h1>PINN canonica — autograd + entrenamiento conjunto + multi-trayectoria</h1>
  <p>Identificacion parametrica de Wilson-Cowan · metodo PINN "sin asteriscos" · ITBA 2026</p>
</header>
<main>
  <div class="nota">
    <b>Que la distingue del metodo de <code>noise_sweep.py</code>:</b> aca las derivadas
    salen por <b>autograd</b> (no diferencias finitas) y la red + los 4 pesos fisicos se
    entrenan <b>a la vez</b> con la fisica en la perdida (no en dos etapas con la red
    congelada). Es la PINN canonica; el otro metodo es el surrogate + matching clasico
    que sirve de comparacion.
  </div>

  <h2>Configuracion</h2>
  <table>
    <tr><th>Trayectorias</th><td>{len(TRAJ_STIMULI)} — P y Q distintos (incl. una con Q alto para identificar wII)</td></tr>
    <tr><th>Entrenamiento</th><td>conjunto, Adam, {EPOCHS} epochs · autograd · raw_w compartido (lr MLP={LR}, lr raw_w={LR_W})</td></tr>
    <tr><th>Pesos de perdida</th><td>datos={W_DATA} · fisica={W_PHYSICS} (lambda) · ic={W_IC}</td></tr>
    <tr><th>Red</th><td>MLP {HIDDEN_DIM}x{N_LAYERS}, tanh · Fourier features={N_FOURIER} (escala {FOURIER_SCALE})</td></tr>
    <tr><th>Validacion temporal</th><td>datos del {100*(1-VAL_FRACTION):.0f}% inicial; fisica en todo el dominio; test el {100*VAL_FRACTION:.0f}% final</td></tr>
    <tr><th>Verdaderos</th><td>wEE=6.4 · wEI=4.8 · wIE=6.0 · wII=1.2</td></tr>
  </table>

  <h2>Resultados por nivel de ruido</h2>
  <table>
    <tr>
      <th class="num">noise_std</th>
      <th class="num">wEE err%</th><th class="num">wEI err%</th>
      <th class="num">wIE err%</th><th class="num">wII err%</th>
      <th class="num">MSE train</th><th class="num">MSE val</th>
      <th class="num">max err%</th>
    </tr>
    {filas}
  </table>

  <h2>Error parametrico vs nivel de ruido</h2>
  {img_cmp}

  <h2>Ajuste + convergencia de pesos (por nivel de ruido)</h2>
  {imgs}
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
    print(f"=== PINN CONJUNTA (autograd, multi-trayectoria): {len(NOISE_LEVELS)} niveles ===")
    results = []
    for noise_std in NOISE_LEVELS:
        results.append(run_one(noise_std))

    fig_cmp = OUT_DIR / "figures" / "joint_noise_sweep_comparison.png"
    _plot_comparison(results, fig_cmp)
    _build_html(results, fig_cmp, OUT_DIR / "reporte_pinn_joint.html")

    print("\n=== RESUMEN FINAL (PINN conjunta) ===")
    print(f"{'noise_std':>10} {'wEE%':>7} {'wEI%':>7} {'wIE%':>7} {'wII%':>7} {'max%':>7}")
    for r in results:
        print(
            f"{r['noise_std']:>10.2f} "
            + " ".join(f"{r['errors'][k]:>7.2f}" for k in IDENTIFY)
            + f" {r['max_err']:>7.2f}"
        )


if __name__ == "__main__":
    main()
