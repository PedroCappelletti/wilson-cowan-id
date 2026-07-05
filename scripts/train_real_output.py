#!/usr/bin/env python3
# =============================================================================
#  IDENTIFICACION WILSON-COWAN POR SALIDA  (datos reales, LFP)
#  ---  MULTIPLE SHOOTING con estado latente por ventana  ---
# =============================================================================
#
#  Solo observamos un escalar s (el LFP), NO el estado [I,E]. Por eso:
#    - el estimulo entra a E:  P = c_P * u,  Q = 0   (decision D2);
#    - la salida modelada es    y = c_out * (E - I)  (LFP, decision D3);
#    - x0=[I0,E0] de cada VENTANA es una variable latente (no lo medimos).
#
#  POR QUE MULTIPLE SHOOTING Y NO UN SOLO ROLLOUT:
#    Un rollout de ~2400 pasos ajustado con MSE en el tiempo colapsa a "salida
#    cero": si la fase no coincide casi exacto, predecir 0 tiene MENOS error que
#    una oscilacion desfasada -> el gradiente cae al minimo trivial. Ademas es
#    lentisimo (loop secuencial). Cortar en VENTANAS cortas (~0.8 s) con un
#    estado latente por ventana:
#       (1) hace la fase LOCAL -> tratable, no colapsa;
#       (2) batchea las ventanas -> ~37x mas rapido;
#       (3) una penalizacion de CONTINUIDAD (x0 de la ventana k+1 ~ fin de la k)
#           re-ata todo, asi los parametros WC siguen identificandose globalmente.
#
#  Loss de datos: por ventana se comparan y y s DEMEDIADOS (quitar la media local
#  aproxima el pasa-banda [0.5,19] Hz sin filtro global; el modelo ya es low-pass).
#
#  EVALUACION (el test honesto): rollout LIBRE (sin resets) de toda la grabacion
#  con los parametros identificados, y se compara en el dominio FILTRADO (FFT
#  banda [0.5,19] Hz). Mide si el modelo REPRODUCE la trayectoria, no solo si
#  interpola ventana a ventana.
#
#  MODOS:  single (F2) | loo (protocolo A) | forecast (protocolo B)
#  USO:
#    python scripts/train_real_output.py --mode single --rec 0 --variant v0 --plot
#    python scripts/train_real_output.py --mode loo --variant v0
#    python scripts/train_real_output.py --mode forecast --rec 0 --frac 0.7 --plot
# =============================================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.neural_ode import GrayBoxWC, rollout

F_LO, F_HI = 0.5, 19.0
WARMUP_S = 1.0
WEIGHTS = ("wEE", "wEI", "wIE", "wII")
PHYS = ("te", "ti", "ae", "ai", "thetae", "thetai")


def fft_bandpass(x: torch.Tensor, fs: float, f_lo=F_LO, f_hi=F_HI) -> torch.Tensor:
    T = x.shape[0]
    X = torch.fft.rfft(x, dim=0)
    freqs = torch.fft.rfftfreq(T, d=1.0 / fs)
    mask = ((freqs >= f_lo) & (freqs <= f_hi)).to(x.dtype)
    return torch.fft.irfft(X * mask, n=T, dim=0)


class OutputModel(torch.nn.Module):
    """WC gray-box + ganancias c_P (u->E), c_out (LFP). x0 latentes se guardan aparte."""
    def __init__(self, variant: str):
        super().__init__()
        fixed = dict(te=0.02, ti=0.04, ae=1.2, ai=1.0, thetae=2.8, thetai=4.0, ke=0.0, ki=0.0)
        w_init = dict(wEE=6.4, wEI=4.8, wIE=6.0, wII=1.2)
        self.wc = GrayBoxWC(fixed, w_init, learnable_weights=True,
                            learnable_params=True, use_correction=(variant == "v1"))
        self.c_P = torch.nn.Parameter(torch.tensor(3.0))
        self.c_out = torch.nn.Parameter(torch.tensor(0.05))

    def y_of(self, traj):                      # traj (...,2)=[I,E] -> LFP
        return self.c_out * (traj[..., 1] - traj[..., 0])


# ---------------------------------------------------------------------------
#  Ventanas: para cada grabacion, trocea en ventanas de W pasos.
# ---------------------------------------------------------------------------
def build_windows(recs, us, ss, W):
    Pw, Sw, meta = [], [], []            # meta: (rec, k) para continuidad
    for r in recs:
        u, s = us[r], ss[r]
        T = len(u); nwin = (T - 1) // W
        for k in range(nwin):
            a = k * W
            Pw.append(u[a:a + W])
            Sw.append(s[a:a + W + 1])
            meta.append((r, k))
    Pw = torch.stack(Pw).T.unsqueeze(-1)          # (W, Nw, 1)
    Sw = torch.stack(Sw).T                         # (W+1, Nw)
    return Pw, Sw, meta


def cont_pairs(meta):
    """indices (i,j) de ventanas consecutivas de la misma grabacion."""
    pairs = []
    for i in range(len(meta) - 1):
        if meta[i][0] == meta[i + 1][0] and meta[i + 1][1] == meta[i][1] + 1:
            pairs.append((i, i + 1))
    return pairs


def demean(x):                                     # quita media por ventana (dim tiempo)
    return x - x.mean(dim=0, keepdim=True)


def fit_windows(model, X0, Pw, Sw, pairs, dt, epochs, lr_w, lr_phys, lr_gain,
                lam_cont, lbfgs, verbose=True):
    Qw = torch.zeros_like(Pw)
    groups = [
        {"params": [model.wc.raw_w], "lr": lr_w},
        {"params": [getattr(model.wc, f"raw_{k}") for k in PHYS], "lr": lr_phys},
        {"params": [model.c_P, model.c_out], "lr": lr_gain},
        {"params": [X0], "lr": lr_gain},
    ]
    if model.wc.use_correction:
        groups.append({"params": list(model.wc.g.parameters()), "lr": lr_gain})
    pi = torch.tensor([p[0] for p in pairs]); pj = torch.tensor([p[1] for p in pairs])

    def losses():
        traj = rollout(model.wc, X0, Pw, Qw, dt)      # (W+1,Nw,2)
        y = model.y_of(traj)                           # (W+1,Nw)
        data = ((demean(y) - demean(Sw)) ** 2).mean()
        cont = ((X0[pj] - traj[-1][pi]) ** 2).mean()
        return data, cont

    def total():
        d, c = losses()
        return d + lam_cont * c

    opt = torch.optim.Adam(groups)
    for ep in range(epochs):
        opt.zero_grad(); loss = total(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 5.0)
        opt.step()
        if verbose and (ep % max(epochs // 8, 1) == 0 or ep == epochs - 1):
            d, c = losses(); p = model.wc.params_dict()
            print(f"  {ep:4d} | data={d.item():.3e} cont={c.item():.3e} | "
                  f"te={p['te']:.4f} ti={p['ti']:.4f} wEE={p['wEE']:.2f} wEI={p['wEI']:.2f} "
                  f"c_P={model.c_P.item():.2f} c_out={model.c_out.item():.3f}")

    if lbfgs > 0:
        params = [p for g in groups for p in g["params"]]
        opt2 = torch.optim.LBFGS(params, lr=0.5, max_iter=20, line_search_fn="strong_wolfe")
        def closure():
            opt2.zero_grad(); l = total(); l.backward(); return l
        for _ in range(lbfgs):
            opt2.step(closure)
        if verbose:
            d, c = losses(); print(f"  --- L-BFGS --- data={d.item():.3e} cont={c.item():.3e}")


@torch.no_grad()
def free_rollout(model, u, dt):
    """Rollout LIBRE de toda la grabacion desde reposo. Devuelve y (T,)."""
    T = len(u)
    P = (model.c_P * u).reshape(T, 1, 1)
    x0 = torch.zeros(1, 2)
    traj = rollout(model.wc, x0, P, torch.zeros_like(P), dt)[:-1, 0, :]   # (T,2)
    return model.y_of(traj)


def eval_filtered(model, u, s, fs, i0, sl=None):
    dt = 1.0 / fs
    y = free_rollout(model, u, dt)
    yb, sb = fft_bandpass(y, fs), fft_bandpass(s, fs)
    reg = slice(i0, None) if sl is None else sl
    yr, sr = yb[reg], sb[reg]
    nrmse = float(torch.sqrt(((yr - sr) ** 2).mean()) / (sr.std() + 1e-12))
    ss_res = ((yr - sr) ** 2).sum(); ss_tot = ((sr - sr.mean()) ** 2).sum()
    r2 = float(1 - ss_res / (ss_tot + 1e-12))
    return nrmse, r2, yb, sb


def report_params(model):
    p = model.wc.params_dict()
    print("  WC:", " ".join(f"{k}={p[k]:.3f}" for k in WEIGHTS + PHYS),
          f"| c_P={model.c_P.item():.3f} c_out={model.c_out.item():.4f}")
    return p


def load(fs_target):
    d = np.load(f"data/processed/real/data8_fs{fs_target}.npz", allow_pickle=True)
    fs = float(d["fs"])
    us = [torch.tensor(a, dtype=torch.float32) for a in d["u"]]
    ss = [torch.tensor(a, dtype=torch.float32) for a in d["s"]]
    return fs, us, ss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "loo", "forecast"], default="single")
    ap.add_argument("--rec", type=int, default=0)
    ap.add_argument("--variant", choices=["v0", "v1"], default="v0")
    ap.add_argument("--fs", type=int, default=125)
    ap.add_argument("--W", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lbfgs", type=int, default=60)
    ap.add_argument("--frac", type=float, default=0.7)
    ap.add_argument("--lam_cont", type=float, default=10.0)
    ap.add_argument("--lr_w", type=float, default=3e-2)
    ap.add_argument("--lr_phys", type=float, default=1e-2)
    ap.add_argument("--lr_gain", type=float, default=2e-2)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    fs, us, ss = load(args.fs)
    dt = 1.0 / fs; i0 = int(WARMUP_S * fs)
    n_rec = len(us)
    print(f"=== train_real_output MS | mode={args.mode} variant={args.variant} "
          f"fs={fs:.0f}Hz W={args.W}({args.W/fs:.2f}s) ===")

    if args.mode == "single":
        model = OutputModel(args.variant)
        Pw, Sw, meta = build_windows([args.rec], us, ss, args.W)
        X0 = torch.nn.Parameter(torch.zeros(len(meta), 2))
        print(f"    {len(meta)} ventanas")
        fit_windows(model, X0, Pw, Sw, cont_pairs(meta), dt, args.epochs,
                    args.lr_w, args.lr_phys, args.lr_gain, args.lam_cont, args.lbfgs)
        report_params(model)
        nrmse, r2, yb, sb = eval_filtered(model, us[args.rec], ss[args.rec], fs, i0)
        print(f"\n  rec{args.rec} [rollout LIBRE, filtrado]: NRMSE={nrmse:.3f}  R2={r2:.3f}")
        if args.plot:
            plot_fit(sb, yb, fs, i0, f"single_rec{args.rec}_{args.variant}")

    elif args.mode == "loo":
        print("  Protocolo A (leave-one-out): params compartidos, held-out por rollout libre")
        for held in range(n_rec):
            tr = [r for r in range(n_rec) if r != held]
            model = OutputModel(args.variant)
            Pw, Sw, meta = build_windows(tr, us, ss, args.W)
            X0 = torch.nn.Parameter(torch.zeros(len(meta), 2))
            fit_windows(model, X0, Pw, Sw, cont_pairs(meta), dt, args.epochs,
                        args.lr_w, args.lr_phys, args.lr_gain, args.lam_cont, args.lbfgs, verbose=False)
            p = model.wc.params_dict()
            # error in-sample (train) y held-out, ambos por rollout libre filtrado
            tr_r2 = np.mean([eval_filtered(model, us[r], ss[r], fs, i0)[1] for r in tr])
            _, ho_r2, _, _ = eval_filtered(model, us[held], ss[held], fs, i0)
            print(f"  held rec{held}: R2 train={tr_r2:.3f}  held-out={ho_r2:.3f} | "
                  f"te={p['te']:.3f} ti={p['ti']:.3f} wEE={p['wEE']:.2f} wEI={p['wEI']:.2f} "
                  f"wIE={p['wIE']:.2f} wII={p['wII']:.2f}")

    elif args.mode == "forecast":
        model = OutputModel(args.variant)
        n = len(us[args.rec]); nfit = int(n * args.frac)
        u_fit, s_fit = us[args.rec][:nfit], ss[args.rec][:nfit]
        Pw, Sw, meta = build_windows([0], [u_fit], [s_fit], args.W)
        X0 = torch.nn.Parameter(torch.zeros(len(meta), 2))
        print(f"    ajuste en primer {args.frac:.0%} ({nfit} pts, {len(meta)} ventanas)")
        fit_windows(model, X0, Pw, Sw, cont_pairs(meta), dt, args.epochs,
                    args.lr_w, args.lr_phys, args.lr_gain, args.lam_cont, args.lbfgs)
        report_params(model)
        _, r2_in, yb, sb = eval_filtered(model, us[args.rec], ss[args.rec], fs, i0, sl=slice(i0, nfit))
        _, r2_fc, _, _ = eval_filtered(model, us[args.rec], ss[args.rec], fs, i0, sl=slice(nfit, None))
        print(f"\n  rec{args.rec} frac={args.frac}: R2 in-sample={r2_in:.3f}  R2 forecast(cola)={r2_fc:.3f}")
        if args.plot:
            plot_forecast(sb, yb, fs, i0, nfit, f"forecast_rec{args.rec}_{args.variant}")


def plot_fit(sb, yb, fs, i0, tag):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from scipy.signal import spectrogram
    t = np.arange(len(sb)) / fs
    fig, ax = plt.subplots(3, 1, figsize=(11, 8))
    ax[0].plot(t, sb.numpy(), lw=.6, label="s real (filt)")
    ax[0].plot(t, yb.numpy(), lw=.6, alpha=.8, label="modelo (rollout libre, filt)")
    ax[0].axvspan(0, i0 / fs, color="gray", alpha=.15); ax[0].legend(loc="upper right")
    ax[0].set_title(f"Ajuste por salida (MS) — {tag}"); ax[0].set_ylabel("s")
    zi = slice(i0, i0 + int(4 * fs))
    ax[1].plot(t[zi], sb.numpy()[zi], lw=1, label="real"); ax[1].plot(t[zi], yb.numpy()[zi], lw=1, label="modelo")
    ax[1].legend(); ax[1].set_title("zoom 4 s"); ax[1].set_ylabel("s")
    f, tt, Sxx = spectrogram(yb.numpy(), fs, nperseg=256, noverlap=192)
    ax[2].pcolormesh(tt, f, 10 * np.log10(Sxx + 1e-20), shading="gouraud", cmap="magma")
    ax[2].set_ylim(0, 20); ax[2].set_title("espectrograma modelo"); ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("Hz")
    fig.tight_layout(); out = f"results/figures/real_{tag}.png"; fig.savefig(out, dpi=130)
    print(f"  figura: {out}")


def plot_forecast(sb, yb, fs, i0, nfit, tag):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t = np.arange(len(sb)) / fs
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, sb.numpy(), lw=.6, label="s real")
    ax.plot(t, yb.numpy(), lw=.6, alpha=.8, label="modelo")
    ax.axvline(nfit / fs, color="red", ls="--", label="corte fit|forecast")
    ax.axvspan(0, i0 / fs, color="gray", alpha=.15); ax.legend(loc="upper right")
    ax.set_title(f"Forecast — {tag}"); ax.set_xlabel("t [s]")
    fig.tight_layout(); out = f"results/figures/real_{tag}.png"; fig.savefig(out, dpi=130)
    print(f"  figura: {out}")


if __name__ == "__main__":
    main()
