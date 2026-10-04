#!/usr/bin/env python3
"""Piloto: el mismo gray-box, pero observando solo S en vez de I y E.
EN EVALUACIÓN, ver README.md. No forma parte del pipeline del escalado.

Qué cambia respecto de esc_run.py y nada más:
  - La pérdida compara S = h(I, E, P, Q) contra el S observado. I y E no se
    usan para entrenar; quedan en el dataset solo para el diagnóstico.
  - Sin I y E observados, el estado al arranque de cada ventana no se conoce.
    Pasa a ser una incógnita por ventana, con una penalización de continuidad
    contra el final de la ventana anterior, como en scripts/train_real_output.py.
    La primera ventana de cada escenario arranca del reposo, que sí se conoce
    porque el experimento lo fija.

El modelo, las variantes, el arranque ignorante y la selección por validación
son los mismos del escalado. Por ahora solo las variantes de build_model
(whitebox, B, H, K, S); el estado de filtro y la latent ODE se entrenan con
fit_aug y quedan para después.

USO
  python experimental/lfp/entrenar_lfp.py --data refrac1_amp__E_menos_I__snr20_chi1 \
      --variant whitebox --tag piloto_wb
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("WC_THREADS", "1")))

from src.neural_ode.graybox_train import (TrainConfig, build_model, make_windows,
                                          MejorVal, PHYS, ALL_P)
from src.neural_ode.integrate import rollout
from esc_eval import _rollout_lote, nrmse_test, r2_delta, seleccion
from observacion import PROXIES

DATOS = RAIZ / "data/processed/lfp_piloto"
SALIDA = RAIZ / "results/lfp_piloto"


def S_modelo(m, proxy, traj, P, Q):
    """S que predice el modelo. traj (..., 2) = [I, E]; P, Q del mismo tamaño
    que traj[..., 0]. El proxy puede depender de los pesos, y entonces usa los
    del modelo, no los verdaderos."""
    return PROXIES[proxy](traj[..., 0], traj[..., 1], P, Q, m.weights())


def actual(u):
    """Valor actual del comando: con historia llega como (..., K) y es el canal 0."""
    return u[..., 0]


@torch.no_grad()
def nrmse_S(m, d, proxy, conjunto, contra="S"):
    """NRMSE % de S en corrida libre desde el reposo, promedio por escenario."""
    sel = seleccion(d, conjunto)
    P, Q = d["P"][sel], d["Q"][sel]
    traj = torch.tensor(_rollout_lote(m, np.zeros(sel.sum()), np.zeros(sel.sum()),
                                      P, Q, float(d["dt"])))
    S_hat = S_modelo(m, proxy, traj, torch.tensor(P.T), torch.tensor(Q.T)).numpy().T
    S = d[contra][sel]
    rng = S.max(1) - S.min(1)
    return float(np.mean(100 * np.sqrt(((S_hat - S) ** 2).mean(1)) / rng))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="nombre en data/processed/lfp_piloto")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--hist", type=int, default=0)
    ap.add_argument("--n-fir", type=int, default=4)
    ap.add_argument("--lam-cont", type=float, default=1.0,
                    help="peso de la continuidad entre ventanas")
    ap.add_argument("--lr-x0", type=float, default=1e-2)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    d = dict(np.load(DATOS / f"{a.data}.npz", allow_pickle=True))
    proxy = str(d["proxy"])
    tr = seleccion(d, "train")
    dt = float(d["dt"])
    W = a.window

    cfg = TrainConfig(variant=a.variant, window=W, epochs=a.epochs, hist=a.hist,
                      n_fir=a.n_fir, seed=a.seed, val_every=a.val_every)
    torch.manual_seed(a.seed)
    m = build_model(cfg)

    I, E, P, Q, S = (d[k][tr] for k in ("I", "E", "P", "Q", "S"))
    # x0 de make_windows es el estado verdadero: no se usa para entrenar.
    _, Pw, Qw, _ = make_windows(I, E, P, Q, W, hist=a.hist)
    n, T = S.shape
    nwin = (T - 1) // W
    Sw = torch.tensor(np.stack([S[s, w * W:w * W + W] for s in range(n)
                                for w in range(nwin)], axis=1), dtype=torch.float32)
    primera = torch.tensor([w == 0 for s in range(n) for w in range(nwin)])
    sig = torch.tensor([i + 1 < n * nwin and (i + 1) % nwin != 0
                        for i in range(n * nwin)])
    idx = torch.arange(n * nwin)
    prev, nxt = idx[sig], idx[sig] + 1
    var_S = float(S.var())

    # Estados de arranque: la corrida libre del modelo recién inicializado,
    # que es consistente con su propia dinámica aunque esté lejos de la real.
    with torch.no_grad():
        libre = _rollout_lote(m, np.zeros(n), np.zeros(n), P, Q, dt)   # (T, n, 2)
    X0 = torch.nn.Parameter(torch.tensor(
        np.stack([libre[w * W, s] for s in range(n) for w in range(nwin)]),
        dtype=torch.float32))
    reposo = torch.tensor(np.stack([d["I"][tr][:, 0], d["E"][tr][:, 0]], 1),
                          dtype=torch.float32).repeat_interleave(nwin, 0)

    groups = [{"params": [m.raw_w], "lr": cfg.lr_w},
              {"params": [getattr(m, f"raw_{k}") for k in PHYS], "lr": cfg.lr_phys},
              {"params": [X0], "lr": a.lr_x0}]
    if m.use_correction:
        groups.append({"params": list(m.g.parameters()), "lr": cfg.lr_g})
    if m.structured:
        groups.append({"params": [m.raw_r_i, m.raw_r_e, m.raw_alpha], "lr": cfg.lr_phys})
    opt = torch.optim.Adam(groups)

    seg = None
    if seleccion(d, "val").any():
        seg = MejorVal(m, lambda mod: nrmse_S(mod, d, proxy, "val"))

    print(f"=== piloto S · {a.tag} · {a.variant} · {a.data} · proxy={proxy} "
          f"· {n * nwin} ventanas ===", flush=True)
    t0 = time.time()
    for ep in range(a.epochs):
        opt.zero_grad()
        x0 = torch.where(primera[:, None], reposo, X0)
        traj = rollout(m, x0, Pw, Qw, dt)                            # (W+1, Nw, 2)
        S_hat = S_modelo(m, proxy, traj[:-1], actual(Pw), actual(Qw))  # (W, Nw)
        dato = ((S_hat - Sw) ** 2).mean() / var_S
        cont = ((x0[nxt] - traj[-1][prev]) ** 2).mean()
        (dato + a.lam_cont * cont).backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 10.0)
        opt.step()
        if ep % 250 == 0 or ep == a.epochs - 1:
            err = np.mean([100 * abs(m.params_dict()[k] - float(d[k])) / abs(float(d[k]))
                           for k in ALL_P])
            print(f"    ep {ep:5d} | S={float(dato.detach()):.3e} cont={float(cont.detach()):.3e} "
                  f"| err_param={err:6.2f}%", flush=True)
        if seg and ((ep + 1) % a.val_every == 0 or ep == a.epochs - 1):
            print(f"    ep {ep:5d} | val_S={seg.evaluar(ep):6.2f}%", flush=True)

    val = seg.restaurar() if seg else {}
    mins = (time.time() - t0) / 60
    m.eval()

    true = {k: float(d[k]) for k in ALL_P}
    p = m.params_dict()
    perr = {k: 100 * abs(p[k] - true[k]) / abs(true[k]) for k in true}
    filas_IE = nrmse_test(m, d, "test")
    out = {
        "tag": a.tag, "data": a.data, "proxy": proxy, "variant": a.variant,
        "window": W, "epochs": a.epochs, "hist": a.hist, "n_fir": a.n_fir,
        "lam_cont": a.lam_cont, "seed": a.seed, "minutos": mins,
        "nrmse_S_test": nrmse_S(m, d, proxy, "test"),
        "nrmse_S_test_limpio": nrmse_S(m, d, proxy, "test", contra="S_limpio"),
        # Lo que el modelo nunca vio: si reconstruye I y E por separado.
        "nrmse_IE_oculto_test": float(np.mean([f["nrmse"] for f in filas_IE])),
        "r2_delta_test": r2_delta(m, d),
        "mean_param_error": float(np.mean(list(perr.values()))),
        "param_errors": perr, "params": p,
        **{k: v for k, v in val.items() if k != "historia_val"},
    }
    SALIDA.joinpath("models").mkdir(parents=True, exist_ok=True)
    torch.save({"state": m.state_dict(), "args": vars(a)},
               SALIDA / "models" / f"{a.tag}.pt")
    (SALIDA / f"{a.tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\n  RESULTADO {a.tag}: S_test={out['nrmse_S_test']:.2f}% "
          f"(limpio {out['nrmse_S_test_limpio']:.2f}) I,E ocultos={out['nrmse_IE_oculto_test']:.2f}% "
          f"err_param={out['mean_param_error']:.2f}% [{mins:.1f} min]", flush=True)


if __name__ == "__main__":
    main()
