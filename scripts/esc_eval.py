#!/usr/bin/env python3
# =============================================================================
#  EVALUACION UNIFICADA DEL ESCALADO  (plan: docs/plan_escalado_progresivo.md)
# =============================================================================
#
#  Una sola vara para medir CUALQUIER modelo del escalado (white-box, gray-box,
#  estructurado, con memoria). Reporta las tres metricas del plan:
#
#    1. NRMSE % de rollout open-loop sobre los 7 estimulos de TEST (la reina).
#    2. R2 de la correccion IMPLICITA contra el Delta f verdadero del dataset.
#    3. Error porcentual de los 10 parametros WC.
#
#  La correccion implicita se define como
#        delta(x, ...) = f_modelo − f_WC(θ̂)
#  con los θ̂ APRENDIDOS. Para el gray-box aditivo coincide con g_φ; para el
#  estructurado y los modelos con memoria captura lo que sea que agregaron por
#  encima del WC puro. Asi la metrica 2 compara peras con peras.
#
#  Para los modelos con estado oculto, el estado se integra "teacher-forced":
#  se recorre la trayectoria REAL (I,E) observada y solo el estado oculto se
#  integra con la dinamica del modelo. Es la misma convencion con la que el
#  dataset guarda dfI/dfE (evaluados sobre la trayectoria real de la planta).
#
#  USO:  python scripts/esc_eval.py results/escalado/models/<tag>.pt data/processed/uncertain/<ds>.npz
# =============================================================================

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

torch.set_num_threads(4)

from src.neural_ode import GrayBoxWC, rollout
from src.neural_ode.graybox_train import ALL_P, WEIGHTS
from src.neural_ode.memory import LagGrayBox, LatentGrayBox


# =============================================================================
#  SECCION 1: CARGA (los checkpoints del escalado llevan un campo "kind")
# =============================================================================
def cargar(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    kind = ck.get("kind", "graybox")
    if kind == "graybox":
        m = GrayBoxWC({k: 1.0 for k in ALL_P}, {k: 1.0 for k in WEIGHTS},
                      learnable_weights=True, learnable_params=True,
                      use_correction=ck.get("use_correction", False),
                      correction_inputs=ck.get("correction_inputs", "x"),
                      structured=ck.get("structured", False))
    elif kind == "lag":
        m = LagGrayBox()
    elif kind == "latent":
        m = LatentGrayBox(n_hidden=ck.get("n_hidden", 2))
    else:
        raise ValueError(f"kind desconocido: {kind}")
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


def _wc_plano(params: dict) -> GrayBoxWC:
    """Wilson-Cowan puro con los parametros dados (la vara contra la que se mide
    la correccion implicita). learnable_params=True solo para que ke,ki se
    deriven de ae,thetae; no se entrena nada."""
    m = GrayBoxWC(params, {k: params[k] for k in WEIGHTS},
                  learnable_weights=True, learnable_params=True)
    m.eval()
    return m


# =============================================================================
#  SECCION 2: ROLLOUT OPEN-LOOP  (con o sin estados ocultos)
# =============================================================================
@torch.no_grad()
def _rollout_traj(m, I0, E0, P, Q, dt):
    T = len(P)
    x0 = torch.tensor([[I0, E0]], dtype=torch.float32)
    Ps = torch.tensor(P, dtype=torch.float32).reshape(T, 1, 1)
    Qs = torch.tensor(Q, dtype=torch.float32).reshape(T, 1, 1)
    if getattr(m, "n_hidden", 0) > 0:
        x0 = torch.cat([x0, m.h0(x0, Ps[0], Qs[0])], dim=-1)
    return rollout(m, x0, Ps[:-1], Qs[:-1], dt)[:, 0, :2].numpy()


@torch.no_grad()
def nrmse_test(m, d, solo_test=True):
    """NRMSE % por escenario de test (mismas convenciones que exp_reproduccion:
    normalizado por el rango pico-a-pico de la senal real, por canal)."""
    sel = d["is_test"].astype(bool)
    if not solo_test:
        sel = ~sel
    I, E, P, Q = d["I"][sel], d["E"][sel], d["P"][sel], d["Q"][sel]
    labels = [str(x) for x in d["labels"][sel]]
    dt = float(d["dt"])
    filas = []
    for s in range(len(I)):
        pred = _rollout_traj(m, I[s, 0], E[s, 0], P[s], Q[s], dt)
        real = np.stack([I[s], E[s]], 1)
        rng_ = real.max(0) - real.min(0)
        rng_[rng_ < 1e-9] = 1.0
        nr = 100 * np.sqrt(((pred - real) ** 2).mean(0)) / rng_
        corr = [float(np.corrcoef(pred[:, c], real[:, c])[0, 1]) for c in (0, 1)]
        filas.append({"label": labels[s], "nrmse_I": float(nr[0]),
                      "nrmse_E": float(nr[1]),
                      "nrmse": float(nr.mean()),
                      "corr_I": corr[0], "corr_E": corr[1]})
    return filas


# =============================================================================
#  SECCION 3: LA CORRECCION IMPLICITA CONTRA EL DELTA F VERDADERO
# =============================================================================
@torch.no_grad()
def _hidden_teacher_forced(m, I, E, P, Q, dt):
    """Integra SOLO el estado oculto recorriendo la trayectoria real (Euler).
    Devuelve (T, n_hidden)."""
    T = len(I)
    x = torch.tensor(np.stack([I, E], 1), dtype=torch.float32)
    Ps = torch.tensor(P, dtype=torch.float32).reshape(T, 1)
    Qs = torch.tensor(Q, dtype=torch.float32).reshape(T, 1)
    if hasattr(m, "filtered_inputs"):
        # el estado del actuador depende solo del comando: se computa exacto
        Ph, Qh = m.filtered_inputs(Ps.unsqueeze(-1), Qs.unsqueeze(-1), dt)
        return torch.cat([Ph[:, 0, :], Qh[:, 0, :]], dim=-1)
    h = m.h0(x[0:1], Ps[0:1], Qs[0:1])[0]
    hs = [h]
    for t in range(T - 1):
        y = torch.cat([x[t], h]).unsqueeze(0)
        dy = m(y, Ps[t:t + 1], Qs[t:t + 1])[0]
        h = h + dt * dy[2:]
        hs.append(h)
    return torch.stack(hs, dim=0)


@torch.no_grad()
def r2_delta(m, d, solo_test=True):
    """R2 de (f_modelo − f_WC(θ̂)) contra el (dfI, dfE) guardado en el dataset.
    Se evalua punto a punto sobre las trayectorias reales."""
    plano = _wc_plano(m.params_dict())
    sel = d["is_test"].astype(bool)
    if not solo_test:
        sel = ~sel
    I, E, P, Q = d["I"][sel], d["E"][sel], d["P"][sel], d["Q"][sel]
    dfI, dfE = d["dfI"][sel], d["dfE"][sel]
    dt = float(d["dt"])
    preds, tgts = [], []
    for s in range(len(I)):
        T = I.shape[1]
        x = torch.tensor(np.stack([I[s], E[s]], 1), dtype=torch.float32)
        Ps = torch.tensor(P[s], dtype=torch.float32).reshape(T, 1)
        Qs = torch.tensor(Q[s], dtype=torch.float32).reshape(T, 1)
        if getattr(m, "n_hidden", 0) > 0:
            hs = _hidden_teacher_forced(m, I[s], E[s], P[s], Q[s], dt)
            y = torch.cat([x, hs], dim=-1)
        else:
            y = x
        f_full = m(y, Ps, Qs)[..., :2]
        f_wc = plano.backbone(x, Ps, Qs)
        preds.append((f_full - f_wc).numpy())
        tgts.append(np.stack([dfI[s], dfE[s]], 1))
    pred = np.concatenate(preds)
    tgt = np.concatenate(tgts)
    ss_res = ((pred - tgt) ** 2).sum()
    ss_tot = ((tgt - tgt.mean(0)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


# =============================================================================
#  SECCION 4: TODO JUNTO
# =============================================================================
def evaluar(m, d, true: dict | None = None) -> dict:
    filas = nrmse_test(m, d)
    if true is None:
        true = {k: float(d[k]) for k in ALL_P}
    p = m.params_dict()
    perr = {k: 100.0 * abs(p[k] - true[k]) / abs(true[k]) for k in true}
    out = {
        "nrmse_test": float(np.mean([f["nrmse"] for f in filas])),
        "nrmse_I": float(np.mean([f["nrmse_I"] for f in filas])),
        "nrmse_E": float(np.mean([f["nrmse_E"] for f in filas])),
        "r2_delta_test": r2_delta(m, d),
        "mean_param_error": float(np.mean(list(perr.values()))),
        "max_param_error": float(max(perr.values())),
        "param_errors": perr,
        "por_escenario": filas,
    }
    if hasattr(m, "extras_dict"):
        out["extras"] = m.extras_dict()
    return out


def main():
    ckpt, data = sys.argv[1], sys.argv[2]
    m, _ = cargar(ckpt)
    d = np.load(data, allow_pickle=True)
    r = evaluar(m, d)
    print(f"{Path(ckpt).stem:28} NRMSE={r['nrmse_test']:6.2f}% "
          f"(I={r['nrmse_I']:.2f} E={r['nrmse_E']:.2f}) "
          f"R2df={r['r2_delta_test']:6.3f} err_param={r['mean_param_error']:6.2f}%")
    print(json.dumps({k: v for k, v in r.items() if k != "por_escenario"}, indent=2))


if __name__ == "__main__":
    main()
