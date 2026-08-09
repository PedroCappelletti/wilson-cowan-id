#!/usr/bin/env python3
# =============================================================================
#  F6 — LAZO CERRADO: ¿EL GRAY-BOX RECUPERA LO QUE EL MISMATCH DEGRADA?
# =============================================================================
#
#  Es el cierre end-to-end del roadmap. El proyecto ya mostro que la fragilidad
#  de la identificacion NO se propaga al control cuando el modelo es correcto
#  (la accion integral del IMC la absorbe). La pregunta nueva es que pasa cuando
#  el modelo esta ESTRUCTURALMENTE EQUIVOCADO.
#
#  Todas las corridas se hacen contra LA MISMA planta: la real perturbada (el
#  "cerebro"). Lo unico que cambia es con que conocimiento se construye el
#  controlador:
#
#    1. oraculo    : θ verdaderos. Cota inferior inalcanzable en la practica.
#    2. white-box  : θ̂ identificados ignorando el hueco  -> cuanto degrada?
#    3. gray-box   : θ̂ + ĝ en la cancelacion             -> cuanto recupera?
#
#  La cancelacion gray-box es exacta y explicita porque ĝ depende solo del
#  estado (ver IMCController.__init__). Con g(I,E,P,Q) habria que resolver un
#  punto fijo en cada paso.
#
#  USO:  python scripts/exp_f6_closed_loop.py --graybox <ckpt.pt>
# =============================================================================

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

torch.set_num_threads(2)

from src.wilson_cowan import WilsonCowanParams, default_uncertainty, NoPerturbation
from src.neural_ode import (
    GrayBoxWC, IMCController, simulate_closed_loop, theta_gamma_refs,
    make_perturbed_plant, make_graybox_correction,
)
from src.neural_ode.graybox_train import ALL_P

OUT = Path("results/uncertainty/f6_closed_loop.json")
T_SPAN = (0.0, 50.0)
DT = 0.005


def fixed_de(params: dict) -> dict:
    """Arma el dict `fixed` que necesita el controlador a partir de 10 params."""
    ae, ai = params["ae"], params["ai"]
    the, thi = params["thetae"], params["thetai"]
    return {"te": params["te"], "ti": params["ti"], "ae": ae, "ai": ai,
            "thetae": the, "thetai": thi,
            "ke": 1.0 / (1.0 + np.exp(ae * the)),
            "ki": 1.0 / (1.0 + np.exp(ai * thi))}


def carga_ckpt(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = GrayBoxWC({k: 1.0 for k in ALL_P}, {k: 1.0 for k in ("wEE", "wEI", "wIE", "wII")},
                  learnable_weights=True, learnable_params=True,
                  use_correction=ck.get("use_correction", False),
                  correction_inputs=ck.get("correction_inputs", "xpq"))
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


def rmse(sol, n_skip=5):
    """RMSE de seguimiento, descartando el transitorio inicial."""
    n0 = len(sol["t"]) // n_skip
    eE = float(np.sqrt(np.mean((sol["E"][n0:] - sol["rE"][n0:]) ** 2)))
    eI = float(np.sqrt(np.mean((sol["I"][n0:] - sol["rI"][n0:]) ** 2)))
    return eI, eE


def corre(nombre, fixed, weights, planta, g_hat=None, sat_ref=None):
    ctrl = IMCController(fixed, weights, g_hat=g_hat, sat_ref=sat_ref)
    refs = theta_gamma_refs(freq_hz=120.0, time_in_ms=True)
    sol = simulate_closed_loop(planta, ctrl, refs, t_span=T_SPAN, dt=DT)
    eI, eE = rmse(sol)
    # Fraccion de pasos con la actuacion saturada. Es el chequeo de salud que
    # falta si uno solo mira si el RMSE es finito: con la saturacion puesta, un
    # g_hat divergente igual da un numero finito y de aspecto razonable.
    sat = max(ctrl.n_sat_i, ctrl.n_sat_e) / max(ctrl.n_pasos, 1)
    ok = bool(np.isfinite(eE) and np.isfinite(eI) and sat < 0.5)
    aviso = ""
    if not np.isfinite(eE) or not np.isfinite(eI):
        aviso = "   <-- DIVERGIO"
    elif sat >= 0.5:
        aviso = f"   <-- SATURADO {100*sat:.0f}% de los pasos (no comparable)"
    elif sat > 0.05:
        aviso = f"   (saturado {100*sat:.0f}%)"
    print(f"  {nombre:34} RMSE_I={eI:.4f}  RMSE_E={eE:.4f}{aviso}", flush=True)
    return {"nombre": nombre, "rmse_I": eI, "rmse_E": eE, "ok": ok,
            "frac_saturado": sat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--whitebox", default=None,
                    help="por defecto, el white-box del MISMO eps que la planta")
    ap.add_argument("--graybox", default=None, help="checkpoint .pt de la mejor variante")
    a = ap.parse_args()
    # El white-box tiene que ser el identificado al MISMO nivel de mismatch que
    # la planta contra la que se lo evalua; si no, la fila 2 de la tabla usa
    # parametros de otro experimento.
    if a.whitebox is None:
        a.whitebox = f"results/uncertainty/f2_eps{a.eps:g}.json"

    p_true = WilsonCowanParams()
    true_params = {k: getattr(p_true, k) for k in ALL_P}
    w_true = {k: true_params[k] for k in ("wEE", "wEI", "wIE", "wII")}

    # LA PLANTA es siempre la misma: la real con incertidumbre dinamica.
    pert = default_uncertainty(a.eps)
    planta = make_perturbed_plant(p_true, pert)
    print(f"=== F6 · planta perturbada (eps={a.eps}, {pert.metadata()['pert_name']}) ===")

    filas = []
    # Todos los controladores enfrentan la MISMA planta, asi que tienen que tener
    # el mismo rango de actuacion. Si cada uno derivara sus topes de su propio
    # theta_hat, el que tenga el `ai` estimado mas chico tendria mas autoridad de
    # control y la tabla mediria eso en vez de la calidad del modelo.
    SAT = fixed_de(true_params)

    # --- Referencia sin perturbacion, para ver cuanto molesta el hueco.
    planta_limpia = make_perturbed_plant(p_true, NoPerturbation())
    filas.append(corre("0. planta LIMPIA + theta verdaderos",
                       fixed_de(true_params), w_true, planta_limpia, sat_ref=SAT))

    # --- 1. Oraculo: θ verdaderos contra la planta perturbada.
    filas.append(corre("1. oraculo (theta verdaderos)",
                       fixed_de(true_params), w_true, planta, sat_ref=SAT))

    # --- 2. White-box: los parametros identificados ignorando el hueco.
    wb_path = Path(a.whitebox)
    if wb_path.exists():
        wb = json.loads(wb_path.read_text())
        ph = wb["params"] if isinstance(wb, dict) else wb[0]["params"]
        filas.append(corre("2. white-box (theta_hat sin correccion)",
                           fixed_de(ph), {k: ph[k] for k in w_true}, planta,
                           sat_ref=SAT))
    else:
        print(f"  (falta {wb_path}: no se pudo evaluar el white-box)")

    # --- 3. Gray-box: θ̂ + la correccion aprendida en la cancelacion.
    if a.graybox:
        m, ck = carga_ckpt(a.graybox)
        if float(ck.get("eps", a.eps)) != a.eps:
            raise SystemExit(f"ERROR: el checkpoint se entreno con eps={ck['eps']} "
                             f"pero la planta es eps={a.eps}.")
        ph = ck["params"]
        fx = fixed_de(ph)
        wh = {k: ph[k] for k in w_true}
        filas.append(corre("3a. gray-box theta_hat, SIN cancelar g", fx, wh, planta,
                           sat_ref=SAT))
        g_hat = make_graybox_correction(m)
        if g_hat is None:
            print("  (el checkpoint no tiene correccion: no hay nada que cancelar)")
        else:
            filas.append(corre("3b. gray-box theta_hat + g cancelada", fx, wh, planta,
                               g_hat=g_hat, sat_ref=SAT))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"eps": a.eps, "filas": filas}, indent=2))
    print(f"\n  -> {OUT}")


if __name__ == "__main__":
    main()
