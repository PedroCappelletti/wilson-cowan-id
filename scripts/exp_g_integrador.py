#!/usr/bin/env python3
# =============================================================================
#  G — ¿EL ERROR DEL INTEGRADOR CONTAMINA LOS RESULTADOS DE RUIDO?
# =============================================================================
#
#  El problema. Los scripts del proyecto generan los datos con solve_ivp y
#  rtol=1e-3. Contra una referencia de alta precision eso acumula un error de
#  ~1.5e-2. Y los barridos de robustez usan sigma = 0, 0.01, 0.05, 0.10.
#
#  O sea que el error de integracion (0.015) cae ENTRE los dos primeros niveles
#  de ruido, y el caso "limpio" (sigma=0) en realidad no lo es: arrastra un
#  error del tamano del segundo nivel del barrido. Ademas es un error
#  SISTEMATICO (la misma trayectoria da siempre el mismo desvio), no aleatorio,
#  asi que el promediado no lo elimina.
#
#  Que hace este script. Genera el MISMO dataset multi-escenario de tres
#  maneras y identifica los 10 parametros con cada una:
#
#     historico  : solve_ivp RK45 rtol=1e-3   (lo que se usa hoy)
#     paso_fijo  : RK4 de paso fijo           (el camino nuevo)
#     referencia : solve_ivp DOP853 rtol=1e-11 (la verdad numerica)
#
#  Si los tres dan lo mismo, el error de integracion es inocuo y no hay nada que
#  corregir. Si el historico se aparta, hay que regenerar los resultados de
#  robustez con un integrador mejor.
#
#  USO:  python scripts/exp_g_integrador.py
# =============================================================================

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import numpy as np
import torch

torch.set_num_threads(2)

from src.wilson_cowan import WilsonCowan, WilsonCowanParams, NoPerturbation
from src.neural_ode.graybox_train import TrainConfig, fit, ALL_P

from gen_multi_dataset import build_scenarios, T_SPAN, N_EVAL, I0, E0

OUT = Path("results/uncertainty/g_integrador.json")
NIVELES_RUIDO = (0.0, 0.01)
SEED = 42

INTEGRADORES = {
    "historico":  dict(modo="ivp", method="RK45",  rtol=1e-3,  atol=1e-6),
    "paso_fijo":  dict(modo="rk4", n_sub=4),
    "referencia": dict(modo="ivp", method="DOP853", rtol=1e-11, atol=1e-13),
}


def genera(cfg: dict) -> dict:
    """Construye el dataset multi-escenario con el integrador indicado."""
    params = WilsonCowanParams()
    t_eval = np.linspace(T_SPAN[0], T_SPAN[1], N_EVAL)
    I_all, E_all, P_all, Q_all, is_test = [], [], [], [], []
    for label, Pf, Qf, test in build_scenarios():
        if cfg["modo"] == "rk4":
            # NoPerturbation = mismas ecuaciones, pero por el camino de paso fijo
            m = WilsonCowan(params=params, P=Pf, Q=Qf, perturbation=NoPerturbation())
            sol = m.simulate(I0=I0, E0=E0, t_span=T_SPAN, t_eval=t_eval,
                             n_sub=cfg["n_sub"])
        else:
            m = WilsonCowan(params=params, P=Pf, Q=Qf)
            sol = m.simulate(I0=I0, E0=E0, t_span=T_SPAN, t_eval=t_eval,
                             rel_tol=cfg["rtol"], abs_tol=cfg["atol"],
                             method=cfg["method"])
        I_all.append(sol["I"]); E_all.append(sol["E"])
        P_all.append(sol["P"]); Q_all.append(sol["Q"])
        is_test.append(test)
    return {
        "I": np.stack(I_all), "E": np.stack(E_all),
        "P": np.stack(P_all), "Q": np.stack(Q_all),
        "is_test": np.asarray(is_test),
        "dt": float(t_eval[1] - t_eval[0]),
        "true": {k: float(getattr(params, k)) for k in ALL_P},
    }


def con_ruido(d: dict, sigma: float, seed: int) -> dict:
    """Agrega ruido de OBSERVACION (despues de integrar, no toca la dinamica)."""
    if sigma <= 0:
        out = dict(d)
    else:
        rng = np.random.default_rng(seed)
        out = dict(d)
        out["I"] = d["I"] + rng.normal(0, sigma, d["I"].shape)
        out["E"] = d["E"] + rng.normal(0, sigma, d["E"].shape)
    it = out["is_test"]
    return {
        "I": out["I"][~it], "E": out["E"][~it], "P": out["P"][~it], "Q": out["Q"][~it],
        "I_te": out["I"][it], "E_te": out["E"][it],
        "P_te": out["P"][it], "Q_te": out["Q"][it],
        "dt": out["dt"], "true": out["true"],
    }


def main():
    print("=== G · ¿el integrador contamina los resultados de ruido? ===\n")

    datos = {}
    ref = None
    print("  Desvio de la trayectoria contra la referencia de alta precision:")
    for nombre, cfg in INTEGRADORES.items():
        datos[nombre] = genera(cfg)
        if nombre == "referencia":
            ref = datos[nombre]
    for nombre in INTEGRADORES:
        d = datos[nombre]
        err = float(np.sqrt(np.mean((d["E"] - ref["E"]) ** 2 +
                                    (d["I"] - ref["I"]) ** 2) / 2))
        mx = float(np.max(np.abs(d["E"] - ref["E"])))
        print(f"    {nombre:11} RMS = {err:.2e}   max = {mx:.2e}")
        d["err_vs_ref"] = err

    filas = []
    print(f"\n  Identificacion de los 10 parametros (arranque ignorante):")
    print(f"  {'integrador':12} {'sigma':>6} {'err_medio%':>11} {'err_max%':>10} "
          f"{'peor':>8} {'wII%':>8}")
    for sigma in NIVELES_RUIDO:
        for nombre in INTEGRADORES:
            data = con_ruido(datos[nombre], sigma, SEED)
            cfg = TrainConfig(variant="whitebox", epochs=1500, lbfgs_steps=30,
                              verbose=False)
            r = fit(data, cfg)
            peor = max(r["param_errors"], key=r["param_errors"].get)
            print(f"  {nombre:12} {sigma:6.2f} {r['mean_param_error']:11.2f} "
                  f"{r['max_param_error']:10.2f} {peor:>8} "
                  f"{r['param_errors']['wII']:8.2f}", flush=True)
            filas.append({"integrador": nombre, "sigma": sigma,
                          "err_vs_ref": datos[nombre]["err_vs_ref"],
                          "mean_param_error": r["mean_param_error"],
                          "max_param_error": r["max_param_error"],
                          "peor": peor, "param_errors": r["param_errors"],
                          "mse_test": r["mse_test"]})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(filas, indent=2))

    # --- Veredicto ---
    print("\n=== VEREDICTO ===")
    for sigma in NIVELES_RUIDO:
        sub = {f["integrador"]: f for f in filas if f["sigma"] == sigma}
        if "historico" in sub and "referencia" in sub:
            h, r = sub["historico"]["mean_param_error"], sub["referencia"]["mean_param_error"]
            dif = abs(h - r)
            rel = 100 * dif / max(r, 1e-9)
            print(f"  sigma={sigma}: historico {h:.2f}% vs referencia {r:.2f}%  "
                  f"(diferencia {dif:.2f} pp, {rel:.0f}% relativo)")
    print(f"\n  -> {OUT}")


if __name__ == "__main__":
    main()
