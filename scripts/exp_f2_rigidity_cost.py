#!/usr/bin/env python3
# =============================================================================
#  F2 — EL COSTO DE LA RIGIDEZ
# =============================================================================
#
#  Pregunta: si la planta real tiene fisica que el modelo NO contempla, cuanto
#  se degrada la identificacion de los 10 parametros?
#
#  Este experimento NO usa la correccion neuronal: es el white-box de siempre
#  (train_neural_ode_full.py) corriendo sobre datos cada vez mas perturbados.
#  Sirve para DEMOSTRAR QUE EL PROBLEMA EXISTE antes de proponer la solucion.
#  Si el white-box aguantara perfecto, no habria nada que arreglar y todo el
#  gray-box seria decoracion.
#
#  Que se mira:
#    - error paramétrico (maximo y medio sobre los 10)
#    - MSE open-loop en estimulos de TEST (nunca vistos)
#    - QUE parametro se rompe primero -> se compara con lo que predice la FIM
#
#  USO:  python scripts/exp_f2_rigidity_cost.py
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

# 2 hilos por proceso: mas no acelera (medido) y deja lugar para correr varios
# eps en paralelo, que si escala.
torch.set_num_threads(2)

from src.neural_ode.graybox_train import TrainConfig, fit, load_split, ALL_P

EPS_GRID = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
DATA_DIR = Path("data/processed/uncertain")
OUT_DIR = Path("results/uncertainty")
OUT = OUT_DIR / "f2_rigidity.json"
EPOCHS = 1500
LBFGS = 30


def corre_uno(eps: float) -> dict:
    path = DATA_DIR / f"eps{eps:g}.npz"
    data = load_split(path)
    print(f"\n=== eps = {eps} — white-box (sin correccion), 10 parametros ===",
          flush=True)
    cfg = TrainConfig(variant="whitebox", epochs=EPOCHS, lbfgs_steps=LBFGS,
                      log_every=300)
    r = fit(data, cfg)
    peor = max(r["param_errors"], key=r["param_errors"].get)
    print(f"  err_max={r['max_param_error']:.2f}%  err_medio={r['mean_param_error']:.2f}%"
          f"  peor={peor}  mse_test={r['mse_test']:.3e}", flush=True)
    return {
        "eps": eps,
        "max_param_error": r["max_param_error"],
        "mean_param_error": r["mean_param_error"],
        "peor_param": peor,
        "param_errors": r["param_errors"],
        "params": r["params"],
        "mse_train": r["mse_train"],
        "mse_test": r["mse_test"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=None,
                    help="corre un solo eps y guarda su json parcial")
    ap.add_argument("--merge", action="store_true",
                    help="junta los json parciales y muestra el resumen")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if a.eps is not None:
        fila = corre_uno(a.eps)
        (OUT_DIR / f"f2_eps{a.eps:g}.json").write_text(json.dumps(fila, indent=2))
        return

    if a.merge:
        filas = []
        for eps in EPS_GRID:
            p = OUT_DIR / f"f2_eps{eps:g}.json"
            if p.exists():
                filas.append(json.loads(p.read_text()))
    else:
        filas = [corre_uno(eps) for eps in EPS_GRID
                 if (DATA_DIR / f"eps{eps:g}.npz").exists()]

    filas.sort(key=lambda f: f["eps"])
    OUT.write_text(json.dumps(filas, indent=2))

    print("\n=== RESUMEN — costo de la rigidez ===")
    print(f"  {'eps':>5} {'err_max%':>9} {'err_medio%':>11} {'peor':>8} {'mse_test':>11}")
    for f in filas:
        print(f"  {f['eps']:5.2f} {f['max_param_error']:9.2f} "
              f"{f['mean_param_error']:11.2f} {f['peor_param']:>8} {f['mse_test']:11.3e}")

    # Que parametro se rompe primero, y como escala cada uno.
    if filas:
        print("\n  Error por parametro (%):")
        print("  " + "eps".rjust(5) + "".join(f"{k:>9}" for k in ALL_P))
        for f in filas:
            print(f"  {f['eps']:5.2f}" +
                  "".join(f"{f['param_errors'][k]:9.2f}" for k in ALL_P))
    print(f"\n  -> {OUT}")


if __name__ == "__main__":
    main()
