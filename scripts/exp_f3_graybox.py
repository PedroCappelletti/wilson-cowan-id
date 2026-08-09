#!/usr/bin/env python3
# =============================================================================
#  F3 — ENCENDER g_φ Y ENFRENTAR LA TRAMPA DE IDENTIFICABILIDAD
# =============================================================================
#
#  Es la fase central del roadmap.
#
#  La trampa: g_φ es un aproximador universal sobre el MISMO dominio que el
#  backbone de Wilson-Cowan. Entonces, para cualquier θ equivocado existe un g
#  que compensa el error y reproduce los datos igual de bien. Con g libre, los
#  10 parametros dejan de ser identificables.
#
#  El sintoma esperado: el MSE MEJORA y el error parametrico EMPEORA. Por eso
#  hay que reportar SIEMPRE las dos metricas juntas: mirar solo el MSE haria
#  parecer que el gray-box es un exito cuando en realidad rompio la
#  identificacion, que es el resultado central del proyecto.
#
#  Las cuatro variantes (ver src/neural_ode/graybox_train.py):
#     A  g(I,E,P,Q) libre        -> deberia mostrar la patologia
#     B  g(I,E)                  -> el estimulo solo lo puede explicar θ
#     C  g(I,E) + λ‖g‖²          -> prior de correccion minima
#     D  g(I,E) + λ‖proyeccion‖² -> prohibe a g moverse en las direcciones que
#                                   un cambio de parametros ya podria explicar
#
#  Metrica clave que reporta este script:
#     frac_redundante = que fraccion de lo que hace g_φ podria haberlo hecho un
#     cambio de θ.
#
#  OJO CON EL PUNTO CERO — no es 0. Medido sobre eps=1 (ver seccion 15.2 del
#  manual, reproducible con exp_f4b_geometria_mismatch.py):
#
#       0.002  ruido blanco sin estructura
#       0.466  una red g(I,E) con pesos ALEATORIOS  <- solo por ser suave
#       0.670  el Delta f VERDADERO                 <- LA REFERENCIA CORRECTA
#       0.86-0.94  las correcciones aprendidas
#
#  O sea: una correccion PERFECTA puntuaria 0.67, no 0. El objetivo no es
#  minimizar esta metrica sino acercarla a la redundancia que tiene la fisica
#  real. Empujarla por debajo (lo que hace la variante D con lambda grande)
#  ALEJA a g del Delta f verdadero y devuelve el sesgo a los parametros.
#
#  USO:  python scripts/exp_f3_graybox.py --variant D --eps 1.0 --lam 10
# =============================================================================

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch

torch.set_num_threads(2)

from src.neural_ode.graybox_train import TrainConfig, fit, load_split

DATA_DIR = Path("data/processed/uncertain")
OUT_DIR = Path("results/uncertainty")
EPOCHS = 1500
LBFGS = 30


def etiqueta(variant, eps, lam):
    base = f"f3_{variant}_eps{eps:g}"
    return base if lam is None else f"{base}_lam{lam:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["whitebox", "A", "B", "C", "D", "S"])
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=None,
                    help="peso de la regularizacion (C: norma, D: proyeccion)")
    ap.add_argument("--data", default=None, help="npz alternativo (para F7)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    a = ap.parse_args()

    path = Path(a.data) if a.data else DATA_DIR / f"eps{a.eps:g}.npz"
    data = load_split(path)

    cfg = TrainConfig(
        variant=a.variant, epochs=a.epochs, lbfgs_steps=LBFGS, log_every=300,
        lam_norm=(a.lam or 0.0) if a.variant == "C" else 0.0,
        lam_orth=(a.lam or 0.0) if a.variant == "D" else 0.0,
    )

    print(f"\n=== F3 · variante {a.variant} · {path.name} · lam={a.lam} ===", flush=True)
    r = fit(data, cfg)

    fila = {
        "variant": a.variant, "eps": a.eps, "lam": a.lam, "data": str(path),
        "max_param_error": r["max_param_error"],
        "mean_param_error": r["mean_param_error"],
        "param_errors": r["param_errors"], "params": r["params"],
        "mse_train": r["mse_train"], "mse_test": r["mse_test"],
        "g_rms": r.get("g_rms"), "g_rel": r.get("g_rel"),
        "frac_redundante": r.get("frac_redundante"),
        "structured": r.get("structured"),
    }
    if r.get("structured"):
        s = r["structured"]
        # Los valores VERDADEROS con los que se genero el dataset, para ver si
        # los 3 parametros fisicos se recuperan (default_uncertainty: r=0.10*eps,
        # sat=3.0/eps).
        rv, sv = 0.10 * a.eps, (3.0 / a.eps if a.eps > 0 else float("inf"))
        print(f"  Correccion FISICA identificada (verdadero -> estimado):")
        print(f"     r_i : {rv:8.4f} -> {s['r_i']:8.4f}")
        print(f"     r_e : {rv:8.4f} -> {s['r_e']:8.4f}")
        print(f"     sat : {sv:8.4f} -> {s['sat']:8.4f}")
    peor = max(r["param_errors"], key=r["param_errors"].get)
    print(f"\n  RESULTADO {a.variant}: err_max={r['max_param_error']:.2f}% "
          f"err_medio={r['mean_param_error']:.2f}% peor={peor} "
          f"mse_test={r['mse_test']:.3e} "
          f"g_rel={r.get('g_rel', 0) or 0:.3f} "
          f"redundante={r.get('frac_redundante', 0) or 0:.3f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = a.tag or etiqueta(a.variant, a.eps, a.lam)
    (OUT_DIR / f"{tag}.json").write_text(json.dumps(fila, indent=2))

    # El checkpoint hace falta para F5 (recuperacion funcional) y F6 (lazo cerrado).
    ck = OUT_DIR / "models"
    ck.mkdir(parents=True, exist_ok=True)
    torch.save({"state": r["model"].state_dict(), "variant": a.variant,
                "eps": a.eps, "lam": a.lam,
                "use_correction": r["model"].use_correction,
                "correction_inputs": r["model"].correction_inputs,
                "structured": r["model"].structured,
                "structured_params": r.get("structured"),
                "params": r["params"]}, ck / f"{tag}.pt")
    print(f"  -> {OUT_DIR / (tag + '.json')}")


if __name__ == "__main__":
    main()
