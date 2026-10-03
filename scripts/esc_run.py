#!/usr/bin/env python3
# =============================================================================
#  RUNNER DEL ESCALADO: entrena UNA configuracion, la evalua y guarda todo.
# =============================================================================
#
#  Une las piezas: graybox_train.fit (variantes sin memoria), memory.fit_aug
#  (variantes con estado oculto) y esc_eval.evaluar (las 3 metricas del plan).
#
#  Cada corrida deja:
#    results/escalado/models/<tag>.pt   <- checkpoint con flags de arquitectura
#    results/escalado/<tag>.json        <- config + metricas
#
#  USO (ejemplos):
#    python scripts/esc_run.py --data refrac1 --variant whitebox --tag e1_wb
#    python scripts/esc_run.py --data refrac1 --variant B --window 400 --tag e1_B_w400
#    python scripts/esc_run.py --data act1 --variant lag --tag e2_lag
# =============================================================================

from __future__ import annotations

import argparse
import os
import hashlib
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import numpy as np
import torch

# Hilos por corrida. La máquina tiene 4 núcleos físicos, así que corridas en
# paralelo por hilos no debe pasar de 4: con 4 corridas de 4 hilos cada una, un
# lote de 25 minutos por corrida tardó 25 horas.
torch.set_num_threads(int(os.environ.get("WC_THREADS", "4")))

from src.neural_ode.graybox_train import TrainConfig, fit, load_split
from src.neural_ode.memory import (AugTrainConfig, LagGrayBox, LatentGrayBox,
                                   fit_aug)
from esc_eval import evaluar, nrmse_test

OUT_DIR = Path("results/escalado")


def cargar_params(core, ruta: str) -> None:
    """Copia los diez parametros fisicos de una corrida previa a los crudos del
    backbone. Sirve igual para el GrayBoxWC suelto y para el .core de los
    modelos de estado aumentado, que es el mismo objeto."""
    ph = json.loads(Path(ruta).read_text())["params"]
    with torch.no_grad():
        w = torch.tensor([ph[k] for k in ("wEE", "wEI", "wIE", "wII")])
        core.raw_w.copy_(torch.log(torch.expm1(w)))
        for k in ("te", "ti", "ae", "ai", "thetae", "thetai"):
            getattr(core, f"raw_{k}").copy_(
                torch.log(torch.expm1(torch.tensor(float(ph[k])))))


def suavizar(data: dict, k: int) -> dict:
    """Media movil de ancho k sobre I y E, en train, test y el crudo de la
    evaluacion. P y Q quedan intactos: son el comando y se conocen sin ruido."""
    ker = np.ones(k, dtype=np.float32) / k

    def mm(x):
        return np.stack([np.convolve(f, ker, mode="same") for f in x])

    for c in ("I", "E", "I_te", "E_te"):
        if c in data:
            data[c] = mm(data[c])
    # raw es el NpzFile, que es de solo lectura: se copia a un dict plano.
    crudo = {k: data["raw"][k] for k in data["raw"].files}
    crudo["I"], crudo["E"] = mm(crudo["I"]), mm(crudo["E"])
    data["raw"] = crudo
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="nombre del .npz en data/processed/uncertain (sin extension)")
    ap.add_argument("--variant", required=True,
                    choices=["whitebox", "A", "B", "C", "D", "S", "Sg", "H",
                             "K", "lag", "latent"])
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=1500)
    # Expuesto para el smoke de punta a punta: con el modelo recien inicializado
    # un solo .step() de L-BFGS agota la busqueda de linea y cuesta mas que el
    # entrenamiento entero.
    ap.add_argument("--lbfgs-steps", type=int, default=60)
    ap.add_argument("--lam-norm", type=float, default=0.0)
    ap.add_argument("--lam-orth", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-hidden", type=int, default=2)
    ap.add_argument("--init-params", default=None,
                    help="json de una corrida previa: warm-start de los 10 θ")
    ap.add_argument("--r-init", type=float, default=None,
                    help="valor inicial de r_i, r_e (variante S)")
    ap.add_argument("--hist", type=int, default=0,
                    help="largo de la historia de comando (variantes H, K)")
    ap.add_argument("--n-fir", type=int, default=4,
                    help="canales del filtro FIR (variante K)")
    ap.add_argument("--hidden", type=int, default=32,
                    help="ancho de la MLP de g (eje 1.4)")
    ap.add_argument("--wd-fir", type=float, default=0.0,
                    help="decaimiento de pesos sobre el FIR (variante K)")
    ap.add_argument("--smooth", type=int, default=0,
                    help="ancho de la media movil sobre I y E (0 = sin suavizar)")
    ap.add_argument("--freeze-phys", action="store_true",
                    help="β fijo en el de --init-params; solo se entrena g")
    ap.add_argument("--val-every", type=int, default=100,
                    help="épocas entre mediciones de validación (si el dataset la tiene)")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    if args.freeze_phys and not args.init_params:
        ap.error("--freeze-phys necesita --init-params: congelar el arranque "
                 "ignorante dejaría β en 1,0")
    if args.freeze_phys and args.variant in ("lag", "latent"):
        ap.error("--freeze-phys no está implementado para el estado aumentado: "
                 "fit_aug siempre entrena β")

    data_path = Path("data/processed/uncertain") / f"{args.data}.npz"
    # Hash del dataset: sin esto, un .npz regenerado deja los resultados viejos
    # sin forma de saber si salieron de los mismos bytes.
    data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()[:16]
    data = load_split(data_path)
    if args.smooth > 1:
        data = suavizar(data, args.smooth)
    t0 = time.time()

    # Con validación, el modelo que se evalúa y se guarda es el de menor NRMSE
    # de validación en corrida libre, no el de la última época.
    val_fn = None
    if data["is_val"].any():
        crudo = data["raw"]
        val_fn = lambda m: np.mean([f["nrmse"] for f in nrmse_test(m, crudo, "val")])

    print(f"=== escalado · {args.tag} · {args.variant} · {args.data} "
          f"· W={args.window} · epochs={args.epochs} ===", flush=True)

    if args.variant in ("lag", "latent"):
        model = (LagGrayBox() if args.variant == "lag"
                 else LatentGrayBox(n_hidden=args.n_hidden))
        # El estado aumentado tambien arranca del beta del white-box. fit_aug
        # no congela beta, asi que --freeze-phys no aplica aca.
        if args.init_params:
            cargar_params(model.core, args.init_params)
        cfg = AugTrainConfig(window=args.window, epochs=args.epochs,
                             seed=args.seed, val_every=args.val_every)
        res = fit_aug(data, model, cfg, val_fn=val_fn)
        ck = {"kind": args.variant, "state": model.state_dict(),
              "n_hidden": getattr(model, "n_hidden", 0)}
    else:
        cfg = TrainConfig(variant=args.variant, window=args.window,
                          epochs=args.epochs, lam_norm=args.lam_norm,
                          lam_orth=args.lam_orth, seed=args.seed,
                          hist=args.hist, n_fir=args.n_fir,
                          hidden=args.hidden, lbfgs_steps=args.lbfgs_steps,
                          wd_fir=args.wd_fir, freeze_phys=args.freeze_phys,
                          val_every=args.val_every)
        warm = None
        if args.init_params or args.r_init is not None:
            # warm-start: mismo modelo que build_model pero con los crudos
            # sobreescritos ANTES de entrenar (θ de una corrida previa, r > 0).
            from src.neural_ode.graybox_train import build_model
            torch.manual_seed(args.seed)
            warm = build_model(cfg)
            if args.init_params:
                cargar_params(warm, args.init_params)
            if args.r_init is not None and warm.structured:
                with torch.no_grad():
                    warm.raw_r_i.fill_(args.r_init)
                    warm.raw_r_e.fill_(args.r_init)
        res = fit(data, cfg, model=warm, val_fn=val_fn)
        model = res["model"]
        ck = {"kind": "graybox", "state": model.state_dict(),
              "use_correction": model.use_correction,
              "correction_inputs": model.correction_inputs,
              "hist_len": getattr(model, "hist_len", 0),
              "structured": model.structured}

    mins = (time.time() - t0) / 60.0
    model.eval()

    # El checkpoint va antes de evaluar: una falla en la evaluacion no tiene
    # por que costar la corrida entera.
    OUT_DIR.joinpath("models").mkdir(parents=True, exist_ok=True)
    torch.save(ck, OUT_DIR / "models" / f"{args.tag}.pt")

    ev = evaluar(model, data["raw"], data["true"])

    print(f"\n  RESULTADO {args.tag}: NRMSE_test={ev['nrmse_test']:.2f}% "
          f"(I={ev['nrmse_I']:.2f} E={ev['nrmse_E']:.2f}) "
          f"R2df={ev['r2_delta_test']:.3f} "
          f"err_param={ev['mean_param_error']:.2f}% [{mins:.1f} min]", flush=True)

    out = {
        "tag": args.tag, "data": args.data, "data_sha256": data_sha,
        "variant": args.variant,
        "window": args.window, "epochs": args.epochs, "hist": args.hist, "n_fir": args.n_fir,
        "hidden": args.hidden, "wd_fir": args.wd_fir,
        "smooth": args.smooth,
        "init_params": args.init_params, "freeze_phys": args.freeze_phys,
        "lam_norm": args.lam_norm, "lam_orth": args.lam_orth,
        "seed": args.seed, "minutos": mins,
        **{k: v for k, v in ev.items()},
        "params": res["params"],
        # g_rms y la fraccion de redundancia miden la ambiguedad entre beta y g:
        # quedaban solo en el dict que devuelve fit y no llegaban al artefacto.
        **{k: res[k] for k in ("g_rms", "g_rel", "frac_redundante",
                               "nrmse_val_mejor", "ep_mejor_val",
                               "nrmse_val_ultima", "historia_val") if k in res},
        **({"extras": res["extras"]} if "extras" in res else {}),
    }
    (OUT_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(f"  -> {OUT_DIR / (args.tag + '.json')}")


if __name__ == "__main__":
    main()
