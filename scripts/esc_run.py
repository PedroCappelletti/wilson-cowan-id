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
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import numpy as np
import torch

torch.set_num_threads(4)

from src.neural_ode.graybox_train import TrainConfig, fit, load_split
from src.neural_ode.memory import (AugTrainConfig, LagGrayBox, LatentGrayBox,
                                   fit_aug)
from esc_eval import evaluar

OUT_DIR = Path("results/escalado")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="nombre del .npz en data/processed/uncertain (sin extension)")
    ap.add_argument("--variant", required=True,
                    choices=["whitebox", "A", "B", "C", "D", "S", "lag", "latent"])
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lam-norm", type=float, default=0.0)
    ap.add_argument("--lam-orth", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-hidden", type=int, default=2)
    ap.add_argument("--init-params", default=None,
                    help="json de una corrida previa: warm-start de los 10 θ")
    ap.add_argument("--r-init", type=float, default=None,
                    help="valor inicial de r_i, r_e (variante S)")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    data_path = Path("data/processed/uncertain") / f"{args.data}.npz"
    data = load_split(data_path)
    t0 = time.time()

    print(f"=== escalado · {args.tag} · {args.variant} · {args.data} "
          f"· W={args.window} · epochs={args.epochs} ===", flush=True)

    if args.variant in ("lag", "latent"):
        model = (LagGrayBox() if args.variant == "lag"
                 else LatentGrayBox(n_hidden=args.n_hidden))
        cfg = AugTrainConfig(window=args.window, epochs=args.epochs,
                             seed=args.seed)
        res = fit_aug(data, model, cfg)
        ck = {"kind": args.variant, "state": model.state_dict(),
              "n_hidden": getattr(model, "n_hidden", 0)}
    else:
        cfg = TrainConfig(variant=args.variant, window=args.window,
                          epochs=args.epochs, lam_norm=args.lam_norm,
                          lam_orth=args.lam_orth, seed=args.seed)
        warm = None
        if args.init_params or args.r_init is not None:
            # warm-start: mismo modelo que build_model pero con los crudos
            # sobreescritos ANTES de entrenar (θ de una corrida previa, r > 0).
            from src.neural_ode.graybox_train import build_model
            torch.manual_seed(args.seed)
            warm = build_model(cfg)
            if args.init_params:
                ph = json.loads(Path(args.init_params).read_text())["params"]
                with torch.no_grad():
                    w = torch.tensor([ph[k] for k in ("wEE", "wEI", "wIE", "wII")])
                    warm.raw_w.copy_(torch.log(torch.expm1(w)))
                    for k in ("te", "ti", "ae", "ai", "thetae", "thetai"):
                        getattr(warm, f"raw_{k}").copy_(
                            torch.log(torch.expm1(torch.tensor(float(ph[k])))))
            if args.r_init is not None and warm.structured:
                with torch.no_grad():
                    warm.raw_r_i.fill_(args.r_init)
                    warm.raw_r_e.fill_(args.r_init)
        res = fit(data, cfg, model=warm)
        model = res["model"]
        ck = {"kind": "graybox", "state": model.state_dict(),
              "use_correction": model.use_correction,
              "correction_inputs": model.correction_inputs,
              "structured": model.structured}

    mins = (time.time() - t0) / 60.0
    model.eval()
    ev = evaluar(model, data["raw"], data["true"])

    print(f"\n  RESULTADO {args.tag}: NRMSE_test={ev['nrmse_test']:.2f}% "
          f"(I={ev['nrmse_I']:.2f} E={ev['nrmse_E']:.2f}) "
          f"R2df={ev['r2_delta_test']:.3f} "
          f"err_param={ev['mean_param_error']:.2f}% [{mins:.1f} min]", flush=True)

    OUT_DIR.joinpath("models").mkdir(parents=True, exist_ok=True)
    torch.save(ck, OUT_DIR / "models" / f"{args.tag}.pt")
    out = {
        "tag": args.tag, "data": args.data, "variant": args.variant,
        "window": args.window, "epochs": args.epochs,
        "lam_norm": args.lam_norm, "lam_orth": args.lam_orth,
        "seed": args.seed, "minutos": mins,
        **{k: v for k, v in ev.items()},
        "params": res["params"],
        **({"extras": res["extras"]} if "extras" in res else {}),
    }
    (OUT_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(f"  -> {OUT_DIR / (args.tag + '.json')}")


if __name__ == "__main__":
    main()
