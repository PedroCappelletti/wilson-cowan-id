#!/usr/bin/env python3
# =============================================================================
#  DATASETS CON INCERTIDUMBRE DINAMICA  (barrido de la perilla eps)
# =============================================================================
#
#  Genera el mismo dataset multi-escenario de siempre (gen_multi_dataset.py),
#  pero con el simulador PERTURBADO: la planta deja de ser las mismas ecuaciones
#  que el modelo que despues se entrena.
#
#  Se generan varios niveles de eps para poder medir "cuanto cuesta la rigidez":
#      eps = 0    -> Wilson-Cowan puro (baseline)
#      eps = 1    -> punto de operacion nominal calibrado
#      eps > 1    -> mismatch fuerte
#
#  IMPORTANTE — el baseline eps=0 se genera con NoPerturbation, o sea POR EL
#  MISMO INTEGRADOR de paso fijo que los perturbados. Si se generara con el
#  camino historico (solve_ivp rtol=1e-3) la comparacion mezclaria el efecto de
#  la perturbacion con un cambio de integrador: ese camino tiene un error de
#  ~1.5e-2, del mismo orden que el ruido que el proyecto estudia.
#
#  Ademas de las trayectorias se guarda el Delta f VERDADERO en cada instante:
#      Delta f = (campo de la planta real) - (campo Wilson-Cowan con θ verdadero)
#  Es exactamente lo que g_φ deberia aprender. Conocerlo permite medir despues
#  si la red aprendio la FISICA CORRECTA y no simplemente algo que ajusta
#  (algo que casi ningun trabajo de gray-box puede hacer).
#
#  USO:  python scripts/gen_uncertain_dataset.py
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import numpy as np
from scipy.special import expit

from src.wilson_cowan import WilsonCowanParams, NoPerturbation, default_uncertainty
from src.data import generate_dataset

from gen_multi_dataset import build_scenarios, T_SPAN, N_EVAL, I0, E0, SEED

# #############################################################################
# ##   ZONA EDITABLE                                                         ##
# #############################################################################

PARAMS = WilsonCowanParams()
EPS_GRID = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
NOISE = 0.0                     # sin ruido de observacion: aislamos el mismatch
OUT_DIR = Path("data/processed/uncertain")

# #############################################################################
# ##   FIN ZONA EDITABLE                                                     ##
# #############################################################################

PARAM_KEYS = ("te", "ti", "wEE", "wEI", "wIE", "wII", "ae", "ai", "thetae", "thetai")


def campo_wc(p, I, E, P, Q):
    """Campo de Wilson-Cowan puro con los parametros VERDADEROS (vectorizado).
    Es el backbone que la red va a usar; la diferencia con la planta real es
    justamente el hueco que tiene que tapar la correccion."""
    u_i = p.wIE * E - p.wII * I + Q - p.thetai
    u_e = p.wEE * E - p.wEI * I + P - p.thetae
    dI = (1.0 / p.ti) * (-I + expit(p.ai * u_i) - p.ki)
    dE = (1.0 / p.te) * (-E + expit(p.ae * u_e) - p.ke)
    return dI, dE


def delta_f_verdadero(params, pert, Pf, Qf, t, I, E, P_cmd, Q_cmd, extra):
    """Delta f = campo real - campo WC nominal, evaluado sobre la trayectoria.

    Se evalua con el estimulo COMANDADO, que es lo unico que el modelo conoce:
    si el actuador distorsiona la entrada, esa distorsion forma parte del hueco
    que g_φ tiene que descubrir."""
    from src.wilson_cowan import WilsonCowan
    m = WilsonCowan(params=params, P=Pf, Q=Qf, perturbation=pert)
    n = len(t)
    dfI = np.zeros(n)
    dfE = np.zeros(n)
    nom_I, nom_E = campo_wc(params, I, E, P_cmd, Q_cmd)
    for k in range(n):
        s = np.concatenate(([I[k], E[k]], extra[k] if extra is not None else []))
        d = m.rhs_aug(float(t[k]), s)
        dfI[k] = d[0] - nom_I[k]
        dfE[k] = d[1] - nom_E[k]
    return dfI, dfE


def generar_con(pert_factory, out_path: Path, extra_meta: dict | None = None):
    """Genera el dataset multi-escenario con la perturbacion que devuelva
    `pert_factory()`. Se llama a la fabrica UNA VEZ POR ESCENARIO: las
    perturbaciones con estado interno pre-generado (el ruido, por ejemplo) no
    deben compartirse entre trayectorias."""
    pert_ref = pert_factory()
    scenarios = build_scenarios()
    print(f"\n=== {pert_ref.metadata()['pert_name']} -> {out_path.name} — "
          f"{len(scenarios)} escenarios ===")

    I_all, E_all, P_all, Q_all = [], [], [], []
    Pe_all, Qe_all, dfI_all, dfE_all = [], [], [], []
    labels, is_test = [], []
    t_ref = None

    for label, Pf, Qf, test in scenarios:
        pert = pert_factory()
        ds = generate_dataset(params=PARAMS, P=Pf, Q=Qf, I0=I0, E0=E0,
                              t_span=T_SPAN, n_eval=N_EVAL, noise_std=NOISE,
                              seed=SEED, perturbation=pert)

        t_ref = ds["t"]
        extra = ds.get("pert_extra")
        dfI, dfE = delta_f_verdadero(PARAMS, pert, Pf, Qf, ds["t"], ds["I"], ds["E"],
                                     ds["P"], ds["Q"], extra)

        I_all.append(ds["I"]); E_all.append(ds["E"])
        P_all.append(ds["P"]); Q_all.append(ds["Q"])
        Pe_all.append(ds.get("P_eff", ds["P"])); Qe_all.append(ds.get("Q_eff", ds["Q"]))
        dfI_all.append(dfI); dfE_all.append(dfE)
        labels.append(label); is_test.append(test)

        rel = np.sqrt(dfI ** 2 + dfE ** 2).mean()
        flag = "TEST " if test else "train"
        print(f"  [{flag}] {label:18} E=[{ds['E'].min():6.3f},{ds['E'].max():6.3f}] "
              f"|Df|={rel:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        t=t_ref,
        I=np.stack(I_all), E=np.stack(E_all),
        P=np.stack(P_all), Q=np.stack(Q_all),
        # diagnostico (NUNCA se usa para entrenar)
        P_eff=np.stack(Pe_all), Q_eff=np.stack(Qe_all),
        dfI=np.stack(dfI_all), dfE=np.stack(dfE_all),
        is_test=np.asarray(is_test), labels=np.asarray(labels),
        dt=float(t_ref[1] - t_ref[0]), t_span=np.asarray(T_SPAN),
        noise_std=np.asarray(NOISE), seed=np.asarray(SEED),
        **{k: np.asarray(v) for k, v in (extra_meta or {}).items()},
        **{k: np.asarray(v) for k, v in pert_ref.metadata().items()},
        **{k: np.asarray(getattr(PARAMS, k)) for k in PARAM_KEYS},
    )
    print(f"  -> {out_path}")


def generar(eps: float, out_path: Path):
    """El par recomendado (refractariedad + actuador) con la perilla eps."""
    fab = (lambda: NoPerturbation()) if eps <= 0 else (lambda: default_uncertainty(eps))
    generar_con(fab, out_path, {"eps": eps})


def main():
    for eps in EPS_GRID:
        generar(eps, OUT_DIR / f"eps{eps:g}.npz")
    print("\nListo.")


if __name__ == "__main__":
    main()
