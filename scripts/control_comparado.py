#!/usr/bin/env python3
# Controlador original contra extendido, sobre todos los modelos de una tanda.
#
# Las dos plantas verdaderas (refractariedad y actuador) y, por cada modelo:
#
#   original, solo β̂        el IMC de Martínez et al. (2022) con β̂, ignorando
#                            la corrección (vale para todos los modelos)
#   original, con modelo     el mismo IMC cancelando la corrección cuando su
#                            despeje cerrado lo permite: g(I, E) o forma exacta
#   extendido                IMCExtendido (closed_loop_ext.py): despeje con la
#                            red (historia y filtro) o con la copia del
#                            actuador (estado de filtro)
#
# Todos con el mismo lazo: controlador actualizado cada 0,005 ms (comando retenido
# entre actualizaciones), planta integrada a 0,005 ms, referencias theta-gamma
# de 120 Hz, 50 ms, mismos topes de actuación. RMSE de seguimiento sin el
# primer quinto.
#
#   python scripts/control_comparado.py e13 [procesos]
#   python scripts/control_comparado.py e16 [procesos]

from __future__ import annotations

import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import numpy as np

RES = RAIZ / "results" / "escalado"
DATOS = RAIZ / "data" / "processed" / "uncertain"
W = ("wEE", "wEI", "wIE", "wII")
# El controlador se actualiza al mismo paso que la planta: a 0,05 ms, el retardo
# del actuador interactúa con el comando retenido y empeora a todos, incluso al
# oráculo (0,069 contra 0,093), y la comparación mediría eso.
DT, DT_CTRL, T_SPAN = 0.005, 0.005, (0.0, 50.0)

# (base del tag, nombre, tipo, se corre desde cero, se corre desde el white-box)
MODELOS = {
    "refrac": [("e1wb", "white-box", "wb", True, False),
               ("B100", "agnóstica 5 ms", "g", True, True),
               ("B400", "agnóstica 20 ms", "g", True, True),
               ("e1S", "forma exacta", "forma", True, True)],
    "act": [("e2wb", "white-box", "wb", True, False),
            ("e2B", "agnóstica", "g", True, True),
            ("H100", "historia, 5 ms", "historia", True, True),
            ("H400", "historia, 20 ms", "historia", True, True),
            ("K400", "filtro, 4 canales", "historia", True, True),
            ("K400f8", "filtro, 8 canales", "historia", True, True),
            ("K400f16", "filtro, 16 canales", "historia", True, True),
            ("lag", "estado de filtro", "lag", True, True)],
}


def tags(tanda, base, arranque):
    """Tags de un modelo en una tanda: [(tag, nivel, semilla)]."""
    if base == "e1S":       # la forma exacta: e1S2 arranca del white-box, e1S_cero desde cero
        b = "e1S2" if arranque == "wrm" else "e1S_cero"
    else:
        b = base + ("_wrm" if arranque == "wrm" else "")
    if tanda == "e13":
        return [(f"e13_{b}{s}", s or "sin ruido", 0) for s in ("", "_n01", "_n05")]
    if b == "e1S_cero":
        return []           # e16 no corre la forma exacta desde cero
    b = "e1S2_wrm" if b == "e1S2" else b
    return [(f"e16_{b}{s}_s{k}", s, k) for s in ("_n01", "_n05") for k in (0, 1, 2)]


def una(args):
    """Una simulación: (planta, nombre de fila, tag o None, controlador, tipo)."""
    import torch
    torch.set_num_threads(1)
    from src.wilson_cowan import WilsonCowanParams, Refractoriness, Actuator
    from src.neural_ode import (IMCController, theta_gamma_refs, make_perturbed_plant,
                                make_graybox_correction)
    from src.neural_ode.closed_loop_ext import IMCExtendido, simulate_closed_loop_ext
    from src.neural_ode.graybox_train import ALL_P
    from esc_eval import cargar
    from exp_f6_closed_loop import fixed_de, rmse

    planta_n, fila, tag, ctrl_n, tipo = args
    p = WilsonCowanParams()
    true = {k: getattr(p, k) for k in ALL_P}
    pert = Refractoriness(r=0.10) if planta_n == "refrac" else Actuator(sat=3.0, tau_act=1.0)
    planta = make_perturbed_plant(p, pert)
    SAT = fixed_de(true)
    kw = dict(sat_ref=SAT)
    if tag is None:                                   # oráculos
        ph = true
        estr = {"r_i": 0.10, "r_e": 0.10, "sat": 1e6} if ctrl_n == "oráculo + forma" else None
        ctrl = IMCController(fixed_de(ph), {k: ph[k] for k in W}, structured=estr, **kw)
    else:
        if not (RES / "models" / f"{tag}.pt").exists():
            return None
        m, _ = cargar(RES / "models" / f"{tag}.pt")
        ph = m.params_dict()
        fx, wh = fixed_de(ph), {k: ph[k] for k in W}
        if ctrl_n == "original, solo β̂":
            ctrl = IMCController(fx, wh, **kw)
        elif ctrl_n == "original, con modelo":
            if tipo == "g":
                ctrl = IMCController(fx, wh, g_hat=make_graybox_correction(m), **kw)
            else:
                ctrl = IMCController(fx, wh, structured=m.structured_dict(), **kw)
        else:
            datos = json.loads((RES / f"{tag}.json").read_text())["data"]
            dt_m = float(np.load(DATOS / f"{datos}.npz")["dt"])
            ctrl = IMCExtendido(fx, wh, m, tipo, dt_m, DT_CTRL, **kw)
    sol = simulate_closed_loop_ext(planta, ctrl, theta_gamma_refs(120.0, True),
                                   t_span=T_SPAN, dt=DT, dt_ctrl=DT_CTRL)
    eI, eE = rmse(sol)
    eb = (float(np.mean([100 * abs(ph[k] - true[k]) / abs(true[k]) for k in ALL_P]))
          if tag else 0.0)
    sat = max(ctrl.n_sat_i, ctrl.n_sat_e) / max(ctrl.n_pasos, 1)
    return {"planta": planta_n, "modelo": fila, "tag": tag, "controlador": ctrl_n,
            "rmse": float(math.sqrt((eI ** 2 + eE ** 2) / 2)), "rmse_I": eI, "rmse_E": eE,
            "error_beta": eb, "frac_saturado": float(sat),
            "fallas_inversion": int(getattr(ctrl, "n_fallas", 0)),
            "P_max": float(np.abs(sol["P"]).max()),
            "ok": bool(np.isfinite(eI) and np.isfinite(eE) and sat < 0.5)}


def tareas(tanda):
    out = [("refrac", "oráculo", None, "oráculo", None),
           ("refrac", "oráculo + forma", None, "oráculo + forma", None),
           ("act", "oráculo", None, "oráculo", None)]
    for planta, lista in MODELOS.items():
        for base, nombre, tipo, cero, wrm in lista:
            for arranque, va in (("cero", cero), ("wrm", wrm)):
                if not va:
                    continue
                for tag, nivel, k in tags(tanda, base, arranque):
                    fila = f"{nombre}" + ("" if tipo == "wb" else f", {'desde cero' if arranque == 'cero' else 'warm start'}")
                    out.append((planta, fila, tag, "original, solo β̂", tipo))
                    if tipo in ("g", "forma"):
                        out.append((planta, fila, tag, "original, con modelo", tipo))
                    if tipo in ("historia", "lag"):
                        out.append((planta, fila, tag, "extendido", tipo))
    # las del extendido con historia son las lentas: primero, para repartirlas
    return sorted(out, key=lambda a: a[3] != "extendido" or a[4] != "historia")


def main():
    tanda = sys.argv[1]
    procesos = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    ts = tareas(tanda)
    print(f"{len(ts)} simulaciones, {procesos} procesos", flush=True)
    filas = []
    with Pool(procesos) as pool:
        for r in pool.imap_unordered(una, ts):
            if r is None:
                continue
            filas.append(r)
            print(f"  {r['planta']:6} {r['modelo']:34} {r['controlador']:22} {r['tag'] or '':26} "
                  f"RMSE {r['rmse']:.4f}  β {r['error_beta']:5.1f} %"
                  + ("" if r["ok"] else "  <-- divergió o saturó"), flush=True)
    out = RAIZ / "results" / f"control_comparado_{tanda}.json"
    out.write_text(json.dumps(filas, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
