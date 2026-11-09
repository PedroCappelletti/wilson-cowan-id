#!/usr/bin/env python3
# Distancia entre lazos y referencias con el comando dentro del rango de
# entrenamiento. Mismo controlador del paper y mismas plantas que
# control_paper_planta_modelo.py, con dos agregados:
#
#   distancia   además del RMSE de seguimiento de cada lazo, cuánto se separa el
#               lazo con el modelo del lazo con el simulador: E, I por un lado y
#               el comando P, Q por otro (último 80 % de la corrida). Es el
#               criterio de Gevers 2005; dos RMSE de seguimiento parecidos no
#               dicen que los lazos se parezcan.
#   en rango    dos referencias nuevas, theta_rango (8 Hz, 500 ms) y gamma_rango
#               (40 Hz, 250 ms). Son la respuesta de la planta ideal a un
#               estímulo con P ∈ [0,2; 1,8] y Q ∈ [0,15; 1,35], dentro de lo que
#               cubren los datos de entrenamiento. Con theta2s y gamma50 el
#               comando sale de [0, 2] la mayor parte del tiempo, así que ahí
#               la comparación mezcla arquitectura con extrapolación.
#
# Plantas: los e16 de control_planta_modelo.json, más la historia pasada (Hp),
# el filtro entrenable (K400f8) y los e19 con x0 entrenable cuando existen. El
# actuador con recorte en ±3 (actc) va con sus modelos e21. Se entrenó solo con
# comandos positivos, así que los negativos que pide el controlador miden
# extrapolación.
#
# «forma exacta, sat 3» es la misma forma exacta con el nivel de saturación del
# diseño (1/α = 3) en lugar del entrenado. Con comandos de entrenamiento entre 0
# y 2 ese nivel no se identifica (α̂ queda en 0,1 con recorte y se va a 0,001 con
# tanh), y el controlador pide hasta ±11. En el tag va como sufijo «@sat3».
#
# actf es la misma planta con filtro y sin saturación (el recorte en ±10⁹). Sus
# modelos son los e21: con comandos de entrenamiento en [0, 2] el recorte en ±3
# nunca actúa, así que los datos de actc y de actf son los mismos. En la forma
# exacta se saca el recorte («@satinf»); las redes no tienen un nivel separable.
#
#   python scripts/control_distancia_y_rango.py [procesos]
#   python scripts/control_distancia_y_rango.py [procesos] --nuevas   (solo lo que falta en el json)
#   python scripts/control_distancia_y_rango.py --probar    (simulador, refs en rango)

from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import numpy as np

from control_paper_modelos import CON_RED, EXACTA, WB, NIVELES, SEMILLAS, DT, RES, W
from control_paper_planta_modelo import planta_de

OUT = RAIZ / "results" / "control_distancia_rango.json"
REFS = ("theta2s", "gamma50", "theta_rango", "gamma_rango")
# (frecuencia en Hz, duración en ms) de las referencias en rango
EN_RANGO = {"theta_rango": (8.0, 500.0), "gamma_rango": (40.0, 250.0)}
RANGO_P, RANGO_Q = (0.0, 2.0), (0.0, 1.5)
SUBMUESTREO = 10       # se guarda cada 0,05 ms


def modelos(planta, suf, k):
    """(etiqueta, tag) de los modelos de una planta, ruido y semilla."""
    if planta == "actc":
        # actuador con recorte en ±3 (e21), sin segunda etapa x0
        return [(n, f"e21_{c}{suf}_s{k}{s}") for n, c, s in
                (("white-box", "e2wb", ""), ("con red", "H100_wrm", ""), ("forma exacta", "lag_wrm", ""),
                 ("forma exacta, sat 3", "lag_wrm", "@sat3"),
                 ("historia pasada", "Hp100_wrm", ""), ("filtro entrenable", "K400f8_wrm", ""),
                 ("agnóstica", "e2B_wrm", ""))]
    if planta == "actf":
        return [(n, f"e21_{c}{suf}_s{k}{s}") for n, c, s in
                (("white-box", "e2wb", ""), ("con red", "H100_wrm", ""),
                 ("forma exacta, sin saturación", "lag_wrm", "@satinf"),
                 ("historia pasada", "Hp100_wrm", ""), ("filtro entrenable", "K400f8_wrm", ""),
                 ("agnóstica", "e2B_wrm", ""))]
    out = [("white-box", f"e16_{WB[planta]}{suf}_s{k}"),
           ("con red", f"e16_{CON_RED[(planta, suf)]}{suf}_s{k}"),
           ("forma exacta", f"e16_{EXACTA[planta]}{suf}_s{k}")]
    if planta == "act":
        # e16 tiene la agnóstica solo con la semilla 0
        out += [("historia pasada", f"e16_Hp100_wrm{suf}_s{k}"),
                ("filtro entrenable", f"e16_K400f8_wrm{suf}_s{k}"),
                ("forma exacta, sat 3", f"e16_{EXACTA[planta]}{suf}_s{k}@sat3"),
                ("agnóstica", f"e16_e2B_wrm{suf}_s{k}")]
        e19 = ("e2wb", "H100", "Hp100", "K400f8")
        nombres = ("white-box", "historia 5 ms", "historia pasada", "filtro entrenable")
    else:
        e19 = ("e1wb", "B100", "e1S2")
        nombres = ("white-box", "agnóstica", "forma exacta")
    out += [(f"{n} x0", f"e19_{c}_x0{suf}_s{k}") for c, n in zip(e19, nombres)]
    return out


def tareas():
    out = []
    for ref in REFS:
        for planta in ("ideal", "refrac", "act", "actc", "actf"):
            out.append((ref, planta, "simulador", None, "", -1))
        for planta in ("refrac", "act", "actc", "actf"):
            for suf in NIVELES:
                for k in SEMILLAS:
                    for nombre, tag in modelos(planta, suf, k):
                        if (RES / "models" / f"{tag.split('@')[0]}.pt").exists():
                            out.append((ref, planta, nombre, tag, suf, k))
                        else:
                            print(f"  falta {tag}, se saltea", flush=True)
    return out


@lru_cache(maxsize=None)
def ref_en_rango(nombre):
    """Respuesta de la planta ideal al estímulo en rango, en la grilla de DT."""
    from src.wilson_cowan import WilsonCowanParams, NoPerturbation
    from src.neural_ode import make_perturbed_plant
    f, T = EN_RANGO[nombre]
    w = 2 * math.pi * f / 1000.0
    rhs = make_perturbed_plant(WilsonCowanParams(), NoPerturbation())

    def estimulo(t):
        return 1.0 + 0.8 * math.sin(w * t), 0.75 + 0.6 * math.sin(w * t - 1.0)

    n = int(round(T / DT))
    t = np.arange(n + 1) * DT
    I, E = np.zeros(n + 1), np.zeros(n + 1)

    def campo(tt, i, e):
        P, Q = estimulo(tt)
        dI, dE, _ = rhs(i, e, P, Q, (), tt)
        return dI, dE

    for j in range(n):
        tt, i, e = t[j], I[j], E[j]
        k1 = campo(tt, i, e)
        k2 = campo(tt + DT / 2, i + DT / 2 * k1[0], e + DT / 2 * k1[1])
        k3 = campo(tt + DT / 2, i + DT / 2 * k2[0], e + DT / 2 * k2[1])
        k4 = campo(tt + DT, i + DT * k3[0], e + DT * k3[1])
        I[j + 1] = i + DT / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        E[j + 1] = e + DT / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return t, I, E


def referencia(ref):
    from src.neural_ode import theta_gamma_refs
    from src.neural_ode.closed_loop_ext import theta_gamma_ventana_refs
    if ref == "theta2s":
        return theta_gamma_ventana_refs()[0], (0.0, 2000.0)
    if ref == "gamma50":
        return theta_gamma_refs(120.0, True), (0.0, 50.0)
    t, I, E = ref_en_rango(ref)
    return (lambda tt: (float(np.interp(tt, t, I)), float(np.interp(tt, t, E)))), (0.0, t[-1])


def en_rango(P, Q):
    return float(np.mean((P >= RANGO_P[0]) & (P <= RANGO_P[1]) & (Q >= RANGO_Q[0]) & (Q <= RANGO_Q[1])))


def una(t):
    import torch
    torch.set_num_threads(1)
    from src.wilson_cowan import WilsonCowanParams, Refractoriness, Actuator, NoPerturbation
    from src.neural_ode import IMCController, make_perturbed_plant
    from src.neural_ode.closed_loop_ext import simulate_closed_loop_ext
    from src.neural_ode.graybox_train import ALL_P
    from esc_eval import cargar
    from exp_f6_closed_loop import fixed_de, rmse

    ref, pert, modelo, tag, suf, k = t
    p = WilsonCowanParams()
    true = {kk: getattr(p, kk) for kk in ALL_P}
    refs, t_span = referencia(ref)
    n_pasos = int(round((t_span[1] - t_span[0]) / DT))
    if tag is None:
        planta = make_perturbed_plant(p, {"ideal": NoPerturbation(), "refrac": Refractoriness(r=0.10),
                                          "act": Actuator(sat=3.0, tau_act=1.0),
                                          "actc": Actuator(sat=3.0, tau_act=1.0, recorte=True),
                                          "actf": Actuator(sat=1e9, tau_act=1.0, recorte=True)}[pert])
    else:
        base, _, sat = tag.partition("@sat")
        planta = planta_de(cargar(RES / "models" / f"{base}.pt")[0], n_pasos)
        if sat:
            # 1/α = ∞ daría división por cero en PlantaFiltro; 10⁹ no recorta nunca
            planta.alpha = 1e-9 if sat == "inf" else 1.0 / float(sat)
    ctrl = IMCController(fixed_de(true), {kk: true[kk] for kk in W}, sat_ref=fixed_de(true))
    with np.errstate(over="ignore", invalid="ignore"):
        sol = simulate_closed_loop_ext(planta, ctrl, refs, t_span=t_span, dt=DT, dt_ctrl=DT)
    finito = bool(np.all(np.isfinite(sol["E"])) and np.all(np.isfinite(sol["I"])))
    eI, eE = rmse(sol) if finito else (float("nan"), float("nan"))
    n0 = len(sol["t"]) // 5
    fila = {"referencia": ref, "perturbacion": pert, "planta": modelo, "tag": tag,
            "ruido": suf, "semilla": k, "finito": finito,
            "rmse": math.sqrt((eI ** 2 + eE ** 2) / 2), "rmse_I": eI, "rmse_E": eE,
            "en_rango": en_rango(sol["P"][n0:], sol["Q"][n0:]),
            "P_p5_p95": [float(v) for v in np.nanpercentile(sol["P"][n0:], [5, 95])],
            "Q_p5_p95": [float(v) for v in np.nanpercentile(sol["Q"][n0:], [5, 95])]}
    traza = {kk: sol[kk][n0::SUBMUESTREO].astype(np.float32) for kk in ("E", "I", "P", "Q")}
    return fila, traza


def rms2(a, b):
    return float(np.sqrt((np.mean(a ** 2) + np.mean(b ** 2)) / 2))


def distancias(filas, trazas):
    """Agrega a cada fila de modelo su distancia al lazo con el simulador."""
    for i, f in enumerate(filas):
        if f["tag"] is None or not f["finito"]:
            continue
        j = next(j for j, g in enumerate(filas) if g["tag"] is None and g["referencia"] == f["referencia"]
                 and g["perturbacion"] == f["perturbacion"])
        m, s = trazas[i], trazas[j]
        f["dist_x"] = rms2(m["E"] - s["E"], m["I"] - s["I"])
        f["dist_u"] = rms2(m["P"] - s["P"], m["Q"] - s["Q"])
        f["dist_u_rel"] = f["dist_u"] / rms2(s["P"], s["Q"])


def main():
    if "--probar" in sys.argv:
        for ref in ("theta_rango", "gamma_rango"):
            for pl in ("ideal", "act"):
                fila, _ = una((ref, pl, "simulador", None, "", -1))
                print(f"  {ref:12} {pl:6} RMSE {fila['rmse']:.4f}  en rango {fila['en_rango']:.0%}  "
                      f"P {fila['P_p5_p95']}  Q {fila['Q_p5_p95']}", flush=True)
        return
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    procesos = int(args[0]) if args else 1
    ts = tareas()
    viejas = []
    if "--nuevas" in sys.argv and OUT.exists():
        # las trazas no se guardan, así que el lazo con el simulador se vuelve a
        # correr para las distancias; es determinista y reemplaza a su fila vieja
        viejas = json.loads(OUT.read_text(encoding="utf-8"))
        hechas = {(f["referencia"], f["perturbacion"], f["tag"]) for f in viejas if f["tag"]}
        ts = [t for t in ts if (t[0], t[1], t[3]) not in hechas]
        perts = {t[1] for t in ts if t[3]}
        ts = [t for t in ts if t[3] or t[1] in perts]
        sims = {(t[0], t[1]) for t in ts if t[3] is None}
        viejas = [f for f in viejas if f["tag"] or (f["referencia"], f["perturbacion"]) not in sims]
    print(f"{len(ts)} simulaciones, {procesos} procesos", flush=True)
    filas, trazas = [], []
    with Pool(procesos) as pool:
        for fila, traza in pool.imap_unordered(una, ts):
            filas.append(fila)
            trazas.append(traza)
            print(f"  {fila['referencia']:11} {fila['perturbacion']:6} {fila['planta']:22} "
                  f"{fila['ruido']:5} s{fila['semilla']:>2}  RMSE {fila['rmse']:.4f}  "
                  f"en rango {fila['en_rango']:.0%}", flush=True)
    distancias(filas, trazas)
    OUT.write_text(json.dumps(viejas + filas, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
