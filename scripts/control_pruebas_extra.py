#!/usr/bin/env python3
# Cuatro pruebas de lazo cerrado que faltaban para el informe (21/11/2026).
#
#   repeticion  Cada modelo en lazo abierto con el comando registrado en el lazo
#               cerrado del simulador. Si el modelo sigue al simulador con ese
#               comando y aun así su lazo se separa, la diferencia la amplifica
#               la realimentación; si no lo sigue, el modelo falla en esa región
#               de comando y estado.
#   bb_chica    Las black-box de amplitud chica (e24) como planta del IMC, con
#               las mismas referencias que control_distancia_y_rango.py.
#   ruido       El controlador mide I y E con ruido. El ruido es el de los
#               datasets (desvío por muestra 0,0032 y 0,016 para σ = 0,01 y
#               0,05) retenido 0,5 ms, el paso de muestreo de los datos. El
#               simulador y el modelo reciben la misma realización.
#   ruido_todo  Lo mismo que ruido, con todos los lazos del informe (los dos
#               regímenes, e24, e28 y las cuatro referencias).
#   rizo        Actuador con recorte: la distancia entre lazos separada en la
#               parte lenta (< 200 Hz) y el rizo del ciclo límite (> 200 Hz).
#
# Todas las filas llevan dist_x y su separación en lenta y rizo.
#
#   python scripts/control_pruebas_extra.py <prueba> [procesos]
#   WC_FGAMMA=100 ...   -> results/pruebas_<prueba>_100hz.json

from __future__ import annotations

import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import control_distancia_y_rango as C  # noqa: E402
from control_paper_modelos import NIVELES, SEMILLAS, RES, DT, W  # noqa: E402

SUFIJO = C.SUFIJO
CORTE_HZ = 200.0
SIGMA_MED = {"_n01": (0.0032, 0.0032), "_n05": (0.016, 0.016)}
RETENCION_MS = 0.5

AMPLITUD = {
    "actc": [("white-box", "e25_e2wb"), ("agnóstica", "e25_e2B_wrm"),
             ("historia con comando actual", "e25_H100_wrm"),
             ("historia sin comando actual", "e25_Hp100_wrm"),
             ("filtro 8 canales", "e25_K400f8_wrm"), ("estado de filtro", "e25_lag_wrm"),
             ("black-box", "e27_BB64_act"), ("black-box con historia", "e27_BBH64_act")],
    "refrac": [("white-box", "e25_e1wb"), ("agnóstica", "e25_B100_wrm"),
               ("forma exacta", "e25_e1S2_wrm"), ("black-box", "e27_BB64_ref")],
}
E28_ACTC = [("agnóstica, β verdadero", "e28_e2B_vrd"),
            ("historia con comando actual, β verdadero", "e28_H100_vrd"),
            ("historia sin comando actual, β verdadero", "e28_Hp100_vrd"),
            ("filtro 8 canales, β verdadero", "e28_K400f8_vrd")]
BB_CHICA = {"actc": [("black-box", "e24_BB64_act"), ("black-box con historia", "e24_BBH64_act")],
            "refrac": [("black-box", "e24_BB64_ref"), ("black-box con historia", "e24_BBH64_ref")]}


def existe(tag):
    return (RES / "models" / f"{tag.split('@')[0]}.pt").exists()


def _expandir(refs, plantas_modelos, regimen):
    out = []
    for ref in refs:
        for planta, mods in plantas_modelos.items():
            out.append(dict(ref=ref, pert=planta, modelo="simulador", tag=None, suf="", k=-1, regimen=""))
            for suf in NIVELES:
                for k in SEMILLAS:
                    for nombre, pre in mods:
                        tag = f"{pre}{suf}_s{k}"
                        if existe(tag):
                            out.append(dict(ref=ref, pert=planta, modelo=nombre, tag=tag, suf=suf,
                                            k=k, regimen=regimen))
    return out


def tareas(prueba):
    lazo = ("theta2s", "gamma50")
    if prueba == "bb_chica":
        return _expandir(C.REFS, BB_CHICA, "chica")
    if prueba == "repeticion":
        return _expandir(lazo, {"actc": AMPLITUD["actc"] + E28_ACTC, "refrac": AMPLITUD["refrac"]},
                         "amplitud")
    if prueba == "ruido":
        out = []
        for med in NIVELES:
            for t in _expandir(lazo, AMPLITUD, "amplitud"):
                # solo la semilla de datos con el mismo nivel que el ruido de medición
                if t["tag"] is None or t["suf"] == med:
                    out.append({**t, "med": med})
        return out
    if prueba == "ruido_todo":
        # todos los lazos del informe con ruido de medición: los dos regímenes,
        # e24, e28, las cuatro referencias (dos con 100 Hz). Cada modelo se
        # cierra con el ruido del mismo nivel con el que se entrenó.
        import control_amplitud as A
        import control_e28 as E
        base = {}
        for planta in ("refrac", "actc"):
            for suf in NIVELES:
                for k in SEMILLAS:
                    for nombre, tag in C.modelos(planta, suf, k):
                        base.setdefault((planta, "chica"), []).append((nombre, tag, suf, k))
        for fuente, regimen in ((A.MODELOS, "amplitud"), (E.MODELOS, "amplitud"), (BB_CHICA, "chica")):
            for planta, mods in fuente.items():
                for suf in NIVELES:
                    for k in SEMILLAS:
                        for nombre, pre in mods:
                            base.setdefault((planta, regimen), []).append((nombre, f"{pre}{suf}_s{k}", suf, k))
        out = []
        for med in NIVELES:
            for ref in C.REFS:
                for planta in ("refrac", "actc"):
                    out.append(dict(ref=ref, pert=planta, modelo="simulador", tag=None, suf="", k=-1,
                                    regimen="", med=med))
                for (planta, regimen), lst in base.items():
                    out += [dict(ref=ref, pert=planta, modelo=n, tag=tg, suf=s, k=k, regimen=regimen, med=med)
                            for n, tg, s, k in lst if s == med and existe(tg)]
        return out
    if prueba == "rizo":
        chica = {}
        for suf in NIVELES:
            for k in SEMILLAS:
                for nombre, tag in C.modelos("actc", suf, k):
                    chica.setdefault(nombre, []).append((tag, suf, k))
        out = []
        for ref in lazo:
            out.append(dict(ref=ref, pert="actc", modelo="simulador", tag=None, suf="", k=-1, regimen=""))
            for nombre, lst in chica.items():
                out += [dict(ref=ref, pert="actc", modelo=nombre, tag=tg, suf=s, k=k, regimen="chica")
                        for tg, s, k in lst if existe(tg)]
        out += [t for t in _expandir(lazo, {"actc": AMPLITUD["actc"] + E28_ACTC}, "amplitud")
                if t["tag"] is not None]
        return out
    raise SystemExit(f"prueba desconocida: {prueba}")


class ConRuido:
    """El IMC del paper midiendo I y E con ruido retenido entre muestras."""

    def __init__(self, ctrl, sI, sE, n_pasos, semilla=12345):
        self.ctrl = ctrl
        cada = int(round(RETENCION_MS / DT))
        rng = np.random.default_rng(semilla)
        m = n_pasos // cada + 2
        self.nI = np.repeat(rng.standard_normal(m) * sI, cada)
        self.nE = np.repeat(rng.standard_normal(m) * sE, cada)
        self.i = 0

    def compute(self, z1, z2, I, E, rI, rE):
        j = min(self.i, len(self.nI) - 1)
        self.i += 1
        return self.ctrl.compute(z1, z2, I + self.nI[j], E + self.nE[j], rI, rE)


class Repeticion:
    """'Controlador' que manda el comando registrado, paso por paso."""

    def __init__(self, P, Q):
        self.P, self.Q, self.i = P, Q, 0

    def compute(self, z1, z2, I, E, rI, rE):
        j = min(self.i, len(self.P) - 1)
        self.i += 1
        return float(self.P[j]), float(self.Q[j]), None, None


def _planta(t, n_pasos):
    from src.wilson_cowan import WilsonCowanParams, Refractoriness, Actuator
    from src.neural_ode import make_perturbed_plant
    from esc_eval import cargar
    if t["tag"] is None:
        p = WilsonCowanParams()
        return make_perturbed_plant(p, {"refrac": Refractoriness(r=0.10),
                                        "actc": Actuator(sat=3.0, tau_act=1.0, recorte=True)}[t["pert"]])
    base, _, sat = t["tag"].partition("@sat")
    pl = C.planta_de(cargar(RES / "models" / f"{base}.pt")[0], n_pasos)
    if sat:
        pl.alpha = 1e-9 if sat == "inf" else 1.0 / float(sat)
    return pl


def _lazo(t, planta, ctrl, refs, t_span):
    from src.neural_ode.closed_loop_ext import simulate_closed_loop_ext
    with np.errstate(over="ignore", invalid="ignore"):
        return simulate_closed_loop_ext(planta, ctrl, refs, t_span=t_span, dt=DT, dt_ctrl=DT)


# lazo del simulador por (referencia, planta), uno por proceso
_SIM = {}


def una(t):
    import torch
    torch.set_num_threads(1)
    from src.wilson_cowan import WilsonCowanParams
    from src.neural_ode import IMCController
    from src.neural_ode.graybox_train import ALL_P
    from exp_f6_closed_loop import fixed_de, rmse

    p = WilsonCowanParams()
    true = {kk: getattr(p, kk) for kk in ALL_P}
    refs, t_span = C.referencia(t["ref"])
    n_pasos = int(round((t_span[1] - t_span[0]) / DT))
    ctrl = IMCController(fixed_de(true), {kk: true[kk] for kk in W}, sat_ref=fixed_de(true))
    if "med" in t:
        ctrl = ConRuido(ctrl, *SIGMA_MED[t["med"]], n_pasos)
    sol = _lazo(t, _planta(t, n_pasos), ctrl, refs, t_span)
    fila = {k: v for k, v in t.items()}
    sim = None
    if t.get("repeticion") and t["tag"] is not None:
        # el lazo del simulador se corre acá mismo para tener su comando completo
        clave = (t["ref"], t["pert"])
        if clave not in _SIM:
            _SIM[clave] = _lazo({**t, "tag": None}, _planta({**t, "tag": None}, n_pasos),
                                IMCController(fixed_de(true), {kk: true[kk] for kk in W},
                                              sat_ref=fixed_de(true)), refs, t_span)
        sim = _SIM[clave]
        rep = _lazo(t, _planta(t, n_pasos), Repeticion(sim["P"], sim["Q"]), refs, t_span)
        n0 = len(sim["t"]) // 5
        ok = bool(np.all(np.isfinite(rep["E"])))
        fila["rep_dist_x"] = rms2(rep["E"][n0:] - sim["E"][n0:], rep["I"][n0:] - sim["I"][n0:]) if ok else float("nan")
        fila["rep_finito"] = ok
    finito = bool(np.all(np.isfinite(sol["E"])) and np.all(np.isfinite(sol["I"])))
    eI, eE = rmse(sol) if finito else (float("nan"), float("nan"))
    n0 = len(sol["t"]) // 5
    fila.update(finito=finito, rmse=math.sqrt((eI ** 2 + eE ** 2) / 2), rmse_I=eI, rmse_E=eE,
                P_p5_p95=[float(v) for v in np.nanpercentile(sol["P"][n0:], [5, 95])])
    traza = {kk: sol[kk][n0::C.SUBMUESTREO].astype(np.float64) for kk in ("E", "I", "P", "Q")}
    return fila, traza


def rms2(a, b):
    return float(np.sqrt((np.mean(a ** 2) + np.mean(b ** 2)) / 2))


def _partir(x, fs):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, CORTE_HZ, fs=fs, output="sos")
    lento = sosfiltfilt(sos, x)
    return lento, x - lento


def distancias(filas, trazas):
    fs = 1000.0 / (DT * C.SUBMUESTREO)
    for i, f in enumerate(filas):
        if f["tag"] is None or not f["finito"]:
            continue
        j = next(j for j, g in enumerate(filas) if g["tag"] is None and g["ref"] == f["ref"]
                 and g["pert"] == f["pert"] and g.get("med") == f.get("med"))
        m, s = trazas[i], trazas[j]
        dE, dI = m["E"] - s["E"], m["I"] - s["I"]
        (lE, rE), (lI, rI) = _partir(dE, fs), _partir(dI, fs)
        f["dist_x"] = rms2(dE, dI)
        f["dist_x_lenta"] = rms2(lE, lI)
        f["dist_x_rizo"] = rms2(rE, rI)
        f["dist_u"] = rms2(m["P"] - s["P"], m["Q"] - s["Q"])


if __name__ == "__main__":
    prueba = sys.argv[1]
    procesos = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ts = tareas(prueba)
    if prueba == "repeticion":
        ts = [{**t, "repeticion": True} for t in ts]
    out = RAIZ / "results" / f"pruebas_{prueba}{SUFIJO}.json"
    print(f"{prueba}: {len(ts)} simulaciones, {procesos} procesos -> {out.name}", flush=True)
    filas, trazas = [], []
    with Pool(procesos) as pool:
        for fila, traza in pool.imap_unordered(una, ts):
            filas.append(fila)
            trazas.append(traza)
            extra = f"  rep {fila['rep_dist_x']:.4f}" if "rep_dist_x" in fila else ""
            print(f"  {fila['ref']:11} {fila['pert']:6} {fila['modelo']:42} {fila['suf']:5} "
                  f"s{fila['k']:>2} {fila.get('med', ''):5} RMSE {fila['rmse']:.4f}{extra}", flush=True)
    distancias(filas, trazas)
    out.write_text(json.dumps(filas, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {out}")
