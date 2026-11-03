#!/usr/bin/env python3
# Tres pruebas sobre las configuraciones de e16 (línea de diseño), en una cola.
#
#   e19_<conf>_x0   segunda etapa con x0 entrenable (λ = 1). Arranca del modelo
#                   e16 entero (--init-model: β y red), así los x0 solo corrigen
#                   el ruido. En el white-box de wc0_2s bajó el error medio de
#                   parámetros con σ 0,05 de 9,8 a 0,5 % (resumen_sesgo_ventanas.py).
#                   La forma exacta del actuador (lag) no entra: x0 libre no está
#                   implementado para el estado aumentado.
#   e19_<conf>_W40  ventanas de 20 ms con x0 medido, desde el white-box W10 y
#                   desde cero. La refractariedad ya tiene la suya (e16_B400).
#   e20_<conf>_vrd  β fijo en los parámetros verdaderos y sin perturbación en el
#                   backbone; se entrena solo la red. Mide cuánto del término
#                   faltante puede representar cada red: r2_residuo_test es contra
#                   f_planta − f_WC(β verdadero), que es el término exacto.
#
# Prioridad: semilla 0 de las tres pruebas, con σ 0,05 antes que 0,01, y
# después las semillas 1 y 2 de e19_x0 y e20.
#
#   python scripts/cola_x0_libre_y_beta_verdadero.py

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import RES, correr
from cola_semillas_2s import datos

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.wilson_cowan.model import WilsonCowanParams

V5, V40 = ["--window", "10"], ["--window", "40"]
# (nombre e16 del modelo de partida, planta, argumentos de arquitectura)
CONFS = {
    "e1wb":   ("refrac1", ["--variant", "whitebox"]),
    "B100":   ("refrac1", ["--variant", "B"]),
    "e1S2":   ("refrac1", ["--variant", "S"]),
    "e2wb":   ("act1",    ["--variant", "whitebox"]),
    "H100":   ("act1",    ["--variant", "H", "--hist", "10"]),
    "Hp100":  ("act1",    ["--variant", "Hp", "--hist", "11"]),
    "K400f8": ("act1",    ["--variant", "K", "--hist", "40", "--n-fir", "8"]),
}
WB = {"refrac1": "e1wb", "act1": "e2wb"}
# β verdadero: agnóstica de cada planta y las dos con historia que mejor dieron
VRD = {"B100": "refrac1", "e2B": "act1", "H100": "act1", "K400f8": "act1"}
ARQ_VRD = {"B100": ["--variant", "B"], "e2B": ["--variant", "B"],
           "H100": CONFS["H100"][1], "K400f8": CONFS["K400f8"][1]}
NIVELES = ["_n05", "_n01"]
VERDADERO = RES / "parametros_verdaderos.json"


def partida(n, suf, k):
    # los white-box no llevan _wrm en el tag de e16
    return f"e16_{n}{suf}_s{k}" if n in WB.values() else f"e16_{n}_wrm{suf}_s{k}"


def x0(k):
    out = []
    for suf in NIVELES:
        for n, (p, a) in CONFS.items():
            pt = RES / "models" / f"{partida(n, suf, k)}.pt"
            out.append((f"e19_{n}_x0{suf}_s{k}",
                        [*datos(p, suf, k), *a, *V5, "--x0-libre", "1", "--init-model", str(pt)], None))
    return out


def w40(k):
    wrm, cero = [], []
    for suf in NIVELES:
        for n, (p, a) in CONFS.items():
            if n == "B100":
                continue
            if n in WB.values():
                cero.append((f"e19_{n}_W40{suf}_s{k}", [*datos(p, suf, k), *a, *V40], None))
                continue
            wb = RES / f"e16_{WB[p]}{suf}_s{k}.json"
            r = ["--r-init", "0.05"] if n == "e1S2" else []   # como e16_e1S2
            wrm.append((f"e19_{n}_W40_wrm{suf}_s{k}",
                        [*datos(p, suf, k), *a, *V40, *r, "--init-params", str(wb)], None))
            if n != "e1S2":   # la forma exacta solo desde el white-box
                cero.append((f"e19_{n}_W40{suf}_s{k}", [*datos(p, suf, k), *a, *V40], None))
    return wrm + cero


def vrd(k):
    return [(f"e20_{n}_vrd{suf}_s{k}",
             [*datos(p, suf, k), *ARQ_VRD[n], *V5, "--init-params", str(VERDADERO), "--freeze-phys"], None)
            for suf in NIVELES for n, p in VRD.items()]


if __name__ == "__main__":
    VERDADERO.write_text(json.dumps({"params": asdict(WilsonCowanParams())}, indent=2), encoding="utf-8")
    tareas = x0(0) + vrd(0) + w40(0) + x0(1) + vrd(1) + x0(2) + vrd(2)
    correr(tareas, marca="DONE_x0_libre_y_beta_verdadero", lugares=3)
