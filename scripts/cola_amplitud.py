#!/usr/bin/env python3
# Todos los modelos sobre los datasets de amplitud grande (esc_gen_amplitud.py).
#
# Las mismas configuraciones que e16 (refractaria) y e21 (actuador con recorte),
# con la grilla de amplitudes de 1 a 12. Tres semillas, los dos ruidos.
#
#   e25  primera etapa: white-box, y cada corrección desde el white-box (_wrm)
#        y desde cero. La forma exacta refractaria solo desde el white-box.
#        Más el white-box sobre la planta ideal, como referencia del sesgo.
#   e26  segunda etapa con x0 entrenable (--x0-libre 1, --init-model del e25
#        correspondiente), como e19, e22 y e23.
#   e27  referencia black-box, BB y BBH con red de 64 y 3000 épocas, desde cero.
#
# Orden: e25 entero por semilla (la 0 primero), después e26, después e27, que es
# la más larga. Cada corrida espera a la que le da su punto de partida.
#
#   python scripts/esc_gen_amplitud.py
#   python scripts/esc_gen_ruido.py <planta>_amp_2s 0.01 0.05 --ref-fs 20000 [--semilla k]
#   python scripts/cola_amplitud.py [lugares]

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola import RES, correr

V5 = ["--window", "10"]
NIVELES = ["_n05", "_n01"]
SEMILLAS = [0, 1, 2]
X0 = ["--x0-libre", "1"]

# (nombre, argumentos, se corre desde cero)
ACT = [("lag",    ["--variant", "lag"], True),
       ("K400f8", ["--variant", "K", "--hist", "40", "--n-fir", "8"], True),
       ("H100",   ["--variant", "H", "--hist", "10"], True),
       ("Hp100",  ["--variant", "Hp", "--hist", "11"], True),
       ("e2B",    ["--variant", "B"], True)]
REF = [("B100",   ["--variant", "B"], True),
       ("B400",   ["--variant", "B", "--window", "40"], True),
       # la forma exacta solo desde el white-box: desde cero colapsa a él
       ("e1S2",   ["--variant", "S", "--r-init", "0.05"], False)]
PLANTAS = {"actc": ("e2wb", ACT), "refrac": ("e1wb", REF)}
# segunda etapa: (planta, modelo de partida)
SEGUNDA = [("actc", "e2wb"), ("actc", "lag_wrm"), ("actc", "H100_wrm"), ("actc", "Hp100_wrm"),
           ("actc", "K400f8_wrm"), ("refrac", "e1wb"), ("refrac", "B100_wrm"), ("refrac", "e1S2_wrm")]
BLACKBOX = {"BB64": ["--variant", "BB", "--hidden", "64", "--epochs", "3000"],
            "BBH64": ["--variant", "BBH", "--hist", "10", "--hidden", "64", "--epochs", "3000"]}


def datos(planta, suf, k):
    nombre = f"{planta}_amp_2s{suf}" + (f"_s{k}" if k else "")
    return ["--data", nombre, "--seed", str(k)]


def args_de(planta, nombre):
    """Argumentos de esc_run de una configuración, con la ventana por defecto."""
    for n, a, _ in PLANTAS[planta][1]:
        if n == nombre:
            return a if "--window" in a else [*a, *V5]
    if nombre == PLANTAS[planta][0]:
        return ["--variant", "whitebox", *V5]
    raise KeyError(nombre)


def tareas():
    primera, segunda, bb = [], [], []
    for k in SEMILLAS:
        for suf in NIVELES:
            primera.append((f"e25_wc0wb{suf}_s{k}", [*datos("wc0", suf, k), "--variant", "whitebox", *V5], None))
            for planta, (wb, configs) in PLANTAS.items():
                twb = f"e25_{wb}{suf}_s{k}"
                primera.append((twb, [*datos(planta, suf, k), *args_de(planta, wb)], None))
                for n, a, con_cero in configs:
                    aa = args_de(planta, n)
                    primera.append((f"e25_{n}_wrm{suf}_s{k}",
                                    [*datos(planta, suf, k), *aa, "--init-params", str(RES / f"{twb}.json")], twb))
                    if con_cero:
                        primera.append((f"e25_{n}{suf}_s{k}", [*datos(planta, suf, k), *aa], None))
            for planta, partida in SEGUNDA:
                base = f"e25_{partida}{suf}_s{k}"
                n = partida.removesuffix("_wrm")
                segunda.append((f"e26_{n}_x0{suf}_s{k}",
                                [*datos(planta, suf, k), *args_de(planta, n), *X0,
                                 "--init-model", str(RES / "models" / f"{base}.pt")], base))
            for planta, corto in (("actc", "act"), ("refrac", "ref")):
                for n, a in BLACKBOX.items():
                    bb.append((f"e27_{n}_{corto}{suf}_s{k}", [*datos(planta, suf, k), *a, *V5], None))
    return primera + segunda + bb


if __name__ == "__main__":
    lugares = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    correr(tareas(), marca="DONE_amplitud", lugares=lugares)
