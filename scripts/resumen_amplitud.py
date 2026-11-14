#!/usr/bin/env python3
# Tablas de lazo abierto de los modelos entrenados con amplitud grande (e25,
# e26, e27), frente a los mismos modelos entrenados con los datos de [0, 2]
# (e21 para el actuador con recorte, e16 para la refractaria).
#
# Las dos tandas se comparan sobre los mismos 13 escenarios de test que
# interpolan del dataset anterior, que están incluidos en el test nuevo. El
# NRMSE sobre el test nuevo (amplitudes fuera de la grilla) solo existe para e25
# en adelante.
#
#   python scripts/resumen_amplitud.py      -> results/resumen_amplitud.json

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
RES = RAIZ / "results" / "escalado"
OUT = RAIZ / "results" / "resumen_amplitud.json"
NIVELES = ("_n01", "_n05")

from gen_multi_dataset import EXTRAPOLACION, build_scenarios_ampliado  # noqa: E402

VIEJOS = [lab for lab, *_, rol in build_scenarios_ampliado(100.0, 1900.0) if rol == "test"]
VIEJOS_INTERP = [l for l in VIEJOS if l not in EXTRAPOLACION]

# (nombre en las tablas, tag nuevo, tag con los datos anteriores)
ACT = [("white-box", "e25_e2wb", "e21_e2wb"),
       ("agnóstica, warm start", "e25_e2B_wrm", "e21_e2B_wrm"),
       ("agnóstica, desde cero", "e25_e2B", "e21_e2B"),
       ("historia con comando actual, warm start", "e25_H100_wrm", "e21_H100_wrm"),
       ("historia con comando actual, desde cero", "e25_H100", "e21_H100"),
       ("historia sin comando actual, warm start", "e25_Hp100_wrm", "e21_Hp100_wrm"),
       ("historia sin comando actual, desde cero", "e25_Hp100", "e21_Hp100"),
       ("filtro 8 canales, warm start", "e25_K400f8_wrm", "e21_K400f8_wrm"),
       ("filtro 8 canales, desde cero", "e25_K400f8", "e21_K400f8"),
       ("estado de filtro, warm start", "e25_lag_wrm", "e21_lag_wrm"),
       ("estado de filtro, desde cero", "e25_lag", "e21_lag"),
       ("white-box, x0", "e26_e2wb_x0", "e22_e2wb_x0"),
       ("historia con comando actual, x0", "e26_H100_x0", "e22_H100_x0"),
       ("historia sin comando actual, x0", "e26_Hp100_x0", "e22_Hp100_x0"),
       ("filtro 8 canales, x0", "e26_K400f8_x0", "e22_K400f8_x0"),
       ("estado de filtro, x0", "e26_lag_x0", "e23_lag_x0"),
       ("black-box", "e27_BB64_act", "e24_BB64_act"),
       ("black-box con historia", "e27_BBH64_act", "e24_BBH64_act")]
REF = [("white-box", "e25_e1wb", "e16_e1wb"),
       ("agnóstica 5 ms, warm start", "e25_B100_wrm", "e16_B100_wrm"),
       ("agnóstica 5 ms, desde cero", "e25_B100", "e16_B100"),
       ("agnóstica 20 ms, warm start", "e25_B400_wrm", "e16_B400_wrm"),
       ("agnóstica 20 ms, desde cero", "e25_B400", "e16_B400"),
       ("forma exacta, warm start", "e25_e1S2_wrm", "e16_e1S2_wrm"),
       ("white-box, x0", "e26_e1wb_x0", "e19_e1wb_x0"),
       ("agnóstica 5 ms, x0", "e26_B100_x0", "e19_B100_x0"),
       ("forma exacta, x0", "e26_e1S2_x0", "e19_e1S2_x0"),
       ("black-box", "e27_BB64_ref", "e24_BB64_ref"),
       ("black-box con historia", "e27_BBH64_ref", "e24_BBH64_ref")]


def cargar(prefijo, suf):
    out = []
    for k in (0, 1, 2):
        p = RES / f"{prefijo}{suf}_s{k}.json"
        if p.exists():
            out.append(json.loads(p.read_text()))
    return out


def media(vals):
    vals = [v for v in vals if v is not None]
    return round(st.mean(vals), 4) if vals else None


def sobre(d, labels):
    por = {e["label"]: e["nrmse"] for e in d["por_escenario"]}
    return st.mean(por[l] for l in labels if l in por)


def fila(prefijo, suf, nuevo):
    ds = cargar(prefijo, suf)
    if not ds:
        return None
    r = {"n": len(ds),
         "beta": media([d["mean_param_error"] for d in ds]),
         "r2": media([d["r2_delta_test"] for d in ds]),
         "nrmse_viejo13": media([sobre(d, VIEJOS_INTERP) for d in ds]),
         "minutos": media([d["minutos"] for d in ds])}
    if nuevo:
        labs = [e["label"] for e in ds[0]["por_escenario"] if e["label"] not in VIEJOS]
        r["nrmse_nuevo"] = media([sobre(d, labs) for d in ds])
    ex = [d.get("extras") for d in ds if d.get("extras")]
    if ex and "tau" in ex[0]:
        r["tau"] = media([e["tau"] for e in ex])
        r["sat"] = media([e["sat"] for e in ex])
    return r


def tabla(lista):
    out = []
    for nombre, nuevo, viejo in lista:
        for suf in NIVELES:
            out.append({"modelo": nombre, "ruido": suf, "amplitud": fila(nuevo, suf, True),
                        "anterior": fila(viejo, suf, False)})
    return out


if __name__ == "__main__":
    res = {"actuador": tabla(ACT), "refractaria": tabla(REF),
           "ideal_white_box": {suf: fila("e25_wc0wb", suf, True) for suf in NIVELES}}
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    f2 = lambda v, w=6, d=2: f"{v:{w}.{d}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"
    for planta in ("actuador", "refractaria"):
        print(f"== {planta}  (beta % / R2 / NRMSE 13 viejos / NRMSE test nuevo)  [anterior: beta / R2 / NRMSE 13]")
        for f in res[planta]:
            a, b = f["amplitud"], f["anterior"]
            fa = (f"{f2(a['beta'])} {f2(a['r2'],7,3)} {f2(a['nrmse_viejo13'])} {f2(a['nrmse_nuevo'])}"
                  + (f" tau {a['tau']:.3f} sat {a['sat']:.2f}" if 'tau' in a else "")) if a else "  -"
            fb = f"{f2(b['beta'])} {f2(b['r2'],7,3)} {f2(b['nrmse_viejo13'])}" if b else "  -"
            print(f"  {f['modelo']:44s}{f['ruido']}  {fa}   [{fb}]")
    print("ideal white-box:", res["ideal_white_box"])
