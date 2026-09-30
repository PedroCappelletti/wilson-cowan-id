#!/usr/bin/env python3
# Tabla de la comparación entre entrenar desde cero y arrancar desde el
# white-box, para cada corrección con red y cada nivel de ruido.
#
# Escribe results/tabla_arranque.md y reemplaza la tabla de la nota de la vault
# entre los marcadores <!-- tabla --> y <!-- /tabla -->. Las celdas de corridas
# que no terminaron quedan en "…"; volver a correr el script las completa.
#
#   python scripts/tabla_arranque.py

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from divergencia import divergio   # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "results" / "escalado"
NOTA = Path(r"C:\Users\User\Desktop\Vault\01-Projects\Investigación Neurociencia"
            r"\Validación de arquitecturas\Arranque desde el white-box.md")

# (planta, corrección, {arranque: (tag σ=0.01, tag σ=0.05)})
FILAS = [
    ("refrac1", "agnóstica, ventana 20 ms", {
        "desde cero": ("e7_e1B400_n01", "e7_e1B400_n05"),
        "desde el white-box": ("e10_B400_wrm_n01", "e10_B400_wrm_n05"),
        "congelada": ("e10_B400_frz_n01", "e10_B400_frz_n05")}),
    ("refrac1", "agnóstica, ventana 5 ms", {
        "desde cero": ("e11_B100_cero_n01", "e11_B100_cero_n05"),
        "desde el white-box": ("e11_B100_wrm_n01", "e11_B100_wrm_n05")}),
    ("act1", "agnóstica g(I, E)", {
        "desde cero": ("e7_e2B_n01", "e7_e2B_n05"),
        "desde el white-box": ("e11_e2B_wrm_n01", "e11_e2B_wrm_n05")}),
    ("act1", "historia del comando, 100", {
        "desde cero": ("e8_H100_n01", "e8_H100_n05"),
        "desde el white-box": ("e11_H100_wrm_n01", "e11_H100_wrm_n05")}),
    ("act1", "historia del comando, 400", {
        "desde cero": ("e8_H400_n01", "e6_H400_n05_s7"),
        "desde el white-box": ("e11_H400_wrm_n01", "e11_H400_wrm_n05")}),
    ("act1", "filtro entrenable, 4 canales", {
        "desde cero": ("e8_K400_n01", "e6_K400_n05_s7"),
        "desde el white-box": ("e11_K400_wrm_n01", "e11_K400_wrm_n05")}),
    ("act1", "filtro entrenable, 8 canales", {
        "desde cero": ("e8_K400f8_n01", "e8_K400f8_n05"),
        "desde el white-box": ("e10_K400f8_wrm_n01", "e10_K400f8_wrm_n05"),
        "congelada": ("e10_K400f8_frz_n01", "e10_K400f8_frz_n05")}),
    ("act1", "filtro entrenable, 16 canales", {
        "desde cero": ("e9_K400f16_n01", "e9_K400f16_n05"),
        "desde el white-box": ("e11_K400f16_wrm_n01", "e11_K400f16_wrm_n05")}),
    ("act1", "filtro, 4 canales, L2 1e-4", {
        "desde cero": ("e8_K400wd4_n01", "e8_K400wd4_n05"),
        "desde el white-box": ("e11_K400wd4_wrm_n01", "e11_K400wd4_wrm_n05")}),
    ("act1", "filtro, 4 canales, L2 1e-3", {
        "desde cero": ("e8_K400wd3_n01", "e8_K400wd3_n05"),
        "desde el white-box": ("e11_K400wd3_wrm_n01", "e11_K400wd3_wrm_n05")}),
]
# Referencias sin red, para leer las filas.
REFERENCIAS = [
    ("refrac1", "white-box", ("e7_e1wb_n01", "e7_e1wb_n05")),
    ("refrac1", "forma exacta (backbone expandido)", ("e7_e1S2_n01", "e7_e1S2_n05")),
    ("act1", "white-box", ("e7_e2wb_n01", "e6_wb_n05_s7")),
    ("act1", "estado de filtro (backbone expandido)", ("e7_e2lag_n01", "e7_e2lag_n05")),
]


def celda(tag, r2=True):
    f = RES / f"{tag}.json"
    if not f.exists():
        return "…"
    d = json.loads(f.read_text(encoding="utf-8"))
    txt = f"{d['nrmse_test']:.2f} / {d['mean_param_error']:.1f}"
    txt += f" / {d['r2_delta_test']:+.2f}" if r2 else " / —"
    return (txt + " †" if divergio(tag) else txt).replace(".", ",")


def tabla():
    lin = ["| planta | corrección | arranque | σ = 0,01 | σ = 0,05 |",
           "|---|---|---|---|---|"]
    for planta, corr, brazos in FILAS:
        for i, (arranque, (t1, t5)) in enumerate(brazos.items()):
            lin.append(f"| {'`' + planta + '`' if i == 0 else ''} | {corr if i == 0 else ''} "
                       f"| {arranque} | {celda(t1)} | {celda(t5)} |")
    lin.append("| | *referencias sin red* | | | |")
    for planta, nombre, (t1, t5) in REFERENCIAS:
        r2 = "white-box" not in nombre
        lin.append(f"| `{planta}` | {nombre} | | {celda(t1, r2)} | {celda(t5, r2)} |")
    faltan = sum(celda(t) == "…" for _, _, b in FILAS for par in b.values() for t in par)
    return "\n".join(lin), faltan


if __name__ == "__main__":
    t, faltan = tabla()
    (RAIZ / "results" / "tabla_arranque.md").write_text(t + "\n", encoding="utf-8")
    if NOTA.exists():
        s = NOTA.read_text(encoding="utf-8")
        a, b = s.index("<!-- tabla -->"), s.index("<!-- /tabla -->")
        s = s[:a] + "<!-- tabla -->\n" + t + "\n" + s[b:]
        s = __import__("re").sub(r"faltan \d+ corridas", f"faltan {faltan} corridas", s)
        NOTA.write_text(s, encoding="utf-8")
    print(t)
    print(f"\nfaltan {faltan} corridas")
