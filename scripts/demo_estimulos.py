#!/usr/bin/env python3
# =============================================================================
#  DEMO — efecto de entradas P,Q variables en el tiempo (excitacion persistente)
# =============================================================================
#
#  Compara como responde el modelo de Wilson-Cowan a distintos tipos de entrada:
#    1. Pulso cuadrado (baseline, lo que usabamos)
#    2. Senoide
#    3. Multiseno (suma de frecuencias)
#    4. Chirp (barrido de frecuencia)
#
#  Para cada uno grafica: las entradas, los estados I(t),E(t), y el retrato de
#  fase (E vs I). El retrato de fase muestra VISUALMENTE la idea: un pulso
#  cuadrado traza un lazo finito (el ciclo limite); las entradas variables
#  llenan mas plano -> exploran mas estados -> mejor identificabilidad.
#
#  USO:  python scripts/demo_estimulos.py
# =============================================================================

from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.wilson_cowan import (
    WilsonCowan, WilsonCowanParams,
    box_pulse, sine_pulse, multisine_pulse, chirp_pulse,
)

# #############################################################################
# ##   ZONA EDITABLE                                                         ##
# #############################################################################

PARAMS = WilsonCowanParams()   # parametros nominales (ciclo limite)
T_SPAN = (0.0, 600.0)
N_EVAL = 6000
I0, E0 = 0.0, 0.0

# Cada escenario define (P, Q). Las frecuencias estan elegidas en el rango lento
# del sistema (te=1, ti=2). Ajustables.
ESCENARIOS = {
    "Pulso cuadrado (baseline)": (
        box_pulse(0.8, 100.0, 400.0),
        box_pulse(0.6, 200.0, 500.0),
    ),
    "Senoide": (
        sine_pulse(amplitude=0.4, freq=0.02, t_on=100.0, t_off=400.0),
        sine_pulse(amplitude=0.3, freq=0.015, t_on=150.0, t_off=450.0),
    ),
    "Multiseno": (
        multisine_pulse(0.6, [0.01, 0.025, 0.05], 100.0, 400.0),
        multisine_pulse(0.45, [0.012, 0.03], 150.0, 450.0),
    ),
    "Chirp (barrido)": (
        chirp_pulse(0.4, f0=0.005, f1=0.05, t_on=100.0, t_off=400.0),
        chirp_pulse(0.3, f0=0.005, f1=0.04, t_on=150.0, t_off=450.0),
    ),
}

OUT_DIR = Path("results")

# #############################################################################
# ##   FIN ZONA EDITABLE                                                     ##
# #############################################################################


def simular(P, Q):
    modelo = WilsonCowan(params=PARAMS, P=P, Q=Q)
    t = np.linspace(T_SPAN[0], T_SPAN[1], N_EVAL)
    return modelo.simulate(I0=I0, E0=E0, t_span=T_SPAN, t_eval=t)


def main():
    import matplotlib.pyplot as plt

    sols = {nombre: simular(P, Q) for nombre, (P, Q) in ESCENARIOS.items()}

    n = len(sols)
    fig, axes = plt.subplots(n, 3, figsize=(14, 3.2 * n))
    cobertura = {}

    for i, (nombre, sol) in enumerate(sols.items()):
        t, I, E = sol["t"], sol["I"], sol["E"]

        # Col 1: entradas
        ax = axes[i, 0]
        ax.plot(t, sol["P"], label="P → E", lw=1.2)
        ax.plot(t, sol["Q"], label="Q → I", lw=1.2)
        ax.set_ylabel(nombre, fontsize=9)
        if i == 0:
            ax.set_title("Entradas P, Q")
        ax.legend(fontsize=7, loc="upper right"); ax.grid(True, alpha=0.3)

        # Col 2: estados
        ax = axes[i, 1]
        ax.plot(t, I, label="I", lw=1.0)
        ax.plot(t, E, label="E", lw=1.0)
        if i == 0:
            ax.set_title("Estados I(t), E(t)")
        ax.legend(fontsize=7, loc="upper right"); ax.grid(True, alpha=0.3)

        # Col 3: retrato de fase E vs I (cobertura del espacio de estados)
        ax = axes[i, 2]
        ax.plot(E, I, lw=0.6, alpha=0.8, color="#5b2a9e")
        ax.set_xlabel("E"); ax.set_ylabel("I")
        if i == 0:
            ax.set_title("Retrato de fase (E vs I)")
        ax.grid(True, alpha=0.3)

        # Metrica simple de "cuanto explora": dispersion en el plano (E,I).
        cobertura[nombre] = float(np.std(E) * np.std(I))

    axes[-1, 0].set_xlabel("tiempo (s)")
    axes[-1, 1].set_xlabel("tiempo (s)")
    fig.tight_layout()

    fig_path = OUT_DIR / "figures" / "demo_estimulos.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=120); plt.close(fig)

    # --- Tabla de cobertura (normalizada al baseline) ---
    base = cobertura["Pulso cuadrado (baseline)"]
    print("\n=== Cobertura del espacio de estados  std(E)*std(I) ===")
    print(f"{'escenario':32} {'cobertura':>12} {'x baseline':>12}")
    for nombre, c in cobertura.items():
        print(f"{nombre:32} {c:12.4e} {c/base:11.2f}x")

    _build_html(cobertura, base, fig_path)
    print(f"\nFigura: {fig_path}")


def _build_html(cobertura, base, fig_path):
    img = base64.b64encode(fig_path.read_bytes()).decode()
    filas = "".join(
        f"<tr><td>{nombre}</td><td class='num'>{c:.3e}</td>"
        f"<td class='num'><b>{c/base:.2f}×</b></td></tr>"
        for nombre, c in cobertura.items()
    )
    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
<title>Demo — entradas variables en el tiempo</title>
<style>
 body{{font-family:-apple-system,"Segoe UI",Arial,sans-serif;color:#1c2733;line-height:1.6;
  margin:0;background:#f4f6f8;padding:2rem 1rem}}
 .wrap{{max-width:1000px;margin:0 auto}}
 header{{background:#1f4e79;color:#fff;border-radius:12px 12px 0 0;padding:1.5rem 1.9rem}}
 header h1{{margin:0;font-size:1.3rem}}
 main{{background:#fff;border:1px solid #d7dee5;border-top:none;border-radius:0 0 12px 12px;padding:1.5rem 1.9rem 2rem}}
 h2{{color:#1f4e79;font-size:1.1rem;border-bottom:2px solid #e8f0f8;padding-bottom:.3rem}}
 img{{width:100%;border:1px solid #d7dee5;border-radius:8px;margin:.5rem 0}}
 table{{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:.9rem}}
 th,td{{text-align:left;padding:.45rem .7rem;border-bottom:1px solid #d7dee5}}
 th{{background:#e8f0f8;color:#1f4e79}} td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
 .nota{{background:#e8f0f8;border-left:4px solid #1f4e79;padding:.7rem 1rem;border-radius:0 6px 6px 0;margin:1rem 0;font-size:.92rem}}
</style></head><body><div class="wrap">
<header><h1>Efecto de entradas P, Q variables en el tiempo</h1></header>
<main>
 <div class="nota">Un pulso cuadrado es casi constante mientras está prendido → excita poco.
 Entradas que varían en el tiempo recorren más estados (ver retrato de fase) → el problema de
 identificación queda mejor condicionado (excitación persistente).</div>
 <h2>Cobertura del espacio de estados</h2>
 <table><tr><th>Escenario</th><th class="num">std(E)·std(I)</th><th class="num">vs baseline</th></tr>
 {filas}</table>
 <h2>Comparación (entradas · estados · retrato de fase)</h2>
 <img src="data:image/png;base64,{img}">
</main></div></body></html>"""
    out = OUT_DIR / "reporte_demo_estimulos.html"
    out.write_text(html, encoding="utf-8")
    print(f"Reporte HTML: {out}")


if __name__ == "__main__":
    main()
