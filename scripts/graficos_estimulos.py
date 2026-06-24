#!/usr/bin/env python3
# =============================================================================
#  GRÁFICOS DE ESTÍMULOS — antes vs ahora + comportamiento del sistema
# =============================================================================
#  Para cada estímulo grafica la entrada (P, Q) junto a la respuesta del sistema
#  Wilson-Cowan (I, E). Tres bloques:
#    ANTES  — los estímulos originales (box + senoidales).
#    AHORA  — la librería nueva tipo pulso (cuadrada, APRBS, PRBS, theta-gamma, Poisson).
#    AJUSTE — el cambio clave para wII: Q débil (~1) vs Q fuerte (~5) y su efecto en I.
#  Genera docs/informe_estimulos.html (figuras embebidas).
#  USO: python scripts/graficos_estimulos.py
# =============================================================================

from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.wilson_cowan import (
    WilsonCowan, WilsonCowanParams, zero_input,
    box_pulse, sine_pulse, multisine_pulse, chirp_pulse,
    aprbs_pulse, theta_gamma_pulse, square_wave_pulse, prbs_pulse, poisson_pulse,
)

PARAMS = WilsonCowanParams()
T_SPAN = (0.0, 600.0)
N = 6000
t_eval = np.linspace(T_SPAN[0], T_SPAN[1], N)
FIGDIR = Path("results/figures")
OUT_HTML = Path("docs/informe_estimulos.html")


def sim(P, Q):
    return WilsonCowan(PARAMS, P=P, Q=Q).simulate(t_span=T_SPAN, t_eval=t_eval)


# --- ANTES: estímulos originales (box + senoidales, luego descartadas) ---
ANTES = [
    ("Pulso cuadrado (box) — baseline", box_pulse(0.8, 100, 400), box_pulse(0.6, 200, 500)),
    ("Senoide", sine_pulse(0.4, 0.02, 100, 400), sine_pulse(0.3, 0.015, 150, 450)),
    ("Multiseno", multisine_pulse(0.6, [0.01, 0.025, 0.05], 100, 400),
     multisine_pulse(0.45, [0.012, 0.03], 150, 450)),
    ("Chirp", chirp_pulse(0.4, 0.005, 0.05, 100, 400), chirp_pulse(0.3, 0.005, 0.04, 150, 450)),
]

# --- AHORA: librería nueva, tipo pulso (on/off, realizable) ---
AHORA = [
    ("Onda cuadrada / tren de pulsos", square_wave_pulse(0.8, 0.03, 100, 500, 0.4),
     square_wave_pulse(0.6, 0.02, 100, 500, 0.5)),
    ("APRBS (escalones aleatorios)", aprbs_pulse(0.9, 100, 500, 15, 60, seed=1, amp_min=0.2),
     aprbs_pulse(0.7, 100, 500, 18, 65, seed=2, amp_min=0.1)),
    ("PRBS (binario)", prbs_pulse(0.8, 100, 500, 20, seed=1), prbs_pulse(0.6, 100, 500, 25, seed=2)),
    ("Theta-gamma (ráfagas)", theta_gamma_pulse(0.8, 0.05, 0.008, 100, 500, 0.5),
     theta_gamma_pulse(0.6, 0.04, 0.007, 100, 500, 0.5)),
    ("Tren de Poisson", poisson_pulse(0.8, 0.05, 100, 500, 8, seed=1),
     poisson_pulse(0.6, 0.04, 100, 500, 8, seed=2)),
    ("Chirp (conservado)", chirp_pulse(0.4, 0.005, 0.05, 100, 400),
     chirp_pulse(0.3, 0.005, 0.04, 150, 450)),
]

# --- AJUSTE clave para wII: MISMO estímulo (APRBS), solo cambia la amplitud de Q.
#     P apagado para aislar el efecto de Q sobre la población inhibitoria I.
AJUSTE = [
    ("APRBS — Q débil (~1)", zero_input, aprbs_pulse(1.0, 100, 500, 18, 55, seed=3, amp_min=0.3)),
    ("APRBS — Q fuerte (~5)", zero_input, aprbs_pulse(5.0, 100, 500, 18, 55, seed=3, amp_min=2.0)),
]


def make_fig(sets, path, suptitle):
    import matplotlib.pyplot as plt
    n = len(sets)
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.3 * n))
    if n == 1:
        axes = axes.reshape(1, 2)
    for r, (label, P, Q) in enumerate(sets):
        s = sim(P, Q)
        ax = axes[r, 0]
        ax.plot(s["t"], s["P"], label="P → E", lw=1.2, color="#1f4e79")
        ax.plot(s["t"], s["Q"], label="Q → I", lw=1.2, color="#d62728")
        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, alpha=0.3)
        if r == 0:
            ax.set_title("Estímulo (P, Q)")
            ax.legend(fontsize=7, loc="upper right")
        ax = axes[r, 1]
        ax.plot(s["t"], s["E"], label="E (excitatoria)", lw=1.0, color="#0a7d33")
        ax.plot(s["t"], s["I"], label="I (inhibitoria)", lw=1.0, color="#9467bd")
        ax.grid(True, alpha=0.3)
        if r == 0:
            ax.set_title("Comportamiento del sistema (I, E)")
            ax.legend(fontsize=7, loc="upper right")
    axes[-1, 0].set_xlabel("tiempo (s)")
    axes[-1, 1].set_xlabel("tiempo (s)")
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=115)
    plt.close(fig)


def _b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


def main():
    f_antes = FIGDIR / "estim_antes.png"
    f_ahora = FIGDIR / "estim_ahora.png"
    f_ajuste = FIGDIR / "estim_ajuste.png"
    make_fig(ANTES, f_antes, "ANTES — estímulos originales (box + senoidales)")
    make_fig(AHORA, f_ahora, "AHORA — librería nueva tipo pulso (realizable on/off)")
    make_fig(AJUSTE, f_ajuste, "AJUSTE clave para wII — efecto de Q en la población inhibitoria I")
    print("Figuras:", f_antes, f_ahora, f_ajuste)

    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
<title>Estímulos: antes vs ahora + comportamiento del sistema</title>
<style>
 body{{font-family:-apple-system,"Segoe UI",Arial,sans-serif;color:#1c2733;line-height:1.6;margin:0;background:#f4f6f8;padding:2rem 1rem}}
 .wrap{{max-width:1040px;margin:0 auto}}
 header{{background:#1f4e79;color:#fff;border-radius:12px 12px 0 0;padding:1.5rem 1.9rem}}
 header h1{{margin:0;font-size:1.3rem}}
 main{{background:#fff;border:1px solid #d7dee5;border-top:none;border-radius:0 0 12px 12px;padding:1.5rem 1.9rem 2rem}}
 h2{{color:#1f4e79;font-size:1.15rem;border-bottom:2px solid #e8f0f8;padding-bottom:.3rem;margin-top:1.8rem}}
 img{{width:100%;border:1px solid #d7dee5;border-radius:8px;margin:.5rem 0}}
 .nota{{background:#e8f0f8;border-left:4px solid #1f4e79;padding:.7rem 1rem;border-radius:0 6px 6px 0;margin:1rem 0;font-size:.92rem}}
 .ok{{background:#e7f6ec;border-left-color:#0a7d33}}
 .ajuste{{font-size:.9rem;color:#555}}
</style></head><body><div class="wrap">
<header><h1>Estímulos utilizados — antes vs ahora, con la respuesta del sistema</h1></header>
<main>
 <div class="nota">Cada fila muestra el <b>estímulo</b> (P→E en azul, Q→I en rojo) junto al
 <b>comportamiento del sistema</b> (E verde, I violeta). Simulación de Wilson-Cowan, t en segundos.</div>

 <h2>1 · ANTES — estímulos originales</h2>
 <p>Se usaba el pulso cuadrado (box) y, como primer intento de excitación más rica, senoidales
 (senoide, multiseno, chirp). El pulso cuadrado "mueve poco" el sistema; las senoidales no son on/off
 (no realizables con optogenética), por eso se descartaron.</p>
 <img src="data:image/png;base64,{_b64(f_antes)}">
 <p class="ajuste"><b>Figura 1 — Estímulos originales.</b> Una fila por estímulo (box, senoide,
 multiseno, chirp). Columna izquierda: la entrada (P→E azul, Q→I rojo); columna derecha: la respuesta
 del sistema (E verde, I violeta). Se ve que el box es casi constante (excita poco) y que las
 senoidales recorren más estados pero no prenden/apagan.</p>

 <h2>2 · AHORA — librería nueva tipo pulso</h2>
 <p>Estímulos realizables (on/off, ≥ 0): onda cuadrada, APRBS, PRBS, theta-gamma, Poisson (+ chirp
 conservado por su cobertura espectral). Excitan el sistema de forma más rica y variada.</p>
 <img src="data:image/png;base64,{_b64(f_ahora)}">
 <p class="ajuste"><b>Figura 2 — Librería nueva.</b> Misma disposición que la Figura 1 (izq.: entrada
 P/Q; der.: respuesta I/E), una fila por estímulo nuevo. Comparado con el box, estas entradas hacen
 que I y E recorran muchos más estados — mejor para identificar.</p>

 <h2>3 · El ajuste clave para wII — Q débil vs Q fuerte (mismo estímulo)</h2>
 <p>Las dos filas usan <b>el mismo estímulo (APRBS)</b> con P apagado; lo único que cambia es la
 <b>amplitud de Q</b>. Así se aísla el efecto de Q sobre la población inhibitoria I. Con <b>Q débil
 (~1)</b>, I casi no se mueve (no se puede identificar wII). Con <b>Q fuerte (~5)</b>, I se activa de
 verdad y se mueve independiente de E — la condición para identificar wII.</p>
 <img src="data:image/png;base64,{_b64(f_ajuste)}">
 <p class="ajuste"><b>Figura 3 — Efecto de la amplitud de Q.</b> Mismo APRBS en ambas filas (P apagado),
 solo cambia la amplitud de Q (izq.). A la derecha, la respuesta: con Q débil I queda casi plana; con
 Q fuerte I se levanta. La diferencia es por la amplitud, no por el tipo de estímulo.</p>
 <div class="nota ok">Este es el cambio que destrabó la identificación de wII: empujar la población
 inhibitoria con Q grande para que I sea grande y decorrelado de E.</div>
</main></div></body></html>"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Informe: {OUT_HTML}")


if __name__ == "__main__":
    main()
