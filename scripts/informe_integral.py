#!/usr/bin/env python3
# =============================================================================
#  INFORME INTEGRAL — Lazo cerrado por TIPO de referencia (estímulo a seguir)
# =============================================================================
#
#  Para cada TIPO de referencia (theta-gamma, APRBS, chirp) y varias FORMAS de
#  cada una, se evalúa la matriz 2x2 de configuraciones:
#
#     PLANTA \ CONTROLADOR     params identificados (θ̂)   params reales
#     Simulador (WC verdadero)        caso real                baseline ideal
#     Neural ODE (aprendido)          lazo 100% aprendido      planta apr.+ctrl ideal
#
#  En lazo cerrado la entrada P,Q la genera el controlador; lo que varía es la
#  REFERENCIA (la consigna que debe imponer a las poblaciones I,E). Se prueban
#  referencias de distinta forma y se mide el RMSE de seguimiento de cada config.
#
#  Genera docs/informe_integral.html. USO: python scripts/informe_integral.py
# =============================================================================

from __future__ import annotations

import base64
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.wilson_cowan import WilsonCowanParams
from src.neural_ode import (
    GrayBoxWC, IMCController, make_true_plant, make_neural_plant, simulate_closed_loop,
)

CKPT = Path("results/models/neural_ode.pt")
OUT_HTML = Path("docs/informe_integral.html")
FIGDIR = Path("results/figures")

T_SPAN = (0.0, 50.0)   # ms
DT = 0.005

# Rango fisiológico alcanzable de las referencias (de la forma theta-gamma base).
RI_LO, RI_HI = 0.05, 0.45
RE_LO, RE_HI = 0.15, 0.75


# -----------------------------------------------------------------------------
#  Constructores de referencia  refs(t) -> (rI, rE),  t en ms.
# -----------------------------------------------------------------------------
def ref_theta_gamma(freq_hz: float):
    """Sinusoide en banda gamma (la forma estándar del controlador del tutor)."""
    f = freq_hz / 1000.0
    def refs(t):
        return (0.2 * math.sin(2 * math.pi * f * t - 0.94) + 0.25,
                0.3 * math.sin(2 * math.pi * f * t) + 0.45)
    return refs


def ref_aprbs(dwell_ms: float, seed: int):
    """Escalones de nivel aleatorio (set-point cambiante) dentro del rango alcanzable."""
    rng = np.random.default_rng(seed)
    bordes = [0.0]; vI = []; vE = []; t = 0.0
    while t < T_SPAN[1]:
        t += rng.uniform(0.6 * dwell_ms, 1.4 * dwell_ms)
        bordes.append(t)
        vI.append(rng.uniform(RI_LO, RI_HI)); vE.append(rng.uniform(RE_LO, RE_HI))
    bordes = np.asarray(bordes)
    def refs(t):
        i = min(max(int(np.searchsorted(bordes, t, side="right")) - 1, 0), len(vI) - 1)
        return float(vI[i]), float(vE[i])
    return refs


def ref_chirp(f0_hz: float, f1_hz: float):
    """Sinusoide con frecuencia que barre de f0 a f1 (sweep gamma)."""
    f0, f1 = f0_hz / 1000.0, f1_hz / 1000.0
    k = (f1 - f0) / T_SPAN[1]
    def refs(t):
        ph = 2 * math.pi * (f0 * t + 0.5 * k * t * t)
        return 0.2 * math.sin(ph - 0.94) + 0.25, 0.3 * math.sin(ph) + 0.45
    return refs


# Tipos de referencia y sus formas. "ajuste" describe qué se eligió en cada caso.
REF_TYPES = {
    "theta-gamma": {
        "desc": ("Sinusoide en banda gamma: es el ritmo objetivo natural del proyecto y la forma "
                 "estándar del controlador. Se prueban dos frecuencias; a mayor frecuencia, más "
                 "exigente el seguimiento."),
        "ajuste": "Frecuencias 120 y 160 Hz (banda gamma típica).",
        "forms": [("120 Hz (base)", ref_theta_gamma(120)),
                  ("160 Hz", ref_theta_gamma(160))],
    },
    "APRBS": {
        "desc": ("Referencia de escalones de nivel y duración aleatorios (set-point cambiante). "
                 "Pone a prueba la respuesta del lazo a saltos abruptos en la consigna."),
        "ajuste": (f"Niveles dentro del rango alcanzable (rI∈[{RI_LO},{RI_HI}], rE∈[{RE_LO},{RE_HI}]); "
                   "dwell ~6 y ~12 ms (más corto = más exigente)."),
        "forms": [("escalones lentos (~12 ms)", ref_aprbs(12.0, 11)),
                  ("escalones rápidos (~6 ms)", ref_aprbs(6.0, 12))],
    },
    "chirp": {
        "desc": ("Sinusoide cuya frecuencia barre un rango (sweep). Muestra cómo se degrada el "
                 "seguimiento a medida que sube la frecuencia de la consigna."),
        "ajuste": "Barridos 80→200 Hz (ascendente) y 200→80 Hz (descendente).",
        "forms": [("80→200 Hz", ref_chirp(80, 200)),
                  ("200→80 Hz", ref_chirp(200, 80))],
    },
}

COMBOS = [
    ("Simulador + θ̂",      "true",   "hat"),
    ("Simulador + reales",  "true",   "real"),
    ("Neural ODE + θ̂",     "neural", "hat"),
    ("Neural ODE + reales", "neural", "real"),
]
COL = {"Simulador + θ̂": "#1f4e79", "Simulador + reales": "#7aa6cc",
       "Neural ODE + θ̂": "#d62728", "Neural ODE + reales": "#f4a3a3"}


def _rmse(sol):
    n0 = len(sol["t"]) // 5
    rI = float(np.sqrt(np.mean((sol["I"][n0:] - sol["rI"][n0:]) ** 2)))
    rE = float(np.sqrt(np.mean((sol["E"][n0:] - sol["rE"][n0:]) ** 2)))
    return rI, rE


def main():
    p = WilsonCowanParams()
    fixed = {"te": p.te, "ti": p.ti, "ae": p.ae, "ai": p.ai,
             "thetae": p.thetae, "thetai": p.thetai, "ke": p.ke, "ki": p.ki}
    w_real = {"wEE": p.wEE, "wEI": p.wEI, "wIE": p.wIE, "wII": p.wII}

    ck = torch.load(CKPT, weights_only=False)
    model = GrayBoxWC(ck["fixed"], {k: 1.0 for k in GrayBoxWC.WEIGHTS},
                      learnable_weights=ck["learn_weights"], use_correction=ck["use_correction"])
    model.load_state_dict(ck["state"]); model.eval()
    w_hat = model.weights_dict()

    plants = {"true": make_true_plant(fixed, w_real), "neural": make_neural_plant(model)}
    ctrls = {"real": IMCController(fixed, w_real), "hat": IMCController(fixed, w_hat)}

    results = {}  # (type, form_label, combo_label) -> {"rmse":(rI,rE),"sol":sol}
    for tname, tinfo in REF_TYPES.items():
        for flabel, refs in tinfo["forms"]:
            for clabel, pk, wk in COMBOS:
                sol = simulate_closed_loop(plants[pk], ctrls[wk], refs, t_span=T_SPAN, dt=DT)
                results[(tname, flabel, clabel)] = {"rmse": _rmse(sol), "sol": sol}
            rE = results[(tname, flabel, "Simulador + θ̂")]["rmse"][1]
            print(f"  {tname:12} {flabel:24} RMSE_E(Sim+θ̂)={rE:.2e}")

    _figs(results)
    _build_html(results, w_real, w_hat)
    print(f"\nInforme: {OUT_HTML}")


def _figs(results):
    import matplotlib.pyplot as plt
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for tname, tinfo in REF_TYPES.items():
        forms = tinfo["forms"]
        # (a) las referencias (formas) del tipo
        t = np.linspace(0, T_SPAN[1], 2000)
        fig, ax = plt.subplots(1, 2, figsize=(11, 3), sharex=True)
        for flabel, refs in forms:
            ax[0].plot(t, [refs(ti)[0] for ti in t], lw=1.1, label=flabel)
            ax[1].plot(t, [refs(ti)[1] for ti in t], lw=1.1, label=flabel)
        ax[0].set_title(f"{tname} — referencia rI"); ax[1].set_title(f"{tname} — referencia rE")
        for a in ax:
            a.set_xlabel("t (ms)"); a.grid(True, alpha=0.3); a.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(FIGDIR / f"int_ref_{tname}.png", dpi=115); plt.close(fig)

        # (b) seguimiento de las 4 configs sobre la 1ª forma del tipo
        flabel0 = forms[0][0]
        sol0 = results[(tname, flabel0, COMBOS[0][0])]["sol"]
        tt = sol0["t"]
        fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
        # Las 4 configs casi coinciden -> grosor decreciente para que se vean las 4
        # como bandas anidadas; la referencia (negra) se dibuja al final, arriba.
        lws = [3.8, 2.7, 1.7, 0.9]
        for (clabel, _, _), lw in zip(COMBOS, lws):
            s = results[(tname, flabel0, clabel)]["sol"]
            ax[0].plot(s["t"], s["I"], lw=lw, color=COL[clabel], label=clabel, alpha=0.9)
            ax[1].plot(s["t"], s["E"], lw=lw, color=COL[clabel], label=clabel, alpha=0.9)
        ax[0].plot(tt, sol0["rI"], "k--", lw=1.3, label="referencia")
        ax[1].plot(tt, sol0["rE"], "k--", lw=1.3, label="referencia")
        ax[0].set_ylabel("I"); ax[1].set_ylabel("E"); ax[1].set_xlabel("t (ms)")
        ax[0].set_title(f"{tname} — seguimiento ({flabel0})")
        for a in ax:
            a.grid(True, alpha=0.3); a.legend(fontsize=7, ncol=3, loc="upper right")
        fig.tight_layout(); fig.savefig(FIGDIR / f"int_track_{tname}.png", dpi=115); plt.close(fig)


def _b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


def _img(path, alt=""):
    p = Path(path)
    if not p.exists():
        return f"<p class='ajuste'>(figura no disponible: {p.name})</p>"
    return f'<img alt="{alt}" src="data:image/png;base64,{_b64(p)}">'


def _build_html(results, w_real, w_hat):
    wr = " · ".join(f"{k}={w_real[k]:.3f}" for k in w_real)
    wh = " · ".join(f"{k}={w_hat[k]:.3f}" for k in w_hat)

    secciones = ""
    for tname, tinfo in REF_TYPES.items():
        th = "".join(f"<th class='num'>{c[0]}</th>" for c in COMBOS)
        filas = ""
        for flabel, _ in tinfo["forms"]:
            celdas = ""
            for clabel, _, _ in COMBOS:
                rI, rE = results[(tname, flabel, clabel)]["rmse"]
                celdas += f"<td class='num'>I={rI:.2e}<br>E={rE:.2e}</td>"
            filas += f"<tr><td>{flabel}</td>{celdas}</tr>"
        secciones += f"""
 <h2>Referencia tipo: {tname}</h2>
 <p>{tinfo['desc']}</p>
 <p class="ajuste"><b>Ajuste:</b> {tinfo['ajuste']}</p>
 <img src="data:image/png;base64,{_b64(FIGDIR / f'int_ref_{tname}.png')}">
 <table><tr><th>Forma</th>{th}</tr>{filas}</table>
 <img src="data:image/png;base64,{_b64(FIGDIR / f'int_track_{tname}.png')}">
 <p class="ajuste">Las 4 configuraciones están dibujadas con grosor decreciente y se superponen casi
 exactamente (se ven como bandas anidadas) — esa coincidencia <b>es</b> el resultado: las 4 se
 comportan igual. La línea negra punteada es la referencia.</p>
"""

    F = Path("results/figures")
    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
<title>Informe integral — Identificación de Wilson-Cowan + control con Neural ODE</title>
<style>
 body{{font-family:-apple-system,"Segoe UI",Arial,sans-serif;color:#1c2733;line-height:1.6;margin:0;background:#f4f6f8;padding:2rem 1rem}}
 .wrap{{max-width:1040px;margin:0 auto}}
 header{{background:#1f4e79;color:#fff;border-radius:12px 12px 0 0;padding:1.5rem 1.9rem}}
 header h1{{margin:0;font-size:1.35rem}} header p{{margin:.3rem 0 0;opacity:.9;font-size:.95rem}}
 main{{background:#fff;border:1px solid #d7dee5;border-top:none;border-radius:0 0 12px 12px;padding:1.5rem 1.9rem 2rem}}
 h2{{color:#1f4e79;font-size:1.2rem;border-bottom:2px solid #e8f0f8;padding-bottom:.3rem;margin-top:2rem}}
 h3{{color:#2c5f8a;font-size:1.02rem;margin-top:1.2rem}}
 img{{width:100%;border:1px solid #d7dee5;border-radius:8px;margin:.5rem 0}}
 table{{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:.85rem}}
 th,td{{text-align:left;padding:.45rem .6rem;border-bottom:1px solid #d7dee5}}
 th{{background:#e8f0f8;color:#1f4e79}} td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
 td.good{{color:#0a7d33}} td.bad{{color:#b00020}}
 code{{background:#eef2f6;padding:.05rem .3rem;border-radius:4px}}
 .ajuste{{font-size:.9rem;color:#555}}
 .nota{{background:#e8f0f8;border-left:4px solid #1f4e79;padding:.7rem 1rem;border-radius:0 6px 6px 0;margin:1rem 0;font-size:.92rem}}
 .ok{{background:#e7f6ec;border-left-color:#0a7d33}}
 .grid{{display:grid;grid-template-columns:1fr 1fr;gap:.6rem;margin:.6rem 0}}
 .card{{border:1px solid #d7dee5;border-radius:8px;padding:.7rem .9rem;font-size:.88rem}}
 nav{{font-size:.9rem;background:#f0f4f8;border-radius:8px;padding:.6rem 1rem;margin:.5rem 0}}
 nav a{{color:#1f4e79;margin-right:.8rem}}
</style></head><body><div class="wrap">
<header><h1>Informe integral — Identificación de Wilson-Cowan + control con Neural ODE</h1>
<p>Librería de estímulos · identificación (PINN y Neural ODE) · lazo cerrado · robustez bajo ruido · ITBA 2026</p></header>
<main>
 <div class="nota">Informe visual de todo el trabajo. El documento de texto completo y detallado
 (con la bitácora de cada experimento, error y corrección) es <code>docs/informe_completo.md</code>,
 pensado para leerse a la par.</div>
 <nav><b>Secciones:</b>
  <a href="#modelo">1·Modelo y estímulos</a><a href="#ident">2·Identificación</a>
  <a href="#nodelo">3·Neural ODE</a><a href="#control">4·Control (matriz)</a>
  <a href="#ruido">5·Ruido</a></nav>

 <h2 id="modelo">1 · El modelo y la librería de estímulos</h2>
 <p>Wilson-Cowan: dos poblaciones (E excitatoria, I inhibitoria). Es el "sistema real" que genera
 los datos. Se inyecta un estímulo externo (P→E, Q→I).</p>
 {_img(F / 'simulacion_wilson_cowan.png', 'simulación WC')}
 <p>Se amplió la librería de estímulos (todos on/off, realizables con optogenética), reemplazando las
 senoides por entradas tipo pulso:</p>
 <table>
  <tr><th>Estímulo</th><th>Qué es</th><th>Para qué</th></tr>
  <tr><td>APRBS</td><td>Escalones de amplitud y duración aleatorias</td><td>Barre amplitud × frecuencia (el más rico)</td></tr>
  <tr><td>Theta-gamma</td><td>Ráfagas gamma bajo envolvente theta</td><td>El régimen propio del proyecto</td></tr>
  <tr><td>Onda cuadrada</td><td>Tren de pulsos (freq + duty)</td><td>DBS / optogenética</td></tr>
  <tr><td>PRBS</td><td>Secuencia binaria pseudo-aleatoria</td><td>Clásico de identificación</td></tr>
  <tr><td>Poisson</td><td>Pulsos en tiempos aleatorios</td><td>Naturalista (spikes)</td></tr>
 </table>

 <h2 id="ident">2 · Identificación paramétrica (en limpio)</h2>
 <p>Verdaderos: wEE=6.4 · wEI=4.8 · wIE=6.0 · wII=1.2. El peso difícil es <b>wII</b> (solo se
 identifica si I es grande y se mueve distinto que E → hace falta <b>Q grande</b>).</p>
 <table>
  <tr><th>Método</th><th class="num">wEE</th><th class="num">wEI</th><th class="num">wIE</th><th class="num">wII</th></tr>
  <tr><td>PINN conjunta</td><td class="num good">0.1%</td><td class="num good">0.3%</td><td class="num good">0.0%</td><td class="num good">0.1%</td></tr>
  <tr><td>PINN Q-grande</td><td class="num good">1.7%</td><td class="num good">4.5%</td><td class="num good">0.2%</td><td class="num good">1.7%</td></tr>
  <tr><td><b>Neural ODE</b></td><td class="num good">0.1%</td><td class="num good">0.4%</td><td class="num good">0.1%</td><td class="num good">0.4%</td></tr>
 </table>
 {_img(F / 'wii_pesos.png', 'convergencia de pesos')}

 <h2 id="nodelo">3 · El Neural ODE como planta (lazo abierto)</h2>
 <p>Un único Neural ODE <code>f(x,P,Q)</code> entrenado con todos los estímulos. Mismo estímulo
 aplicado al simulador y al Neural ODE: las trayectorias casi coinciden. Rollout MSE: train
 <b>3.4e-5</b>, test held-out <b>8.5e-5</b> (generaliza a estímulos no vistos).</p>
 {_img(F / 'informe_ol.png', 'lazo abierto simulador vs Neural ODE')}

 <h2 id="control">4 · Control en lazo cerrado — matriz por tipo de referencia</h2>
 <p>El controlador IMC lleva un modelo de la planta adentro (sus pesos), por eso importa con qué
 parámetros se lo construye. Se comparan <b>4 configuraciones</b> = 2 plantas × 2 juegos de parámetros:</p>
 <div class="grid">
  <div class="card"><b>Simulador + θ̂</b> — sistema REAL con controlador de parámetros IDENTIFICADOS
   (caso de uso real: ¿el error de identificación degrada el control?).</div>
  <div class="card"><b>Simulador + reales</b> — cota inferior ideal del RMSE.</div>
  <div class="card"><b>Neural ODE + θ̂</b> — lazo 100% aprendido.</div>
  <div class="card"><b>Neural ODE + reales</b> — aísla el efecto de la planta aprendida.</div>
 </div>
 <p>Pesos del controlador — reales: <code>{wr}</code> · identificados θ̂: <code>{wh}</code>.
 RMSE: menor = mejor seguimiento.</p>
 {secciones}
 <div class="nota ok">En los tres tipos de referencia, las 4 configuraciones dan RMSE casi idéntico:
 usar θ̂ en vez de los reales no degrada el control, y la planta aprendida se comporta como el
 simulador. El RMSE varía con la dificultad de la referencia, no con la configuración.</div>

 <h2 id="ruido">5 · Robustez bajo ruido (OE3)</h2>
 <p>Por cada nivel de ruido: se identifica con datos ruidosos (θ̂), se arma el controlador con θ̂ y
 se cierra el lazo sobre la planta verdadera. Método mejorado = estímulo fuerte + suavizado adaptativo.</p>
 <table>
  <tr><th class="num">σ (ruido)</th><th class="num">θ̂ máx — ingenuo</th><th class="num">θ̂ máx — mejorado</th><th class="num">Control RMSE I / E</th></tr>
  <tr><td class="num">0.00</td><td class="num">0.4%</td><td class="num good">0.1%</td><td class="num">3.35e-2 / 3.13e-2</td></tr>
  <tr><td class="num">0.01</td><td class="num">2.1%</td><td class="num good">0.6%</td><td class="num">3.35e-2 / 3.14e-2</td></tr>
  <tr><td class="num">0.05</td><td class="num bad">19.3%</td><td class="num good">1.2%</td><td class="num">3.37e-2 / 3.13e-2</td></tr>
  <tr><td class="num">0.10</td><td class="num bad">106.1%</td><td class="num good">8.9%</td><td class="num">3.42e-2 / 3.06e-2</td></tr>
 </table>
 {_img(F / 'ruido.png', 'robustez bajo ruido')}
 <div class="nota ok">Dos resultados: (1) el control se mantiene cerca del ideal (~3.3e-2) en todos
 los niveles, incluso con θ̂ muy errado (acción integral del IMC); (2) la identificación bajo ruido
 mejora drásticamente con suavizado + estímulo fuerte (a σ=0.10: de 106% a 8.9%).</div>
</main></div></body></html>"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
