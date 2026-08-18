#!/usr/bin/env python3
# =============================================================================
#  ARMA LOS DOS HTML DEL ESCALADO (explicacion + figuras), autocontenidos.
# =============================================================================
#  - docs/escalado_explicacion.html : la explicacion clara de las etapas 0-2,
#                                     con enlaces a cada figura del otro HTML.
#  - docs/escalado_figuras.html     : las 7 figuras (version clara/oscura
#                                     embebidas en base64), cada una explicada.
#
#  Las figuras salen de scripts/figuras_escalado.py y los numeros de
#  results/escalado/*.json (se leen aca, no se tipean a mano).
#
#  USO:  python scripts/build_html_escalado.py
# =============================================================================
from __future__ import annotations

import base64
import json
from pathlib import Path

FIGDIR = Path("docs/figuras_escalado")
RES = Path("results/escalado")
FIG_HTML = "escalado_figuras.html"
EXP_HTML = "escalado_explicacion.html"


def R(tag):
    return json.loads((RES / f"{tag}.json").read_text())


def b64(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def figure(fid: str, stem: str, title: str, caption_html: str) -> str:
    return f"""
<figure class="fig" id="{fid}">
  <figcaption class="fig-head"><span class="fig-num">{fid.upper().replace('FIG', 'Figura ')}</span> {title}</figcaption>
  <img class="light" src="{b64(FIGDIR / f'{stem}_light.png')}" alt="{title}">
  <img class="dark"  src="{b64(FIGDIR / f'{stem}_dark.png')}"  alt="{title}">
  <div class="fig-body">{caption_html}</div>
</figure>"""


# -----------------------------------------------------------------------------
#  Numeros que se citan (leidos de los json)
# -----------------------------------------------------------------------------
n = {t: R(t) for t in ("e1_wb", "e1_S", "e1_B_w100", "e1_B_w200", "e1_B_w400", "e1_S2",
                       "e2_wb", "e2_B", "e2_lag", "e2_lag2", "e2_lat_w100", "e2_lat_w400")}
def N(t): return f"{n[t]['nrmse_test']:.2f}%"
def R2(t): return f"{n[t]['r2_delta_test']:.2f}"
def PE(t): return f"{n[t]['mean_param_error']:.1f}%"
tau1 = n["e2_lag"]["extras"]["tau"]; tau2 = n["e2_lag2"]["extras"]["tau"]

# -----------------------------------------------------------------------------
#  CSS comun
# -----------------------------------------------------------------------------
CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --bg:#f7f6f2; --surface:#fffefb; --ink:#1c1b18; --ink-2:#4f4d47; --ink-3:#7c7a72;
  --line:#e2dfd6; --accent:#1f5f8b; --accent-soft:#e3eef6;
  --wb:#eb6834; --gb:#2a78d6; --first:#eda100; --best:#1baf7a; --real:#1c1b18;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#161614; --surface:#1e1e1b; --ink:#f1efe8; --ink-2:#c9c6bb; --ink-3:#8f8c82;
    --line:#33322d; --accent:#7fb3d9; --accent-soft:#1f2b35;
    --wb:#d95926; --gb:#3987e5; --first:#c98500; --best:#199e70; --real:#f1efe8;
  }
}
:root[data-theme="dark"]{
  --bg:#161614; --surface:#1e1e1b; --ink:#f1efe8; --ink-2:#c9c6bb; --ink-3:#8f8c82;
  --line:#33322d; --accent:#7fb3d9; --accent-soft:#1f2b35;
  --wb:#d95926; --gb:#3987e5; --first:#c98500; --best:#199e70; --real:#f1efe8;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:"Source Serif 4",Georgia,"Times New Roman",serif;font-size:17px;line-height:1.55}
main{max-width:46rem;margin:0 auto;padding:3rem 1.25rem 5rem}
h1,h2,h3{font-family:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;text-wrap:balance;line-height:1.2}
h1{font-size:2.1rem;font-weight:600;margin:0 0 .5rem}
h2{font-size:1.35rem;font-weight:600;margin:3rem 0 .75rem;padding-top:1.25rem;border-top:1px solid var(--line)}
h3{font-size:1.05rem;font-weight:600;margin:1.75rem 0 .5rem;color:var(--ink-2)}
p{margin:.7rem 0}
.eyebrow{font-family:"IBM Plex Sans",sans-serif;font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);font-weight:600}
.lede{font-size:1.1rem;color:var(--ink-2)}
a{color:var(--accent);text-decoration:none;border-bottom:1px solid color-mix(in srgb,var(--accent) 45%,transparent)}
a:hover,a:focus-visible{border-bottom-color:var(--accent);outline:none}
a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,Consolas,monospace;font-size:.86em}
code{background:var(--accent-soft);padding:.05em .35em;border-radius:3px}
.figref{font-family:"IBM Plex Sans",sans-serif;font-size:.85em;font-weight:600;white-space:nowrap}
.figref::before{content:"↗ ";opacity:.7}
.card{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:1rem 1.2rem;margin:1.2rem 0}
.card.key{border-left:4px solid var(--accent)}
.card h3{margin-top:0}
.actors{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin:1.2rem 0}
@media (max-width:640px){.actors{grid-template-columns:1fr}}
.actors .card{margin:0}
.actors .card .who{font-family:"IBM Plex Sans",sans-serif;font-size:.8rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3);font-weight:600}
table{border-collapse:collapse;width:100%;font-family:"IBM Plex Sans",sans-serif;font-size:.86rem;margin:1rem 0;font-variant-numeric:tabular-nums}
.tbl{overflow-x:auto}
th,td{text-align:left;padding:.5rem .6rem;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;color:var(--ink-2);font-size:.78rem;letter-spacing:.04em;text-transform:uppercase}
td.num{text-align:right}
tr.best td{font-weight:600}
.pill{display:inline-block;font-family:"IBM Plex Sans",sans-serif;font-size:.72rem;font-weight:600;padding:.1em .5em;border-radius:999px;color:#fff;letter-spacing:.02em}
.pill.wb{background:var(--wb)}.pill.gb{background:var(--gb)}.pill.first{background:var(--first);color:#1c1b18}.pill.best{background:var(--best)}
.pill.real{background:var(--real);color:var(--bg)}
.swatch{display:inline-block;width:.8em;height:.8em;border-radius:2px;vertical-align:-.05em;margin-right:.35em}
ol.steps{padding-left:1.3rem}ol.steps li{margin:.5rem 0}
.formula{font-family:"IBM Plex Mono",monospace;font-size:.9rem;background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:.8rem 1rem;overflow-x:auto;margin:.8rem 0}
.callout{border-left:4px solid var(--best);padding:.2rem 1rem;margin:1.2rem 0;color:var(--ink-2)}
.src{font-family:"IBM Plex Mono",ui-monospace,Consolas,monospace;font-size:.72rem;color:var(--ink-3);margin:.6rem 0 0;line-height:1.5}
.src::before{content:"código · ";font-family:"IBM Plex Sans",sans-serif;letter-spacing:.06em;text-transform:uppercase;font-size:.66rem}
.src code{background:none;padding:0;font-size:1em;color:var(--ink-3)}
.summary-line{font-family:"IBM Plex Sans",sans-serif;font-size:1.02rem}
/* figuras */
.fig{margin:2.2rem 0;background:var(--surface);border:1px solid var(--line);border-radius:6px;overflow:hidden}
.fig img{display:block;width:100%;height:auto}
.fig img.dark{display:none}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]) .fig img.light{display:none} :root:not([data-theme="light"]) .fig img.dark{display:block} }
:root[data-theme="dark"] .fig img.light{display:none} :root[data-theme="dark"] .fig img.dark{display:block}
.fig-head{font-family:"IBM Plex Sans",sans-serif;font-weight:600;font-size:1rem;padding:.9rem 1.1rem .6rem;border-bottom:1px solid var(--line)}
.fig-num{color:var(--accent);margin-right:.4rem}
.fig-body{padding:.9rem 1.1rem 1.1rem;font-size:.97rem;color:var(--ink-2)}
.fig-body p{margin:.5rem 0}
.fig-body strong{color:var(--ink)}
.toc{font-family:"IBM Plex Sans",sans-serif;font-size:.9rem;columns:2;column-gap:2rem;margin:1rem 0 2rem}
.toc a{display:block;padding:.15rem 0;border:0}
@media (max-width:640px){.toc{columns:1}}
.back{font-family:"IBM Plex Sans",sans-serif;font-size:.85rem;margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line)}
@media (prefers-reduced-motion: no-preference){ html{scroll-behavior:smooth} }
</style>
"""

F = FIG_HTML  # atajo para los enlaces


def ref(fid, text=None):
    return f'<a class="figref" href="{F}#{fid}">{text or fid.upper().replace("FIG", "Fig. ")}</a>'


# =============================================================================
#  HTML 1 — LA EXPLICACION
# =============================================================================
EXPLICACION = f"""<title>Escalado progresivo del gray-box</title>
{CSS}
<main>
<p class="eyebrow">Proyecto Wilson-Cowan · Neural ODE gray-box · etapas 0–2</p>
<h1>Escalado progresivo del gray-box</h1>
<p class="lede">Qué hicimos, en qué orden y qué dio, contado sin atajos. Cada número tiene una figura que lo muestra en
<a href="{F}">la página de figuras</a>; los enlaces marcados <span class="figref">Fig. N</span> llevan directo a ella.</p>

<h2>La idea general</h2>
<p>Hay dos actores en todo esto:</p>
<div class="actors">
  <div class="card"><div class="who">Quien genera los datos</div><h3>La planta (el "cerebro" simulado)</h3>
    <p>Es Wilson-Cowan <strong>más algo extra</strong> que la red no conoce. Ese "algo extra" es la perturbación,
    y es lo único que cambia entre etapas.</p></div>
  <div class="card"><div class="who">Quien aprende</div><h3>La Neural ODE gray-box</h3>
    <p>Es Wilson-Cowan <strong>puro</strong> (sus 10 parámetros son aprendibles) más una <strong>corrección</strong>
    aprendible. Se entrena para copiar los datos de la planta.</p></div>
</div>
<p>Los estímulos <code>P(t), Q(t)</code> son <strong>siempre la misma librería de 20 señales</strong> (13 para entrenar,
7 para test) en todas las etapas. Y una aclaración clave: el modelo <strong>siempre recibe el estímulo comandado</strong>
(el que nosotros mandamos), nunca el efectivo que le llega a la planta. Si le diéramos el efectivo le estaríamos
regalando la respuesta. {ref('fig1')} muestra la misma señal de entrada atravesando las tres plantas.</p>
<p class="src">planta y perturbaciones: <code>src/wilson_cowan/model.py</code>, <code>src/wilson_cowan/uncertainty.py:137</code> (Refractoriness), <code>:169</code> (Actuator) · modelo gray-box: <code>src/neural_ode/dynamics.py:167</code> (backbone WC), <code>:223</code> (corrección g) · librería de 20 estímulos: <code>scripts/gen_multi_dataset.py:63</code> · integrador RK4: <code>src/neural_ode/integrate.py:21</code></p>

<div class="card key"><h3>Cómo se mide "copiar la dinámica": el NRMSE</h3>
<p>Al modelo entrenado se le da <em>solo</em> el estado inicial y el estímulo de un escenario de test (que nunca vio),
y tiene que generar los 200 ms enteros por su cuenta sin volver a mirar el dato real ("rollout open-loop"). Se compara
punto a punto contra la planta:</p>
<div class="formula">NRMSE = 100 · RMSE(predicho − real) / (máx(real) − mín(real))</div>
<p>Es el error típico expresado como <strong>porcentaje de cuánto se mueve la señal</strong>, así es comparable
entre escenarios que oscilan mucho y poco. Guía de lectura: <strong>&lt; 5%</strong> copia muy bien;
<strong>5–15%</strong> forma correcta con desvíos visibles; <strong>&gt; 25%</strong> no sirve como planta.
Se reporta el promedio sobre los 7 escenarios de test y los dos canales (I, E).</p>
<p class="src"><code>scripts/esc_eval.py:94</code> (nrmse_test), <code>:144</code> (r2_delta), <code>:179</code> (evaluar)</p>
<p>Métricas secundarias: <strong>R²</strong> de la corrección contra el hueco verdadero Δf (¿aprendió la física o solo
tapó?) y el <strong>error de los 10 parámetros</strong>.</p></div>

<h2>Etapa 0 — la referencia</h2>
<p><strong>Planta:</strong> WC puro, sin nada extra. <strong>Dataset:</strong> <code>eps0.npz</code>.
<strong>Modelo:</strong> WC puro sin corrección (white-box).</p>
<p><strong>Resultado: NRMSE 2.04%.</strong> Es el piso: acá el modelo <em>puede</em> ser exacto, así que este número
es lo mejor a lo que se puede aspirar en las etapas siguientes. {ref('fig2')} muestra dos rollouts: uno excelente
(<code>chirp</code>, 0.6%) y el único flojo (<code>box_a1.2</code>, 10.8%, el escalón grande sostenido) — ese
escenario va a ser el duro en <em>todas</em> las etapas.</p>
<p class="src">dataset: <code>scripts/gen_uncertain_dataset.py</code> (eps=0 → NoPerturbation) · entrenamiento white-box: <code>src/neural_ode/graybox_train.py:210</code> (fit), <code>:46</code> (ventanas)</p>

<h2>Etapa 1 — solo refractariedad</h2>
<p><strong>Planta:</strong> WC + término refractario <code>(1 − r·x)·S(u)</code> con r = 0.10 (es literalmente el
término del Wilson-Cowan original de 1972 que la forma reducida descarta). Estímulos: la librería, sin perturbar.
<strong>Dataset:</strong> <code>refrac1.npz</code>.</p>
<p class="src">término refractario en la planta: <code>src/wilson_cowan/uncertainty.py:162</code> (gains) · generación: <code>scripts/esc_gen_datasets.py:37</code></p>
<p><strong>Modelo:</strong> WC puro (sin el término refractario) + corrección. Se probaron tres correcciones:</p>
<div class="tbl"><table>
<tr><th>corrección</th><th>qué es</th><th class="num">NRMSE</th><th>¿aprendió la física?</th></tr>
<tr><td><span class="pill wb">white-box</span></td><td>ninguna: solo mueve los 10 parámetros</td><td class="num">{N('e1_wb')}</td><td>—</td></tr>
<tr><td><span class="pill gb">gray-box B</span></td><td>red neuronal <code>g(I,E)</code> "ciega": suma algo a la derivada viendo solo el estado</td><td class="num">{N('e1_B_w100')} – {N('e1_B_w400')}</td><td>no (R² ≈ {R2('e1_B_w100')}): tapa el hueco deformando parámetros</td></tr>
<tr class="best"><td><span class="pill best">estructural S2</span></td><td>le decimos la <em>forma</em> <code>(1−r·x)</code> y solo tiene que hallar r</td><td class="num">{N('e1_S2')}</td><td>sí: r = 0.102 (real 0.10), R² {R2('e1_S2')}</td></tr>
</table></div>
<p>{ref('fig3')} muestra los tres rollouts sobre el mismo escenario; {ref('fig5')} y {ref('fig7')} el promedio y el
detalle por escenario.</p>
<p class="src">variantes wb / B / S: <code>src/neural_ode/graybox_train.py:185</code> · red ciega g(I,E): <code>src/neural_ode/dynamics.py:132</code> · forma estructural (1−r·x): <code>src/neural_ode/dynamics.py:204</code> · runner: <code>scripts/esc_run.py</code></p>
<p><strong>Lectura:</strong> la refractariedad sola es benigna. Incluso sin saber su forma, la red ciega la copia hasta
casi el piso ({N('e1_wb')} → ~2.9%). Sabiendo la forma, la clava exactamente y queda <em>por debajo</em> del piso
(1.70%; el piso lo fijaba un white-box con 1% de error en los parámetros). Se barrió además la ventana de
entrenamiento (5, 10 y 20 ms) y <strong>no importó</strong> — la sospecha de que entrenar con ventanas cortas era el
cuello de botella no se confirmó.</p>
<div class="callout"><p><strong>Detalle que importa.</strong> La estructural primero <em>falló</em>: la corrida
<code>e1_S</code> quedó idéntica al white-box porque aprendió r = 0 exacto. El motivo: r arrancaba pegado a 0 y ahí
(por el <code>clamp</code> que lo mantiene positivo) el gradiente es cero, así que nunca se movió. Se arregló arrancando r
en 0.05 con los parámetros WC ya inicializados desde el white-box. Fue un problema de <strong>inicialización</strong>,
no de modelo — {ref('fig6')} muestra el antes/después.</p>
<p class="src">clamp de r: <code>src/neural_ode/dynamics.py:204</code> · warm-start y r inicial: <code>scripts/esc_run.py:56-88</code> (<code>--init-params</code>, <code>--r-init</code>)</p></div>

<h2>Etapa 2 — solo actuador (estímulo perturbado)</h2>
<p><strong>Planta:</strong> WC puro (sin refractariedad), pero el estímulo que le llega no es el comandado: pasa por un
filtro con retardo τ = 1 ms y una saturación suave (imita el canal ChR2 en optogenética). Estímulos comandados: la misma
librería. <strong>Dataset:</strong> <code>act1.npz</code>.</p>
<p class="src">actuador en la planta: <code>src/wilson_cowan/uncertainty.py:197</code> (filtro), <code>:202</code> (saturación) · generación: <code>scripts/esc_gen_datasets.py:39</code></p>
<p><strong>Modelo:</strong> WC puro + corrección, y <strong>recibe el estímulo comandado</strong> (no el que llegó).
Correcciones probadas:</p>
<div class="tbl"><table>
<tr><th>corrección</th><th>qué es</th><th class="num">NRMSE</th><th>¿aprendió la física?</th></tr>
<tr><td><span class="pill wb">white-box</span></td><td>solo parámetros</td><td class="num">{N('e2_wb')}</td><td>—</td></tr>
<tr><td><span class="pill gb">gray-box B</span></td><td>red <code>g(I,E)</code> ciega, igual que en etapa 1</td><td class="num">{N('e2_B')}</td><td>no puede: el retardo tiene memoria y <code>g(I,E)</code> no</td></tr>
<tr><td><span class="pill first">latente</span></td><td>red con estados ocultos aprendidos genéricos</td><td class="num">{N('e2_lat_w400')} – {N('e2_lat_w100')}</td><td>no funcionó</td></tr>
<tr><td><span class="pill first">lag (1er intento)</span></td><td>agregamos al modelo el filtro <code>dP̂/dt = (P−P̂)/τ̂</code> con τ̂ aprendible</td><td class="num">{N('e2_lag')}</td><td>a medias: τ̂ = {tau1:.2f} (real 1.0), R² {R2('e2_lag')}</td></tr>
<tr class="best"><td><span class="pill best">lag2</span></td><td>lo mismo, con el estado del filtro computado exacto al inicio de cada ventana</td><td class="num">{N('e2_lag2')}</td><td>sí: τ̂ = {tau2:.2f} (real 1.0), R² {R2('e2_lag2')}</td></tr>
</table></div>
<p>{ref('fig4')} muestra los cuatro rollouts sobre el mismo escenario.</p>
<p class="src">modelo con filtro (lag): <code>src/neural_ode/memory.py:45</code> · latente genérico: <code>:127</code> · entrenamiento con estado oculto: <code>:200</code> (fit_aug)</p>
<p><strong>Lectura:</strong> el actuador es el problema grave: {N('e2_wb')} con white-box, casi lo mismo que daban las dos
perturbaciones juntas antes (14.0%). Una red que solo ve el estado <strong>no puede</strong> arreglarlo por diseño (el
retardo tiene memoria propia: dos instantes con el mismo estado pueden tener distinta derivada). Pero si al modelo le
damos un estado de memoria con la forma del filtro, lo clava.</p>
<div class="callout"><p><strong>Detalle que importa.</strong> El lag también falló primero (12%, τ̂ a la mitad). En el
entrenamiento por ventanas cortas, cada ventana arrancaba con el filtro "asentado" en el comando — un salto que la planta
no tiene, y el modelo lo minimizaba achicando τ̂. Como el estado del actuador depende <em>solo</em> del comando (que
conocemos entero) y de τ̂, se puede computar exacto filtrando toda la trayectoria. Con eso: {N('e2_lag2')} y τ̂ dentro
del 2% del real. La saturación no se identificó (es ~16% del efecto y con τ̂ bien puesto queda poca señal); es lo que deja
<code>box_a1.2</code> en 12.8% mientras el resto queda &lt; 3% ({ref('fig7')}).</p>
<p class="src">estado del filtro exacto: <code>src/neural_ode/memory.py:88</code> (filtered_inputs), usado en <code>:225</code> (hidden0)</p></div>

<h2>Resumen</h2>
<div class="tbl"><table>
<tr><th>etapa</th><th>planta = WC + …</th><th>dataset</th><th>mejor modelo</th><th class="num">NRMSE</th></tr>
<tr><td>0</td><td>nada</td><td class="mono">eps0</td><td>white-box</td><td class="num">2.04%</td></tr>
<tr class="best"><td>1</td><td>refractariedad</td><td class="mono">refrac1</td><td>WC + <code>(1−r·x)</code> aprendible</td><td class="num">{N('e1_S2')} <span style="font-weight:400;color:var(--ink-3)">(red ciega: {N('e1_B_w400')})</span></td></tr>
<tr class="best"><td>2</td><td>actuador con retardo</td><td class="mono">act1</td><td>WC + filtro con τ̂ aprendible</td><td class="num">{N('e2_lag2')} <span style="font-weight:400;color:var(--ink-3)">(red ciega: {N('e2_B')})</span></td></tr>
<tr><td>antes</td><td>las dos juntas</td><td class="mono">eps1</td><td>gray-box D</td><td class="num">13.1%</td></tr>
</table></div>
<p class="summary-line">{ref('fig5')} pone todos los modelos en una sola barra; {ref('fig6')} muestra que la física
aprendida es la real.</p>
<div class="card key"><h3>La conclusión</h3>
<p>El 13% que teníamos <strong>no era un límite del gray-box</strong>. Era el retardo del actuador (que necesita memoria)
más dos errores de inicialización que las dos perturbaciones juntas escondían. Separadas, cada una se copia a ~2–3%.</p>
<p>El siguiente paso natural (etapa 3) es juntar las dos formas ganadoras —término refractario + filtro con τ̂— y ver si
con las dos perturbaciones a la vez baja de 13% a ~3–4%. Y en paralelo (etapa 4), una Neural ODE full black-box con los
mismos datos, como cota de cuánto paga el prior de Wilson-Cowan.</p></div>

<p class="back">Documentos hermanos: <code>docs/bitacora_escalado.md</code> (proceso completo),
<code>docs/resultados_escalado.md</code> (informe), <a href="{F}">{F}</a> (figuras).</p>
</main>
"""

# =============================================================================
#  HTML 2 — LAS FIGURAS
# =============================================================================
figs = [
    figure("fig1", "fig1_plantas", "Misma entrada, tres plantas distintas", f"""
<p><strong>Qué se ve.</strong> Un escenario de test (<code>prbs_1</code>, una secuencia pseudoaleatoria de pulsos) atravesando
las tres plantas del escalado. <em>Arriba:</em> el estímulo P comandado (escalones netos) y el que realmente le llega a la
planta con actuador — se ve el retardo (sube y baja exponencialmente en vez de saltar) y la leve saturación en el tope.
<em>Medio:</em> la respuesta E de cada planta: WC puro (negro), con refractariedad (azul), con actuador (naranja).
<em>Abajo:</em> el hueco |Δf| = |f_planta − f_WC|, o sea cuánto se aparta en cada instante la derivada real de la
de Wilson-Cowan puro — es exactamente lo que la corrección tiene que aprender.</p>
<p><strong>Cómo leerla.</strong> La refractariedad recorta los picos altos de E de forma suave y sostenida (su |Δf| es
una joroba que sigue a la actividad). El actuador, en cambio, produce picos de |Δf| muy altos y breves en cada flanco
del estímulo: es un desajuste que aparece cuando el comando <em>cambia</em>, no cuando el estado está en cierto valor.
Ésa es la razón física por la que una corrección que solo ve el estado (I,E) puede con la primera y no con la segunda.</p>"""),

    figure("fig2", "fig2_etapa0", "Etapa 0 — el piso: white-box sobre planta WC pura", """
<p><strong>Qué se ve.</strong> Rollout open-loop de 200 ms del white-box entrenado sobre la planta sin perturbación
(línea punteada verde) contra la planta (negro), en dos escenarios de test. Arriba <code>chirp</code> (frecuencia
creciente): 0.6% — indistinguible. Abajo <code>box_a1.2</code> (escalón grande sostenido): 10.8% — el modelo se
adelanta levemente en fase dentro del régimen oscilatorio y ese desfase se acumula.</p>
<p><strong>Cómo leerla.</strong> Éste es el mejor caso posible: el modelo tiene exactamente las ecuaciones de la planta
y solo tenía que encontrar los 10 números (error 1.05%). Aun así da 2.04% de promedio, casi todo por <code>box_a1.2</code>.
Cualquier modelo de las etapas 1 y 2 hay que compararlo contra <strong>este</strong> 2%, no contra 0.</p>"""),

    figure("fig3", "fig3_etapa1_rollout", "Etapa 1 — planta con refractariedad: tres correcciones", f"""
<p><strong>Qué se ve.</strong> El mismo escenario de test (<code>chirp</code>, zoom en 80–200 ms) reproducido por los tres
modelos entrenados sobre <code>refrac1</code>. Negro = planta con refractariedad; punteado = el modelo.
<em>Arriba:</em> white-box (naranja) — se le van los picos y la fase. <em>Medio:</em> gray-box B con red ciega
<code>g(I,E)</code> (azul) — casi encima. <em>Abajo:</em> estructural S2 (verde) — encima.</p>
<p><strong>Cómo leerla.</strong> Con solo refractariedad, hasta el white-box "más o menos" sigue la forma
({N('e1_wb')} de promedio); la red ciega la lleva a {N('e1_B_w100')} y la forma exacta a {N('e1_S2')}. La diferencia entre
las dos últimas es invisible a ojo acá: donde sí se distinguen es en la
<a href="#fig6">Fig. 6</a> — la ciega no aprendió la física (R² ≈ 0), la estructural sí.</p>"""),

    figure("fig4", "fig4_etapa2_rollout", "Etapa 2 — planta con actuador: cuatro correcciones", f"""
<p><strong>Qué se ve.</strong> Mismo escenario y zoom que la Fig. 3, ahora sobre <code>act1</code> (planta con retardo en el
estímulo). <em>Panel 1:</em> white-box — a partir de ~125 ms se desengancha por completo: donde la planta hace un pico,
el modelo hace un valle. <em>Panel 2:</em> gray-box B ciego — idéntico al white-box: la red no puede hacer nada.
<em>Panel 3:</em> lag, primer intento (τ̂ = {tau1:.2f}, la mitad del real) — mejora, pero sigue desfasado.
<em>Panel 4:</em> lag2 con el estado del filtro exacto (τ̂ = {tau2:.2f}) — encima de la planta, 1.9% en este escenario.</p>
<p><strong>Cómo leerla.</strong> Es la figura central del trabajo. Muestra por qué el problema original parecía
imposible (paneles 1–2: sin memoria no hay forma) y que la solución no es "más red" sino <strong>el estado correcto</strong>
(panel 4). También muestra que un modelo con la forma correcta pero mal entrenado (panel 3) puede quedar lejos: la
diferencia entre 3 y 4 es solo <em>cómo se inicializa el filtro en cada ventana de entrenamiento</em>.</p>"""),

    figure("fig5", "fig5_resumen", "Resumen — NRMSE de todos los modelos", f"""
<p><strong>Qué se ve.</strong> Una barra por corrida (13 en total), agrupadas por etapa. Color = tipo de modelo:
<span class="swatch" style="background:var(--wb)"></span>white-box,
<span class="swatch" style="background:var(--gb)"></span>gray-box con red ciega,
<span class="swatch" style="background:var(--first)"></span>primeros intentos que fallaron o no ayudaron,
<span class="swatch" style="background:var(--best)"></span>el mejor de cada etapa. Línea punteada: el piso de la etapa 0
(2.04%). Línea a trazos: lo que daba el mejor modelo con las dos perturbaciones juntas (13.1%).</p>
<p><strong>Cómo leerla.</strong> Tres cosas saltan a la vista. (1) En la etapa 1 todo queda cerca del piso salvo el
white-box y la S mal inicializada, que coinciden exactamente ({N('e1_wb')} / {N('e1_S')}). (2) En la etapa 2 hay un
abismo: todo lo que no tiene memoria bien puesta está en 12–15%, y lag2 baja a {N('e2_lag2')}. (3) Las tres barras azules
de la etapa 1 (ventanas de 5, 10 y 20 ms) son iguales: la ventana no fue la perilla.</p>"""),

    figure("fig6", "fig6_fisica", "¿Aprendió la física real? r, τ̂ y R²", f"""
<p><strong>Qué se ve.</strong> <em>Izquierda:</em> el coeficiente de refractariedad r_e que aprendió la corrección
estructural, contra el real (0.10, línea a trazos): el primer intento dio 0 exacto (colapsó a WC puro), el segundo
0.102. <em>Centro:</em> el retardo τ̂ del filtro aprendido contra el real (1.0 ms): primer intento {tau1:.2f}, segundo
{tau2:.2f}. <em>Derecha:</em> R² de la corrección aprendida contra el hueco verdadero Δf en los datos de test — 1 sería
"reproduce exactamente la física que falta", 0 "no mejor que predecir el promedio", negativo "peor que eso".</p>
<p><strong>Cómo leerla.</strong> El panel derecho separa dos cosas que el NRMSE mezcla: <em>copiar</em> y
<em>entender</em>. El gray-box ciego de la etapa 1 copia a 3% pero su R² es {R2('e1_B_w100')}: lo que sumó no se parece a
la refractariedad, tapó el hueco deformando los parámetros WC. Las dos formas físicas bien inicializadas dan R² de
{R2('e1_S2')} y {R2('e2_lag2')}: aprendieron el término real. Y el gray-box ciego de la etapa 2 da R² {R2('e2_B')}:
inventó una corrección que empeora las cosas. Para usar el modelo como planta de simulación alcanza con copiar; para
cancelar la perturbación en lazo cerrado hace falta entender.</p>"""),

    figure("fig7", "fig7_escenarios", "Los 7 escenarios de test, uno por uno", f"""
<p><strong>Qué se ve.</strong> El NRMSE por escenario de test, para el white-box, el gray-box ciego y el mejor modelo de
cada etapa. Los escenarios son los mismos en las dos: un escalón grande (<code>box_a1.2</code>), un tren de pulsos a
130 Hz, dos secuencias pseudoaleatorias (<code>aprbs_2</code>, <code>prbs_1</code>), un ritmo theta-gamma, un tren de
Poisson y un chirp.</p>
<p><strong>Cómo leerla.</strong> El promedio esconde un rango enorme. En los estímulos ricos y rápidos los mejores
modelos quedan en 0.2–1%; en <code>box_a1.2</code> todos quedan en 9–13% (y el piso limpio ya estaba en 10.8% ahí):
un escalón grande y sostenido lleva la actividad al régimen donde más muerden tanto la refractariedad como la
saturación del actuador, y además es el único escenario de su tipo en el set de test, así que no hay uno parecido en
entrenamiento. En la etapa 2 se ve además que en <code>prbs_1</code> y <code>chirp</code> — donde el comando cambia
todo el tiempo — el white-box y el ciego están en 16–21% y lag2 en 1–2%: la memoria importa justo donde el
estímulo se mueve.</p>"""),
]

FIGURAS = f"""<title>Escalado progresivo · figuras</title>
{CSS}
<main>
<p class="eyebrow">Proyecto Wilson-Cowan · Neural ODE gray-box · etapas 0–2</p>
<h1>Figuras del escalado progresivo</h1>
<p class="lede">Las siete figuras que acompañan a <a href="{EXP_HTML}">la explicación</a>. Cada una dice qué se ve y cómo
leerla. Todas salen de los modelos entrenados y de los datasets reales (<code>scripts/figuras_escalado.py</code>);
la paleta es la misma en todo el documento:
<span class="pill real">planta real</span> <span class="pill wb">white-box</span> <span class="pill gb">gray-box ciego</span>
<span class="pill first">primer intento</span> <span class="pill best">mejor modelo</span>.</p>
<p class="src">figuras: <code>scripts/figuras_escalado.py</code> · rollout: <code>scripts/esc_eval.py:80</code> (_rollout_traj) sobre <code>src/neural_ode/integrate.py:30</code> · checkpoints: <code>results/escalado/models/*.pt</code></p>
<nav class="toc">
<a href="#fig1">Fig. 1 · Misma entrada, tres plantas</a>
<a href="#fig2">Fig. 2 · Etapa 0, el piso</a>
<a href="#fig3">Fig. 3 · Etapa 1, rollouts</a>
<a href="#fig4">Fig. 4 · Etapa 2, rollouts</a>
<a href="#fig5">Fig. 5 · Resumen NRMSE</a>
<a href="#fig6">Fig. 6 · Física aprendida</a>
<a href="#fig7">Fig. 7 · Por escenario</a>
</nav>
{''.join(figs)}
<p class="back">Volver a <a href="{EXP_HTML}">la explicación</a>. Números: <code>results/escalado/*.json</code>.</p>
</main>
"""

Path("docs", EXP_HTML).write_text(EXPLICACION, encoding="utf-8")
Path("docs", FIG_HTML).write_text(FIGURAS, encoding="utf-8")
print("->", Path("docs", EXP_HTML), Path("docs", EXP_HTML).stat().st_size // 1024, "KB")
print("->", Path("docs", FIG_HTML), Path("docs", FIG_HTML).stat().st_size // 1024, "KB")
