#!/usr/bin/env python3
# =============================================================================
#  FIGURAS DE LA PREGUNTA CENTRAL: ¿EL GRAY-BOX COPIA AL SIMULADOR?
# =============================================================================
#
#  QUE ES EL ROLLOUT OPEN LOOP (la prueba que se dibuja aca)
#  ---------------------------------------------------------
#  Se le da al modelo UNICAMENTE dos cosas: el estado inicial [I,E] del primer
#  instante y la senal de estimulo P,Q completa. Con eso tiene que generar los
#  200 ms enteros integrando su propia derivada, sin volver a mirar nunca el dato
#  real. Ninguna correccion en el camino.
#
#  POR QUE ES EXIGENTE. La alternativa comoda seria medir el error de la DERIVADA
#  punto a punto (o reiniciar el modelo desde el dato real cada ventana, que es lo
#  que hace el entrenamiento por multiple shooting). Ahi cada punto se juzga por
#  separado y un sesgo chico pasa desapercibido. En open loop no: el error del paso
#  k entra como condicion inicial del paso k+1, asi que un sesgo minimo en la
#  derivada se INTEGRA y se convierte en desfase de fase y de amplitud. Es la unica
#  prueba que responde la pregunta que importa para el control: si usaramos este
#  modelo como planta para disenar un controlador, ¿se comportaria como el cerebro?
#
#  Y encima se mide sobre estimulos de TEST, que el modelo no vio al entrenar
#  (el primero de ellos es 'box_a1.2'; ver escenario_test()).
#
#  Se comparan tres cosas contra el cerebro:
#     - el modelo COMPLETO (sin hueco)  -> control positivo
#     - el white-box con hueco          -> solo ecuaciones
#     - el gray-box con hueco           -> ecuaciones + red
#
#  "HUECO" = la perilla epsilon. El SIMULADOR tiene fisica que el MODELO no
#  contempla (refractariedad del Wilson-Cowan de 1972 + un actuador optogenetico
#  con retardo y saturacion). eps=0 -> no falta nada (las dos partes son las
#  mismas ecuaciones); eps=1 -> falta la fisica completa. Ver
#  docs/graybox_manual_completo.md.
#
#  POR QUE EL ERROR VA NORMALIZADO AL RANGO DE LA SENAL (NRMSE %). Un MSE crudo de
#  1e-2 no dice absolutamente nada: depende de las unidades y de cuan grande sea la
#  actividad en ese escenario. Dividiendo por (max - min) de la senal real el numero
#  queda adimensional y comparable entre escenarios y entre niveles de eps: "me
#  equivoco un 15% de lo que la senal se mueve". Referencia usada en todo el
#  proyecto:  <5% copia muy bien | 5-15% sigue la forma con desvios | >25% no sirve
#  como planta.
#
#  OJO: este script NO recalcula los errores que muestra en los titulos y en las
#  barras. Los lee de results/uncertainty/reproduccion.json (lo produce
#  scripts/exp_reproduccion.py) para que las figuras y la tabla del informe digan
#  siempre el mismo numero. Lo que si recalcula son las TRAYECTORIAS que se dibujan.
#
#  USO:  python scripts/figuras_copia.py
#        (requiere que ya existan los .npz de datos, los modelos entrenados y
#         reproduccion.json)
# =============================================================================

from __future__ import annotations

import json
import sys
from pathlib import Path

# El script se corre desde la raiz del repo (python scripts/figuras_copia.py), asi
# que la carpeta del script NO alcanza para importar "src". Se agrega la raiz al
# path a mano; por eso los imports de src van despues de esta linea y no arriba.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import matplotlib
# Backend "Agg" = sin ventana grafica. Imprescindible para que corra por SSH o en
# CI, donde no hay display. Tiene que elegirse ANTES de importar pyplot.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Los rollouts son 4000 pasos RK4 de una red chica: mas hilos no acelera nada y
# ademas hace que el resultado dependa de la maquina. 4 alcanza y es reproducible.
torch.set_num_threads(4)

from src.neural_ode import GrayBoxWC, rollout
from src.neural_ode.graybox_train import ALL_P

RES = Path("results/uncertainty")        # de donde salen los jsons de resultados
OUT = Path("docs/figuras_presentacion")  # donde se escriben los PNG

# --- Paleta. Un color = un significado, y el significado es el MISMO en todo el
#     paquete de figuras (figuras_presentacion.py, figuras_resumen.py):
#       azul  = el simulador, "el cerebro" (la verdad contra la que se compara)
#       naranja = el gray-box (ecuaciones + red)
#       gris  = el white-box (solo ecuaciones)  y tambien ejes y texto secundario
#       verde = el modelo completo / control positivo
#       rojo  = la anotacion que marca el resultado incomodo
#       suave = grillas y bordes (tienen que quedar detras de los datos)
#     Los hex de verde, gris, rojo y tinta son un poco mas oscuros que en los otros
#     scripts porque estas figuras se proyectan. Cambiar el ROL de un color aca y no
#     en las demas figuras es lo que confunde al lector; el tono exacto no.
AZUL = "#2a78d6"
NARA = "#eb6834"
VERDE = "#0d8a4f"
ROJO = "#d63b3a"
GRIS = "#5b6070"
SUAVE = "#dcdcd8"
TINTA = "#12141a"

# --- Estilo global. Estas figuras se proyectan y se ven de lejos: tipografia
#     grande, sin los bordes de arriba y derecha, grilla suave y fondo blanco
#     explicito (si se deja transparente, sobre un slide oscuro no se lee nada).
plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 14,
    "legend.fontsize": 13.5, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.edgecolor": SUAVE, "axes.labelcolor": GRIS, "axes.titlecolor": TINTA,
    "text.color": TINTA, "xtick.color": GRIS, "ytick.color": GRIS,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": SUAVE, "grid.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "lines.linewidth": 2.6,
})


# Guarda la figura como PNG en OUT y la cierra. bbox_inches="tight" recorta el
# margen sobrante (importante al pegarla en un slide) y facecolor="white" fuerza el
# fondo opaco. plt.close libera la figura: sin eso, generar varias seguidas deja
# todas abiertas y matplotlib avisa por consumo de memoria.
def guarda(fig, nombre):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{nombre}.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  {nombre}.png")


# Atajo: lee un json de resultados de RES por nombre de archivo.
def jd(n):
    return json.loads((RES / n).read_text())


# ---------------------------------------------------------------------------
#  RECONSTRUCCION DE LOS MODELOS YA ENTRENADOS
# ---------------------------------------------------------------------------
#  Aca no se entrena nada: se vuelven a armar los modelos que dejaron los
#  experimentos F2 (white-box) y F3 (gray-box) para poder integrarlos de nuevo.

# White-box a partir del json de F2. F2 no guarda checkpoint .pt, solo el
# diccionario de los 10 parametros identificados, asi que el modelo se reconstruye
# desde ahi.
#
# POR QUE learnable_params=True aunque no se vaya a entrenar: con False el
# constructor espera tambien 'ke' y 'ki' en el diccionario -> KeyError, porque el
# json de F2 trae solo los 10 libres. Con True, ke y ki se recalculan en cada
# forward a partir de ae,thetae / ai,thetai (que es lo correcto: conserva el reposo
# E=I=0). Los valores no se alteran: el constructor guarda inv_softplus(p) y el
# forward aplica softplus, asi que se recupera p exactamente.
# use_correction queda en su default False: el white-box es solo las ecuaciones.
def modelo_whitebox(jf):
    p = jd(jf)["params"]
    m = GrayBoxWC(p, {k: p[k] for k in ("wEE", "wEI", "wIE", "wII")},
                  learnable_weights=True, learnable_params=True)
    m.eval(); return m


# Gray-box a partir de un checkpoint .pt de F3.
#
# Los 1.0 de inicializacion son de relleno: load_state_dict pisa TODOS los pesos
# enseguida. Lo que si tiene que coincidir con el checkpoint es la ARQUITECTURA
# (use_correction y correction_inputs), porque define que tensores existen; si no
# coincide, load_state_dict falla por claves faltantes o sobrantes. Por eso los dos
# flags se leen del propio ck y no se hardcodean.
#
# TRAMPA: no se pasa structured=..., asi que esta funcion solo sabe cargar las
# variantes con correccion NEURONAL (A/B/C/D). Un checkpoint de la variante
# estructurada (f3_S_*) traeria raw_r_i, raw_r_e, raw_alpha y load_state_dict
# reventaria. Para eso esta carga() en scripts/exp_reproduccion.py.
def modelo_graybox(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = GrayBoxWC({k: 1.0 for k in ALL_P},
                  {k: 1.0 for k in ("wEE", "wEI", "wIE", "wII")},
                  learnable_weights=True, learnable_params=True,
                  use_correction=ck.get("use_correction", False),
                  correction_inputs=ck.get("correction_inputs", "x"))
    m.load_state_dict(ck["state"]); m.eval(); return m


@torch.no_grad()
def simula(m, d, s):
    """Rollout OPEN LOOP del escenario s del dataset d. Devuelve (I,E) predichos.

    Esta es LA operacion del script. Del dato real se toma una sola cosa: el estado
    del instante 0 (d["I"][s,0], d["E"][s,0]). El resto lo genera el modelo solo,
    guiado unicamente por el estimulo comandado P,Q.

    no_grad porque solo se evalua: sin eso PyTorch construiria el grafo de los 4000
    pasos RK4 y gastaria memoria al por mayor para nada.

    Detalles que es facil equivocar:
      - Ps[:-1] / Qs[:-1]: rollout devuelve T+1 estados si recibe T pasos de
        estimulo (el primero es x0). Con T pasos daria 4001 puntos y ya no se
        podria restar contra el dato, que tiene T. Pasando T-1 pasos salen
        exactamente T puntos, alineados con la grilla del .npz.
      - la forma (T,1,1) es (tiempo, batch=1, canal=1): rollout integra un lote de
        trayectorias en paralelo, aca el lote es de una sola.
      - el [:, 0, :] final saca esa dimension de batch -> queda (T, 2) = [I, E].
      - dt sale del propio archivo, NO se hardcodea: los .npz de esta familia usan
        dt ~ 0.05 ms, y meter otro dt cambiaria la trayectoria integrada.
      - P y Q son los COMANDADOS. Los P_eff/Q_eff (lo que realmente llega a las
        poblaciones despues del actuador) tambien estan en el .npz, pero usarlos
        seria trampa: en un experimento real no se conocen.
    """
    T = d["I"].shape[1]
    x0 = torch.tensor([[d["I"][s, 0], d["E"][s, 0]]], dtype=torch.float32)
    Ps = torch.tensor(d["P"][s], dtype=torch.float32).reshape(T, 1, 1)
    Qs = torch.tensor(d["Q"][s], dtype=torch.float32).reshape(T, 1, 1)
    return rollout(m, x0, Ps[:-1], Qs[:-1], float(d["dt"]))[:, 0, :].numpy()


# Indice del PRIMER escenario de test del dataset (el que se dibuja siempre).
# Test = estimulo que el modelo nunca vio al entrenar; es lo unico que permite
# hablar de generalizacion. El split viene marcado en el .npz (is_test) y es el
# MISMO en todos los niveles de eps, asi que este indice hace comparables las
# figuras de eps0 y eps1: es el mismo estimulo en las dos.
def escenario_test(d):
    return int(np.where(d["is_test"].astype(bool))[0][0])


def nrmse(pred, real):
    """Error de la trayectoria normalizado al rango de la senal, en %.

    Por que normalizado: un RMSE crudo no se puede leer (depende de las unidades y
    de cuanta actividad haya en ese escenario). Dividido por (max - min) del dato
    real queda "me equivoco un X% de lo que la senal se mueve", comparable entre
    escenarios y entre niveles de eps.

    Es la misma definicion que usa scripts/exp_reproduccion.py. Queda aca como
    referencia de la formula: las figuras NO la llaman, toman los numeros ya
    calculados de reproduccion.json para no arriesgar que la figura y la tabla del
    informe muestren valores distintos.
    """
    return 100.0 * np.sqrt(((pred - real) ** 2).mean()) / (real.max() - real.min())


# =============================================================================
#  1 · Control positivo: con el modelo completo, ¿copia?
# =============================================================================
#  Que muestra: eps0 = al modelo NO le falta nada (simulador y modelo son las
#  mismas ecuaciones). Si en este caso el rollout open loop no calzara, el problema
#  seria de la prueba (integrador, alineacion, dt) y no del gray-box. Que calce
#  valida el metodo y da el piso de error alcanzable (~2%).
def copia_sin_hueco():
    d = np.load("data/processed/uncertain/eps0.npz", allow_pickle=True)
    s = escenario_test(d)
    pred = simula(modelo_whitebox("f2_eps0.json"), d, s)
    # Se dibujan solo los primeros 120 ms de los 200: con la ventana completa los
    # ciclos quedan tan apretados que no se distingue si las curvas se separan.
    # searchsorted devuelve el indice del primer tiempo >= 120 ms.
    t = d["t"]; n = np.searchsorted(t, 120)

    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    ax.plot(t[:n], d["E"][s, :n], color=AZUL, lw=3.4, label="el simulador")
    ax.plot(t[:n], pred[:n, 1], color=NARA, lw=2.0, ls="--",
            label="el modelo, corriendo solo")
    ax.set_xlabel("tiempo (ms)"); ax.set_ylabel("actividad  E")
    # Aire arriba (32% sobre el pico) para que la leyenda y el texto verde no se
    # monten encima de la curva.
    ax.set_ylim(top=d["E"][s, :n].max() * 1.32)
    ax.legend(frameon=False, loc="upper right", ncol=2); ax.grid(alpha=0.6)
    ax.text(0.012, 0.94, "las dos curvas se superponen", transform=ax.transAxes,
            fontsize=15, color=VERDE, fontweight="bold", va="top")
    fig.tight_layout()
    guarda(fig, "c_sin_hueco")


# =============================================================================
#  2 · LA FIGURA CENTRAL: white-box vs gray-box contra el simulador
# =============================================================================
#  Ahora con eps1: al modelo LE FALTA fisica. Arriba, solo las ecuaciones; abajo,
#  ecuaciones + la correccion neuronal. La pregunta de la figura es si la red tapa
#  el hueco, y la respuesta honesta que se ve es "casi no" (15% -> 14%).
#
#  El gray-box elegido es la variante D (g depende solo de I,E y se penaliza la
#  parte de g que un cambio de parametros ya podia explicar) con lambda=1: es la
#  principista del estudio, la que no deja que la red compita con los parametros.
#  El [:, 1] toma la columna E: es la poblacion que se grafica en todo el paquete.
def copia_comparacion():
    d = np.load("data/processed/uncertain/eps1.npz", allow_pickle=True)
    s = escenario_test(d)
    real = d["E"][s]
    wb = simula(modelo_whitebox("f2_eps1.json"), d, s)[:, 1]
    gb = simula(modelo_graybox("results/uncertainty/models/f3_D_eps1_lam1.pt"),
                d, s)[:, 1]
    t = d["t"]; n = np.searchsorted(t, 120)

    # Los errores de los titulos salen de reproduccion.json (no de estas curvas).
    # OJO al leer la figura: son el PROMEDIO sobre los 7 escenarios de test, no el
    # error de la trayectoria dibujada, que es solo el primero de ellos. Se hace
    # asi a proposito, para que el numero coincida con la tabla del informe.
    rep = {r["modelo"]: r for r in jd("reproduccion.json")}
    e_wb = rep["white-box, planta CON hueco"]["nrmse_E"]
    e_gb = rep["gray-box D regularizada"]["nrmse_E"]

    # Dos paneles con el eje de tiempo compartido: el ojo compara verticalmente
    # donde se despega cada uno.
    fig, ax = plt.subplots(2, 1, figsize=(13.5, 7), sharex=True)
    for k, (pred, nom, err, col) in enumerate([
            (wb, "SÓLO las ecuaciones (white-box)", e_wb, GRIS),
            (gb, "ecuaciones + la red (gray-box)", e_gb, NARA)]):
        ax[k].plot(t[:n], real[:n], color=AZUL, lw=3.4, label="el simulador")
        ax[k].plot(t[:n], pred[:n], color=col, lw=2.2, ls="--", label=nom)
        ax[k].set_ylabel("actividad  E")
        # El tope mira el maximo de las DOS curvas: cuando el modelo se dispara por
        # encima del dato real (pasa con el white-box), mirar solo el real cortaria
        # la prediccion y esconderia justamente el error.
        ax[k].set_ylim(top=max(real[:n].max(), pred[:n].max()) * 1.34)
        ax[k].grid(alpha=0.6)
        ax[k].set_title(f"{nom}   ·   error {err:.0f} %", fontsize=16.5,
                        loc="left", color=col)
        # La leyenda va una sola vez: las dos filas usan el mismo codigo de color.
        if k == 0:
            ax[k].legend(frameon=False, loc="upper right", ncol=2)
    ax[1].set_xlabel("tiempo (ms)")
    fig.tight_layout()
    guarda(fig, "c_comparacion")


# =============================================================================
#  3 · El error se acumula: por que "copiar" es exigente
# =============================================================================
#  En vez de las trayectorias, el error INSTANTANEO |pred - real| a lo largo del
#  tiempo. Es la figura que explica por que el open loop es una prueba dura: las
#  tres curvas arrancan en cero (todas parten del estado real) y su envolvente se
#  va levantando, porque el error de cada paso entra como condicion inicial del
#  siguiente. No es una rampa limpia: oscila con el ciclo (los picos caen donde la
#  senal se mueve rapido y un desfase chico ya da diferencia grande). Lo que hay que
#  mirar es la envolvente: el modelo completo se queda abajo, los otros dos no.
def copia_error_acumula():
    d = np.load("data/processed/uncertain/eps1.npz", allow_pickle=True)
    d0 = np.load("data/processed/uncertain/eps0.npz", allow_pickle=True)
    s, s0 = escenario_test(d), escenario_test(d0)
    real, real0 = d["E"][s], d0["E"][s0]
    wb = simula(modelo_whitebox("f2_eps1.json"), d, s)[:, 1]
    gb = simula(modelo_graybox("results/uncertainty/models/f3_D_eps1_lam1.pt"),
                d, s)[:, 1]
    # "ok" = el control positivo, que corre sobre SU propio dataset (eps0): es otro
    # cerebro, uno sin fisica faltante. Por eso hace falta el par (d0, s0) aparte.
    ok = simula(modelo_whitebox("f2_eps0.json"), d0, s0)[:, 1]
    # Aca si se usa la ventana completa: la gracia es ver la deriva a lo largo de
    # todo el registro. La grilla termina justo en 200 ms, asi que searchsorted
    # devuelve el indice del ultimo punto y t[:n] es practicamente todo.
    t = d["t"]; n = np.searchsorted(t, 200)

    # Error instantaneo en % del rango. Toma "n" del ambito de la funcion (no es
    # parametro) para no repetirlo en las tres llamadas.
    def err_acum(p, r, rng):
        return 100 * np.abs(p[:n] - r[:n]) / rng

    # Cada dataset se normaliza con SU propio rango: eps0 y eps1 no tienen la misma
    # amplitud de actividad (la fisica que falta cambia la senal), asi que usar un
    # solo rango para las tres curvas las volveria incomparables.
    rng1 = real.max() - real.min()
    rng0 = real0.max() - real0.min()

    fig, ax = plt.subplots(figsize=(13.5, 4.8))
    ax.plot(t[:n], err_acum(ok, real0, rng0), color=VERDE,
            label="modelo completo")
    ax.plot(t[:n], err_acum(wb, real, rng1), color=GRIS,
            label="con hueco · sólo ecuaciones")
    ax.plot(t[:n], err_acum(gb, real, rng1), color=NARA,
            label="con hueco · ecuaciones + red")
    ax.set_xlabel("tiempo (ms)")
    ax.set_ylabel("error instantáneo (% del rango)")
    ax.legend(frameon=False, loc="upper left", ncol=3); ax.grid(alpha=0.6)
    # El error es un valor absoluto: anclar el eje en 0 evita que matplotlib deje un
    # margen negativo que sugiera que el error puede bajar de cero.
    ax.set_ylim(bottom=0)
    ax.text(0.985, 0.93, "el modelo arranca bien y se va despegando",
            transform=ax.transAxes, ha="right", va="top", fontsize=14,
            color=GRIS)
    fig.tight_layout()
    guarda(fig, "c_error_acumula")


# =============================================================================
#  4 · Retrato de fase: ¿reproduce la FORMA del ciclo?
# =============================================================================
#  Se saca el tiempo del dibujo y se grafica E contra I. Sirve para separar dos
#  fracasos que en el tiempo se confunden: un modelo puede tener error grande solo
#  por desfase (mismo ciclo limite, corrido en el tiempo -> el lazo se superpone) o
#  puede estar generando otra dinamica (lazo de otro tamano o de otra forma, o ni
#  siquiera cerrar). Lo primero es aceptable para control; lo segundo no.
def copia_retrato():
    d0 = np.load("data/processed/uncertain/eps0.npz", allow_pickle=True)
    d1 = np.load("data/processed/uncertain/eps1.npz", allow_pickle=True)
    s0, s1 = escenario_test(d0), escenario_test(d1)
    ok = simula(modelo_whitebox("f2_eps0.json"), d0, s0)
    wb = simula(modelo_whitebox("f2_eps1.json"), d1, s1)
    gb = simula(modelo_graybox("results/uncertainty/models/f3_D_eps1_lam1.pt"),
                d1, s1)
    # Descarta los primeros 20 ms: el arranque desde el reposo es un transitorio que
    # dibuja una cola hacia el origen y ensucia el lazo. 20 ms alcanzan para que el
    # sistema ya este sobre el ciclo. La grilla de tiempo es la misma en eps0 y eps1,
    # asi que este unico indice sirve para los tres paneles.
    a = np.searchsorted(d0["t"], 20)     # descarta el transitorio

    # sharex/sharey: los tres paneles con la MISMA escala. Sin eso, cada retrato se
    # autoescala y tres lazos de tamanos distintos se verian iguales.
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True, sharey=True)
    casos = [
        (d0["I"][s0], d0["E"][s0], ok, "Modelo completo", VERDE),
        (d1["I"][s1], d1["E"][s1], wb, "Con hueco · sólo ecuaciones", GRIS),
        (d1["I"][s1], d1["E"][s1], gb, "Con hueco · + la red", NARA),
    ]
    for k, (rI, rE, pred, nom, col) in enumerate(casos):
        # Azul: el lazo del simulador. Punteado: el del modelo. pred[:,0] es I y
        # pred[:,1] es E (mismo orden de estado que el backbone Wilson-Cowan).
        ax[k].plot(rI[a:], rE[a:], color=AZUL, lw=2.4, alpha=.95,
                   label="el simulador")
        ax[k].plot(pred[a:, 0], pred[a:, 1], color=col, lw=1.8, ls="--",
                   label="el modelo")
        ax[k].set_title(nom, fontsize=15, color=col, loc="left")
        ax[k].set_xlabel("población  I")
        ax[k].grid(alpha=0.55)
        if k == 0:
            ax[k].set_ylabel("población  E")
            ax[k].legend(frameon=False, fontsize=12.5, loc="upper left")
    fig.tight_layout()
    guarda(fig, "c_retrato")


# =============================================================================
#  5 · El resumen en una barra
# =============================================================================
#  Toda la historia en tres numeros, promediados sobre los escenarios de test:
#  ~2% el modelo completo, ~15% con hueco y solo ecuaciones, ~14% con hueco y red.
#  La figura esta armada para que se lea el resultado incomodo, no para venderlo:
#  la flecha roja mide cuanto recorta la red, y lo que se ve es que recorta poco.
#  Es la conclusion del estudio: una correccion sin memoria no puede reponer un
#  hueco que tiene estado propio (el retardo del actuador).
def copia_resumen():
    # Mismos numeros que los titulos de la figura 2 (nrmse_E de reproduccion.json),
    # para que las dos figuras no se contradigan.
    rep = {r["modelo"]: r for r in jd("reproduccion.json")}
    nombres = ["modelo\nCOMPLETO", "con hueco\nsólo ecuaciones",
               "con hueco\n+ la red"]
    vals = [rep["white-box, planta SIN hueco"]["nrmse_E"],
            rep["white-box, planta CON hueco"]["nrmse_E"],
            rep["gray-box D regularizada"]["nrmse_E"]]
    cols = [VERDE, GRIS, NARA]

    fig, ax = plt.subplots(figsize=(10.5, 5))
    b = ax.bar(nombres, vals, 0.55, color=cols)
    # El valor escrito arriba de cada barra: en una proyeccion nadie estima la
    # altura contra el eje. El offset de 6 puntos lo despega del borde.
    for bb, v in zip(b, vals):
        ax.annotate(f"{v:.1f} %", (bb.get_x() + bb.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=18, fontweight="bold", color=TINTA)
    ax.set_ylabel("error al copiar el comportamiento (%)")
    # Grilla solo horizontal: en un grafico de barras las verticales no aportan.
    ax.grid(alpha=0.5, axis="y")
    # Desde 0 (obligatorio en barras: recortar la base exagera las diferencias) y
    # con 35% de aire arriba para las etiquetas y el texto de la flecha.
    ax.set_ylim(0, max(vals) * 1.35)
    ax.tick_params(axis="x", labelsize=14)
    # Flecha curva de la barra 1 (solo ecuaciones) a la barra 2 (+ la red): es lo
    # que la red gana. rad negativo la curva por arriba, para que no cruce las
    # barras. Texto vacio porque la leyenda va aparte, centrada en x=1.5.
    ax.annotate("", xy=(2, vals[2]), xytext=(1, vals[1]),
                arrowprops=dict(arrowstyle="->", color=ROJO, lw=2.2,
                                connectionstyle="arc3,rad=-0.32"))
    ax.text(1.5, max(vals) * 1.16, f"la red sólo recorta {vals[1]-vals[2]:.1f} puntos",
            ha="center", fontsize=15, color=ROJO, fontweight="bold")
    fig.tight_layout()
    guarda(fig, "c_resumen")


# =============================================================================
#  EJECUCION DIRECTA: genera las cinco figuras en OUT.
#  Orden = orden del relato: primero se valida la prueba con el modelo completo,
#  despues se abre el hueco, se muestra como se acumula el error, se chequea la
#  forma del ciclo y se cierra con el resumen.
# =============================================================================
if __name__ == "__main__":
    print("Figuras de la pregunta central:")
    copia_sin_hueco()
    copia_comparacion()
    copia_error_acumula()
    copia_retrato()
    copia_resumen()
    print(f"\n-> {OUT}/")
