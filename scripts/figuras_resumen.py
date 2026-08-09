#!/usr/bin/env python3
# =============================================================================
#  FIGURAS DEL RESUMEN PARA COMPANEROS
# =============================================================================
#
#  Genera las figuras del PDF de resumen. Todas salen de las corridas REALES
#  (results/uncertainty/*.json y data/processed/uncertain/*.npz): ninguna es
#  esquematica ni inventada.
#
#  Cada figura contesta UNA pregunta en lenguaje simple; el titulo de cada una
#  es esa pregunta.
#
#  CONTEXTO (lo que hay que saber para entender que muestran las figuras):
#    El proyecto identifica los 10 parametros de Wilson-Cowan con una Neural
#    ODE. Al SIMULADOR se le agrego fisica que el MODELO no contempla
#    (refractariedad del WC de 1972 + un actuador optogenetico con retardo y
#    saturacion), asi el termino de correccion neuronal g_phi tiene algo real
#    que aprender. La perilla epsilon gradua cuanta fisica falta:
#      eps = 0  -> planta = modelo (el mundo comodo de siempre)
#      eps = 1  -> el hueco vale un tercio del campo
#    Reparto de papeles:
#      simulador + perturbacion = "el cerebro real"  (tiene toda la fisica)
#      el .npz                  = lo que se mide     (solo t, I, E, P, Q)
#      backbone WC + g_phi      = la hipotesis + lo que no explica
#    Referencia: docs/graybox_manual_completo.md
#
#  DE DONDE SALE CADA NUMERO:
#    results/uncertainty/f2_eps*.json   -> white-box (sin red) por cada eps
#    results/uncertainty/f3_*.json      -> gray-box (con red) por variante
#    results/uncertainty/f4b_geometria  -> descomposicion del hueco
#    results/uncertainty/f5_recovery    -> cuanto de la fisica real recupero g
#    results/uncertainty/f6_closed_loop -> control en lazo cerrado
#    results/uncertainty/reproduccion   -> rollout open-loop (predecir 200 ms)
#    data/processed/uncertain/eps*.npz  -> las trayectorias crudas
#
#  ORDEN DE LECTURA (es el orden del PDF, y el del __main__ al final):
#    8 (esquema, para ubicarse) -> 1 (que se agrego) -> 2 (el costo) ->
#    3 (por que pasa) -> 4 (el cruce) -> 5 (predice?) -> 6 (aprendio la
#    fisica?) -> 7 (sirve para controlar?).
#
#  TRAMPA DE USO: todas las rutas de datos son RELATIVAS (RES, OUT,
#  "data/processed/..."), asi que el script hay que correrlo DESDE LA RAIZ del
#  repo. Desde otra carpeta falla al abrir el primer .json. Se hizo asi a
#  proposito para que las figuras queden con rutas cortas y reproducibles en
#  el PDF; _ROOT solo se usa para poder importar src/.
#
#  USO:  python scripts/figuras_resumen.py
# =============================================================================

from __future__ import annotations

import json
import sys
from pathlib import Path

# La raiz del repo, para que "from src.neural_ode import ..." funcione aunque
# el script se invoque como scripts/figuras_resumen.py y no como modulo.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib
# Backend sin ventana: esto corre en batch y solo escribe PNGs. Tiene que ir
# ANTES de importar pyplot, si no matplotlib ya eligio backend y el use() no
# hace nada (error clasico y silencioso).
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("results/uncertainty")   # de aca se leen los resultados medidos
OUT = Path("docs/figuras_resumen")  # aca se escriben los PNGs que van al PDF

# --- Paleta validada (skill dataviz, checks corridos) ------------------------
# Solo dos series por panel casi siempre: azul = "lo bueno / la referencia",
# naranja = "lo que se le agrega o lo que se prueba". Manteniendo ese pacto en
# las 8 figuras, el lector no tiene que releer la leyenda cada vez.
AZUL = "#2a78d6"      # serie 1
NARA = "#eb6834"      # serie 2
AQUA = "#1baf7a"      # serie 3 (siempre con etiqueta directa: contraste < 3:1)
ROJO = "#e34948"      # estado: malo
VERDE = "#008300"     # estado: bueno
TINTA = "#0b0b0b"     # texto principal (no negro puro: menos duro impreso)
GRIS = "#52514e"      # texto secundario, ejes y notas
SUAVE = "#d8d8d4"     # grillas y bordes: tienen que estar atras de los datos

# Estilo global: tipografia algo mas grande de lo normal (el PDF se lee en
# pantalla y proyectado), ejes y grillas en gris claro, y sin las dos espinas
# de arriba/derecha para que la tinta este en los datos y no en el marco.
plt.rcParams.update({
    "font.size": 10.5,
    "axes.edgecolor": SUAVE,
    "axes.labelcolor": GRIS,
    "axes.titlecolor": TINTA,
    "text.color": TINTA,
    "xtick.color": GRIS,
    "ytick.color": GRIS,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": SUAVE,
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# =============================================================================
#  DOS AYUDANTES QUE USAN TODAS LAS FIGURAS
# =============================================================================

# Guarda la figura y la cierra. Las decisiones que estan aca:
#   - dpi=170: suficiente para que el texto de 8-9 pt siga nitido impreso, sin
#     que los PNG pesen de mas (a 300 dpi el PDF se vuelve inmanejable).
#   - bbox_inches="tight": recorta el margen sobrante. Es lo que hace que las
#     anotaciones que se salen del area de ejes (las flechas, las notas) no
#     queden cortadas al insertar la figura en el documento.
#   - facecolor="white" de nuevo aunque ya este en rcParams: savefig NO hereda
#     el facecolor de la figura por defecto en todas las versiones, y sin esto
#     el fondo puede salir transparente -> se ve gris sobre el PDF.
#   - plt.close: se generan 8 figuras seguidas; sin cerrarlas matplotlib las
#     acumula en memoria y avisa por warning.
def guarda(fig, nombre):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{nombre}.png", dpi=170, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  {nombre}.png")


# Atajo para leer un resultado medido de results/uncertainty. Todos los numeros
# que aparecen escritos en las figuras salen de aca: NINGUNO esta cableado en
# este archivo, asi que si se re-corre un experimento las figuras se actualizan
# solas. Es la regla mas importante del script.
def jd(nombre):
    return json.loads((RES / nombre).read_text())


# =============================================================================
#  FIG 1 — Que le agregamos al simulador
# =============================================================================
def fig_perturbacion():
    """Pregunta: que es exactamente la "fisica que falta"?

    Es la figura de apertura y tiene que dejar dos ideas, una por panel:
      izquierda: la perturbacion DEFORMA la dinamica pero no la rompe. Si el
        ciclo limite desapareciera, el experimento no probaria nada (seria
        "cambie el sistema por otro" y no "a mi modelo le falta un termino").
      derecha: el estimulo comandado no es el que llega. Esto justifica por que
        el hueco no se puede arreglar midiendo mejor: la distorsion del actuador
        esta del lado de la planta y el modelo solo conoce P comandado.

    Se comparan los MISMOS escenarios de eps0 y eps1: mismo estimulo, misma
    condicion inicial, misma semilla. Lo unico distinto es la perturbacion, asi
    que toda diferencia visible es atribuible a ella.
    """
    d0 = np.load("data/processed/uncertain/eps0.npz", allow_pickle=True)
    d1 = np.load("data/processed/uncertain/eps1.npz", allow_pickle=True)
    # El escenario que MAS oscila: es donde se ve que el ciclo se deforma sin
    # desaparecer, que es lo que la figura tiene que mostrar.
    # (Se elige sobre d0, la planta limpia, para que la eleccion no dependa de
    # la perturbacion; los escenarios estan alineados uno a uno entre los .npz.)
    s = int(np.argmax(d0["E"].std(axis=1)))
    t = d0["t"]
    n = np.searchsorted(t, 110)          # primeros 110 ms: se ve el detalle
    # Por que recortar: la trayectoria dura 200 ms con 4000 muestras. Dibujada
    # entera, los ciclos se apelmazan y las dos curvas parecen la misma cosa.
    # Con ~110 ms entran varios periodos y todavia se distinguen ciclo a ciclo.

    # Dos paneles lado a lado, no uno arriba del otro: son dos afirmaciones
    # independientes (la actividad / el estimulo), no la misma senal en dos
    # escalas, asi que no comparten eje y.
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))

    ax[0].plot(t[:n], d0["E"][s, :n], lw=1.8, color=AZUL,
               label="Wilson-Cowan puro (lo que asume el modelo)")
    ax[0].plot(t[:n], d1["E"][s, :n], lw=1.8, color=NARA,
               label="con la física agregada (el \"cerebro real\")")
    ax[0].set_title("La actividad cambia, pero el ritmo se conserva", fontsize=11)
    ax[0].set_xlabel("tiempo (ms)")
    ax[0].set_ylabel("actividad excitatoria  E")
    # Espacio arriba para que la leyenda no se monte sobre los picos.
    # El 1.42 es empirico: es el factor mas chico con el que la leyenda de dos
    # renglones entra sin tapar el pico mas alto de estos datos.
    ax[0].set_ylim(top=max(d0["E"][s, :n].max(), d1["E"][s, :n].max()) * 1.42)
    ax[0].legend(fontsize=8.5, frameon=False, loc="upper right")
    ax[0].grid(alpha=0.5, lw=0.6)

    # Los dos vienen del MISMO .npz (eps1): P es el comandado y P_eff el
    # efectivo. En eps0 serian identicos, asi que este panel solo tiene sentido
    # con la planta perturbada. P_eff se guarda unicamente como diagnostico:
    # jamas entra al entrenamiento, porque en un experimento real no se mide.
    ax[1].plot(t[:n], d1["P"][s, :n], lw=1.8, color=AZUL,
               label="lo que comandás  (lo único que ve el modelo)")
    ax[1].plot(t[:n], d1["P_eff"][s, :n], lw=1.8, color=NARA,
               label="lo que llega a la neurona")
    ax[1].set_title("El estímulo que mandás no es el que llega", fontsize=11)
    ax[1].set_xlabel("tiempo (ms)")
    ax[1].set_ylabel("estímulo  P")
    # El techo se calcula sobre el COMANDADO a proposito: el efectivo esta
    # saturado y siempre queda por debajo, asi que P manda la escala.
    ax[1].set_ylim(top=d1["P"][s, :n].max() * 1.42)
    ax[1].legend(fontsize=8.5, frameon=False, loc="upper right")
    ax[1].grid(alpha=0.5, lw=0.6)

    fig.tight_layout()
    guarda(fig, "fig1_perturbacion")


# =============================================================================
#  FIG 2 — El costo de la rigidez
# =============================================================================
def fig_costo():
    """Pregunta: cuanto se paga por tener un modelo incompleto?

    La curva de "todo el resultado del proyecto depende de una hipotesis
    comoda": con eps=0 el error es ~1 %, y crece monotonamente hasta ~94 % con
    eps=2. Es un barrido de la perilla, no un caso puntual, y eso es lo que la
    hace convincente: no hay un valor de eps elegido para que quede lindo.

    Sale de los f2_eps*.json, que son las corridas WHITE-BOX (sin red). Aca
    todavia no aparece la correccion: primero el problema, despues el remedio.
    """
    # Glob + sort por eps: asi agregar una corrida nueva (f2_eps3.json) suma un
    # punto a la curva sin tocar el codigo. Ordenar es obligatorio porque glob
    # devuelve orden de directorio y "0.25" ordena antes que "0" como texto.
    filas = sorted([jd(p.name) for p in RES.glob("f2_eps*.json")],
                   key=lambda f: f["eps"])
    eps = [f["eps"] for f in filas]
    med = [f["mean_param_error"] for f in filas]
    wii = [f["param_errors"]["wII"] for f in filas]

    fig, ax = plt.subplots(figsize=(7.2, 4))
    # Dos curvas y no una: el promedio solo esconde que el dano se concentra.
    # wII es SIEMPRE el peor parametro en las 6 corridas (campo "peor_param" de
    # cada json) y llega a ~190 % de error mientras el promedio va por 60 %.
    # No es casualidad: el analisis Fisher+SVD del proyecto ya marcaba a wII
    # como la direccion menos identificable del modelo, y el hueco entra
    # justamente por ahi. Que la prediccion de la FIM se cumpla es parte del
    # resultado, por eso wII merece su propia curva.
    ax.plot(eps, wii, "o-", lw=2, ms=8, color=NARA,
            label="el peor parámetro (siempre $w_{II}$)")
    ax.plot(eps, med, "o-", lw=2, ms=8, color=AZUL,
            label="error medio de los 10 parámetros")
    ax.set_xlabel("ε  —  cuánta física le falta al modelo")
    ax.set_ylabel("error de identificación (%)")
    ax.set_title("Cuanto más incompleto el modelo, peor identifica", fontsize=11.5)
    ax.grid(alpha=0.5, lw=0.6)
    ax.legend(fontsize=9, frameon=False, loc="lower right")

    # Ancla el punto eps=0 al resultado que los companeros YA conocen (~1 %).
    # Sin esta nota la curva se lee como "un experimento nuevo mas"; con ella se
    # lee como "el resultado que ya viste es el borde izquierdo de esto".
    # xytext=(0.13, 45) esta en coordenadas de datos y esta puesto a mano: es un
    # hueco vacio entre las dos curvas para estos valores. Si el barrido de eps
    # cambia mucho, hay que reubicarlo.
    ax.annotate(f"sin hueco: {med[0]:.1f} %\n(el resultado conocido\ndel proyecto)",
                xy=(0, med[0]), xytext=(0.13, 45), fontsize=8.5, color=GRIS,
                arrowprops=dict(arrowstyle="->", color=GRIS, lw=1))
    fig.tight_layout()
    guarda(fig, "fig2_costo_rigidez")


# =============================================================================
#  FIG 3 — La geometria: dos tercios se disfrazan de parametros
# =============================================================================
def fig_geometria():
    """Pregunta: POR QUE el modelo incompleto arruina los parametros?

    Esta es la figura que explica todas las demas, y es la mas conceptual.
    La idea: el hueco (Delta f = campo real - campo Wilson-Cowan nominal) se
    proyecta sobre el espacio que generan las 10 sensibilidades df/dtheta. Esa
    proyeccion se parte en dos pedazos ortogonales:
      - "imitable": la parte que un cambio de theta puede reproducir. El
        optimizador la absorbe DEFORMANDO los parametros, y encima el ajuste
        mejora. De ahi el 187 % de error en wII de la figura 2.
      - "nueva": la parte que ningun theta puede imitar. Es lo unico que g_phi
        podria aportar de verdad.
    Con eps=1 dos tercios del hueco son imitables, y por eso el reparto entre
    theta y g deja de ser unico. Es la causa raiz del trabajo entero.

    El panel derecho es la misma cuenta como porcentaje: se repite a proposito,
    porque el panel de barras se lee en unidades del campo (que nadie tiene
    intuicion de cuanto es) y el porcentaje se lee solo.
    """
    # Tolerancia al formato: el json paso de ser una lista a ser un dict con
    # claves "por_eps"/"por_estimulo". Se acepta cualquiera de las dos formas
    # para no romper con archivos viejos.
    g = jd("f4b_geometria.json")
    filas = g["por_eps"] if isinstance(g, dict) else g
    eps = [f["eps"] for f in filas]
    imit = [f["rms_imitable"] for f in filas]
    nuevo = [f["rms_nuevo"] for f in filas]
    frac = [100 * f["frac_imitable"] for f in filas]
    x = np.arange(len(eps))

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))

    # Barras APILADAS y no lado a lado: se quiere leer "de que esta hecho el
    # hueco", o sea las partes de un total, no comparar dos series.
    # CUIDADO al interpretar la altura total: los dos pedazos son ortogonales,
    # asi que el hueco real es la suma en cuadratura (sqrt(imit^2 + nuevo^2) =
    # df_rms), no la suma lineal que dibuja la barra. La barra apilada exagera
    # el total; lo que la figura tiene que comunicar es la PROPORCION, y para el
    # numero exacto esta el panel derecho.
    ax[0].bar(x, imit, 0.6, color=NARA, label="se disfraza de parámetros")
    # El +0.0004 del bottom es una hairline: separa visualmente los dos
    # segmentos. Es ~2 % del valor tipico, no cambia la lectura.
    ax[0].bar(x, nuevo, 0.6, bottom=np.array(imit) + 0.0004, color=AZUL,
              label="física realmente nueva")
    # Eje x categorico (0,1,2,...) y no el valor de eps: los eps del barrido no
    # estan equiespaciados (0.25, 0.5, 1, 1.5, 2) y con barras eso se veria como
    # anchos desparejos. En el panel derecho, que es una curva, si va eps real.
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{e:g}" for e in eps])
    ax[0].set_xlabel("ε"); ax[0].set_ylabel("tamaño del hueco")
    ax[0].set_title("De qué está hecho el hueco", fontsize=11)
    ax[0].legend(fontsize=8.5, frameon=False, loc="upper left")
    ax[0].grid(alpha=0.4, lw=0.6, axis="y")

    ax[1].plot(eps, frac, "o-", lw=2, ms=8, color=NARA)
    # Etiqueta directa en cada punto: son 5 valores y el mensaje ES el numero
    # ("dos tercios"), asi que obligar a leerlo contra el eje seria peor.
    for e, f in zip(eps, frac):
        ax[1].annotate(f"{f:.0f} %", (e, f), textcoords="offset points",
                       xytext=(0, 9), ha="center", fontsize=8.5, color=GRIS)
    # Eje fijo 0-105 y no automatico: es un porcentaje, y dejarlo autoescalar
    # haria que el crecimiento parezca mas dramatico de lo que es. El 105 (y no
    # 100) es para que la etiqueta del ultimo punto no quede pegada al borde.
    ax[1].set_ylim(0, 105)
    ax[1].set_xlabel("ε"); ax[1].set_ylabel("% del hueco imitable con parámetros")
    ax[1].set_title("La parte que un cambio de θ podría imitar", fontsize=11)
    ax[1].grid(alpha=0.5, lw=0.6)
    fig.tight_layout()
    guarda(fig, "fig3_geometria")


# =============================================================================
#  FIG 4 — El cruce: la red ayuda o arruina segun haya hueco
# =============================================================================
def fig_cruce():
    """Pregunta: la red de correccion ayuda o arruina?

    Respuesta: DEPENDE, y ahi esta la trampa. Cuatro barras, dos grupos:
      eps=0 (el modelo ya es correcto): la red empeora los parametros ~17 veces
             (1.05 % -> 18 %). No tiene fisica que aprender, asi que se dedica a
             tapar parametros equivocados.
      eps=1 (al modelo le falta fisica): la red recupera ~48 % del error
             (59.5 % -> 30.8 %).
    Lo grave: en LOS DOS casos el MSE de ajuste mejora. O sea que mirando el
    ajuste, que es lo unico que se tiene con datos reales, no hay manera de
    saber en cual de los dos escenarios se esta. Eso es lo que la figura
    denuncia, y es la razon por la que 2x2 barras es la forma correcta: el
    mensaje es el CRUCE, y un cruce necesita las dos condiciones juntas.

    Se toma el MEJOR (min) de las variantes de correccion en cada eps: la
    afirmacion tiene que ser "ni en el mejor caso alcanza", no "esta variante
    puntual fallo". Es la lectura conservadora, la que no se puede discutir.
    """
    wb0 = jd("f2_eps0.json")["mean_param_error"]
    wb1 = jd("f2_eps1.json")["mean_param_error"]
    gb0 = min(jd(f"f3_{v}_eps0.json")["mean_param_error"] for v in ("A", "B"))
    # Se excluye la variante "S_" (correccion ESTRUCTURADA: refractariedad +
    # saturacion con forma fisica conocida) porque no es una red de caja negra;
    # es el frente siguiente del proyecto y mezclarla aca arruinaria la
    # comparacion. OJO con la asimetria: en eps=0 las variantes se listan a mano
    # (A, B) y en eps=1 por glob. Si alguna vez se corren C/D con eps=0, hay que
    # acordarse de sumarlas a la tupla o el grupo izquierdo queda desactualizado.
    gb1 = min(jd(p.name)["mean_param_error"]
              for p in RES.glob("f3_*eps1*.json")
              if "S_" not in p.name)

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    # w=0.33 con un separador de 0.01 entre las dos barras del grupo: el hueco
    # dentro del grupo tiene que ser MUCHO menor que el hueco entre grupos, si
    # no se pierde que son pares comparables.
    x = np.array([0, 1]); w = 0.33
    b1 = ax.bar(x - w / 2 - 0.01, [wb0, wb1], w, color=AZUL,
                label="sin red (sólo las ecuaciones)")
    b2 = ax.bar(x + w / 2 + 0.01, [gb0, gb1], w, color=NARA,
                label="con la red de corrección")
    for b in list(b1) + list(b2):
        ax.annotate(f"{b.get_height():.1f} %",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9.5, color=TINTA)
    # Las etiquetas del eje x traducen eps a palabras. Nadie de afuera sabe que
    # es eps; lo que tiene que leer es "modelo correcto" contra "modelo
    # incompleto". El valor numerico va igual, para poder cruzar con la fig 2.
    ax.set_xticks(x)
    ax.set_xticklabels(["ε = 0\nEL MODELO YA ES CORRECTO",
                        "ε = 1\nAL MODELO LE FALTA FÍSICA"], fontsize=9.5)
    ax.set_ylabel("error de identificación (%)")
    ax.set_title("La red de corrección: ¿ayuda o arruina?", fontsize=11.5)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    # Grilla solo en y: en x las categorias son nominales, una grilla vertical
    # ahi no ayuda a comparar nada.
    ax.grid(alpha=0.4, lw=0.6, axis="y")
    # El techo lo fija el grupo eps=1 (la barra mas alta de las cuatro), con 32 %
    # de aire para las anotaciones "PEOR"/"MEJOR" que van encima de las barras.
    ax.set_ylim(0, max(wb1, gb1) * 1.32)

    # Las etiquetas van sobre la barra naranja de cada grupo (la de "con red"),
    # que es la que cambia respecto de la de al lado.
    # ROJO y VERDE se usan aca como semantica de estado (empeora / mejora), no
    # como series: es el unico lugar de la figura donde el color dice el juicio,
    # y el texto lo repite en palabras para no depender del color.
    # El "+5" del xytext esta en unidades de datos (% de error), no en puntos:
    # sube la nota 5 puntos porcentuales por encima de la barra.
    ax.annotate(f"×{gb0/wb0:.0f} PEOR", xy=(w / 2 + 0.01, gb0),
                xytext=(w / 2 + 0.01, gb0 + 5), fontsize=11,
                fontweight="bold", color=ROJO, ha="center")
    ax.annotate(f"{100*(1-gb1/wb1):.0f} % MEJOR", xy=(1 + w / 2 + 0.01, gb1),
                xytext=(1 + w / 2 + 0.01, gb1 + 5), fontsize=11,
                fontweight="bold", color=VERDE, ha="center")
    fig.tight_layout()
    guarda(fig, "fig4_cruce")


# =============================================================================
#  FIG 5 — ¿Copia bien el comportamiento?
# =============================================================================
def fig_reproduccion():
    """Pregunta: el modelo identificado copia bien el comportamiento?

    Es la prueba mas exigente y la que mas le importa a alguien de afuera. No se
    mide el ajuste paso a paso: se le da SOLO el estado inicial y el estimulo, y
    tiene que generar los 200 ms enteros por su cuenta, sin volver a mirar el
    dato real nunca (rollout open loop), y sobre escenarios de TEST que no vio al
    entrenar. Los errores de un paso se acumulan, asi que aca no se puede hacer
    trampa.

    Dos paneles, uno por escenario:
      arriba (eps=0): 2 % de error, correlacion 0.99 -> la maquinaria funciona.
      abajo  (eps=1): 15 % -> arranca bien, sigue la forma de la oscilacion,
             pero se desfasa. Serviria para entender el sistema, no para
             predecir con precision.
    El panel de arriba es el CONTROL POSITIVO: sin el, el 15 % de abajo se podria
    achacar a un bug del rollout en vez de al hueco.

    Los imports de torch son locales a la funcion (y no arriba del archivo) para
    que las figuras que no necesitan la red se generen sin pagar el arranque de
    torch, que tarda segundos.
    """
    import torch
    from src.neural_ode import GrayBoxWC, rollout
    from src.neural_ode.graybox_train import ALL_P
    # Limite de hilos: sin esto torch toma todos los cores y el rollout, que es
    # secuencial y de tensores chiquitos, se vuelve MAS lento por overhead.
    torch.set_num_threads(4)

    # Reconstruye el modelo WHITE-BOX con los 10 parametros identificados que
    # quedaron guardados en el json. use_correction queda en su valor por
    # defecto (apagado): esta figura mide que tan lejos llega la fisica sola.
    # learnable_* van en True solo para que la clase arme los parametros con la
    # misma parametrizacion (softplus) con la que se entrenaron; el modelo se
    # pone en eval() y nunca se le calcula un gradiente.
    def modelo_de(params):
        m = GrayBoxWC(params, {k: params[k] for k in ("wEE", "wEI", "wIE", "wII")},
                      learnable_weights=True, learnable_params=True)
        m.eval(); return m

    casos = []
    for tag, jf, npz, titulo in [
        ("sin", "f2_eps0.json", "eps0.npz",
         "Modelo correcto: la copia es casi perfecta"),
        ("con", "f2_eps1.json", "eps1.npz",
         "Al modelo le falta física: arranca bien y se va desfasando"),
    ]:
        d = np.load(f"data/processed/uncertain/{npz}", allow_pickle=True)
        m = modelo_de(jd(jf)["params"])
        # PRIMER escenario de test. Que sea de test es lo que hace honesta la
        # prueba: el estimulo de este escenario no se uso para identificar. Se
        # toma el primero y no el "mejor" para que la figura no sea una vidriera.
        it = d["is_test"].astype(bool)
        s = int(np.where(it)[0][0])
        T = d["I"].shape[1]
        with torch.no_grad():
            # Del dato real se usa UNICAMENTE la muestra 0 (el arranque) y todo
            # el estimulo comandado. Nada mas. El resto lo genera el modelo.
            x0 = torch.tensor([[d["I"][s, 0], d["E"][s, 0]]], dtype=torch.float32)
            # Forma (T, batch=1, 1): rollout espera la secuencia con eje de
            # batch, aunque aca haya un solo escenario.
            Ps = torch.tensor(d["P"][s], dtype=torch.float32).reshape(T, 1, 1)
            Qs = torch.tensor(d["Q"][s], dtype=torch.float32).reshape(T, 1, 1)
            # [:-1] porque con N estados hay N-1 pasos: el estimulo del ultimo
            # instante no se usa para avanzar hacia ningun lado. Pasar los T
            # daria un estado de mas y desalinearia la curva con el tiempo.
            pred = rollout(m, x0, Ps[:-1], Qs[:-1], float(d["dt"]))[:, 0, :].numpy()
        # Columna 1 = E (la 0 es I). Se grafica E porque es la poblacion cuya
        # oscilacion se reconoce a simple vista.
        casos.append((titulo, d["t"], d["E"][s], pred[:, 1]))

    # El numero del titulo NO se calcula aca: sale de reproduccion.json, que lo
    # promedia sobre TODOS los escenarios de test y lo normaliza al rango de la
    # senal (nrmse). Consecuencia a tener presente: la curva dibujada es un solo
    # escenario, asi que puede verse algo mejor o peor que el % del titulo. Se
    # prefiere el numero agregado porque es el que se puede defender.
    rep = {r["modelo"]: r for r in jd("reproduccion.json")}
    err = {"sin": rep["white-box, planta SIN hueco"]["nrmse_E"],
           "con": rep["white-box, planta CON hueco"]["nrmse_E"]}

    # sharex=True: los dos paneles tienen que estar en la MISMA escala temporal,
    # si no el desfase de abajo no es comparable con la coincidencia de arriba.
    # No se comparte el eje y, porque las amplitudes de los dos escenarios son
    # distintas y aplastarlas al mismo rango esconderia la forma.
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
    for k, (titulo, t, real, pred) in enumerate(casos):
        # 120 de los 200 ms. El desfase crece con el tiempo, asi que mostrar
        # todo haria ver la curva de abajo como ruido contra ruido; con 120 ms se
        # ve el punto importante: arranca pegado y se va separando de a poco.
        n = np.searchsorted(t, 120)
        # Real: linea gruesa y llena (es la verdad). Prediccion: mas fina y
        # punteada, dibujada ENCIMA, para que donde coinciden se vea la de abajo
        # asomando. Si las dos fueran llenas, la de arriba taparia a la otra y no
        # se distinguiria "coincide" de "esta tapada".
        ax[k].plot(t[:n], real[:n], lw=2.2, color=AZUL, label="lo que hace el cerebro")
        ax[k].plot(t[:n], pred[:n], lw=1.6, ls="--", color=NARA,
                   label="lo que predice el modelo, solo")
        e = err["sin" if k == 0 else "con"]
        ax[k].set_title(f"{titulo}   —   error {e:.0f} %", fontsize=11)
        ax[k].set_ylabel("actividad  E")
        ax[k].grid(alpha=0.5, lw=0.6)
        # Leyenda solo en el primer panel: los colores significan lo mismo en los
        # dos, repetirla seria ruido.
        if k == 0:
            ax[k].legend(fontsize=9, frameon=False, loc="upper right")
    ax[1].set_xlabel("tiempo (ms)")
    # El suptitle explica LAS REGLAS DEL JUEGO, no el resultado. Sin esta frase
    # cualquiera supone que el modelo va corrigiendose con el dato real en cada
    # paso, y entonces un 15 % de error parece malisimo en lugar de razonable.
    # Va en gris y chico a proposito: es una aclaracion, no el titulo.
    fig.suptitle("Se le da sólo el arranque y el estímulo, y tiene que generar "
                 "todo lo demás por su cuenta", fontsize=10.5, color=GRIS, y=1.0)
    fig.tight_layout()
    guarda(fig, "fig5_reproduccion")


# =============================================================================
#  FIG 6 — Acierta el tamano, erra la forma
# =============================================================================
def fig_forma():
    """Pregunta: la red aprendio la fisica que falta, o solo tapo el error?

    Es el hallazgo mas fino del trabajo y por eso va casi al final. Se enfrentan
    dos curvas en las MISMAS unidades:
      - dfE = la fisica que de verdad falta, o sea (campo real - campo Wilson-
        Cowan nominal) en la componente dE/dt, evaluada sobre la trayectoria
        medida. Es el "oraculo": se puede calcular porque la perturbacion del
        simulador se conoce, cosa que con datos reales nunca pasa.
      - g_phi = lo que la red efectivamente aprendio.
    Resultado: la amplitud coincide (0.0129 contra 0.0172 de RMS) pero la FORMA
    esta desfasada, y el R2 contra la fisica real es NEGATIVO (-1.21 en test), o
    sea peor que predecir cero. Traduccion: la red ayuda a bajar el error de los
    parametros sin haber descubierto la fisica; es ~90 % redundante con theta.

    Por que esta figura importa: el ajuste (fig 4) decia "la red sirve". Esta
    dice "sirve por la razon equivocada", y de ahi sale directamente la fig 7,
    donde por eso mismo no se la puede usar para cancelar en el controlador.
    """
    import torch
    from src.neural_ode import GrayBoxWC
    from src.neural_ode.graybox_train import ALL_P
    # Se usa el checkpoint de la variante B, o sea g(I,E), la correccion que NO
    # ve el estimulo. Es la comparacion mas justa contra la fisica real: la
    # variante que si ve P,Q (A) es un aproximador universal sobre el mismo
    # dominio que el backbone y puede compensar cualquier theta equivocado.
    ck = torch.load("results/uncertainty/models/f3_B_eps1.pt",
                    map_location="cpu", weights_only=False)
    # Los 1.0 son un ANDAMIO: hay que construir el objeto con la misma
    # arquitectura antes de poder cargarle los pesos, y load_state_dict pisa
    # todos esos valores. No son parametros con significado, no leerlos como
    # tales. Lo que si tiene que coincidir con el checkpoint es la ARQUITECTURA,
    # y de ahi que correction_inputs se lea del propio archivo.
    m = GrayBoxWC({k: 1.0 for k in ALL_P},
                  {k: 1.0 for k in ("wEE", "wEI", "wIE", "wII")},
                  learnable_weights=True, learnable_params=True,
                  use_correction=True,
                  correction_inputs=ck.get("correction_inputs", "x"))
    m.load_state_dict(ck["state"]); m.eval()

    d = np.load("data/processed/uncertain/eps1.npz", allow_pickle=True)
    # Mismo criterio que la fig 5: el primer escenario de TEST.
    it = d["is_test"].astype(bool)
    s = int(np.where(it)[0][0])
    t = d["t"]
    # g se evalua sobre la trayectoria REAL medida (no sobre una simulada): la
    # pregunta es "en los estados por los que el sistema realmente pasa, que
    # dice la red y que dice la fisica".
    X = torch.tensor(np.stack([d["I"][s], d["E"][s]], 1), dtype=torch.float32)
    # Los ceros de P,Q son legitimos SOLO porque este checkpoint es la variante
    # "x": g(I,E) ignora sus argumentos P,Q y g_out los descarta. TRAMPA: si
    # alguna vez se apunta esta figura a un checkpoint "xpq", estos ceros
    # dejarian de ser inocuos y la curva naranja seria la red evaluada sin
    # estimulo, o sea otra cosa. Habria que pasar d["P"][s] y d["Q"][s].
    z = torch.zeros(len(X), 1)
    with torch.no_grad():
        g = m.g_out(X, z, z).numpy()
    real = d["dfE"][s]
    # Ventana 12-100 ms. Se saltan los primeros 12 ms porque el transitorio
    # inicial tiene una excursion mucho mas grande que el regimen y, si entra,
    # aplasta todo el resto de la figura contra el eje. El corte en 100 deja ver
    # el desfase ciclo a ciclo, que es lo que hay que mostrar.
    n0, n = np.searchsorted(t, 12), np.searchsorted(t, 100)

    # Las metricas (RMS de cada curva y R2) se leen del json de la fase F5 y no
    # se recalculan aca: alli estan promediadas sobre todos los escenarios, y se
    # quiere que la figura y la tabla del informe digan el mismo numero.
    rec = {r["tag"]: r for r in jd("f5_recovery.json")}
    r = rec.get("f3_B_eps1", {})

    # Panel unico, ancho y bajo (11 x 3.8): el mensaje es un DESFASE temporal, y
    # un panel ancho estira el eje del tiempo, que es donde esta la evidencia.
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(t[n0:n], real[n0:n], lw=2.2, color=AZUL,
            label="la física que de verdad falta")
    # Columna 1 = la componente dE/dt de g (la 0 es dI/dt), para que se compare
    # contra dfE y no contra otra componente.
    ax.plot(t[n0:n], g[n0:n, 1], lw=1.8, color=NARA,
            label="lo que aprendió la red")
    ax.set_xlabel("tiempo (ms)")
    ax.set_ylabel("corrección sobre  dE/dt")
    ax.set_title("La red aprende algo del tamaño correcto, pero con la forma "
                 "equivocada", fontsize=11.5)
    ax.grid(alpha=0.5, lw=0.6)
    # Franja libre arriba para la leyenda y la nota, que si no se montan
    # sobre las curvas.
    # Las dos curvas son mayormente NEGATIVAS (la fisica que falta frena a E), de
    # modo que el piso lo fija el minimo y arriba queda lugar libre. Se aprovecha
    # esa asimetria: el techo se pone en una fraccion del piso en vez de en el
    # maximo de los datos. OJO: la cuenta supone lo < 0; si algun dia estas
    # curvas fueran positivas, el ylim saldria invertido.
    lo = min(real[n0:n].min(), g[n0:n, 1].min())
    ax.set_ylim(lo * 1.08, abs(lo) * 0.46)
    # ncol=2: la leyenda en una sola linea horizontal ocupa menos alto y deja el
    # renglon de abajo libre para la nota con los numeros.
    ax.legend(fontsize=9.5, frameon=False, loc="upper right", ncol=2)
    # La nota dice explicitamente lo que el ojo NO puede medir: que las dos
    # amplitudes son parecidas (por eso la red "ajusta bien") y que sin embargo
    # el R2 es negativo. Sin este texto el lector ve dos curvas del mismo tamano
    # y concluye lo contrario de lo que hay que concluir.
    ax.annotate(f"amplitud parecida  ({r.get('g_rms',0):.4f} contra "
                f"{r.get('df_rms',0):.4f}),  pero desfasada:\n"
                f"R² = {r.get('r2_test', 0):.2f}  —  peor que no corregir nada",
                xy=(0.015, 0.97), xycoords="axes fraction", fontsize=9,
                color=GRIS, va="top")
    fig.tight_layout()
    guarda(fig, "fig6_forma")


# =============================================================================
#  FIG 7 — Control
# =============================================================================
def fig_control():
    """Pregunta: sirve todo esto para CONTROLAR el sistema?

    Respuesta: los parametros si, la red no. Cuatro controladores contra la MISMA
    planta perturbada, ordenados de mejor conocimiento a peor:
      1. theta verdaderos (el oraculo: el techo alcanzable)
      2. white-box: los theta identificados sin red
      3a. gray-box: los theta que salieron de entrenar CON red, y despues la red
          se descarta               -> el mejor de los realizables
      3b. gray-box: lo mismo, y ademas evaluar g(I,E) y restarlo en cada paso
          (cancelacion por linealizacion por realimentacion)

    ==================== POR QUE VAN LOS DOS LAZOS ====================
    Esta es LA decision de diseno de la figura, y no es cosmetica: es la
    diferencia entre publicar la conclusion correcta y publicar la opuesta.

    Los numeros de 3a -> 3b:
        lazo de E:  0.0784 -> 0.0645   (MEJORA, y casi llega al oraculo 0.0629)
        lazo de I:  0.0550 -> 0.2064   (se DESTRUYE: casi cuatro veces peor)

    Si la figura mostrara solo E -- que es la tentacion natural, porque E es la
    poblacion "protagonista", la que se grafica en todas las demas figuras y la
    que define la salida y = E - I -- la lectura seria "restar la red mejora el
    control y lo acerca al optimo". Es exactamente lo contrario de la verdad. La
    correccion aprendida acierta lo suficiente en un canal como para ayudar ahi, y
    se equivoca lo suficiente en el otro como para arruinarlo; en el balance NO es
    usable para cancelar. Un sistema de dos poblaciones acopladas no se juzga por
    un canal: un lazo de I roto significa que la inhibicion quedo fuera de
    control, y el regimen entero del sistema depende de ella.

    Y no es un accidente ni mala suerte: es la consecuencia PREDICHA por la fig 6.
    La cancelacion por linealizacion usa a g SOLA y AISLADA, y la resta punto a
    punto del objetivo. El ajuste tolera que theta y g se repartan el trabajo de
    cualquier manera (lo que se compensa entre las dos partes no se ve en el
    error); la cancelacion no tolera nada de eso, porque se queda con una sola de
    las dos partes. Con R2 ~ 0 contra la fisica real, g apunta mal, y restar algo
    que apunta mal es peor que no restar nada.

    Moraleja general, que vale para cualquier gray-box orientado a control: que
    una correccion AJUSTE bien los datos no la habilita para CANCELAR. Son dos
    requisitos distintos, y el ajuste no avisa cuando el segundo no se cumple.
    Corolario para el codigo: cualquier metrica de lazo cerrado se reporta por
    canal. Un promedio de los dos lazos habria escondido esto igual que mirar
    solo E.
    ===================================================================

    Los numeros salen de f6_closed_loop.json con la correccion regularizada
    (variante D, lambda=1), que es la que MEJOR se porta. Se elige a proposito la
    mas favorable: la sin regularizar empeora los dos canales a la vez, y con esa
    la conclusion seria facil. El caso interesante es este.
    """
    d = jd("f6_closed_loop.json")["filas"]

    # Busqueda por PREFIJO del nombre y no por indice: el json lleva las filas
    # numeradas ("1. oraculo...", "3b. gray-box...") y asi la figura no se rompe
    # si se agrega o reordena un caso en el experimento.
    def buscar(pref):
        return next((f for f in d if f["nombre"].startswith(pref)), None)

    # El orden es narrativo, de "lo mejor posible" a "lo que se probo ultimo".
    # Se OMITE la fila "0." del json (planta limpia + theta verdaderos, 0.0313):
    # esa mide el costo del hueco en si, que ya es tema de la fig 2. Aca la
    # referencia es el oraculo SOBRE LA PLANTA PERTURBADA, para que las cuatro
    # barras se diferencien solo por el conocimiento del controlador y no por la
    # planta. Meter la fila 0 haria parecer que el problema es la identificacion
    # cuando la mitad del error es estructural e irreparable.
    # Las etiquetas estan en castellano llano: nadie de afuera sabe que es "3b".
    orden = [("1.", "θ verdaderos\n(lo mejor posible)"),
             ("2.", "sólo las ecuaciones\n(white-box)"),
             ("3a", "gray-box:\nusar los parámetros"),
             ("3b", "gray-box:\nrestar también la red")]
    nombres, vI, vE = [], [], []
    for pref, etiqueta in orden:
        # El "if f" tolera un json incompleto: si falta un caso, se dibuja con
        # las barras que haya en vez de reventar.
        f = buscar(pref)
        if f:
            nombres.append(etiqueta); vI.append(f["rmse_I"]); vE.append(f["rmse_E"])

    x = np.arange(len(nombres)); w = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.3))
    # Barras APAREADAS por controlador (no apiladas, no promediadas): la
    # comparacion que importa es I contra E DENTRO de cada controlador, porque el
    # hallazgo es que se mueven en direcciones opuestas. Apilarlas o promediarlas
    # borraria justamente eso.
    b1 = ax.bar(x - w / 2 - 0.01, vI, w, color=AZUL, label="lazo de I")
    b2 = ax.bar(x + w / 2 + 0.01, vE, w, color=NARA, label="lazo de E")
    # Valor sobre cada barra con 3 decimales: los RMSE viven entre 0.03 y 0.21, y
    # con menos decimales varios casos se verian iguales. Etiquetar las 8 barras
    # permite ademas leer los cocientes exactos sin ir al informe.
    for bb in list(b1) + list(b2):
        ax.annotate(f"{bb.get_height():.3f}",
                    (bb.get_x() + bb.get_width() / 2, bb.get_height()),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=9, color=TINTA)
    ax.set_xticks(x); ax.set_xticklabels(nombres, fontsize=9)
    ax.set_ylabel("error de seguimiento del controlador")
    ax.set_title("Para controlar: los parámetros sirven, la red no",
                 fontsize=11.5)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.grid(alpha=0.4, lw=0.6, axis="y")
    # Eje desde 0 (obligatorio en barras: si no, las diferencias se falsean) y con
    # 42 % de aire arriba para la nota que va sobre la barra mas alta.
    ax.set_ylim(0, max(vI + vE) * 1.42)
    # La nota va sobre la barra AZUL del cuarto grupo (la de I en 3b), que es la
    # que se dispara. Es el remate de la figura escrito en palabras: si el lector
    # solo mira este texto ya se lleva la conclusion correcta. El "if" cubre el
    # caso de que el json no traiga las 4 filas (ver el "if f" de arriba): sin el,
    # vI[3] seria un IndexError.
    if len(vI) >= 4:
        ax.annotate("mejora E,\npero DESTROZA I", xy=(3 - w / 2, vI[3]),
                    xytext=(3 - w / 2, vI[3] * 1.16), fontsize=9.5,
                    color=ROJO, ha="center", fontweight="bold")
    fig.tight_layout()
    guarda(fig, "fig7_control")


# =============================================================================
#  FIG 8 — Esquema: que es la planta y que usa el controlador
# =============================================================================
def fig_esquema_control():
    """La confusion tipica: creer que el modelo aprendido es la planta. No lo es.
    La planta es SIEMPRE el cerebro real; lo que cambia es con que conocimiento
    se arma el controlador.

    Es la UNICA figura esquematica del juego (todas las demas son datos medidos).
    Existe porque la fig 7 es imposible de leer sin ella: al ver "gray-box: restar
    tambien la red" la gente entiende "se reemplaza el cerebro por la red", y
    entonces no se explica por que el resultado puede ser malo. El esquema deja
    fijo que la planta no cambia nunca y que lo unico que se mueve entre las
    filas de la tabla es el contenido del controlador.

    Va PRIMERA en el __main__ aunque se llame "fig 8": se genera antes por
    comodidad al armar el PDF, no porque el orden importe.

    Como esta hecho: ejes apagados y un lienzo de coordenadas propias 100 x 62,
    o sea todas las posiciones de las cajas y flechas se leen como porcentaje del
    ancho. Es un dibujo a mano; si se cambia el tamano de una caja hay que
    reacomodar las flechas, que llevan las coordenadas escritas.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

    # Una caja del diagrama: titulo arriba, aclaracion chica abajo, borde de
    # color y relleno del mismo tono pero muy lavado. Las fracciones 0.66 y 0.27
    # de la altura reparten los dos textos con el peso visual en el titulo.
    def caja(x, y, w, h, titulo, sub, color, relleno):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.6,rounding_size=1.2",
                     linewidth=1.8, edgecolor=color, facecolor=relleno))
        ax.text(x + w / 2, y + h * 0.66, titulo, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=color)
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
                fontsize=8.6, color=GRIS, linespacing=1.35)

    # Flecha quebrada que pasa por una lista de puntos. Se dibuja como varios
    # segmentos y la punta se pone SOLO en el ultimo, para que un recorrido en L
    # (como la realimentacion, que baja, cruza por abajo y sube) no quede con una
    # punta de flecha en cada esquina.
    def flecha(pts, color=GRIS):
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            ultimo = (x2, y2) == pts[-1]
            ax.add_patch(FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle=("-|>,head_width=3.2,head_length=6.5" if ultimo else "-"),
                linewidth=1.6, color=color, shrinkA=0, shrinkB=0))

    # --- El lazo ---------------------------------------------------------
    # Azul = lo que se puede cambiar; naranja = lo que esta dado y no se toca. El
    # texto de cada caja repite esa idea por escrito, para no depender del color.
    caja(9, 38, 26, 15, "CONTROLADOR", "lo único que cambia\nentre las 3 filas",
         AZUL, "#eaf2fc")
    caja(60, 38, 32, 15, "PLANTA = el cerebro real",
         "el simulador con la física\nagregada · SIEMPRE la misma", NARA, "#fdeee7")

    flecha([(36.2, 45.5), (58.8, 45.5)])
    ax.text(47.5, 47.2, "estímulo  P, Q", ha="center", fontsize=9.2, color=TINTA)

    # realimentacion: baja de la planta, va por abajo, sube al controlador
    flecha([(76, 36.8), (76, 31.5), (22, 31.5), (22, 36.8)])
    # Lo que vuelve son I y E MEDIDAS, no la salida y = E - I: el controlador es de
    # estado completo. Que se realimenten las dos poblaciones es lo que hace que
    # tenga sentido hablar de "lazo de I" y "lazo de E" por separado en la fig 7.
    ax.text(49, 32.7, "actividad medida  I, E", ha="center", fontsize=9.2,
            color=TINTA)

    flecha([(22, 60), (22, 54.2)])
    ax.text(24, 58.2, "referencia: el ritmo que se le quiere imponer",
            ha="left", fontsize=9.2, color=TINTA, va="center")

    ax.text(93, 45.5, "La red NUNCA\nreemplaza al cerebro", ha="right",
            va="center", fontsize=8.4, color=NARA, style="italic",
            linespacing=1.3, alpha=0)      # reservado; el texto va en el epigrafe

    # --- Que se le pone adentro al controlador ---------------------------
    # Linea divisoria: arriba el lazo (el "que es que"), abajo la leyenda de las
    # tres filas. Separa el dibujo de su explicacion sin necesitar dos figuras.
    ax.plot([4, 96], [26.5, 26.5], color=SUAVE, lw=1)
    ax.text(4, 22.5, "Lo único que cambia entre las tres filas de la tabla es "
                     "qué se le pone adentro al controlador:",
            fontsize=9.6, fontweight="bold", color=TINTA, va="center")

    # Las tres filas, con un chip de color a la izquierda que anticipa el
    # veredicto de la fig 7: gris = la referencia, VERDE = lo que funciona
    # (usar los parametros del gray-box y tirar la red), ROJO = lo que arruina un
    # canal (restar la red en cada paso). El color esta puesto para que el lector
    # llegue a la fig 7 con la expectativa ya formada.
    filas = [
        ("white-box", "los 10 parámetros identificados sin usar la red", GRIS),
        ("gray-box, sólo los parámetros",
         "los 10 parámetros que salieron de entrenar CON la red — y después la "
         "red se descarta", VERDE),
        ("gray-box, restando la red",
         "lo mismo, y además evaluar ĝ(I,E) en cada paso y restarlo del objetivo",
         ROJO),
    ]
    for k, (nom, desc, col) in enumerate(filas):
        # y=16.5 es el primer renglon y 6.2 el paso entre filas, medidos en el
        # lienzo de 62 de alto: las tres entran justo entre la linea divisoria
        # (26.5) y el borde de abajo.
        y = 16.5 - k * 6.2
        ax.add_patch(FancyBboxPatch((5, y - 2.1), 1.2, 4.2,
                     boxstyle="round,pad=0,rounding_size=.5",
                     linewidth=0, facecolor=col))
        ax.text(8.5, y + 1.2, nom, fontsize=9.4, color=col,
                fontweight="bold", va="center")
        ax.text(8.5, y - 1.7, desc, fontsize=8.9, color=GRIS, va="center")

    ax.set_title("Quién es quién en el lazo de control", fontsize=12.5,
                 color=TINTA, pad=6)
    fig.tight_layout()
    guarda(fig, "fig8_esquema_control")


# =============================================================================
#  EJECUCION DIRECTA: regenera las 8 figuras del PDF de resumen.
# =============================================================================
# Requisitos previos: que existan los results/uncertainty/*.json y los
# data/processed/uncertain/eps*.npz (los produce scripts/run_uncertainty_all.sh).
# Hay que correrlo DESDE LA RAIZ del repo (ver la nota del encabezado).
#
# Las dos ultimas (fig 5 y fig 6) son las unicas que cargan torch y modelos, asi
# que si algo tarda es ahi; las seis primeras son solo lectura de json.
if __name__ == "__main__":
    print("Generando figuras del resumen:")
    fig_esquema_control()
    fig_perturbacion()
    fig_costo()
    fig_geometria()
    fig_cruce()
    fig_reproduccion()
    fig_forma()
    fig_control()
    print(f"\n-> {OUT}/")
