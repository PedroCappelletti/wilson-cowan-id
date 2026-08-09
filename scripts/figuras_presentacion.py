#!/usr/bin/env python3
# =============================================================================
#  FIGURAS PARA LA PRESENTACION  (formato apaisado, tipografia grande)
# =============================================================================
#
#  Las del PDF estan pensadas para leerse de cerca. En una presentacion hay que
#  verlas de lejos: menos elementos, letra mas grande, una idea por figura.
#
#  Ademas genera lo que faltaba: el COMPORTAMIENTO EN LAZO CERRADO (las
#  trayectorias, no solo las barras de error).
#
#  USO:  python scripts/figuras_presentacion.py
# =============================================================================
#
#  POR QUE ES OTRO ARCHIVO Y NO UN FLAG DE scripts/figuras_resumen.py
#  ------------------------------------------------------------------
#  Son las MISMAS corridas pero otro medio, y el medio cambia el diseño, no solo
#  el tamaño de la letra:
#    - se ven a 3-5 metros y por pocos segundos -> cada figura tiene que
#      contestar UNA pregunta sola; lo que no se lee de lejos se saca.
#    - se recortan titulos, se cortan sufijos de las etiquetas y se bajan de dos
#      paneles a uno donde el segundo panel era "detalle" (p.ej. geometria, que
#      en el PDF tiene tambien la curva de % imitable).
#    - las anotaciones (×17 PEOR, 48 % MEJOR, "DESTROZA I") reemplazan al
#      epigrafe: en la diapositiva no hay texto al pie que explique.
#    - figsize apaisado (~11-14 de ancho) porque el lienzo es 16:9, no A4.
#  Mantener los dos archivos separados evita que un ajuste para la charla
#  degrade el PDF (y al reves). El precio es duplicacion: si cambia el NOMBRE de
#  una clave de los .json hay que tocar los dos.
#
#  DE DONDE SALEN LOS DATOS (nada es esquematico ni dibujado a mano)
#  ----------------------------------------------------------------
#    results/uncertainty/*.json          metricas de los experimentos F2..F6
#    results/uncertainty/models/*.pt     checkpoints de los gray-box entrenados
#    data/processed/uncertain/eps*.npz   trayectorias del simulador con la
#                                        fisica agregada (la perilla eps)
#  Son rutas RELATIVAS: hay que correr el script desde la raiz del repo.
#
#  QUE PREGUNTA CONTESTA CADA FIGURA (el orden del final del archivo es el orden
#  del guion de la charla):
#    p_perturbacion   que le agregamos al simulador, y se nota?
#    p_costo          cuanto se paga por tener un modelo incompleto?
#    p_geometria      de que esta hecho ese hueco: se disfraza de parametros o
#                     es fisica nueva?
#    p_cruce          la red de correccion, ayuda o arruina? (depende del hueco)
#    p_forma          lo que aprendio la red, se parece a la fisica que falta?
#    p_lazo_abierto   el modelo, corriendo solo, reproduce al "cerebro"?
#    p_lazo_cerrado   y para CONTROLAR, sirve?  (trayectorias)
#    p_control        lo mismo resumido en barras, los dos lazos juntos
#  El prefijo "p_" es de "presentacion" y separa estas de las del PDF
#  (docs/figuras_resumen/fig*.png) y de las de la copia (c_*).
#
#  Referencia conceptual: docs/graybox_manual_completo.md
# =============================================================================

from __future__ import annotations

import json
import sys
from pathlib import Path

# El script vive en scripts/ pero importa src/: se agrega la raiz del repo al
# path ANTES de cualquier "from src...". Por eso los imports de numpy y de src
# quedan mas abajo y no arriba con los de la stdlib.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib
# Backend sin ventana: esto corre por consola y solo escribe PNG. Va ANTES de
# importar pyplot, si no queda fijado el backend interactivo.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("results/uncertainty")          # de donde se leen las metricas
OUT = Path("docs/figuras_presentacion")    # donde se escriben los PNG

# --- Paleta VALIDADA (misma que figuras_resumen.py; checks de contraste ya
#     corridos). NO cambiarla: los colores tienen significado asignado y las
#     figuras de la charla y del PDF tienen que leerse como un solo sistema.
#       AZUL/NARA  el par de series (1 y 2). En todas las figuras AZUL es "lo
#                  que asume el modelo / lo verdadero / el lazo de I" y NARA es
#                  "lo real medido / lo aprendido / el lazo de E".
#       AQUA       tercera serie; queda por debajo de 3:1 sobre blanco, asi que
#                  solo se usa con etiqueta directa (aca no se usa).
#       ROJO/VERDE colores de ESTADO (malo / bueno), no de serie: aparecen para
#                  juzgar un resultado, nunca para distinguir dos curvas
#                  cualesquiera.
#       TINTA/GRIS jerarquia de texto (titulo / secundario).
#       SUAVE      grillas y bordes de eje: tiene que verse menos que los datos.
AZUL = "#2a78d6"
NARA = "#eb6834"
AQUA = "#1baf7a"
ROJO = "#e34948"
VERDE = "#008300"
TINTA = "#0b0b0b"
GRIS = "#52514e"
SUAVE = "#dcdcd8"

# --- Estilo global "para ver de lejos".
#     Los tamaños son ~1.4x los del PDF (que usa font.size 10.5): a distancia de
#     proyector es el minimo que se lee. Lo demas es quitar tinta que no es dato:
#     sin bordes arriba/derecha, grilla clarita y linea gruesa (2.6) para que la
#     curva siga siendo visible aunque el proyector lave los contrastes.
#     Fondo blanco explicito porque las diapositivas son blancas: si quedara
#     transparente, los ejes grises se perderian sobre cualquier otro fondo.
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 14,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.edgecolor": SUAVE,
    "axes.labelcolor": GRIS,
    "axes.titlecolor": TINTA,
    "text.color": TINTA,
    "xtick.color": GRIS, "ytick.color": GRIS,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": SUAVE, "grid.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "lines.linewidth": 2.6,
})


# Guarda la figura y la cierra (si no, matplotlib acumula figuras y avisa).
# bbox_inches="tight" recorta el margen sobrante: por eso los figsize de abajo
# valen como PROPORCION, no como tamaño final. dpi=150 alcanza para proyectar
# (el PDF usa 170, que ahi si se mira de cerca). facecolor="white" se repite
# porque bbox_inches="tight" rearma el lienzo y no siempre hereda el rcParam.
def guarda(fig, nombre):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{nombre}.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  {nombre}.png")


# Atajo para leer un json de resultados (RES / nombre). Nombre corto porque se
# usa en casi todas las figuras.
def jd(n):
    return json.loads((RES / n).read_text())


# =============================================================================
#  LAZO ABIERTO — el modelo corriendo solo
# =============================================================================
#  Pregunta: si al modelo identificado le damos SOLO el arranque y el estimulo,
#  reproduce lo que hizo el "cerebro"? Es la prueba dura: no hay realimentacion
#  que lo reencauce, el error se acumula.
#
#  Dos paneles = el contraste de toda la charla: arriba el modelo COMPLETO
#  (eps=0, la fisica del simulador es la misma que la del modelo), abajo el
#  modelo al que le FALTA fisica (eps=1). Misma escala de tiempo y de eje.
def lazo_abierto():
    # torch y src se importan adentro para que las figuras que solo leen json
    # (perturbacion, costo, geometria, cruce) no paguen el arranque de torch.
    import torch
    from src.neural_ode import GrayBoxWC, rollout
    from src.neural_ode.graybox_train import ALL_P
    torch.set_num_threads(4)

    # Arma un white-box con los 10 parametros YA identificados (no entrena
    # nada). Los flags learnable_* se dejan como en el entrenamiento para que la
    # parametrizacion interna (raw_*) sea la misma; en eval + no_grad no cambian
    # el resultado.
    def modelo(params):
        m = GrayBoxWC(params, {k: params[k] for k in ("wEE", "wEI", "wIE", "wII")},
                      learnable_weights=True, learnable_params=True)
        m.eval(); return m

    casos = []
    for jf, npz, titulo, err in [
        ("f2_eps0.json", "eps0.npz", "Modelo COMPLETO", None),
        ("f2_eps1.json", "eps1.npz", "Al modelo le FALTA física", None),
    ]:
        d = np.load(f"data/processed/uncertain/{npz}", allow_pickle=True)
        m = modelo(jd(jf)["params"])
        # PRIMER escenario de TEST: nunca lo vio el entrenamiento, asi que el
        # ajuste que se ve no es memoria. Se elige el primero y no el "mejor"
        # para no maquillar la figura.
        s = int(np.where(d["is_test"].astype(bool))[0][0])
        T = d["I"].shape[1]
        with torch.no_grad():
            # Free-run: unica informacion que recibe el modelo = estado inicial
            # + las series de estimulo. Se pasa Ps[:-1]/Qs[:-1] porque con T
            # muestras hay T-1 pasos de integracion (el estimulo del ultimo
            # instante ya no se usa).
            x0 = torch.tensor([[d["I"][s, 0], d["E"][s, 0]]], dtype=torch.float32)
            Ps = torch.tensor(d["P"][s], dtype=torch.float32).reshape(T, 1, 1)
            Qs = torch.tensor(d["Q"][s], dtype=torch.float32).reshape(T, 1, 1)
            pred = rollout(m, x0, Ps[:-1], Qs[:-1], float(d["dt"]))[:, 0, :].numpy()
        # El estado es [I, E] -> la columna 1 es E, que es la que se grafica.
        casos.append((titulo, d["t"], d["E"][s], pred[:, 1]))

    # Los porcentajes NO se recalculan aca: se toman de reproduccion.json, que
    # los promedia sobre TODOS los escenarios de test. O sea, el numero del
    # titulo es del conjunto y la curva es de un escenario: la curva ilustra, el
    # numero cuantifica. Si algun dia se quiere que coincidan hay que calcular el
    # nrmse del escenario dibujado.
    rep = {r["modelo"]: r for r in jd("reproduccion.json")}
    errs = [rep["white-box, planta SIN hueco"]["nrmse_E"],
            rep["white-box, planta CON hueco"]["nrmse_E"]]

    fig, ax = plt.subplots(2, 1, figsize=(13.5, 7), sharex=True)
    for k, (titulo, t, real, pred) in enumerate(casos):
        # Solo los primeros 120 ms de los 200 del dataset: alcanzan para ver el
        # desfase y de lejos mas ciclos serian una mancha.
        n = np.searchsorted(t, 120)
        ax[k].plot(t[:n], real[:n], color=AZUL, label="el cerebro")
        ax[k].plot(t[:n], pred[:n], color=NARA, ls="--", lw=2.2,
                   label="el modelo, corriendo solo")
        # El titulo hace de epigrafe: nombre del caso + su error, alineado a la
        # izquierda para que se lea junto con el eje y no centrado en el aire.
        ax[k].set_title(f"{titulo}   ·   error {errs[k]:.0f} %",
                        fontsize=17, loc="left")
        ax[k].set_ylabel("actividad  E")
        ax[k].grid(alpha=0.6)
        # 30 % de aire arriba: es donde va la leyenda, si no se monta sobre los
        # picos (el eje se recorta al pico, no al maximo del dataset completo).
        ax[k].set_ylim(top=max(real[:n].max(), pred[:n].max()) * 1.3)
        if k == 0:
            # Una sola leyenda para los dos paneles (mismos colores abajo): dos
            # leyendas iguales serian ruido.
            ax[k].legend(frameon=False, loc="upper right", ncol=2)
    ax[1].set_xlabel("tiempo (ms)")
    fig.tight_layout()
    guarda(fig, "p_lazo_abierto")


# =============================================================================
#  LAZO CERRADO — el controlador tratando de imponer un ritmo
# =============================================================================
def lazo_cerrado():
    """Lo que faltaba: las TRAYECTORIAS del lazo cerrado, no solo el error.

    Las barras de p_control dicen QUE pasa; esto muestra COMO: en la columna
    verde la actividad sigue la referencia, en la roja el lazo de I se despega.
    Es la unica figura que se calcula corriendo el lazo aca (las demas leen
    resultados ya guardados), asi que es la mas lenta.

    Lo que NO cambia entre las dos columnas: la planta (el cerebro real con
    eps=1), la referencia y los topes del actuador. Lo unico que cambia es que
    se le pone adentro al controlador. Sin eso la comparacion no mediria la
    correccion sino otra cosa.
    """
    import torch
    from src.wilson_cowan import WilsonCowanParams, default_uncertainty, NoPerturbation
    from src.neural_ode import (GrayBoxWC, IMCController, simulate_closed_loop,
                                theta_gamma_refs, make_perturbed_plant,
                                make_graybox_correction)
    from src.neural_ode.graybox_train import ALL_P
    torch.set_num_threads(4)

    p = WilsonCowanParams()                      # los parametros VERDADEROS
    tp = {k: getattr(p, k) for k in ALL_P}       # los mismos, como dict
    w_true = {k: tp[k] for k in ("wEE", "wEI", "wIE", "wII")}

    # Arma el bloque "fixed" que consume IMCController. ke y ki NO son
    # parametros libres: son la sigmoidea en reposo, o sea una funcion de
    # (a, theta) — la misma formula que en src/wilson_cowan/model.py. Hay que
    # recalcularlos con los a/theta ESTIMADOS; mezclar los estimados con los ke,
    # ki verdaderos seria filtrarle al controlador informacion que no tiene.
    def fixed_de(q):
        ae, ai, the, thi = q["ae"], q["ai"], q["thetae"], q["thetai"]
        return {"te": q["te"], "ti": q["ti"], "ae": ae, "ai": ai,
                "thetae": the, "thetai": thi,
                "ke": 1 / (1 + np.exp(ae * the)), "ki": 1 / (1 + np.exp(ai * thi))}

    # Topes de actuacion COMPARTIDOS, calculados con los parametros verdaderos.
    # Representan hasta donde puede empujar el estimulador (una propiedad del
    # equipo, no del modelo). Si cada controlador se armara sus propios topes con
    # sus parametros estimados, el que estimo un `ai` mas chico tendria mas
    # autoridad de control y la figura mediria eso. Ver la nota de sat_ref en
    # src/neural_ode/closed_loop.py.
    SAT = fixed_de(tp)
    # Referencia theta-gamma del controlador del tutor: 120 Hz con t en ms, o sea
    # un ciclo cada ~8.3 ms.
    refs = theta_gamma_refs(freq_hz=120.0, time_in_ms=True)
    # LA PLANTA ES EL CEREBRO REAL: Wilson-Cowan + la fisica agregada a eps=1
    # (refractariedad + actuador con retardo y saturacion). Nunca es el modelo
    # aprendido; el modelo solo alimenta al controlador.
    planta = make_perturbed_plant(p, default_uncertainty(1.0))

    # Gray-box de referencia: variante D (penaliza que g se mueva en las
    # direcciones que un cambio de parametros ya podria explicar) con lam=1.
    ck = torch.load("results/uncertainty/models/f3_D_eps1_lam1.pt",
                    map_location="cpu", weights_only=False)
    # Se reconstruye el modelo con todo en 1.0 y despues load_state_dict pisa los
    # valores: lo unico que tiene que coincidir es la ARQUITECTURA. use_correction
    # y correction_inputs se leen del checkpoint (con .get, por si es viejo y no
    # los guardo) porque si no coinciden, load_state_dict falla o carga otra red.
    m = GrayBoxWC({k: 1.0 for k in ALL_P},
                  {k: 1.0 for k in ("wEE", "wEI", "wIE", "wII")},
                  learnable_weights=True, learnable_params=True,
                  use_correction=ck.get("use_correction", False),
                  correction_inputs=ck.get("correction_inputs", "x"))
    m.load_state_dict(ck["state"]); m.eval()
    ph = ck["params"]     # los 10 parametros identificados por el gray-box

    # Un lazo cerrado: t en ms, dt=0.005 ms. El paso es chico porque la planta
    # tiene el actuador con tau ~1 ms y la referencia va a 120 Hz; son los mismos
    # t_span y dt de exp_f6_closed_loop.py, para poder comparar numeros.
    def corre(fx, wh, g_hat=None):
        c = IMCController(fx, wh, g_hat=g_hat, sat_ref=SAT)
        return simulate_closed_loop(planta, c, refs, t_span=(0.0, 50.0), dt=0.005)

    # Las dos columnas. El color YA anticipa la conclusion (verde = anda,
    # rojo = rompe): son los colores de ESTADO de la paleta, no dos series.
    #   izquierda: el controlador usa solo los parametros del gray-box y la red
    #              se descarta.
    #   derecha:   ademas evalua g_φ(I,E) en cada paso y lo resta del objetivo.
    escenarios = [
        ("Con los parámetros del gray-box", fixed_de(ph),
         {k: ph[k] for k in w_true}, None, VERDE),
        ("Restando además la red", fixed_de(ph),
         {k: ph[k] for k in w_true}, make_graybox_correction(m), ROJO),
    ]

    # Grilla 2x2: filas = poblacion (I arriba, E abajo), columnas = escenario.
    # Asi se compara "misma poblacion, otro controlador" mirando al costado, que
    # es la lectura que interesa.
    fig, ax = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
    for col, (titulo, fx, wh, g, color) in enumerate(escenarios):
        sol = corre(fx, wh, g)
        t = sol["t"]
        n0 = np.searchsorted(t, 10)      # se descarta el transitorio inicial
        # Por que 10 ms: el integrador del PI arranca en Z=0 y tarda en cargar;
        # ese pico inicial no dice nada del modelo y ademas domina la escala.
        # Coincide con el criterio de exp_f6 (descarta el primer 20 % de 50 ms).
        for row, var in enumerate(["I", "E"]):
            a = ax[row][col]
            # Referencia punteada y en gris: es el objetivo, no un dato medido.
            a.plot(t[n0:], sol["r" + var][n0:], color=GRIS, lw=2.0, ls=":",
                   label="lo que se le pide (referencia)")
            a.plot(t[n0:], sol[var][n0:], color=color, lw=2.6,
                   label="lo que hace el cerebro")
            a.grid(alpha=0.6)
            a.set_ylabel(f"población {var}")
            if row == 0:
                # El titulo va del color de la columna: de lejos el color es lo
                # primero que se lee y ya dice si el caso es el bueno o el malo.
                a.set_title(titulo, fontsize=16, color=color, loc="left")
            if row == 1:
                a.set_xlabel("tiempo (ms)")
            if row == 0 and col == 0:
                # Leyenda una sola vez: los cuatro paneles usan el mismo codigo.
                a.legend(frameon=False, fontsize=12, loc="upper right", ncol=2)
        # anotar el error de cada lazo
        # RMS calculado sobre la MISMA ventana dibujada, asi el numero explica lo
        # que se ve. No tiene por que coincidir al ultimo decimal con
        # f6_closed_loop.json, que promedia con su propio criterio de recorte.
        for row, var in enumerate(["I", "E"]):
            e = float(np.sqrt(np.mean((sol[var][n0:] - sol["r" + var][n0:]) ** 2)))
            # caja blanca: si no, la etiqueta se monta sobre la curva
            # Posicion en coordenadas de ejes (0-1), abajo a la izquierda: el
            # unico rincon que las senoides de la referencia dejan libre.
            ax[row][col].text(0.012, 0.04, f"error = {e:.3f}",
                              transform=ax[row][col].transAxes, fontsize=13,
                              color=color, fontweight="bold",
                              bbox=dict(facecolor="white", edgecolor="none",
                                        alpha=0.85, pad=2.5))
    # misma escala vertical por fila, para poder comparar columnas
    # Es imprescindible: con ejes autoescalados, un lazo que se despega se
    # dibuja "prolijo" en su propia escala y la figura mentiria. Se toma la
    # union de los limites, nunca la interseccion (recortaria datos).
    for row in range(2):
        lo = min(ax[row][c].get_ylim()[0] for c in range(2))
        hi = max(ax[row][c].get_ylim()[1] for c in range(2))
        for c in range(2):
            ax[row][c].set_ylim(lo, hi)
    fig.tight_layout()
    guarda(fig, "p_lazo_cerrado")


# =============================================================================
#  Las de siempre, en formato presentacion
# =============================================================================
#  De aca abajo estan las mismas figuras del PDF (docs/figuras_resumen/), pero
#  recortadas para proyectar: un panel en vez de dos donde el segundo era
#  detalle, titulos mas cortos y las conclusiones escritas sobre la figura.
#  Todas leen resultados ya calculados, asi que son instantaneas.


# Pregunta: que le agregamos al simulador y se nota? Dos sintomas del mismo
# cambio: el ciclo se deforma (izquierda) y el estimulo no llega como se manda
# (derecha, el actuador con retardo y saturacion).
def perturbacion():
    d0 = np.load("data/processed/uncertain/eps0.npz", allow_pickle=True)
    d1 = np.load("data/processed/uncertain/eps1.npz", allow_pickle=True)
    # El escenario que MAS oscila: es donde se ve que el ciclo se deforma sin
    # desaparecer, que es justo el mensaje (la fisica agregada no rompe el
    # ritmo, lo corre). Se elige por std de E, no a dedo.
    s = int(np.argmax(d0["E"].std(axis=1)))
    t = d0["t"]; n = np.searchsorted(t, 110)   # 110 ms: se ve el detalle ciclo a ciclo
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    ax[0].plot(t[:n], d0["E"][s, :n], color=AZUL, label="lo que el modelo asume")
    ax[0].plot(t[:n], d1["E"][s, :n], color=NARA, label="el cerebro real")
    # El titulo es la conclusion, no la descripcion del eje: en la charla el
    # titulo es lo unico que se lee seguro.
    ax[0].set_title("El ciclo se deforma, no desaparece", loc="left")
    ax[0].set_xlabel("tiempo (ms)"); ax[0].set_ylabel("actividad  E")
    ax[0].set_ylim(top=max(d0["E"][s, :n].max(), d1["E"][s, :n].max()) * 1.38)
    ax[0].legend(frameon=False, loc="upper right"); ax[0].grid(alpha=0.6)

    # P es el estimulo COMANDADO (lo unico que ve el modelo al entrenar) y P_eff
    # el que realmente llega a la poblacion. P_eff se guarda solo como
    # diagnostico: si el modelo lo viera, el hueco desapareceria por construccion.
    ax[1].plot(t[:n], d1["P"][s, :n], color=AZUL, label="lo que comandás")
    ax[1].plot(t[:n], d1["P_eff"][s, :n], color=NARA, label="lo que llega")
    ax[1].set_title("El estímulo no llega como lo mandaste", loc="left")
    ax[1].set_xlabel("tiempo (ms)"); ax[1].set_ylabel("estímulo  P")
    # El aire de arriba se calcula sobre P (no sobre P_eff) porque el comandado
    # es el mas alto: el actuador satura, nunca amplifica.
    ax[1].set_ylim(top=d1["P"][s, :n].max() * 1.38)
    ax[1].legend(frameon=False, loc="upper right"); ax[1].grid(alpha=0.6)
    fig.tight_layout()
    guarda(fig, "p_perturbacion")


# Pregunta: cuanto cuesta tener un modelo incompleto? Barrido de la perilla eps
# (cuanta fisica le falta al modelo) contra el error con que se identifican los
# 10 parametros. Es la figura que justifica todo el resto.
def costo():
    # Se descubren los eps disponibles por glob y se ORDENAN por el valor de eps:
    # el orden alfabetico de los nombres pondria eps0.25 despues de eps0.5 y la
    # linea saldria en zigzag.
    filas = sorted([jd(p.name) for p in RES.glob("f2_eps*.json")],
                   key=lambda f: f["eps"])
    eps = [f["eps"] for f in filas]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    # Dos curvas y no las diez: el peor parametro y el promedio acotan al resto.
    # wII es SIEMPRE el peor (es el menos identificable: la autoinhibicion casi
    # no se ve en la salida), asi que va con su nombre puesto.
    ax.plot(eps, [f["param_errors"]["wII"] for f in filas], "o-", ms=11,
            color=NARA, label="el peor parámetro ($w_{II}$)")
    ax.plot(eps, [f["mean_param_error"] for f in filas], "o-", ms=11,
            color=AZUL, label="error medio de los 10")
    ax.set_xlabel("ε  ·  cuánta física le falta al modelo")
    ax.set_ylabel("error de identificación (%)")
    ax.grid(alpha=0.6); ax.legend(frameon=False, loc="lower right")
    # El "1 %" esta ESCRITO A MANO (el PDF en cambio lo formatea del json). Sale
    # de f2_eps0.json: mean_param_error = 1.05 %, el resultado ya publicado del
    # proyecto sin hueco. Si se reentrena F2 y ese numero cambia, esta etiqueta
    # miente: hay que actualizarla.
    # xytext esta en coordenadas de DATOS (eps=0.16, error=62 %), no en fraccion
    # de ejes: si cambia el rango del barrido, la flecha se corre.
    ax.annotate("1 %\nel resultado\nconocido", xy=(0, filas[0]["mean_param_error"]),
                xytext=(0.16, 62), fontsize=13, color=GRIS,
                arrowprops=dict(arrowstyle="->", color=GRIS, lw=1.4))
    fig.tight_layout()
    guarda(fig, "p_costo")


# Pregunta: de que esta hecho el hueco? Barras apiladas: la parte que un cambio
# de parametros podria imitar (y que por lo tanto ENVENENA la identificacion) y
# la parte que es fisica realmente nueva. En el PDF hay un segundo panel con la
# curva de % imitable; aca se sacrifica y el % se escribe adentro de la barra.
def geometria():
    g = jd("f4b_geometria.json")
    # Tolerancia al formato: las corridas viejas guardaban una lista suelta y las
    # nuevas un dict con la clave "por_eps".
    filas = g["por_eps"] if isinstance(g, dict) else g
    eps = [f["eps"] for f in filas]
    imit = np.array([f["rms_imitable"] for f in filas])
    nuevo = np.array([f["rms_nuevo"] for f in filas])
    # x = posiciones enteras (barras categoricas), no el valor de eps: el barrido
    # no es equiespaciado y las barras quedarian de anchos distintos.
    x = np.arange(len(eps))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x, imit, 0.62, color=NARA, label="se disfraza de parámetros")
    # El 0.0004 no es fisica: es una hendija (en unidades del eje y, que va en
    # ~1e-2) para que se vea la linea de corte entre los dos colores. Si cambiara
    # la escala del hueco habria que reajustarlo.
    ax.bar(x, nuevo, 0.62, bottom=imit + 0.0004, color=AZUL,
           label="física realmente nueva")
    # El porcentaje va DENTRO de la barra naranja, en blanco: de lejos se lee sin
    # tener que buscar el eje. Es la misma cantidad que frac_imitable del json,
    # recalculada aca desde las dos alturas.
    for i, (a, b) in enumerate(zip(imit, nuevo)):
        ax.text(i, a / 2, f"{100*a/(a+b):.0f}%", ha="center", va="center",
                color="white", fontsize=14, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"ε={e:g}" for e in eps])
    ax.set_ylabel("tamaño del hueco")
    ax.legend(frameon=False, loc="upper left"); ax.grid(alpha=0.5, axis="y")
    # Aire para la leyenda de arriba a la izquierda.
    ax.set_ylim(top=(imit + nuevo).max() * 1.25)
    fig.tight_layout()
    guarda(fig, "p_geometria")


# Pregunta: la red de correccion ayuda o arruina? LA figura de la charla: el
# resultado es que DEPENDE, y el cruce se ve de un golpe.
#   ε=0 (modelo ya correcto): la red solo agrega libertad -> ×17 PEOR.
#   ε=1 (al modelo le falta fisica): la red absorbe parte del hueco -> 48 % MEJOR.
def cruce():
    wb0 = jd("f2_eps0.json")["mean_param_error"]   # sin red, sin hueco
    wb1 = jd("f2_eps1.json")["mean_param_error"]   # sin red, con hueco
    # "Con la red" = la MEJOR de las variantes disponibles en cada eps, para no
    # ganar la comparacion eligiendo a mano la que conviene.
    gb0 = min(jd(f"f3_{v}_eps0.json")["mean_param_error"] for v in ("A", "B"))
    # Se excluye la variante S: no es una red sino una correccion ESTRUCTURADA
    # (tres parametros fisicos dentro del backbone). Meterla aca contestaria otra
    # pregunta; esta figura compara "sin red" contra "con red".
    gb1 = min(jd(p.name)["mean_param_error"] for p in RES.glob("f3_*eps1*.json")
              if "S_" not in p.name)
    fig, ax = plt.subplots(figsize=(11, 5.4))
    # w = ancho de barra; el .012 extra es la ranura entre las dos barras del
    # grupo (que se toquen las hace leer como una sola barra apilada).
    x = np.array([0, 1]); w = 0.34
    b1 = ax.bar(x - w / 2 - .012, [wb0, wb1], w, color=AZUL, label="sin red")
    b2 = ax.bar(x + w / 2 + .012, [gb0, gb1], w, color=NARA, label="con la red")
    # Valor encima de cada barra: en la charla nadie va a leer el eje.
    for b in list(b1) + list(b2):
        ax.annotate(f"{b.get_height():.1f} %",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 5), ha="center",
                    fontsize=14)
    ax.set_xticks(x)
    # La etiqueta del eje x explica que significa cada eps: en la diapositiva no
    # hay epigrafe donde aclararlo.
    ax.set_xticklabels(["ε = 0\nel modelo YA es correcto",
                        "ε = 1\nal modelo le FALTA física"], fontsize=14)
    ax.set_ylabel("error de identificación (%)")
    ax.legend(frameon=False, loc="upper left"); ax.grid(alpha=0.5, axis="y")
    ax.set_ylim(0, max(wb1, gb1) * 1.34)
    # Los dos veredictos, en los colores de ESTADO (rojo malo / verde bueno) y
    # calculados de los datos, no escritos a mano. Van sobre la barra naranja de
    # cada grupo (la de "con la red"), que es la que cambia. El "+5" es un
    # empujon en unidades de %, sobre el numero que ya esta encima de la barra.
    ax.annotate(f"×{gb0/wb0:.0f} PEOR", (w / 2, gb0 + 5), fontsize=17,
                fontweight="bold", color=ROJO, ha="center")
    ax.annotate(f"{100*(1-gb1/wb1):.0f} % MEJOR", (1 + w / 2, gb1 + 5),
                fontsize=17, fontweight="bold", color=VERDE, ha="center")
    fig.tight_layout()
    guarda(fig, "p_cruce")


# Pregunta: lo que aprendio la red, ES la fisica que falta? Se compara g_φ contra
# el Delta f verdadero (que el simulador guarda en dfE). Respuesta y titulo de la
# figura: acierta la amplitud y erra la fase, o sea aprendio "algo del tamaño
# correcto" pero no la fisica. Esto es lo que justifica que en el lazo cerrado
# restar la red no ayude.
def forma():
    import torch
    from src.neural_ode import GrayBoxWC
    from src.neural_ode.graybox_train import ALL_P
    # Variante B: g(I,E), sin regularizacion. Es la mas transparente para mirar
    # la FORMA de lo aprendido (en A la red tambien ve P,Q y puede explicar el
    # estimulo; en C/D la penalizacion ya la deforma a proposito).
    ck = torch.load("results/uncertainty/models/f3_B_eps1.pt",
                    map_location="cpu", weights_only=False)
    m = GrayBoxWC({k: 1.0 for k in ALL_P},
                  {k: 1.0 for k in ("wEE", "wEI", "wIE", "wII")},
                  learnable_weights=True, learnable_params=True,
                  use_correction=True,
                  correction_inputs=ck.get("correction_inputs", "x"))
    m.load_state_dict(ck["state"]); m.eval()
    d = np.load("data/processed/uncertain/eps1.npz", allow_pickle=True)
    s = int(np.where(d["is_test"].astype(bool))[0][0])
    t = d["t"]
    # g se evalua sobre la trayectoria MEDIDA (I,E reales), no sobre una
    # prediccion: se quiere ver la forma de la correccion, sin contaminarla con
    # el error de rollout.
    X = torch.tensor(np.stack([d["I"][s], d["E"][s]], 1), dtype=torch.float32)
    # Los ceros son los P,Q que g_out pide por firma pero IGNORA, porque este
    # modelo es correction_inputs="x". Con la variante A (g(I,E,P,Q)) pasar ceros
    # daria una curva sin sentido.
    z = torch.zeros(len(X), 1)
    with torch.no_grad():
        g = m.g_out(X, z, z).numpy()
    # Ventana 12-95 ms: se saca el arranque (donde el Delta f verdadero todavia
    # arrastra el transitorio del actuador) y se corta antes del final para que
    # de lejos se distingan los ciclos.
    n0, n = np.searchsorted(t, 12), np.searchsorted(t, 95)
    fig, ax = plt.subplots(figsize=(13.5, 4.8))
    ax.plot(t[n0:n], d["dfE"][s][n0:n], color=AZUL, label="la física que falta")
    ax.plot(t[n0:n], g[n0:n, 1], color=NARA, label="lo que aprendió la red")
    # Las dos curvas son NEGATIVAS (la correccion frena a E), asi que el aire
    # libre se pide arriba: lo tope se fija en una fraccion del minimo (0.42*|lo|)
    # para dejar una franja donde entren la leyenda y la nota sin taparlas.
    lo = min(d["dfE"][s][n0:n].min(), g[n0:n, 1].min())
    ax.set_ylim(lo * 1.1, abs(lo) * 0.42)
    ax.set_xlabel("tiempo (ms)"); ax.set_ylabel("corrección sobre  dE/dt")
    ax.legend(frameon=False, loc="upper right", ncol=2); ax.grid(alpha=0.6)
    # La conclusion escrita sobre la figura. El PDF pone los numeros
    # (g_rms contra df_rms, R²); aca solo la frase: de lejos los decimales no se
    # leen y lo que importa es "amplitud si, fase no".
    ax.text(0.015, 0.95, "amplitud parecida · fase equivocada",
            transform=ax.transAxes, fontsize=15, color=GRIS, va="top")
    fig.tight_layout()
    guarda(fig, "p_forma")


# Pregunta: y para CONTROLAR, sirve? Cuatro configuraciones del controlador
# contra el mismo cerebro real, y SIEMPRE los dos lazos (I y E).
# Mostrar los dos no es un detalle: mirando solo E, restar la red parece mejorar
# y la conclusion saldria al reves. Lo que pasa es que acierta un canal y
# destruye el otro.
def control_barras():
    d = jd("f6_closed_loop.json")["filas"]
    # Los nombres del json son largos ("3b. gray-box theta_hat + g cancelada"):
    # se los busca por prefijo y se les pone una etiqueta corta para la charla.
    def b(p): return next((f for f in d if f["nombre"].startswith(p)), None)
    # El json trae tambien la fila "0." (planta LIMPIA), que aca se OMITE a
    # proposito: no es una opcion de control sino el piso teorico con un cerebro
    # sin la fisica agregada, y en la charla confundiria.
    orden = [("1.", "θ verdaderos"), ("2.", "sólo ecuaciones"),
             ("3a", "gray-box:\nparámetros"), ("3b", "gray-box:\n+ restar la red")]
    nom, vI, vE = [], [], []
    # El `if f` deja la figura armarse igual si alguna corrida falta (dibuja
    # menos barras en vez de explotar).
    for pref, et in orden:
        f = b(pref)
        if f: nom.append(et); vI.append(f["rmse_I"]); vE.append(f["rmse_E"])
    x = np.arange(len(nom)); w = 0.37
    fig, ax = plt.subplots(figsize=(11, 5.2))
    r1 = ax.bar(x - w / 2 - .012, vI, w, color=AZUL, label="lazo de I")
    r2 = ax.bar(x + w / 2 + .012, vE, w, color=NARA, label="lazo de E")
    # Tres decimales porque los errores estan en ~0.05-0.2 y la diferencia que
    # cuenta esta en el segundo decimal.
    for bb in list(r1) + list(r2):
        ax.annotate(f"{bb.get_height():.3f}",
                    (bb.get_x() + bb.get_width() / 2, bb.get_height()),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=12.5)
    ax.set_xticks(x); ax.set_xticklabels(nom, fontsize=13)
    ax.set_ylabel("error de seguimiento")
    ax.legend(frameon=False, loc="upper left"); ax.grid(alpha=0.5, axis="y")
    ax.set_ylim(0, max(vI + vE) * 1.35)
    # La anotacion apunta a la barra azul (lazo de I) del cuarto grupo: el indice
    # 3 es "3b, restar la red". El guard len(vI) >= 4 es para no indexar fuera de
    # rango si esa corrida no estaba.
    if len(vI) >= 4:
        ax.annotate("mejora E,\npero DESTROZA I", (3 - w / 2, vI[3] * 1.14),
                    fontsize=14, color=ROJO, ha="center", fontweight="bold")
    fig.tight_layout()
    guarda(fig, "p_control")


# =============================================================================
#  EJECUCION DIRECTA: genera las ocho figuras en docs/figuras_presentacion/
# =============================================================================
#  El orden es el del guion de la charla, no el del archivo: primero las cuatro
#  que solo leen json (instantaneas), y al final las dos que levantan torch y
#  simulan (lazo_abierto y sobre todo lazo_cerrado, que integra el lazo dos
#  veces y es la que tarda).
if __name__ == "__main__":
    print("Figuras de la presentacion:")
    perturbacion()
    costo()
    geometria()
    cruce()
    forma()
    lazo_abierto()
    lazo_cerrado()
    control_barras()
    print(f"\n-> {OUT}/")
