#!/usr/bin/env python3
# =============================================================================
#  F7 — CONTROLES: ¿DONDE DEJA DE FUNCIONAR EL GRAY-BOX SIN MEMORIA?
# =============================================================================
#
#  Las fases anteriores muestran que el gray-box funciona con el par elegido
#  (refractariedad + actuador). Un resultado honesto tiene que mostrar tambien
#  DONDE NO funciona, y por que. Hay TRES maneras distintas de que falle, y
#  conviene no confundirlas porque cada una se arregla (o no) de otra forma:
#
#    (a) ESTADO OCULTO LENTO. Si la fisica que falta tiene memoria propia mas
#        lenta que el sistema, g_φ(I,E) no la puede representar por mas que
#        entrene: le falta la variable. Se barre tau_a de la adaptacion.
#        Se arregla dandole MEMORIA a la correccion (estado latente / RNN).
#
#    (b) NO AUTONOMA. La deriva de wEE(t) depende explicitamente del tiempo, y
#        t NO es argumento de g_φ(I,E,P,Q). No es azar: es reproducible, pero
#        no como funcion de lo que la red ve. Se arregla dandole t (o un
#        estimador del estado lento de neuromodulacion), no mas capacidad.
#
#    (c) IRREDUCIBLE. El ruido de proceso no es funcion de nada: ninguna
#        correccion determinista lo puede predecir. Es el piso. No se arregla:
#        marca hasta donde puede bajar cualquier correccion.
#
#  Los tres casos "malos" dan R2 bajo, asi que MIRANDO SOLO EL NUMERO parecen el
#  mismo problema. La distincion la hace el diagnostico, no la metrica: por eso
#  la columna "tipo" de CASOS esta escrita a mano y no deducida del resultado.
#
#  Como se mide sin gastar un entrenamiento completo por caso: se ajusta la
#  arquitectura EXACTA de g_φ directamente contra el Delta f verdadero (que
#  conocemos). Eso da el TECHO de lo que la correccion podria aprender en el
#  mejor de los casos, con estimulos de test nunca vistos.
#
#  POR QUE EL TECHO Y NO EL ENTRENAMIENTO COMPLETO — son dos razones:
#    1) Separa dos preguntas que el entrenamiento mezcla: "¿este Delta f es
#       REPRESENTABLE con estos argumentos?" (geometria, es lo que se mide aca)
#       y "¿el entrenamiento lo ENCUENTRA?" (optimizacion). Si el techo ya es
#       bajo, no hay ajuste de hiperparametros que salve el caso.
#    2) Costo: 12 casos x 2 juegos de argumentos = 24 ajustes. Un rollout
#       completo con g_φ dentro del RK4 es ~3x mas lento que el white-box (ver
#       run_uncertainty_all.sh); ajustar la red contra un target ya calculado es
#       una regresion punto a punto, sin integrador de por medio.
#  El precio de esta decision: el techo es OPTIMISTA a proposito. Un caso que
#  aca sale "capturable" todavia puede fallar al entrenar; uno que sale
#  "IMPOSIBLE" no tiene salvacion posible con esos argumentos.
#
#  UNIDADES: todo en ms (convencion de gen_multi_dataset.py, el regimen del
#  control). Las constantes propias del sistema son te = 1 ms y ti = 2 ms: son
#  la referencia contra la que se compara si un tau_a es "rapido" o "lento".
#
#  Ver docs/graybox_manual_completo.md, seccion 18.
#
#  USO:  python scripts/exp_f7_controls.py
# =============================================================================

from __future__ import annotations

import json
import sys
from pathlib import Path

# Dos rutas al sys.path, y las dos hacen falta:
#   - la raiz, para poder importar el paquete src.wilson_cowan.
#   - scripts/, porque gen_uncertain_dataset es un SCRIPT hermano (no un modulo
#     del paquete) y se lo importa por nombre mas abajo para reusar generar_con.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import numpy as np
import torch
import torch.nn as nn

# 2 hilos y no todos: el runner (run_uncertainty_all.sh) lanza 4 procesos en
# paralelo. Medido ahi: torch satura en ~2 hilos, y si cada proceso toma todos
# los nucleos se pelean entre si y la ola tarda mas.
torch.set_num_threads(2)

from src.wilson_cowan import (
    Adaptation, ProcessNoise, SynapticDepression, HiddenPopulation,
    HeterogeneousSigmoid, Refractoriness, WeightDrift, default_uncertainty,
)

OUT = Path("results/uncertainty/f7_controls.json")
DATA_DIR = Path("data/processed/uncertain/controls")
FIG = Path("results/figures/f7_memoria.png")


# =============================================================================
#  EL CATALOGO DE CASOS
# =============================================================================
#  Cada fila es (nombre, fabrica_de_perturbacion, tipo). El nombre es tambien el
#  nombre del .npz cacheado, y en la familia de adaptacion se PARSEA mas abajo
#  para sacar el tau_a de la curva: si se renombra "adapt_tauXX" se rompe la
#  figura en silencio.
#
#  El orden no es alfabetico ni casual: primero las capturables (para que la
#  tabla arranque mostrando que el metodo si funciona cuando puede), despues el
#  par del roadmap, despues la perilla de memoria, y al final los controles
#  negativos. Asi la tabla impresa se lee como el argumento del paper.
#
# --- Los casos. tau_a barre de rapido (esclavizado al estado) a lento (memoria).
#  Cada fabrica recibe el indice del escenario (n) para poder variar la semilla
#  en las familias con aleatoriedad; las deterministas simplemente lo ignoran.
CASOS = [
    ("refract_r0.1",   lambda n: Refractoriness(r=0.10),                  "capturable"),
    ("hetero_0.8",     lambda n: HeterogeneousSigmoid(spread=0.8),        "capturable"),
    ("par_nominal",    lambda n: default_uncertainty(1.0),                "el del roadmap"),
    ("adapt_tau1",     lambda n: Adaptation(b=0.5, tau_a=1.0),            "memoria rapida"),
    ("adapt_tau3",     lambda n: Adaptation(b=0.5, tau_a=3.0),            "memoria rapida"),
    ("adapt_tau10",    lambda n: Adaptation(b=0.5, tau_a=10.0),           "memoria media"),
    ("adapt_tau30",    lambda n: Adaptation(b=0.5, tau_a=30.0),           "memoria lenta"),
    ("adapt_tau100",   lambda n: Adaptation(b=0.5, tau_a=100.0),          "memoria lenta"),
    ("depresion",      lambda n: SynapticDepression(U=0.15, tau_d=30.0),  "estado oculto"),
    ("poblacion_oculta", lambda n: HiddenPopulation(w_back=0.8),          "estado oculto"),
    ("deriva_wEE",     lambda n: WeightDrift(amp=0.15),                   "no autonoma"),
    # t_max=210 ms con trayectorias de 200 ms: el camino de ruido se pre-genera
    # en una grilla y hay que pasarse de largo, porque el RK4 y el calculo del
    # Delta f evaluan la derivada en instantes intermedios. Si el camino se
    # acabara antes, ProcessNoise levanta ValueError a proposito (congelarlo
    # convertiria el ruido en un sesgo DC determinista, o sea APRENDIBLE, y el
    # control negativo dejaria de ser negativo).
    ("ruido_proceso",  lambda n: ProcessNoise(sigma=0.02, seed=1000 + n,
                                              t_max=210.0),               "irreducible"),
]


# =============================================================================
#  GENERACION (con cache en disco)
# =============================================================================
#  Un dataset por caso: las 20 trayectorias multi-estimulo de siempre, pero con
#  la planta perturbada por esa familia, y con el Delta f verdadero guardado.
#  Generar los 12 cuesta bastante, asi que si el .npz ya existe se reusa.
#
#  TRAMPA DEL CACHE: la clave es SOLO el nombre. Si se cambia un parametro de un
#  caso sin cambiarle el nombre (p.ej. b=0.5 -> b=1.0 en adapt_tau30), el .npz
#  viejo se reusa y el resultado sale del experimento anterior sin avisar. Al
#  editar CASOS hay que borrar data/processed/uncertain/controls/<nombre>.npz.
def genera_si_falta(nombre, fabrica):
    path = DATA_DIR / f"{nombre}.npz"
    if path.exists():
        return path
    from gen_uncertain_dataset import generar_con
    # Contador para que cada escenario reciba una perturbacion distinta cuando la
    # familia tenga aleatoriedad. Con la semilla fija, las 20 trayectorias del
    # dataset compartirian EXACTAMENTE la misma realizacion de ruido y el
    # "control negativo" se volveria una senal repetida y por lo tanto aprendible.
    contador = {"n": 0}

    # OJO: generar_con llama a la fabrica una vez de mas, al principio, solo para
    # leer los metadatos de la perturbacion. Ese consumo del contador es inocuo
    # (las semillas de los datos arrancan en 1002 en vez de 1001) pero explica
    # por que las semillas no coinciden con el indice del escenario.
    def fab():
        contador["n"] += 1
        return fabrica(contador["n"])
    generar_con(fab, path, {"caso": nombre})
    return path


# =============================================================================
#  EL AJUSTE DEL TECHO
# =============================================================================
def ajusta_g(Xtr, Ytr, Xte, Yte, epochs=1200, seed=0):
    """Arquitectura EXACTA de la correccion de src/neural_ode/dynamics.py.

    Que devuelve: (R2 en train, R2 en test). El de test es el que importa: mide
    si el techo se sostiene con estimulos que la red nunca vio, no si la red
    puede memorizar la trayectoria.

    Por que la arquitectura tiene que ser la MISMA (2 capas de 32, tanh) y no
    una red grande: el numero que buscamos es el techo de ESA correccion, la que
    despues se entrena de verdad. Con una red mas ancha el techo subiria y la
    conclusion ("hace falta memoria, no capacidad") dejaria de estar medida.
    """
    torch.manual_seed(seed)
    g = nn.Sequential(nn.Linear(Xtr.shape[1], 32), nn.Tanh(),
                      nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 2))
    # Estandarizacion de la entrada con la media/desvio de TRAIN, aplicada tal
    # cual al test (si se recalcularan en test habria fuga de informacion y el
    # techo saldria inflado). El 1e-8 evita dividir por 0 en una columna quieta.
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    xt, xv = (Xtr - mu) / sd, (Xte - mu) / sd
    # Escala del target. Delta f es chico (RMS ~0.003 a 0.02 segun la familia),
    # asi que el MSE crudo vive en ~1e-5 y con lr=3e-3 la red practicamente no
    # se mueve: se quedaria en la solucion trivial cero y TODOS los casos darian
    # "imposible". Se entrena contra Y/sy (target de orden 1) y despues se
    # desescala para reportar el R2 en unidades fisicas.
    sy = Ytr.pow(2).mean().sqrt() + 1e-12
    opt = torch.optim.Adam(g.parameters(), lr=3e-3)
    # Cosine annealing hasta 0 en el ultimo paso: sin el, el ruido del minibatch
    # deja el R2 rebotando en la tercera cifra y casos vecinos de la curva de
    # tau_a se cruzan entre si.
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        # Minibatch con reposicion (randint, no permutacion): el train son ~13
        # escenarios x 4000 puntos = ~52k muestras, asi que 4096 es ~8% del
        # total. No hace falta recorrer epocas exactas, solo muchas muestras
        # i.i.d.; 1200 x 4096 = ~5M presentaciones, ~95 pasadas equivalentes.
        idx = torch.randint(0, len(xt), (4096,))
        opt.zero_grad()
        ((g(xt[idx]) - Ytr[idx] / sy) ** 2).mean().backward()
        opt.step(); sch.step()

    # R2 con las dos salidas juntas (dI y dE): la varianza total se mide contra
    # la media de cada columna, pero las sumas se agregan. Interpretacion:
    #   1  = reproduce el Delta f exactamente.
    #   0  = igual que predecir la media (no aprendio nada util).
    #   <0 = PEOR que la media. No es un bug: pasa cuando el Delta f no es
    #        funcion de los argumentos y lo que se ajusto en train no vale en
    #        test (es lo que le pasa al par del roadmap con g(I,E): -0.11).
    def r2(x, y):
        with torch.no_grad():
            p = g(x) * sy
        ss_res = float(((p - y) ** 2).sum())
        ss_tot = float(((y - y.mean(0)) ** 2).sum())
        return 1.0 - ss_res / max(ss_tot, 1e-30)
    return r2(xt, Ytr), r2(xv, Yte)


# =============================================================================
#  CARGA DEL DATASET
# =============================================================================
#  Arma la regresion (X -> Delta f) a partir del .npz, separando train y test
#  POR ESCENARIO COMPLETO (la mascara is_test que ya trae el dataset). El split
#  no es por puntos al azar: puntos vecinos de una misma trayectoria estan
#  correlacionados y un split aleatorio daria un techo falsamente alto.
#
#  solo_estado elige el juego de argumentos de g_φ, y esa es la comparacion
#  central del experimento:
#    True  -> g(I,E)      : la variante que el proyecto usa de verdad, porque el
#                           controlador IMC puede cancelarla de forma exacta.
#    False -> g(I,E,P,Q)  : mas expresiva, pero al ver el estimulo puede tapar
#                           parametros equivocados (rompe la identificabilidad).
#  Sirve como diagnostico: si el techo sube mucho al agregar P,Q, el hueco entra
#  por el CANAL DE ACTUACION y no por el estado.
def carga(path, solo_estado):
    d = np.load(path, allow_pickle=True)
    it = d["is_test"].astype(bool)

    # ravel(): se apilan todos los instantes de todos los escenarios elegidos en
    # una nube de puntos. Se pierde el orden temporal a proposito -> esta
    # regresion NO tiene memoria, igual que la g_φ real.
    def arma(sel):
        cols = [d["I"][sel].ravel(), d["E"][sel].ravel()]
        if not solo_estado:
            cols += [d["P"][sel].ravel(), d["Q"][sel].ravel()]
        X = torch.tensor(np.stack(cols, 1), dtype=torch.float32)
        # El target es el Delta f VERDADERO que guardo el generador: el campo de
        # la planta real menos el de Wilson-Cowan con los parametros verdaderos.
        # Es exactamente lo que g_φ deberia aprender, y conocerlo es lo que
        # permite medir un techo en vez de un residuo de entrenamiento.
        Y = torch.tensor(np.stack([d["dfI"][sel].ravel(),
                                   d["dfE"][sel].ravel()], 1), dtype=torch.float32)
        return X, Y
    # Se usan P,Q COMANDADOS (nunca P_eff/Q_eff): en un experimento real es lo
    # unico que se conoce. Meter el estimulo efectivo seria hacer trampa, porque
    # ahi vive parte del hueco que la correccion tiene que descubrir.
    return arma(~it), arma(it), d


# =============================================================================
#  BARRIDO PRINCIPAL: una fila por familia
# =============================================================================
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filas = []
    print("=== F7 · techo de lo que g_φ puede capturar, por familia ===")
    print("    (se ajusta la arquitectura real de g_φ contra el Delta f VERDADERO;")
    print("     test = estimulos held-out del propio dataset)\n")
    hdr = f"{'caso':20} {'tipo':16} {'|Df|':>8} {'R2_x':>7} {'R2_xpq':>8}  veredicto"
    # El +16 del separador cubre el ancho de la columna "veredicto", que en el
    # encabezado va sin padding.
    print(hdr); print("-" * (len(hdr) + 16))

    for nombre, fabrica, tipo in CASOS:
        path = genera_si_falta(nombre, fabrica)
        (Xtr, Ytr), (Xte, Yte), d = carga(path, solo_estado=True)
        r2_x = ajusta_g(Xtr, Ytr, Xte, Yte)[1]
        (Xtr2, Ytr2), (Xte2, Yte2), _ = carga(path, solo_estado=False)
        r2_xpq = ajusta_g(Xtr2, Ytr2, Xte2, Yte2)[1]
        # |Df| = RMS del Delta f en train. Es el TAMANO del hueco, y hay que
        # leerlo junto al R2: un R2 malo con |Df| chico casi no molesta, y un R2
        # bueno con |Df| grande es donde la correccion realmente sirve. La
        # referencia natural es el ruido de observacion del proyecto (0.02).
        mag = float(Ytr.pow(2).mean().sqrt())

        # El veredicto usa el MEJOR de los dos juegos de argumentos: la pregunta
        # es si el Delta f es representable con ALGO de lo que la red puede ver.
        # Con eso, "IMPOSIBLE" significa que no lo salva ni la version generosa.
        mejor = max(r2_x, r2_xpq)
        # De donde salen los cortes (ver la tabla de la seccion 18 del manual):
        #   0.9 -> las familias que son funcion pura del estado llegan ahi
        #          (refractariedad 0.997, sigmoidea heterogenea 0.984).
        #   0.6 -> limite practico: por debajo queda mas hueco sin explicar que
        #          explicado, y el gray-box ya no compra gran cosa.
        #   0.1 -> el piso irreducible medido: el ruido de proceso da ~0.11-0.12
        #          con TODOS los argumentos. Por debajo de eso no hay senal.
        if mejor > 0.9:
            v = "CAPTURABLE"
        elif mejor > 0.6:
            v = "parcial"
        elif mejor > 0.1:
            v = "POBRE"
        else:
            v = "IMPOSIBLE"
        # flush: cada caso tarda, y el runner redirige a un log. Sin flush no se
        # ve nada hasta el final.
        print(f"{nombre:20} {tipo:16} {mag:8.4f} {r2_x:7.3f} {r2_xpq:8.3f}  {v}",
              flush=True)
        filas.append({"caso": nombre, "tipo": tipo, "df_rms": mag,
                      "r2_g_x": r2_x, "r2_g_xpq": r2_xpq, "veredicto": v})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(filas, indent=2))

    # --- La curva que resume todo: R2 vs cuan lenta es la memoria oculta.
    #  Se queda con r2_g_x (solo estado) porque la pregunta es si la correccion
    #  SIN MEMORIA y sin ver el estimulo alcanza. El tau_a se lee del nombre del
    #  caso ("adapt_tau30" -> 30.0): es fragil, pero evita duplicar la grilla.
    taus, r2s = [], []
    for f in filas:
        if f["caso"].startswith("adapt_tau"):
            taus.append(float(f["caso"].split("tau")[1]))
            r2s.append(f["r2_g_x"])
    if taus:
        print("\n--- La perilla de la memoria: adaptacion con tau_a creciente ---")
        print("    (te = 1 ms, ti = 2 ms son las constantes propias del sistema)")
        for t, r in sorted(zip(taus, r2s)):
            # Barra ASCII de 40 caracteres = R2 de 1.0. El max(r,0) es porque el
            # R2 puede ser negativo y "#" * negativo daria cadena vacia sin avisar.
            barra = "#" * int(max(r, 0) * 40)
            print(f"    tau_a = {t:6.1f} ms   R2 = {r:6.3f}  {barra}")
        _figura(sorted(zip(taus, r2s)))
    print(f"\n  -> {OUT}")


# =============================================================================
#  LA FIGURA: el techo contra la lentitud del estado oculto
# =============================================================================
#  Eje x logaritmico porque tau_a barre 1 -> 100 ms (dos decadas). La linea
#  vertical en te = 1 ms es la referencia que da sentido a "rapido" y "lento":
#  a la izquierda el estado oculto queda esclavizado al estado visible y la
#  correccion lo reconstruye; a la derecha guarda historia propia que (I,E) no
#  contiene y el techo se cae. Es el argumento cuantitativo de cuando hace falta
#  darle memoria a la correccion.
def _figura(pares):
    # Backend Agg antes de importar pyplot: el script corre sin pantalla (dentro
    # de las olas del runner) y con el backend interactivo fallaria.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = [p[0] for p in pares]; r = [p[1] for p in pares]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.semilogx(t, r, "o-", color="#1f4e79", lw=2)
    # Las mismas dos cotas que el veredicto de main: si se cambian alla hay que
    # cambiarlas aca, no estan compartidas.
    ax.axhline(0.9, ls=":", color="#2ca02c", label="capturable")
    ax.axhline(0.6, ls=":", color="#d62728", label="limite practico")
    ax.axvline(1.0, ls="--", color="#888888", lw=1)
    # Coordenadas en unidades de datos: 1.05 ms a la derecha de la linea, y 0.05
    # de R2 para que el texto quede abajo sin taparse con la curva.
    ax.text(1.05, 0.05, "te = 1 ms", fontsize=8, color="#888888")
    ax.set_xlabel("tau_a del estado oculto [ms]")
    ax.set_ylabel("R2 alcanzable por g_φ(I,E)")
    ax.set_title("Cuando la correccion sin memoria deja de alcanzar")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
