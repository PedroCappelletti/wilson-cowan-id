#!/usr/bin/env python3
# =============================================================================
#  A (validacion) — ¿SIRVE DE ALGO EL ESTIMULO DISENADO?
# =============================================================================
#
#  El diseno bajo la fraccion imitable del 71.4% (mejor estimulo de libreria) al
#  59.5%, con el mismo |Delta f| y la misma amplitud media. Pero eso es una
#  metrica geometrica: falta ver si se traduce en algo util.
#
#  La prediccion concreta, que es lo que este script pone a prueba:
#    Si menos del mismatch es imitable moviendo parametros, entonces
#      (a) el white-box tiene menos margen para "esconder" el hueco en theta
#          -> deberia sesgarse distinto, y
#      (b) la correccion g_phi tiene menos donde competir y mas fisica genuina
#          que aprender -> su R2 contra el Delta f VERDADERO deberia subir.
#
#  El (b) es el importante: es la diferencia entre una correccion que ajusta y
#  una que aprende fisica.
#
#  Como se arma el dataset. El optimizador devuelve UN estimulo; un dataset
#  necesita variedad. Se generan 20 escenarios a partir del disenado aplicando
#  corrimientos circulares y jitter de amplitud: eso conserva la ESTADISTICA de
#  forma que lo hace bueno y da diversidad. Se verifica que los escenarios
#  derivados mantengan la fraccion imitable baja (no la mantuvieron: ver abajo).
#
#  LO QUE SALIO — y por que este script vale igual aunque el resultado sea
#  negativo. La fraccion imitable CONJUNTA del dataset de 20 variantes subio a
#  ~73.9%, PEOR que la del dataset de libreria (~67.1%), a pesar de que cada
#  variante por separado esta en el 59.5%. La explicacion es la parte valiosa:
#  la fraccion se mide pidiendo UN SOLO delta-theta que explique todos los
#  escenarios A LA VEZ. Si los escenarios son distintos entre si, la
#  degeneracion se rompe sola (la libreria tiene 7 familias de estimulo);
#  20 variantes de la MISMA forma comparten casi la misma degeneracion y no
#  aportan nada nuevo.
#  Moraleja (y correccion del plan): hay dos palancas y la segunda pesa mas
#  -> que cada estimulo sea individualmente bueno, y que los estimulos sean
#  COMPLEMENTARIOS entre si. El siguiente paso es disenar el CONJUNTO
#  (scripts/exp_a_set_design.py), no la senal aislada.
#
#  TRAMPA AL LEER LOS NUMEROS: el 59.5% que sale del JSON de diseno es
#  POR ESCENARIO (2500 puntos de UNA trayectoria); el que imprime este script es
#  CONJUNTO (4000 puntos tomados de las 14 trayectorias de train). No son la
#  misma cantidad y no se pueden comparar entre si; lo que si es comparable es
#  conjunto-vs-conjunto: la fila "libreria" contra la fila "DISENADO".
#
#  QUE NECESITA ANTES (las rutas son RELATIVAS -> correr desde la raiz del repo):
#    - results/uncertainty/a_stimulus_design_fraccion.json
#         <- python scripts/exp_a_stimulus_design.py --objetivo fraccion
#    - data/processed/uncertain/eps1.npz  (dataset de libreria al mismo eps=1)
#         <- python scripts/gen_uncertain_dataset.py
#  QUE DEJA:
#    - data/processed/uncertain/disenado.npz  (listo para el trainer gray-box)
#    - results/uncertainty/a_validacion.json  (las dos filas de la tabla)
#
#  USO:  python scripts/exp_a_validate.py
# =============================================================================

from __future__ import annotations

import json
import sys
from pathlib import Path

# Dos rutas al sys.path: la raiz del repo (para "src.*") y scripts/ (para
# importar los scripts hermanos como si fueran modulos). Lo segundo es a
# proposito: se REUSAN escalon/normaliza/T_SPAN/EPS del script de diseno en vez
# de copiarlos, asi la parametrizacion del estimulo y el nivel de mismatch son
# identicos a los de la busqueda. Si se copiaran, cualquier cambio alla dejaria
# este script validando otra cosa.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import numpy as np
import torch

# Lo unico que hace torch aca es una pseudoinversa de (2N, 10): matrices chicas.
# Mas hilos no aceleran nada y torch por defecto se queda con toda la maquina.
torch.set_num_threads(2)

from src.wilson_cowan import WilsonCowanParams, default_uncertainty
from src.data import generate_dataset
from src.neural_ode import GrayBoxWC
from src.neural_ode.graybox_train import projection_operator, projected_fraction, ALL_P

# T_ON/T_OFF vienen por referencia (la ventana del estimulo); escalon() ya los
# usa por dentro como globales de su propio modulo, aca no se tocan.
from exp_a_stimulus_design import escalon, normaliza, T_ON, T_OFF, T_SPAN, EPS
from gen_uncertain_dataset import delta_f_verdadero, PARAM_KEYS

# =============================================================================
#  SECCION 1: DE DONDE SALEN LOS DATOS Y A DONDE VAN
# =============================================================================

# Se lee el ganador del objetivo "fraccion", NO el de "residuo". Los dos existen
# y miden cosas distintas: el de residuo agranda el hueco pero puede dejar la
# fraccion imitable igual, y lo que este experimento pone a prueba es justamente
# si se rompio la degeneracion.
DISENO = Path("results/uncertainty/a_stimulus_design_fraccion.json")
OUT_NPZ = Path("data/processed/uncertain/disenado.npz")
OUT_JSON = Path("results/uncertainty/a_validacion.json")

# 4000 puntos en (0, 200) -> dt ~ 0.05, EL MISMO muestreo que eps1.npz (que sale
# de gen_multi_dataset). Igualarlo no es un detalle: si un dataset estuviera mas
# grueso que el otro, la diferencia de fraccion imitable podria ser resolucion
# disfrazada de diseno. La busqueda corrio a 2000 puntos porque cada evaluacion
# es cara; aca se paga el doble una sola vez.
N_EVAL = 4000

# 20 escenarios, los 6 ultimos reservados para test -> 14 de train. Mismo
# reparto que los datasets de libreria, para que las metricas sean comparables.
N_ESC = 20
N_TEST = 6

# Techo de amplitud del estimulo (lo que el actuador puede entregar).
# OJO: es el mismo 1.4 que usa exp_a_stimulus_design, pero alla es una variable
# LOCAL de main() y no se exporta. Si se cambia alla hay que cambiarlo aca a
# mano, o los escenarios derivados dejan de respetar el presupuesto con el que
# se optimizo el diseno y la comparacion pierde sentido.
AMP_MAX = 1.4


# =============================================================================
#  SECCION 2: DEL ESTIMULO UNICO AL DATASET (las 20 variantes)
# =============================================================================

def escenarios(pv, qv, media_P, media_Q, seed=0):
    """20 variantes del estimulo disenado: corrimiento circular + jitter.
    Conserva la forma (que es lo que lo hace bueno) y da diversidad.

    Entra: pv, qv = las amplitudes por tramo que devolvio el optimizador (24
    tramos con la parametrizacion actual del diseno);
    media_P, media_Q = el presupuesto de amplitud media con el que se optimizo.
    Sale: lista de (label, P(t), Q(t), es_test) lista para generate_dataset.

    POR QUE roll + jitter y no estimulos nuevos al azar: sortear formas nuevas
    tira a la basura lo que el optimizador encontro. El corrimiento circular y un
    jitter suave conservan la distribucion de amplitudes y de duraciones de tramo
    (la "forma" es justamente lo que baja la fraccion imitable) y solo cambian
    donde cae cada cosa en el tiempo.

    TRAMPA: esa fidelidad es tambien la razon por la que el experimento salio
    negativo. Las 20 variantes comparten casi la misma degeneracion, asi que la
    fraccion CONJUNTA no baja (ver encabezado). Para bajarla harian falta formas
    COMPLEMENTARIAS, no variantes de una sola.
    """
    rng = np.random.default_rng(seed)
    out = []
    for k in range(N_ESC):
        if k == 0:
            # El escenario 0 es el diseno tal cual salio del optimizador: queda
            # como referencia dentro del propio dataset.
            p, q = pv.copy(), qv.copy()
        else:
            # Corrimiento circular del patron de tramos. Desde 1 y no 0 porque un
            # corrimiento nulo repetiria el escenario 0. Son dos sorteos
            # independientes, asi que cambia tambien el desfasaje relativo entre P
            # y Q: eso es lo que mueve el punto de operacion del sistema.
            p = np.roll(pv, rng.integers(1, len(pv)))
            q = np.roll(qv, rng.integers(1, len(qv)))
            # Jitter multiplicativo de +-15%: alcanza para que las trayectorias no
            # sean calcos, y es poco como para no destruir la forma optimizada.
            p = p * rng.uniform(0.85, 1.15, size=p.shape)
            q = q * rng.uniform(0.85, 1.15, size=q.shape)
        # Re-ancla la media al presupuesto DESPUES del jitter (que la corrio) y
        # recorta a [0, AMP_MAX]. Sin esto cada variante gastaria una energia
        # distinta y las diferencias entre escenarios se confundirian con
        # amplitud en vez de forma.
        p = normaliza(p, media_P, AMP_MAX)
        q = normaliza(q, media_Q, AMP_MAX)
        # Los ultimos N_TEST quedan como test: el modelo nunca los ve.
        # Se parte por indice y no al azar para que el reparto sea reproducible
        # sin tener que guardar la mascara aparte.
        out.append((f"dis_{k:02d}", escalon(p), escalon(q), k >= N_ESC - N_TEST))
    return out


# =============================================================================
#  SECCION 3: LA METRICA — FRACCION IMITABLE CONJUNTA
# =============================================================================

def frac_del_dataset(d, n_pts=4000, seed=0):
    """Fraccion imitable del dataset completo (todos los escenarios juntos).

    Descompone el hueco verdadero como  Delta f = S·delta_theta* + residuo  y
    devuelve tres numeros:
      - fraccion imitable: cuanta energia del hueco se explica moviendo los 10
        parametros (o sea: cuanto podria "esconder" el white-box en theta).
      - |Delta f| RMS: el tamano total del hueco.
      - residuo RMS: la parte que NINGUN cambio de parametros puede imitar; es la
        fisica genuina que le queda para aprender a g_phi.
    Hacen falta los tres juntos porque la fraccion sola se puede bajar hacien-
    do el hueco chiquito, y eso no sirve de nada: no habria nada que aprender.

    Lo CONJUNTO es la clave del experimento: la proyeccion se hace en el ESPACIO
    DE FUNCIONES, con todos los escenarios apilados en una sola nube de puntos, o
    sea exigiendo un unico delta_theta que los explique a TODOS a la vez. Por eso
    este numero puede ser peor que el de cada escenario por separado: escenarios
    parecidos comparten degeneracion y no se ayudan entre si.
    """
    # Solo TRAIN. La degeneracion que importa es la de los datos que el modelo va
    # a ver; el test se guarda para medir generalizacion, no para esto.
    # OJO: is_test tiene que ser BOOL. Si viniera como entero, "~it" seria
    # complemento a bits (~0 = -1, ~1 = -2) y numpy dejaria de leerlo como mascara
    # para leerlo como POSICIONES: se elegirian las filas -1 y -2 en silencio.
    # De ahi el .astype(bool) al cargar el .npz de referencia mas abajo.
    it = d["is_test"]
    I, E, P, Q = (d[k][~it].ravel() for k in ("I", "E", "P", "Q"))
    dfI, dfE = d["dfI"][~it].ravel(), d["dfE"][~it].ravel()
    # Submuestreo de n_pts puntos: la pseudoinversa se calcula sobre una matriz de
    # (2N, 10) y N seria el dataset entero (14 escenarios x 4000 instantes).
    # Semilla fija -> el numero que se reporta es reproducible y, con el mismo
    # n_pts en los dos datasets, comparable entre ellos.
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(I), min(n_pts, len(I)), replace=False)
    X = torch.tensor(np.stack([I[idx], E[idx]], 1), dtype=torch.float32)
    Pt = torch.tensor(P[idx].reshape(-1, 1), dtype=torch.float32)
    Qt = torch.tensor(Q[idx].reshape(-1, 1), dtype=torch.float32)
    DF = torch.tensor(np.stack([dfI[idx], dfE[idx]], 1), dtype=torch.float32)
    # Las sensibilidades se evaluan en el theta VERDADERO: es el punto de
    # linealizacion honesto (donde estaria el white-box si no tuviera que
    # deformar nada para tapar el hueco).
    p = WilsonCowanParams()
    true = {k: getattr(p, k) for k in ALL_P}
    # Los dos flags en True son OBLIGATORIOS: backbone_sensitivities devuelve
    # columnas de CEROS para los parametros que no son aprendibles. Con alguno en
    # False la matriz S quedaria mutilada, el subespacio imitable seria mas chico
    # y la fraccion saldria artificialmente baja (buena noticia falsa).
    m = GrayBoxWC(true, {k: true[k] for k in ("wEE", "wEI", "wIE", "wII")},
                  learnable_weights=True, learnable_params=True)
    S = m.backbone_sensitivities(X, Pt, Qt)
    # A = pseudoinversa de S aplanada: c = A @ vec(Delta f) son los delta_theta
    # que mejor imitan el hueco sobre TODOS los puntos a la vez.
    A, Sf = projection_operator(S)
    _, frac = projected_fraction(DF, A, Sf)
    # projected_fraction ya calcula la componente proyectada, pero solo devuelve
    # energia y fraccion. El residuo en unidades ABSOLUTAS hay que rearmarlo aca.
    # El reshape(-1, 2) deshace el aplanado de projection_operator (orden n*2+c).
    c = A @ DF.reshape(-1)
    res = DF - (Sf @ c).reshape(-1, 2)
    return float(frac), float(DF.pow(2).mean().sqrt()), float(res.pow(2).mean().sqrt())


# =============================================================================
#  SECCION 4: EL EXPERIMENTO — generar, medir y comparar contra la libreria
# =============================================================================

def main():
    # Sin el JSON del diseno no hay nada que validar: este script no busca el
    # estimulo, solo consume el que dejo el optimizador.
    if not DISENO.exists():
        print(f"Falta {DISENO}: corre exp_a_stimulus_design.py --objetivo fraccion")
        return
    dis = json.loads(DISENO.read_text())
    # P_seg / Q_seg: las amplitudes de los 24 tramos del estimulo ganador.
    # amp_media_*: el presupuesto de amplitud media con el que se lo optimizo (el
    # del APRBS de libreria). Hay que respetarlo o la comparacion mediria energia.
    pv = np.array(dis["P_seg"]); qv = np.array(dis["Q_seg"])
    media_P, media_Q = dis["amp_media_P"], dis["amp_media_Q"]

    print("=== A (validacion) · dataset con el estimulo disenado ===")
    print(f"    diseno base: imitable {100*dis['disenado']['frac_imitable']:.1f}%  "
          f"(referencia de libreria: {100*dis['referencia']['frac_imitable']:.1f}%)\n")

    params = WilsonCowanParams()
    escs = escenarios(pv, qv, media_P, media_Q)
    I_all, E_all, P_all, Q_all, dfI_all, dfE_all, labels, is_test = \
        [], [], [], [], [], [], [], []
    t_ref = None
    for label, Pf, Qf, test in escs:
        # Una perturbacion NUEVA por escenario. Las perturbaciones con estado
        # interno pre-generado (el retardo del actuador, el ruido) no se pueden
        # compartir entre trayectorias: se arrastraria el final de una al comienzo
        # de la siguiente y los escenarios dejarian de ser independientes.
        pert = default_uncertainty(EPS)
        # noise_std=0.0 a proposito: se aisla el MISMATCH. Con ruido de medicion
        # encima no se podria atribuir el Delta f a la fisica faltante. La semilla
        # queda fija por prolijidad, aunque sin ruido no cambie nada.
        ds = generate_dataset(params=params, P=Pf, Q=Qf, I0=0.0, E0=0.0,
                              t_span=T_SPAN, n_eval=N_EVAL, noise_std=0.0,
                              seed=42, perturbation=pert)
        # Todos los escenarios comparten la misma grilla de tiempo (mismo t_span y
        # n_eval), asi que guardar la ultima alcanza; de ahi sale tambien el dt.
        t_ref = ds["t"]
        # El hueco verdadero instante a instante: campo de la planta real menos
        # campo Wilson-Cowan nominal. Se evalua con el estimulo COMANDADO
        # (ds["P"], no P_eff) porque es lo unico que el modelo conoce: la
        # distorsion del actuador forma parte de lo que g_phi debe descubrir.
        # pert_extra son los estados ocultos de la perturbacion, necesarios para
        # reevaluar el campo real en el mismo punto de la trayectoria.
        dfI, dfE = delta_f_verdadero(params, pert, Pf, Qf, ds["t"], ds["I"], ds["E"],
                                     ds["P"], ds["Q"], ds.get("pert_extra"))
        I_all.append(ds["I"]); E_all.append(ds["E"])
        P_all.append(ds["P"]); Q_all.append(ds["Q"])
        dfI_all.append(dfI); dfE_all.append(dfE)
        labels.append(label); is_test.append(test)
        flag = "TEST " if test else "train"
        # Chequeo de sanidad por escenario: el rango de E dice si la trayectoria
        # se quedo pegada a la meseta de la sigmoidea (escenario poco informativo),
        # y |Df| si el hueco tiene el tamano esperado para este eps.
        print(f"  [{flag}] {label}  E=[{ds['E'].min():6.3f},{ds['E'].max():6.3f}] "
              f"|Df|={np.sqrt(dfI**2+dfE**2).mean():.4f}")

    # El .npz con el formato que espera el trainer gray-box (load_split lee
    # I, E, P, Q, dt, is_test, labels, dfI/dfE y los 10 parametros verdaderos).
    # Los 10 params se guardan aunque sean los de fabrica: el trainer los usa como
    # referencia para el error parametrico, y asi el archivo se explica solo.
    # NOTA: aca NO se guardan P_eff/Q_eff ni la metadata de la perturbacion, que
    # si guarda gen_uncertain_dataset. Los scripts de figuras que leen P_eff no
    # funcionan sobre este dataset (para entrenar y medir no hacen falta: son
    # diagnostico, y el modelo nunca puede verlos).
    d = {"t": t_ref,
         "I": np.stack(I_all), "E": np.stack(E_all),
         "P": np.stack(P_all), "Q": np.stack(Q_all),
         "dfI": np.stack(dfI_all), "dfE": np.stack(dfE_all),
         "is_test": np.asarray(is_test), "labels": np.asarray(labels),
         "dt": float(t_ref[1] - t_ref[0]), "eps": np.asarray(EPS),
         **{k: np.asarray(getattr(params, k)) for k in PARAM_KEYS}}
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, **d)

    # Las dos mediciones que se comparan. La referencia es el dataset de libreria
    # al MISMO eps y con el mismo muestreo: la unica diferencia es el estimulo
    # (7 familias distintas alla, 20 variantes de una sola forma aca), que es
    # justamente la variable del experimento.
    frac, df, res = frac_del_dataset(d)
    ref = np.load("data/processed/uncertain/eps1.npz", allow_pickle=True)
    # Se le pasa solo lo que frac_del_dataset necesita. El astype(bool) no es
    # decorativo: is_test entra como mascara con "~", y si el .npz lo devolviera
    # como entero la negacion seria a nivel de bits y dejaria de ser una mascara
    # (ver el comentario en frac_del_dataset).
    frac_ref, df_ref, res_ref = frac_del_dataset(
        {k: ref[k] for k in ("I", "E", "P", "Q", "dfI", "dfE")} |
        {"is_test": ref["is_test"].astype(bool)})

    # LA tabla del experimento. Leer la columna "imitable" de las dos filas: si la
    # del DISENADO no baja respecto de la libreria, el diseno de UNA senal no
    # alcanza y el problema es el conjunto (que es lo que paso: ~73.9% vs ~67.1%).
    # La columna "residuo" dice cuanta fisica genuina le queda a g_phi por
    # aprender, y sirve de control: una fraccion baja con residuo chico seria una
    # victoria vacia.
    print(f"\n=== El dataset entero (no un solo escenario) ===")
    print(f"  {'':22} {'|Delta f|':>11} {'imitable':>11} {'residuo':>11}")
    print(f"  {'libreria (eps1.npz)':22} {df_ref:11.5f} {100*frac_ref:10.1f}% {res_ref:11.5f}")
    print(f"  {'DISENADO':22} {df:11.5f} {100*frac:10.1f}% {res:11.5f}")
    print(f"\n  -> {OUT_NPZ}")

    # Se deja el JSON con las dos filas para que las figuras y el informe no
    # tengan que volver a correr todo esto (cada corrida son 20 simulaciones con
    # la planta perturbada mas dos pseudoinversas).
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "libreria": {"df_rms": df_ref, "frac_imitable": frac_ref, "residuo": res_ref},
        "disenado": {"df_rms": df, "frac_imitable": frac, "residuo": res},
    }, indent=2))
    print(f"  -> {OUT_JSON}")
    # La fraccion imitable es una metrica geometrica: predice, no demuestra. La
    # prueba final es entrenar sobre este dataset y ver si el R2 de g_phi contra el
    # Delta f VERDADERO sube. Ese R2 es lo que separa una correccion que ajusta de
    # una que aprendio fisica.
    print("\n  Siguiente paso: entrenar white-box y variante B sobre este dataset")
    print("  y comparar el R2 de g_phi contra el Delta f verdadero.")


if __name__ == "__main__":
    main()
