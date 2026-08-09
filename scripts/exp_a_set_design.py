#!/usr/bin/env python3
# =============================================================================
#  A (conjunto) — DISENAR EL EXPERIMENTO ENTERO, NO UN ESTIMULO SUELTO
# =============================================================================
#
#  Por que hizo falta esto. Optimizar UN estimulo funciono: bajo la fraccion
#  imitable del 71.4% (el mejor de libreria) al 59.5%. Pero al armar un dataset
#  con 20 variantes de ese unico diseno, la fraccion CONJUNTA subio a 73.9% —
#  peor que el dataset de libreria (67.1%).
#
#  La explicacion es la clave del asunto: cuando se exige un unico δθ que
#  explique TODOS los escenarios a la vez, la degeneracion se rompe sola si los
#  escenarios son DISTINTOS entre si. La libreria tiene 7 familias diferentes
#  (box, cuadrada, APRBS, PRBS, theta-gamma, Poisson, chirp) y esa diversidad
#  hace mucho trabajo gratis. Veinte variantes de la misma forma, no.
#
#  O sea que hay DOS palancas distintas y la segunda pesa mas:
#     (1) que cada estimulo sea individualmente bueno,
#     (2) que los estimulos sean COMPLEMENTARIOS entre si.
#
#  Por eso el objetivo correcto no es por estimulo sino del CONJUNTO: se optimiza
#  un juego de N estimulos para minimizar la fraccion imitable conjunta. Es
#  diseno de experimentos de verdad, no diseno de senal.
#
#  QUE ES LA "FRACCION IMITABLE CONJUNTA". Para cada escenario se mide el hueco
#  instantaneo entre la planta (con refractariedad + actuador optogenetico) y el
#  modelo Wilson-Cowan nominal, Delta f = f_planta - f_WC, evaluado en el MISMO
#  estado. Despues se pregunta cuanto de ese hueco se consigue simplemente
#  moviendo los 10 parametros: con S = df_WC/dtheta (dos filas por punto, una
#  columna por parametro) se resuelve por minimos cuadrados
#
#      Delta f  =  S · dtheta*  +  residuo
#      fraccion imitable = ‖S · dtheta*‖² / ‖Delta f‖²
#
#  Lo CONJUNTO esta en que los puntos de los N escenarios se APILAN en un unico
#  sistema antes de resolver: el dtheta* que sale es UNO SOLO para todos. Si en
#  cambio se midiera la fraccion escenario por escenario y se promediara, saldria
#  la metrica POR ESTIMULO — justo la que llevo al resultado equivocado de arriba.
#  La diferencia es toda la idea: estimulos parecidos comparten la misma direccion
#  de degeneracion, asi que un unico dtheta la imita en todos a la vez (fraccion
#  alta); estimulos complementarios exigen correcciones incompatibles entre si y
#  ningun dtheta unico las cubre (fraccion baja). Bajar esta fraccion es lo que
#  deja algo REAL para que aprenda la correccion neuronal g_phi.
#
#  COMO BUSCA. Cada estimulo son N_SEG escalones (on/off, realizable con
#  optogenetica); un candidato es la matriz (N_STIM, 2*N_SEG) con los tramos de P
#  y de Q de los seis estimulos JUNTOS — se optimiza el experimento entero como un
#  solo objeto. Busqueda (1+lambda) con sigma decreciente, sin gradientes (el
#  objetivo pasa por un integrador y una pseudoinversa). Todos los candidatos se
#  renormalizan a la misma amplitud media que la referencia, para que lo que
#  compita sea la FORMA y no la energia.
#
#  Convencion de tiempo: ms (el regimen del control), igual que gen_multi_dataset.
#
#  USO:  python scripts/exp_a_set_design.py
# =============================================================================

from __future__ import annotations

import json
import sys
from pathlib import Path

# La raiz del repo va al path para poder importar src.*, y la carpeta scripts/
# tambien porque este script reusa a sus HERMANOS como si fueran modulos
# (exp_a_stimulus_design y gen_multi_dataset), en vez de duplicar su codigo.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import numpy as np
import torch

# El trabajo de torch aca es una pseudoinversa de una matriz flaca (~8400 x 10):
# no escala con hilos y, si se corren varias busquedas en paralelo, mas hilos solo
# se pelean por los cores. Con 2 alcanza.
torch.set_num_threads(2)

from src.wilson_cowan import WilsonCowan, WilsonCowanParams, default_uncertainty
from src.neural_ode import GrayBoxWC
from src.neural_ode.graybox_train import projection_operator, projected_fraction, ALL_P

# Se reusa la maquinaria del diseno de UN estimulo: la parametrizacion en escalones
# (escalon), el presupuesto de amplitud (normaliza), el campo WC nominal (campo_wc)
# y la ventana temporal. Lo unico nuevo de este script es la metrica CONJUNTA.
#
# OJO: AMP_MAX_DEF (el techo de amplitud) NO existe hoy en exp_a_stimulus_design —
# ahi el techo es la variable local AMP_MAX = 1.4 dentro de main(). Con lo cual este
# import falla y el script no arranca; el manual lo tiene como "escrito, falta
# correr", asi que nunca se ejecuto. Hay que exportar la constante alla (o definirla
# aca) antes de la primera corrida. No se toca desde este archivo a proposito.
from exp_a_stimulus_design import (
    escalon, normaliza, campo_wc, T_ON, T_OFF, T_SPAN, N_SEG, EPS, AMP_MAX_DEF,
)

OUT = Path("results/uncertainty/a_set_design.json")
FIG = Path("results/figures/a_set_design.png")

# --- Perillas -----------------------------------------------------------------
# Costo: cada evaluacion de un candidato simula los N_STIM escenarios completos,
# asi que el precio total es N_ITER * POB * N_STIM simulaciones (45*6*6 = 1620).
# De ahi que la poblacion y las iteraciones sean chicas y el muestreo, grosero.
N_STIM = 6            # cuantos estimulos tiene el experimento
                      # (6 = el mismo tamano que la referencia de libreria, para
                      #  que la comparacion sea a igual numero de trayectorias)
N_EVAL = 1200         # dt ~0.17 ms durante la busqueda (barato)
N_SUB = 2             # subpasos de RK4 por punto -> paso real ~0.083 ms
                      # OJO: default_uncertainty esta validada con dt ~0.0125 ms y
                      # el actuador tiene tau_act = 1 ms; aca quedan ~12 pasos por
                      # tau, que alcanza para buscar pero no para el numero final.
                      # Por eso el ganador se re-mide con n_eval=3000, n_sub=4.
N_PTS_POR = 700       # puntos por escenario para la proyeccion
                      # 6*700 = 4200 puntos -> Sf de (8400, 10). Se submuestrea
                      # porque la pseudoinversa se hace en cada evaluacion; con la
                      # grilla completa la busqueda no cierra en tiempo razonable.
N_ITER = 45
POB = 6               # (1+lambda): se muta el mejor y se queda el mejor hijo
SIGMA0 = 0.4          # desvio inicial de la mutacion, en unidades de amplitud
SEED = 0

# Modelo de referencia SOLO para sacar sensibilidades. Se instancia en los valores
# VERDADEROS y nunca se entrena: lo unico que se le pide es S = df_WC/dtheta por
# diferencias centradas (ver backbone_sensitivities). Los dos flags en True son
# imprescindibles: un parametro marcado como no aprendible devuelve columna de
# ceros y quedaria fuera de la proyeccion, inflando artificialmente el residuo.
_P0 = WilsonCowanParams()
_TRUE = {k: getattr(_P0, k) for k in ALL_P}
_MODEL = GrayBoxWC(_TRUE, {k: _TRUE[k] for k in ("wEE", "wEI", "wIE", "wII")},
                   learnable_weights=True, learnable_params=True)


# =============================================================================
#  UN ESCENARIO: SIMULAR LA PLANTA Y MEDIR EL HUECO Delta f
# =============================================================================

def traza(Pf, Qf, n_eval=N_EVAL, n_sub=N_SUB, seed=0):
    """Simula un escenario y devuelve (X, P, Q, Delta f) muestreados."""
    # La planta lleva la fisica que el modelo NO tiene (refractariedad del WC de
    # 1972 + actuador optogenetico con retardo y saturacion). EPS gradua cuanta
    # falta; EPS=1.0 es el punto nominal del roadmap.
    pert = default_uncertainty(EPS)
    m = WilsonCowan(params=_P0, P=Pf, Q=Qf, perturbation=pert)
    t = np.linspace(T_SPAN[0], T_SPAN[1], n_eval)
    sol = m.simulate(I0=0.0, E0=0.0, t_span=T_SPAN, t_eval=t, n_sub=n_sub)
    # Pc, Qc son los estimulos COMANDADOS (no los efectivos): es lo unico que
    # conoceria un experimento real y lo unico que puede ver el modelo.
    # 'extra' son los estados ocultos de la perturbacion (estado del actuador,
    # variable de refractariedad). No se miden; hacen falta solo para poder
    # reevaluar el campo verdadero en el punto exacto de la trayectoria.
    I, E, Pc, Qc, extra = sol["I"], sol["E"], sol["P"], sol["Q"], sol["extra"]
    # Campo NOMINAL: lo que predeciria el Wilson-Cowan puro en ese mismo estado.
    nomI, nomE = campo_wc(I, E, Pc, Qc)
    dfI = np.zeros(n_eval); dfE = np.zeros(n_eval)
    for k in range(n_eval):
        # Delta f punto a punto = campo verdadero - campo nominal EN EL MISMO
        # estado. Restar asi (y no comparar trayectorias) aisla el mismatch
        # instantaneo, sin que se mezcle la acumulacion del error en el tiempo.
        d = m.rhs_aug(float(t[k]), np.concatenate(([I[k], E[k]], extra[k])))
        dfI[k] = d[0] - nomI[k]; dfE[k] = d[1] - nomE[k]
    # Submuestreo ALEATORIO y no cada-k-puntos: los estimulos son periodicos o
    # casi (onda cuadrada, theta-gamma) y un submuestreo regular puede aliasarse
    # con el periodo y quedarse siempre en la misma fase de la trayectoria.
    # La semilla la fija quien llama -> reproducible, y distinta por escenario.
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_eval, min(N_PTS_POR, n_eval), replace=False)
    return (np.stack([I[idx], E[idx]], 1), Pc[idx], Qc[idx],
            np.stack([dfI[idx], dfE[idx]], 1))


# =============================================================================
#  LA METRICA DEL CONJUNTO (el aporte de este script)
# =============================================================================

def evalua_conjunto(stims, n_eval=N_EVAL, n_sub=N_SUB) -> dict:
    """Fraccion imitable CONJUNTA: un unico δθ tiene que explicar TODOS los
    escenarios a la vez. Es la metrica que realmente importa para un dataset."""
    Xs, Ps, Qs, DFs = [], [], [], []
    for i, (Pf, Qf) in enumerate(stims):
        # seed=i: cada escenario submuestrea instantes distintos, asi la nube de
        # puntos no queda correlacionada entre escenarios. Es deterministico
        # respecto del INDICE, no del contenido: dos candidatos con el mismo
        # estimulo en la posicion i se comparan sobre los mismos instantes.
        X, P, Q, DF = traza(Pf, Qf, n_eval, n_sub, seed=i)
        Xs.append(X); Ps.append(P); Qs.append(Q); DFs.append(DF)
    # ACA esta todo el asunto: los puntos de los N escenarios se apilan en UNA
    # sola nube antes de proyectar. Con eso el sistema de minimos cuadrados tiene
    # 10 incognitas (un solo δθ) y 2*N_STIM*N_PTS_POR ecuaciones que vienen de
    # trayectorias distintas. Si en vez de apilar se llamara a projected_fraction
    # por escenario y se promediara, cada uno podria usar SU propio δθ y saldria
    # la metrica por estimulo — la que hace creer que 20 clones de un buen diseno
    # son un buen dataset.
    X = torch.tensor(np.vstack(Xs), dtype=torch.float32)
    P = torch.tensor(np.concatenate(Ps).reshape(-1, 1), dtype=torch.float32)
    Q = torch.tensor(np.concatenate(Qs).reshape(-1, 1), dtype=torch.float32)
    DF = torch.tensor(np.vstack(DFs), dtype=torch.float32)
    # S (N,2,10) por diferencias centradas; projection_operator la aplana a
    # (2N,10) y devuelve la pseudoinversa (no ridge: borraria justo las
    # direcciones debiles, que son las que la red aprende a robar).
    S = _MODEL.backbone_sensitivities(X, P, Q)
    A, Sf = projection_operator(S)
    _, frac = projected_fraction(DF, A, Sf)
    # Se recalcula c = A·vec(Delta f) para poder reportar tambien el RESIDUO en
    # valor absoluto (projected_fraction solo devuelve la parte proyectada).
    # Los tres numeros hacen falta juntos: la fraccion sola se puede bajar
    # haciendo el hueco chiquito, y entonces no hay nada que aprender.
    c = A @ DF.reshape(-1)
    res = DF - (Sf @ c).reshape(-1, 2)
    return {"frac_imitable": float(frac),
            "df_rms": float(DF.pow(2).mean().sqrt()),   # tamano del hueco total
            "residuo": float(res.pow(2).mean().sqrt())}  # lo NO imitable = lo util


# =============================================================================
#  BUSQUEDA: REFERENCIA, PRESUPUESTO Y EVOLUCION
# =============================================================================

def main():
    # Import adentro de main a proposito: gen_multi_dataset es un script de
    # generacion de datos y no hace falta arrastrarlo si alguien importa este
    # archivo solo para usar evalua_conjunto.
    from gen_multi_dataset import build_scenarios
    rng = np.random.default_rng(SEED)

    # --- Referencia: 6 estimulos de libreria, los mas diversos posibles -----
    # La eleccion NO es al azar: uno por FAMILIA (APRBS, cuadrada, PRBS,
    # theta-gamma, Poisson, chirp), que es la configuracion mas dificil de batir
    # justamente porque la diversidad entre familias ya rompe degeneracion gratis.
    # Compararse contra 6 variantes de la misma familia seria hacerse trampa.
    todos = build_scenarios()
    quiero = ["aprbs_0", "square_a1.0_f50", "prbs_0", "thetagamma_0",
              "poisson_0", "chirp"]
    ref_stims = [(Pf, Qf) for lab, Pf, Qf, _ in todos if lab in quiero]
    ref = evalua_conjunto(ref_stims)
    print("=== A (conjunto) · disenar el EXPERIMENTO, no un estimulo ===\n")
    print(f"  Referencia: {len(ref_stims)} estimulos de libreria, familias distintas")
    print(f"    imitable CONJUNTA = {100*ref['frac_imitable']:.1f}%   "
          f"|Df| = {ref['df_rms']:.5f}   residuo = {ref['residuo']:.5f}\n")

    # presupuesto de amplitud: el de la referencia
    # Por que igualar: un estimulo mas fuerte excita mas no linealidad y agranda el
    # residuo por la via facil. Si el conjunto disenado gana con MAS energia, no se
    # demostro nada sobre el diseno. Aca se promedia sobre los 6 estimulos y sobre
    # la ventana [T_ON, T_OFF) -> un unico numero por canal.
    ts = np.linspace(T_ON, T_OFF, 200)
    mp = float(np.mean([[Pf(t) for t in ts] for Pf, _ in ref_stims]))
    mq = float(np.mean([[Qf(t) for t in ts] for _, Qf in ref_stims]))
    print(f"  Presupuesto de amplitud media (igualado): P={mp:.3f}  Q={mq:.3f}\n")

    def arma(v):
        """v: (N_STIM, 2*N_SEG) -> lista de (Pf, Qf) con amplitud media fijada."""
        # Cada mitad de la fila es un canal: los primeros N_SEG tramos son P (a la
        # poblacion E) y los ultimos N_SEG son Q (a la poblacion I). P y Q se
        # normalizan por separado porque sus presupuestos son distintos.
        #
        # OJO con lo que este presupuesto le PROHIBE al diseno: normaliza fija la
        # media de CADA estimulo en mp/mq, o sea que los seis salen con la misma
        # energia. La libreria no: ahi hay estimulos flojos y fuertes, y esa
        # dispersion de amplitud es en si misma una fuente de diversidad (la
        # sigmoidea responde distinto en cada punto de operacion). El conjunto
        # disenado tiene que conseguir toda su diversidad de la FORMA. Es una
        # comparacion honesta pero conservadora: le ata una mano.
        out, segs = [], []
        for k in range(N_STIM):
            p = normaliza(v[k, :N_SEG], mp, AMP_MAX_DEF)
            q = normaliza(v[k, N_SEG:], mq, AMP_MAX_DEF)
            # Se devuelven las dos vistas: las funciones f(t) para simular y los
            # vectores de tramos para poder graficarlos y guardarlos en el JSON.
            out.append((escalon(p), escalon(q))); segs.append((p, q))
        return out, segs

    # Arranque ALEATORIO (a diferencia del script de un estimulo, que arranca de la
    # APRBS de libreria discretizada). Es deliberado: sembrar con la libreria
    # empuja hacia seis variantes de una misma forma, que es exactamente el error
    # que este script trata de evitar. El costo es que hay que llegar solo, con
    # N_ITER*POB = 270 evaluaciones; si la historia no baja, faltan iteraciones,
    # no es que el conjunto no exista.
    mejor_v = rng.uniform(0, AMP_MAX_DEF, size=(N_STIM, 2 * N_SEG))
    mejor = evalua_conjunto(arma(mejor_v)[0])
    print(f"  {'iter':>5} {'imitable':>10} {'|Df|':>10} {'residuo':>10}")
    print(f"  {'init':>5} {100*mejor['frac_imitable']:9.1f}% {mejor['df_rms']:10.5f} "
          f"{mejor['residuo']:10.5f}")

    hist, sigma, quieto = [], SIGMA0, 0
    for it in range(N_ITER):
        # (1+lambda): POB hijos gaussianos alrededor del mejor, recortados al rango
        # fisico [0, AMP_MAX_DEF] (el estimulo optogenetico no puede ser negativo
        # ni pasarse de la potencia del laser). El clip antes de normaliza importa:
        # si no, un tramo negativo bajaria la media y normaliza compensaria
        # subiendo todos los demas.
        cands = [np.clip(mejor_v + rng.normal(0, sigma, mejor_v.shape),
                         0, AMP_MAX_DEF) for _ in range(POB)]
        res = [evalua_conjunto(arma(c)[0]) for c in cands]
        # Se MINIMIZA la fraccion imitable, que es el test directo de "romper la
        # degeneracion". OJO: aca no hay penalizacion por |Delta f| chico como en
        # exp_a_stimulus_design (donde el objetivo 'fraccion' exige |Df| >= el de
        # la referencia). El unico freno contra bajar la fraccion achicando el
        # hueco es el presupuesto de amplitud. Por eso se imprimen |Df| y residuo
        # en cada tramo: si |Df| se derrumba, la mejora no sirve.
        j = int(np.argmin([r["frac_imitable"] for r in res]))
        if res[j]["frac_imitable"] < mejor["frac_imitable"]:
            mejor, mejor_v, quieto = res[j], cands[j], 0
        else:
            quieto += 1
            # Recocido simple: si 4 generaciones seguidas no mejoran, es que sigma
            # es mas grande que la escala del optimo local -> se afina el paso.
            # 0.65 y 4 son heuristicos; con 270 evaluaciones no conviene enfriar
            # mas rapido porque la busqueda se congela antes de explorar.
            if quieto >= 4:
                sigma *= 0.65; quieto = 0
        hist.append(mejor["frac_imitable"])
        if it % 5 == 0 or it == N_ITER - 1:
            print(f"  {it:5d} {100*mejor['frac_imitable']:9.1f}% "
                  f"{mejor['df_rms']:10.5f} {mejor['residuo']:10.5f}", flush=True)

    # Validacion del ganador con el integrador FINO (paso ~0.017 ms, ya dentro del
    # rango validado de la perturbacion) y mas puntos. Es necesaria porque durante
    # la busqueda el paso es grosero y parte de la mejora podria ser error de
    # discretizacion en vez de fisica.
    # TRAMPA que queda abierta: 'ref' se midio con la resolucion GROSERA y aca se
    # compara contra 'fino'. Los dos numeros de la tabla final no salen del mismo
    # integrador; para una comparacion estricta habria que re-medir la referencia
    # con n_eval=3000, n_sub=4. (La curva de historia si es comparable con la
    # linea de la libreria, porque ambas usan la resolucion de busqueda.)
    stims, segs = arma(mejor_v)
    fino = evalua_conjunto(stims, n_eval=3000, n_sub=4)

    print("\n=== RESULTADO (conjunto de 6 estimulos, amplitud media igualada) ===")
    print(f"  {'':26} {'imitable':>11} {'|Delta f|':>11} {'residuo':>11}")
    print(f"  {'6 familias de libreria':26} {100*ref['frac_imitable']:10.1f}% "
          f"{ref['df_rms']:11.5f} {ref['residuo']:11.5f}")
    print(f"  {'6 DISENADOS en conjunto':26} {100*fino['frac_imitable']:10.1f}% "
          f"{fino['df_rms']:11.5f} {fino['residuo']:11.5f}")
    # Dos lecturas del mismo resultado: d_frac en PUNTOS PORCENTUALES (cuanta
    # degeneracion se rompio; positivo = el conjunto disenado es mejor) y d_res en
    # PORCENTAJE relativo (cuanta fisica aprovechable gano g_phi). Pueden moverse
    # en sentidos distintos, y por eso se reportan las dos.
    d_frac = 100 * (ref["frac_imitable"] - fino["frac_imitable"])
    d_res = 100 * (fino["residuo"] / ref["residuo"] - 1)
    print(f"\n  Fraccion imitable: {d_frac:+.1f} puntos")
    print(f"  Residuo aprovechable: {d_res:+.1f}%")

    # Se guardan los SEGMENTOS, no las funciones: con (p, q) por estimulo se puede
    # reconstruir el conjunto exacto con escalon() y alimentar gen_multi_dataset
    # para entrenar con este dataset. Es el entregable real del script.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "referencia": ref, "disenado": fino,
        "delta_frac_pp": d_frac, "delta_residuo_pct": d_res,
        "segmentos": [[p.tolist(), q.tolist()] for p, q in segs],
        "n_stim": N_STIM, "n_seg": N_SEG, "amp_media_P": mp, "amp_media_Q": mq,
        "historia": hist,
    }, indent=2))

    # --- Figura: los 6 estimulos + la curva de convergencia -------------------
    import matplotlib
    matplotlib.use("Agg")      # backend sin pantalla (corre en servidor / por lote)
    import matplotlib.pyplot as plt
    # Bordes izquierdos de los tramos; se descarta el ultimo borde porque step()
    # con where="post" dibuja un escalon por valor, no por borde.
    b = np.linspace(T_ON, T_OFF, N_SEG + 1)[:-1]
    # Grilla 2x4 = 8 paneles: 6 estimulos + convergencia + uno vacio. Los indices
    # 6 y 7 de mas abajo estan atados a N_STIM=6: si se cambia N_STIM hay que
    # rehacer la grilla (con N_STIM=8 los paneles de convergencia se pisarian).
    fig, ax = plt.subplots(2, 4, figsize=(17, 6))
    for k in range(N_STIM):
        a = ax.flat[k]
        a.step(b, segs[k][0], where="post", label="P", color="#1f4e79")
        a.step(b, segs[k][1], where="post", label="Q", color="#d62728")
        a.set_title(f"estimulo disenado {k+1}", fontsize=9)
        a.set_xlabel("t [ms]"); a.grid(alpha=0.3)
        if k == 0:
            a.legend(fontsize=7)
    # Convergencia contra la linea de la libreria: si la curva azul no cruza para
    # abajo la linea roja, el conjunto disenado NO le gano a las 6 familias.
    ax.flat[6].plot([100 * h for h in hist], color="#1f4e79")
    ax.flat[6].axhline(100 * ref["frac_imitable"], ls="--", color="#d62728",
                       label="libreria")
    ax.flat[6].set_title("fraccion imitable conjunta [%]", fontsize=9)
    ax.flat[6].legend(fontsize=7); ax.flat[6].grid(alpha=0.3)
    ax.flat[7].axis("off")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=120); plt.close(fig)
    print(f"\n  -> {OUT}")


if __name__ == "__main__":
    main()
