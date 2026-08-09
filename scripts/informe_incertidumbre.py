#!/usr/bin/env python3
# =============================================================================
#  INFORME CONSOLIDADO — incertidumbre dinamica y gray-box
# =============================================================================
#
#  Junta todo lo que produjeron las fases F1..F7 y arma las tablas y figuras
#  que van al documento. No entrena nada: solo lee los .json de results/.
#
#  USO:  python scripts/informe_incertidumbre.py
#
#  QUE ES ESTO EN UNA FRASE: el ultimo eslabon de la cadena. Cada script
#  exp_fN_*.py corre un experimento y deja su resultado en un .json; este
#  archivo los lee todos, los ordena y los imprime como el informe de texto
#  que se copia al documento (results/uncertainty/informe.txt).
#
#  DE QUE VA LA HISTORIA (para que las tablas se entiendan):
#    El simulador tiene fisica que el modelo NO contempla (refractariedad del
#    WC de 1972 + un actuador optogenetico con retardo y saturacion). La perilla
#    eps gradua cuanta fisica falta: eps=0 -> el modelo es exacto; eps=1 -> el
#    hueco vale un tercio del campo de Wilson-Cowan. La pregunta del trabajo es
#    si un termino de correccion neuronal g_phi puede tapar ese hueco sin
#    arruinar la identificacion de los 10 parametros.
#
#  UNA FUNCION POR FASE, y cada una contesta una pregunta distinta:
#    f1 · el hueco esta bien calibrado?            <- f1_caracterizacion.json
#    f2 · cuanto cuesta ignorarlo (white-box)?     <- f2_eps*.json
#    f3 · la correccion ayuda o tapa parametros?   <- f3_*.json
#    f4 · por donde le roba a los parametros?      <- f4_fim.json
#    f5 · aprendio la fisica correcta?             <- f5_recovery.json
#    f6 · sirve para controlar?                    <- f6_closed_loop.json
#    f7 · donde deja de alcanzar?                  <- f7_controls.json
#
#  TOLERA RESULTADOS PARCIALES: si falta un .json, esa seccion simplemente no
#  se imprime (ver carga()). Asi se puede mirar el informe a mitad de camino,
#  sin esperar a que termine la cadena entera (scripts/run_uncertainty_all.sh).
#
#  Referencias: docs/graybox_manual_completo.md (los mismos numeros, con la
#  discusion completa) y docs/recorrido_estimulos_y_entrenamiento.md.
# =============================================================================

from __future__ import annotations

import json
import sys
from pathlib import Path

# Agregamos la raiz del repo al path para poder importar el paquete src/ (lo
# usan otros scripts de la familia; aca se conserva para que todos arranquen
# igual). OJO: las rutas de resultados de abajo SI son relativas, asi que este
# informe hay que correrlo DESDE LA RAIZ del repo.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np

# Rutas ANCLADAS A LA RAIZ del repo, no relativas al directorio de trabajo.
# Con rutas relativas, correr el informe desde otra carpeta hacia que carga()
# devolviera None para todo, cada seccion saliera por su "if not d: return" y el
# script imprimiera NADA terminando con exit 0: un fallo perfectamente silencioso.
RES = _ROOT / "results" / "uncertainty"   # donde cada exp_fN_*.py dejo su .json
FIG = _ROOT / "results" / "figures"       # donde va la figura resumen

# Orden canonico de los 10 parametros de Wilson-Cowan. Es el MISMO orden que
# usan fisher_identifiability.py y los scripts de las fases, y por eso las
# columnas de la tabla "error por parametro" se pueden comparar entre fases.
# Si se cambia el orden aca, las tablas dejan de alinearse con el resto.
ALL_P = ("wEE", "wEI", "wIE", "wII", "te", "ti", "ae", "ai", "thetae", "thetai")


# =============================================================================
#  UTILIDADES DE LECTURA E IMPRESION
# =============================================================================

# Lee un .json de results/uncertainty. Devuelve None si no existe en vez de
# explotar: es lo que permite armar el informe con lo que haya listo.
def carga(nombre):
    p = RES / nombre
    return json.loads(p.read_text()) if p.exists() else None


# Titulo de seccion con el separador de 78 "=" (mismo ancho que los bloques de
# comentario del proyecto, para que el informe.txt quede prolijo).
def seccion(t):
    print(f"\n{'='*78}\n  {t}\n{'='*78}")


# ---------------------------------------------------------------------------
# F1 · ESTA BIEN CALIBRADA LA PERTURBACION?
# ---------------------------------------------------------------------------
# Antes de sacar conclusiones hay que mostrar que la perilla eps hace algo
# razonable: que degrade de forma suave y monotona, que el hueco se vea POR
# ENCIMA del ruido de observacion, y que NO destruya el regimen del sistema
# (si el ciclo limite desapareciera, ya no estariamos midiendo lo mismo).
#
# Columnas:
#   D_traj%    deformacion de la trayectoria contra el caso eps=0, en % de |y|.
#   D_abs      la misma deformacion en unidades absolutas de y.
#   xSigma     D_abs dividido el ruido de observacion tipico de los barridos
#              del proyecto (sigma=0.02, fijado en exp_f1_characterize.py).
#              >1 = el hueco estructural es mas grande que el ruido; si fuera
#              <1 el experimento no tendria sentido: el mismatch quedaria
#              enterrado bajo el ruido.
#   |Df|/|f|%  tamano del termino faltante respecto del campo de Wilson-Cowan.
#              Es la medida honesta de "cuanta fisica falta" (34% en eps=1).
#   E_max      pico de actividad excitatoria, y
#   cruces     cuantas veces y(t) cruza su media -> juntos dicen que el ciclo
#              limite se DEFORMA pero no desaparece (los cruces se mantienen
#              en 21 hasta eps=1.5).
def f1():
    d = carga("f1_caracterizacion.json")
    if not d:
        return None
    seccion("F1 · Caracterizacion: cuanto deforma la perturbacion")
    print(f"  {'eps':>5} {'D_traj%':>9} {'D_abs':>8} {'xSigma':>8} "
          f"{'|Df|/|f|%':>11} {'E_max':>8} {'cruces':>8}")
    for r in d:
        print(f"  {r['eps']:5.2f} {r['D_traj_pct']:9.2f} {r['D_abs']:8.4f} "
              f"{r['veces_sigma']:8.1f} {r['df_rel_pct']:11.2f} "
              f"{r['E_max']:8.3f} {r['cruces']:8d}")
    return d


# ---------------------------------------------------------------------------
# F2 · QUE CUESTA LA RIGIDEZ
# ---------------------------------------------------------------------------
# Se identifican los 10 parametros con el white-box de siempre (sin correccion)
# sobre datos cada vez mas perturbados. Es el precio de insistir en que el
# modelo es exacto cuando no lo es.
#
# Un .json por nivel de eps (f2_eps0.json, f2_eps0.25.json, ...), por eso hay
# que juntarlos con glob y reordenarlos: el orden alfabetico de los nombres NO
# es el orden numerico de eps ("f2_eps1.5" cae antes que "f2_eps2").
#
# COMO LEERLO:
#   - eps=0 es el CONTROL POSITIVO: tiene que dar ~1% de error medio, que es el
#     resultado ya conocido del proyecto. Si no da eso, el problema esta en la
#     maquinaria nueva y no en la perturbacion.
#   - mse_train / mse_test no son un buen termometro del mismatch: no crecen de
#     forma monotona con eps (eps=2 llega a dar un mse mas chico que eps=1) y
#     sin embargo el error parametrico se triplica. Es el aviso de fondo del
#     trabajo: un ajuste bueno no garantiza parametros buenos.
def f2():
    filas = []
    for p in sorted(RES.glob("f2_eps*.json")):
        filas.append(json.loads(p.read_text()))
    if not filas:
        return None
    filas.sort(key=lambda f: f["eps"])
    seccion("F2 · El costo de la rigidez (white-box sobre planta perturbada)")
    print(f"  {'eps':>5} {'err_max%':>10} {'err_medio%':>12} {'peor':>8} "
          f"{'mse_train':>12} {'mse_test':>12}")
    for f in filas:
        print(f"  {f['eps']:5.2f} {f['max_param_error']:10.2f} "
              f"{f['mean_param_error']:12.2f} {f['peor_param']:>8} "
              f"{f['mse_train']:12.3e} {f['mse_test']:12.3e}")

    # Desglose parametro por parametro: muestra que los PESOS sinapticos se
    # rompen mucho mas que la forma de la sigmoidea. Tiene sentido fisico: la
    # refractariedad multiplica la salida de la sigmoidea, y eso se imita facil
    # reescalando pesos.
    print("\n  Error por parametro (%):")
    print("  " + "eps".rjust(5) + "".join(f"{k:>9}" for k in ALL_P))
    for f in filas:
        print(f"  {f['eps']:5.2f}" +
              "".join(f"{f['param_errors'][k]:9.1f}" for k in ALL_P))

    # El resultado mas fuerte de la fase: en TODOS los niveles el peor
    # parametro es el mismo (wII). Solo se afirma si el conjunto de peores
    # tiene un unico elemento; si algun nivel se sale del patron, la conclusion
    # no se imprime en vez de forzarla.
    peores = [f["peor_param"] for f in filas]
    print(f"\n  Parametro mas castigado en cada nivel: {peores}")
    if len(set(peores)) == 1:
        print(f"  -> SIEMPRE es {peores[0]}: el mismatch estructural degrada "
              f"preferentemente\n     la direccion que la FIM ya marcaba como mas debil.")
    return filas


# ---------------------------------------------------------------------------
# F3 · ENCENDER LA CORRECCION: AYUDA O TAPA PARAMETROS?
# ---------------------------------------------------------------------------
# Variantes que se comparan (una por .json f3_<variante>_eps<eps>[_lam<lam>]):
#   A  g(I,E,P,Q) libre        B  g(I,E) (sin ver el estimulo)
#   C  g(I,E) + lam*||g||^2    D  penalizacion ortogonal (castiga que g sea
#                                 redundante con un cambio de theta)
#   S  correccion estructurada
#
# LA TRAMPA QUE ESTA SECCION EXISTE PARA DETECTAR: si el mse baja y el error
# parametrico sube, la red no aprendio fisica; le esta tapando la boca a
# parametros mal estimados. Por eso las dos metricas van SIEMPRE juntas.
#
# Las dos ultimas columnas son las que no dependen del ajuste:
#   g_rel     = |g| / |f_WC|, el tamano de la correccion. Con lam grande tiende
#               a 0: la red queda anulada.
#   redund.   = frac_redundante, cuanto de lo que hace g lo podria haber hecho
#               un cambio de theta. OJO CON EL PUNTO CERO: no es 0. La fisica
#               real (el Delta f verdadero) puntua 0.67 y una red sin entrenar
#               ~0.47, asi que el objetivo NO es minimizarlo sino acercarlo a
#               0.67. Comparar el 0.94 aprendido contra 0 exagera la patologia.
def f3():
    filas = [json.loads(p.read_text()) for p in sorted(RES.glob("f3_*.json"))]
    if not filas:
        return None
    seccion("F3 · Gray-box: ¿mejora el ajuste a costa de los parametros?")
    print("  LEER LAS DOS METRICAS JUNTAS. Si el mse baja y el error parametrico")
    print("  sube, la red esta tapando parametros malos, no aprendiendo fisica.\n")
    print(f"  {'variante':10} {'eps':>5} {'lam':>7} {'err_max%':>10} "
          f"{'err_med%':>10} {'mse_test':>11} {'g_rel':>8} {'redund.':>9}")
    # El "or 0" del criterio de orden es solo para que las variantes sin
    # regularizacion (lam=None) no rompan la comparacion; no cambia nada.
    for f in sorted(filas, key=lambda x: (x["eps"], x["variant"], x["lam"] or 0)):
        lam = "-" if f["lam"] is None else f"{f['lam']:g}"
        # g_rel y frac_redundante son opcionales: las corridas viejas no los
        # traen, y ahi se imprime "-" en lugar de fallar.
        gr = f.get("g_rel")
        fr = f.get("frac_redundante")
        print(f"  {f['variant']:10} {f['eps']:5.2f} {lam:>7} "
              f"{f['max_param_error']:10.2f} {f['mean_param_error']:10.2f} "
              f"{f['mse_test']:11.3e} "
              f"{('-' if gr is None else f'{gr:8.3f}')} "
              f"{('-' if fr is None else f'{fr:9.3f}')}")

    # comparacion contra el white-box del mismo eps
    # Sin esta referencia los numeros de arriba no dicen nada: 31% de error
    # parametrico es malisimo con eps=0 y muy bueno con eps=1.
    for eps in sorted({f["eps"] for f in filas}):
        # El formato :g es lo que hace coincidir el nombre del archivo: 1.0
        # tiene que quedar "f2_eps1.json". Con str(1.0) saldria "f2_eps1.0" y
        # la comparacion se saltearia en silencio.
        wb = carga(f"f2_eps{eps:g}.json")
        if not wb:
            continue
        sub = [f for f in filas if f["eps"] == eps]
        # Los dos ganadores no tienen por que ser el mismo: uno optimiza los
        # parametros y el otro el ajuste. Cuando difieren, ahi esta el conflicto.
        mejor_par = min(sub, key=lambda f: f["mean_param_error"])
        mejor_mse = min(sub, key=lambda f: f["mse_test"])
        def etiq(f):
            return f["variant"] if f["lam"] is None else f"{f['variant']}(lam={f['lam']:g})"

        print(f"\n  --- eps = {eps} ---")
        print(f"  white-box            err_med={wb['mean_param_error']:7.2f}%  "
              f"mse_test={wb['mse_test']:.3e}")
        print(f"  mejor en PARAMETROS  {etiq(mejor_par):>12}  "
              f"err_med={mejor_par['mean_param_error']:7.2f}%  "
              f"mse_test={mejor_par['mse_test']:.3e}")
        print(f"  mejor en AJUSTE      {etiq(mejor_mse):>12}  "
              f"err_med={mejor_mse['mean_param_error']:7.2f}%  "
              f"mse_test={mejor_mse['mse_test']:.3e}")
        # El resultado central del trabajo se lee comparando esta linea entre
        # eps=0 y eps=1: con eps=0 (modelo ya correcto) NINGUNA variante mejora
        # los parametros -> la correccion hace dano. Con eps=1 recupera ~la
        # mitad del error. O sea: g_phi sirve justo cuando el modelo esta mal.
        if mejor_par["mean_param_error"] < wb["mean_param_error"]:
            g = 100 * (1 - mejor_par["mean_param_error"] / wb["mean_param_error"])
            print(f"  -> el gray-box RECUPERA {g:.0f}% del error parametrico")
        else:
            print("  -> ninguna variante mejora los parametros respecto del white-box")
    return filas


# ---------------------------------------------------------------------------
# F4 · POR DONDE LE ROBA g_phi A LOS PARAMETROS
# ---------------------------------------------------------------------------
# Hipotesis: la correccion entra por las direcciones MAS DEBILES de la matriz
# de Fisher, porque son las mas baratas (mover el modelo ahi casi no cambia la
# trayectoria, asi que la red lo puede hacer casi gratis).
#
# Que hay en el .json:
#   sing_norm  espectro singular normalizado (sigma_i/sigma_1) del jacobiano
#              relativo de la trayectoria respecto de theta.
#   V          las direcciones singulares: V[i] es un vector de 10 componentes
#              en el espacio de parametros relativos. Las ULTIMAS son las
#              debiles, de ahi el recorrido al reves (len-1, len-2, len-3).
#   cond       sigma1/sigma10, el condicionamiento del problema inverso.
def f4():
    d = carga("f4_fim.json")
    if not d:
        return None
    seccion("F4 · FIM del hibrido: por donde roba identificabilidad g_phi")
    for r in d:
        sn = r["sing_norm"]
        print(f"\n  dataset: {Path(r['data']).name}   condicion = {r['cond']:.2e}")
        V = np.array(r["V"]); P = r["pnames"]
        # Las 3 direcciones mas debiles, de la ultima hacia atras.
        for i in (len(sn) - 1, len(sn) - 2, len(sn) - 3):
            # Ordenamos por peso ABSOLUTO y mostramos las 3 componentes que mas
            # pesan: es la forma de ponerle nombre a una direccion (sigma10
            # resulta ser "wII", con trazas de ai y ti).
            o = np.argsort(-np.abs(V[i]))
            print(f"    sigma{i+1:2d} rel={sn[i]:8.2e}  " +
                  " ".join(f"{P[j]}={V[i][j]:+.2f}" for j in o[:3]))
        # Este bloque solo aparece si a la fase se le paso un checkpoint con
        # correccion entrenada (exp_f4_fim_hybrid.py --ckpt ...).
        if "frac_explicable" in r:
            print(f"    correccion: {Path(r['ckpt']).name}")
            # ~0.94: casi todo lo que hace la red equivale a mover theta.
            print(f"    fraccion de g EXPLICABLE por un cambio de theta = "
                  f"{r['frac_explicable']:.3f}")
            # El azar es 30% porque son 3 de las 10 direcciones. Medir 99.5%
            # confirma la hipotesis: la FIM predice por donde va a entrar la
            # correccion SIN entrenar nada.
            print(f"    energia robada en las 3 direcciones mas debiles = "
                  f"{100*r['energia_3_mas_debiles']:.1f}%   (azar = 30%)")
    return d


# ---------------------------------------------------------------------------
# F5 · APRENDIO LA RED LA FISICA CORRECTA?
# ---------------------------------------------------------------------------
# Esta es la pregunta que el mse NO contesta. Aca se compara la correccion
# aprendida contra el Delta f VERDADERO, que solo conocemos porque nosotros
# escribimos el simulador.
#
# Columnas:
#   |g| vs |Df|  tamano de la correccion contra tamano del hueco real. Suelen
#                coincidir (0.0170 vs 0.0172): aprende algo del tamano correcto.
#   R2_test      pero el R2 contra el Delta f real es NEGATIVO: la forma esta
#                mal. Peor que predecir la media.
#   techo        el oraculo: la mejor correccion posible con esos MISMOS
#                argumentos, ajustada directamente contra el Delta f real.
#                Separa "no se puede representar" de "el entrenamiento no lo
#                encontro".
#   R2_lejos     el mismo R2 fuera del dominio visto (extrapolacion).
def f5():
    d = carga("f5_recovery.json")
    if not d:
        return None
    seccion("F5 · ¿Aprendio la red la fisica correcta?")
    print(f"  {'modelo':22} {'|g|':>8} {'|Df|':>8} {'R2_test':>9} "
          f"{'techo':>8} {'aprov.':>9} {'R2_lejos':>10}")
    for r in d:
        # El "aprovechamiento del techo" solo tiene sentido si el techo es
        # positivo. Si el oraculo mismo da R2<=0, el Delta f NO ES FUNCION de los
        # argumentos de g: no hay techo, y el cociente da numeros sin sentido.
        # El 0.05 es un margen: con techos casi nulos el cociente ya explota.
        ap = (f"{100*r['r2_test']/r['oraculo_test']:8.1f}%"
              if r["oraculo_test"] > 0.05 else "sin techo")
        print(f"  {r['tag'][:22]:22} {r['g_rms']:8.4f} {r['df_rms']:8.4f} "
              f"{r['r2_test']:9.3f} {r['oraculo_test']:8.3f} {ap:>9} "
              f"{r['r2_lejos']:10.3f}")
    # El caso g(I,E) da "sin techo" y eso NO es un fallo del entrenamiento: la
    # mitad del hueco es el actuador, que entra por P,Q y arrastra su propio
    # estado interno. Con P,Q entre los argumentos el techo sube a 0.775.
    print("\n  'techo' = mejor correccion sin memoria posible (oraculo ajustado al Df real).")
    print("  'sin techo' = el oraculo tampoco puede: el Df no es funcion de esos argumentos,")
    print("  asi que NINGUNA correccion con esos argumentos lo puede recuperar.")
    return d


# ---------------------------------------------------------------------------
# F6 · SIRVE PARA CONTROLAR?
# ---------------------------------------------------------------------------
# QUIEN ES QUIEN, porque se presta a confusion: la PLANTA es siempre la misma
# (el simulador con la perturbacion). Lo unico que cambia entre filas es con
# QUE CONOCIMIENTO se armo el controlador: los theta estimados y, opcionalmente,
# la correccion g restada del objetivo ("cancelacion").
#
# Las filas vienen numeradas desde exp_f6_closed_loop.py: 0. planta limpia,
# 1. oraculo (theta verdaderos), 2. white-box, 3a. gray-box sin cancelar g,
# 3b. gray-box cancelando g.
def f6():
    d = carga("f6_closed_loop.json")
    if not d:
        return None
    seccion("F6 · Lazo cerrado contra la planta perturbada")
    print(f"  {'controlador construido con':38} {'RMSE_I':>9} {'RMSE_E':>9}")
    for f in d["filas"]:
        # El aviso DIVERGIO importa: un g mal apuntado produce igual un RMSE
        # finito de aspecto sano, y sin la marca pasaria inadvertido.
        print(f"  {f['nombre'][:38]:38} {f['rmse_I']:9.4f} {f['rmse_E']:9.4f}"
              + ("" if f["ok"] else "   DIVERGIO"))
    # Indexamos las filas por los 2 primeros caracteres del nombre ("0.", "1.",
    # "2.", "3a", "3b") para poder restar pares concretos. OJO: es un dict, asi
    # que si alguna vez el .json trae DOS filas 3a/3b (por ejemplo la variante B
    # y la D en la misma corrida), la ultima tapa a la anterior en silencio.
    filas = {f["nombre"][:2]: f for f in d["filas"]}
    # Cuanto cuesta el hueco estructural POR SI SOLO, con los parametros
    # verdaderos: ninguna identificacion puede arreglar esta parte (~+100%).
    if "0." in filas and "1." in filas:
        base, orac = filas["0."]["rmse_E"], filas["1."]["rmse_E"]
        print(f"\n  Costo de la perturbacion en si (theta verdaderos): "
              f"{base:.4f} -> {orac:.4f}  ({100*(orac/base-1):+.0f}%)")
    # Y esto es lo que agrega la correccion cuando se la usa para cancelar.
    # Cuidado al leerlo: mira solo el canal E. La conclusion practica de la
    # fase es que el lazo de E mejora pero el de I se destruye (mirar la tabla
    # de arriba, no solo este porcentaje). Que una correccion ajuste bien los
    # datos NO la habilita para cancelacion: la cancelacion usa g sola y
    # aislada, y no tolera que apunte mal.
    if "3b" in filas and "2." in filas:
        wb, gb = filas["2."]["rmse_E"], filas["3b"]["rmse_E"]
        print(f"  White-box {wb:.4f}  ->  gray-box con cancelacion {gb:.4f} "
              f"({100*(gb/wb-1):+.0f}%)")
    return d


# ---------------------------------------------------------------------------
# F7 · DONDE DEJA DE ALCANZAR UNA CORRECCION SIN MEMORIA
# ---------------------------------------------------------------------------
# Barrido por FAMILIA de fisica faltante (refractariedad, adaptacion con
# distintos tau, depresion sinaptica, poblacion oculta, deriva de wEE(t), ruido
# de proceso). En cada caso se ajusta la arquitectura exacta de g_phi contra el
# Delta f verdadero: los numeros son TECHOS, no resultados de entrenamiento.
#
# Las dos columnas de R2 comparan que argumentos le damos a la correccion:
#   R2 g(I,E)        solo el estado
#   R2 g(I,E,P,Q)    tambien el estimulo comandado
# La brecha entre las dos dice cuanto del hueco entra por la ENTRADA.
#
# Los tres bloques fallan por razones distintas y conviene no confundirlas:
#   capturable   -> Delta f es funcion pura de estado y entrada (R2 ~ 1).
#   estado oculto -> el limite lo pone cuan LENTA es la variable escondida
#                   (adapt_tau1 = 0.91 baja a adapt_tau100 = 0.31).
#   irreducible  -> el ruido de proceso no es funcion de nada (piso ~0.11);
#                   la deriva de wEE(t) falla por otro motivo: depende de t, y
#                   t no es argumento de g_phi.
def f7():
    d = carga("f7_controls.json")
    if not d:
        return None
    seccion("F7 · Donde deja de funcionar la correccion sin memoria")
    print(f"  {'caso':22} {'tipo':16} {'|Df|':>8} {'R2 g(I,E)':>11} "
          f"{'R2 g(I,E,P,Q)':>15}  veredicto")
    for r in d:
        print(f"  {r['caso']:22} {r['tipo']:16} {r['df_rms']:8.4f} "
              f"{r['r2_g_x']:11.3f} {r['r2_g_xpq']:15.3f}  {r['veredicto']}")
    return d


# =============================================================================
#  FIGURA RESUMEN (3 paneles): la version grafica de F1, F2 y F3
# =============================================================================
# Solo se dibuja si estan F1 y F2 (los dos barridos en eps); F3 es opcional y
# agrega el tercer panel.
def figura_resumen(d1, d2, d3):
    if not (d1 and d2):
        return
    import matplotlib
    # Backend "Agg" = sin ventana. Este informe se corre por consola / ssh y
    # solo guarda el PNG; con el backend interactivo fallaria sin display.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) deformacion vs eps
    # Las dos curvas son dos formas de medir el mismo hueco: cuanto se movio la
    # trayectoria (efecto) y cuanto vale el termino faltante (causa).
    ax[0].plot([r["eps"] for r in d1], [r["D_traj_pct"] for r in d1],
               "o-", color="#1f4e79", label="deformacion de la trayectoria")
    ax[0].plot([r["eps"] for r in d1], [r["df_rel_pct"] for r in d1],
               "s--", color="#ff7f0e", label="|Delta f| / |f_WC|")
    ax[0].set_xlabel("eps"); ax[0].set_ylabel("%")
    ax[0].set_title("F1 · tamano del hueco estructural")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    # (b) costo de la rigidez
    # El error MAXIMO es casi siempre wII y satura cerca del 200%: mas alla de
    # ese punto el parametro ya perdio todo sentido.
    ax[1].plot([f["eps"] for f in d2], [f["mean_param_error"] for f in d2],
               "o-", color="#d62728", label="error medio de theta")
    ax[1].plot([f["eps"] for f in d2], [f["max_param_error"] for f in d2],
               "s--", color="#8b0000", label="error maximo")
    ax[1].set_xlabel("eps"); ax[1].set_ylabel("error parametrico [%]")
    ax[1].set_title("F2 · lo que cuesta ignorar el hueco")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    # (c) gray-box: las dos metricas a la vez
    # El panel existe para que no se pueda hacer trampa mirando una sola: cada
    # punto es una variante, con el ajuste en x y el error parametrico en y.
    if d3:
        # eps=1 es el punto NOMINAL del trabajo (el hueco vale un tercio del
        # campo de WC). Esta fijo a proposito: es el caso que se discute en el
        # documento. Para otro nivel hay que cambiarlo aca.
        eps_ref = 1.0
        sub = [f for f in d3 if f["eps"] == eps_ref]
        wb = carga(f"f2_eps{eps_ref:g}.json")
        xs, ys, names = [], [], []
        # El white-box va primero y etiquetado "WB": es la referencia contra la
        # que se miden todas las variantes.
        if wb:
            xs.append(wb["mse_test"]); ys.append(wb["mean_param_error"]); names.append("WB")
        for f in sub:
            xs.append(f["mse_test"]); ys.append(f["mean_param_error"])
            names.append(f["variant"] + ("" if f["lam"] is None else f"·{f['lam']:g}"))
        ax[2].scatter(xs, ys, s=60, c=range(len(xs)), cmap="viridis", zorder=3)
        for x, y, n in zip(xs, ys, names):
            ax[2].annotate(n, (x, y), fontsize=8, xytext=(4, 4),
                           textcoords="offset points")
        # Escala log en x porque los mse abarcan ordenes de magnitud.
        ax[2].set_xscale("log")
        ax[2].set_xlabel("MSE open-loop (test)")
        ax[2].set_ylabel("error medio de theta [%]")
        ax[2].set_title("F3 · abajo-izquierda = gana en las dos")
        ax[2].grid(alpha=0.3)

    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "informe_incertidumbre.png", dpi=120)
    plt.close(fig)
    # Se imprime la ruta RELATIVA a la raiz (mas corta y legible en el
    # informe); la escritura usa FIG, que si esta anclada.
    print(f"\n  Figura: {(FIG / 'informe_incertidumbre.png').relative_to(_ROOT)}")


# =============================================================================
#  EJECUCION: imprime las siete secciones en orden y guarda la figura
# =============================================================================
# El orden es el del relato (hay hueco -> cuesta -> la correccion -> por donde
# roba -> que aprendio -> sirve para controlar -> donde deja de alcanzar).
# Los resultados de F1..F3 se guardan porque la figura los reusa; F4..F7 solo
# imprimen.
#
# Para volcar el informe a archivo, como esta en results/uncertainty/informe.txt:
#   python scripts/informe_incertidumbre.py > results/uncertainty/informe.txt
def main():
    d1 = f1(); d2 = f2(); d3 = f3()
    f4(); f5(); f6(); f7()
    figura_resumen(d1, d2, d3)


if __name__ == "__main__":
    main()
