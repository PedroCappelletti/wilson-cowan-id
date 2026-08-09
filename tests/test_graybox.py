"""Tests del entrenamiento gray-box (src/neural_ode/graybox_train.py).

Lo que se protege aca es la matematica de la variante D (proyeccion ortogonal),
que es la parte principista del experimento y la mas facil de romper en silencio:
si el orden de aplanado de las sensibilidades y de la correccion no coincide, la
penalizacion mezcla componentes y deja de significar lo que dice el comentario,
pero NO falla ruidosamente: simplemente da numeros sin sentido.

POR QUE HACEN FALTA TESTS ACA Y NO EN OTRAS PARTES DEL PIPELINE:
  El entrenamiento no tiene un "resultado correcto" con el que comparar: es una
  optimizacion. Si la penalizacion de redundancia esta mal calculada, el
  entrenamiento igual converge, igual imprime un error de parametros bajo e igual
  genera el informe. Nadie se entera de nada. Estos tests son la unica barrera
  contra ese modo de falla, y por eso apuntan justo a los cuatro lugares donde un
  error queda invisible:

    1. LA PROYECCION (variante D): que marque como redundante lo que un cambio de
       parametros puede imitar -- incluso en las direcciones DEBILES, que son las
       que importan -- y que NO marque como redundante la fisica nueva.
    2. EL ORDEN DE APLANADO: que la componente dI de la correccion se compare con
       la sensibilidad de dI, y no con la de dE.
    3. LAS SENSIBILIDADES: que el calculo por diferencias finitas (que escribe
       sobre raw.data) deje el modelo intacto, y que tenga el signo fisico bien.
    4. EL ARMADO DE LOS DATOS: que g(I,E) realmente ignore el estimulo y que las
       ventanas de multiple shooting esten alineadas. Un corrimiento de un paso
       no rompe nada visible: solo sesga los 10 parametros identificados.

Lo que este archivo NO cubre: la variante S (correccion estructurada) y el lazo
cerrado, que viven en tests/test_uncertainty.py y tests/test_neural_ode.py.

Referencias: docs/graybox_manual_completo.md (variantes A/B/C/D y por que D),
docs/recorrido_estimulos_y_entrenamiento.md (el pipeline completo).

Correr:  .venv/bin/python -m pytest tests/test_graybox.py -q
"""

import numpy as np
import torch

from src.neural_ode import GrayBoxWC
from src.neural_ode.graybox_train import (
    projection_operator, projected_fraction, make_windows, ALL_P,
)


# =============================================================================
#  FIXTURE: SENSIBILIDADES SINTETICAS MAL CONDICIONADAS
# =============================================================================
#  Por que sinteticas y no las del modelo real: aca se esta testeando el ALGEBRA
#  de la proyeccion, no el modelo. Con sensibilidades fabricadas se sabe de
#  antemano cual es la respuesta correcta (una correccion construida como S·e_j
#  es redundante por definicion), asi que el test tiene un oraculo exacto en vez
#  de un umbral inventado.

def _sens_mal_condicionada(N=400, seed=0):
    """Sensibilidades sinteticas con 6 ordenes de magnitud entre direcciones,
    como las reales (el mal condicionamiento es el hecho central del proyecto).

    De donde sale el 0.6 del exponente: 10^(-0.6*k) para k=0..9 escalona las 10
    direcciones desde 1 hasta 10^-5.4 (los "6 ordenes" son en realidad 5.4, o sea
    un numero de condicion medido de ~2.3e5). Es el orden de magnitud de las
    sensibilidades reales del Wilson-Cowan
    (hay parametros a los que el campo vectorial casi no responde). Si se usaran
    escalas parejas, los tests pasarian igual pero no probarian NADA: el bug que
    se quiere cazar (borrar las direcciones debiles) solo aparece cuando hay
    direcciones debiles.

    La semilla es fija a proposito: los tests comparan contra umbrales duros
    (0.99 y 1e-6) y una realizacion desafortunada del randn podria acercarse al
    limite y hacer el test intermitente.

    Devuelve DOS vistas de lo mismo:
      S  (N,2,10) -- el formato que entrega GrayBoxWC.backbone_sensitivities.
      Sf (2N,10)  -- ya aplanada, para poder CONSTRUIR correcciones de prueba en
                     el mismo orden de filas (n*2+c) que usa la libreria. Que el
                     test use este Sf y evalue con el que devuelve
                     projection_operator es intencional: si los dos reshapes
                     dejaran de coincidir, los tests se caen.
    """
    torch.manual_seed(seed)
    escalas = torch.tensor([10.0 ** (-k * 0.6) for k in range(10)])
    Sf = torch.randn(2 * N, 10) * escalas
    return Sf.reshape(N, 2, 10), Sf


# =============================================================================
#  LA PROYECCION ORTOGONAL (variante D)
# =============================================================================

def test_proyeccion_captura_la_direccion_mas_debil():
    """El punto de la pseudoinversa: una correccion que imita un cambio en la
    direccion MAS DEBIL tiene que detectarse como redundante. Con un ridge fijo
    esas direcciones se borran y la penalizacion no ve nada.

    ESTE ES EL TEST MAS IMPORTANTE DEL ARCHIVO. Medido con este mismo fixture,
    cambiar la pinv por el ridge fijo de 1e-8 que tenia la primera version da:
    j=0 -> 1.000, j=5 -> 1.000, j=9 -> 0.345. O sea: el ridge sigue detectando
    las direcciones fuertes (el test parece sano) y solo pierde la debil, que es
    exactamente por donde la red le roba identificabilidad a los parametros. Sin
    el caso j=9 el bug pasaba desapercibido.
    """
    S, Sf = _sens_mal_condicionada()
    A, Sf2 = projection_operator(S)
    for j in (0, 5, 9):                      # fuerte, media y la mas debil
        # c = e_j  ->  G = S·e_j = "mover solo el parametro j". Por construccion
        # esta ENTERO dentro del span de S: la respuesta correcta es fraccion 1.
        c = torch.zeros(10); c[j] = 1.0
        G = (Sf @ c).reshape(-1, 2)
        _, frac = projected_fraction(G, A, Sf2)
        # 0.99 y no == 1.0 porque todo esto es float32: en la practica da
        # 1.000000 / 1.000001 (puede pasarse un poco de 1 por redondeo).
        assert float(frac) > 0.99, f"direccion {j}: fraccion {float(frac):.3f}"


def test_correccion_ortogonal_no_es_redundante():
    """Una correccion fuera del espacio de los parametros es fisica nueva.

    Es el contrapeso del test anterior: sin este, una proyeccion que devolviera
    "todo es redundante" (por ejemplo si A no truncara nada y Sf tuviera rango
    completo por accidente) los pasaria los dos.

    Como se fabrica una direccion ortogonal sin equivocarse: la SVD completa de
    Sf (2N,10) da U de 2N columnas; las primeras 10 generan el espacio columna y
    las demas son su complemento ortogonal. Por eso U[:, 10] es la primera
    columna GARANTIZADA fuera del span -- y no hace falta ortogonalizar a mano.
    """
    S, Sf = _sens_mal_condicionada()
    A, Sf2 = projection_operator(S)
    U, _, _ = torch.linalg.svd(Sf, full_matrices=True)
    G = U[:, 10].reshape(-1, 2)              # ortogonal al span de S
    _, frac = projected_fraction(G, A, Sf2)
    # El umbral es holgado: lo medido es ~2e-15, o sea puro redondeo.
    assert float(frac) < 1e-6


def test_orden_de_aplanado_consistente():
    """S se aplana (N,2,10)->(2N,10) y g (N,2)->(2N,). Si los ordenes no
    coincidieran, una sensibilidad que solo toca dI 'explicaria' una correccion
    que solo toca dE, que es absurdo.

    El orden correcto es fila = n*2 + c (los dos canales de un punto van juntos y
    consecutivos), y lo fijan dos reshape que viven en ARCHIVOS distintos y no se
    hablan entre si: el de projection_operator y el de projected_fraction. Nada
    en el codigo los ata; este test es lo unico que los ata.

    El truco del test es construir un caso donde la respuesta correcta es 1 y 0
    exactos, para que la mezcla sea imposible de disimular: si se aplanara g
    "por canal" (todos los dI y despues todos los dE) en vez de por punto, las
    dos aserciones se caen a la vez -- lo medido es ~0.25 en los dos casos, ni
    cerca de 1 ni cerca de 0.
    """
    N = 200
    S = torch.zeros(N, 2, 10); S[:, 0, 0] = 1.0        # solo afecta dI
    A, Sf = projection_operator(S)

    g_en_dI = torch.zeros(N, 2); g_en_dI[:, 0] = 1.0
    g_en_dE = torch.zeros(N, 2); g_en_dE[:, 1] = 1.0

    # dI: coincide exactamente con la unica direccion de S -> redundante total.
    assert float(projected_fraction(g_en_dI, A, Sf)[1]) > 0.99
    # dE: ningun parametro puede tocar dE en este S -> fisica nueva, penalizacion 0.
    assert float(projected_fraction(g_en_dE, A, Sf)[1]) < 1e-6


def test_gradiente_fluye_por_g_y_no_por_las_sensibilidades():
    # Lo que protege: la penalizacion tiene que ser DERIVABLE respecto de la
    # correccion (si no, la variante D no entrena y su curva de perdida se ve
    # identica a la de la variante B, sin error visible en ninguna parte).
    #
    # El otro lado del nombre -- que NO fluya por las sensibilidades -- lo
    # garantizan dos cosas del codigo, no este assert: backbone_sensitivities
    # esta decorada con @torch.no_grad(), y projected_fraction devuelve la
    # fraccion con .detach(). S es una referencia GEOMETRICA (donde estan hoy los
    # parametros), no algo que se deba derivar: si el gradiente pasara por ahi, la
    # red podria bajar la penalizacion moviendo los parametros para "esconder" su
    # correccion, que es precisamente lo que la variante D quiere impedir.
    S, _ = _sens_mal_condicionada()
    A, Sf = projection_operator(S)
    G = torch.randn(S.shape[0], 2, requires_grad=True)
    e, _ = projected_fraction(G, A, Sf)
    e.backward()
    # isfinite y no solo "not None": con pinv de una matriz casi singular es facil
    # terminar con inf/nan y que el entrenamiento muera 200 epocas despues.
    assert G.grad is not None and torch.isfinite(G.grad).all()


# ---------------------------------------------------------------------------
#  Sensibilidades del backbone
# ---------------------------------------------------------------------------
#  Aca se pasa del algebra al modelo real. Las sensibilidades ∂f/∂θ se calculan
#  por diferencias centradas ESCRIBIENDO sobre raw.data (no hay otra forma de
#  perturbar un parametro sin construir el grafo), y eso es peligroso: si la
#  restauracion falla, el entrenamiento sigue con parametros corridos.

def _modelo():
    # Arranque IGNORANTE: los 10 parametros valen 1.0, igual que en build_model.
    # No importa que 1.0 no sea el valor verdadero de nada: estos tests miran
    # signos y invariantes, no precision numerica.
    init = {k: 1.0 for k in ALL_P}
    return GrayBoxWC(init, {k: 1.0 for k in ("wEE", "wEI", "wIE", "wII")},
                     learnable_weights=True, learnable_params=True)


def test_sensibilidades_no_alteran_el_modelo():
    """El calculo por diferencias finitas toca raw.data; tiene que dejar el
    modelo EXACTAMENTE como estaba, si no el entrenamiento se corrompe.

    Como se corromperia en la practica: durante fit() las sensibilidades se
    recalculan cada sens_every epocas. Un residuo de +h que no se restaura se
    acumularia epoca tras epoca y arrastraria los parametros de forma
    sistematica. No habria excepcion ni NaN: solo un sesgo en el informe final.

    Se comparan las dos cosas por separado a proposito: el state_dict (los
    parametros crudos) y la SALIDA del backbone. Lo segundo cubre el caso de que
    algo quede alterado fuera del state_dict (un buffer, un cache).
    """
    m = _modelo()
    x = torch.rand(50, 2)
    P = torch.rand(50, 1); Q = torch.rand(50, 1)
    antes = {k: v.detach().clone() for k, v in m.state_dict().items()}
    f_antes = m.backbone(x, P, Q).detach().clone()

    m.backbone_sensitivities(x, P, Q)

    for k, v in m.state_dict().items():
        assert torch.allclose(antes[k], v), f"la sensibilidad altero {k}"
    assert torch.allclose(f_antes, m.backbone(x, P, Q))


def test_sensibilidades_tienen_signo_correcto():
    """Chequeo de cordura: subir wEE tiene que AUMENTAR dE (mas autoexcitacion).

    Es el unico test que ata el numero 'sensibilidad' a la FISICA. Sin el, un
    error de signo (un f_minus - f_plus dado vuelta, o un indice de columna
    corrido) daria una matriz perfectamente bien condicionada y absolutamente
    incorrecta, y la proyeccion seguiria dando fracciones plausibles.

    DOS TRAMPAS que explican por que el test esta escrito asi:

    1. El rango de x: rand*0.5 + 0.2 pone I,E en [0.2, 0.7], LEJOS de cero.
       ∂dE/∂wEE = (1/te)·ae·S'(u_e)·E, o sea proporcional a E. Con E=0 la
       sensibilidad es exactamente 0 y el assert estricto (> 0) se caeria sin que
       haya ningun bug. El mismo cuidado vale para la desigualdad con wEI.

    2. Las diferencias finitas perturban el parametro CRUDO (el de antes del
       softplus), no wEE directamente. No invalida el chequeo porque softplus es
       monotona creciente: su derivada es siempre positiva y el signo se conserva.

    P=Q=0 para que el signo no dependa del estimulo: se quiere el efecto puro del
    acoplamiento interno.
    """
    m = _modelo()
    x = torch.rand(30, 2) * 0.5 + 0.2
    P = torch.zeros(30, 1); Q = torch.zeros(30, 1)
    S = m.backbone_sensitivities(x, P, Q)          # (N,2,10)
    # El indice se busca por NOMBRE en PARAM_ORDER en vez de escribir 0 y 1 a
    # mano: si alguien reordena PARAM_ORDER el test lo sigue, en lugar de empezar
    # a chequear el signo del parametro equivocado.
    i_wEE = list(GrayBoxWC.PARAM_ORDER).index("wEE")
    assert (S[:, 1, i_wEE] > 0).all(), "dE deberia crecer con wEE"
    i_wEI = list(GrayBoxWC.PARAM_ORDER).index("wEI")
    assert (S[:, 1, i_wEI] < 0).all(), "dE deberia bajar con wEI (mas inhibicion)"


# ---------------------------------------------------------------------------
#  La correccion restringida
# ---------------------------------------------------------------------------

def test_correccion_solo_estado_ignora_el_estimulo():
    """La variante B/C/D usa g(I,E): cambiar P,Q NO debe cambiar la correccion.
    Es lo que fuerza a que el estimulo lo explique el backbone, y lo que hace que
    el controlador pueda cancelar g de forma explicita.

    Si correction_inputs='x' dejara de respetarse (por ejemplo si g_out cayera al
    camino 'xpq' por un default mal puesto), las variantes B/C/D se volverian la
    variante A sin avisar: el experimento entero compararia A contra A y la
    conclusion del informe seria falsa.
    """
    init = {k: 1.0 for k in ALL_P}
    m = GrayBoxWC(init, {k: 1.0 for k in ("wEE", "wEI", "wIE", "wII")},
                  learnable_weights=True, learnable_params=True,
                  use_correction=True, correction_inputs="x")
    # sacar la correccion de cero para que el test sea informativo
    # (GrayBoxWC inicializa la ultima capa en cero para arrancar = WC puro; con
    # esos pesos g == 0 y el allclose de abajo pasaria por razones triviales,
    # incluso si g SI estuviera mirando P,Q).
    with torch.no_grad():
        m.g[-1].weight.normal_(0, 0.5)
    x = torch.rand(20, 2)
    g1 = m.g_out(x, torch.zeros(20, 1), torch.zeros(20, 1))
    g2 = m.g_out(x, torch.rand(20, 1) * 3, torch.rand(20, 1) * 3)
    assert torch.allclose(g1, g2)
    assert g1.abs().max() > 1e-6          # la correccion no es trivialmente cero


def test_forward_es_backbone_mas_correccion():
    # Protege la descomposicion aditiva  f = f_WC + g_φ, que NO es un detalle de
    # implementacion: de ella dependen (a) que las sensibilidades del backbone
    # sean las del termino correcto, (b) que 'g_rel' del informe mida algo, y
    # (c) que el controlador IMC pueda restar g y quedarse con WC puro. Si alguien
    # metiera la correccion adentro de la sigmoidea, todo eso deja de valer.
    init = {k: 1.0 for k in ALL_P}
    m = GrayBoxWC(init, {k: 1.0 for k in ("wEE", "wEI", "wIE", "wII")},
                  learnable_weights=True, learnable_params=True,
                  use_correction=True, correction_inputs="x")
    with torch.no_grad():
        m.g[-1].weight.normal_(0, 0.5)
    x = torch.rand(15, 2); P = torch.rand(15, 1); Q = torch.rand(15, 1)
    assert torch.allclose(m(x, P, Q), m.backbone(x, P, Q) + m.g_out(x, P, Q))


def test_make_windows_alinea_estado_y_estimulo():
    """La ventana w arranca en el estado observado y usa el estimulo de ESE
    instante. Un corrimiento de un paso sesgaria todos los parametros.

    Por que este test es facil de escribir y por eso vale la pena: make_windows
    hace cuatro reshape/transpose/permute seguidos y ninguno falla si el orden
    esta mal -- las formas salen iguales. La unica forma de verificarlo es con
    datos MARCADOS: se rellena I,E,P,Q con numeros que codifican su propia
    posicion (E = I+100, P = I+1000, Q = I+10000), asi cada valor dice de donde
    vino y una permutacion equivocada se ve a simple vista.
    """
    n, T, W = 2, 41, 10
    I = np.arange(n * T, dtype=float).reshape(n, T)
    E = I + 100.0
    P = I + 1000.0
    Q = I + 10000.0
    x0, Pw, Qw, tgt = make_windows(I, E, P, Q, W)
    # (T-1)//W = 4 ventanas por trayectoria, 8 en total. Se usa T-1 y no T porque
    # cada ventana necesita W+1 puntos de target (el estado inicial mas los W
    # pasos integrados), asi que hay un punto menos que puntos de arranque.
    assert x0.shape[0] == n * (T - 1) // W
    # primera ventana de la primera trayectoria
    # x0 = [I,E] en ese orden (el mismo del estado en model.py: I primero).
    assert float(x0[0, 0]) == I[0, 0] and float(x0[0, 1]) == E[0, 0]
    # Pw es (W, Nw, 1): primero el tiempo, despues la ventana. Ese orden es el que
    # espera rollout(). Si estuviera transpuesta, con n y W parecidos ni la forma
    # protestaria.
    assert float(Pw[0, 0, 0]) == P[0, 0]
    # tgt es (W+1, Nw, 2): el primer punto del target ES el estado inicial (no el
    # siguiente). Aca esta el corrimiento de un paso que el docstring menciona.
    assert float(tgt[0, 0, 0]) == I[0, 0]
    assert float(tgt[W, 0, 0]) == I[0, W]
    # segunda ventana: arranca donde termina la primera
    # O sea: las ventanas se PISAN en un punto (el borde es a la vez el ultimo
    # target de una y el arranque de la siguiente). Es a proposito: asi no queda
    # ningun tramo de la trayectoria sin cubrir.
    assert float(x0[1, 0]) == I[0, W]
