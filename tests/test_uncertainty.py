"""Tests de la incertidumbre dinamica (src/wilson_cowan/uncertainty.py).

Lo que se protege aca:
  1. REGRESION: sin perturbacion, el simulador es EXACTAMENTE el de antes.
  2. eps=0 por el camino nuevo coincide con Wilson-Cowan puro.
  3. Todas las perturbaciones preservan el reposo E=I=0 como equilibrio.
  4. El estimulo COMANDADO nunca se contamina con el efectivo.
  5. Los estados ocultos se reparten bien cuando se combinan perturbaciones.
  6. La perilla eps es monotona: mas eps -> mas deformacion.

POR QUE ESTE ARCHIVO EXISTE (el riesgo que cubre):
  La extension de incertidumbre le agrega al SIMULADOR fisica que el MODELO no
  contempla (refractariedad del Wilson-Cowan de 1972 + actuador optogenetico con
  retardo y saturacion). Ese hueco es lo que el termino de correccion neuronal
  g_phi tiene que aprender. Y ahi hay dos maneras faciles de arruinar todo el
  experimento sin que nadie se de cuenta:

    (i)  Que la extension le cambie el comportamiento al camino viejo. Todos los
         resultados ya publicados del proyecto (identificacion de los 10
         parametros, robustez al ruido, FIM) salieron de perturbation=None. Si
         eso se mueve, no se puede comparar nada con nada.
    (ii) Que el estimulo EFECTIVO se filtre a los datos de entrenamiento. El
         modelo solo puede conocer lo que se COMANDO; si entrenara con lo que
         realmente llego, le estariamos regalando la respuesta y el hueco a
         aprender desapareceria. Es la trampa mas silenciosa de todo el diseno.

  Los tests de aca son la unica barrera contra esas dos cosas.

ADVERTENCIA sobre los umbrales numericos: casi ninguno es arbitrario. Estan
atados al ruido de observacion que estudia el proyecto (sigma = 0.01-0.02) o al
paso del integrador. Si un test falla, la pregunta correcta no es "subo el
umbral?" sino "que cambio de precision se colo?".

Referencia conceptual: docs/graybox_manual_completo.md
"""

import numpy as np
import pytest

from src.wilson_cowan import (
    WilsonCowan, WilsonCowanParams, box_pulse, aprbs_pulse,
    NoPerturbation, Refractoriness, Actuator, Adaptation, SynapticDepression,
    HiddenPopulation, HeterogeneousSigmoid, ProcessNoise, WeightDrift,
    CompositePerturbation, default_uncertainty,
)

# Ventana corta a proposito: los experimentos de verdad corren hasta t=600, pero
# aca alcanza con ver el transitorio y unos ciclos. El camino perturbado integra
# con RK4 de paso fijo (mas lento que solve_ivp), asi que la duracion es lo que
# pone el precio de la suite.
T_SPAN = (0.0, 60.0)
# 1200 puntos en 60 unidades -> dt de salida = 0.05. Con n_sub=4 subpasos el paso
# real del integrador queda en 0.0125, que es justo el dt con el que se valido el
# rango util de eps (ver default_uncertainty). Cambiar cualquiera de los dos
# numeros cambia el error de discretizacion y hace fallar los tests de precision.
T_EVAL = np.linspace(0.0, 60.0, 1200)


# Un solo escenario compartido por casi todos los tests: mismos parametros,
# mismos pulsos, misma grilla. Lo unico que cambia entre llamadas es la
# perturbacion -> cualquier diferencia observada es atribuible SOLO a ella.
# Los pulsos se solapan parcialmente (P prende en 5, Q en 10) para que el sistema
# pase por varios regimenes en lugar de quedarse en un punto de operacion.
def _run(pert, n_sub=4):
    m = WilsonCowan(params=WilsonCowanParams(),
                    P=box_pulse(0.8, 5.0, 55.0), Q=box_pulse(0.6, 10.0, 50.0),
                    perturbation=pert)
    return m.simulate(t_span=T_SPAN, t_eval=T_EVAL, n_sub=n_sub)


# =============================================================================
#  1. REGRESION: perturbation=None no cambia NADA del comportamiento previo.
# =============================================================================
#  Es la garantia de compatibilidad hacia atras. Todo lo que el proyecto ya
#  midio se genero con este camino; si se mueve, los resultados historicos
#  dejan de ser comparables con los nuevos.
# =============================================================================
def test_sin_perturbacion_usa_el_camino_viejo():
    """Sin perturbacion el resultado no trae las claves nuevas: es el mismo
    dict de siempre, generado por solve_ivp igual que antes."""
    # Chequear el conjunto EXACTO de claves (y no solo que esten las viejas)
    # detecta el error de haber ruteado None por el camino perturbado: ese
    # devuelve ademas P_eff, Q_eff y extra. El test falla por "sobran claves",
    # que es la senal temprana de que se cambio de integrador sin querer.
    sol = _run(None)
    assert set(sol) == {"t", "I", "E", "y", "P", "Q"}


def test_regresion_valores_identicos_sin_perturbacion():
    """La trayectoria sin perturbacion coincide con integrar el rhs original."""
    # Import local: scipy solo se necesita aca, no en toda la suite.
    from scipy.integrate import solve_ivp
    p = WilsonCowanParams()
    m = WilsonCowan(params=p, P=box_pulse(0.8, 5.0, 55.0), Q=box_pulse(0.6, 10.0, 50.0))
    # Esta llamada replica a mano, linea por linea, la que hace simulate() por
    # dentro: mismo metodo (RK45 ~ ode45) y mismas tolerancias que el MATLAB.
    # Por eso la comparacion puede ser allclose y no "parecido": tiene que dar
    # bit a bit lo mismo, porque literalmente es el mismo calculo.
    ref = solve_ivp(m.rhs, T_SPAN, [0.0, 0.0], t_eval=T_EVAL,
                    rtol=1e-3, atol=1e-6, method="RK45")
    sol = _run(None)
    assert np.allclose(sol["I"], ref.y[0])
    assert np.allclose(sol["E"], ref.y[1])


# =============================================================================
#  2. eps = 0 POR EL CAMINO NUEVO = Wilson-Cowan puro.
# =============================================================================
#  Aca se compara el camino perturbado (con la perturbacion identidad) contra
#  Wilson-Cowan. La sutileza: los dos caminos usan INTEGRADORES DISTINTOS
#  (solve_ivp adaptativo vs RK4 de paso fijo), asi que nunca van a coincidir
#  exactamente. Lo que hay que demostrar es que la diferencia es solo error de
#  discretizacion y no un cambio de ecuaciones.
# =============================================================================
def _referencia_exacta():
    """Trayectoria de referencia con tolerancia muy fina (la 'verdad' numerica)."""
    from scipy.integrate import solve_ivp
    m = WilsonCowan(params=WilsonCowanParams(),
                    P=box_pulse(0.8, 5.0, 55.0), Q=box_pulse(0.6, 10.0, 50.0))
    # DOP853 (Runge-Kutta de orden 8) con rtol=1e-11 / atol=1e-13: unas 8 ordenes
    # de magnitud mas fino que las tolerancias de produccion. A ese nivel el error
    # de integracion es despreciable frente a todo lo demas, asi que sirve de
    # ARBITRO NEUTRAL. Por eso NO se comparan los dos caminos entre si: si
    # difieren, comparandolos entre ellos no se sabe cual de los dos esta mal.
    ref = solve_ivp(m.rhs, T_SPAN, [0.0, 0.0], t_eval=T_EVAL,
                    rtol=1e-11, atol=1e-13, method="DOP853")
    return ref.y[0], ref.y[1]


def test_no_perturbation_coincide_con_wc_puro():
    """NoPerturbation pasa por el integrador de paso fijo pero las ecuaciones son
    las mismas -> tiene que coincidir con la referencia exacta de Wilson-Cowan."""
    _, Eref = _referencia_exacta()
    # n_sub=8 (el doble del default) para que el error de discretizacion quede
    # claramente por debajo del umbral y el test no viva al borde de fallar.
    cero = _run(NoPerturbation(), n_sub=8)
    err = np.max(np.abs(cero["E"] - Eref))
    # 1e-3 no es un numero redondo cualquiera: es un orden de magnitud MENOS que
    # el ruido de observacion mas chico que usa el proyecto (sigma=0.01). Si el
    # camino nuevo se desviara mas que eso, la perturbacion identidad ya estaria
    # metiendo un efecto visible en los datos.
    assert err < 1e-3, f"desvio {err:.2e} demasiado grande para ser discretizacion"


def test_paso_fijo_es_mas_preciso_que_el_camino_historico():
    """DOCUMENTA UN HALLAZGO: el camino historico (solve_ivp con rtol=1e-3) tiene
    un error de integracion de ~1.5e-2, del mismo orden que el ruido de
    observacion que el proyecto estudia (sigma=0.01-0.02). El RK4 de paso fijo
    del camino nuevo es ~30x mas preciso.

    Consecuencia practica: el baseline eps=0 de los experimentos DEBE generarse
    con NoPerturbation (mismo integrador), no con perturbation=None, para no
    confundir 'efecto de la perturbacion' con 'cambio de integrador'."""
    # Este test no protege codigo: protege una CONCLUSION. Es el que evita que
    # alguien, mas adelante, arme la curva de eps usando perturbation=None como
    # punto eps=0 y atribuya a la refractariedad un desvio de 1.5e-2 que en
    # realidad es la tolerancia floja de solve_ivp.
    _, Eref = _referencia_exacta()
    historico = np.max(np.abs(_run(None)["E"] - Eref))
    paso_fijo = np.max(np.abs(_run(NoPerturbation(), n_sub=8)["E"] - Eref))
    # Aviso: este assert depende del control de paso interno de scipy. Si una
    # version futura de scipy fuera mas precisa con rtol=1e-3, el test falla. Eso
    # es DESEADO: significa que la premisa del hallazgo cambio y hay que releerlo,
    # no que haya que subir el umbral.
    assert historico > 5e-3, "el camino historico resulto mas preciso de lo esperado"
    # El factor 10 es holgado a proposito: lo medido es ~30x. Deja margen para que
    # el test no sea fragil, pero sigue detectando si el paso fijo se degrada.
    assert paso_fijo < historico / 10.0


def test_default_uncertainty_eps_cero_es_identidad():
    """eps=0 tiene que devolver NoPerturbation, NO un composite con r=0.

    Por que importa: si eps=0 se armara escalando el composite, tau_act tenderia
    a 0 y la ecuacion del actuador se volveria rigida -> el RK4 de paso fijo se
    desestabiliza en silencio. El limite 'sin perturbacion' se alcanza por un
    caso especial, no por continuidad de la perilla."""
    assert isinstance(default_uncertainty(0.0), NoPerturbation)


# =============================================================================
#  3. TODAS preservan el reposo E=I=0 (sin estimulo el sistema duerme en cero).
#     Es la propiedad que evita cambiar el punto de operacion sin darse cuenta.
# =============================================================================
#  Por que es LA invariante clave: los parametros de Wilson-Cowan del proyecto
#  estan elegidos para que el reposo sea equilibrio (de ahi salen ke y ki). Una
#  perturbacion que rompa esa propiedad no agrega "fisica que falta": corre el
#  sistema a otro punto de operacion, y entonces g_phi ya no estaria aprendiendo
#  el mismo problema. Es un error facil de cometer al escribir una perturbacion
#  nueva (basta sumar una corriente constante en drive()).
# =============================================================================
@pytest.mark.parametrize("pert", [
    NoPerturbation(),
    Refractoriness(r=0.3),                    # r=0.3 >> el nominal 0.10: caso duro
    Actuator(sat=2.0, tau_act=1.0),
    Adaptation(b=0.5, tau_a=30.0),
    SynapticDepression(U=0.2, tau_d=30.0),
    HiddenPopulation(w_back=0.8),
    HeterogeneousSigmoid(spread=0.8),         # el unico que recalcula ke,ki
    WeightDrift(amp=0.2),
    default_uncertainty(1.0),                 # tambien el composite del roadmap
])
def test_reposo_es_equilibrio(pert):
    """Con estimulo nulo y arrancando en cero, el sistema no se mueve."""
    # Sin pasar P ni Q: el modelo usa zero_input, o sea el sistema "duerme".
    m = WilsonCowan(params=WilsonCowanParams(), perturbation=pert)
    sol = m.simulate(I0=0.0, E0=0.0, t_span=(0.0, 40.0),
                     t_eval=np.linspace(0, 40, 400))
    # 1e-9 y no 0 exacto: HeterogeneousSigmoid recalcula los offsets de reposo con
    # una cuadratura de Gauss-Hermite, asi que la cancelacion queda al nivel del
    # redondeo de punto flotante. Cualquier deriva REAL es ordenes de magnitud
    # mas grande que esto, asi que el umbral no le perdona nada de fisica.
    assert np.max(np.abs(sol["E"])) < 1e-9
    assert np.max(np.abs(sol["I"])) < 1e-9


def test_ruido_de_proceso_si_saca_del_reposo():
    """Control: el ruido de proceso SI mueve al sistema sin estimulo. Es la
    diferencia entre incertidumbre estructural (se activa con el estado) y
    ruido (esta siempre, aunque el sistema duerma)."""
    # ProcessNoise esta ausente a proposito de la lista de arriba: no es una
    # incertidumbre estructural sino el CONTROL NEGATIVO del experimento (marca
    # el piso irreducible que ninguna g_phi determinista puede bajar). Este test
    # verifica que efectivamente rompe el reposo -> que no quedo desconectado.
    m = WilsonCowan(params=WilsonCowanParams(),
                    perturbation=ProcessNoise(sigma=0.02, t_max=40.0))
    # t_max=40 tiene que cubrir el t_span: ProcessNoise pre-genera su camino en
    # una grilla y levanta ValueError si se le pide un t mas alla del final.
    sol = m.simulate(t_span=(0.0, 40.0), t_eval=np.linspace(0, 40, 400))
    # 1e-4 es un piso deliberadamente flojo: con sigma=0.02 el desvio real es
    # mucho mayor. Solo interesa distinguir "se movio" de "no se movio".
    assert np.max(np.abs(sol["E"])) > 1e-4


# =============================================================================
#  4. EL ESTIMULO COMANDADO NUNCA SE CONTAMINA CON EL EFECTIVO.
# =============================================================================
#  Es la trampa mas peligrosa del diseno gray-box. En un experimento real solo
#  se conoce lo que se COMANDO (la intensidad de luz que se pidio); lo que
#  efectivamente llego a las neuronas pasa por la cinetica del canal y por la
#  saturacion de la opsina, y NO se mide. Si el dataset guardara el efectivo en
#  la clave "P", el entrenamiento lo tomaria como entrada conocida y estaria
#  usando informacion que en el laboratorio no existe: el hueco que g_phi
#  deberia aprender se cerraria solo. Los resultados quedarian inflados y el
#  bug seria invisible (nada falla, todo "anda mejor").
# =============================================================================
def test_p_comandado_no_se_contamina_con_el_efectivo():
    """P es lo que comandaste; P_eff es lo que llego. Con un actuador que satura
    y filtra tienen que ser DISTINTOS, y P debe seguir siendo el pulso limpio."""
    # sat=0.5 con un pulso de amplitud 0.8: la saturacion es visible (recorta) y
    # tau_act=2.0 mete un retardo del orden de la constante de tiempo de E (te=1).
    # Elegido para que el efecto sea grande y el test no dependa de detalles finos.
    pert = Actuator(sat=0.5, tau_act=2.0)
    sol = _run(pert)
    esperado = np.array([box_pulse(0.8, 5.0, 55.0)(t) for t in T_EVAL])
    # (1) P sigue siendo, punto por punto, el pulso cuadrado que se comando.
    assert np.allclose(sol["P"], esperado), "P dejo de ser el comandado"
    # (2) Y ademas P_eff es realmente distinto. Sin este assert el test pasaria
    #     igual con un actuador roto que no hiciera nada (P_eff == P): estaria
    #     verificando la no contaminacion de un canal que no contamina.
    assert np.max(np.abs(sol["P_eff"] - sol["P"])) > 0.05, "el actuador no hizo nada"
    # (3) El techo de la saturacion se respeta. El +1e-9 es solo tolerancia de
    #     redondeo: tanh nunca llega a 1, asi que P_eff < sat estrictamente.
    assert sol["P_eff"].max() <= 0.5 + 1e-9, "la saturacion no se respeto"


def test_dataset_guarda_comandado_y_efectivo_por_separado():
    """El mismo cuidado, un nivel mas arriba: lo que llega al .npz.

    generate_dataset es lo que alimenta a los entrenadores, asi que es el punto
    exacto donde el efectivo podria filtrarse a los datos. Tiene que guardar los
    dos por separado (P = comandado, P_eff = solo diagnostico) y ademas
    pert_name, sin el cual el dataset no seria reproducible: no se sabria que
    fisica extra tenia el simulador que lo genero."""
    from src.data import generate_dataset
    ds = generate_dataset(WilsonCowanParams(), P=box_pulse(0.8, 5, 55),
                          Q=box_pulse(0.6, 10, 50), t_span=T_SPAN, n_eval=600,
                          perturbation=Actuator(sat=0.5, tau_act=2.0))
    assert "P_eff" in ds and "pert_name" in ds
    # No basta con que las dos claves existan: tienen que tener contenido
    # distinto. Si alguien copiara P en P_eff (o viceversa) el test lo caza.
    assert not np.allclose(ds["P"], ds["P_eff"])


# =============================================================================
#  5. COMPOSICION: los estados ocultos se reparten bien.
# =============================================================================
#  Al combinar perturbaciones, el vector de estados ocultos es uno solo y cada
#  parte se queda con su rebanada. Si las rebanadas se corrieran, una parte
#  leeria el estado de otra: el simulador seguiria corriendo y dando numeros
#  plausibles, pero serian de otro sistema. Un error asi no se detecta mirando
#  las trayectorias, solo con estos tests.
# =============================================================================
def test_composite_reparte_estados_ocultos():
    comp = CompositePerturbation([Adaptation(b=0.5), Actuator(), SynapticDepression()])
    # 1 (corriente de adaptacion) + 2 (los dos filtros del actuador) + 1 (recurso
    # sinaptico). Escrito como suma y no como "4" para que se vea de donde sale.
    assert comp.n_extra == 1 + 2 + 1
    # el estado inicial del recurso sinaptico (D=1) tiene que quedar en su lugar
    # Este es el chequeo de ORDEN, no solo de tamano: el unico extra que no
    # arranca en cero es el de SynapticDepression (sinapsis con las vesiculas
    # llenas), y tiene que aparecer en la ultima posicion porque es la ultima
    # parte de la lista. Si el 1.0 apareciera en otro indice, las rebanadas
    # estarian desalineadas.
    assert comp.extra0().tolist() == [0.0, 0.0, 0.0, 1.0]


def test_composite_equivale_a_aplicar_las_partes():
    """Refractariedad + actuador combinados == la refractariedad actuando sobre
    la entrada que el actuador deja pasar. Se verifica sobre la derivada."""
    # Se compara la DERIVADA y no una trayectoria: asi el resultado no depende
    # del integrador y el error esperado es cero de verdad (ver el 1e-12 abajo).
    p = WilsonCowanParams()
    r, act = Refractoriness(r=0.2), Actuator(sat=1.0, tau_act=1.0)
    comp = CompositePerturbation([r, act])
    # P y Q constantes (lambdas): al evaluar una sola derivada no interesa la
    # forma temporal del estimulo, solo su valor en ese instante.
    m_comp = WilsonCowan(params=p, P=lambda t: 1.5, Q=lambda t: 1.0, perturbation=comp)

    s = np.array([0.3, 0.5, 0.9, 0.4])       # [I, E, Plag, Qlag]
    # Refractoriness no tiene estados ocultos, asi que las dos ultimas
    # componentes son los filtros del actuador. Valores elegidos distintos entre
    # si (0.9 != 0.4) y distintos de I,E para que un cruce de indices se note.
    d = m_comp.rhs_aug(1.0, s)

    # Calculo manual: entrada saturada del actuador + factor refractario.
    from scipy.special import expit
    # OJO: el actuador satura su ESTADO INTERNO (Plag=0.9), no el comando (1.5).
    # El comando 1.5 solo aparece en la derivada del filtro, que este test no
    # verifica. Escribir tanh(1.5) aca seria el error tipico.
    Pe, Qe = 1.0 * np.tanh(0.9 / 1.0), 1.0 * np.tanh(0.4 / 1.0)
    u_i = p.wIE * 0.5 - p.wII * 0.3 + Qe - p.thetai
    u_e = p.wEE * 0.5 - p.wEI * 0.3 + Pe - p.thetae
    # El factor refractario (1 - r*x) multiplica la SALIDA de la sigmoidea, no su
    # entrada, y usa el estado de la propia poblacion (I para dI, E para dE). Los
    # offsets de reposo ki/ke quedan FUERA del factor: por eso el reposo se
    # preserva (en x=0 el factor vale 1 y todo se cancela como siempre).
    dI = (1 / p.ti) * (-0.3 + (1 - 0.2 * 0.3) * expit(p.ai * u_i) - p.ki)
    dE = (1 / p.te) * (-0.5 + (1 - 0.2 * 0.5) * expit(p.ae * u_e) - p.ke)
    # 1e-12 = igualdad de punto flotante. No se pide "parecido": se pide que el
    # composite haga LA MISMA ARITMETICA que el calculo a mano.
    assert abs(d[0] - dI) < 1e-12
    assert abs(d[1] - dE) < 1e-12


# =============================================================================
#  6. LA PERILLA eps ES MONOTONA: mas eps -> mas deformacion.
# =============================================================================
#  eps gradua "cuanta fisica le falta al modelo". Todo el roadmap la usa como
#  eje de barrido, y eso solo tiene sentido si la deformacion crece con eps. Si
#  no fuera monotona (por ejemplo porque a eps grande el sistema se sale del
#  regimen oscilatorio y la diferencia se achica), las curvas de "error de g_phi
#  vs eps" no serian interpretables.
# =============================================================================
def test_eps_es_monotona():
    # El baseline es NoPerturbation, NO perturbation=None: mismo integrador que
    # los casos perturbados, asi la distancia medida es efecto de la perturbacion
    # y no del cambio de integrador (es exactamente la leccion del hallazgo del
    # test de la seccion 2).
    base = _run(NoPerturbation())
    prev = 0.0
    # 0.5, 1.0, 2.0 caen dentro del rango validado [0.25, 2.0] de
    # default_uncertainty. Por debajo de ~0.15 la perilla deja de ser continua
    # (tau_act toca su piso) y arriba de 2.0 el regimen oscilatorio se destruye.
    for eps in (0.5, 1.0, 2.0):
        sol = _run(default_uncertainty(eps))
        # RMSE sobre E contra el baseline: una sola cifra por eps, comparable.
        d = float(np.sqrt(np.mean((sol["E"] - base["E"]) ** 2)))
        assert d > prev, f"eps={eps} no deformo mas que el anterior"
        prev = d
