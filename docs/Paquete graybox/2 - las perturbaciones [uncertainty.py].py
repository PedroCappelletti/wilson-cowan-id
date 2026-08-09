# =============================================================================
#  INCERTIDUMBRE DINAMICA: hacer que el simulador NO sea las mismas ecuaciones
#  que el modelo que despues entrenamos.
# =============================================================================
#
#  Para que sirve: hoy el simulador (model.py) y el backbone de la red
#  (neural_ode/dynamics.py) son EXACTAMENTE las mismas ecuaciones. Por eso el
#  termino extra de la red (g_phi) no tiene nada que aprender: sobra.
#
#  Un "gray-box" de verdad necesita un HUECO: que la planta que genera los datos
#  tenga fisica que el modelo no contempla. Este archivo agrega ese hueco.
#
#  IMPORTANTE — donde vive cada cosa:
#    - La perturbacion vive ACA y entra en el simulador  -> es "el cerebro real".
#    - La red NUNCA la ve. Solo ve (I, E) y el estimulo (P, Q) que comandaste.
#    - Todo lo que no cierre con Wilson-Cowan puro lo tiene que descubrir g_phi.
#
#  CUANDO se aplica: en TODO instante, es parte permanente del campo vectorial.
#  No es un golpe en un momento dado: es fisica que siempre estuvo ahi.
#
#  NO es aleatoria (salvo ProcessNoise, que esta a proposito como control
#  negativo): g_phi es una funcion determinista de (I,E,P,Q) y solo puede
#  aprender lo que es REPRODUCIBLE. Si el sistema vuelve a pasar por el mismo
#  estado con el mismo estimulo, la correccion tiene que valer lo mismo.
#
#  TODAS las perturbaciones de aca preservan el reposo E=I=0 como equilibrio:
#  sin estimulo el sistema sigue durmiendo en cero, igual que antes.
# =============================================================================

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.special import expit, roots_hermitenorm


# =============================================================================
#  SECCION 1: LA INTERFAZ
# =============================================================================
#  Una perturbacion puede meterse en CINCO lugares distintos de la ecuacion.
#  Se escriben como "ganchos" (hooks) para poder COMBINAR varias a la vez.
#
#  Recordemos la ecuacion base (una por poblacion, aca la de I):
#
#      u_i = wIE*E - wII*I + Q - thetai          <- 1) entrada total
#      dI  = (1/ti) * ( -I + S(u_i) - ki )       <- 2) sigmoidea y relajacion
#
#  Los cinco lugares donde se puede meter mano:
#
#    (a) inputs()  : el estimulo que LLEGA no es el que comandaste (Q -> Q_eff).
#                    Es la incertidumbre del CANAL DE ACTUACION.
#    (b) drive()   : se suma algo a u_i / u_e (una corriente extra).
#                    Es lo que hacen la adaptacion y una poblacion no modelada.
#    (c) gains()   : multiplica la salida de la sigmoidea.
#                    Es la refractariedad del Wilson-Cowan original.
#    (d) weights() : cambia los pesos sinapticos (p.ej. que dependan del tiempo
#                    o del uso).
#    (e) deriv()   : suma directamente a dI/dE (el ruido de proceso).
#
#  Ademas una perturbacion puede tener ESTADOS PROPIOS OCULTOS (n_extra): la
#  corriente de adaptacion, el recurso sinaptico, el estado del actuador.
#  Esos estados NO se miden -> no estan entre los argumentos de g_phi -> son
#  justamente los que hacen que la correccion sin memoria se quede corta.
# =============================================================================

class Perturbation:
    """Clase base. Por defecto no hace nada: todos los ganchos son la identidad."""

    name = "base"
    n_extra = 0                    # cuantos estados ocultos propios agrega
    requires_fixed_step = False    # True -> obliga al integrador de paso fijo

    # --- estados ocultos ---
    def extra0(self) -> np.ndarray:
        """Valor inicial de los estados ocultos."""
        return np.zeros(self.n_extra)

    def d_extra(self, t, I, E, extra, P, Q, p) -> np.ndarray:
        """Derivada de los estados ocultos."""
        return np.zeros(self.n_extra)

    # --- los cinco ganchos ---
    def inputs(self, t, I, E, extra, P, Q, p):
        """(a) Estimulo efectivo que realmente llega a las poblaciones."""
        return P, Q

    def drive(self, t, I, E, extra, p):
        """(b) Corriente extra que se suma a (u_i, u_e)."""
        return 0.0, 0.0

    def gains(self, t, I, E, extra, p):
        """(c) Factor multiplicativo sobre la salida de las sigmoideas (i, e)."""
        return 1.0, 1.0

    def weights(self, t, I, E, extra, p):
        """(d) Pesos efectivos (wEE, wEI, wIE, wII). None = los nominales."""
        return None

    def deriv(self, t, I, E, extra, p):
        """(e) Termino que se suma directamente a (dI, dE)."""
        return 0.0, 0.0

    # --- la no linealidad estatica y el offset de reposo ---
    def sigmoid_i(self, u, p):
        return expit(p.ai * u)

    def sigmoid_e(self, u, p):
        return expit(p.ae * u)

    def rest_offsets(self, p):
        """(ke, ki): los offsets que hacen que E=I=0 sea equilibrio."""
        return p.ke, p.ki

    def metadata(self) -> dict:
        """Lo que se guarda en el .npz para poder reproducir el experimento."""
        return {"pert_name": self.name, "pert_n_extra": self.n_extra}

    def __repr__(self) -> str:
        return f"<{self.name}>"


class NoPerturbation(Perturbation):
    """Identidad explicita. Sirve para generar el caso eps=0 POR EL MISMO CAMINO
    de codigo que los perturbados, y que la comparacion sea justa (mismo
    integrador, mismo error de discretizacion)."""
    name = "none"


# =============================================================================
#  SECCION 2: LAS FAMILIAS DE INCERTIDUMBRE
# =============================================================================
#  Cada una dice: que es fisicamente, donde entra, y si g_phi puede aprenderla.
#  Los R2 que aparecen los medimos en scripts/uncertainty_probe/.
# =============================================================================

class Refractoriness(Perturbation):
    """(1) REFRACTARIEDAD — el termino del Wilson-Cowan ORIGINAL (1972).

    Que es fisicamente: una neurona que acaba de disparar no puede volver a
    disparar de inmediato (periodo refractario). Entonces la fraccion de la
    poblacion DISPONIBLE para disparar no es 1 sino (1 - r*x), donde x es la
    fraccion que ya esta activa.

    Por que esta es la mejor eleccion: NO es una perturbacion inventada. Es
    literalmente el factor que Wilson y Cowan derivaron en 1972 y que la forma
    reducida (la que usamos) descarta. O sea: la planta pasa a ser el WC
    ORIGINAL y el modelo el WC REDUCIDO. Es un mismatch estructural real y
    citable, no un capricho.

    Donde entra: gancho (c), multiplicando la sigmoidea.
    Preserva el reposo: en E=I=0 el factor vale 1 -> todo queda igual.
    Aprendible por g_phi(I,E): SI, R2 = 0.97 (es funcion pura del estado).
    """

    name = "refractoriness"

    def __init__(self, r: float = 0.10, r_i: float | None = None):
        self.r_e = float(r)                                # sobre la poblacion E
        self.r_i = float(r if r_i is None else r_i)        # sobre la poblacion I

    def gains(self, t, I, E, extra, p):
        return 1.0 - self.r_i * I, 1.0 - self.r_e * E

    def metadata(self):
        return {**super().metadata(), "pert_r_e": self.r_e, "pert_r_i": self.r_i}


class Actuator(Perturbation):
    """(2) CANAL DE ACTUACION NO IDEAL — lo que comandas no es lo que llega.

    Que es fisicamente: en optogenetica vos comandas intensidad de luz, pero lo
    que la neurona recibe paso por (i) la cinetica del canal ChR2, que es un
    filtro de primer orden con constante tau_act, y (ii) la saturacion de la
    opsina, que no puede entregar corriente infinita.

    Por que importa: rompe el supuesto de que P,Q son perfectamente conocidos,
    que es JUSTAMENTE el supuesto que hoy ancla la escala de los parametros
    (lo dice el encabezado de train_neural_ode_full.py). Es un test duro y
    honesto. Ademas entra literalmente por las DOS entradas del sistema.

    Donde entra: gancho (a), reemplazando (P,Q) por (P_eff,Q_eff).
    Estados ocultos: 2 (el estado del filtro de cada canal).
    Preserva el reposo: con P=Q=0 los filtros quedan en 0 -> nada cambia.
    Aprendible por g_phi: PARCIAL. La saturacion si (es funcion de P). El lag
      NO, porque su estado es una variable independiente que g_phi no ve.
      Medido: R2 = 0.34 con tau_act = 0.5 ms (el caso dificil).
    """

    name = "actuator"
    n_extra = 2

    def __init__(self, sat: float = 3.0, tau_act: float = 1.0):
        self.sat = float(sat)          # nivel de saturacion (mas chico = satura antes)
        self.tau_act = float(tau_act)  # constante del filtro del canal [ms]

    def d_extra(self, t, I, E, extra, P, Q, p):
        # El filtro persigue al comando: dPl/dt = (P_comandado - Pl)/tau
        return np.array([(P - extra[0]) / self.tau_act,
                         (Q - extra[1]) / self.tau_act])

    def inputs(self, t, I, E, extra, P, Q, p):
        A = self.sat
        # tanh: saturacion suave. Para |x| << A es casi lineal (no distorsiona);
        # para |x| >> A se aplana en +-A.
        return A * np.tanh(extra[0] / A), A * np.tanh(extra[1] / A)

    def metadata(self):
        return {**super().metadata(), "pert_sat": self.sat, "pert_tau_act": self.tau_act}


class Adaptation(Perturbation):
    """(3) ADAPTACION POR FRECUENCIA DE DISPARO (SFA).

    Que es fisicamente: cuando una poblacion dispara sostenidamente se activan
    corrientes lentas de potasio que la frenan. Es el mecanismo por el que una
    neurona "se cansa" ante un estimulo constante.

    Donde entra: gancho (b), restando corriente a la poblacion E.
    Estado oculto: 1 (la corriente de adaptacion A).
    Aprendible por g_phi: DEPENDE DE tau_a, y esa es la gracia.
      tau_a = 1 ms  -> R2 = 0.97  (el estado oculto queda esclavizado a E)
      tau_a = 30 ms -> R2 = 0.54
      tau_a = 100 ms-> R2 = 0.33  (guarda historia propia que (I,E,P,Q) no tiene)
    Es la perilla continua entre "capturable" y "no capturable".
    """

    name = "adaptation"
    n_extra = 1

    def __init__(self, b: float = 0.5, tau_a: float = 30.0):
        self.b = float(b)          # fuerza de la adaptacion
        self.tau_a = float(tau_a)  # que tan LENTA es [ms] -> controla la memoria

    def d_extra(self, t, I, E, extra, P, Q, p):
        return np.array([(E - extra[0]) / self.tau_a])

    def drive(self, t, I, E, extra, p):
        return 0.0, -self.b * extra[0]      # frena a E

    def metadata(self):
        return {**super().metadata(), "pert_b": self.b, "pert_tau_a": self.tau_a}


class SynapticDepression(Perturbation):
    """(4) DEPRESION SINAPTICA DE CORTO PLAZO (Tsodyks-Markram).

    Que es fisicamente: cada sinapsis tiene un pozo finito de vesiculas. Si la
    usas mucho se vacia y el peso efectivo BAJA; se recupera con constante tau_d.
    O sea: los pesos sinapticos dejan de ser constantes.

    Donde entra: gancho (d), escalando los pesos que salen de E.
    Estado oculto: 1 (el recurso disponible D en [0,1]).
    Aprendible por g_phi: parcial, R2 = 0.71.
    """

    name = "depression"
    n_extra = 1

    def __init__(self, U: float = 0.15, tau_d: float = 30.0):
        self.U = float(U)
        self.tau_d = float(tau_d)

    def extra0(self):
        return np.array([1.0])       # arranca con los recursos llenos

    def d_extra(self, t, I, E, extra, P, Q, p):
        D = extra[0]
        return np.array([(1.0 - D) / self.tau_d - self.U * D * E])

    def weights(self, t, I, E, extra, p):
        D = extra[0]
        return (p.wEE * D, p.wEI, p.wIE * D, p.wII)

    def metadata(self):
        return {**super().metadata(), "pert_U": self.U, "pert_tau_d": self.tau_d}


class HiddenPopulation(Perturbation):
    """(5) DINAMICA NO MODELADA — una tercera poblacion acoplada.

    Que es fisicamente: el modelo de dos poblaciones es una simplificacion; en
    el tejido real hay mas subtipos (p.ej. otra clase de interneurona) o una
    columna vecina acoplada. Esa poblacion existe, influye, y no la modelas.

    Donde entra: gancho (b). Estado oculto: 1.
    Aprendible por g_phi: parcial, R2 = 0.84.
    """

    name = "hidden_population"
    n_extra = 1

    def __init__(self, w_back: float = 0.8, tj: float = 8.0,
                 aj: float = 1.0, thj: float = 3.0, wJE: float = 5.0):
        self.w_back = float(w_back)   # cuanto influye de vuelta sobre E
        self.tj, self.aj, self.thj, self.wJE = float(tj), float(aj), float(thj), float(wJE)
        self.kj = float(expit(-self.aj * self.thj))

    def d_extra(self, t, I, E, extra, P, Q, p):
        J = extra[0]
        return np.array([(1.0 / self.tj) *
                         (-J + expit(self.aj * (self.wJE * E - self.thj)) - self.kj)])

    def drive(self, t, I, E, extra, p):
        return 0.0, -self.w_back * extra[0]

    def metadata(self):
        return {**super().metadata(), "pert_w_back": self.w_back, "pert_tj": self.tj}


# Nodos de Gauss-Hermite para promediar la sigmoidea sobre umbrales dispersos.
_GH_X, _GH_W = roots_hermitenorm(21)
_GH_W = _GH_W / _GH_W.sum()


class HeterogeneousSigmoid(Perturbation):
    """(6) LA NO LINEALIDAD ESTATICA NO ES UNA LOGISTICA.

    Que es fisicamente: la sigmoidea de Wilson-Cowan sale de promediar muchas
    neuronas suponiendo que todas tienen el mismo umbral. En la realidad los
    umbrales estan DISPERSOS, y el promedio de logisticas con umbral disperso
    no es una logistica: es mas suave y con colas mas gordas.

    Donde entra: reemplaza sigmoid_i/sigmoid_e Y recalcula ke,ki (si no, se
    rompe el reposo).
    Aprendible por g_phi: SI, R2 = 0.98 (es funcion pura del estado).
    """

    name = "hetero_sigmoid"

    def __init__(self, spread: float = 0.8):
        self.spread = float(spread)

    def _s(self, u, a):
        return float(np.sum(_GH_W * expit(a * (u - self.spread * _GH_X))))

    def sigmoid_i(self, u, p):
        return self._s(u, p.ai)

    def sigmoid_e(self, u, p):
        return self._s(u, p.ae)

    def rest_offsets(self, p):
        # Recalculados para que E=I=0 siga siendo equilibrio con la curva nueva.
        return self._s(-p.thetae, p.ae), self._s(-p.thetai, p.ai)

    def metadata(self):
        return {**super().metadata(), "pert_spread": self.spread}


class ProcessNoise(Perturbation):
    """(7) RUIDO DE PROCESO — el CONTROL NEGATIVO del experimento.

    Que es fisicamente: una poblacion real tiene un numero finito de neuronas,
    asi que su tasa fluctua aun sin estimulo (ruido de tamano finito). Se modela
    como un proceso de Ornstein-Uhlenbeck sumado a las derivadas.

    OJO — ES DISTINTO DEL RUIDO DE OBSERVACION que ya usa el proyecto:
      - ruido de observacion (noise_std): se suma DESPUES de integrar, no
        cambia la trayectoria; es "medis mal".
      - ruido de proceso (este): se suma DENTRO, SI cambia la trayectoria.

    Aprendible por g_phi: NO, R2 = 0.04. Y no puede serlo: g_phi es determinista
    y el ruido no es funcion de (I,E,P,Q). Por eso sirve como PISO IRREDUCIBLE:
    marca hasta donde puede bajar cualquier correccion.

    Implementacion: el camino de ruido se PRE-GENERA en una grilla a partir de
    la semilla y despues se interpola. Asi xi(t) es una funcion pura del tiempo
    (mismo t -> mismo valor), que es lo que el integrador necesita para poder
    llamar a la derivada varias veces en el mismo instante.
    """

    name = "process_noise"
    requires_fixed_step = True

    def __init__(self, sigma: float = 0.02, tau_n: float = 1.0,
                 seed: int = 0, t_max: float = 200.0, dt_grid: float = 0.01):
        self.sigma, self.tau_n, self.seed = float(sigma), float(tau_n), int(seed)
        self.dt_grid = float(dt_grid)
        n = int(np.ceil(t_max / self.dt_grid)) + 2
        rng = np.random.default_rng(seed)
        xi = np.zeros((n, 2))
        for k in range(n - 1):
            xi[k + 1] = (xi[k] - xi[k] / self.tau_n * self.dt_grid
                         + self.sigma * np.sqrt(2 * self.dt_grid / self.tau_n)
                         * rng.normal(size=2))
        self._xi = xi
        self._tgrid = np.arange(n) * self.dt_grid

    def deriv(self, t, I, E, extra, p):
        # Fuera del rango pre-generado el camino se congelaria en una constante,
        # o sea el "ruido" se convertiria en un sesgo DC determinista y el
        # control negativo dejaria de serlo. Mejor fallar que mentir.
        if t > self._tgrid[-1] + 1e-9:
            raise ValueError(
                f"ProcessNoise: t={t:.3f} supera el camino pre-generado "
                f"(t_max={self._tgrid[-1]:.3f}). Construilo con t_max mas grande.")
        k = int(np.clip(np.searchsorted(self._tgrid, t) - 1, 0, len(self._tgrid) - 2))
        # interpolacion lineal entre los dos nodos de la grilla
        w = (t - self._tgrid[k]) / self.dt_grid
        w = min(max(w, 0.0), 1.0)
        v = (1 - w) * self._xi[k] + w * self._xi[k + 1]
        return float(v[0]), float(v[1])

    def metadata(self):
        return {**super().metadata(), "pert_sigma_proc": self.sigma,
                "pert_tau_n": self.tau_n, "pert_noise_seed": self.seed}


class WeightDrift(Perturbation):
    """(8) DERIVA LENTA DE wEE — el otro control negativo.

    Que es fisicamente: neuromodulacion. El estado del cerebro cambia despacio
    (atencion, vigilia, dopamina) y con el cambia la ganancia de las conexiones.

    Aprendible por g_phi: NO, R2 = 0.33 — pero por una razon DISTINTA al ruido:
    depende explicitamente de t, y t NO es argumento de g_phi(I,E,P,Q). Es
    incertidumbre NO AUTONOMA. Sirve para mostrar que hay dos maneras distintas
    de que la correccion falle: por azar y por dependencia temporal.
    """

    name = "weight_drift"

    def __init__(self, amp: float = 0.15, f_slow_hz: float = 5.0):
        self.amp = float(amp)
        self.f = float(f_slow_hz) / 1000.0    # Hz -> ciclos/ms

    def weights(self, t, I, E, extra, p):
        s = 1.0 + self.amp * np.sin(2.0 * np.pi * self.f * t)
        return (p.wEE * s, p.wEI, p.wIE, p.wII)

    def metadata(self):
        return {**super().metadata(), "pert_drift_amp": self.amp,
                "pert_drift_f": self.f}


# =============================================================================
#  SECCION 3: COMBINAR VARIAS PERTURBACIONES
# =============================================================================
#  El diseno del proyecto usa DOS entradas a la vez:
#     entrada 1 (lado del ESTADO)  : refractariedad
#     entrada 2 (lado de la ENTRADA): actuador optogenetico
#  Este objeto las encadena y reparte los estados ocultos entre ellas.
# =============================================================================

class CompositePerturbation(Perturbation):
    """Aplica varias perturbaciones a la vez. Cada una recibe SOLO su porcion
    del vector de estados ocultos."""

    name = "composite"

    def __init__(self, parts: Sequence[Perturbation]):
        self.parts = list(parts)
        self.n_extra = sum(p.n_extra for p in self.parts)
        self.requires_fixed_step = any(p.requires_fixed_step for p in self.parts)
        # rebanadas del vector de extras, una por parte
        self._slices, a = [], 0
        for p in self.parts:
            self._slices.append(slice(a, a + p.n_extra))
            a += p.n_extra

    def _sub(self, extra):
        return [extra[s] for s in self._slices]

    def extra0(self):
        return (np.concatenate([p.extra0() for p in self.parts])
                if self.parts else np.zeros(0))

    def d_extra(self, t, I, E, extra, P, Q, p):
        subs = self._sub(extra)
        outs = [part.d_extra(t, I, E, subs[i], P, Q, p)
                for i, part in enumerate(self.parts)]
        return np.concatenate(outs) if outs else np.zeros(0)

    def inputs(self, t, I, E, extra, P, Q, p):
        subs = self._sub(extra)
        for i, part in enumerate(self.parts):
            P, Q = part.inputs(t, I, E, subs[i], P, Q, p)   # encadenadas
        return P, Q

    def drive(self, t, I, E, extra, p):
        subs = self._sub(extra)
        di = de = 0.0
        for i, part in enumerate(self.parts):
            a, b = part.drive(t, I, E, subs[i], p)
            di += a; de += b
        return di, de

    def gains(self, t, I, E, extra, p):
        subs = self._sub(extra)
        gi = ge = 1.0
        for i, part in enumerate(self.parts):
            a, b = part.gains(t, I, E, subs[i], p)
            gi *= a; ge *= b
        return gi, ge

    def weights(self, t, I, E, extra, p):
        # Cada parte devuelve pesos ABSOLUTOS. Para combinarlas se acumulan sus
        # FACTORES respecto de los nominales (una parte que multiplica wEE por
        # 0.8 y otra que lo multiplica por 0.5 dejan wEE nominal * 0.4).
        subs = self._sub(extra)
        nom = (p.wEE, p.wEI, p.wIE, p.wII)
        factor = [1.0, 1.0, 1.0, 1.0]
        tocado = False
        for i, part in enumerate(self.parts):
            wi = part.weights(t, I, E, subs[i], p)
            if wi is not None:
                tocado = True
                factor = [f * (a / b) for f, a, b in zip(factor, wi, nom)]
        return tuple(f * b for f, b in zip(factor, nom)) if tocado else None

    def deriv(self, t, I, E, extra, p):
        subs = self._sub(extra)
        di = de = 0.0
        for i, part in enumerate(self.parts):
            a, b = part.deriv(t, I, E, subs[i], p)
            di += a; de += b
        return di, de

    # La no linealidad y el reposo los define la ULTIMA parte que los redefina
    # (en la practica solo HeterogeneousSigmoid los toca, y no se combina con si
    # misma, asi que no hay ambiguedad).
    def _owner(self):
        # Se excluye a CompositePerturbation a proposito: como ella TAMBIEN
        # redefine sigmoid_i, un composite anidado siempre "ganaria" la
        # propiedad y taparia en silencio a un HeterogeneousSigmoid anterior.
        for part in reversed(self.parts):
            if isinstance(part, CompositePerturbation):
                sub = part._owner()
                if sub is not None:
                    return sub
                continue
            if type(part).sigmoid_i is not Perturbation.sigmoid_i:
                return part
        return None

    def sigmoid_i(self, u, p):
        o = self._owner()
        return o.sigmoid_i(u, p) if o else expit(p.ai * u)

    def sigmoid_e(self, u, p):
        o = self._owner()
        return o.sigmoid_e(u, p) if o else expit(p.ae * u)

    def rest_offsets(self, p):
        o = self._owner()
        return o.rest_offsets(p) if o else (p.ke, p.ki)

    def metadata(self):
        md = {"pert_name": "+".join(p.name for p in self.parts),
              "pert_n_extra": self.n_extra}
        for part in self.parts:
            md.update({k: v for k, v in part.metadata().items()
                       if k not in ("pert_name", "pert_n_extra")})
        return md


# =============================================================================
#  SECCION 4: EL PAR RECOMENDADO (el que usa todo el roadmap)
# =============================================================================

def default_uncertainty(eps: float = 1.0) -> Perturbation:
    """Las DOS entradas del diseno, con una sola perilla `eps` que las escala.

        eps = 0.0  -> sin perturbacion (Wilson-Cowan puro)
        eps = 1.0  -> punto de operacion nominal, calibrado:
                      r = 0.10 (refractariedad), A = 3.0, tau_act = 1.0 ms

    La calibracion (ver docs/incertidumbre_dinamica_graybox.md) busca que la
    deformacion quede BIEN por encima del ruido de observacion que usa el
    proyecto (~3x sigma=0.02) sin destruir el regimen oscilatorio.

    Como escala eps:
      - refractariedad: r crece linealmente con eps (r=0 es "sin efecto").
      - actuador: la saturacion se AFLOJA cuando eps->0 (A -> infinito, o sea
        sin saturacion) y el lag se vuelve mas rapido (tau_act -> 0).

    OJO CON EL LIMITE eps -> 0. Bajar tau_act vuelve RIGIDA la ecuacion del
    actuador: cuando tau_act se acerca al paso del integrador, el RK4 de paso
    fijo se vuelve inestable. O sea que el limite ideal NO se alcanza bajando
    eps de forma continua; para eps=0 hay que usar NoPerturbation (que es lo que
    devuelve esta funcion). Por eso tau_act tiene un piso y se avisa si el eps
    pedido cae por debajo del rango util.
    Rango validado: eps en [0.25, 2.0], con dt de integracion ~0.0125 ms.
    """
    if eps <= 0.0:
        return NoPerturbation()
    r = 0.10 * eps
    sat = 3.0 / eps            # eps chico -> satura muy tarde -> casi lineal
    # Piso de tau_act: por debajo de ~10 pasos del integrador la ecuacion del
    # actuador es rigida y el RK4 de paso fijo se desestabiliza en silencio.
    TAU_MIN = 0.15
    tau_act = max(1.0 * eps, TAU_MIN)
    if 1.0 * eps < TAU_MIN:
        import warnings
        warnings.warn(
            f"default_uncertainty(eps={eps:g}): tau_act se limito a {TAU_MIN} ms "
            f"para que el integrador sea estable. Por debajo de eps~{TAU_MIN:g} la "
            f"perilla deja de ser continua; para 'sin perturbacion' usa eps=0.",
            RuntimeWarning, stacklevel=2)
    return CompositePerturbation([Refractoriness(r=r), Actuator(sat=sat, tau_act=tau_act)])


# Registro por nombre, para que los scripts puedan pedir una familia por string.
REGISTRY = {
    "none": NoPerturbation,
    "refractoriness": Refractoriness,
    "actuator": Actuator,
    "adaptation": Adaptation,
    "depression": SynapticDepression,
    "hidden_population": HiddenPopulation,
    "hetero_sigmoid": HeterogeneousSigmoid,
    "process_noise": ProcessNoise,
    "weight_drift": WeightDrift,
}
