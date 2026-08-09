# =============================================================================
#  MODELO DE WILSON-COWAN
#  Traduccion fiel del simulador de MATLAB: documentos/simulador_wilson_cowan.m
# =============================================================================
#
#  Que es: un modelo de dos poblaciones de neuronas que se influyen entre si.
#    - E = poblacion EXCITATORIA (empuja la actividad hacia arriba)
#    - I = poblacion INHIBITORIA (frena la actividad)
#  La interaccion entre las dos puede producir oscilaciones sostenidas
#  (un "ciclo limite"), parecidas a un ritmo cerebral.
#
#  El sistema son dos ecuaciones diferenciales (una para I y otra para E).
#  El estado del sistema en cada instante es el par X = [I, E]:
#
#    dI/dt = (1/ti) * ( -I + S_i(wIE*E - wII*I + Q(t) - thetai) - ki )
#    dE/dt = (1/te) * ( -E + S_e(wEE*E - wEI*I + P(t) - thetae) - ke )
#
#  donde:
#    - S(u) = 1/(1+exp(-a*u)) es la "sigmoidea" (curva en forma de S que
#      convierte la entrada total de una poblacion en su tasa de disparo).
#    - ke, ki son constantes que se restan para que el reposo (E=I=0) sea
#      un punto de equilibrio cuando no hay entrada externa.
#    - P(t) y Q(t) son las ENTRADAS EXTERNAS (la "fuerza" que se le inyecta
#      al sistema desde afuera) a las poblaciones E e I respectivamente.
#
#  La salida que nos interesa es el "potencial":  y(t) = E(t) - I(t).
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit


# =============================================================================
#  SECCION 1: PARAMETROS DEL SISTEMA
# =============================================================================
# De donde salen: son exactamente los valores del simulador de MATLAB.
# Estan elegidos para que el sistema genere oscilaciones sostenidas
# (ciclo limite). Si los cambias, cambia el comportamiento del modelo.

@dataclass
class WilsonCowanParams:
    # --- Constantes de tiempo [s]: que tan rapido reacciona cada poblacion.
    #     Mas grande = mas lenta.
    te: float = 1.0   # poblacion excitatoria
    ti: float = 2.0   # poblacion inhibitoria

    # --- Pesos sinapticos: cuanto influye cada poblacion sobre cada otra.
    wEE: float = 6.4  # E sobre si misma (autoexcitacion)
    wEI: float = 4.8  # I sobre E (cuanto la frena I)
    wIE: float = 6.0  # E sobre I (cuanto la activa E)
    wII: float = 1.2  # I sobre si misma (autoinhibicion)

    # --- Ganancia de las sigmoideas: que tan "abrupta" es la curva en S.
    ae: float = 1.2   # de la poblacion excitatoria
    ai: float = 1.0   # de la poblacion inhibitoria

    # --- Umbrales de las sigmoideas: a partir de que entrada la poblacion
    #     empieza a dispararse con fuerza.
    thetae: float = 2.8  # de la poblacion excitatoria
    thetai: float = 4.0  # de la poblacion inhibitoria

    @property
    def ke(self) -> float:
        # Offset de la sigmoidea excitatoria. Es el valor de la sigmoidea
        # en reposo (entrada total = 0). Se resta para que, sin estimulo,
        # E=0 sea un equilibrio (que el sistema "duerma" en cero).
        return 1.0 / (1.0 + np.exp(self.ae * self.thetae))

    @property
    def ki(self) -> float:
        # Lo mismo que ke, pero para la poblacion inhibitoria.
        return 1.0 / (1.0 + np.exp(self.ai * self.thetai))


# La funcion sigmoidea: convierte la entrada total "u" de una poblacion en
# una tasa de disparo entre 0 y 1. El umbral NO entra aca: viene ya restado
# dentro de "u" (igual que en el MATLAB). Usamos expit (= 1/(1+exp(-x)))
# porque es la misma formula pero numericamente estable (no se desborda).
def sigmoid(u: np.ndarray | float, a: float) -> np.ndarray | float:
    return expit(a * u)


# =============================================================================
#  SECCION 2: SEÑALES DE ENTRADA / "FUERZA" EXTERNA  P(t) y Q(t)
# =============================================================================
# Que son: el estimulo externo que se le inyecta al sistema desde afuera.
#   - P(t) entra a la poblacion E.
#   - Q(t) entra a la poblacion I.
# Como se obtienen: en el MATLAB son pulsos cuadrados (la señal vale una
#   amplitud constante durante una ventana de tiempo, y 0 fuera de ella).
# Donde se usan: dentro de la funcion derivada (rhs), sumados a la entrada
#   total de cada poblacion. Sin estimulo el sistema queda quieto en reposo;
#   al prender el pulso, arranca a oscilar.

# Fabrica un pulso cuadrado: vale "amplitude" entre t_on y t_off, 0 si no.
# En el MATLAB los valores son:  P = box_pulse(0.8, 100, 400)
#                                Q = box_pulse(0.6, 200, 500)
def box_pulse(amplitude: float, t_on: float, t_off: float) -> Callable[[float], float]:
    def f(t: float) -> float:
        return amplitude if (t_on <= t < t_off) else 0.0
    return f


# Entrada nula: para cuando no se quiere aplicar ningun estimulo externo.
def zero_input(t: float) -> float:
    return 0.0


# =============================================================================
#  ENTRADAS QUE VARIAN EN EL TIEMPO (excitacion persistente)
# =============================================================================
# Por que: un pulso cuadrado es casi "DC" (constante mientras esta prendido) y
# excita el sistema de forma pobre -> lo lleva por un rango limitado de estados.
# Entradas que VARIAN en el tiempo (senoides, multiseno, chirp) recorren mas
# estados -> el problema inverso queda mejor condicionado y los parametros se
# identifican con mas precision (excitacion persistente, Ljung). El simulador
# de MATLAB ya soporta esto: pre-genera P,Q en una grilla y las interpola.
#
# Todas estan GATEADAS a una ventana [t_on, t_off) (0 fuera) y por defecto llevan
# un offset = amplitude para que la senal quede >= 0 (estimulo no negativo, como
# en la actuacion optogenetica). Pasar offset explicito para cambiarlo.

# Senoide de una frecuencia: offset + amplitude*sin(2*pi*freq*(t - t_on)).
def sine_pulse(amplitude: float, freq: float, t_on: float, t_off: float,
               offset: float | None = None) -> Callable[[float], float]:
    off = amplitude if offset is None else offset
    def f(t: float) -> float:
        if t_on <= t < t_off:
            return off + amplitude * np.sin(2.0 * np.pi * freq * (t - t_on))
        return 0.0
    return f


# Multiseno: suma de senoides de varias frecuencias (excitacion mas rica).
# La amplitud total se reparte entre las frecuencias.
def multisine_pulse(amplitude: float, freqs, t_on: float, t_off: float,
                    offset: float | None = None) -> Callable[[float], float]:
    freqs = list(freqs)
    a = amplitude / len(freqs)
    off = amplitude if offset is None else offset
    def f(t: float) -> float:
        if t_on <= t < t_off:
            return off + sum(a * np.sin(2.0 * np.pi * fr * (t - t_on)) for fr in freqs)
        return 0.0
    return f


# Chirp: barrido lineal de frecuencia f0 -> f1 dentro de la ventana. Excita un
# rango continuo de frecuencias (muy bueno para identificabilidad).
def chirp_pulse(amplitude: float, f0: float, f1: float, t_on: float, t_off: float,
                offset: float | None = None) -> Callable[[float], float]:
    off = amplitude if offset is None else offset
    dur = max(t_off - t_on, 1e-9)
    k = (f1 - f0) / dur
    def f(t: float) -> float:
        if t_on <= t < t_off:
            tau = t - t_on
            phase = 2.0 * np.pi * (f0 * tau + 0.5 * k * tau * tau)
            return off + amplitude * np.sin(phase)
        return 0.0
    return f


# =============================================================================
#  ENTRADAS TIPO PULSO / ESCALON  (on-off, realizables con optogenetica)
# =============================================================================
# Por que estas y no senoides: para IDENTIFICAR un sistema no lineal lo que mas
# importa es recorrer muchas AMPLITUDES (la sigmoidea responde distinto en cada
# punto de operacion) ademas de muchas frecuencias. Las senoides (seno, multiseno,
# chirp) varian poco la amplitud y NO son on/off, asi que no son realizables con
# estimulacion optogenetica (que solo prende/apaga). Estas entradas si: son todas
# >= 0 y conmutan, y barren amplitud y/o frecuencia mucho mejor.
#
# Todas estan GATEADAS a una ventana [t_on, t_off) (0 fuera) y son >= 0.
#
# DETALLE IMPORTANTE de implementacion: el integrador (solve_ivp) llama a la
# entrada en tiempos salteados y repetidos, no en orden. Por eso las entradas
# aleatorias PRE-CALCULAN su "agenda" (cuando cambian y a que valor) a partir de
# la semilla, y la funcion f(t) solo CONSULTA en que tramo cae t. Asi f(t) es una
# funcion pura del tiempo (mismo t -> mismo valor siempre), reproducible con seed.
#
# NOTA sobre las frecuencias: van en las MISMAS unidades de tiempo que la
# simulacion (segundos aca; t hasta ~600, te=1, ti=2). No son los Hz biologicos
# (gamma ~40 Hz, theta ~6 Hz): a esa velocidad el sistema ni responde. Lo que se
# conserva es la RELACION entre frecuencias (gamma/theta ~6:1) y que caigan en la
# banda en la que el sistema reacciona (~0.005-0.05 en estas unidades).

# --- APRBS: escalones de amplitud Y duracion aleatorias. El "caballo de batalla"
#     para identificacion no lineal: cubre amplitud x frecuencia. En cada tramo
#     elige una duracion al azar en [dwell_min, dwell_max] y una amplitud al azar
#     en [0, amplitude].
def aprbs_pulse(amplitude: float, t_on: float, t_off: float,
                dwell_min: float, dwell_max: float, seed: int = 0,
                amp_min: float = 0.0) -> Callable[[float], float]:
    # amp_min: piso de amplitud. Con 0.0 cada tramo puede caer cerca de cero (la
    # excitacion se "apaga" seguido). Subirlo mantiene la señal viva (mejor para
    # identificar) sin perder el barrido de amplitudes hasta `amplitude`.
    rng = np.random.default_rng(seed)
    bordes = [t_on]          # tiempos donde cambia el escalon
    valores: list[float] = []  # amplitud de cada tramo
    t = t_on
    while t < t_off:
        t = t + rng.uniform(dwell_min, dwell_max)
        bordes.append(min(t, t_off))
        valores.append(rng.uniform(amp_min, amplitude))
    bordes_a = np.asarray(bordes)
    valores_a = np.asarray(valores)

    def f(t: float) -> float:
        if t_on <= t < t_off:
            # indice del ultimo borde <= t  ->  en que tramo cae t.
            i = int(np.searchsorted(bordes_a, t, side="right")) - 1
            i = min(max(i, 0), len(valores_a) - 1)
            return float(valores_a[i])
        return 0.0
    return f


# --- Theta-gated gamma: rafagas de pulsos rapidos (gamma) cuya intensidad la
#     modula una envolvente lenta (theta). Es el REGIMEN PROPIO del proyecto (lo
#     que el controlador induce). El tren gamma da el on/off; la envolvente theta
#     (en [0,1]) sube y baja la amplitud de las rafagas.
def theta_gamma_pulse(amplitude: float, f_gamma: float, f_theta: float,
                      t_on: float, t_off: float, duty: float = 0.5) -> Callable[[float], float]:
    def f(t: float) -> float:
        if t_on <= t < t_off:
            tau = t - t_on
            env = 0.5 * (1.0 + np.sin(2.0 * np.pi * f_theta * tau))  # envolvente theta en [0,1]
            fase = (f_gamma * tau) % 1.0                              # fase del tren gamma
            gamma_on = 1.0 if fase < duty else 0.0                    # pulso gamma prendido/apagado
            return amplitude * env * gamma_on
        return 0.0
    return f


# --- Onda cuadrada / tren de pulsos: prende/apaga periodico. 'duty' = fraccion
#     del periodo que esta prendido. Estandar en DBS y optogenetica.
def square_wave_pulse(amplitude: float, freq: float, t_on: float, t_off: float,
                      duty: float = 0.5) -> Callable[[float], float]:
    def f(t: float) -> float:
        if t_on <= t < t_off:
            fase = (freq * (t - t_on)) % 1.0
            return amplitude if fase < duty else 0.0
        return 0.0
    return f


# --- PRBS: secuencia binaria pseudo-aleatoria. Cada 'bit_period' el valor es 0 o
#     'amplitude' al azar. Banda ancha, clasico de identificacion (Ljung). Binario
#     puro = optogenetica directa. Limitacion: una sola amplitud (no barre el eje
#     amplitud como APRBS), por eso suele ir como complemento.
def prbs_pulse(amplitude: float, t_on: float, t_off: float,
               bit_period: float, seed: int = 0) -> Callable[[float], float]:
    rng = np.random.default_rng(seed)
    n_bits = max(int(np.ceil((t_off - t_on) / bit_period)), 1)
    bits = rng.integers(0, 2, size=n_bits)   # 0 o 1 en cada bit

    def f(t: float) -> float:
        if t_on <= t < t_off:
            i = min(int((t - t_on) // bit_period), len(bits) - 1)
            return amplitude * float(bits[i])
        return 0.0
    return f


# --- Tren de Poisson: pulsos cortos (ancho 'pulse_width') en tiempos aleatorios,
#     con 'rate' eventos por unidad de tiempo (inter-arribos exponenciales).
#     Naturalista: imita la estadistica de disparos neuronales reales.
def poisson_pulse(amplitude: float, rate: float, t_on: float, t_off: float,
                  pulse_width: float, seed: int = 0) -> Callable[[float], float]:
    rng = np.random.default_rng(seed)
    tiempos = []             # instantes en que ocurre cada evento
    t = t_on
    while True:
        t = t + rng.exponential(1.0 / rate)
        if t >= t_off:
            break
        tiempos.append(t)
    tiempos_a = np.asarray(tiempos)

    def f(t: float) -> float:
        if t_on <= t < t_off and tiempos_a.size:
            # ultimo evento <= t; si t cae dentro de su ancho de pulso -> prendido.
            i = int(np.searchsorted(tiempos_a, t, side="right")) - 1
            if i >= 0 and (t - tiempos_a[i]) < pulse_width:
                return amplitude
        return 0.0
    return f


# =============================================================================
#  CLASE PRINCIPAL: junta los parametros + las entradas + la derivada + la
#  integracion en un solo objeto facil de usar.
# =============================================================================
class WilsonCowan:
    # Al crear el modelo se le pasan:
    #   - params: los parametros (Seccion 1). Si no, usa los del MATLAB.
    #   - P, Q: las funciones de entrada externa (Seccion 2). Por defecto 0.
    #   - perturbation: incertidumbre dinamica OPCIONAL (ver uncertainty.py).
    #       None -> el simulador es Wilson-Cowan puro, exactamente como antes.
    #       Si se pasa una, el simulador deja de ser las mismas ecuaciones que
    #       el modelo que despues se entrena: ese hueco es lo que aprende la
    #       correccion neuronal g_phi (gray-box).
    def __init__(
        self,
        params: WilsonCowanParams | None = None,
        P: Callable[[float], float] = zero_input,
        Q: Callable[[float], float] = zero_input,
        perturbation=None,
    ) -> None:
        self.params = params or WilsonCowanParams()
        self.P = P
        self.Q = Q
        self.perturbation = perturbation

    # =========================================================================
    #  SECCION 3: FUNCION DERIVADA (el "corazon" del modelo)
    # =========================================================================
    # Dada la situacion actual (tiempo t y estado [I, E]), devuelve la
    # VELOCIDAD de cambio del estado: [dI/dt, dE/dt]. El integrador la llama
    # muchisimas veces para ir avanzando la solucion paso a paso.
    def rhs(self, t: float, state: np.ndarray) -> list[float]:
        p = self.params
        I, E = state  # desarmamos el estado en sus dos componentes

        # Entrada total que recibe cada poblacion en este instante:
        #   suma de las influencias internas + el estimulo externo - el umbral.
        u_i = p.wIE * E - p.wII * I + self.Q(t) - p.thetai  # entrada a I
        u_e = p.wEE * E - p.wEI * I + self.P(t) - p.thetae  # entrada a E

        # Cada ecuacion: la poblacion tiende a relajarse (-I, -E) y al mismo
        # tiempo es empujada por su sigmoidea (menos el offset de reposo).
        # Todo escalado por 1/constante_de_tiempo.
        dI = (1.0 / p.ti) * (-I + sigmoid(u_i, p.ai) - p.ki)
        dE = (1.0 / p.te) * (-E + sigmoid(u_e, p.ae) - p.ke)
        return [dI, dE]

    # =========================================================================
    #  SECCION 3-bis: DERIVADA CON INCERTIDUMBRE DINAMICA
    # =========================================================================
    #  Misma ecuacion que rhs(), pero pasando por los cinco "ganchos" de la
    #  perturbacion (ver uncertainty.py). El estado es mas largo: ademas de
    #  [I, E] lleva los ESTADOS OCULTOS que la perturbacion necesite (el estado
    #  del actuador, la corriente de adaptacion, etc.).
    #
    #  Esos estados ocultos NO se miden y NO se guardan como observacion: son
    #  precisamente lo que hace que una correccion g_phi(I,E,P,Q) sin memoria se
    #  quede corta.
    def rhs_aug(self, t: float, state: np.ndarray) -> np.ndarray:
        P_cmd, Q_cmd = self.P(t), self.Q(t)   # estimulo COMANDADO
        dI, dE, d_extra = self.perturbed_field(t, state[0], state[1], state[2:],
                                               P_cmd, Q_cmd)
        return np.concatenate(([dI, dE], d_extra))

    # El campo perturbado con el estimulo dado EXPLICITAMENTE. Se separa de
    # rhs_aug porque en lazo cerrado P y Q no vienen de una funcion del tiempo
    # sino del controlador, y la planta tiene que poder evaluarse igual.
    def perturbed_field(self, t, I, E, extra, P_cmd, Q_cmd):
        p = self.params
        pert = self.perturbation

        # (a) estimulo EFECTIVO (lo que realmente llega a las poblaciones).
        P_eff, Q_eff = pert.inputs(t, I, E, extra, P_cmd, Q_cmd, p)

        # (d) pesos efectivos (pueden depender del tiempo o del uso).
        w = pert.weights(t, I, E, extra, p)
        wEE, wEI, wIE, wII = w if w is not None else (p.wEE, p.wEI, p.wIE, p.wII)

        # (b) corriente extra sumada a la entrada total de cada poblacion.
        du_i, du_e = pert.drive(t, I, E, extra, p)

        u_i = wIE * E - wII * I + Q_eff - p.thetai + du_i
        u_e = wEE * E - wEI * I + P_eff - p.thetae + du_e

        # (c) ganancia multiplicativa sobre la salida de las sigmoideas.
        g_i, g_e = pert.gains(t, I, E, extra, p)
        ke, ki = pert.rest_offsets(p)

        dI = (1.0 / p.ti) * (-I + g_i * pert.sigmoid_i(u_i, p) - ki)
        dE = (1.0 / p.te) * (-E + g_e * pert.sigmoid_e(u_e, p) - ke)

        # (e) termino sumado directamente a las derivadas (ruido de proceso).
        adI, adE = pert.deriv(t, I, E, extra, p)
        dI += adI
        dE += adE

        d_extra = pert.d_extra(t, I, E, extra, P_cmd, Q_cmd, p)
        return dI, dE, d_extra

    # Estimulo efectivo en cada instante (solo para diagnostico y figuras).
    # NUNCA se usa para entrenar: el modelo solo conoce el comandado.
    def effective_inputs(self, t: float, state: np.ndarray) -> tuple[float, float]:
        if self.perturbation is None:
            return self.P(t), self.Q(t)
        return self.perturbation.inputs(t, state[0], state[1], state[2:],
                                        self.P(t), self.Q(t), self.params)

    # =========================================================================
    #  SECCION 4: INTEGRACION NUMERICA (resolver las ecuaciones en el tiempo)
    # =========================================================================
    # Avanza el sistema desde t_inicial hasta t_final partiendo de [I0, E0].
    # Equivale al ode45 del MATLAB: usa Runge-Kutta 4-5 de paso adaptativo.
    def simulate(
        self,
        I0: float = 0.0,                       # condicion inicial de I
        E0: float = 0.0,                       # condicion inicial de E
        t_span: tuple[float, float] = (0.0, 600.0),  # (t_inicial, t_final)
        t_eval: np.ndarray | None = None,      # tiempos donde devolver la sol.
        rel_tol: float = 1e-3,                 # tolerancia relativa (como MATLAB)
        abs_tol: float = 1e-6,                 # tolerancia absoluta (como MATLAB)
        method: str = "RK45",                  # RK45 ≈ ode45
        n_sub: int = 4,                        # subpasos por punto (solo paso fijo)
    ) -> dict[str, np.ndarray]:
        # --- Camino con INCERTIDUMBRE: estado aumentado + RK4 de paso fijo.
        #     Se usa paso fijo porque (i) hay perturbaciones que lo exigen (el
        #     ruido de proceso no se puede integrar con paso adaptativo) y
        #     (ii) es el mismo integrador con el que despues se entrena la red.
        if self.perturbation is not None:
            return self._simulate_perturbed(I0, E0, t_span, t_eval, n_sub)

        # solve_ivp hace todo el trabajo pesado: llama a rhs muchas veces y
        # va construyendo la trayectoria. Si t_eval es None, devuelve los
        # puntos que el propio integrador elige (paso adaptativo, como ode45).
        sol = solve_ivp(
            fun=self.rhs,
            t_span=t_span,
            y0=[I0, E0],
            method=method,
            t_eval=t_eval,
            rtol=rel_tol,
            atol=abs_tol,
        )

        # Separamos los resultados en variables con nombre claro.
        t = sol.t
        I = sol.y[0]  # tasa de la poblacion inhibitoria a lo largo del tiempo
        E = sol.y[1]  # tasa de la poblacion excitatoria a lo largo del tiempo

        # Reconstruimos las entradas externas en esos mismos tiempos (para
        # poder graficarlas despues).
        P = np.array([self.P(ti) for ti in t])
        Q = np.array([self.Q(ti) for ti in t])

        # Devolvemos todo en un diccionario. 'y' = E - I es la salida final.
        return {
            "t": t,
            "I": I,
            "E": E,
            "y": E - I,
            "P": P,
            "Q": Q,
        }

    # =========================================================================
    #  INTEGRACION CON PERTURBACION (RK4 de paso fijo, estado aumentado)
    # =========================================================================
    def _simulate_perturbed(self, I0, E0, t_span, t_eval, n_sub):
        pert = self.perturbation

        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 6000)
        t_eval = np.asarray(t_eval, dtype=float)
        n_out = len(t_eval)

        # Estado aumentado: [I, E, *estados_ocultos_de_la_perturbacion]
        s = np.concatenate(([float(I0), float(E0)], pert.extra0()))
        dim = len(s)

        out = np.zeros((n_out, dim))
        out[0] = s
        Pc = np.zeros(n_out); Qc = np.zeros(n_out)
        Pe = np.zeros(n_out); Qe = np.zeros(n_out)
        Pc[0], Qc[0] = self.P(t_eval[0]), self.Q(t_eval[0])
        Pe[0], Qe[0] = self.effective_inputs(t_eval[0], s)

        f = self.rhs_aug
        for k in range(n_out - 1):
            # Entre dos puntos de salida damos n_sub subpasos, para que el error
            # de discretizacion no dependa de cuan finito sea el muestreo pedido.
            h = (t_eval[k + 1] - t_eval[k]) / n_sub
            t = t_eval[k]
            for _ in range(n_sub):
                k1 = f(t, s)
                k2 = f(t + 0.5 * h, s + 0.5 * h * k1)
                k3 = f(t + 0.5 * h, s + 0.5 * h * k2)
                k4 = f(t + h, s + h * k3)
                s = s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                t += h
            out[k + 1] = s
            Pc[k + 1], Qc[k + 1] = self.P(t_eval[k + 1]), self.Q(t_eval[k + 1])
            Pe[k + 1], Qe[k + 1] = self.effective_inputs(t_eval[k + 1], s)

        I, E = out[:, 0], out[:, 1]
        return {
            "t": t_eval,
            "I": I,
            "E": E,
            "y": E - I,
            # P y Q son SIEMPRE los COMANDADOS: es lo unico que se conoce en un
            # experimento real y lo unico que puede ver el modelo al entrenar.
            "P": Pc,
            "Q": Qc,
            # El estimulo EFECTIVO y los estados ocultos son solo diagnostico.
            "P_eff": Pe,
            "Q_eff": Qe,
            "extra": out[:, 2:],
        }


# =============================================================================
#  SECCION 5: VISUALIZACION (generar y guardar los graficos)
# =============================================================================
# Toma el resultado de simulate() y arma una figura de 3 paneles, igual que
# el MATLAB:
#   1) las entradas externas P y Q
#   2) los estados I y E
#   3) la salida y = E - I
# Donde se guarda: en la ruta "save_path". Por defecto en
#   results/figures/simulacion_wilson_cowan.png  (relativo a la raiz del repo).
# Si show=True ademas la muestra en pantalla.
def plot_results(
    sol: dict[str, np.ndarray],
    save_path: str | Path | None = "results/figures/simulacion_wilson_cowan.png",
    show: bool = False,
):
    # Importamos matplotlib aca adentro para no exigirlo si solo se simula.
    import matplotlib.pyplot as plt

    t = sol["t"]

    # 3 paneles, uno arriba del otro, compartiendo el eje del tiempo.
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    # Panel 1: entradas externas (la "fuerza" aplicada al sistema).
    axes[0].plot(t, sol["Q"], label="Q: entrada a poblacion I", linewidth=1.3)
    axes[0].plot(t, sol["P"], label="P: entrada a poblacion E", linewidth=1.3)
    axes[0].set_ylabel("[xA]")
    axes[0].set_title("Entradas")
    axes[0].legend(loc="upper right")
    axes[0].grid(True)

    # Panel 2: estados (la actividad de cada poblacion en el tiempo).
    axes[1].plot(t, sol["I"], label="I: actividad inhibitoria", linewidth=1.3)
    axes[1].plot(t, sol["E"], label="E: actividad excitatoria", linewidth=1.3)
    axes[1].set_ylabel("[rate]")
    axes[1].set_title("Estados")
    axes[1].legend()
    axes[1].grid(True)

    # Panel 3: salida y = E - I (el "potencial" que observamos).
    axes[2].plot(t, sol["y"], linewidth=1.3)
    axes[2].set_ylabel("[xV]")
    axes[2].set_xlabel("tiempo (s)")
    axes[2].set_title("Salida")
    axes[2].grid(True)

    fig.tight_layout()

    # Guardado: creamos la carpeta destino si no existe y escribimos el PNG.
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig


# =============================================================================
#  EJECUCION DIRECTA: si corres este archivo (python -m src.wilson_cowan.model)
#  reproduce la simulacion completa del MATLAB y guarda el grafico.
# =============================================================================
if __name__ == "__main__":
    # Mismos parametros, entradas y condiciones iniciales que el simulador.
    modelo = WilsonCowan(
        params=WilsonCowanParams(),
        P=box_pulse(0.8, 100.0, 400.0),
        Q=box_pulse(0.6, 200.0, 500.0),
    )

    # Integramos de 0 a 600 s, muestreando en una grilla uniforme para graficar.
    resultado = modelo.simulate(
        I0=0.0,
        E0=0.0,
        t_span=(0.0, 600.0),
        t_eval=np.linspace(0.0, 600.0, 6000),
    )

    # Generamos y guardamos la figura de 3 paneles.
    plot_results(resultado)
    print("Listo. Grafico guardado en results/figures/simulacion_wilson_cowan.png")
