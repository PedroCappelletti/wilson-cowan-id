"""Modelo de Wilson-Cowan.

Implementación fiel al simulador de MATLAB en
`documentos/simulador_wilson_cowan.m`.

Sistema de dos EDOs con estado X = [I, E] (inhibitoria, excitatoria):

    dI/dt = (1/ti) * ( -I + S_i(wIE*E - wII*I + Q(t) - thetai) - ki )
    dE/dt = (1/te) * ( -E + S_e(wEE*E - wEI*I + P(t) - thetae) - ke )

con la respuesta sigmoidea  S_x(u) = 1 / (1 + exp(-a_x * u))  y las
constantes  ke = S_e(-thetae) = 1/(1+exp(ae*thetae)),
            ki = S_i(-thetai) = 1/(1+exp(ai*thetai)),
que se restan para que el reposo E = I = 0 sea un equilibrio sin entrada.

La salida del sistema es el "potencial"  y(t) = E(t) - I(t).

P(t) y Q(t) son las entradas externas a las poblaciones E e I
respectivamente (por defecto, cero; ver `box_pulse`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit


@dataclass
class WilsonCowanParams:
    """Parámetros del modelo de Wilson-Cowan.

    Los valores por defecto son los del simulador de MATLAB, que generan
    oscilaciones sostenidas (ciclo límite).
    """

    # Constantes de tiempo de cada población [s]
    te: float = 1.0
    ti: float = 2.0

    # Pesos sinápticos
    wEE: float = 6.4
    wEI: float = 4.8
    wIE: float = 6.0
    wII: float = 1.2

    # Ganancia de las sigmoideas
    ae: float = 1.2
    ai: float = 1.0

    # Umbrales de las sigmoideas
    thetae: float = 2.8
    thetai: float = 4.0

    @property
    def ke(self) -> float:
        """Offset de la sigmoidea excitatoria: S_e evaluada en reposo (u=0)."""
        return 1.0 / (1.0 + np.exp(self.ae * self.thetae))

    @property
    def ki(self) -> float:
        """Offset de la sigmoidea inhibitoria: S_i evaluada en reposo (u=0)."""
        return 1.0 / (1.0 + np.exp(self.ai * self.thetai))


def sigmoid(u: np.ndarray | float, a: float) -> np.ndarray | float:
    """Respuesta sigmoidea S(u) = 1 / (1 + exp(-a * u)).

    El umbral no entra acá: se incluye en `u` como (... - theta), igual que
    en el simulador de MATLAB. Se usa expit (= 1/(1+exp(-x))) por estabilidad
    numérica.
    """
    return expit(a * u)


def box_pulse(
    amplitude: float, t_on: float, t_off: float
) -> Callable[[float], float]:
    """Devuelve una función de pulso cuadrado.

    f(t) = amplitude  si  t_on <= t < t_off,  y  0 en otro caso.

    Reproduce las entradas Pstim/Qstim del simulador de MATLAB. Los valores
    del MATLAB son:
        P (a la población E): box_pulse(0.8, 100, 400)
        Q (a la población I): box_pulse(0.6, 200, 500)
    """

    def f(t: float) -> float:
        return amplitude if (t_on <= t < t_off) else 0.0

    return f


def zero_input(t: float) -> float:
    """Entrada nula (sin estímulo externo)."""
    return 0.0


class WilsonCowan:
    """Modelo de Wilson-Cowan con entradas externas dependientes del tiempo.

    Args:
        params: parámetros del modelo.
        P: entrada externa a la población excitatoria, P(t). Por defecto 0.
        Q: entrada externa a la población inhibitoria, Q(t). Por defecto 0.
    """

    def __init__(
        self,
        params: WilsonCowanParams | None = None,
        P: Callable[[float], float] = zero_input,
        Q: Callable[[float], float] = zero_input,
    ) -> None:
        self.params = params or WilsonCowanParams()
        self.P = P
        self.Q = Q

    def rhs(self, t: float, state: np.ndarray) -> list[float]:
        """Lado derecho del sistema de EDOs.

        Args:
            t: tiempo actual.
            state: vector de estado [I, E].

        Returns:
            [dI/dt, dE/dt].
        """
        p = self.params
        I, E = state

        # Argumentos de cada sigmoidea (incluyen el umbral y el estímulo).
        u_i = p.wIE * E - p.wII * I + self.Q(t) - p.thetai
        u_e = p.wEE * E - p.wEI * I + self.P(t) - p.thetae

        dI = (1.0 / p.ti) * (-I + sigmoid(u_i, p.ai) - p.ki)
        dE = (1.0 / p.te) * (-E + sigmoid(u_e, p.ae) - p.ke)
        return [dI, dE]

    def simulate(
        self,
        I0: float = 0.0,
        E0: float = 0.0,
        t_span: tuple[float, float] = (0.0, 600.0),
        t_eval: np.ndarray | None = None,
        rel_tol: float = 1e-3,
        abs_tol: float = 1e-6,
        method: str = "RK45",
    ) -> dict[str, np.ndarray]:
        """Integra el sistema (equivalente a ode45 del simulador de MATLAB).

        Args:
            I0, E0: condiciones iniciales de las poblaciones I y E.
            t_span: (t_inicial, t_final).
            t_eval: tiempos en los que devolver la solución. Si es None, usa
                los puntos adaptativos que elige el integrador (como ode45).
            rel_tol, abs_tol: tolerancias del integrador.
            method: método de integración de scipy (RK45 ≈ ode45).

        Returns:
            dict con:
                't': tiempos, shape (N,)
                'I': tasa inhibitoria, shape (N,)
                'E': tasa excitatoria, shape (N,)
                'y': salida y = E - I, shape (N,)
                'P': estímulo P(t) muestreado, shape (N,)
                'Q': estímulo Q(t) muestreado, shape (N,)
        """
        sol = solve_ivp(
            fun=self.rhs,
            t_span=t_span,
            y0=[I0, E0],
            method=method,
            t_eval=t_eval,
            rtol=rel_tol,
            atol=abs_tol,
        )

        t = sol.t
        I = sol.y[0]
        E = sol.y[1]
        P = np.array([self.P(ti) for ti in t])
        Q = np.array([self.Q(ti) for ti in t])

        return {
            "t": t,
            "I": I,
            "E": E,
            "y": E - I,
            "P": P,
            "Q": Q,
        }
