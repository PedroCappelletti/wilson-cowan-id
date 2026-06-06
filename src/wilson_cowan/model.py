"""Modelo de Wilson-Cowan.

Sistema de dos ecuaciones diferenciales ordinarias que describe la dinámica
de poblaciones de neuronas excitatorias (E) e inhibitorias (I):

    tau_E * dE/dt = -E + (1 - r_E * E) * S_E(c_EE * E - c_EI * I + P)
    tau_I * dI/dt = -I + (1 - r_I * I) * S_I(c_IE * E - c_II * I + Q)

donde S(x) es una función sigmoidea de respuesta.

Este archivo define la estructura; la implementación queda como TODO.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WilsonCowanParams:
    """Parámetros del modelo de Wilson-Cowan."""

    # Constantes de tiempo
    tau_E: float = 1.0
    tau_I: float = 1.0

    # Pesos de acoplamiento
    c_EE: float = 16.0
    c_EI: float = 12.0
    c_IE: float = 15.0
    c_II: float = 3.0

    # Entradas externas
    P: float = 1.0
    Q: float = 1.0

    # Parámetros de la sigmoidea
    a_E: float = 1.3
    theta_E: float = 4.0
    a_I: float = 2.0
    theta_I: float = 3.7

    # Términos refractarios
    r_E: float = 1.0
    r_I: float = 1.0


def sigmoid(x: np.ndarray, a: float, theta: float) -> np.ndarray:
    """Función de respuesta sigmoidea S(x) = 1 / (1 + exp(-a (x - theta)))."""
    raise NotImplementedError


class WilsonCowan:
    """Modelo de Wilson-Cowan."""

    def __init__(self, params: WilsonCowanParams | None = None) -> None:
        self.params = params or WilsonCowanParams()

    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        """Lado derecho del sistema de ODEs: devuelve [dE/dt, dI/dt]."""
        raise NotImplementedError

    def simulate(
        self,
        E0: float,
        I0: float,
        t_span: tuple[float, float],
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integra el sistema y devuelve (t, estados).

        Returns:
            t: vector de tiempos, shape (N,)
            states: trayectoria [E, I], shape (N, 2)
        """
        raise NotImplementedError
