# =============================================================================
#  LAZO CERRADO — controlador IMC + planta (verdadera o aprendida)
# =============================================================================
#
#  Port fiel a Python de  simulador_wilson_cowan_con_control.m  (el controlador
#  que dio el tutor). Es un controlador IMC con linealizacion por realimentacion:
#
#    - Estados del controlador (integradores):  dZ1 = rI - I ,  dZ2 = rE - E
#    - Salida LTI (PI):   Ulti_I = kp_I*(rI-I) + ki_I*Z1   (lazo de I)
#                         Ulti_E = kp_E*(rE-E) + ki_E*Z2   (lazo de E)
#    - Saturacion al dominio de la sigmoidea inversa.
#    - Sigmoidea inversa -> uq, up.
#    - CANCELACION (feedback linearization, usa los pesos):
#          Q = uq - (wIE*E - wII*I)
#          P = up - (wEE*E - wEI*I)
#
#  CLAVE: el controlador usa los pesos (wEE,wEI,wIE,wII) para cancelar el
#  acoplamiento. Si se lo construye con los pesos IDENTIFICADOS (θ̂) y se corre
#  contra la planta verdadera (o contra el modelo aprendido), se mide cuanto
#  degrada el control el error de identificacion -> "validacion orientada al
#  control" (OE3).
#
#  El controlador y la planta estan SEPARADOS: la planta es un callable
#  plant_rhs(I,E,P,Q)->(dI,dE), que puede ser la WC verdadera o el Neural ODE
#  aprendido. Asi se enchufa el modelo aprendido sin tocar el controlador.
#
#  UNIDADES (reconciliado): TODA la cadena trabaja en MILISEGUNDOS. La decision ya
#  estaba tomada en gen_multi_dataset.py ("régimen ms"): el dataset de control, la
#  identificacion (te=1 ms, ti=2 ms; estimulos con freq en Hz -> ciclos/ms via f/1000)
#  y este lazo cerrado (tf=50 ms, refs 120 Hz = 0.12 ciclos/ms) usan la MISMA unidad.
#  Los parametros del modelo son numericamente identicos en ambos lados; lo unico que
#  cambia es el paso de integracion (dt~0.05 ms en identificacion vs dt=0.005 ms aca),
#  que es una eleccion numerica, no de unidades. No hay reescalado de parametros.
#
#  NOTA: el archivo original usa realimentacion de ESTADO COMPLETO (I,E directos,
#  sin filtro de Kalman). Se replica igual. El EKF del paper seria una capa extra.
# =============================================================================

from __future__ import annotations

import math
import numpy as np


# -----------------------------------------------------------------------------
#  Referencias theta-gamma (las senoides del archivo del controlador).
# -----------------------------------------------------------------------------
def theta_gamma_refs(freq_hz: float = 120.0, time_in_ms: bool = True):
    """Devuelve refs(t) -> (rI, rE). Por defecto, las del MATLAB:
        rI = 0.2*sin(2π f t - 0.94) + 0.25
        rE = 0.3*sin(2π f t)        + 0.45
    con f=120 Hz y t en ms (f/1000 ciclos por ms)."""
    f = freq_hz / 1000.0 if time_in_ms else freq_hz

    def refs(t):
        rI = 0.2 * math.sin(2 * math.pi * f * t - 0.94) + 0.25
        rE = 0.3 * math.sin(2 * math.pi * f * t) + 0.45
        return rI, rE
    return refs


# -----------------------------------------------------------------------------
#  Controlador IMC + sigmoidea inversa + cancelacion.
# -----------------------------------------------------------------------------
class IMCController:
    def __init__(
        self,
        fixed: dict,            # ae,ai,thetae,thetai,ke,ki
        weights: dict,          # wEE,wEI,wIE,wII (los que usa la cancelacion: verdaderos o θ̂)
        kp_I: float = 10.0, ki_I: float = 5.0,   # PI del lazo de I  (Ulti1 = 5*Z1 + 10*err)
        kp_E: float = 5.0,  ki_E: float = 5.0,   # PI del lazo de E  (Ulti2 = 5*Z2 + 5*err)
        argmin: float = -100.0, argmax: float = 100.0,
    ) -> None:
        self.f = fixed
        self.w = weights
        self.kp_I, self.ki_I = kp_I, ki_I
        self.kp_E, self.ki_E = kp_E, ki_E

        ai, ki = fixed["ai"], fixed["ki"]
        ae, ke = fixed["ae"], fixed["ke"]
        ti_th, te_th = fixed["thetai"], fixed["thetae"]
        # Limites de saturacion: dominio valido de la sigmoidea inversa.
        self.fim = 0.99999 * (1 / (1 + math.exp(-ai * (argmin - ti_th))) - ki)
        self.fiM = 0.99999 * (1 / (1 + math.exp(-ai * (argmax - ti_th))) - ki)
        self.fem = 0.99999 * (1 / (1 + math.exp(-ae * (argmin - te_th))) - ke)
        self.feM = 0.99999 * (1 / (1 + math.exp(-ae * (argmax - te_th))) - ke)

    def compute(self, Z1, Z2, I, E, rI, rE):
        """Devuelve (P, Q, dZ1, dZ2) dado el estado del controlador y de la planta."""
        f, w = self.f, self.w
        ai, ki, ti_th = f["ai"], f["ki"], f["thetai"]
        ae, ke, te_th = f["ae"], f["ke"], f["thetae"]

        # PI del controlador.
        Ulti_I = self.kp_I * (rI - I) + self.ki_I * Z1
        Ulti_E = self.kp_E * (rE - E) + self.ki_E * Z2

        # Saturacion al dominio de la inversa.
        Usat_I = min(max(Ulti_I, self.fim), self.fiM)
        Usat_E = min(max(Ulti_E, self.fem), self.feM)

        # Sigmoidea inversa.
        uq = (-1.0 / ai) * math.log(-1.0 + 1.0 / (Usat_I + ki)) + ti_th
        up = (-1.0 / ae) * math.log(-1.0 + 1.0 / (Usat_E + ke)) + te_th

        # Cancelacion del acoplamiento (feedback linearization) -> estimulos.
        Q = uq - (w["wIE"] * E - w["wII"] * I)
        P = up - (w["wEE"] * E - w["wEI"] * I)

        dZ1 = rI - I
        dZ2 = rE - E
        return P, Q, dZ1, dZ2


# -----------------------------------------------------------------------------
#  Plantas: callables plant_rhs(I,E,P,Q) -> (dI,dE).
# -----------------------------------------------------------------------------
def make_true_plant(fixed: dict, weights: dict):
    """Planta verdadera: ecuaciones de Wilson-Cowan (V0 / referencia)."""
    te, ti = fixed["te"], fixed["ti"]
    ae, ai = fixed["ae"], fixed["ai"]
    the, thi = fixed["thetae"], fixed["thetai"]
    ke, ki = fixed["ke"], fixed["ki"]
    wEE, wEI, wIE, wII = (weights["wEE"], weights["wEI"], weights["wIE"], weights["wII"])

    def rhs(I, E, P, Q):
        u_i = wIE * E - wII * I + Q - thi
        u_e = wEE * E - wEI * I + P - the
        dI = (1.0 / ti) * (-I + 1.0 / (1.0 + math.exp(-ai * u_i)) - ki)
        dE = (1.0 / te) * (-E + 1.0 / (1.0 + math.exp(-ae * u_e)) - ke)
        return dI, dE
    return rhs


def make_neural_plant(model):
    """Planta aprendida: envuelve un GrayBoxWC (torch) como callable escalar."""
    import torch

    def rhs(I, E, P, Q):
        with torch.no_grad():
            x = torch.tensor([[float(I), float(E)]], dtype=torch.float32)
            dx = model(x, torch.tensor([[float(P)]]), torch.tensor([[float(Q)]]))
        return float(dx[0, 0]), float(dx[0, 1])
    return rhs


# -----------------------------------------------------------------------------
#  Simulacion del lazo cerrado (RK4 sobre el estado aumentado [Z1,Z2,I,E]).
# -----------------------------------------------------------------------------
def simulate_closed_loop(plant_rhs, controller: IMCController, refs,
                         t_span=(0.0, 50.0), dt=0.005, x0=(0.0, 0.0)):
    """Integra controlador + planta. Devuelve dict con arrays t, I, E, y, P, Q, rI, rE."""

    def aug_rhs(state, t):
        Z1, Z2, I, E = state
        rI, rE = refs(t)
        P, Q, dZ1, dZ2 = controller.compute(Z1, Z2, I, E, rI, rE)
        dI, dE = plant_rhs(I, E, P, Q)
        return np.array([dZ1, dZ2, dI, dE]), P, Q, rI, rE

    t0, tf = t_span
    n = int(round((tf - t0) / dt))
    state = np.array([0.0, 0.0, x0[0], x0[1]])

    T, I_a, E_a, P_a, Q_a, rI_a, rE_a = [], [], [], [], [], [], []
    t = t0
    for _ in range(n + 1):
        d1, P, Q, rI, rE = aug_rhs(state, t)
        # Registro en el punto actual.
        T.append(t); I_a.append(state[2]); E_a.append(state[3])
        P_a.append(P); Q_a.append(Q); rI_a.append(rI); rE_a.append(rE)
        # Paso RK4 (refs evaluadas en los sub-tiempos).
        k1, _, _, _, _ = aug_rhs(state, t)
        k2, _, _, _, _ = aug_rhs(state + 0.5 * dt * k1, t + 0.5 * dt)
        k3, _, _, _, _ = aug_rhs(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4, _, _, _, _ = aug_rhs(state + dt * k3, t + dt)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt

    out = {k: np.asarray(v) for k, v in {
        "t": T, "I": I_a, "E": E_a, "P": P_a, "Q": Q_a, "rI": rI_a, "rE": rE_a,
    }.items()}
    out["y"] = out["E"] - out["I"]
    return out
