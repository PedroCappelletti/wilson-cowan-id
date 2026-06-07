# =============================================================================
#  FUNCIONES DE PERDIDA DE LA PINN
# =============================================================================
#
#  La perdida total combina:
#
#      L = w_data * L_datos  +  w_physics * L_fisica  +  w_ic * L_inicial
#
#    - L_datos  : que la red pase por las mediciones [I, E].
#    - L_fisica : que la red cumpla las ecuaciones de Wilson-Cowan. ACA entran
#                 los parametros (identificados y/o fijos): se arma el residuo
#                 de la ODE y se lo manda a cero.
#    - L_inicial: que la red respete la condicion inicial [I0, E0].
#
#  El residuo se construye con la estructura EXACTA de model.py: estado [I, E],
#  sigmoide con el umbral adentro de la entrada, y offset de reposo ke/ki restado.
# =============================================================================

from __future__ import annotations

import torch
from torch.autograd import grad


# -----------------------------------------------------------------------------
#  L_datos: error cuadratico medio entre prediccion y observaciones.
# -----------------------------------------------------------------------------
def data_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean()      # pred, target: (N, 2) = [I, E]


# -----------------------------------------------------------------------------
#  L_fisica: residuo de las ecuaciones de Wilson-Cowan en puntos de colocacion.
#
#  `params` es un dict con TODOS los parametros necesarios:
#     te, ti, ae, ai, thetae, thetai, ke, ki, wEE, wEI, wIE, wII
#  Cada valor puede ser un float (parametro fijo) o un tensor (peso identificado,
#  con gradiente). El residuo funciona igual en ambos casos.
# -----------------------------------------------------------------------------
def physics_loss(
    model: torch.nn.Module,
    t_c: torch.Tensor,      # (M, 1) tiempos de colocacion donde exigimos la ODE
    P_c: torch.Tensor,      # (M, 1) estimulo P(t) en esos tiempos
    Q_c: torch.Tensor,      # (M, 1) estimulo Q(t) en esos tiempos
    params: dict,           # todos los parametros (fijos y/o identificados)
) -> torch.Tensor:
    # Necesitamos derivar la salida respecto de t -> t_c registra gradiente.
    t_c = t_c.clone().requires_grad_(True)

    out = model(t_c)        # (M, 2)
    I = out[:, 0:1]
    E = out[:, 1:2]

    # Derivadas temporales por autograd. create_graph=True: la perdida fisica
    # tiene que poder backpropagarse (depende de estas derivadas).
    dI = grad(I, t_c, torch.ones_like(I), create_graph=True)[0]
    dE = grad(E, t_c, torch.ones_like(E), create_graph=True)[0]

    p = params  # alias corto

    # MISMA estructura que rhs() en model.py (umbral adentro de la entrada):
    u_i = p["wIE"] * E - p["wII"] * I + Q_c - p["thetai"]   # entrada a I
    u_e = p["wEE"] * E - p["wEI"] * I + P_c - p["thetae"]   # entrada a E

    # torch.sigmoid(a*u) == expit(a*u). Se resta el offset de reposo ke/ki.
    rhs_I = (1.0 / p["ti"]) * (-I + torch.sigmoid(p["ai"] * u_i) - p["ki"])
    rhs_E = (1.0 / p["te"]) * (-E + torch.sigmoid(p["ae"] * u_e) - p["ke"])

    # Residuo = (derivada de la red) - (lo que dice la ecuacion). Ideal ~ 0.
    res_I = dI - rhs_I
    res_E = dE - rhs_E
    return (res_I ** 2 + res_E ** 2).mean()


# -----------------------------------------------------------------------------
#  L_inicial: penaliza la desviacion respecto de la condicion inicial [I0, E0].
# -----------------------------------------------------------------------------
def initial_condition_loss(
    model: torch.nn.Module,
    t0: torch.Tensor,       # (1, 1) tiempo inicial
    ic: torch.Tensor,       # (1, 2) condicion inicial [I0, E0]
) -> torch.Tensor:
    return ((model(t0) - ic) ** 2).mean()
