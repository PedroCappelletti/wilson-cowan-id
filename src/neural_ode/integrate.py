# =============================================================================
#  INTEGRADOR  (Runge-Kutta 4 de paso fijo, diferenciable)
# =============================================================================
#
#  Avanza el modelo de dinamica f(x,P,Q) en el tiempo. RK4 de paso fijo:
#    - no agrega dependencias (no necesita torchdiffeq),
#    - es diferenciable (sirve para entrenar el Neural ODE por backprop),
#    - el control P,Q se mantiene constante durante cada paso (zero-order hold),
#      que es como llega un control muestreado.
#
#  Para el entrenamiento conviene "multiple shooting": integrar en ventanas cortas
#  reiniciando desde el estado observado (ver train_ode.py, a futuro), no de un
#  tiron los 600 s. Estas funciones son los ladrillos para eso.
# =============================================================================

from __future__ import annotations

import torch


def rk4_step(f, x: torch.Tensor, P, Q, dt: float) -> torch.Tensor:
    """Un paso RK4. f(x,P,Q)->dx. P,Q constantes en el paso (ZOH)."""
    k1 = f(x, P, Q)
    k2 = f(x + 0.5 * dt * k1, P, Q)
    k3 = f(x + 0.5 * dt * k2, P, Q)
    k4 = f(x + dt * k3, P, Q)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rollout(f, x0: torch.Tensor, P_seq: torch.Tensor, Q_seq: torch.Tensor,
            dt: float) -> torch.Tensor:
    """Integra una trayectoria completa dado el estimulo muestreado.

    x0: (...,2) estado inicial.
    P_seq, Q_seq: (T, ...,1) estimulo en cada paso.
    Devuelve (T+1, ...,2) con la trayectoria de estados (incluye x0).
    """
    x = x0
    xs = [x]
    for t in range(P_seq.shape[0]):
        x = rk4_step(f, x, P_seq[t], Q_seq[t], dt)
        xs.append(x)
    return torch.stack(xs, dim=0)
