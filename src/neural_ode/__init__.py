# =============================================================================
#  Modulo: modelo de dinamica (Neural ODE gray-box) + controlador IMC + lazo cerrado
# =============================================================================
#  Para la extension orientada al control (OE3): el modelo aprendido como PLANTA
#  que responde a los estimulos P,Q del controlador. Ver docs/plan_planta_neural.html
# =============================================================================

from .dynamics import GrayBoxWC  # noqa: F401
from .integrate import rk4_step, rollout  # noqa: F401
from .closed_loop import (  # noqa: F401
    IMCController,
    make_true_plant,
    make_neural_plant,
    simulate_closed_loop,
    theta_gamma_refs,
)
