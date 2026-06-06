"""Physics-Informed Neural Network para el modelo de Wilson-Cowan."""

from .network import PINN  # noqa: F401
from .losses import data_loss, physics_loss, initial_condition_loss  # noqa: F401
from .train import Trainer  # noqa: F401
