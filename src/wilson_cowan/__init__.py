# Subpaquete del modelo de Wilson-Cowan (fiel al simulador de MATLAB).
# Reexporta lo principal para poder importarlo comodo desde fuera.

from .model import (  # noqa: F401
    WilsonCowan,
    WilsonCowanParams,
    sigmoid,
    box_pulse,
    zero_input,
    plot_results,
)
