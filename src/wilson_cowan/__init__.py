# Subpaquete del modelo de Wilson-Cowan (fiel al simulador de MATLAB).
# Reexporta lo principal para poder importarlo comodo desde fuera.

from .model import (  # noqa: F401
    WilsonCowan,
    WilsonCowanParams,
    sigmoid,
    box_pulse,
    zero_input,
    sine_pulse,
    multisine_pulse,
    chirp_pulse,
    aprbs_pulse,
    theta_gamma_pulse,
    square_wave_pulse,
    prbs_pulse,
    poisson_pulse,
    plot_results,
)

# Incertidumbre dinamica: perturbaciones que vuelven al simulador distinto del
# modelo que se entrena (lo que le da trabajo real a la correccion gray-box).
from .uncertainty import (  # noqa: F401
    Perturbation,
    NoPerturbation,
    Refractoriness,
    Actuator,
    Adaptation,
    SynapticDepression,
    HiddenPopulation,
    HeterogeneousSigmoid,
    ProcessNoise,
    WeightDrift,
    CompositePerturbation,
    default_uncertainty,
    REGISTRY,
)
