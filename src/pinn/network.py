# =============================================================================
#  ARQUITECTURA DE LA PINN  (Physics-Informed Neural Network)
# =============================================================================
#
#  Que es: una red neuronal que cumple DOS roles a la vez:
#    1) APROXIMAR LA TRAYECTORIA: mapea el tiempo t -> [I, E]. Es una curva
#       suave y derivable que pasa por los datos medidos.
#    2) GUARDAR LOS PARAMETROS A IDENTIFICAR: los pesos de Wilson-Cowan que
#       queremos recuperar viven aca adentro como parametros entrenables.
#
#  CUALES PESOS SON ENTRENABLES lo decide el argumento `identify`:
#    - identify=()                          -> NINGUNO. La red solo ajusta la
#                                              trayectoria (problema FORWARD,
#                                              Paso 4.0: chequeo de plomeria).
#    - identify=("wEE","wEI","wIE","wII")   -> los 4 pesos (problema INVERSO,
#                                              Pasos 4.1 y 4.2).
#
#  Por que la red predice el estado si lo que queremos son los parametros:
#  porque para escribir la "perdida fisica" (que la ecuacion se cumpla) hace
#  falta el estado [I, E] y sus derivadas dI/dt, dE/dt. La red da el estado
#  suave y de ahi salen las derivadas con autograd. Ver losses.py.
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
#  Utilidad: inversa de softplus.
#  softplus(x) = log(1 + e^x) convierte cualquier real en uno POSITIVO. La
#  usamos para forzar que los pesos identificados sean > 0 (en el modelo los
#  signos ya estan explicitos: -wEI, -wII). Guardamos un valor "crudo" sin
#  restriccion y el peso real es softplus(crudo). Para arrancar el crudo desde
#  un guess inicial dado, hay que invertir softplus.
# -----------------------------------------------------------------------------
def inv_softplus(y: float) -> torch.Tensor:
    return torch.log(torch.expm1(torch.tensor(float(y))))   # x = log(e^y - 1)


class PINN(nn.Module):
    """Red t -> [I, E], con cero o mas pesos de Wilson-Cowan entrenables.

    Args:
        hidden_dim, n_layers: tamano del MLP.
        t_min, t_max: rango temporal, para NORMALIZAR la entrada t a [-1, 1]
            (clave: si se le mete t=0..600 crudo a tanh, se satura).
        identify: tupla con los nombres de los pesos a identificar. () = ninguno.
        w_init: dict {nombre: valor_inicial} para los pesos identificados.
            Es el punto de partida de la busqueda, NO el valor real.
            Si es None, arranca todos en 1.0.
    """

    WEIGHT_NAMES = ("wEE", "wEI", "wIE", "wII")

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 4,
        t_min: float = 0.0,
        t_max: float = 600.0,
        identify: tuple[str, ...] = ("wEE", "wEI", "wIE", "wII"),
        w_init: dict[str, float] | None = None,
        n_fourier: int = 128,
        fourier_scale: float = 6.0,
    ) -> None:
        super().__init__()

        # --- Rango temporal para normalizar (buffer: viaja con el modelo, no se entrena).
        self.register_buffer("t_min", torch.tensor(float(t_min)))
        self.register_buffer("t_max", torch.tensor(float(t_max)))

        # --- FOURIER FEATURES: en vez de meter el tiempo "pelado" al MLP, lo
        #     codificamos en senos y cosenos de muchas frecuencias. Esto rompe el
        #     "sesgo espectral" (un MLP con tanh no puede ajustar oscilaciones
        #     marcadas a partir de t crudo). Las frecuencias B son FIJAS (no se
        #     entrenan); el MLP aprende como combinarlas.
        #       features(t) = [sin(2*pi*B*tn), cos(2*pi*B*tn)]
        #     n_fourier  = cuantas frecuencias (0 = desactivar, MLP clasico).
        #     fourier_scale = dispersion de las frecuencias (mas grande = capta
        #                     oscilaciones mas rapidas / picos mas agudos).
        self.n_fourier = n_fourier
        if n_fourier > 0:
            gen = torch.Generator().manual_seed(0)        # reproducible
            B = torch.randn(n_fourier, generator=gen) * fourier_scale
            self.register_buffer("B", B)
            in_features = 2 * n_fourier
        else:
            self.B = None
            in_features = 1

        # --- El MLP: features(t) -> 2 salidas [I, E].
        capas: list[nn.Module] = [nn.Linear(in_features, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            capas += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        capas += [nn.Linear(hidden_dim, 2)]   # capa final sin activacion
        self.net = nn.Sequential(*capas)

        # --- Pesos a IDENTIFICAR (si los hay). Guardamos el "crudo"; el peso
        #     real es softplus(crudo) > 0. Es nn.Parameter -> el optimizador lo ajusta.
        self.identify = tuple(identify)
        if self.identify:
            w_init = w_init or {name: 1.0 for name in self.identify}
            crudos = torch.stack([inv_softplus(w_init[name]) for name in self.identify])
            self.raw_w = nn.Parameter(crudos)
        else:
            # Forward puro: no hay pesos entrenables (Paso 4.0).
            self.register_parameter("raw_w", None)

    # -------------------------------------------------------------------------
    #  Forward: dado t, devuelve [I, E]. Normaliza t antes del MLP.
    # -------------------------------------------------------------------------
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        tn = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0   # -> [-1, 1]
        if self.B is not None:
            # tn: (N,1), B: (n_fourier,)  ->  proj: (N, n_fourier)
            proj = 2.0 * torch.pi * tn * self.B.view(1, -1)
            feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)
        else:
            feats = tn
        return self.net(feats)                                         # (N, 2) = [I, E]

    # -------------------------------------------------------------------------
    #  Los pesos identificados, en su valor REAL (positivo), como diccionario
    #  {nombre: tensor}. Vacio si no se identifica ninguno.
    # -------------------------------------------------------------------------
    def identified_weights(self) -> dict[str, torch.Tensor]:
        if not self.identify:
            return {}
        w = F.softplus(self.raw_w)
        return {name: w[i] for i, name in enumerate(self.identify)}

    # Version comoda para leer/loguear (floats, sin gradiente).
    def weights_dict(self) -> dict[str, float]:
        return {k: float(v.detach()) for k, v in self.identified_weights().items()}
