# =============================================================================
#  MODELO DE DINAMICA  ẋ = f_θ(x, P, Q)   (Neural ODE gray-box)
# =============================================================================
#
#  A diferencia de la PINN (que mapea t -> [I,E] para un estimulo FIJO), este
#  modelo es de ESTADOS: dado el estado actual [I,E] y el control [P,Q], devuelve
#  la derivada [dI/dt, dE/dt]. Se integra paso a paso -> se puede meter en un lazo
#  de control donde P,Q se generan online (lo que la PINN no permite).
#
#  Es GRAY-BOX: backbone con la estructura EXACTA de Wilson-Cowan (misma que
#  losses.py / model.py) + una correccion neuronal g_φ opcional. Conservar la
#  estructura WC es REQUISITO para que el controlador IMC (feedback linearization)
#  pueda cancelar el acoplamiento. Ver docs/plan_planta_neural.html.
#
#  Variantes (segun los flags):
#    V0 (white-box): use_correction=False, learnable_weights=False
#        -> son las ecuaciones WC con los pesos dados (planta de referencia).
#    V1 (gray-box):  use_correction=True
#        -> WC + MLP que captura lo que el backbone no modela (la parte aprendida).
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GrayBoxWC(nn.Module):
    WEIGHTS = ("wEE", "wEI", "wIE", "wII")

    def __init__(
        self,
        fixed: dict,                 # te,ti,ae,ai,thetae,thetai,ke,ki (floats)
        w_init: dict,                # wEE,wEI,wIE,wII (p.ej. los identificados por la PINN)
        learnable_weights: bool = False,  # True -> los 4 pesos se siguen ajustando
        use_correction: bool = False,     # True -> agrega la correccion neuronal g_φ (V1)
        hidden: int = 32,
    ) -> None:
        super().__init__()

        # Constantes del modelo (no se entrenan): viajan como buffers.
        for k in ("te", "ti", "ae", "ai", "thetae", "thetai", "ke", "ki"):
            self.register_buffer(k, torch.tensor(float(fixed[k])))

        # Pesos sinapticos. Si son entrenables, se guardan "crudos" (softplus -> >0).
        w = torch.tensor([float(w_init[k]) for k in self.WEIGHTS])
        self._learn = learnable_weights
        if learnable_weights:
            self.raw_w = nn.Parameter(torch.log(torch.expm1(w)))   # inv_softplus
        else:
            self.register_buffer("w_fixed", w)

        # Correccion neuronal opcional. Se inicializa en ~0 (arranca = backbone WC).
        self.use_correction = use_correction
        if use_correction:
            self.g = nn.Sequential(
                nn.Linear(4, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, 2),
            )
            for m in self.g:
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.bias)
            nn.init.zeros_(self.g[-1].weight)   # salida 0 al inicio

    # Pesos reales (positivos).
    def weights(self) -> torch.Tensor:
        return F.softplus(self.raw_w) if self._learn else self.w_fixed

    def weights_dict(self) -> dict[str, float]:
        w = self.weights().detach()
        return {k: float(w[i]) for i, k in enumerate(self.WEIGHTS)}

    # -------------------------------------------------------------------------
    #  ẋ = f_θ(x, P, Q).  x: (...,2)=[I,E];  P,Q: tensores broadcastables a (...,1).
    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        I = x[..., 0:1]
        E = x[..., 1:2]
        P = torch.as_tensor(P, dtype=x.dtype, device=x.device) * torch.ones_like(I)
        Q = torch.as_tensor(Q, dtype=x.dtype, device=x.device) * torch.ones_like(I)

        w = self.weights()
        wEE, wEI, wIE, wII = w[0], w[1], w[2], w[3]

        # MISMA estructura que rhs() en model.py (umbral adentro, offset ke/ki).
        u_i = wIE * E - wII * I + Q - self.thetai
        u_e = wEE * E - wEI * I + P - self.thetae
        dI = (1.0 / self.ti) * (-I + torch.sigmoid(self.ai * u_i) - self.ki)
        dE = (1.0 / self.te) * (-E + torch.sigmoid(self.ae * u_e) - self.ke)

        if self.use_correction:
            corr = self.g(torch.cat([I, E, P, Q], dim=-1))
            dI = dI + corr[..., 0:1]
            dE = dE + corr[..., 1:2]

        return torch.cat([dI, dE], dim=-1)
