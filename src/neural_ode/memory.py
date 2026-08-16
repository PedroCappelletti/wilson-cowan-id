# =============================================================================
#  MODELOS CON MEMORIA  (etapa 2 del escalado progresivo)
# =============================================================================
#
#  El problema que resuelven: el actuador real tiene un ESTADO PROPIO (el filtro
#  de primer orden P_lag). Una correccion g(I,E) no puede representarlo: dos
#  instantes con el mismo estado observado pueden tener distinta derivada.
#  La unica salida es darle al modelo estados adicionales que se integran junto
#  con (I,E).
#
#  Dos formas de hacerlo, de mas informada a mas agnostica:
#
#    LagGrayBox    -> la forma FISICA exacta: agrega el filtro dP̂/dt=(P−P̂)/τ̂
#                     con τ̂ (y la saturacion) aprendibles. Si esto no funciona,
#                     nada va a funcionar: es la cota superior del enfoque.
#
#    LatentGrayBox -> estados latentes GENERICOS z con dinamica aprendida
#                     ż = h_ψ(I,E,z,P,Q) y una correccion g_φ(I,E,z) que ahora
#                     si puede depender de la historia (a traves de z). Es lo
#                     que generaliza a fisica desconocida.
#
#  Los dos comparten la convencion de estado aumentado:
#      y = [I, E, h_1 ... h_k]      (solo las dos primeras se observan)
#  y el mismo rollout RK4 de integrate.py (forward recibe y, devuelve dy).
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamics import GrayBoxWC
from .integrate import rollout
from .graybox_train import make_windows, param_errors, ALL_P, WEIGHTS, PHYS


def _inv_softplus(v: float) -> torch.Tensor:
    return torch.log(torch.expm1(torch.tensor(float(v))))


class LagGrayBox(nn.Module):
    """WC + el estado del actuador con forma fisica exacta y parametros libres.

    Estado aumentado y = [I, E, P̂, Q̂]:
        dP̂/dt = (P − P̂) / τ̂            <- el filtro que imita al canal ChR2
        ẋ     = f_WC(x, sat(P̂), sat(Q̂))  <- el backbone ve el estimulo FILTRADO
    con sat(u) = tanh(α·u)/α  (α = 1/saturacion; α→0 es el canal lineal).

    Arranque ignorante: τ̂ parte lejos del valor verdadero (1.0 ms) y α parte en
    ~0 (sin saturacion). Los 10 parametros WC parten de 1.0 como siempre.
    """

    n_hidden = 2

    def __init__(self, init_value: float = 1.0, tau_init: float = 0.3,
                 hidden: int = 32):
        super().__init__()
        init = {k: init_value for k in ALL_P}
        self.core = GrayBoxWC(init, {k: init_value for k in WEIGHTS},
                              learnable_weights=True, learnable_params=True,
                              use_correction=False)
        # softplus + piso: τ̂ muy chico vuelve rigida la ecuacion y desestabiliza
        # el RK4 de paso fijo (mismo motivo que el piso de tau_act en la planta).
        self.raw_tau = nn.Parameter(_inv_softplus(tau_init))
        # misma parametrizacion que la variante S: α directo con clamp, para que
        # el gradiente no muera cerca del canal lineal. Arranca en 0.1 (no en
        # ~0): en la primera corrida arranco en 0.01 y se fue al borde inferior
        # del clamp (1e-3), donde el gradiente es nulo y no vuelve a salir.
        self.raw_alpha = nn.Parameter(torch.tensor(0.1))

    def tau(self) -> torch.Tensor:
        return F.softplus(self.raw_tau) + 0.15

    def _eff(self, u: torch.Tensor) -> torch.Tensor:
        alpha = torch.clamp(self.raw_alpha, 1e-3, 10.0)
        return torch.tanh(alpha * u) / alpha

    def h0(self, x0: torch.Tensor, P0: torch.Tensor, Q0: torch.Tensor) -> torch.Tensor:
        """Estado inicial del filtro cuando NO se conoce la historia del comando
        (rollout desde t=0): se asume asentado en el comando inicial. En t=0 la
        planta tambien arranca con P_lag = 0 = P(0), asi que es exacto ahi."""
        return torch.cat([P0, Q0], dim=-1)

    def filtered_inputs(self, P_seq: torch.Tensor, Q_seq: torch.Tensor,
                        dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Recorre el filtro dP̂/dt=(P−P̂)/τ̂ sobre una secuencia COMPLETA de
        comandos (T, ...) con la solucion exacta del ZOH, diferenciable en τ̂.

        Para que sirve: el estado del actuador depende SOLO del comando (que se
        conoce entero) y de τ̂ (aprendible). Entonces en multiple shooting no
        hace falta suponer nada al inicio de cada ventana: se computa el P̂
        verdadero-segun-el-modelo en cada instante y se usa como estado inicial.
        Sin esto, arrancar cada ventana con P̂=P(t0) mete un transitorio falso
        de ~τ̂ por ventana, que sesga τ̂ hacia abajo (medido: 0.56 vs 1.0)."""
        a = torch.exp(-dt / self.tau())
        Ph = [P_seq[0]]
        Qh = [Q_seq[0]]
        for t in range(1, P_seq.shape[0]):
            Ph.append(a * Ph[-1] + (1 - a) * P_seq[t - 1])
            Qh.append(a * Qh[-1] + (1 - a) * Q_seq[t - 1])
        return torch.stack(Ph), torch.stack(Qh)

    def forward(self, y: torch.Tensor, P, Q) -> torch.Tensor:
        x = y[..., 0:2]
        Ph = y[..., 2:3]
        Qh = y[..., 3:4]
        P = torch.as_tensor(P, dtype=y.dtype, device=y.device) * torch.ones_like(Ph)
        Q = torch.as_tensor(Q, dtype=y.dtype, device=y.device) * torch.ones_like(Qh)
        tau = self.tau()
        dPh = (P - Ph) / tau
        dQh = (Q - Qh) / tau
        dx = self.core.backbone(x, self._eff(Ph), self._eff(Qh))
        return torch.cat([dx, dPh, dQh], dim=-1)

    def params_dict(self) -> dict:
        return self.core.params_dict()

    def extras_dict(self) -> dict:
        alpha = float(torch.clamp(self.raw_alpha, 1e-3, 10.0))
        return {"tau": float(self.tau()), "sat": 1.0 / alpha, "alpha": alpha}


class LatentGrayBox(nn.Module):
    """WC + estados latentes genericos con dinamica aprendida.

    Estado aumentado y = [I, E, z]  (z de dimension n_hidden):
        ż = h_ψ(I, E, z, P, Q)                 <- memoria aprendida
        ẋ = f_WC(x, P, Q) + g_φ(I, E, z)       <- correccion que VE la memoria
    z parte de 0 en cada ventana; el modelo decide que guardar ahi.

    Las dos redes arrancan con salida 0: el modelo empieza siendo WC puro (igual
    que el gray-box sin memoria) y la memoria solo crece si el gradiente la pide.
    """

    def __init__(self, init_value: float = 1.0, n_hidden: int = 2,
                 hidden: int = 32):
        super().__init__()
        self.n_hidden = n_hidden
        init = {k: init_value for k in ALL_P}
        self.core = GrayBoxWC(init, {k: init_value for k in WEIGHTS},
                              learnable_weights=True, learnable_params=True,
                              use_correction=False)
        self.h = nn.Sequential(
            nn.Linear(2 + n_hidden + 2, hidden), nn.Tanh(),
            nn.Linear(hidden, n_hidden),
        )
        self.g = nn.Sequential(
            nn.Linear(2 + n_hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )
        for net in (self.h, self.g):
            for m in net:
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.bias)
            nn.init.zeros_(net[-1].weight)

    def h0(self, x0: torch.Tensor, P0: torch.Tensor, Q0: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*x0.shape[:-1], self.n_hidden,
                           dtype=x0.dtype, device=x0.device)

    def forward(self, y: torch.Tensor, P, Q) -> torch.Tensor:
        x = y[..., 0:2]
        z = y[..., 2:]
        I = y[..., 0:1]
        P = torch.as_tensor(P, dtype=y.dtype, device=y.device) * torch.ones_like(I)
        Q = torch.as_tensor(Q, dtype=y.dtype, device=y.device) * torch.ones_like(I)
        dz = self.h(torch.cat([x, z, P, Q], dim=-1))
        dx = self.core.backbone(x, P, Q) + self.g(torch.cat([x, z], dim=-1))
        return torch.cat([dx, dz], dim=-1)

    def params_dict(self) -> dict:
        return self.core.params_dict()

    def extras_dict(self) -> dict:
        return {"n_hidden": self.n_hidden}


# =============================================================================
#  ENTRENAMIENTO  (el analogo de graybox_train.fit para estados aumentados)
# =============================================================================

@dataclass
class AugTrainConfig:
    window: int = 100
    epochs: int = 1500
    lr_w: float = 5e-2
    lr_phys: float = 2e-2
    lr_extra: float = 2e-2      # τ̂, α (LagGrayBox)
    lr_g: float = 3e-3          # redes g y h (LatentGrayBox)
    seed: int = 0
    log_every: int = 250
    verbose: bool = True


def fit_aug(data: dict, model: nn.Module, cfg: AugTrainConfig) -> dict:
    """Entrena un modelo de estado aumentado con multiple shooting.

    Identico en espiritu a graybox_train.fit: ventanas de W pasos reiniciadas
    desde el dato observado. La diferencia es que el estado oculto NO se observa,
    asi que en cada ventana se inicializa con model.h0 (el filtro asentado en el
    comando, o z=0) y la perdida se evalua SOLO sobre (I,E).
    """
    torch.manual_seed(cfg.seed)

    x0, Pw, Qw, tgt = make_windows(data["I"], data["E"], data["P"], data["Q"], cfg.window)
    dt = data["dt"]

    # Estado oculto al inicio de cada ventana.
    #  - LagGrayBox: se computa EXACTO filtrando el comando completo de cada
    #    trayectoria con el τ̂ actual (depende de τ̂ -> se recalcula por epoca,
    #    dentro del grafo). Ver filtered_inputs.
    #  - LatentGrayBox: z=0 (no hay nada mejor sin observar z).
    W = cfg.window
    n, T = data["I"].shape
    nwin = (T - 1) // W
    P_full = torch.tensor(data["P"], dtype=torch.float32).T.unsqueeze(-1)  # (T,n,1)
    Q_full = torch.tensor(data["Q"], dtype=torch.float32).T.unsqueeze(-1)
    starts = [(s, w * W) for s in range(n) for w in range(nwin)]   # mismo orden que make_windows

    def hidden0():
        if hasattr(model, "filtered_inputs"):
            Ph, Qh = model.filtered_inputs(P_full, Q_full, dt)      # (T,n,1)
            h = torch.stack([torch.cat([Ph[a, s], Qh[a, s]]) for s, a in starts])
            return h
        return model.h0(x0, Pw[0, :, :], Qw[0, :, :])

    core = model.core
    phys_raw = [getattr(core, f"raw_{k}") for k in PHYS]
    groups = [{"params": [core.raw_w], "lr": cfg.lr_w},
              {"params": phys_raw, "lr": cfg.lr_phys}]
    net_params = [p for name in ("g", "h") if hasattr(model, name)
                  for p in getattr(model, name).parameters()]
    if net_params:
        groups.append({"params": net_params, "lr": cfg.lr_g})
    own = [p for n, p in model.named_parameters(recurse=False)]
    if own:
        groups.append({"params": own, "lr": cfg.lr_extra})
    opt = torch.optim.Adam(groups)

    hist = []
    for ep in range(cfg.epochs):
        opt.zero_grad()
        y0 = torch.cat([x0, hidden0()], dim=-1)
        pred = rollout(model, y0, Pw, Qw, dt)[..., :2]   # solo lo observable
        loss = ((pred - tgt) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in groups for p in g["params"]], 10.0)
        opt.step()
        if cfg.verbose and (ep % cfg.log_every == 0 or ep == cfg.epochs - 1):
            _, mx = param_errors(model, data["true"])
            extra = " ".join(f"{k}={v:.3f}" for k, v in model.extras_dict().items()
                             if isinstance(v, float))
            print(f"    ep {ep:5d} | data={float(loss.detach()):.3e} "
                  f"| err_max={mx:6.2f}% {extra}", flush=True)
        hist.append({"ep": ep, "data": float(loss.detach())})

    errs, mx = param_errors(model, data["true"])
    return {
        "params": model.params_dict(),
        "extras": model.extras_dict(),
        "param_errors": errs,
        "max_param_error": mx,
        "mean_param_error": float(np.mean(list(errs.values()))),
        "model": model,
        "hist": hist,
    }
