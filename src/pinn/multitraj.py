# =============================================================================
#  PINN MULTI-TRAYECTORIA  (Experimento 2)
# =============================================================================
#
#  Idea: usar VARIAS trayectorias del mismo sistema (mismo theta) pero con
#  estimulos / condiciones iniciales distintos, y un UNICO juego de pesos
#  theta COMPARTIDO que tiene que explicar TODAS a la vez.
#
#  Por que: en una sola trayectoria, wIE y wII se compensan en la entrada
#  inhibitoria u_i = wIE*E - wII*I (degeneracion) y no se pueden separar. Con
#  varias trayectorias que exciten E e I de formas distintas, esa compensacion
#  ya no funciona en todas a la vez -> el par queda determinado.
#
#  Como: una red por trayectoria (cada una mapea t -> [I,E] de SU grabacion) y
#  un solo theta compartido. Receta que funciona (heredada del paso 4.2):
#    1) WARMUP: cada red aprende su trayectoria solo con datos (fisica apagada).
#    2) CONGELAR todas las redes.
#    3) Ajustar SOLO theta (con lr propio y alto) minimizando la suma de los
#       residuos fisicos de todas las trayectorias.
# =============================================================================

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .network import PINN, inv_softplus
from .losses import data_loss, physics_loss, initial_condition_loss


class MultiTrajPINN(nn.Module):
    """N redes (una por trayectoria) + un theta (4 pesos) COMPARTIDO."""

    WEIGHT_NAMES = ("wEE", "wEI", "wIE", "wII")

    def __init__(
        self,
        n_traj: int,
        t_min: float,
        t_max: float,
        hidden_dim: int = 64,
        n_layers: int = 4,
        n_fourier: int = 128,
        fourier_scale: float = 6.0,
        w_init: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        # Una red t->[I,E] por trayectoria (identify=() -> sin pesos propios:
        # los pesos viven aparte, compartidos).
        self.nets = nn.ModuleList([
            PINN(hidden_dim=hidden_dim, n_layers=n_layers, t_min=t_min, t_max=t_max,
                 identify=(), n_fourier=n_fourier, fourier_scale=fourier_scale)
            for _ in range(n_traj)
        ])
        # theta compartido (crudo; el valor real es softplus(crudo) > 0).
        w_init = w_init or {k: 1.0 for k in self.WEIGHT_NAMES}
        self.raw_w = nn.Parameter(torch.stack([inv_softplus(w_init[k]) for k in self.WEIGHT_NAMES]))

    def weights(self) -> torch.Tensor:
        return F.softplus(self.raw_w)              # [wEE, wEI, wIE, wII] > 0

    def weights_dict(self) -> dict[str, float]:
        w = self.weights().detach()
        return {k: float(w[i]) for i, k in enumerate(self.WEIGHT_NAMES)}


class MultiTrajectoryTrainer:
    """Entrena N redes (warmup) y luego el theta compartido (congelando las redes)."""

    def __init__(self, model: MultiTrajPINN, config, fixed_params: dict) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.fixed = fixed_params
        # Dos grupos: las redes con su lr, el theta compartido con su lr propio.
        grupos = [{"params": list(self.model.nets.parameters()), "lr": config.lr},
                  {"params": [self.model.raw_w], "lr": config.lr_weights}]
        self.opt = torch.optim.Adam(grupos)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, factor=config.lr_factor, patience=config.lr_patience, min_lr=1e-5)

    # Junta los parametros fijos con el theta compartido (tensores).
    def _params(self) -> dict:
        w = self.model.weights()
        p = dict(self.fixed)
        for i, k in enumerate(self.model.WEIGHT_NAMES):
            p[k] = w[i]
        return p

    def _prepare(self, datasets: list[dict]):
        dev = self.config.device

        def col(x):
            return torch.tensor(np.asarray(x), dtype=torch.float32, device=dev).reshape(-1, 1)

        trajs = []
        for ds in datasets:
            t = col(ds["t"])
            target = torch.cat([col(ds["I"]), col(ds["E"])], dim=1)
            trajs.append({
                "t": t, "target": target, "P": col(ds["P"]), "Q": col(ds["Q"]),
                "t0": t[0:1], "ic": target[0:1], "n": t.shape[0],
            })
        return trajs

    def train(self, datasets: list[dict]) -> dict:
        cfg = self.config
        trajs = self._prepare(datasets)
        dev = cfg.device
        warmup = cfg.physics_warmup

        hist: dict[str, list] = {"loss": [], "data": [], "physics": []}
        for k in self.model.WEIGHT_NAMES:
            hist[k] = []

        ema, best_ema, sin_mejora = None, float("inf"), 0
        stop_epoch = cfg.epochs - 1

        for epoch in range(cfg.epochs):
            # Transicion fin del warmup: congelar redes, ajustar solo theta.
            if warmup > 0 and epoch == warmup:
                self.opt.param_groups[0]["lr"] = cfg.lr
                self.opt.param_groups[1]["lr"] = cfg.lr_weights
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.opt, factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=1e-5)
                ema, best_ema, sin_mejora = None, float("inf"), 0
                for p in self.model.nets.parameters():
                    p.requires_grad_(False)
                print(f"--> fin del warmup en epoch {epoch}: congelo las redes, ajusto solo theta.")

            en_warmup = epoch < warmup
            self.opt.zero_grad()

            if en_warmup:
                # FASE 1: cada red aprende su trayectoria (datos + condicion inicial).
                L_d, L_ic = 0.0, 0.0
                for i, tr in enumerate(trajs):
                    bi = torch.randperm(tr["n"], device=dev)[: cfg.batch_size]
                    L_d = L_d + data_loss(self.model.nets[i](tr["t"][bi]), tr["target"][bi])
                    L_ic = L_ic + initial_condition_loss(self.model.nets[i], tr["t0"], tr["ic"])
                L = cfg.w_data * L_d + cfg.w_ic * L_ic
                l_data, l_phys = L_d.item(), 0.0
            else:
                # FASE 2: theta compartido contra la suma de residuos fisicos.
                params = self._params()
                L_f = 0.0
                for i, tr in enumerate(trajs):
                    ci = torch.randperm(tr["n"], device=dev)[: cfg.n_collocation]
                    L_f = L_f + physics_loss(self.model.nets[i], tr["t"][ci], tr["P"][ci], tr["Q"][ci], params)
                L = cfg.w_physics * L_f
                l_data, l_phys = 0.0, L_f.item()

            L.backward()
            self.opt.step()

            l_total = L.item()
            hist["loss"].append(l_total)
            hist["data"].append(l_data)
            hist["physics"].append(l_phys)
            w = self.model.weights_dict()
            for k in self.model.WEIGHT_NAMES:
                hist[k].append(w[k])

            ema = l_total if ema is None else 0.99 * ema + 0.01 * l_total
            self.scheduler.step(ema)

            if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
                pesos = " ".join(f"{k}={w[k]:.3f}" for k in self.model.WEIGHT_NAMES)
                print(f"epoch {epoch:5d} | L={l_total:.3e} (ema={ema:.2e}) | {pesos}")
                if cfg.early_stop and epoch >= warmup:
                    if ema < best_ema * (1.0 - cfg.plateau_tol):
                        best_ema, sin_mejora = ema, 0
                    else:
                        sin_mejora += 1
                        if sin_mejora >= cfg.plateau_patience:
                            stop_epoch = epoch
                            print(f"--> meseta detectada en epoch {epoch}. Corto.")
                            break

        hist["stop_epoch"] = stop_epoch
        hist["warmup"] = warmup
        return hist

    @torch.no_grad()
    def predict(self, i: int, t_array: np.ndarray) -> np.ndarray:
        t = torch.tensor(np.asarray(t_array), dtype=torch.float32,
                         device=self.config.device).reshape(-1, 1)
        return self.model.nets[i](t).cpu().numpy()
