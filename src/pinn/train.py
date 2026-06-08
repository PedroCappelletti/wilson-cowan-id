# =============================================================================
#  ENTRENAMIENTO DE LA PINN
# =============================================================================
#
#  Ajusta, con UN SOLO optimizador y a la vez:
#    - los pesos internos de la red (para reproducir la trayectoria), y
#    - los pesos de Wilson-Cowan que se esten identificando (si los hay).
#  Minimiza  L = w_data*L_datos + w_physics*L_fisica + w_ic*L_inicial.
#
#  VALIDACION TEMPORAL (extrapolacion):
#    Entrenamos con DATOS solo hasta cierto tiempo (primeros (1 - val_fraction)
#    de la trayectoria) y evaluamos como predice el TRAMO FINAL que no vio.
#    La fisica (colocacion) si se aplica en TODO el dominio: es justamente lo
#    que permite "propagar" la solucion hacia el tramo sin datos. Si predice
#    bien ahi, aprendio la dinamica (no memorizo puntos).
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .losses import data_loss, physics_loss, initial_condition_loss


@dataclass
class TrainConfig:
    """Hiperparametros de entrenamiento."""

    epochs: int = 30_000        # tope maximo (la parada por meseta suele cortar antes)
    lr: float = 1e-3            # learning rate de la red
    lr_weights: float = 1e-3   # learning rate de los pesos a identificar (theta).
                               # Conviene mas alto que el de la red: theta arranca
                               # lejos y tiene que viajar mucho (ej. 1.0 -> 6.4).
    w_data: float = 1.0         # peso del termino de datos
    w_physics: float = 1.0      # peso del termino fisico (el "lambda")
    w_ic: float = 1.0           # peso de la condicion inicial
    batch_size: int = 8_000     # puntos de DATOS por paso (minibatch)
    n_collocation: int = 4_000  # puntos de FISICA por paso
    val_fraction: float = 0.2   # fraccion final reservada para test (no se entrena con sus datos)
    device: str = "cpu"
    log_every: int = 200
    checkpoint_dir: str = "results/models"
    # --- Curriculum "datos primero" (clave para el inverso desde arranque lejano) ---
    # Durante los primeros `physics_warmup` epochs se entrena SOLO con datos
    # (fisica apagada) para que la red aprenda la trayectoria verdadera. Recien
    # despues se prende la fisica, ya con la curva correcta: asi el residuo
    # empuja los pesos hacia su valor real en vez de trabarse en un minimo local.
    physics_warmup: int = 0
    # Al terminar el warmup, CONGELAR la red y ajustar SOLO los pesos de
    # Wilson-Cowan. Evita que la red "haga trampa" deformando la trayectoria
    # para satisfacer la fisica con pesos errados; obliga a que se muevan los pesos.
    freeze_net_after_warmup: bool = False
    # --- Parada automatica al amesetar la perdida (meseta) ---
    early_stop: bool = True
    plateau_tol: float = 1e-3   # mejora relativa minima para considerar "todavia baja"
    plateau_patience: int = 10  # cuantos chequeos seguidos sin mejora antes de cortar
    lr_factor: float = 0.5      # cuanto se reduce el lr cuando se estanca
    lr_patience: int = 6        # chequeos sin mejora antes de bajar el lr


class Trainer:
    """Orquesta el entrenamiento de la PINN sobre UNA trayectoria."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainConfig,
        fixed_params: dict,     # te,ti,ae,ai,thetae,thetai,ke,ki + pesos NO identificados
    ) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.fixed = fixed_params
        # Un solo optimizador, pero con DOS grupos: la red con su lr y los pesos
        # a identificar (raw_w) con un lr propio (normalmente mas alto). Asi theta
        # puede moverse rapido sin desestabilizar a la red.
        grupos = [{"params": list(self.model.net.parameters()), "lr": config.lr}]
        if self.model.raw_w is not None:
            grupos.append({"params": [self.model.raw_w], "lr": config.lr_weights})
        self.opt = torch.optim.Adam(grupos)
        # Reduce el lr cuando la perdida (suavizada) se estanca, para cerrar mejor.
        # min_lr evita que colapse a cero y congele el entrenamiento.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, factor=config.lr_factor, patience=config.lr_patience, min_lr=1e-5,
        )

    # -------------------------------------------------------------------------
    #  Junta los parametros fijos con los identificados por la red (tensores).
    # -------------------------------------------------------------------------
    def _params(self) -> dict:
        return {**self.fixed, **self.model.identified_weights()}

    # -------------------------------------------------------------------------
    #  Pasa el dataset a tensores y separa train (80% inicial) / test (20% final).
    # -------------------------------------------------------------------------
    def _prepare(self, dataset: dict[str, np.ndarray]):
        dev = self.config.device

        def col(x):
            return torch.tensor(np.asarray(x), dtype=torch.float32, device=dev).reshape(-1, 1)

        t = col(dataset["t"])
        target = torch.cat([col(dataset["I"]), col(dataset["E"])], dim=1)   # (N, 2)
        P = col(dataset["P"])
        Q = col(dataset["Q"])

        # Corte temporal: train = primeros (1 - val_fraction); test = resto.
        n = t.shape[0]
        n_train = int(round((1.0 - self.config.val_fraction) * n))
        train_idx = torch.arange(0, n_train, device=dev)
        test_idx = torch.arange(n_train, n, device=dev)

        t0 = t[0:1]            # tiempo inicial
        ic = target[0:1]       # condicion inicial [I0, E0]
        return t, target, P, Q, t0, ic, train_idx, test_idx

    # -------------------------------------------------------------------------
    #  Bucle de entrenamiento.
    # -------------------------------------------------------------------------
    def train(self, dataset: dict[str, np.ndarray]) -> dict:
        cfg = self.config
        t, target, P, Q, t0, ic, train_idx, test_idx = self._prepare(dataset)
        dev = cfg.device
        n_train = train_idx.shape[0]

        hist: dict[str, list] = {"loss": [], "data": [], "physics": [], "ic": [], "val": []}
        for k in self.model.identify:
            hist[k] = []

        # Para detectar la meseta usamos una media movil (EMA) de la perdida:
        # asi ignoramos el ruido del minibatch y miramos la tendencia.
        ema = None
        best_ema = float("inf")
        sin_mejora = 0          # chequeos seguidos sin mejora apreciable
        stop_epoch = cfg.epochs - 1

        warmup = cfg.physics_warmup
        for epoch in range(cfg.epochs):
            # --- Transicion fin del warmup: se prende la fisica. Reiniciamos el
            #     lr y la deteccion de meseta, porque la perdida da un salto.
            if warmup > 0 and epoch == warmup:
                self.opt.param_groups[0]["lr"] = cfg.lr              # red
                if len(self.opt.param_groups) > 1:
                    self.opt.param_groups[1]["lr"] = cfg.lr_weights  # pesos theta
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.opt, factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=1e-5)
                ema, best_ema, sin_mejora = None, float("inf"), 0
                print(f"--> fin del warmup (datos solos) en epoch {epoch}: activo la fisica.")
                if cfg.freeze_net_after_warmup:
                    for p in self.model.net.parameters():
                        p.requires_grad_(False)
                    print("    red congelada: ahora solo se ajustan los pesos de Wilson-Cowan.")

            en_warmup = epoch < warmup
            wf = 0.0 if en_warmup else cfg.w_physics   # fisica apagada durante el warmup

            self.opt.zero_grad()

            # --- L_datos: minibatch del tramo de ENTRENAMIENTO (primeros 80%).
            if cfg.batch_size < n_train:
                bi = train_idx[torch.randperm(n_train, device=dev)[: cfg.batch_size]]
            else:
                bi = train_idx
            L_d = data_loss(self.model(t[bi]), target[bi])

            # --- L_fisica: minibatch de TODO el dominio (incluye el tramo de test).
            #     Esto es lo que permite extrapolar hacia el tramo sin datos.
            ci = torch.randperm(t.shape[0], device=dev)[: cfg.n_collocation]
            L_f = physics_loss(self.model, t[ci], P[ci], Q[ci], self._params())

            # --- L_inicial.
            L_ic = initial_condition_loss(self.model, t0, ic)

            L = cfg.w_data * L_d + wf * L_f + cfg.w_ic * L_ic
            L.backward()
            self.opt.step()

            # --- Registro (incluye error de validacion en el tramo no visto).
            l_total, l_d, l_f, l_ic = L.item(), L_d.item(), L_f.item(), L_ic.item()
            hist["loss"].append(l_total)
            hist["data"].append(l_d)
            hist["physics"].append(l_f)
            hist["ic"].append(l_ic)
            w = self.model.weights_dict()
            for k in self.model.identify:
                hist[k].append(w[k])

            # Media movil de la perdida + ajuste del lr segun la tendencia.
            ema = l_total if ema is None else 0.99 * ema + 0.01 * l_total
            self.scheduler.step(ema)

            if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
                val = self._val_mse(t, target, test_idx)
                hist["val"].append((epoch, val))
                extra = ""
                if self.model.identify:
                    extra = " | " + " ".join(f"{k}={w[k]:.3f}" for k in self.model.identify)
                print(
                    f"epoch {epoch:5d} | L={l_total:.3e} (ema={ema:.2e}) "
                    f"(datos={l_d:.2e} fis={l_f:.2e} ic={l_ic:.2e}) "
                    f"| val_MSE={val:.3e}{extra}"
                )

                # --- Deteccion de meseta sobre la EMA (recien despues del warmup) ---
                if cfg.early_stop and epoch >= warmup:
                    if ema < best_ema * (1.0 - cfg.plateau_tol):
                        best_ema = ema          # hubo mejora apreciable
                        sin_mejora = 0
                    else:
                        sin_mejora += 1
                        if sin_mejora >= cfg.plateau_patience:
                            stop_epoch = epoch
                            print(f"--> meseta detectada en epoch {epoch} (la perdida ya casi no baja). Corto.")
                            break

        hist["stop_epoch"] = stop_epoch
        hist["warmup"] = warmup
        return hist

    # -------------------------------------------------------------------------
    #  Error en el tramo de test (datos que la red NO uso para entrenar).
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _val_mse(self, t, target, test_idx) -> float:
        if test_idx.numel() == 0:
            return float("nan")
        pred = self.model(t[test_idx])
        return float(((pred - target[test_idx]) ** 2).mean())

    # -------------------------------------------------------------------------
    #  Prediccion sobre tiempos dados (para graficar). Devuelve numpy (N, 2).
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, t_array: np.ndarray) -> np.ndarray:
        t = torch.tensor(np.asarray(t_array), dtype=torch.float32,
                         device=self.config.device).reshape(-1, 1)
        return self.model(t).cpu().numpy()

    # -------------------------------------------------------------------------
    #  Checkpoints.
    # -------------------------------------------------------------------------
    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": self.model.state_dict(),
             "opt_state": self.opt.state_dict(),
             "fixed": self.fixed},
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.opt.load_state_dict(ckpt["opt_state"])
        self.fixed = ckpt.get("fixed", self.fixed)
