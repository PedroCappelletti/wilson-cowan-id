# =============================================================================
#  ENTRENAMIENTO DEL GRAY-BOX  (compartido por todos los experimentos)
# =============================================================================
#
#  Que resuelve: dado un dataset de trayectorias (I,E) forzadas por estimulos
#  (P,Q), ajustar el modelo  ẋ = f_WC(x,P,Q;θ) + g_φ(x[,P,Q])  para recuperar
#  los 10 parametros θ y, si esta activada, la correccion g_φ.
#
#  EL PROBLEMA CENTRAL que este archivo tiene que manejar:
#    g_φ es un aproximador universal sobre el mismo dominio que el backbone.
#    Entonces, para CUALQUIER θ equivocado existe un g que compensa el error y
#    reproduce los datos igual de bien. Con g libre, θ deja de ser identificable:
#    la red aprende a tapar parametros malos en vez de aprender fisica faltante.
#
#  Las cuatro variantes que se comparan:
#    A "libre"  : g(I,E,P,Q), sin regularizacion  -> muestra la patologia.
#    B "sin_pq" : g(I,E)                          -> el estimulo solo lo explica θ.
#    C "norma"  : g(I,E) + λ‖g‖²                  -> prior de "correccion minima".
#    D "ortogonal": g(I,E) + λ‖proyeccion de g sobre span{∂f/∂θ}‖²
#                   -> le PROHIBE a la red moverse en las direcciones que un
#                      cambio de parametros ya podria explicar.
#
#  La variante D es la principista: no penaliza que g sea grande, penaliza que
#  g sea REDUNDANTE con los parametros.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .dynamics import GrayBoxWC
from .integrate import rollout

WEIGHTS = ("wEE", "wEI", "wIE", "wII")
PHYS = ("te", "ti", "ae", "ai", "thetae", "thetai")
ALL_P = WEIGHTS + PHYS


# =============================================================================
#  SECCION 1: PREPARAR LOS DATOS
# =============================================================================

def make_windows(I, E, P, Q, W):
    """Parte las trayectorias en ventanas cortas (multiple shooting).

    Por que ventanas y no la trayectoria entera: integrar 4000 pasos de un tiron
    hace que el gradiente explote o se desvanezca. Reiniciando desde el estado
    observado cada W pasos el problema queda mucho mejor condicionado.

    Devuelve x0 (Nw,2), Pw (W,Nw,1), Qw (W,Nw,1), target (W+1,Nw,2).
    """
    n, T = I.shape
    nwin = (T - 1) // W
    x0, Pw, Qw, tgt = [], [], [], []
    for s in range(n):
        for w in range(nwin):
            a = w * W
            x0.append([I[s, a], E[s, a]])
            Pw.append(P[s, a:a + W])
            Qw.append(Q[s, a:a + W])
            tgt.append(np.stack([I[s, a:a + W + 1], E[s, a:a + W + 1]], axis=1))
    x0 = torch.tensor(np.asarray(x0), dtype=torch.float32)
    Pw = torch.tensor(np.asarray(Pw), dtype=torch.float32).T.unsqueeze(-1)
    Qw = torch.tensor(np.asarray(Qw), dtype=torch.float32).T.unsqueeze(-1)
    tgt = torch.tensor(np.asarray(tgt), dtype=torch.float32).permute(1, 0, 2)
    return x0, Pw, Qw, tgt


def sample_points(I, E, P, Q, n_max=3000, seed=0):
    """Muestra puntos sueltos (I,E,P,Q) del dataset. Se usan para evaluar las
    penalizaciones de g: se calculan sobre datos OBSERVADOS, no sobre la
    trayectoria predicha, para que no dependan del estado del entrenamiento."""
    rng = np.random.default_rng(seed)
    X = np.stack([I.ravel(), E.ravel()], axis=1)
    U = np.stack([P.ravel(), Q.ravel()], axis=1)
    n = len(X)
    idx = rng.choice(n, size=min(n_max, n), replace=False)
    return (torch.tensor(X[idx], dtype=torch.float32),
            torch.tensor(U[idx, 0:1], dtype=torch.float32),
            torch.tensor(U[idx, 1:2], dtype=torch.float32))


# =============================================================================
#  SECCION 2: LAS PENALIZACIONES
# =============================================================================

def projection_operator(S: torch.Tensor, rtol: float = 1e-10):
    """Dada la matriz de sensibilidades S (N,2,10), devuelve el operador A (10,2N)
    tal que  c = A @ vec(G)  son los coeficientes de la mejor aproximacion de una
    correccion G por un cambio de parametros.

    Idea: si existe un δθ tal que  S·δθ ≈ G  sobre todo el dataset, entonces esa
    parte de G es INDISTINGUIBLE de haber movido los parametros. Esa es
    exactamente la componente que hay que penalizar.

    OJO 1 — la proyeccion es en el ESPACIO DE FUNCIONES (G evaluada en los N puntos
    a la vez), no punto a punto. Punto a punto no serviria: 10 columnas en R^2
    generan todo R^2 y la penalizacion mataria g por completo.

    OJO 2 — POR QUE PSEUDOINVERSA Y NO RIDGE FIJO. S esta MUY mal condicionada (es
    el hecho central del proyecto: hay direcciones de parametros casi planas). Un
    ridge fijo del orden de 1e-8 domina a los autovalores mas chicos de SᵀS y
    borra de la proyeccion justamente las direcciones DEBILES. Y esas son las que
    importan: son las mas baratas de imitar para la red, o sea por donde se espera
    que robe identificabilidad. Con ridge fijo, la variante D dejaria de penalizar
    lo unico que tiene que penalizar.
    La pseudoinversa con tolerancia relativa conserva todas las direcciones con
    señal real y descarta solo las numericamente nulas (donde mover el parametro
    no hace nada, asi que no hay redundancia posible).

    Nota: la proyeccion sobre el espacio columna es invariante al reescalado de
    columnas, asi que no hace falta normalizar los parametros entre si.
    """
    N = S.shape[0]
    Sf = S.reshape(2 * N, -1)                       # (2N, 10); orden n*2+c
    A = torch.linalg.pinv(Sf, rtol=rtol)            # (10, 2N)
    return A, Sf


def projected_fraction(g_vals: torch.Tensor, A: torch.Tensor, Sf: torch.Tensor):
    """Parte de g que un cambio de parametros podria imitar.
    Devuelve (energia_proyectada, fraccion_del_total)."""
    Gv = g_vals.reshape(-1)                          # (2N,)
    c = A @ Gv                                       # coeficientes δθ equivalentes
    proj = Sf @ c                                    # componente redundante
    e_proj = (proj ** 2).mean()
    e_tot = (Gv ** 2).mean() + 1e-30
    return e_proj, (e_proj / e_tot).detach()


# =============================================================================
#  SECCION 3: METRICAS
# =============================================================================

@torch.no_grad()
def open_loop_mse(model, I, E, P, Q, dt):
    """Rollout COMPLETO de cada trayectoria (sin reinicios) -> MSE.
    Es el test de generalizacion real: si el modelo solo funciona en ventanas
    cortas, no sirve como planta."""
    n, T = I.shape
    errs = []
    for s in range(n):
        x0 = torch.tensor([[I[s, 0], E[s, 0]]], dtype=torch.float32)
        Ps = torch.tensor(P[s], dtype=torch.float32).reshape(T, 1, 1)
        Qs = torch.tensor(Q[s], dtype=torch.float32).reshape(T, 1, 1)
        traj = rollout(model, x0, Ps[:-1], Qs[:-1], dt)[:, 0, :]
        tgt = torch.tensor(np.stack([I[s], E[s]], axis=1), dtype=torch.float32)
        errs.append(float(((traj - tgt) ** 2).mean()))
    return float(np.mean(errs))


def param_errors(model, true: dict) -> tuple[dict, float]:
    """Error porcentual por parametro y el maximo."""
    p = model.params_dict()
    errs = {k: 100.0 * abs(p[k] - true[k]) / abs(true[k]) for k in true if k in p}
    return errs, max(errs.values())


# =============================================================================
#  SECCION 4: EL ENTRENAMIENTO
# =============================================================================

@dataclass
class TrainConfig:
    variant: str = "whitebox"     # whitebox | A | B | C | D
    window: int = 100
    epochs: int = 1500
    lbfgs_steps: int = 60
    lr_w: float = 5e-2
    lr_phys: float = 2e-2
    lr_g: float = 3e-3
    lam_norm: float = 0.0         # peso de  ‖g‖²          (variante C)
    lam_orth: float = 0.0         # peso de  ‖proj de g‖²  (variante D)
    init_value: float = 1.0       # arranque ignorante
    hidden: int = 32
    seed: int = 0
    sens_every: int = 25          # cada cuantas epocas se recalculan ∂f/∂θ
    log_every: int = 250
    verbose: bool = True


VARIANTS = {
    "whitebox": dict(use_correction=False, correction_inputs="xpq"),
    "A":        dict(use_correction=True,  correction_inputs="xpq"),
    "B":        dict(use_correction=True,  correction_inputs="x"),
    "C":        dict(use_correction=True,  correction_inputs="x"),
    "D":        dict(use_correction=True,  correction_inputs="x"),
    # S = correccion ESTRUCTURADA: 3 parametros fisicos (r_i, r_e, sat) en lugar
    # de una red. No es un termino aditivo sino una extension del propio
    # backbone, asi que el controlador puede invertirla exactamente.
    "S":        dict(use_correction=False, correction_inputs="x", structured=True),
}


def build_model(cfg: TrainConfig) -> GrayBoxWC:
    """Modelo con arranque IGNORANTE: todos los parametros parten de 1.0, sin
    ver jamas los valores verdaderos del dataset."""
    spec = VARIANTS[cfg.variant]
    init = {k: cfg.init_value for k in ALL_P}
    return GrayBoxWC(
        init, {k: cfg.init_value for k in WEIGHTS},
        learnable_weights=True, learnable_params=True,
        hidden=cfg.hidden, **spec,
    )


def fit(data: dict, cfg: TrainConfig, model: GrayBoxWC | None = None) -> dict:
    """Entrena una variante y devuelve el resultado + el historial.

    `data` necesita: I,E,P,Q (train), I_te,E_te,P_te,Q_te (test), dt, true.
    `model` opcional: permite arrancar de un modelo ya inicializado (warm-start)
    en vez del arranque ignorante de build_model.
    """
    torch.manual_seed(cfg.seed)
    if model is None:
        model = build_model(cfg)

    x0, Pw, Qw, tgt = make_windows(data["I"], data["E"], data["P"], data["Q"], cfg.window)
    Xs, Ps, Qs = sample_points(data["I"], data["E"], data["P"], data["Q"], seed=cfg.seed)
    dt = data["dt"]

    phys_raw = [getattr(model, f"raw_{k}") for k in PHYS]
    groups = [{"params": [model.raw_w], "lr": cfg.lr_w},
              {"params": phys_raw, "lr": cfg.lr_phys}]
    if model.use_correction:
        groups.append({"params": list(model.g.parameters()), "lr": cfg.lr_g})
    struct_raw = []
    if model.structured:
        struct_raw = [model.raw_r_i, model.raw_r_e, model.raw_alpha]
        groups.append({"params": struct_raw, "lr": cfg.lr_phys})
    opt = torch.optim.Adam(groups)

    A = Sf = None
    hist = []

    def penalties():
        """Devuelve (penalizacion_total, diagnosticos)."""
        if not model.use_correction:
            return torch.zeros((), dtype=torch.float32), {}
        gv = model.g_out(Xs, Ps, Qs)
        norm = (gv ** 2).mean()
        pen = cfg.lam_norm * norm
        frac = float("nan")
        if cfg.lam_orth > 0.0 and A is not None:
            e_proj, frac_t = projected_fraction(gv, A, Sf)
            pen = pen + cfg.lam_orth * e_proj
            frac = float(frac_t)
        # Se loguea la RAIZ cuadratica media (no la media cuadratica): es
        # directamente comparable con el |Delta f| verdadero, que en el punto
        # nominal vale ~0.017. Con la media cuadratica todo se veia como "0.000".
        return pen, {"g_rms": float(norm.detach()) ** 0.5, "redund": frac}

    for ep in range(cfg.epochs):
        # Las sensibilidades ∂f/∂θ dependen de θ, que va cambiando -> se
        # recalculan cada tanto (son baratas: 20 evaluaciones del backbone).
        if model.use_correction and cfg.lam_orth > 0.0 and ep % cfg.sens_every == 0:
            S = model.backbone_sensitivities(Xs, Ps, Qs)
            A, Sf = projection_operator(S)

        opt.zero_grad()
        pred = rollout(model, x0, Pw, Qw, dt)
        data_loss = ((pred - tgt) ** 2).mean()
        pen, diag = penalties()
        (data_loss + pen).backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in groups for p in g["params"]], 10.0)
        opt.step()

        if cfg.verbose and (ep % cfg.log_every == 0 or ep == cfg.epochs - 1):
            _, mx = param_errors(model, data["true"])
            extra = " ".join(f"{k}={v:.4f}" for k, v in diag.items())
            print(f"    ep {ep:5d} | data={float(data_loss.detach()):.3e} "
                  f"| err_max={mx:6.2f}% {extra}", flush=True)
        hist.append({"ep": ep, "data": float(data_loss.detach()),
                     "pen": float(pen.detach()) if torch.is_tensor(pen) else 0.0})

    # --- Refinamiento L-BFGS (solo sobre los parametros fisicos: es lo que
    #     mas se beneficia de un metodo de segundo orden).
    if cfg.lbfgs_steps > 0:
        params = [model.raw_w] + phys_raw + struct_raw
        opt2 = torch.optim.LBFGS(params, lr=1.0, max_iter=20,
                                 line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            loss = ((rollout(model, x0, Pw, Qw, dt) - tgt) ** 2).mean()
            loss.backward()
            return loss
        for _ in range(cfg.lbfgs_steps):
            opt2.step(closure)

    # --- Resultados
    errs, mx = param_errors(model, data["true"])
    out = {
        "variant": cfg.variant,
        "params": model.params_dict(),
        "param_errors": errs,
        "max_param_error": mx,
        "mean_param_error": float(np.mean(list(errs.values()))),
        "mse_train": open_loop_mse(model, data["I"], data["E"], data["P"], data["Q"], dt),
        "mse_test": open_loop_mse(model, data["I_te"], data["E_te"],
                                  data["P_te"], data["Q_te"], dt),
        "model": model,
        "hist": hist,
    }
    if model.structured:
        out["structured"] = model.structured_dict()
    if model.use_correction:
        with torch.no_grad():
            gv = model.g_out(Xs, Ps, Qs)
            fb = model.backbone(Xs, Ps, Qs)
        out["g_rms"] = float(gv.pow(2).mean().sqrt())
        out["g_rel"] = out["g_rms"] / float(fb.pow(2).mean().sqrt())
        S = model.backbone_sensitivities(Xs, Ps, Qs)
        Ao, Sfo = projection_operator(S)
        _, frac = projected_fraction(gv, Ao, Sfo)
        # Cuanto de la correccion aprendida es REDUNDANTE con los parametros:
        # 1.0 = todo lo que hace g lo podria haber hecho un cambio de θ.
        #
        # PERO EL PUNTO DE COMPARACION NO ES 0. Medido en eps=1: el Delta f
        # VERDADERO ya puntua 0.67, y una red con pesos aleatorios puntua 0.47
        # solo por ser una funcion suave del estado. Bajar de 0.67 no es
        # "aprender fisica mas pura": es alejarse de la fisica real.
        out["frac_redundante"] = float(frac)
    return out


# =============================================================================
#  SECCION 5: CARGA DEL DATASET
# =============================================================================

def load_split(path) -> dict:
    """Carga un .npz multi-escenario y lo parte en train/test segun is_test."""
    d = np.load(path, allow_pickle=True)
    is_test = d["is_test"].astype(bool)
    out = {
        "I": d["I"][~is_test], "E": d["E"][~is_test],
        "P": d["P"][~is_test], "Q": d["Q"][~is_test],
        "I_te": d["I"][is_test], "E_te": d["E"][is_test],
        "P_te": d["P"][is_test], "Q_te": d["Q"][is_test],
        "dt": float(d["dt"]),
        "true": {k: float(d[k]) for k in ALL_P},
        "labels": d["labels"], "is_test": is_test,
        "raw": d,
    }
    return out
