"""Tests del Neural ODE: identificacion completa (learnable_params) y utilidades."""

import numpy as np
import torch

from src.wilson_cowan import WilsonCowanParams
from src.neural_ode import GrayBoxWC, rollout


def _fixed_true():
    p = WilsonCowanParams()
    return {"te": p.te, "ti": p.ti, "ae": p.ae, "ai": p.ai,
            "thetae": p.thetae, "thetai": p.thetai, "ke": p.ke, "ki": p.ki}


def _w_true():
    p = WilsonCowanParams()
    return {"wEE": p.wEE, "wEI": p.wEI, "wIE": p.wIE, "wII": p.wII}


def test_learnable_params_exposes_ten_params():
    """Con learnable_params=True hay 6 params fisicos entrenables + los pesos."""
    init = {**_fixed_true(), **_w_true()}
    m = GrayBoxWC(init, _w_true(), learnable_weights=True, learnable_params=True)
    names = {n for n, _ in m.named_parameters()}
    for k in ("te", "ti", "ae", "ai", "thetae", "thetai"):
        assert f"raw_{k}" in names
    assert "raw_w" in names
    assert set(m.params_dict()) == set(("wEE", "wEI", "wIE", "wII",
                                        "te", "ti", "ae", "ai", "thetae", "thetai"))


def test_params_dict_recovers_init_values():
    """params_dict() debe devolver los valores con que se inicializo (softplus inverso)."""
    init = {**_fixed_true(), **_w_true()}
    m = GrayBoxWC(init, _w_true(), learnable_weights=True, learnable_params=True)
    p = m.params_dict()
    for k, v in {**_fixed_true(), **_w_true()}.items():
        if k in p:
            assert abs(p[k] - v) < 1e-4


def test_forward_matches_whitebox_at_true_params():
    """Inicializado en los verdaderos, el forward con learnable_params=True coincide
    con el modo de parametros fijos (no cambia el nucleo de la dinamica)."""
    fixed, w = _fixed_true(), _w_true()
    m_fixed = GrayBoxWC(fixed, w, learnable_weights=False, learnable_params=False)
    m_learn = GrayBoxWC({**fixed, **w}, w, learnable_weights=True, learnable_params=True)

    x = torch.tensor([[0.3, 0.5], [0.1, 0.2]])
    P = torch.tensor([[0.8], [0.2]])
    Q = torch.tensor([[0.6], [0.1]])
    with torch.no_grad():
        d_fixed = m_fixed(x, P, Q)
        d_learn = m_learn(x, P, Q)
    assert torch.allclose(d_fixed, d_learn, atol=1e-6)


def test_rest_state_is_equilibrium_with_derived_ke_ki():
    """ke,ki recalculados deben mantener E=I=0 como equilibrio (sin estimulo)."""
    init = {**_fixed_true(), **_w_true()}
    m = GrayBoxWC(init, _w_true(), learnable_weights=True, learnable_params=True)
    x0 = torch.zeros(1, 2)
    z = torch.zeros(1, 1)
    with torch.no_grad():
        d = m(x0, z, z)
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-6)


def _make_target(T=300, dt=0.05):
    """Trayectoria verdadera limpia bajo un estimulo rico (escalones) + sus ventanas."""
    rng = np.random.default_rng(0)
    Pseq = torch.tensor(np.repeat(rng.uniform(0, 1.2, T // 20), 20)[:T],
                        dtype=torch.float32).reshape(T, 1, 1)
    Qseq = torch.tensor(np.repeat(rng.uniform(0, 4.0, T // 20), 20)[:T],
                        dtype=torch.float32).reshape(T, 1, 1)
    true_model = GrayBoxWC(_fixed_true(), _w_true(),
                           learnable_weights=False, learnable_params=False)
    with torch.no_grad():
        traj = rollout(true_model, torch.zeros(1, 2), Pseq[:-1], Qseq[:-1], dt)
    W = 100
    x0s, Pw, Qw, tgt = [], [], [], []
    for a in range(0, T - W, W):
        x0s.append(traj[a, 0])
        Pw.append(Pseq[a:a + W, 0]); Qw.append(Qseq[a:a + W, 0])
        tgt.append(traj[a:a + W + 1, 0])
    return (torch.stack(x0s), torch.stack(Pw, dim=1), torch.stack(Qw, dim=1),
            torch.stack(tgt, dim=1), dt)


def test_weight_identification_recovers_weights():
    """Caso BIEN condicionado: identificar solo los 4 pesos (params fisicos conocidos)
    desde arranque ignorante (1.0) sobre datos limpios DEBE recuperar los pesos."""
    torch.manual_seed(0)
    x0s, Pw, Qw, tgt, dt = _make_target()
    w_true = _w_true()
    m = GrayBoxWC(_fixed_true(), {k: 1.0 for k in w_true},
                  learnable_weights=True, learnable_params=False)

    def max_err():
        p = m.weights_dict()
        return max(100 * abs(p[k] - w_true[k]) / w_true[k] for k in w_true)

    err0 = max_err()
    opt = torch.optim.Adam([m.raw_w], lr=5e-2)
    for _ in range(400):
        opt.zero_grad()
        loss = ((rollout(m, x0s, Pw, Qw, dt) - tgt) ** 2).mean()
        loss.backward(); opt.step()
    assert max_err() < err0           # mejora respecto al arranque ignorante (84%)
    assert max_err() < 25.0           # y converge razonablemente cerca


def test_learnable_params_loss_decreases_and_grads_flow():
    """Maquinaria de learnable_params: el ajuste baja la perdida y TODOS los 10 params
    reciben gradiente (no se mide recuperacion: con una sola trayectoria el problema de
    10 params es mal condicionado -> ver el barrido FIM/ruido)."""
    torch.manual_seed(0)
    x0s, Pw, Qw, tgt, dt = _make_target()
    keys = ("wEE", "wEI", "wIE", "wII", "te", "ti", "ae", "ai", "thetae", "thetai")
    m = GrayBoxWC({k: 1.0 for k in keys}, {k: 1.0 for k in ("wEE", "wEI", "wIE", "wII")},
                  learnable_weights=True, learnable_params=True)
    phys = [getattr(m, f"raw_{k}") for k in ("te", "ti", "ae", "ai", "thetae", "thetai")]
    opt = torch.optim.Adam([{"params": [m.raw_w], "lr": 5e-2},
                            {"params": phys, "lr": 2e-2}])
    with torch.no_grad():
        loss0 = float(((rollout(m, x0s, Pw, Qw, dt) - tgt) ** 2).mean())
    for _ in range(150):
        opt.zero_grad()
        loss = ((rollout(m, x0s, Pw, Qw, dt) - tgt) ** 2).mean()
        loss.backward(); opt.step()
    assert float(loss.detach()) < loss0
    assert m.raw_w.grad is not None and torch.isfinite(m.raw_w.grad).all()
    for r in phys:
        assert r.grad is not None and torch.isfinite(r.grad).all()


def test_closed_loop_tracks_with_true_params():
    """Regresion del lazo cerrado (convencion de unidades: ms en toda la cadena).
    Con los parametros reales, el IMC debe SEGUIR las referencias theta-gamma (RMSE
    chico y acotado). Guarda que la reconciliacion de unidades no rompio el control."""
    from src.neural_ode import (
        IMCController, make_true_plant, simulate_closed_loop, theta_gamma_refs,
    )
    fixed, w = _fixed_true(), _w_true()
    plant = make_true_plant(fixed, w)
    ctrl = IMCController(fixed, w)
    refs = theta_gamma_refs(freq_hz=120.0, time_in_ms=True)
    sol = simulate_closed_loop(plant, ctrl, refs, t_span=(0.0, 50.0), dt=0.005)
    n0 = len(sol["t"]) // 5
    rmse_E = float(np.sqrt(np.mean((sol["E"][n0:] - sol["rE"][n0:]) ** 2)))
    assert np.isfinite(rmse_E)
    assert rmse_E < 0.1            # sigue la referencia (valor historico ~3.1e-2)


def test_smooth_reduces_noise_variance():
    """El suavizado (promedio movil) reduce la varianza del ruido de alta frecuencia."""
    from scripts.noise_improve import smooth
    rng = np.random.default_rng(0)
    clean = np.sin(np.linspace(0, 6 * np.pi, 500))[None, :]
    noisy = clean + rng.normal(0, 0.1, clean.shape)
    sm = smooth(noisy, 7)
    assert np.var(sm - clean) < np.var(noisy - clean)
