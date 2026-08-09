#!/usr/bin/env python3
"""
BANCO DE PRUEBAS: candidatos de incertidumbre dinamica sobre Wilson-Cowan.

Para cada familia de perturbacion y cada intensidad epsilon mide:
  D_traj   : deformacion relativa de la trayectoria vs la WC nominal
  D_abs    : deformacion absoluta (comparable con el sigma del ruido de observacion)
  D_field  : |Delta f| / |f_WC|  -> tamano del termino que g_phi deberia aprender
  R2_mem   : R^2 del MEJOR predictor SIN MEMORIA de Delta f a partir de (I,E,P,Q),
             estimado con kNN + bloqueo temporal. Es la cota superior de lo que
             g_phi(I,E,P,Q) puede capturar. R2~1 -> aprendible; R2 bajo -> hay
             memoria/estado oculto que la correccion memoryless NO puede representar.
  regimen  : rango de E y numero de cruces por la media (sigue oscilando?)
"""
from __future__ import annotations

import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_P(__file__).resolve().parent))
from pathlib import Path


import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit, roots_hermitenorm

from src.wilson_cowan import (
    WilsonCowanParams, aprbs_pulse, theta_gamma_pulse, square_wave_pulse,
)

P0 = WilsonCowanParams()
KE, KI = P0.ke, P0.ki

DT = 0.02          # ms
T_END = 200.0
N = int(T_END / DT)
HZ = lambda f: f / 1000.0


# ---------------------------------------------------------------------------
#  Campo nominal de Wilson-Cowan (el backbone del GrayBoxWC)
# ---------------------------------------------------------------------------
def f_wc(I, E, P, Q):
    u_i = P0.wIE * E - P0.wII * I + Q - P0.thetai
    u_e = P0.wEE * E - P0.wEI * I + P - P0.thetae
    dI = (1.0 / P0.ti) * (-I + expit(P0.ai * u_i) - KI)
    dE = (1.0 / P0.te) * (-E + expit(P0.ae * u_e) - KE)
    return dI, dE


# ---------------------------------------------------------------------------
#  Familias de perturbacion.
#  Cada una devuelve (n_extra, rhs) donde rhs(t, s, P, Q, hist, k) -> ds
#  s = [I, E, *extras]. 'hist' es la trayectoria ya calculada (para retardos).
# ---------------------------------------------------------------------------
def make_nominal(eps):
    def rhs(t, s, P, Q, hist, k):
        dI, dE = f_wc(s[0], s[1], P, Q)
        return np.array([dI, dE])
    return 0, rhs


def make_refract(eps):
    """Refractariedad de Wilson-Cowan 1972: factor (1 - r*x) sobre la sigmoidea.
    Es el termino que la forma reducida (la que usamos) TIRA. eps = r."""
    def rhs(t, s, P, Q, hist, k):
        I, E = s[0], s[1]
        u_i = P0.wIE * E - P0.wII * I + Q - P0.thetai
        u_e = P0.wEE * E - P0.wEI * I + P - P0.thetae
        dI = (1.0 / P0.ti) * (-I + (1.0 - eps * I) * expit(P0.ai * u_i) - KI)
        dE = (1.0 / P0.te) * (-E + (1.0 - eps * E) * expit(P0.ae * u_e) - KE)
        return np.array([dI, dE])
    return 0, rhs


def make_adapt(eps, tau_a=20.0):
    """Adaptacion por frecuencia de disparo (SFA): estado lento A que resta
    corriente a E. eps = b (fuerza de la adaptacion)."""
    def rhs(t, s, P, Q, hist, k):
        I, E, A = s[0], s[1], s[2]
        u_i = P0.wIE * E - P0.wII * I + Q - P0.thetai
        u_e = P0.wEE * E - P0.wEI * I + P - P0.thetae - eps * A
        dI = (1.0 / P0.ti) * (-I + expit(P0.ai * u_i) - KI)
        dE = (1.0 / P0.te) * (-E + expit(P0.ae * u_e) - KE)
        dA = (E - A) / tau_a
        return np.array([dI, dE, dA])
    return 1, rhs


def make_depress(eps, tau_d=30.0):
    """Depresion sinaptica de corto plazo (Tsodyks-Markram): el peso wEE y wIE
    se debilitan con el uso. eps = U (fraccion de recurso consumida)."""
    def rhs(t, s, P, Q, hist, k):
        I, E, D = s[0], s[1], s[2]
        u_i = P0.wIE * D * E - P0.wII * I + Q - P0.thetai
        u_e = P0.wEE * D * E - P0.wEI * I + P - P0.thetae
        dI = (1.0 / P0.ti) * (-I + expit(P0.ai * u_i) - KI)
        dE = (1.0 / P0.te) * (-E + expit(P0.ae * u_e) - KE)
        dD = (1.0 - D) / tau_d - eps * D * E
        return np.array([dI, dE, dD])
    return 1, rhs, np.array([1.0])   # D arranca en 1 (recursos llenos)


def make_delay(eps):
    """Retardo axonal en las conexiones cruzadas E->I e I->E. eps = tau [ms]."""
    lag = eps

    def rhs(t, s, P, Q, hist, k):
        I, E = s[0], s[1]
        # historia: hist[k] es el estado en el paso k (t = k*DT)
        kd = k - int(round(lag / DT))
        if kd < 0:
            Ed, Id = 0.0, 0.0
        else:
            Id, Ed = hist[kd, 0], hist[kd, 1]
        u_i = P0.wIE * Ed - P0.wII * I + Q - P0.thetai
        u_e = P0.wEE * E - P0.wEI * Id + P - P0.thetae
        dI = (1.0 / P0.ti) * (-I + expit(P0.ai * u_i) - KI)
        dE = (1.0 / P0.te) * (-E + expit(P0.ae * u_e) - KE)
        return np.array([dI, dE])
    return 0, rhs


# Nodos/pesos de Gauss-Hermite para promediar la sigmoidea sobre umbrales dispersos.
_GH_X, _GH_W = roots_hermitenorm(21)
_GH_W = _GH_W / _GH_W.sum()


def _sig_hetero(u, a, spread):
    """Sigmoidea efectiva de una poblacion con umbrales gaussianos dispersos:
    E_delta[ sigma(a*(u - delta)) ], delta ~ N(0, spread^2). Cambia la FORMA
    de la curva (mas suave, colas mas gordas), no solo su ganancia."""
    return float(np.sum(_GH_W * expit(a * (u - spread * _GH_X))))


def make_shape(eps):
    """Mismatch de la no linealidad estatica: la poblacion real no es una
    logistica pura sino la mezcla sobre umbrales heterogeneos. eps = dispersion."""
    ke = _sig_hetero(-P0.thetae, P0.ae, eps)
    ki = _sig_hetero(-P0.thetai, P0.ai, eps)

    def rhs(t, s, P, Q, hist, k):
        I, E = s[0], s[1]
        u_i = P0.wIE * E - P0.wII * I + Q - P0.thetai
        u_e = P0.wEE * E - P0.wEI * I + P - P0.thetae
        dI = (1.0 / P0.ti) * (-I + _sig_hetero(u_i, P0.ai, eps) - ki)
        dE = (1.0 / P0.te) * (-E + _sig_hetero(u_e, P0.ae, eps) - ke)
        return np.array([dI, dE])
    return 0, rhs


def make_actuator(eps, tau_act=1.0):
    """Canal de actuacion no ideal (optogenetica): lo que COMANDAS no es lo que
    LLEGA. Retardo de primer orden + saturacion blanda. eps = nivel de saturacion
    (mas chico = satura antes). Entra por las DOS entradas P y Q."""
    def rhs(t, s, P, Q, hist, k):
        I, E, Pl, Ql = s[0], s[1], s[2], s[3]
        Pe = eps * np.tanh(Pl / eps)
        Qe = eps * np.tanh(Ql / eps)
        u_i = P0.wIE * E - P0.wII * I + Qe - P0.thetai
        u_e = P0.wEE * E - P0.wEI * I + Pe - P0.thetae
        dI = (1.0 / P0.ti) * (-I + expit(P0.ai * u_i) - KI)
        dE = (1.0 / P0.te) * (-E + expit(P0.ae * u_e) - KE)
        return np.array([dI, dE, (P - Pl) / tau_act, (Q - Ql) / tau_act])
    return 2, rhs


def make_hidden(eps, tj=8.0, aj=1.0, thj=3.0, wJE=5.0):
    """Dinamica no modelada: una tercera poblacion (columna vecina / poblacion
    lenta) acoplada al nodo E. eps = fuerza del acoplamiento de vuelta."""
    kj = expit(-aj * thj)

    def rhs(t, s, P, Q, hist, k):
        I, E, J = s[0], s[1], s[2]
        u_i = P0.wIE * E - P0.wII * I + Q - P0.thetai
        u_e = P0.wEE * E - P0.wEI * I + P - P0.thetae - eps * J
        dI = (1.0 / P0.ti) * (-I + expit(P0.ai * u_i) - KI)
        dE = (1.0 / P0.te) * (-E + expit(P0.ae * u_e) - KE)
        dJ = (1.0 / tj) * (-J + expit(aj * (wJE * E - thj)) - kj)
        return np.array([dI, dE, dJ])
    return 1, rhs


def make_drift(eps, f_slow_hz=5.0):
    """Neuromodulacion: el peso wEE deriva lentamente en el tiempo. Es
    incertidumbre NO AUTONOMA (depende de t, no del estado). eps = amplitud rel."""
    fs = HZ(f_slow_hz)

    def rhs(t, s, P, Q, hist, k):
        I, E = s[0], s[1]
        wEE = P0.wEE * (1.0 + eps * np.sin(2 * np.pi * fs * t))
        u_i = P0.wIE * E - P0.wII * I + Q - P0.thetai
        u_e = wEE * E - P0.wEI * I + P - P0.thetae
        dI = (1.0 / P0.ti) * (-I + expit(P0.ai * u_i) - KI)
        dE = (1.0 / P0.te) * (-E + expit(P0.ae * u_e) - KE)
        return np.array([dI, dE])
    return 0, rhs


def make_ou(eps, tau_n=1.0, seed=0):
    """Ruido de PROCESO (no de observacion): fluctuaciones de tamano finito de la
    poblacion, modeladas como Ornstein-Uhlenbeck sumado a las derivadas.
    eps = desvio estacionario. Es la parte IRREDUCIBLE (ningun g_phi la predice)."""
    rng = np.random.default_rng(seed)
    xi = np.zeros((N + 2, 2))
    for k in range(N + 1):
        xi[k + 1] = xi[k] + (-xi[k] / tau_n) * DT + eps * np.sqrt(2 * DT / tau_n) * rng.normal(size=2)

    def rhs(t, s, P, Q, hist, k):
        I, E = s[0], s[1]
        dI, dE = f_wc(I, E, P, Q)
        kk = min(k, N)
        return np.array([dI + xi[kk, 0], dE + xi[kk, 1]])
    return 0, rhs


# ---------------------------------------------------------------------------
#  Integrador RK4 de paso fijo con historia (necesario para el retardo).
# ---------------------------------------------------------------------------
def simulate(spec, Pf, Qf):
    if len(spec) == 3:
        n_extra, rhs, x_extra0 = spec
    else:
        n_extra, rhs = spec
        x_extra0 = np.zeros(n_extra)

    dim = 2 + n_extra
    hist = np.zeros((N + 1, dim))
    hist[0, 2:] = x_extra0
    Ps = np.array([Pf(k * DT) for k in range(N + 1)])
    Qs = np.array([Qf(k * DT) for k in range(N + 1)])

    for k in range(N):
        s = hist[k]
        t = k * DT
        P, Q = Ps[k], Qs[k]      # ZOH, igual que el rollout del entrenamiento
        k1 = rhs(t, s, P, Q, hist, k)
        k2 = rhs(t + 0.5 * DT, s + 0.5 * DT * k1, P, Q, hist, k)
        k3 = rhs(t + 0.5 * DT, s + 0.5 * DT * k2, P, Q, hist, k)
        k4 = rhs(t + DT, s + DT * k3, P, Q, hist, k)
        hist[k + 1] = s + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Delta f = campo real - campo WC nominal, evaluado sobre LA TRAYECTORIA REAL
    # y con el estimulo COMANDADO (que es lo unico que el modelo conoce).
    df = np.zeros((N + 1, 2))
    fw = np.zeros((N + 1, 2))
    for k in range(N + 1):
        s = hist[k]
        full = rhs(k * DT, s, Ps[k], Qs[k], hist, k)[:2]
        nom = np.array(f_wc(s[0], s[1], Ps[k], Qs[k]))
        fw[k] = nom
        df[k] = full - nom
    return hist[:, :2], df, fw, Ps, Qs


# ---------------------------------------------------------------------------
#  R^2 del mejor predictor SIN MEMORIA de Delta f desde (I,E,P,Q).
#  kNN con BLOQUEO TEMPORAL: se prohiben los vecinos temporalmente cercanos, si
#  no el vecino trivial (el paso siguiente) comparte el estado oculto y da R2~1
#  aunque la perturbacion tenga memoria.
# ---------------------------------------------------------------------------
def memoryless_r2(X, df, block=150, k=12, sub=6):
    Xs = X[::sub]
    ds = df[::sub]
    blk = max(block // sub, 1)
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-12
    Z = (Xs - mu) / sd
    tree = cKDTree(Z)
    nq = min(3000, len(Z))
    idx_q = np.linspace(0, len(Z) - 1, nq).astype(int)
    _, nb = tree.query(Z[idx_q], k=min(6 * k + blk * 2 + 1, len(Z)))
    pred = np.zeros((nq, 2))
    ok = np.zeros(nq, dtype=bool)
    for r, i in enumerate(idx_q):
        cand = [j for j in nb[r] if abs(int(j) - i) > blk][:k]
        if len(cand) >= 3:
            pred[r] = ds[cand].mean(0)
            ok[r] = True
    if ok.sum() < 50:
        return float("nan")
    tgt = ds[idx_q][ok]
    res = tgt - pred[ok]
    ss_tot = ((tgt - tgt.mean(0)) ** 2).sum()
    if ss_tot < 1e-30:
        return float("nan")
    return float(1.0 - (res ** 2).sum() / ss_tot)


# ---------------------------------------------------------------------------
#  Estimulos: los mismos del multi_dataset (regimen ms).
# ---------------------------------------------------------------------------
STIM = {
    "aprbs": (aprbs_pulse(1.0, 10, 190, 2, 8, seed=71, amp_min=0.2),
              aprbs_pulse(0.8, 10, 190, 2.6, 9.6, seed=81, amp_min=0.1)),
    "thetagamma": (theta_gamma_pulse(1.0, HZ(40), HZ(10), 10, 190, 0.5),
                   theta_gamma_pulse(0.7, HZ(36), HZ(10), 10, 190, 0.5)),
    "square": (square_wave_pulse(0.6, HZ(50), 10, 190, 0.4),
               square_wave_pulse(0.42, HZ(40), 10, 190, 0.5)),
}

FAMILIES = {
    "refract  (refractariedad WC-1972)": (make_refract, [0.2, 0.5, 1.0]),
    "shape    (sigmoidea heterogenea)":  (make_shape,   [0.3, 0.8, 1.5]),
    "adapt    (adaptacion SFA)":         (make_adapt,   [0.2, 0.5, 1.0]),
    "depress  (depresion sinaptica)":    (make_depress, [0.05, 0.15, 0.4]),
    "hidden   (poblacion no modelada)":  (make_hidden,  [0.3, 0.8, 1.5]),
    "delay    (retardo axonal ms)":      (make_delay,   [0.3, 1.0, 2.0]),
    "actuator (canal opto no ideal)":    (make_actuator,[1.5, 0.8, 0.4]),
    "drift    (deriva de wEE en t)":     (make_drift,   [0.05, 0.15, 0.30]),
    "ou_noise (ruido de proceso)":       (make_ou,      [0.005, 0.02, 0.05]),
}


def main():
    # Referencia nominal por estimulo
    nom = {}
    for name, (Pf, Qf) in STIM.items():
        x, _, fw, Ps, Qs = simulate(make_nominal(0), Pf, Qf)
        nom[name] = (x, fw, Ps, Qs)
        print(f"[nominal:{name:11}] E in [{x[:,1].min():.3f},{x[:,1].max():.3f}]  "
              f"I in [{x[:,0].min():.3f},{x[:,0].max():.3f}]  "
              f"RMS|f_WC| = {np.sqrt((fw**2).sum(1)).mean():.4f}")

    print()
    hdr = (f"{'familia':36} {'eps':>6} {'D_traj%':>8} {'D_abs':>8} "
           f"{'D_field%':>9} {'R2_mem':>7} {'E_max':>6} {'osc':>5}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for fam, (maker, eps_list) in FAMILIES.items():
        for eps in eps_list:
            Dt, Da, Df, R2, Emax, osc = [], [], [], [], [], []
            Xall, dfall = [], []
            for sname, (Pf, Qf) in STIM.items():
                xn, fwn, Ps, Qs = nom[sname]
                spec = maker(eps)
                xp, dfp, fwp, _, _ = simulate(spec, Pf, Qf)
                d = xp - xn
                sig = xn - xn.mean(0)
                Dt.append(100 * np.sqrt((d ** 2).mean()) / np.sqrt((sig ** 2).mean()))
                Da.append(np.sqrt((d ** 2).mean()))
                Df.append(100 * np.sqrt((dfp ** 2).sum(1)).mean()
                          / np.sqrt((fwp ** 2).sum(1)).mean())
                Emax.append(xp[:, 1].max())
                e = xp[:, 1] - xp[:, 1].mean()
                osc.append(int(np.sum(np.diff(np.sign(e)) != 0)))
                Xall.append(np.column_stack([xp[:, 0], xp[:, 1], Ps, Qs]))
                dfall.append(dfp)
            R2 = memoryless_r2(np.vstack(Xall), np.vstack(dfall))
            print(f"{fam:36} {eps:6.3g} {np.mean(Dt):8.2f} {np.mean(Da):8.4f} "
                  f"{np.mean(Df):9.2f} {R2:7.3f} {np.mean(Emax):6.3f} {int(np.mean(osc)):5d}")
            rows.append((fam, eps, np.mean(Dt), np.mean(Da), np.mean(Df), R2))
        print()

    np.save("/tmp/claude-1000/-home-pedro-Documentos-investigacion/"
            "51ae4f64-36e9-4d36-81d7-dd17b77f5087/scratchpad/probe_rows.npy",
            np.array(rows, dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()
