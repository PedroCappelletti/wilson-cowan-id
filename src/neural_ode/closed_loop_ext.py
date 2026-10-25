# =============================================================================
#  CONTROLADOR EXTENDIDO: el IMC de Martínez et al. (2022) con el modelo entero
# =============================================================================
#
#  IMCController (closed_loop.py) despeja el comando con una cuenta cerrada
#  (ec. 6 del paper): invierte la sigmoidea y resta el acoplamiento. Eso supone
#  que P entra en un solo lugar y al instante. Los modelos con memoria rompen
#  alguno de los dos supuestos:
#
#    historia y filtro entrenable   P(t) entra también adentro de la red
#    estado de filtro               P pasa por el retardo del actuador
#
#  IMCExtendido conserva todo lo demás del original (el PI, la saturación de
#  u_lti, la inversa de la sigmoidea y los topes) y cambia solo el despeje:
#
#  - Historia y filtro. El objetivo es el mismo que el del original, que el
#    campo del modelo valga (1/τ̂)(−x + u_sat). En vez de una cuenta, se
#    resuelve f_modelo(x, P, Q, pasado) = objetivo con Newton en (P, Q),
#    arrancando del despeje cerrado del original. El pasado son los comandos
#    que el propio controlador ya mandó, muestreados al paso del modelo.
#  - Estado de filtro. El despeje cerrado da el P efectivo que hace falta. Se
#    invierte la saturación identificada y se compensa el retardo con una copia
#    interna del filtro: P = P̂ + (τ̂/τ_f)(P*_lag − P̂). Así P̂ sigue al valor
#    deseado con constante τ_f en vez de τ̂. τ_f es un parámetro de diseño.
#
#  Con la agnóstica g(I, E) y la forma exacta, el original ya cancela todo el
#  modelo, así que el extendido no les cambia nada.
# =============================================================================

from __future__ import annotations

import math

import numpy as np
import torch

from .closed_loop import IMCController


def _inv_sig(y, a, theta):
    return (-1.0 / a) * math.log(-1.0 + 1.0 / y) + theta


class _CampoNumpy:
    """El campo de un GrayBoxWC con historia o filtro, evaluado en numpy. Da lo
    mismo que el modelo de torch y, con lotes de 3 filas, es mucho más rápido:
    en el controlador se evalúa unas 1,2 millones de veces por simulación de 2 s."""

    def __init__(self, m):
        p = m.params_dict()
        self.w = np.array([p[k] for k in ("wEE", "wEI", "wIE", "wII")])
        self.te, self.ti, self.ae, self.ai = p["te"], p["ti"], p["ae"], p["ai"]
        self.the, self.thi = p["thetae"], p["thetai"]
        self.ke = 1.0 / (1.0 + math.exp(self.ae * self.the))
        self.ki = 1.0 / (1.0 + math.exp(self.ai * self.thi))
        g = m.g
        self.fir = None
        if m.correction_inputs == "xconv":
            self.fir = g.fir.weight.detach().numpy().astype(float).T   # (2K, n_fir)
            g = g.mlp
        capas = [c for c in g if hasattr(c, "weight")]
        self.W = [c.weight.detach().numpy().astype(float).T for c in capas]
        self.b = [c.bias.detach().numpy().astype(float) for c in capas]

    def __call__(self, x, Ph, Qh):
        I, E = x[:, 0], x[:, 1]
        P, Q = Ph[:, 0], Qh[:, 0]
        wEE, wEI, wIE, wII = self.w
        ui = wIE * E - wII * I + Q - self.thi
        ue = wEE * E - wEI * I + P - self.the
        dI = (-I + 1.0 / (1.0 + np.exp(-self.ai * ui)) - self.ki) / self.ti
        dE = (-E + 1.0 / (1.0 + np.exp(-self.ae * ue)) - self.ke) / self.te
        u = np.concatenate([Ph, Qh], axis=1)
        h = np.concatenate([x, u @ self.fir if self.fir is not None else u], axis=1)
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if k < len(self.W) - 1:
                h = np.tanh(h)
        return np.stack([dI, dE], axis=1) + h


class IMCExtendido(IMCController):
    """modelo: GrayBoxWC con historia o filtro, o LagGrayBox. dt_modelo: el paso
    de muestreo con que se entrenó (ms), que fija el espaciado de la historia."""

    def __init__(self, fixed, weights, modelo, tipo: str, dt_modelo: float,
                 dt_ctrl: float, tau_f: float = 0.2, newton_iter: int = 4, **kw):
        super().__init__(fixed, weights, **kw)
        self.modelo, self.tipo = modelo, tipo
        modelo.requires_grad_(False)   # solo se deriva respecto de (P, Q)
        self.tau_f = tau_f
        self.newton_iter = newton_iter
        self.n_fallas = 0
        if tipo == "historia":
            self.K = int(modelo.hist_len)
            self.paso = max(1, int(round(dt_modelo / dt_ctrl)))
            n = self.K * self.paso + 1
            self.bufP = np.zeros(n)          # comandos ya mandados, uno por actualización
            self.bufQ = np.zeros(n)
            self.idx = np.arange(1, self.K) * self.paso   # retardos 1..K-1
            self._campo = _CampoNumpy(modelo)
        if tipo == "lag":
            ex = modelo.extras_dict()
            self.tau_hat, self.alpha_hat = ex["tau"], ex["alpha"]

    # El controlador del estado de filtro lleva su copia del actuador (P̂, Q̂).
    @property
    def n_estados(self):
        return 2 if self.tipo == "lag" else 0

    def registrar(self, P, Q):
        """Se llama en cada actualización del controlador con el comando nuevo."""
        if self.tipo == "historia":
            self.bufP = np.roll(self.bufP, 1); self.bufP[0] = P
            self.bufQ = np.roll(self.bufQ, 1); self.bufQ[0] = Q

    def _objetivo(self, Z1, Z2, I, E):
        f = self.f
        Ulti_I = self.kp_I * (self._rI - I) + self.ki_I * Z1
        Ulti_E = self.kp_E * (self._rE - E) + self.ki_E * Z2
        Usat_I = min(max(Ulti_I, self.fim), self.fiM)
        Usat_E = min(max(Ulti_E, self.fem), self.feM)
        return (-I + Usat_I) / f["ti"], (-E + Usat_E) / f["te"]

    def _cotas(self, I, E):
        f, w = self.f, self.w
        up = (_inv_sig(self.lo_e, f["ae"], f["thetae"]), _inv_sig(self.hi_e, f["ae"], f["thetae"]))
        uq = (_inv_sig(self.lo_i, f["ai"], f["thetai"]), _inv_sig(self.hi_i, f["ai"], f["thetai"]))
        cP = w["wEE"] * E - w["wEI"] * I
        cQ = w["wIE"] * E - w["wII"] * I
        return (up[0] - cP, up[1] - cP), (uq[0] - cQ, uq[1] - cQ)

    def _newton(self, P0, Q0, I, E, dI_obj, dE_obj):
        # Newton con jacobiano por diferencias finitas: en cada iteración se
        # evalúan el punto y sus dos desplazamientos en un solo lote. Con
        # derivadas automáticas, una simulación de 50 ms tardaba más de 10 min;
        # así, con el campo en numpy, unos segundos.
        x = np.array([[I, E]] * 3)
        pasP = np.tile(self.bufP[self.idx], (3, 1))
        pasQ = np.tile(self.bufQ[self.idx], (3, 1))
        obj = np.array([dI_obj, dE_obj])
        (Plo, Phi), (Qlo, Qhi) = self._cotas(I, E)
        h = 1e-3

        def residuos(P, Q):
            Pc = np.array([[P], [P + h], [P]])
            Qc = np.array([[Q], [Q], [Q + h]])
            return self._campo(x, np.hstack([Pc, pasP]), np.hstack([Qc, pasQ])) - obj

        R = residuos(P0, Q0)
        r0 = float(np.abs(R[0]).max())
        # Arranca del despeje cerrado o de la solución del paso anterior, la mejor.
        v = np.array([P0, Q0])
        if getattr(self, "_ult", None) is not None:
            Ru = residuos(*self._ult)
            if np.abs(Ru[0]).max() < r0:
                v, R = np.array(self._ult), Ru
        for _ in range(self.newton_iter):
            r = R[0]
            if np.abs(r).max() < 1e-6:
                break
            J = np.stack([(R[1] - r) / h, (R[2] - r) / h], axis=1)
            try:
                d = np.linalg.solve(J, r)
            except np.linalg.LinAlgError:
                break
            v = v - d
            v[0] = min(max(v[0], Plo), Phi)
            v[1] = min(max(v[1], Qlo), Qhi)
            R = residuos(*v)
        r = float(np.abs(R[0]).max())
        if not np.isfinite(r) or r > max(1e-4, r0):
            # Sin una solución mejor que el despeje cerrado dentro de los topes:
            # se cuenta y se usa el despeje cerrado.
            self.n_fallas += 1
            self._ult = None
            return P0, Q0
        self._ult = (float(v[0]), float(v[1]))
        return self._ult

    def compute_ext(self, zs, I, E, rI, rE):
        """zs = [Z1, Z2, *estados del controlador]. Devuelve (P, Q, dzs)."""
        Z1, Z2 = zs[0], zs[1]
        self._rI, self._rE = rI, rE
        P, Q, dZ1, dZ2 = self.compute(Z1, Z2, I, E, rI, rE)   # despeje cerrado del original
        dz = [dZ1, dZ2]
        if self.tipo == "historia":
            with torch.enable_grad():
                dI_obj, dE_obj = self._objetivo(Z1, Z2, I, E)
                P, Q = self._newton(P, Q, I, E, dI_obj, dE_obj)
        elif self.tipo == "lag":
            Ph, Qh = zs[2], zs[3]
            a = self.alpha_hat
            # inversa de la saturación identificada, acotada al dominio de atanh
            lim = 0.999 / a
            Pl = math.atanh(a * min(max(P, -lim), lim)) / a
            Ql = math.atanh(a * min(max(Q, -lim), lim)) / a
            # compensación del retardo con la copia interna del filtro
            g = self.tau_hat / self.tau_f
            P = Ph + g * (Pl - Ph)
            Q = Qh + g * (Ql - Qh)
            dz += [(P - Ph) / self.tau_hat, (Q - Qh) / self.tau_hat]
        return P, Q, dz


def simulate_closed_loop_ext(plant_rhs, controller, refs, t_span=(0.0, 50.0),
                             dt=0.005, dt_ctrl=0.05, x0=(0.0, 0.0),
                             tope=None, antiwindup: bool = False):
    """Lazo cerrado con un controlador DIGITAL: el comando se calcula cada
    dt_ctrl con el estado medido en ese instante y se mantiene constante hasta
    la actualización siguiente. La planta se integra con RK4 a paso dt.

    Sirve igual para IMCController y para IMCExtendido, y todas las
    comparaciones usan el mismo dt_ctrl. Los estados del controlador se
    integran en continuo: el integrador del PI con el error de cada instante, y
    la copia del actuador (estado de filtro) con el comando retenido.

    tope = {"P": (lo, hi), "Q": (lo, hi)} acota el comando después de calcularlo
    y antes de aplicarlo: la luz tiene un máximo y no puede ser negativa. El
    controlador no se entera, salvo que antiwindup sea True: entonces el
    integrador del lazo cuyo comando está contra el tope deja de acumular
    mientras el error lo empujaría más allá (integración condicional). La
    historia y la copia del actuador reciben siempre el comando aplicado.
    La salida suma el rango del comando antes de acotar y la fracción de las
    actualizaciones con cada comando contra el tope."""
    usa_extra = hasattr(plant_rhs, "extra0")
    nc = int(getattr(controller, "n_estados", 0))
    ext = hasattr(controller, "compute_ext")
    lag = getattr(controller, "tipo", None) == "lag"
    cada = max(1, int(round(dt_ctrl / dt)))

    def rhs(state, t, P, Q):
        I, E = state[2 + nc], state[3 + nc]
        extra = state[4 + nc:]
        rI, rE = refs(t)
        dz = [rI - I, rE - E]
        if antiwindup:
            # Z1 alimenta a Q (lazo de I) y Z2 a P (lazo de E); los dos suben el
            # comando cuando crecen. Se congela si el error empuja contra el tope.
            if topeQ * dz[0] > 0:
                dz[0] = 0.0
            if topeP * dz[1] > 0:
                dz[1] = 0.0
        if lag:
            th = controller.tau_hat
            dz += [(P - state[2]) / th, (Q - state[3]) / th]
        if usa_extra:
            dI, dE, dex = plant_rhs(I, E, P, Q, extra, t)
        else:
            dI, dE = plant_rhs(I, E, P, Q)
            dex = np.zeros(0)
        return np.concatenate((dz, [dI, dE], np.asarray(dex, dtype=float)))

    t0, tf = t_span
    n = int(round((tf - t0) / dt))
    ex0 = np.asarray(plant_rhs.extra0(), dtype=float) if usa_extra else np.zeros(0)
    state = np.concatenate(([0.0, 0.0], np.zeros(nc), [x0[0], x0[1]], ex0))
    T, I_a, E_a, P_a, Q_a, rI_a, rE_a = [], [], [], [], [], [], []
    t = t0
    P = Q = 0.0
    topeP = topeQ = 0          # -1 contra el piso, +1 contra el techo
    rango = [np.inf, -np.inf, np.inf, -np.inf]
    n_upd = n_tP = n_tQ = n_t = 0
    for i in range(n + 1):
        rI, rE = refs(t)
        if i % cada == 0:
            zs = state[:2 + nc]
            I, E = state[2 + nc], state[3 + nc]
            if ext:
                P, Q, _ = controller.compute_ext(zs, I, E, rI, rE)
            else:
                P, Q, _, _ = controller.compute(zs[0], zs[1], I, E, rI, rE)
            n_upd += 1
            rango = [min(rango[0], P), max(rango[1], P), min(rango[2], Q), max(rango[3], Q)]
            if tope is not None:
                (pl, ph), (ql, qh) = tope["P"], tope["Q"]
                topeP = -1 if P < pl else (1 if P > ph else 0)
                topeQ = -1 if Q < ql else (1 if Q > qh else 0)
                P, Q = min(max(P, pl), ph), min(max(Q, ql), qh)
                n_tP += topeP != 0; n_tQ += topeQ != 0; n_t += (topeP != 0 or topeQ != 0)
            if hasattr(controller, "registrar"):
                controller.registrar(P, Q)
        T.append(t); I_a.append(state[2 + nc]); E_a.append(state[3 + nc])
        P_a.append(P); Q_a.append(Q); rI_a.append(rI); rE_a.append(rE)
        k1 = rhs(state, t, P, Q)
        k2 = rhs(state + 0.5 * dt * k1, t + 0.5 * dt, P, Q)
        k3 = rhs(state + 0.5 * dt * k2, t + 0.5 * dt, P, Q)
        k4 = rhs(state + dt * k3, t + dt, P, Q)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    out = {k: np.asarray(v) for k, v in {
        "t": T, "I": I_a, "E": E_a, "P": P_a, "Q": Q_a, "rI": rI_a, "rE": rE_a}.items()}
    out["y"] = out["E"] - out["I"]
    out["rango_P"], out["rango_Q"] = (rango[0], rango[1]), (rango[2], rango[3])
    out["frac_tope_P"] = n_tP / max(n_upd, 1)
    out["frac_tope_Q"] = n_tQ / max(n_upd, 1)
    out["frac_tope"] = n_t / max(n_upd, 1)
    return out


# -----------------------------------------------------------------------------
#  Referencia theta-gamma con ventana, como en Martínez et al. (2022), fig. 7
# -----------------------------------------------------------------------------
def theta_gamma_ventana_refs(t_fin: float = 2000.0, f_gamma: float = 120.0,
                             banda_theta=(4.0, 12.0), ancho_fase: float = math.pi / 4,
                             semilla: int = 0, dt: float = 0.5):
    """Referencias gamma de 120 Hz encendidas solo cerca de los valles de una
    onda theta (ecs. 13 y 14 del paper multiplicadas por T_w).

    La theta es ruido blanco filtrado entre 4 y 12 Hz, con semilla fija, para
    que no sea una senoidal perfecta (como la theta registrada de la fig. 7a).
    T_w vale 1 cuando la fase de la theta está a menos de `ancho_fase` del valle
    (por defecto un cuarto de ciclo centrado en él) y 0 fuera. Fuera de la
    ventana la referencia es el reposo, E = I = 0.

    Devuelve refs(t) -> (rI, rE), y (t, theta, ventana) para graficar."""
    from scipy.signal import butter, filtfilt, hilbert
    n = int(round(t_fin / dt)) + 1
    t = np.arange(n) * dt
    rng = np.random.default_rng(semilla)
    fs = 1000.0 / dt
    b, a = butter(4, [banda_theta[0] / (fs / 2), banda_theta[1] / (fs / 2)], btype="band")
    theta = filtfilt(b, a, rng.standard_normal(n))
    theta /= np.abs(theta).max()
    fase = np.angle(hilbert(theta))           # el valle de una coseno es fase ±π
    ventana = (np.pi - np.abs(fase)) < ancho_fase
    f = f_gamma / 1000.0

    def refs(tt):
        k = min(int(tt / dt), n - 1)
        if not ventana[k]:
            return 0.0, 0.0
        rI = 0.2 * math.sin(2 * math.pi * f * tt - 0.94) + 0.25
        rE = 0.3 * math.sin(2 * math.pi * f * tt) + 0.45
        return rI, rE
    return refs, (t, theta, ventana)
