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
        # derivadas automáticas, una simulación de 50 ms tardaba más de 10 min.
        x = torch.tensor([[I, E]] * 3, dtype=torch.float32)
        pasP = torch.tensor(self.bufP[self.idx], dtype=torch.float32).repeat(3, 1)
        pasQ = torch.tensor(self.bufQ[self.idx], dtype=torch.float32).repeat(3, 1)
        obj = np.array([dI_obj, dE_obj])
        (Plo, Phi), (Qlo, Qhi) = self._cotas(I, E)
        h = 1e-3

        def residuos(P, Q):
            Pc = torch.tensor([[P], [P + h], [P]], dtype=torch.float32)
            Qc = torch.tensor([[Q], [Q], [Q + h]], dtype=torch.float32)
            with torch.no_grad():
                f = self.modelo(x, torch.cat([Pc, pasP], 1), torch.cat([Qc, pasQ], 1))[:, :2]
            return f.numpy().astype(float) - obj

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
                             dt=0.005, dt_ctrl=0.05, x0=(0.0, 0.0)):
    """Lazo cerrado con un controlador DIGITAL: el comando se calcula cada
    dt_ctrl con el estado medido en ese instante y se mantiene constante hasta
    la actualización siguiente. La planta se integra con RK4 a paso dt.

    Sirve igual para IMCController y para IMCExtendido, y todas las
    comparaciones usan el mismo dt_ctrl. Los estados del controlador se
    integran en continuo: el integrador del PI con el error de cada instante, y
    la copia del actuador (estado de filtro) con el comando retenido."""
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
    for i in range(n + 1):
        rI, rE = refs(t)
        if i % cada == 0:
            zs = state[:2 + nc]
            I, E = state[2 + nc], state[3 + nc]
            if ext:
                P, Q, _ = controller.compute_ext(zs, I, E, rI, rE)
            else:
                P, Q, _, _ = controller.compute(zs[0], zs[1], I, E, rI, rE)
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
    return out
