# =============================================================================
#  LAZO CERRADO — controlador IMC + planta (verdadera o aprendida)
# =============================================================================
#
#  Port fiel a Python de  simulador_wilson_cowan_con_control.m  (el controlador
#  que dio el tutor). Es un controlador IMC con linealizacion por realimentacion:
#
#    - Estados del controlador (integradores):  dZ1 = rI - I ,  dZ2 = rE - E
#    - Salida LTI (PI):   Ulti_I = kp_I*(rI-I) + ki_I*Z1   (lazo de I)
#                         Ulti_E = kp_E*(rE-E) + ki_E*Z2   (lazo de E)
#    - Saturacion al dominio de la sigmoidea inversa.
#    - Sigmoidea inversa -> uq, up.
#    - CANCELACION (feedback linearization, usa los pesos):
#          Q = uq - (wIE*E - wII*I)
#          P = up - (wEE*E - wEI*I)
#
#  CLAVE: el controlador usa los pesos (wEE,wEI,wIE,wII) para cancelar el
#  acoplamiento. Si se lo construye con los pesos IDENTIFICADOS (θ̂) y se corre
#  contra la planta verdadera (o contra el modelo aprendido), se mide cuanto
#  degrada el control el error de identificacion -> "validacion orientada al
#  control" (OE3).
#
#  El controlador y la planta estan SEPARADOS: la planta es un callable
#  plant_rhs(I,E,P,Q)->(dI,dE), que puede ser la WC verdadera o el Neural ODE
#  aprendido. Asi se enchufa el modelo aprendido sin tocar el controlador.
#
#  UNIDADES (reconciliado): TODA la cadena trabaja en MILISEGUNDOS. La decision ya
#  estaba tomada en gen_multi_dataset.py ("régimen ms"): el dataset de control, la
#  identificacion (te=1 ms, ti=2 ms; estimulos con freq en Hz -> ciclos/ms via f/1000)
#  y este lazo cerrado (tf=50 ms, refs 120 Hz = 0.12 ciclos/ms) usan la MISMA unidad.
#  Los parametros del modelo son numericamente identicos en ambos lados; lo unico que
#  cambia es el paso de integracion (dt~0.05 ms en identificacion vs dt=0.005 ms aca),
#  que es una eleccion numerica, no de unidades. No hay reescalado de parametros.
#
#  NOTA: el archivo original usa realimentacion de ESTADO COMPLETO (I,E directos,
#  sin filtro de Kalman). Se replica igual. El EKF del paper seria una capa extra.
# =============================================================================

from __future__ import annotations

import math
import numpy as np


# -----------------------------------------------------------------------------
#  Referencias theta-gamma (las senoides del archivo del controlador).
# -----------------------------------------------------------------------------
def theta_gamma_refs(freq_hz: float = 120.0, time_in_ms: bool = True):
    """Devuelve refs(t) -> (rI, rE). Por defecto, las del MATLAB:
        rI = 0.2*sin(2π f t - 0.94) + 0.25
        rE = 0.3*sin(2π f t)        + 0.45
    con f=120 Hz y t en ms (f/1000 ciclos por ms)."""
    f = freq_hz / 1000.0 if time_in_ms else freq_hz

    def refs(t):
        rI = 0.2 * math.sin(2 * math.pi * f * t - 0.94) + 0.25
        rE = 0.3 * math.sin(2 * math.pi * f * t) + 0.45
        return rI, rE
    return refs


# -----------------------------------------------------------------------------
#  Controlador IMC + sigmoidea inversa + cancelacion.
# -----------------------------------------------------------------------------
def _inv_sat(u_eff: float, sat: float) -> float:
    """Invierte la saturacion  u_eff = sat*tanh(u_cmd/sat)  ->  u_cmd.

    Si el objetivo pide mas de lo que el actuador puede entregar (|u_eff| >= sat)
    la inversa no existe: se comanda el maximo. Es saturacion fisica real, no un
    problema numerico."""
    r = u_eff / sat
    r = min(max(r, -0.999), 0.999)
    return float(sat * math.atanh(r))


class IMCController:
    def __init__(
        self,
        fixed: dict,            # ae,ai,thetae,thetai,ke,ki (+ te,ti si hay g_hat)
        weights: dict,          # wEE,wEI,wIE,wII (los que usa la cancelacion: verdaderos o θ̂)
        kp_I: float = 10.0, ki_I: float = 5.0,   # PI del lazo de I  (Ulti1 = 5*Z1 + 10*err)
        kp_E: float = 5.0,  ki_E: float = 5.0,   # PI del lazo de E  (Ulti2 = 5*Z2 + 5*err)
        argmin: float = -100.0, argmax: float = 100.0,
        g_hat=None,             # correccion gray-box aprendida: (I,E) -> (gI,gE)
        sat_ref: dict | None = None,   # parametros con que se calculan los topes
        structured: dict | None = None,  # {r_i, r_e, sat} de la correccion fisica
    ) -> None:
        self.f = fixed
        self.w = weights
        self.kp_I, self.ki_I = kp_I, ki_I
        self.kp_E, self.ki_E = kp_E, ki_E
        # CANCELACION GRAY-BOX
        # -------------------
        # Sin correccion, el controlador quiere lograr   dI = (1/ti)(-I + Ulti_I)
        # y para eso invierte la sigmoidea pidiendo      σ_i = Ulti_I + ki.
        #
        # Si el modelo aprendido es  dI = (1/ti)(-I + σ_i - ki) + ĝ_I(I,E),
        # igualar al objetivo da                          σ_i = Ulti_I + ki - ti·ĝ_I.
        #
        # O sea: alcanza con CORRER EL OBJETIVO de la sigmoidea por -ti·ĝ_I.
        # Esto es exacto y explicito SOLO porque ĝ depende unicamente del estado.
        # Si ĝ dependiera tambien de P,Q, la ecuacion quedaria implicita en P,Q y
        # habria que resolver un punto fijo en cada paso.
        self.g_hat = g_hat

        # CANCELACION ESTRUCTURADA (la alternativa que SI funciona).
        # -----------------------------------------------------------
        # Si la correccion no es una red sino fisica con forma conocida
        #   dI = (1/ti)( -I + (1 - r_i·I)·σ_i - ki ),  con Q_eff = sat·tanh(Q/sat)
        # entonces la inversion se hace a mano y es EXACTA:
        #
        #   objetivo:   dI = (1/ti)(-I + Ulti_I)
        #   igualando:  (1 - r_i·I)·σ_i - ki = Ulti_I
        #   despejando: σ_i = (Ulti_I + ki) / (1 - r_i·I)      <- explicito en I
        #   y al final: Q_comandado = sat·artanh(Q_efectivo / sat)
        #
        # Todo explicito, sin punto fijo. Es la ventaja concreta de usar una
        # correccion con forma fisica en vez de una caja negra.
        self.structured = structured

        # Los TOPES DE ACTUACION representan hasta donde puede empujar el
        # estimulador: son una propiedad del equipo, no del modelo. Por defecto
        # salen de `fixed` (comportamiento historico), pero al comparar varios
        # controladores contra LA MISMA planta hay que pasarles el mismo `sat_ref`.
        # Si no, cada controlador enfrenta un rango de actuacion distinto —
        # el que tiene el `ai` estimado mas chico gana autoridad de control—
        # y la comparacion mide eso en vez de medir la calidad del modelo.
        s = sat_ref or fixed
        ai, ki = s["ai"], s["ki"]
        ae, ke = s["ae"], s["ke"]
        ti_th, te_th = s["thetai"], s["thetae"]
        # Limites de saturacion: dominio valido de la sigmoidea inversa.
        self.fim = 0.99999 * (1 / (1 + math.exp(-ai * (argmin - ti_th))) - ki)
        self.fiM = 0.99999 * (1 / (1 + math.exp(-ai * (argmax - ti_th))) - ki)
        self.fem = 0.99999 * (1 / (1 + math.exp(-ae * (argmin - te_th))) - ke)
        self.feM = 0.99999 * (1 / (1 + math.exp(-ae * (argmax - te_th))) - ke)
        # Los mismos limites, ya expresados como objetivo de la SALIDA de la
        # sigmoidea (que es sobre lo que actua la correccion gray-box).
        #
        # Se aplican SIEMPRE, con o sin correccion. Es importante: si el
        # reclampeo existiera solo en la rama con g_hat, el controlador cambiaria
        # de comportamiento por el solo hecho de pasarle una correccion NULA, y
        # la comparacion "con g" vs "sin g" mediria el cambio de saturacion en
        # vez de medir la correccion. Con g_hat=None estos limites son un no-op
        # exacto, porque Usat ya viene saturado al mismo rango.
        self.lo_i, self.hi_i = self.fim + ki, self.fiM + ki
        self.lo_e, self.hi_e = self.fem + ke, self.feM + ke
        # Diagnostico: cuantas veces la saturacion mordio (ver nota en compute()).
        self.n_pasos = 0
        self.n_sat_i = 0
        self.n_sat_e = 0

    def compute(self, Z1, Z2, I, E, rI, rE):
        """Devuelve (P, Q, dZ1, dZ2) dado el estado del controlador y de la planta."""
        f, w = self.f, self.w
        ai, ki, ti_th = f["ai"], f["ki"], f["thetai"]
        ae, ke, te_th = f["ae"], f["ke"], f["thetae"]

        # PI del controlador.
        Ulti_I = self.kp_I * (rI - I) + self.ki_I * Z1
        Ulti_E = self.kp_E * (rE - E) + self.ki_E * Z2

        # Saturacion al dominio de la inversa.
        Usat_I = min(max(Ulti_I, self.fim), self.fiM)
        Usat_E = min(max(Ulti_E, self.fem), self.feM)

        # Objetivo para la salida de cada sigmoidea.
        tgt_i = Usat_I + ki
        tgt_e = Usat_E + ke

        # Compensacion de la correccion gray-box (ver __init__).
        if self.g_hat is not None:
            gI, gE = self.g_hat(I, E)
            tgt_i -= f["ti"] * gI
            tgt_e -= f["te"] * gE

        # Compensacion ESTRUCTURADA: se divide por el factor refractario.
        if self.structured is not None:
            fi = 1.0 - self.structured["r_i"] * I
            fe = 1.0 - self.structured["r_e"] * E
            # Si el factor se acerca a cero la poblacion esta saturada de
            # refractariedad y no hay actuacion posible: se acota para no dividir
            # por ~0, y el contador de saturacion lo registra mas abajo.
            tgt_i = tgt_i / max(fi, 1e-3)
            tgt_e = tgt_e / max(fe, 1e-3)

        # Saturacion al dominio valido de la sigmoidea inversa. SIEMPRE, para que
        # el camino de codigo no dependa de si hay correccion o no.
        #
        # OJO al interpretar resultados: cuando esta saturacion muerde, la
        # cancelacion DEJA DE SER EXACTA (el controlador no puede pedir lo que
        # querria). Sin contarlo, un g_hat con la escala equivocada produce un
        # lazo finito y de aspecto sano, porque el clamp le tapa la divergencia.
        # Por eso se cuenta y se reporta.
        self.n_pasos += 1
        if not (self.lo_i <= tgt_i <= self.hi_i):
            self.n_sat_i += 1
            tgt_i = min(max(tgt_i, self.lo_i), self.hi_i)
        if not (self.lo_e <= tgt_e <= self.hi_e):
            self.n_sat_e += 1
            tgt_e = min(max(tgt_e, self.lo_e), self.hi_e)

        # Sigmoidea inversa.
        uq = (-1.0 / ai) * math.log(-1.0 + 1.0 / tgt_i) + ti_th
        up = (-1.0 / ae) * math.log(-1.0 + 1.0 / tgt_e) + te_th

        # Cancelacion del acoplamiento (feedback linearization) -> estimulos.
        Q = uq - (w["wIE"] * E - w["wII"] * I)
        P = up - (w["wEE"] * E - w["wEI"] * I)

        # Si el modelo dice que el actuador satura, hay que COMANDAR de mas para
        # que llegue lo que se quiere: se invierte la saturacion.
        if self.structured is not None:
            sat = self.structured["sat"]
            Q = _inv_sat(Q, sat)
            P = _inv_sat(P, sat)

        dZ1 = rI - I
        dZ2 = rE - E
        return P, Q, dZ1, dZ2


# -----------------------------------------------------------------------------
#  Plantas: callables plant_rhs(I,E,P,Q) -> (dI,dE).
# -----------------------------------------------------------------------------
def make_true_plant(fixed: dict, weights: dict):
    """Planta verdadera: ecuaciones de Wilson-Cowan (V0 / referencia)."""
    te, ti = fixed["te"], fixed["ti"]
    ae, ai = fixed["ae"], fixed["ai"]
    the, thi = fixed["thetae"], fixed["thetai"]
    ke, ki = fixed["ke"], fixed["ki"]
    wEE, wEI, wIE, wII = (weights["wEE"], weights["wEI"], weights["wIE"], weights["wII"])

    def rhs(I, E, P, Q):
        u_i = wIE * E - wII * I + Q - thi
        u_e = wEE * E - wEI * I + P - the
        dI = (1.0 / ti) * (-I + 1.0 / (1.0 + math.exp(-ai * u_i)) - ki)
        dE = (1.0 / te) * (-E + 1.0 / (1.0 + math.exp(-ae * u_e)) - ke)
        return dI, dE
    return rhs


def make_neural_plant(model):
    """Planta aprendida: envuelve un GrayBoxWC (torch) como callable escalar."""
    import torch

    def rhs(I, E, P, Q):
        with torch.no_grad():
            x = torch.tensor([[float(I), float(E)]], dtype=torch.float32)
            dx = model(x, torch.tensor([[float(P)]]), torch.tensor([[float(Q)]]))
        return float(dx[0, 0]), float(dx[0, 1])
    return rhs


def make_perturbed_plant(params, perturbation):
    """Planta REAL con incertidumbre dinamica: Wilson-Cowan + la fisica que el
    modelo no contempla (ver src/wilson_cowan/uncertainty.py).

    A diferencia de las otras plantas, esta puede tener ESTADOS OCULTOS propios
    (el estado del actuador, la corriente de adaptacion...). Por eso expone
    n_extra y extra0(), y su rhs recibe y devuelve tambien esos estados.

    Es la planta contra la que se mide todo en F6: representa el cerebro real.
    """
    from src.wilson_cowan import WilsonCowan

    m = WilsonCowan(params=params, perturbation=perturbation)

    def rhs(I, E, P, Q, extra, t=0.0):
        dI, dE, dex = m.perturbed_field(t, I, E, extra, P, Q)
        return dI, dE, dex

    rhs.n_extra = perturbation.n_extra
    rhs.extra0 = perturbation.extra0
    return rhs


def make_graybox_correction(model):
    """Envuelve la correccion g_φ de un GrayBoxWC como callable (I,E)->(gI,gE),
    que es lo que consume IMCController(g_hat=...).

    Solo tiene sentido si el modelo se entreno con correction_inputs='x': si g
    dependiera de P,Q la cancelacion no seria explicita."""
    import torch

    if not model.use_correction:
        return None
    if model.correction_inputs != "x":
        raise ValueError(
            "la cancelacion explicita requiere g(I,E); este modelo usa g(I,E,P,Q)")

    def g_hat(I, E):
        with torch.no_grad():
            x = torch.tensor([[float(I), float(E)]], dtype=torch.float32)
            z = torch.zeros(1, 1)
            g = model.g_out(x, z, z)
        return float(g[0, 0]), float(g[0, 1])
    return g_hat


# -----------------------------------------------------------------------------
#  Simulacion del lazo cerrado (RK4 sobre el estado aumentado [Z1,Z2,I,E]).
# -----------------------------------------------------------------------------
def simulate_closed_loop(plant_rhs, controller: IMCController, refs,
                         t_span=(0.0, 50.0), dt=0.005, x0=(0.0, 0.0)):
    """Integra controlador + planta. Devuelve dict con arrays t, I, E, y, P, Q, rI, rE.

    El estado integrado es [Z1, Z2, I, E, *estados_ocultos_de_la_planta]. Los
    estados ocultos solo existen si la planta los declara (n_extra > 0): son la
    fisica que el controlador NO conoce y que tiene que rechazar."""

    # Que convencion de llamada usa la planta se decide por la INTERFAZ, no por
    # cuantos estados ocultos tenga: una planta perturbada con n_extra=0 (p.ej.
    # NoPerturbation) igual espera el argumento `extra`.
    usa_extra = hasattr(plant_rhs, "extra0")
    n_extra = int(getattr(plant_rhs, "n_extra", 0))

    def aug_rhs(state, t):
        Z1, Z2, I, E = state[0], state[1], state[2], state[3]
        extra = state[4:]
        rI, rE = refs(t)
        P, Q, dZ1, dZ2 = controller.compute(Z1, Z2, I, E, rI, rE)
        if usa_extra:
            dI, dE, dex = plant_rhs(I, E, P, Q, extra, t)
        else:
            dI, dE = plant_rhs(I, E, P, Q)
            dex = np.zeros(0)
        return np.concatenate(([dZ1, dZ2, dI, dE], np.asarray(dex, dtype=float))), P, Q, rI, rE

    t0, tf = t_span
    n = int(round((tf - t0) / dt))
    ex0 = np.asarray(plant_rhs.extra0(), dtype=float) if usa_extra else np.zeros(0)
    state = np.concatenate(([0.0, 0.0, x0[0], x0[1]], ex0))

    T, I_a, E_a, P_a, Q_a, rI_a, rE_a = [], [], [], [], [], [], []
    t = t0
    for _ in range(n + 1):
        d1, P, Q, rI, rE = aug_rhs(state, t)
        # Registro en el punto actual.
        T.append(t); I_a.append(state[2]); E_a.append(state[3])
        P_a.append(P); Q_a.append(Q); rI_a.append(rI); rE_a.append(rE)
        # Paso RK4 (refs evaluadas en los sub-tiempos).
        k1, _, _, _, _ = aug_rhs(state, t)
        k2, _, _, _, _ = aug_rhs(state + 0.5 * dt * k1, t + 0.5 * dt)
        k3, _, _, _, _ = aug_rhs(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4, _, _, _, _ = aug_rhs(state + dt * k3, t + dt)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt

    out = {k: np.asarray(v) for k, v in {
        "t": T, "I": I_a, "E": E_a, "P": P_a, "Q": Q_a, "rI": rI_a, "rE": rE_a,
    }.items()}
    out["y"] = out["E"] - out["I"]
    return out
