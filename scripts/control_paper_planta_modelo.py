#!/usr/bin/env python3
# El controlador del paper, con los parámetros del .m, sobre dos plantas: el
# simulador perturbado y cada modelo gray-box identificado de ese simulador.
#
# Controlador: IMCController con los β verdaderos (los del .m), sin tope ni
# anti-windup. Es el mismo en todas las filas; lo único que cambia es la planta:
#
#   simulador     Wilson-Cowan sin perturbación (ideal), con refractariedad o
#                 con actuador (make_perturbed_plant)
#   white-box     el modelo e16 sin corrección
#   con red       refractariedad: agnóstica g(I, E); actuador: historia de 5 ms
#   forma exacta  refractariedad: (1 − r x) y saturación; actuador: estado de filtro
#
# Los modelos se evalúan en numpy (CampoModelo): con torch, una corrida de 2 s
# son 1,6 millones de forwards y tarda más de diez minutos. verificar() compara
# los dos campos antes de simular.
#
#   python scripts/control_paper_planta_modelo.py [procesos]

from __future__ import annotations

import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import numpy as np

from control_paper_modelos import CON_RED, EXACTA, WB, REFS, NIVELES, SEMILLAS, DT, RES, W

OUT = RAIZ / "results" / "control_planta_modelo.json"
TRAZAS = RAIZ / "results" / "control_planta_modelo_trazas.npz"
DT_MODELO = 0.5          # paso de muestreo de los datos e16 (2 kHz), en ms


class CampoModelo:
    """El campo de un GrayBoxWC (o del backbone de un LagGrayBox) en numpy."""

    def __init__(self, core):
        p = core.params_dict()
        self.w = [p[k] for k in W]
        self.te, self.ti, self.ae, self.ai = p["te"], p["ti"], p["ae"], p["ai"]
        self.the, self.thi = p["thetae"], p["thetai"]
        self.ke = 1.0 / (1.0 + math.exp(self.ae * self.the))
        self.ki = 1.0 / (1.0 + math.exp(self.ai * self.thi))
        s = core.structured_dict() if core.structured else {}
        self.alpha = s.get("alpha")
        self.r_i, self.r_e = s.get("r_i", 0.0), s.get("r_e", 0.0)
        self.entradas = core.correction_inputs if core.use_correction else None
        self.fir, self.capas = None, []
        if self.entradas is not None:
            g = core.g
            if self.entradas == "xconv":
                self.fir = g.fir.weight.detach().numpy().astype(float).T
                g = g.mlp
            self.capas = [(c.weight.detach().numpy().astype(float).T,
                           c.bias.detach().numpy().astype(float))
                          for c in g if hasattr(c, "weight")]

    def __call__(self, I, E, P, Q):
        """P y Q: escalares, o vectores [actual, retardo 1, ...] con historia."""
        P0 = P[0] if np.ndim(P) else P
        Q0 = Q[0] if np.ndim(Q) else Q
        if self.alpha is not None:
            P0 = math.tanh(self.alpha * P0) / self.alpha
            Q0 = math.tanh(self.alpha * Q0) / self.alpha
        wEE, wEI, wIE, wII = self.w
        ui = wIE * E - wII * I + Q0 - self.thi
        ue = wEE * E - wEI * I + P0 - self.the
        dI = (-I + (1.0 - self.r_i * I) / (1.0 + math.exp(-self.ai * ui)) - self.ki) / self.ti
        dE = (-E + (1.0 - self.r_e * E) / (1.0 + math.exp(-self.ae * ue)) - self.ke) / self.te
        if self.entradas is None:
            return dI, dE
        if self.entradas == "x":
            h = np.array([I, E])
        elif self.entradas == "xconv":
            h = np.concatenate(([I, E], np.concatenate((P, Q)) @ self.fir))
        elif self.entradas == "xhistp":
            h = np.concatenate(([I, E], P[1:], Q[1:]))
        else:
            h = np.concatenate(([I, E], P, Q))
        for k, (Wk, b) in enumerate(self.capas):
            h = h @ Wk + b
            if k < len(self.capas) - 1:
                h = np.tanh(h)
        return dI + h[0], dE + h[1]


class PlantaHistoria:
    """Modelo con historia del comando. Guarda el comando aplicado en cada paso
    y le pasa a la red P y Q en t, t − 0,5 ms, ..., t − (K−1)·0,5 ms."""

    def __init__(self, campo, K, n_pasos):
        self.campo, self.K = campo, K
        self.logP = np.zeros(n_pasos + 2)
        self.logQ = np.zeros(n_pasos + 2)

    def extra0(self):
        return np.zeros(0)

    def __call__(self, I, E, P, Q, extra, t):
        s = t / DT
        # Tolerancia de 1e-3 pasos, no de 1e-6. t se acumula sumando dt y el
        # error de redondeo pasa de 1e-6 pasos hacia los 1292 ms: con la
        # tolerancia vieja el comando dejaba de registrarse, la red veía una
        # historia en cero y el lazo se trababa en un punto fijo falso (theta2s
        # con H y K, hasta el 10/11/2026). Las etapas intermedias del RK4 caen en
        # medio paso, lejos de cualquier tolerancia chica.
        i = int(math.floor(s + 1e-3))
        # Al principio de cada paso el comando nuevo pisa al que dejó la última
        # etapa del RK4 anterior, así que cada casilla termina con el aplicado.
        if abs(s - round(s)) < 1e-3:
            self.logP[i], self.logQ[i] = P, Q
        Ph, Qh = np.empty(self.K), np.empty(self.K)
        Ph[0], Qh[0] = P, Q
        for k in range(1, self.K):
            j = int(math.floor((t - k * DT_MODELO) / DT + 1e-3))
            Ph[k], Qh[k] = (self.logP[j], self.logQ[j]) if j >= 0 else (0.0, 0.0)
        dI, dE = self.campo(I, E, Ph, Qh)
        return dI, dE, ()


class PlantaFiltro:
    """LagGrayBox: estado [P̂, Q̂] con dP̂/dt = (P − P̂)/τ̂, y el backbone ve
    tanh(α P̂)/α, o P̂ recortado en ±1/α si el modelo es el de la planta con recorte."""

    def __init__(self, m):
        self.campo = CampoModelo(m.core)
        ex = m.extras_dict()
        self.tau, self.alpha = ex["tau"], ex["alpha"]
        self.recorte = ex.get("recorte", False)

    def extra0(self):
        return np.zeros(2)

    def __call__(self, I, E, P, Q, extra, t):
        Ph, Qh = extra
        a = self.alpha
        if self.recorte:
            dI, dE = self.campo(I, E, min(1 / a, max(-1 / a, Ph)), min(1 / a, max(-1 / a, Qh)))
        else:
            dI, dE = self.campo(I, E, math.tanh(a * Ph) / a, math.tanh(a * Qh) / a)
        return dI, dE, ((P - Ph) / self.tau, (Q - Qh) / self.tau)


def planta_de(m, n_pasos):
    if hasattr(m, "extras_dict") and "tau" in m.extras_dict():
        return PlantaFiltro(m)
    campo = CampoModelo(m)
    if campo.entradas in ("xhist", "xhistp", "xconv"):
        return PlantaHistoria(campo, m.hist_len, n_pasos)
    return lambda I, E, P, Q: campo(I, E, P, Q)


def verificar(m, n=200, semilla=0):
    """Diferencia máxima entre el campo de torch y el de numpy en puntos al azar."""
    import torch
    rng = np.random.default_rng(semilla)
    x = rng.uniform(-0.2, 0.9, (n, 2))
    lag = hasattr(m, "extras_dict") and "tau" in m.extras_dict()
    K = getattr(m, "hist_len", 0) if getattr(m, "correction_inputs", "") in ("xhist", "xhistp", "xconv") else 0
    u = rng.uniform(-15, 15, (n, 2, max(K, 1)))
    with torch.no_grad():
        if lag:
            y = torch.tensor(np.concatenate([x, u[:, :, 0]], 1), dtype=torch.float32)
            ref = m(y, torch.tensor(u[:, :1, 0], dtype=torch.float32),
                    torch.tensor(u[:, 1:, 0], dtype=torch.float32)).numpy()[:, :2]
        else:
            P = torch.tensor(u[:, 0, :] if K else u[:, 0, :1], dtype=torch.float32)
            Q = torch.tensor(u[:, 1, :] if K else u[:, 1, :1], dtype=torch.float32)
            ref = m(torch.tensor(x, dtype=torch.float32), P, Q).numpy()
    if lag:
        pl = PlantaFiltro(m)
        mio = np.array([pl(x[k, 0], x[k, 1], 0.0, 0.0, u[k, :, 0], 0.0)[:2] for k in range(n)])
    else:
        c = CampoModelo(m)
        mio = np.array([c(x[k, 0], x[k, 1], u[k, 0] if K else u[k, 0, 0],
                          u[k, 1] if K else u[k, 1, 0]) for k in range(n)])
    return float(np.abs(mio - ref).max() / max(np.abs(ref).max(), 1e-12))


def tareas():
    out = []
    for ref in REFS:
        for planta in ("ideal", "refrac", "act"):
            out.append((ref, planta, "simulador", None, "", -1))
        for planta in ("refrac", "act"):
            for suf in NIVELES:
                for k in SEMILLAS:
                    out.append((ref, planta, "white-box", f"e16_{WB[planta]}{suf}_s{k}", suf, k))
                    out.append((ref, planta, "con red", f"e16_{CON_RED[(planta, suf)]}{suf}_s{k}", suf, k))
                    out.append((ref, planta, "forma exacta", f"e16_{EXACTA[planta]}{suf}_s{k}", suf, k))
    return out


def una(t):
    import torch
    torch.set_num_threads(1)
    from src.wilson_cowan import WilsonCowanParams, Refractoriness, Actuator, NoPerturbation
    from src.neural_ode import IMCController, theta_gamma_refs, make_perturbed_plant
    from src.neural_ode.closed_loop_ext import simulate_closed_loop_ext, theta_gamma_ventana_refs
    from src.neural_ode.graybox_train import ALL_P
    from esc_eval import cargar
    from exp_f6_closed_loop import fixed_de, rmse

    ref, pert, modelo, tag, suf, k = t
    p = WilsonCowanParams()
    true = {kk: getattr(p, kk) for kk in ALL_P}
    if ref == "theta2s":
        refs, t_span = theta_gamma_ventana_refs()[0], (0.0, 2000.0)
    else:
        refs, t_span = theta_gamma_refs(120.0, True), (0.0, 50.0)
    n_pasos = int(round((t_span[1] - t_span[0]) / DT))
    if tag is None:
        planta = make_perturbed_plant(p, {"ideal": NoPerturbation(), "refrac": Refractoriness(r=0.10),
                                          "act": Actuator(sat=3.0, tau_act=1.0)}[pert])
    else:
        m = cargar(RES / "models" / f"{tag}.pt")[0]
        planta = planta_de(m, n_pasos)
    ctrl = IMCController(fixed_de(true), {kk: true[kk] for kk in W}, sat_ref=fixed_de(true))
    with np.errstate(over="ignore", invalid="ignore"):
        sol = simulate_closed_loop_ext(planta, ctrl, refs, t_span=t_span, dt=DT, dt_ctrl=DT)
    finito = bool(np.all(np.isfinite(sol["E"])) and np.all(np.isfinite(sol["I"])))
    eI, eE = rmse(sol) if finito else (float("nan"), float("nan"))
    fila = {"referencia": ref, "perturbacion": pert, "planta": modelo, "tag": tag,
            "ruido": suf, "semilla": k, "rmse": math.sqrt((eI ** 2 + eE ** 2) / 2),
            "rmse_I": eI, "rmse_E": eE, "finito": finito,
            "P_rango": [float(np.nanmin(sol["P"])), float(np.nanmax(sol["P"]))],
            "Q_rango": [float(np.nanmin(sol["Q"])), float(np.nanmax(sol["Q"]))]}
    traza = None
    if k in (0, -1) and suf in ("", "_n01"):
        paso = 20 if ref == "theta2s" else 2
        traza = {kk: sol[kk][::paso].astype(np.float32) for kk in ("t", "E", "I", "rE", "rI", "P")}
    return fila, traza


def main():
    if "--verificar" in sys.argv:
        from esc_eval import cargar
        tags = sorted({t[3] for t in tareas() if t[3]})
        for tag in tags:
            print(f"  {tag:24} {verificar(cargar(RES / 'models' / f'{tag}.pt')[0]):.2e}", flush=True)
        return
    procesos = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ts = tareas()
    print(f"{len(ts)} simulaciones, {procesos} procesos", flush=True)
    filas, trazas = [], {}
    with Pool(procesos) as pool:
        for fila, traza in pool.imap_unordered(una, ts):
            filas.append(fila)
            print(f"  {fila['referencia']:8} {fila['perturbacion']:7} {fila['planta']:13} "
                  f"{fila['ruido']:5} s{fila['semilla']:>2}  RMSE {fila['rmse']:.4f}", flush=True)
            if traza is not None:
                clave = f"{fila['referencia']}|{fila['perturbacion']}|{fila['planta']}"
                for kk, v in traza.items():
                    trazas[f"{clave}|{kk}"] = v
    OUT.write_text(json.dumps(filas, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(TRAZAS, **trazas)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
