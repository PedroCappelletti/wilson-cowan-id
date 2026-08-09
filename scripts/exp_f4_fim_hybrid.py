#!/usr/bin/env python3
# =============================================================================
#  F4 — FIM DEL HIBRIDO: ¿POR DONDE LE ROBA IDENTIFICABILIDAD g_φ A θ?
# =============================================================================
#
#  Contexto: el proyecto ya sabe, por Fisher+SVD, que direcciones del espacio de
#  parametros estan mal condicionadas (σ10 = wII, σ9 = acople ae-te-wEE...).
#  Ese analisis se hizo para el white-box.
#
#  La pregunta nueva: cuando se agrega la correccion neuronal, ¿por donde entra
#  a competir con los parametros? La hipotesis es que g_φ roba por las
#  direcciones MAS DEBILES, porque son las que menos le cuestan al ajuste de
#  datos: mover el modelo en una direccion mal condicionada casi no cambia la
#  trayectoria, asi que la red puede hacerlo "gratis".
#
#  Como se mide (misma convencion que scripts/fisher_identifiability.py:
#  jacobiano de la TRAYECTORIA, sensibilidad RELATIVA):
#
#    1. J = ∂y/∂θ sobre los estimulos del dataset  ->  SVD  ->  σ_i, V.
#    2. Se calcula cuanto MUEVE LA TRAYECTORIA la correccion aprendida:
#           Δy_g = y(θ, con g) - y(θ, sin g)
#    3. Se busca el cambio de parametros EQUIVALENTE:
#           c = argmin ‖J_rel · c - Δy_g‖
#       Si el residuo es chico, todo lo que hizo g lo podria haber hecho un
#       cambio de θ -> g es redundante y rompe la identificacion.
#       Si el residuo es grande, g esta aportando fisica genuinamente nueva.
#    4. Se descompone c en la base de vectores singulares -> POR QUE DIRECCION
#       roba. Ahi se confirma (o no) la hipotesis.
#
#  USO:  python scripts/exp_f4_fim_hybrid.py --ckpt results/uncertainty/models/f3_D_eps1_lam10.pt
# =============================================================================

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from torch.func import jacfwd

torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)

from src.wilson_cowan import WilsonCowanParams
from src.neural_ode import rollout, GrayBoxWC
from src.neural_ode.graybox_train import ALL_P

PNAMES = list(ALL_P)
SUBSAMPLE = 5
N_SCEN = 4
T_MAX = 1500          # pasos por escenario (recortado: el jacobiano es caro)
OUT = Path("results/uncertainty/f4_fim.json")


def true_theta(d) -> torch.Tensor:
    return torch.tensor([float(d[k]) for k in PNAMES])


def make_rhs(theta, g=None):
    """Espejo diferenciable del forward de GrayBoxWC (no toca el nucleo).
    Si se pasa g, se le suma la correccion aprendida (congelada)."""
    wEE, wEI, wIE, wII, te, ti, ae, ai, thetae, thetai = theta
    ke = torch.sigmoid(-ae * thetae)
    ki = torch.sigmoid(-ai * thetai)

    def rhs(x, P, Q):
        I = x[..., 0:1]
        E = x[..., 1:2]
        u_i = wIE * E - wII * I + Q - thetai
        u_e = wEE * E - wEI * I + P - thetae
        dI = (1.0 / ti) * (-I + torch.sigmoid(ai * u_i) - ki)
        dE = (1.0 / te) * (-E + torch.sigmoid(ae * u_e) - ke)
        out = torch.cat([dI, dE], dim=-1)
        if g is not None:
            # OJO: se le pasan P,Q reales. La variante A usa g(I,E,P,Q), asi que
            # evaluarla con P=Q=0 mediria mal justamente la variante que se
            # supone que muestra la patologia.
            out = out + g(x, P, Q)
        return out
    return rhs


def build_inputs(d, n_scen=N_SCEN):
    """Toma escenarios de entrenamiento del dataset perturbado."""
    is_test = d["is_test"].astype(bool)
    idx = np.where(~is_test)[0][:n_scen]
    dt = float(d["dt"])
    out = []
    for s in idx:
        T = min(T_MAX, d["I"].shape[1])
        x0 = torch.tensor([[d["I"][s, 0], d["E"][s, 0]]])
        Ps = torch.tensor(d["P"][s, :T]).reshape(T, 1, 1)
        Qs = torch.tensor(d["Q"][s, :T]).reshape(T, 1, 1)
        out.append((x0, Ps, Qs))
    return out, dt


def predict(theta, inputs, dt, g=None):
    outs = []
    f = make_rhs(theta, g)
    for x0, Ps, Qs in inputs:
        traj = rollout(f, x0, Ps[:-1], Qs[:-1], dt)[:, 0, :]
        outs.append(traj[::SUBSAMPLE].reshape(-1))
    return torch.cat(outs)


def cargar_g(ckpt_path):
    """Reconstruye la correccion g_φ aprendida y la congela."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not ck.get("use_correction"):
        return None, ck
    init = {k: 1.0 for k in ALL_P}
    m = GrayBoxWC(init, {k: 1.0 for k in ("wEE", "wEI", "wIE", "wII")},
                  learnable_weights=True, learnable_params=True,
                  use_correction=True, correction_inputs=ck["correction_inputs"])
    m.load_state_dict(ck["state"])
    m = m.double().eval()

    def g(x, P, Q):
        with torch.no_grad():
            # g_out ignora P,Q por si sola cuando correction_inputs == "x",
            # asi que pasarlos siempre es correcto para las dos variantes.
            return m.g_out(x, P, Q)
    return g, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/uncertain/eps1.npz")
    ap.add_argument("--ckpt", default=None, help="checkpoint de una variante gray-box")
    a = ap.parse_args()

    d = np.load(a.data, allow_pickle=True)
    theta = true_theta(d)
    inputs, dt = build_inputs(d)
    print(f"=== F4 · FIM del hibrido · {Path(a.data).name} · "
          f"{len(inputs)} escenarios, dt={dt:.4f} ms ===")

    # --- 1) Jacobiano de la trayectoria respecto de θ (sensibilidad relativa).
    J = jacfwd(lambda th: predict(th, inputs, dt))(theta)
    Jr = J * theta.unsqueeze(0)
    U, S, Vt = torch.linalg.svd(Jr, full_matrices=False)
    Sn = (S / S[0]).numpy()
    V = Vt.numpy()
    print(f"    salidas = {Jr.shape[0]}   condicion σ1/σ10 = {float(S[0]/S[-1]):.3e}")

    print("\n--- Espectro (normalizado) y direcciones debiles ---")
    for i in range(len(S)):
        vec = V[i]
        order = np.argsort(-np.abs(vec))
        comps = " ".join(f"{PNAMES[j]}={vec[j]:+.2f}" for j in order[:3])
        print(f"  σ{i+1:2d} rel={Sn[i]:9.2e}   {comps}")

    fila = {"data": a.data, "sing_norm": Sn.tolist(),
            "cond": float(S[0] / S[-1]),
            "V": V.tolist(), "pnames": PNAMES}

    # --- 2) Si hay checkpoint: por donde roba la correccion aprendida.
    if a.ckpt:
        g, ck = cargar_g(a.ckpt)
        if g is None:
            print("\n  (el checkpoint no tiene correccion: nada que analizar)")
        else:
            y0 = predict(theta, inputs, dt)
            yg = predict(theta, inputs, dt, g=g)
            dy = yg - y0                       # cuanto mueve la trayectoria g_φ

            # Cambio de parametros EQUIVALENTE (en unidades relativas).
            sol = torch.linalg.lstsq(Jr, dy.unsqueeze(1))
            c = sol.solution.squeeze(1)
            resid = dy - Jr @ c
            frac_expl = float(1.0 - (resid @ resid) / (dy @ dy + 1e-30))

            print(f"\n--- Correccion aprendida: {Path(a.ckpt).name} ---")
            print(f"  ‖Δy por g‖ / ‖y‖            = {float(dy.norm()/y0.norm()):.4f}")
            print(f"  fraccion EXPLICABLE por δθ  = {frac_expl:.4f}")
            print("     (1.0 = todo lo que hace g lo podia hacer un cambio de "
                  "parametros -> redundante)")
            print(f"  δθ equivalente (relativo): " +
                  " ".join(f"{PNAMES[j]}={float(c[j]):+.3f}" for j in range(10)))

            # Descomposicion en la base singular: por que direccion roba.
            coef = V @ c.numpy()               # componentes de c en la base de V
            peso = coef ** 2 / (np.sum(coef ** 2) + 1e-30)
            print("\n  Por que direccion roba (fraccion de la energia de δθ):")
            for i in np.argsort(-peso)[:5]:
                vec = V[i]
                order = np.argsort(-np.abs(vec))
                comps = " ".join(f"{PNAMES[j]}" for j in order[:3])
                print(f"    σ{i+1:2d} (rel {Sn[i]:8.2e})  {peso[i]*100:5.1f}%   [{comps}]")

            # La prueba de la hipotesis: correlacion entre debilidad y robo.
            rank_debil = np.argsort(Sn)                    # mas debil primero
            top = rank_debil[:3]
            print(f"\n  Energia robada en las 3 direcciones MAS DEBILES: "
                  f"{100*peso[top].sum():.1f}%  (azar = 30%)")

            fila.update({"ckpt": a.ckpt, "frac_explicable": frac_expl,
                         "delta_theta_equiv": c.tolist(),
                         "peso_por_direccion": peso.tolist(),
                         "energia_3_mas_debiles": float(peso[top].sum())})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(OUT.read_text()) if OUT.exists() else []
    prev.append(fila)
    OUT.write_text(json.dumps(prev, indent=2))
    print(f"\n  -> {OUT}")


if __name__ == "__main__":
    main()
