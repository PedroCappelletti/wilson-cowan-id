#!/usr/bin/env python3
# =============================================================================
#  F5 — ¿APRENDIO LA RED LA FISICA CORRECTA, O SOLO ALGO QUE AJUSTA?
# =============================================================================
#
#  Esta es la pregunta que casi ningun trabajo de gray-box / UDE puede
#  contestar, porque no conoce la verdad. Nosotros SI: como fabricamos el
#  simulador, sabemos exactamente cual es el termino que falta (Delta f), y lo
#  guardamos junto con cada dataset.
#
#  Entonces se puede comparar directamente:
#        g_φ(I,E)   aprendida de los datos
#     vs Delta f    la fisica que realmente falta
#
#  Dos mediciones distintas y las dos importan:
#
#  1. R2 DENTRO del dominio visitado: ¿la red capturo el termino donde hay datos?
#
#  2. R2 FUERA del dominio visitado: esto decide si el modelo sirve como PLANTA.
#     En lazo cerrado el controlador lleva al sistema a estados que el dataset
#     de entrenamiento no cubrio. Una correccion que extrapola mal ahi es peor
#     que no tener correccion.
#
#  TECHO ALCANZABLE: cuidado con exigir R2=1. Parte del Delta f verdadero
#  depende de estados ocultos (el lag del actuador) que g_φ(I,E) NO PUEDE ver.
#  Por eso se calcula tambien un ORACULO: la mejor correccion sin memoria
#  posible, ajustada directamente contra el Delta f verdadero. Ese oraculo es el
#  techo real, y el R2 de la red hay que leerlo contra ese techo, no contra 1.
#
#  USO:  python scripts/exp_f5_functional_recovery.py --ckpt <ckpt.pt>
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
import torch.nn as nn

torch.set_num_threads(2)

from src.neural_ode import GrayBoxWC
from src.neural_ode.graybox_train import ALL_P

OUT = Path("results/uncertainty/f5_recovery.json")
FIGDIR = Path("results/figures")


def cargar_modelo(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    init = {k: 1.0 for k in ALL_P}
    m = GrayBoxWC(init, {k: 1.0 for k in ("wEE", "wEI", "wIE", "wII")},
                  learnable_weights=True, learnable_params=True,
                  use_correction=ck.get("use_correction", False),
                  correction_inputs=ck.get("correction_inputs", "xpq"))
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


def r2(pred, tgt):
    ss_res = float(((pred - tgt) ** 2).sum())
    ss_tot = float(((tgt - tgt.mean(0)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-30)


def oraculo(Xtr, Ytr, Xte, Yte, epochs=1200, seed=0):
    """La MEJOR correccion sin memoria posible: se ajusta directamente contra el
    Delta f verdadero (con informacion que el entrenamiento real no tiene).
    Marca el techo que g_φ(I,E) podria alcanzar en el mejor de los casos."""
    torch.manual_seed(seed)
    g = nn.Sequential(nn.Linear(Xtr.shape[1], 32), nn.Tanh(),
                      nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 2))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    xt, xv = (Xtr - mu) / sd, (Xte - mu) / sd
    sy = Ytr.pow(2).mean().sqrt() + 1e-12
    opt = torch.optim.Adam(g.parameters(), lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        idx = torch.randint(0, len(xt), (4096,))
        opt.zero_grad()
        ((g(xt[idx]) - Ytr[idx] / sy) ** 2).mean().backward()
        opt.step(); sch.step()
    with torch.no_grad():
        return r2(g(xt) * sy, Ytr), r2(g(xv) * sy, Yte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=None,
                    help="por defecto se deduce del eps con que se entreno el checkpoint")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    m, ck = cargar_modelo(a.ckpt)

    # El dataset TIENE que ser el mismo con el que se entreno el checkpoint: si
    # no, se compara la correccion aprendida a un nivel de mismatch contra el
    # Delta f verdadero de otro, y el R2 no significa nada.
    if a.data is None:
        a.data = f"data/processed/uncertain/eps{float(ck['eps']):g}.npz"
        print(f"  (dataset deducido del checkpoint: {a.data})")
    else:
        esperado = f"eps{float(ck['eps']):g}.npz"
        if Path(a.data).name != esperado:
            raise SystemExit(
                f"ERROR: el checkpoint se entreno con eps={ck['eps']} (=> {esperado}) "
                f"pero se pidio {Path(a.data).name}. Comparar g_hat de un eps contra "
                f"el Delta f de otro no mide nada.")
    d = np.load(a.data, allow_pickle=True)
    if not m.use_correction:
        print("  el checkpoint no tiene correccion: nada que recuperar")
        return

    is_test = d["is_test"].astype(bool)

    def arma(sel):
        I = d["I"][sel].ravel(); E = d["E"][sel].ravel()
        P = d["P"][sel].ravel(); Q = d["Q"][sel].ravel()
        X = torch.tensor(np.stack([I, E], 1), dtype=torch.float32)
        U = torch.tensor(np.stack([P, Q], 1), dtype=torch.float32)
        Y = torch.tensor(np.stack([d["dfI"][sel].ravel(),
                                   d["dfE"][sel].ravel()], 1), dtype=torch.float32)
        return X, U, Y

    Xtr, Utr, Ytr = arma(~is_test)
    Xte, Ute, Yte = arma(is_test)

    with torch.no_grad():
        gtr = m.g_out(Xtr, Utr[:, 0:1], Utr[:, 1:2])
        gte = m.g_out(Xte, Ute[:, 0:1], Ute[:, 1:2])

    r2_tr, r2_te = r2(gtr, Ytr), r2(gte, Yte)

    # Techo: el mejor ajuste memoryless posible contra el Delta f verdadero.
    feat_tr = Xtr if m.correction_inputs == "x" else torch.cat([Xtr, Utr], 1)
    feat_te = Xte if m.correction_inputs == "x" else torch.cat([Xte, Ute], 1)
    o_tr, o_te = oraculo(feat_tr, Ytr, feat_te, Yte)

    # -----------------------------------------------------------------------
    #  DOMINIO DE VALIDEZ. Esto decide si el modelo sirve como planta: en lazo
    #  cerrado el controlador lleva al sistema a estados que el dataset no cubrio.
    #
    #  Hay que medirlo con un criterio ABSOLUTO, no relativo. Partir los puntos
    #  de test por la mediana de su distancia al entrenamiento da siempre 50/50
    #  y los deja a todos DENTRO de la nube: no mide extrapolacion ninguna.
    #  Se compara la distancia contra el espaciado propio de la nube de
    #  entrenamiento, y se reporta cuantos puntos caen realmente afuera.
    # -----------------------------------------------------------------------
    from scipy.spatial import cKDTree
    sub = slice(None, None, 17)
    Xtr_s, Xte_s = Xtr.numpy()[sub], Xte.numpy()[sub]
    tree = cKDTree(Xtr_s)

    # espaciado tipico DENTRO del entrenamiento (vecino mas cercano distinto de si mismo)
    d_self, _ = tree.query(Xtr_s, k=2)
    espaciado = float(np.median(d_self[:, 1]))

    dist, _ = tree.query(Xte_s)
    # "lejos" = mas de 10 espaciados tipicos de cualquier dato de entrenamiento
    umbral = 10.0 * espaciado
    lejos = dist > umbral
    lo, hi = Xtr_s.min(0), Xtr_s.max(0)
    fuera_caja = float(np.mean(np.any((Xte_s < lo) | (Xte_s > hi), axis=1)))

    Yte_n, gte_n = Yte.numpy()[sub], gte.numpy()[sub]
    r2_cerca = r2(torch.tensor(gte_n[~lejos]), torch.tensor(Yte_n[~lejos]))
    r2_lejos = (r2(torch.tensor(gte_n[lejos]), torch.tensor(Yte_n[lejos]))
                if lejos.sum() > 30 else float("nan"))

    # Riesgo de extrapolacion medido SIN necesitar el Delta f verdadero: cuanto
    # crece la correccion al salir de la caja visitada. Si se dispara, meterla en
    # la cancelacion del controlador es peligroso aunque el R2 interior sea bueno.
    margen = 0.4
    ext_lo = lo - margen * (hi - lo)
    ext_hi = hi + margen * (hi - lo)
    gi = np.linspace(ext_lo[0], ext_hi[0], 80)
    ge = np.linspace(ext_lo[1], ext_hi[1], 80)
    GI, GE = np.meshgrid(gi, ge)
    Gp = torch.tensor(np.stack([GI.ravel(), GE.ravel()], 1), dtype=torch.float32)
    zc = torch.zeros(len(Gp), 1)
    with torch.no_grad():
        g_grid = m.g_out(Gp, zc, zc).numpy()
    dentro = ((Gp.numpy() >= lo) & (Gp.numpy() <= hi)).all(1)
    g_in = float(np.sqrt((g_grid[dentro] ** 2).mean())) if dentro.any() else float("nan")
    g_out_ = float(np.sqrt((g_grid[~dentro] ** 2).mean()))
    amplif = g_out_ / max(g_in, 1e-12)

    tag = a.tag or Path(a.ckpt).stem
    print(f"\n=== F5 · {tag} · correccion g({m.correction_inputs}) ===")
    print(f"  |Delta f| verdadero (RMS)         = {float(Ytr.pow(2).mean().sqrt()):.4f}")
    print(f"  |g_φ| aprendida     (RMS)         = {float(gtr.pow(2).mean().sqrt()):.4f}")
    print(f"  R2 vs Delta f  — train            = {r2_tr:.3f}")
    print(f"  R2 vs Delta f  — test (held-out)  = {r2_te:.3f}")
    print(f"  TECHO (oraculo sin memoria) train = {o_tr:.3f}")
    print(f"  TECHO (oraculo sin memoria) test  = {o_te:.3f}")
    # El "aprovechamiento" solo tiene sentido si el techo es positivo. Si el
    # oraculo mismo da R2 <= 0, significa que el Delta f NO ES FUNCION de los
    # argumentos que recibe g_phi: no hay techo que aprovechar, y dividir por el
    # da porcentajes sin sentido.
    if o_te > 0.05:
        print(f"  -> aprovechamiento del techo      = {100*r2_te/o_te:.1f}%")
    else:
        print(f"  -> NO HAY TECHO POSITIVO: el Delta f no es funcion de "
              f"({m.correction_inputs}).")
        print(f"     Ninguna correccion con estos argumentos puede recuperarlo.")
    print(f"\n  --- dominio de validez (criterio absoluto) ---")
    print(f"  espaciado tipico de la nube train = {espaciado:.2e}")
    print(f"  puntos de test a >10 espaciados   = {100*lejos.mean():.1f}%")
    print(f"  puntos de test fuera de la caja   = {100*fuera_caja:.1f}%")
    print(f"  R2 cerca del dominio              = {r2_cerca:.3f}")
    print(f"  R2 lejos del dominio              = {r2_lejos:.3f}"
          + ("  (pocos puntos: no concluyente)" if not np.isfinite(r2_lejos) else ""))
    print(f"  |g| dentro de la caja visitada    = {g_in:.4f}")
    print(f"  |g| fuera (extrapolacion +40%)    = {g_out_:.4f}   "
          f"amplificacion x{amplif:.2f}")
    if amplif > 3:
        print("  AVISO: la correccion se dispara fuera del dominio visitado.")
        print("         Meterla en la cancelacion del controlador es riesgoso.")

    fila = {"tag": tag, "ckpt": a.ckpt, "data": a.data,
            "correction_inputs": m.correction_inputs,
            "df_rms": float(Ytr.pow(2).mean().sqrt()),
            "g_rms": float(gtr.pow(2).mean().sqrt()),
            "r2_train": r2_tr, "r2_test": r2_te,
            "oraculo_train": o_tr, "oraculo_test": o_te,
            "r2_cerca": r2_cerca, "r2_lejos": r2_lejos,
            "espaciado_train": espaciado,
            "frac_test_lejos": float(lejos.mean()),
            "frac_test_fuera_caja": fuera_caja,
            "g_rms_dentro": g_in, "g_rms_fuera": g_out_,
            "amplificacion_extrapolacion": amplif}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(OUT.read_text()) if OUT.exists() else []
    prev = [p for p in prev if p.get("tag") != tag] + [fila]
    OUT.write_text(json.dumps(prev, indent=2))

    _figura(m, d, Xtr, Ytr, tag)
    print(f"  -> {OUT}")


def _figura(m, d, Xtr, Ytr, tag):
    """Mapa 2D: Delta f verdadero vs g_φ aprendida sobre el plano (I,E)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ig = np.linspace(float(Xtr[:, 0].min()), float(Xtr[:, 0].max()), 60)
    Eg = np.linspace(float(Xtr[:, 1].min()), float(Xtr[:, 1].max()), 60)
    II, EE = np.meshgrid(Ig, Eg)
    G = torch.tensor(np.stack([II.ravel(), EE.ravel()], 1), dtype=torch.float32)
    z = torch.zeros(len(G), 1)
    with torch.no_grad():
        gv = m.g_out(G, z, z).numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for k, (comp, nom) in enumerate(zip((0, 1), ("dI", "dE"))):
        im = ax[k].pcolormesh(II, EE, gv[:, comp].reshape(II.shape),
                              cmap="RdBu_r", shading="auto")
        ax[k].set_title(f"correccion aprendida  g_φ[{nom}]")
        ax[k].set_xlabel("I"); ax[k].set_ylabel("E")
        fig.colorbar(im, ax=ax[k], fraction=0.046)
    # nube de puntos visitados, para ver donde hay datos de verdad
    sub = slice(None, None, 37)
    ax[2].scatter(Xtr[sub, 0], Xtr[sub, 1], s=2, alpha=0.3, color="#1f4e79")
    ax[2].set_title("dominio visitado en entrenamiento")
    ax[2].set_xlabel("I"); ax[2].set_ylabel("E")
    ax[2].set_xlim(Ig[0], Ig[-1]); ax[2].set_ylim(Eg[0], Eg[-1])
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"f5_recovery_{tag}.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
