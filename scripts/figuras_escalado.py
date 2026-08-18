#!/usr/bin/env python3
# =============================================================================
#  FIGURAS DEL ESCALADO PROGRESIVO  (para docs/escalado_figuras.html)
# =============================================================================
#  Genera cada figura en dos versiones (clara y oscura) como PNG en
#  docs/figuras_escalado/, y un json con las metricas que el HTML cita.
#  Todo sale de los checkpoints reales en results/escalado/models/.
#
#  USO:  python scripts/figuras_escalado.py
# =============================================================================
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(4)

from src.neural_ode import GrayBoxWC
from src.neural_ode.graybox_train import ALL_P, WEIGHTS
from esc_eval import cargar, _rollout_traj

OUT = Path("docs/figuras_escalado")
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("data/processed/uncertain")
MODELS = Path("results/escalado/models")

# Paleta validada (scripts/validate_palette.js del skill dataviz), light / dark.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e6e5e0",
                  real="#0b0b0b", wb="#eb6834", B="#2a78d6", first="#eda100", best="#1baf7a"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#33332f",
                  real="#f2f2ee", wb="#d95926", B="#3987e5", first="#c98500", best="#199e70"),
}


def style(th):
    c = THEMES[th]
    plt.rcParams.update({
        "figure.facecolor": c["surface"], "axes.facecolor": c["surface"],
        "savefig.facecolor": c["surface"], "text.color": c["ink"],
        "axes.labelcolor": c["ink2"], "xtick.color": c["ink2"], "ytick.color": c["ink2"],
        "axes.edgecolor": c["grid"], "grid.color": c["grid"], "axes.grid": True,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 1.6,
    })
    return c


def save(fig, name, th):
    fig.savefig(OUT / f"{name}_{th}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_ds(name):
    return np.load(DATA / f"{name}.npz", allow_pickle=True)


def wb_from_json(path):
    ph = json.loads(Path(path).read_text())
    ph = ph["params"] if isinstance(ph, dict) else ph[0]["params"]
    m = GrayBoxWC(ph, {k: ph[k] for k in WEIGHTS}, learnable_weights=True, learnable_params=True)
    m.eval()
    return m


def traj(m, d, label):
    idx = int(np.where(d["labels"].astype(str) == label)[0][0])
    I, E, P, Q = d["I"][idx], d["E"][idx], d["P"][idx], d["Q"][idx]
    pred = _rollout_traj(m, I[0], E[0], P, Q, float(d["dt"]))
    return d["t"], np.stack([I, E], 1), pred, P


def nrmse(pred, real):
    r = real.max(0) - real.min(0)
    return float((100 * np.sqrt(((pred - real) ** 2).mean(0)) / r).mean())


def R(tag):
    return json.loads((Path("results/escalado") / f"{tag}.json").read_text())


# =============================================================================
#  FIG 1 — las tres plantas: misma senal de entrada, tres respuestas
# =============================================================================
def fig1(th):
    c = style(th)
    d0, dr, da = load_ds("eps0"), load_ds("refrac1"), load_ds("act1")
    lab = "prbs_1"
    i = int(np.where(d0["labels"].astype(str) == lab)[0][0])
    t = d0["t"]
    fig, ax = plt.subplots(3, 1, figsize=(9, 6.4), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1.6, 1.6]})
    ax[0].plot(t, d0["P"][i], color=c["ink2"], lw=1.2, label="P comandado (el que mandamos)")
    ax[0].plot(t, da["P_eff"][i], color=c["wb"], lw=1.4, label="P efectivo con actuador (retardo + saturación)")
    ax[0].set_ylabel("estímulo P"); ax[0].legend(fontsize=8, loc="upper right")
    ax[0].set_title("Misma entrada, tres plantas distintas — escenario de test «prbs_1»")
    ax[1].plot(t, d0["E"][i], color=c["real"], lw=1.4, label="WC puro (etapa 0)")
    ax[1].plot(t, dr["E"][i], color=c["B"], lw=1.4, label="WC + refractariedad (etapa 1)")
    ax[1].plot(t, da["E"][i], color=c["wb"], lw=1.4, label="WC + actuador (etapa 2)")
    ax[1].set_ylabel("E (excitatoria)"); ax[1].legend(fontsize=8, loc="upper right", ncol=3)
    ax[2].plot(t, np.hypot(dr["dfI"][i], dr["dfE"][i]), color=c["B"], lw=1.2,
               label="|Δf| refractariedad")
    ax[2].plot(t, np.hypot(da["dfI"][i], da["dfE"][i]), color=c["wb"], lw=1.2,
               label="|Δf| actuador")
    ax[2].set_ylabel("|Δf| = |f_planta − f_WC|"); ax[2].set_xlabel("tiempo [ms]")
    ax[2].legend(fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout(); save(fig, "fig1_plantas", th)


# =============================================================================
#  FIG 2 — etapa 0: el piso
# =============================================================================
def fig2(th):
    c = style(th)
    d = load_ds("eps0")
    m = wb_from_json("results/uncertainty/f2_eps0.json")
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for k, lab in enumerate(["chirp", "box_a1.2"]):
        t, real, pred, _ = traj(m, d, lab)
        ax[k].plot(t, real[:, 1], color=c["real"], lw=1.6, label="E real (planta)")
        ax[k].plot(t, pred[:, 1], color=c["best"], lw=1.3, ls="--", label="E del modelo (rollout 200 ms)")
        ax[k].set_title(f"«{lab}» — NRMSE {nrmse(pred, real):.1f}%")
        ax[k].set_ylabel("E")
    ax[0].legend(fontsize=8, loc="upper right"); ax[1].set_xlabel("tiempo [ms]")
    fig.suptitle("Etapa 0 — planta WC pura, white-box: el piso (2.04% promedio en test)",
                 fontweight="bold")
    fig.tight_layout(); save(fig, "fig2_etapa0", th)


# =============================================================================
#  FIG 3 — etapa 1: rollouts real vs modelos
# =============================================================================
def fig3(th):
    c = style(th)
    d = load_ds("refrac1")
    ms = {"white-box": (cargar(MODELS / "e1_wb.pt")[0], c["wb"]),
          "gray-box B  g(I,E)": (cargar(MODELS / "e1_B_w100.pt")[0], c["B"]),
          "estructural S2  (1−r·x)": (cargar(MODELS / "e1_S2.pt")[0], c["best"])}
    lab = "chirp"
    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for k, (name, (m, col)) in enumerate(ms.items()):
        t, real, pred, _ = traj(m, d, lab)
        ax[k].plot(t, real[:, 1], color=c["real"], lw=1.6, label="E real (planta con refractariedad)")
        ax[k].plot(t, pred[:, 1], color=col, lw=1.3, ls="--", label=f"E {name}")
        ax[k].set_title(f"{name} — NRMSE en «{lab}»: {nrmse(pred, real):.1f}%")
        ax[k].set_ylabel("E"); ax[k].legend(fontsize=8, loc="upper right")
        ax[k].set_xlim(80, 200)
    ax[-1].set_xlabel("tiempo [ms]  (zoom 80–200 ms)")
    fig.suptitle("Etapa 1 — planta WC + refractariedad: tres correcciones", fontweight="bold")
    fig.tight_layout(); save(fig, "fig3_etapa1_rollout", th)


# =============================================================================
#  FIG 4 — etapa 2: rollouts real vs modelos
# =============================================================================
def fig4(th):
    c = style(th)
    d = load_ds("act1")
    ms = {"white-box": (cargar(MODELS / "e2_wb.pt")[0], c["wb"]),
          "gray-box B  g(I,E)": (cargar(MODELS / "e2_B.pt")[0], c["B"]),
          "lag (1er intento, τ̂=0.56)": (cargar(MODELS / "e2_lag.pt")[0], c["first"]),
          "lag2 (estado exacto, τ̂=0.98)": (cargar(MODELS / "e2_lag2.pt")[0], c["best"])}
    lab = "chirp"
    fig, ax = plt.subplots(4, 1, figsize=(9, 8.6), sharex=True)
    for k, (name, (m, col)) in enumerate(ms.items()):
        t, real, pred, _ = traj(m, d, lab)
        ax[k].plot(t, real[:, 1], color=c["real"], lw=1.6, label="E real (planta con actuador)")
        ax[k].plot(t, pred[:, 1], color=col, lw=1.3, ls="--", label=f"E {name}")
        ax[k].set_title(f"{name} — NRMSE en «{lab}»: {nrmse(pred, real):.1f}%")
        ax[k].set_ylabel("E"); ax[k].legend(fontsize=8, loc="upper right")
        ax[k].set_xlim(80, 200)
    ax[-1].set_xlabel("tiempo [ms]  (zoom 80–200 ms)")
    fig.suptitle("Etapa 2 — planta WC + actuador con retardo: cuatro correcciones", fontweight="bold")
    fig.tight_layout(); save(fig, "fig4_etapa2_rollout", th)


# =============================================================================
#  FIG 5 — resumen NRMSE de todos los modelos
# =============================================================================
ROWS = [  # (etiqueta, tag, color-key, etapa)
    ("E0 · white-box (piso)", None, "best", 0),
    ("E1 · white-box", "e1_wb", "wb", 1),
    ("E1 · estructural S (1er intento)", "e1_S", "first", 1),
    ("E1 · gray-box B, 5 ms", "e1_B_w100", "B", 1),
    ("E1 · gray-box B, 10 ms", "e1_B_w200", "B", 1),
    ("E1 · gray-box B, 20 ms", "e1_B_w400", "B", 1),
    ("E1 · estructural S2", "e1_S2", "best", 1),
    ("E2 · white-box", "e2_wb", "wb", 2),
    ("E2 · gray-box B", "e2_B", "B", 2),
    ("E2 · latente, 5 ms", "e2_lat_w100", "first", 2),
    ("E2 · latente, 20 ms", "e2_lat_w400", "first", 2),
    ("E2 · lag (1er intento)", "e2_lag", "first", 2),
    ("E2 · lag2", "e2_lag2", "best", 2),
]


def fig5(th):
    c = style(th)
    names, vals, cols = [], [], []
    for name, tag, ck, _ in ROWS:
        names.append(name); cols.append(c[ck])
        vals.append(2.04 if tag is None else R(tag)["nrmse_test"])
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=cols, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + 0.25, yi, f"{v:.2f}%", va="center", fontsize=8.5, color=c["ink"])
    ax.axvline(2.04, color=c["ink2"], lw=1, ls=":"); ax.text(2.2, y[0] + 0.7, "piso 2.04%", fontsize=8, color=c["ink2"])
    ax.axvline(13.1, color=c["ink2"], lw=1, ls="--"); ax.text(13.3, y[0] + 0.7, "antes: las dos juntas 13.1%", fontsize=8, color=c["ink2"])
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("NRMSE de rollout en test [%]  (menor es mejor)")
    ax.set_xlim(0, 18.5); ax.grid(axis="y", visible=False)
    ax.set_title("Resumen — cuánto se aparta cada modelo de la planta")
    fig.tight_layout(); save(fig, "fig5_resumen", th)


# =============================================================================
#  FIG 6 — fisica aprendida: r, tau, R2
# =============================================================================
def fig6(th):
    c = style(th)
    fig, ax = plt.subplots(1, 3, figsize=(9.5, 3.4))
    # r
    S, S2 = R("e1_S"), R("e1_S2")
    from esc_eval import cargar as _c
    rS = _c(MODELS / "e1_S.pt")[0].structured_dict()["r_e"]
    rS2 = _c(MODELS / "e1_S2.pt")[0].structured_dict()["r_e"]
    ax[0].bar([0, 1], [rS, rS2], color=[c["first"], c["best"]], width=0.6)
    ax[0].axhline(0.10, color=c["ink"], lw=1.2, ls="--"); ax[0].text(1.85, 0.103, "real 0.10", va="bottom", ha="right", fontsize=8, color=c["ink2"])
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["S\n(1er intento)", "S2"]); ax[0].set_ylim(0, 0.13)
    ax[0].set_title("Refractariedad r_e aprendida"); ax[0].set_xlim(-0.5, 1.9)
    for x, v in zip([0, 1], [rS, rS2]): ax[0].text(x, v + 0.004, f"{v:.3f}", ha="center", fontsize=8.5)
    # tau
    tl, tl2 = R("e2_lag")["extras"]["tau"], R("e2_lag2")["extras"]["tau"]
    ax[1].bar([0, 1], [tl, tl2], color=[c["first"], c["best"]], width=0.6)
    ax[1].axhline(1.0, color=c["ink"], lw=1.2, ls="--"); ax[1].text(1.85, 1.02, "real 1.0", va="bottom", ha="right", fontsize=8, color=c["ink2"])
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["lag\n(1er intento)", "lag2"]); ax[1].set_ylim(0, 1.25)
    ax[1].set_title("Retardo τ̂ [ms] aprendido"); ax[1].set_xlim(-0.5, 1.9)
    for x, v in zip([0, 1], [tl, tl2]): ax[1].text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=8.5)
    # R2
    tags = [("E1 B", "e1_B_w100", "B"), ("E1 S2", "e1_S2", "best"), ("E2 B", "e2_B", "B"),
            ("E2 lag", "e2_lag", "first"), ("E2 lag2", "e2_lag2", "best")]
    v = [R(t)["r2_delta_test"] for _, t, _ in tags]
    ax[2].bar(range(len(v)), v, color=[c[k] for _, _, k in tags], width=0.6)
    ax[2].axhline(0, color=c["ink"], lw=0.8)
    ax[2].set_xticks(range(len(v))); ax[2].set_xticklabels([n for n, _, _ in tags], fontsize=8)
    ax[2].set_ylim(-2.1, 1.1); ax[2].set_title("R² de la corrección vs Δf real")
    for x, val in enumerate(v):
        ax[2].text(x, val + (0.05 if val > 0 else -0.18), f"{val:.2f}", ha="center", fontsize=8)
    fig.tight_layout(); save(fig, "fig6_fisica", th)


# =============================================================================
#  FIG 7 — por escenario de test
# =============================================================================
def fig7(th):
    c = style(th)
    tags = [("E1 white-box", "e1_wb", "wb"), ("E1 gray-box B", "e1_B_w100", "B"), ("E1 S2", "e1_S2", "best"),
            ("E2 white-box", "e2_wb", "wb"), ("E2 gray-box B", "e2_B", "B"), ("E2 lag2", "e2_lag2", "best")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for a, sub, title in ((ax[0], tags[:3], "Etapa 1 — refractariedad"),
                          (ax[1], tags[3:], "Etapa 2 — actuador")):
        labs = [f["label"] for f in R(sub[0][1])["por_escenario"]]
        x = np.arange(len(labs)); w = 0.26
        for j, (name, tag, ck) in enumerate(sub):
            v = [f["nrmse"] for f in R(tag)["por_escenario"]]
            a.bar(x + (j - 1) * w, v, width=w - 0.02, color=c[ck], label=name)
        a.set_xticks(x); a.set_xticklabels(labs, rotation=35, ha="right", fontsize=8)
        a.set_title(title); a.legend(fontsize=8); a.grid(axis="x", visible=False)
    ax[0].set_ylabel("NRMSE [%]")
    fig.suptitle("Los 7 escenarios de test — el escalón grande «box_a1.2» es el duro en todas las etapas",
                 fontweight="bold", fontsize=10.5)
    fig.tight_layout(); save(fig, "fig7_escenarios", th)


if __name__ == "__main__":
    for th in ("light", "dark"):
        for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
            f(th); print("ok", f.__name__, th, flush=True)
