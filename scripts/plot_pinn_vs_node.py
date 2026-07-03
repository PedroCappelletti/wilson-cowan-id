#!/usr/bin/env python3
# =============================================================================
#  DIAGRAMA — PINN vs Neural ODE (prototipo estilo matplotlib)
# =============================================================================
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import matplotlib.pyplot as plt

FIG = Path("results/figures/pinn_vs_node.png")

fig, ax = plt.subplots(figsize=(12, 6.2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

BLUE, GREEN, GREY = "#1f4e79", "#2e7d32", "#555555"


def box(x, y, text, color):
    ax.text(x, y, text, ha="center", va="center", fontsize=10.5, color="black",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=color, lw=2))


def arrow(x, y0, y1, color):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2))


def footer(x, text, color):
    ax.text(x, 0.06, text, ha="center", va="center", fontsize=9.5, color="white",
            bbox=dict(boxstyle="round,pad=0.5", fc=color, ec="none"), wrap=True)


ax.text(0.5, 0.975, "PINN  vs  Neural ODE", ha="center", fontsize=15, fontweight="bold")
ax.text(0.25, 0.925, "PINN", ha="center", fontsize=13, fontweight="bold", color=BLUE)
ax.text(0.75, 0.925, "Neural ODE", ha="center", fontsize=13, fontweight="bold", color=GREEN)

# --- Columna PINN (izq) ---
xL = 0.25
box(xL, 0.84, "t  (tiempo)", BLUE)
box(xL, 0.65, "Fourier features", BLUE)
box(xL, 0.46, "MLP  →  x(t) = (I, E)", BLUE)
box(xL, 0.24, "Pérdida = datos  +  residuo físico\n(autograd  ∂x/∂t  vs  WC)", BLUE)
for y0, y1 in [(0.80, 0.70), (0.61, 0.51), (0.42, 0.30)]:
    arrow(xL, y0, y1, BLUE)
footer(xL, "estímulo FIJO · NO es planta · evita estimar derivadas de datos", BLUE)

# --- Columna Neural ODE (der) ---
xR = 0.75
box(xR, 0.84, "estado x = (I, E)  +  P, Q", GREEN)
box(xR, 0.65, "f_θ = WC(θ)  +  corrección g_φ", GREEN)
box(xR, 0.46, "RK4  (integrador diferenciable)", GREEN)
box(xR, 0.24, "Pérdida = rollout MSE\n(real vs integrado)", GREEN)
for y0, y1 in [(0.80, 0.70), (0.61, 0.51), (0.42, 0.30)]:
    arrow(xR, y0, y1, GREEN)
# lazo de rollout
ax.annotate("", xy=(xR + 0.17, 0.84), xytext=(xR + 0.17, 0.46),
            arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.4,
                            connectionstyle="arc3,rad=0.4"))
ax.text(xR + 0.235, 0.65, "rollout", fontsize=8, color=GREY, rotation=90, va="center")
footer(xR, "estímulo VARIABLE · gemelo digital / PLANTA · aprende 10 params", GREEN)

# separador
ax.plot([0.5, 0.5], [0.12, 0.90], color="#cccccc", lw=1, ls="--")

fig.tight_layout()
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=140, bbox_inches="tight"); plt.close(fig)
print(f"Figura: {FIG}")
