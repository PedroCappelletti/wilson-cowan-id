#!/usr/bin/env python3
# =============================================================================
#  DIAGRAMA CONCEPTUAL — el "valle plano" de la verosimilitud (SVD/FIM -> wII)
# =============================================================================
#
#  Ilustra POR QUE un parametro es poco identificable: cerca del optimo, la
#  verosimilitud (o el costo = -log L) se aproxima por una cuadratica cuyo Hessiano
#  ES la matriz de informacion de Fisher. Si la FIM tiene un autovalor chico, hay
#  una direccion PLANA: moverse por ella casi no cambia el costo -> el parametro (o
#  la combinacion) en esa direccion no se puede determinar. En el proyecto esa
#  direccion esta dominada por wII.
#
#  NO usa datos: es un diagrama pedagogico (aproximacion cuadratica local). El eje
#  largo de la elipse de confianza es ~ 1/sqrt(lambda_min) = cota de Cramer-Rao.
#
#  USO:  python -u scripts/plot_valle_plano.py
# =============================================================================
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrow

FIG = Path("results/figures/valle_plano_svd_fim.png")

# Autovalores de la FIM: uno grande (bien condicionado) y uno chico (plano).
LAM_STEEP, LAM_FLAT = 1.0, 0.03        # cond ~ 33
ALPHA = np.deg2rad(22.0)               # angulo de la direccion PLANA respecto de x
CHI2 = 1.0                             # nivel de la elipse de confianza (~1 sigma)

# Direcciones (autovectores).
v_flat = np.array([np.cos(ALPHA), np.sin(ALPHA)])
v_steep = np.array([-np.sin(ALPHA), np.cos(ALPHA)])
# Hessiano = FIM = suma de proyecciones.
H = LAM_FLAT * np.outer(v_flat, v_flat) + LAM_STEEP * np.outer(v_steep, v_steep)


def cost(X, Y):
    return 0.5 * (H[0, 0] * X**2 + 2 * H[0, 1] * X * Y + H[1, 1] * Y**2)


fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2))

# --- Panel 1: contorno del costo + elipse de confianza + direccion plana --------
g = np.linspace(-7, 7, 400)
X, Y = np.meshgrid(g, g)
C = cost(X, Y)
cf = ax[0].contourf(X, Y, C, levels=25, cmap="viridis")
ax[0].contour(X, Y, C, levels=12, colors="white", linewidths=0.4, alpha=0.5)
fig.colorbar(cf, ax=ax[0], fraction=0.046, label="costo = − log verosimilitud")

# Elipse de confianza: eje largo = direccion plana (~ CRB grande).
w_flat = 2 * np.sqrt(CHI2 / LAM_FLAT)     # diametro a lo largo de la dir. plana
h_steep = 2 * np.sqrt(CHI2 / LAM_STEEP)   # diametro a lo largo de la dir. empinada
el = Ellipse((0, 0), w_flat, h_steep, angle=np.rad2deg(ALPHA),
             fill=False, edgecolor="#ff3b3b", lw=2.2, ls="-")
ax[0].add_patch(el)

# Flecha sobre la direccion plana + "soluciones casi equivalentes" a lo largo del valle.
L = 5.2
ax[0].annotate("", xy=(L * v_flat[0], L * v_flat[1]),
               xytext=(-L * v_flat[0], -L * v_flat[1]),
               arrowprops=dict(arrowstyle="<->", color="#ff3b3b", lw=2))
ax[0].text(0.0, 3.1, "dirección PLANA (≈ wII)", color="#ff3b3b", fontsize=11,
           fontweight="bold", ha="center",
           bbox=dict(boxstyle="round", fc="#ffffffcc", ec="#ff3b3b", lw=1))
for s in (-4.2, -2.1, 2.1, 4.2):
    ax[0].plot(s * v_flat[0], s * v_flat[1], "o", color="#ffd400", ms=7, mec="k", mew=0.6)
ax[0].plot(0, 0, "*", color="white", ms=16, mec="k", mew=0.8)
ax[0].text(-6.6, 6.0, "los puntos amarillos ajustan\ncasi igual de bien → wII ambiguo",
           fontsize=9, color="white",
           bbox=dict(boxstyle="round", fc="#00000088", ec="none"))
ax[0].set_xlabel("parámetro (combinación 1)")
ax[0].set_ylabel("parámetro (combinación 2)")
ax[0].set_title("Valle de la verosimilitud: plano a lo largo de wII")
ax[0].set_aspect("equal"); ax[0].set_xlim(-7, 7); ax[0].set_ylim(-7, 7)

# --- Panel 2: perfil de verosimilitud a lo largo de cada direccion --------------
t = np.linspace(-5, 5, 300)
ax[1].plot(t, 0.5 * LAM_STEEP * t**2, color="#1f4e79", lw=2.4,
           label="dir. bien condicionada (σ grande)")
ax[1].plot(t, 0.5 * LAM_FLAT * t**2, color="#ff3b3b", lw=2.4,
           label="dir. plana ≈ wII (σ chico)")
ax[1].axhline(0.5, ls=":", color="#888", lw=1)
ax[1].text(0.1, 0.56, "umbral de confianza", fontsize=8, color="#666")
ax[1].annotate("mínimo neto\n→ identificable", xy=(1.0, 0.5 * LAM_STEEP * 1.0**2),
               xytext=(1.4, 4.2), fontsize=9, color="#1f4e79",
               arrowprops=dict(arrowstyle="->", color="#1f4e79"))
ax[1].annotate("casi chato\n→ NO identificable", xy=(3.5, 0.5 * LAM_FLAT * 3.5**2),
               xytext=(-4.6, 3.2), fontsize=9, color="#ff3b3b",
               arrowprops=dict(arrowstyle="->", color="#ff3b3b"))
ax[1].set_xlabel("desplazamiento a lo largo de la dirección")
ax[1].set_ylabel("costo = − log verosimilitud")
ax[1].set_title("Perfil de verosimilitud (profile likelihood)")
ax[1].set_ylim(-0.2, 6.5); ax[1].grid(True, alpha=0.3); ax[1].legend(fontsize=9, loc="upper center")

fig.suptitle("¿Por qué wII no se identifica? La FIM tiene una dirección de autovalor chico (dirección plana)",
             fontsize=12, y=1.0)
fig.tight_layout()
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=130, bbox_inches="tight"); plt.close(fig)
print(f"Figura: {FIG}")
print(f"  cond(FIM) = lambda_steep/lambda_flat = {LAM_STEEP/LAM_FLAT:.1f}")
print(f"  eje largo de la elipse (CRB) = {np.sqrt(CHI2/LAM_FLAT):.2f}  vs corto = {np.sqrt(CHI2/LAM_STEEP):.2f}")
