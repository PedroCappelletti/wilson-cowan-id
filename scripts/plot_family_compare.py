#!/usr/bin/env python3
# =============================================================================
#  COMPARACION dt grande vs dt fino — identificacion NODE por familia
#  (solo grafica, lee los dos JSON ya generados; NO entrena)
# =============================================================================
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import matplotlib.pyplot as plt

G = json.loads(Path("results/ident_family_neural_ode_dtgrande.json").read_text(encoding="utf-8")) \
    if Path("results/ident_family_neural_ode_dtgrande.json").exists() \
    else json.loads(Path("results/ident_family_neural_ode.json").read_text(encoding="utf-8"))
F = json.loads(Path("results/ident_family_neural_ode_dtfino.json").read_text(encoding="utf-8"))

fams = [n for n in G["results"] if n in F["results"]]
crb = G["crb_wII_pred"]
wII_g = [G["results"][n]["err_wII"] for n in fams]
wII_f = [F["results"][n]["err_wII"] for n in fams]

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
x = np.arange(len(fams)); w = 0.38
ax[0].bar(x - w/2, wII_g, w, label="dt grande (0.2)", color="#ff7f0e")
ax[0].bar(x + w/2, wII_f, w, label="dt fino (0.05)", color="#1f4e79")
ax[0].axhline(10, ls=":", color="#2ca02c", label="10% (referencia)")
ax[0].set_yscale("log")
ax[0].set_xticks(x); ax[0].set_xticklabels(fams, rotation=45, ha="right")
ax[0].set_ylabel("error de wII (%) — escala log")
ax[0].set_title("Error de wII por familia: dt grande vs dt fino")
ax[0].legend(fontsize=8); ax[0].grid(True, axis="y", which="both", alpha=0.3)

# CRB predicha (Exp C) vs error empirico a dt fino — ¿hay correlacion?
cx = [crb[n] for n in fams]
ax[1].scatter(cx, wII_f, color="#1f4e79", label="dt fino")
ax[1].scatter(cx, wII_g, color="#ff7f0e", marker="^", alpha=0.6, label="dt grande")
for n, xx, yy in zip(fams, cx, wII_f):
    ax[1].annotate(n, (xx, yy), fontsize=8, xytext=(3, 3), textcoords="offset points")
ax[1].set_xlabel("CRB(wII) predicha — Exp C (menor = mejor)")
ax[1].set_ylabel("error empírico de wII (%)")
ax[1].set_yscale("log")
ax[1].set_title("¿Confirma la NODE la predicción CRB? (no concluyente)")
ax[1].legend(fontsize=8); ax[1].grid(True, which="both", alpha=0.3)

fig.tight_layout()
out = Path("results/figures/family_compare_dt.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120); plt.close(fig)
print(f"Figura: {out}")

# Tabla resumen a consola.
print(f"\n{'familia':12} {'dtGrande':>10} {'dtFino':>10} {'CRB pred':>10}")
for n in fams:
    print(f"{n:12} {G['results'][n]['err_wII']:9.1f}% {F['results'][n]['err_wII']:9.1f}% {crb[n]:10.2f}")
