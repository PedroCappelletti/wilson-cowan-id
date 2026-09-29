#!/usr/bin/env python3
# Figura de tau, el núcleo y la rugosidad (tau-nucleo-y-rugosidad.png) y los
# números del ejemplo del zigzag que cita el complemento: un núcleo suave y el
# mismo con un zigzag sumado, aplicados a un comando PRBS del dataset.
#
#   python scripts/explicar_nucleo.py
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load(Path(__file__).resolve().parents[1] / "data" / "processed" / "uncertain" / "act1.npz",
            allow_pickle=True)
dt, tau, K = float(d["dt"]), 1.0, 400
k = np.arange(K)
t_k = k * dt

# nucleo verdadero del actuador (respuesta al impulso discreta)
h = (dt / tau) * np.exp(-t_k / tau)
# el mismo nucleo con un zigzag encima: +a, -a, +a, ...
a = 0.6 * h[0]
h_rug = h + a * (-1.0) ** k


def rugosidad(w):
    return float(np.sum(np.diff(w) ** 2))


def filtrar(w, u):
    return np.convolve(u, w)[:len(u)]


labels = [str(x) for x in d["labels"]]
s = next(i for i, l in enumerate(labels) if l.startswith("prbs"))
u = d["P"][s]
y, y_rug = filtrar(h, u), filtrar(h_rug, u)
dif = np.sqrt(np.mean((y - y_rug) ** 2)) / np.sqrt(np.mean(y ** 2))

fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
ax = axs.ravel()

# 1. tau: respuesta a un escalon
t = np.linspace(0, 6, 400)
ax[0].plot(t, 1 - np.exp(-t / tau), lw=2, color="#1F4E79")
ax[0].axhline(1, color="0.6", lw=0.8, ls=":")
for n, txt in ((1, "63 %"), (3, "95 %")):
    v = 1 - np.exp(-n)
    ax[0].plot([n, n], [0, v], color="0.5", lw=0.8, ls="--")
    ax[0].annotate(f"{n}τ: {txt}", (n, v), xytext=(n + 0.15, v - 0.12), fontsize=10)
ax[0].set_xlabel("tiempo desde el escalón del comando [ms]")
ax[0].set_ylabel("estímulo que llega a la población")
ax[0].set_title("1. τ: qué tan lento sigue el actuador al comando")

# 2. el nucleo
ax[1].plot(t_k, h / h[0], lw=2, color="#1F4E79")
ax[1].plot([tau, tau], [0, np.exp(-1)], color="0.5", lw=0.8, ls="--")
ax[1].annotate("τ: 37 % del peso", (tau, np.exp(-1)), xytext=(1.3, 0.45), fontsize=10)
ax[1].set_xlim(0, 6)
ax[1].set_xlabel("antigüedad del comando [ms]")
ax[1].set_ylabel("peso relativo")
ax[1].set_title("2. El núcleo: cuánto pesa cada instante pasado")

# 3. dos nucleos
ax[2].plot(t_k[:60], h_rug[:60], color="#C0504D", lw=1.1, label="en zigzag")
ax[2].plot(t_k[:60], h[:60], color="#1F4E79", lw=2.4, label="suave")
ax[2].set_xlabel("antigüedad del comando [ms]")
ax[2].set_ylabel("peso")
ax[2].set_title("3. Dos núcleos muy distintos")
ax[2].legend(frameon=False)

# 4. casi la misma salida
seg = slice(1000, 1600)
tt = np.arange(len(u))[seg] * dt
ax[3].plot(tt, y[seg], lw=2.4, color="#1F4E79", label="con el núcleo suave")
ax[3].plot(tt, y_rug[seg], lw=1.2, color="#C0504D", ls="--", label="con el núcleo en zigzag")
ax[3].set_ylim(-0.1, 1.35)
ax[3].set_xlabel("tiempo [ms]")
ax[3].set_ylabel("salida del filtro")
ax[3].set_title("4. Casi la misma salida sobre un comando real")
ax[3].legend(frameon=False, loc="upper center", ncol=2)

for x in ax:
    x.spines[["top", "right"]].set_visible(False)

out = Path(__file__).resolve().parents[1] / "results" / "figures" / "tau-nucleo-y-rugosidad.png"
fig.savefig(out, dpi=130)

print(f"escenario: {labels[s]}")
print(f"diferencia entre las salidas: {dif:.2%} del tamaño de la salida")
print(f"los nucleos difieren en {np.sqrt(np.mean((h - h_rug) ** 2)) / np.sqrt(np.mean(h ** 2)):.0%}")
print(f"rugosidad suave {rugosidad(h):.2e}  zigzag {rugosidad(h_rug):.2e}  "
      f"(x{rugosidad(h_rug) / rugosidad(h):.0f})")
print(f"L2 suave {np.sum(h**2):.2e}  zigzag {np.sum(h_rug**2):.2e}")
