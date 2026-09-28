#!/usr/bin/env python3
# Grafica el nucleo FIR aprendido por la variante K contra la exponencial del
# actuador. La comparacion es el resultado interpretable del eje 1.3: si el
# filtro decae con la constante de tiempo verdadera sin que se le haya dicho
# nada, la correccion descubrio la escala temporal de la fisica omitida.
#
#   python scripts/plot_fir_kernel.py e3_K400 [e3_K400_f8 ...] [--out nombre]
#
# Sin --out escribe fir_kernel.pdf, que es la figura del informe del 11-09
# (e3_K400 contra e3_K400_f8). Cualquier otra comparación tiene que ir con su
# propio nombre: una vez se pisó esa figura y el informe salió con el panel
# equivocado debajo de un pie que describía otra corrida.

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

RES = Path("results/escalado")
FIG = Path("results/figures")
DT = 0.05001250312578145      # ms por muestra, de act1.npz
TAU = 1.0                     # ms, la constante del actuador


def kernel_nulo(n_fir: int, K: int, seed: int = 0) -> np.ndarray:
    """El mismo filtro sin entrenar. Es el control que hace falta para decir
    si la cola del nucleo aprendido tiene estructura o es ruido: torch.nn
    inicializa Linear con U(-1/sqrt(2K), 1/sqrt(2K)), plano en el retardo."""
    g = torch.Generator().manual_seed(seed)
    lim = 1.0 / np.sqrt(2 * K)
    return (torch.rand((n_fir, K), generator=g) * 2 - 1).numpy() * lim


def kernels(tag: str):
    """(K_P, K_Q) de forma (n_fir, K), en orden de retardo creciente."""
    ck = torch.load(RES / "models" / f"{tag}.pt", map_location="cpu",
                    weights_only=False)
    w = ck["state"]["g.fir.weight"].numpy()
    K = ck["hist_len"]
    return w[:, :K], w[:, K:]


def energia(k: np.ndarray) -> np.ndarray:
    """Energia por retardo, sumada sobre canales y normalizada a su maximo.

    Los canales por separado no son comparables entre si: la MLP puede
    reescalar cualquiera sin cambiar la funcion. Lo invariante es cuanta
    energia total pone el filtro en cada retardo.
    """
    e = np.sqrt((k ** 2).sum(0))
    return e / e.max()


def tau_estimado(e: np.ndarray, t: np.ndarray, piso_desde: float = 10.0):
    """Constante de tiempo del nucleo, despues de restarle su piso.

    Los pesos de retardo largo son casi libres y forman un piso plano que no
    corresponde a ninguna fisica. Restarlo es lo que permite preguntar si lo
    que queda decae, y con que constante. El ajuste es una recta sobre el
    logaritmo, en el tramo donde la curva todavia esta por encima del piso.
    """
    piso = float(e[t >= piso_desde].mean())
    y = e - piso
    # Tiempo hasta 1/e del pico. No supone que el nucleo SEA exponencial, que
    # es justo lo que esta en duda, asi que es el descriptor primario.
    bajo = np.where(y < y[0] / np.e)[0]
    t_1e = float(t[bajo[0]]) if len(bajo) else float("nan")
    usable = (y > 0.15 * y[0]) & (t < piso_desde)
    if usable.sum() < 5:
        return float("nan"), piso, 0, t_1e
    c = np.polyfit(t[usable], np.log(y[usable]), 1)
    return float(-1.0 / c[0]), piso, int(usable.sum()), t_1e


def largo_efectivo(e: np.ndarray, t: np.ndarray) -> float:
    """Centroide temporal: el retardo medio ponderado por energia."""
    return float((t * e).sum() / e.sum())


def main(tags, nombre="fir_kernel"):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(tags), figsize=(6 * len(tags), 4.2),
                             squeeze=False, layout="constrained")
    resumen = {}

    for ax, tag in zip(axes[0], tags):
        kP, kQ = kernels(tag)
        t = np.arange(kP.shape[1]) * DT
        eP, eQ = energia(kP), energia(kQ)

        ax.plot(t, eP, lw=1.8, label="$P$, el canal excitado")
        ax.plot(t, eQ, lw=1.8, alpha=0.8, label="$Q$")
        eN = energia(kernel_nulo(*kP.shape))
        ax.plot(t, eN, color="0.7", lw=1.0, label="sin entrenar")
        ax.plot(t, np.exp(-t / TAU), "k--", lw=1.4,
                label=fr"$e^{{-t/\tau}}$, $\tau={TAU}$ ms")
        ax.set_xlim(0, min(20.0, t[-1]))
        ax.set_xlabel("retardo [ms]")
        ax.set_ylabel("energía del núcleo, normalizada")
        ax.set_title(f"{tag}  ({kP.shape[0]} canales, K={kP.shape[1]})")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

        resumen[tag] = {
            "n_fir": int(kP.shape[0]), "K": int(kP.shape[1]),
            "centroide_P_ms": largo_efectivo(eP, t),
            "centroide_Q_ms": largo_efectivo(eQ, t),
            "centroide_exponencial_ms": largo_efectivo(np.exp(-t / TAU), t),
            "energia_primeros_5ms_P": float(eP[t <= 5].sum() / eP.sum()),
            "energia_primeros_5ms_nulo": float(eN[t <= 5].sum() / eN.sum()),
            "energia_primeros_5ms_exp": float(
                np.exp(-t[t <= 5] / TAU).sum() / np.exp(-t / TAU).sum()),
            "cola_P_sobre_pico": float(eP[t > 10].mean() / eP[0]),
            "cola_nulo_sobre_pico": float(eN[t > 10].mean() / eN[0]),
        }
        tau_ap, piso, n, t1e = tau_estimado(eP, t)
        tau_nulo, _, _, t1e_nulo = tau_estimado(eN, t)
        resumen[tag].update(tau_P_ms=tau_ap, piso_P=piso, n_ajuste=n,
                            tau_nulo_ms=tau_nulo, t_1e_ms=t1e,
                            t_1e_nulo_ms=t1e_nulo)

    fig.savefig(FIG / f"{nombre}.pdf")
    fig.savefig(FIG / f"{nombre}.png", dpi=150)
    (RES / f"{nombre}.json").write_text(json.dumps(resumen, indent=2))
    for tag, r in resumen.items():
        print(f"{tag}: centroide {r['centroide_P_ms']:.2f} ms "
              f"(exponencial {r['centroide_exponencial_ms']:.2f})  |  "
              f"energía en los primeros 5 ms: aprendido "
              f"{r['energia_primeros_5ms_P']:.1%}, sin entrenar "
              f"{r['energia_primeros_5ms_nulo']:.1%}, exponencial "
              f"{r['energia_primeros_5ms_exp']:.1%}  |  cola/pico: "
              f"{r['cola_P_sobre_pico']:.2f} vs {r['cola_nulo_sobre_pico']:.2f}"
              f"  |  t hasta 1/e: {r['t_1e_ms']:.2f} ms "
              f"(verdadero {TAU}, sin entrenar {r['t_1e_nulo_ms']:.2f})  |  "
              f"ajuste exponencial {r['tau_P_ms']:.2f} ms, piso {r['piso_P']:.2f}")
    print(f"-> {FIG / (nombre + '.pdf')}")


if __name__ == "__main__":
    args = sys.argv[1:]
    nombre = "fir_kernel"
    if "--out" in args:
        i = args.index("--out")
        nombre = args[i + 1]
        del args[i:i + 2]
    main(args or ["e3_K400"], nombre)
