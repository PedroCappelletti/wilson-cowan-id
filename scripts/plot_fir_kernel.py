#!/usr/bin/env python3
# Grafica el nucleo FIR aprendido por la variante K contra la exponencial del
# actuador. La comparacion es el resultado interpretable del eje 1.3: si el
# filtro decae con la constante de tiempo verdadera sin que se le haya dicho
# nada, la correccion descubrio la escala temporal de la fisica omitida.
#
#   python scripts/plot_fir_kernel.py e3_K400 [e3_K400_f8 ...]

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


def largo_efectivo(e: np.ndarray, t: np.ndarray) -> float:
    """Centroide temporal: el retardo medio ponderado por energia."""
    return float((t * e).sum() / e.sum())


def main(tags):
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

    fig.savefig(FIG / "fir_kernel.pdf")
    fig.savefig(FIG / "fir_kernel.png", dpi=150)
    (RES / "fir_kernel.json").write_text(json.dumps(resumen, indent=2))
    for tag, r in resumen.items():
        print(f"{tag}: centroide {r['centroide_P_ms']:.2f} ms "
              f"(exponencial {r['centroide_exponencial_ms']:.2f})  |  "
              f"energía en los primeros 5 ms: aprendido "
              f"{r['energia_primeros_5ms_P']:.1%}, sin entrenar "
              f"{r['energia_primeros_5ms_nulo']:.1%}, exponencial "
              f"{r['energia_primeros_5ms_exp']:.1%}  |  cola/pico: "
              f"{r['cola_P_sobre_pico']:.2f} vs {r['cola_nulo_sobre_pico']:.2f}")
    print(f"-> {FIG / 'fir_kernel.pdf'}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["e3_K400"])
