"""Ruido de observación para el piloto de S. EN EVALUACIÓN: ver README.md.

Hoy los datasets del escalado llevan ruido blanco gaussiano por muestra, sumado a
20 kHz. Casi toda su potencia cae fuera de la banda donde vive la señal, y por
eso dentro de banda queda en 27 y 41 dB de SNR, contra una cota de unos −7 dB en
las grabaciones reales (scripts/snr_real_vs_sim.py). El ruido de un registro de
LFP no es blanco: tiene un espectro aperiódico que cae como 1/f^χ.
"""

from __future__ import annotations

import numpy as np


def coloreado(n_tray: int, T: int, dt_ms: float, chi: float, rng,
              f_min_hz: float = 0.5) -> np.ndarray:
    """Ruido gaussiano con densidad espectral ∝ 1/f^chi, varianza unitaria por
    trayectoria. chi = 0 es blanco, 1 rosa, 2 marrón.

    Se le da forma al espectro de un ruido blanco: la amplitud de cada
    frecuencia se multiplica por f^(-chi/2). Debajo de f_min_hz la ganancia
    queda plana, porque si no, la componente casi continua se lleva toda la
    varianza con 200 ms de trayectoria."""
    blanco = rng.standard_normal((n_tray, T))
    X = np.fft.rfft(blanco, axis=1)
    f = np.fft.rfftfreq(T, d=dt_ms / 1000.0)
    g = np.maximum(f, f_min_hz) ** (-chi / 2.0)
    g[0] = 0.0
    x = np.fft.irfft(X * g, n=T, axis=1)
    return x / x.std(axis=1, keepdims=True)


def agregar(S: np.ndarray, snr_db: float, chi: float, dt_ms: float,
            seed: int = 7) -> np.ndarray:
    """Suma a S (n_tray, T) ruido 1/f^chi con la SNR pedida, en potencia total
    por trayectoria. La SNR se fija contra la varianza de S de cada trayectoria
    y no contra una global, para que un escenario de baja actividad no quede
    tapado por el ruido calibrado para uno de alta."""
    rng = np.random.default_rng(seed)
    r = coloreado(*S.shape, dt_ms=dt_ms, chi=chi, rng=rng)
    sigma = S.std(axis=1, keepdims=True) * 10 ** (-snr_db / 20.0)
    return S + sigma * r
