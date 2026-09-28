#!/usr/bin/env python3
# Qué relación señal a ruido tienen los datos reales, y cuál la de los datasets
# con ruido del escalado, medidas de la misma forma.
#
# Real: dos estimaciones independientes.
#   - Repeticiones. Si los tres estímulos son el mismo chirp, la parte de s que
#     se repite entre grabaciones es la respuesta y lo que no se repite es ruido.
#   - Coherencia entre u y s. La fracción de potencia de s explicada linealmente
#     por u, frecuencia por frecuencia. Cuenta como ruido cualquier respuesta no
#     lineal, así que da una cota pesimista.
#
# Simulado: el ruido blanco se agrega por muestra a 20 kHz, y la mayor parte de
# su potencia cae fuera de la banda donde vive la señal. Por eso se reporta
# también la SNR dentro de banda, que es la comparable con una grabación ya
# filtrada.

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import coherence, welch

RAIZ = Path(__file__).resolve().parents[1]


def db(x):
    return 10 * np.log10(x)


def real():
    d = np.load(RAIZ / "data/processed/real/data8_fs250.npz", allow_pickle=True)
    fs = float(d["fs"])
    u, s = list(d["u"]), list(d["s"])
    n = min(len(x) for x in s)
    U = np.stack([x[:n] for x in u])
    S = np.stack([x[:n] for x in s])
    print(f"real: {len(S)} grabaciones, fs = {fs:.0f} Hz, {n / fs:.1f} s en común")

    cu = np.corrcoef(U)
    print(f"  correlación entre estímulos: {cu[0,1]:.3f} {cu[0,2]:.3f} {cu[1,2]:.3f}")

    media = S.mean(0)
    resid = S - media
    # la media de N repeticiones todavía lleva ruido/N en varianza: se corrige
    k = len(S)
    p_ruido = resid.var() * k / (k - 1)
    p_senal = media.var() - p_ruido / k
    print(f"  por repeticiones: SNR = {db(p_senal / p_ruido):.1f} dB")

    for i in range(k):
        f, c = coherence(U[i], S[i], fs=fs, nperseg=int(4 * fs))
        banda = (f >= 1) & (f <= 10)
        cm = float(np.clip(c[banda].mean(), 1e-6, 1 - 1e-6))
        print(f"  grabación {i + 1}, coherencia media 1-10 Hz = {cm:.2f} "
              f"-> SNR lineal {db(cm / (1 - cm)):.1f} dB")


def simulado(nombre, k_suave=7):
    c = np.load(RAIZ / f"data/processed/uncertain/{nombre}.npz", allow_pickle=True)
    fs = 1000.0 / float(c["dt"])        # dt en ms
    E = c["E"]
    for n in ("01", "05"):
        r = np.load(RAIZ / f"data/processed/uncertain/{nombre}_n{n}.npz",
                    allow_pickle=True)
        ruido = r["E"] - E
        banda_sin = db(E.var() / ruido.var())

        # banda de la señal: donde está el 99 % de su potencia
        f, pe = welch(E, fs=fs, nperseg=1024, axis=-1)
        pe = pe.mean(0)
        fc = f[np.searchsorted(np.cumsum(pe) / pe.sum(), 0.99)]
        _, pr = welch(ruido, fs=fs, nperseg=1024, axis=-1)
        pr = pr.mean(0)
        en_banda = db(pe[f <= fc].sum() / pr[f <= fc].sum())

        ker = np.ones(k_suave) / k_suave
        rs = np.stack([np.convolve(x, ker, mode="same") for x in ruido])
        suave = db(E.var() / rs.var())
        print(f"  {nombre}_n{n}: SNR por muestra {banda_sin:5.1f} dB, "
              f"tras media móvil de {k_suave} {suave:5.1f} dB, "
              f"dentro de la banda de la señal (hasta {fc:.0f} Hz) {en_banda:5.1f} dB")


if __name__ == "__main__":
    real()
    print("simulado:")
    simulado("act1")
    simulado("refrac1")
