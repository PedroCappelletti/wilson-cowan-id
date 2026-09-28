#!/usr/bin/env python3
# Cuántas corridas en paralelo y con cuántos hilos rinden más en esta máquina.
#
# Cada proceso entrena N épocas de Adam de una corrida típica y reporta cuánto
# tardó. El lanzador corre k procesos a la vez con t hilos cada uno y mide
# épocas por minuto en total, que es lo que importa para un lote.
#
#   python scripts/bench_hilos.py            # corre todas las configuraciones
#   python scripts/bench_hilos.py --hijo     # uso interno

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
EPOCAS = 25
CONFIGS = [(1, 4), (2, 2), (4, 1), (4, 2)]   # (corridas a la vez, hilos cada una)


def hijo():
    import torch
    torch.set_num_threads(int(os.environ["WC_THREADS"]))
    sys.path.insert(0, str(RAIZ))
    from src.neural_ode.graybox_train import (TrainConfig, build_model,
                                              load_split, make_windows)
    from src.neural_ode.integrate import rollout

    torch.manual_seed(0)
    d = load_split(RAIZ / "data/processed/uncertain/act1_n05.npz")
    m = build_model(TrainConfig(variant="B"))
    x0, Pw, Qw, tgt = make_windows(d["I"], d["E"], d["P"], d["Q"], 100)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    t0 = time.perf_counter()
    for _ in range(EPOCAS):
        opt.zero_grad()
        loss = ((rollout(m, x0, Pw, Qw, d["dt"]) - tgt) ** 2).mean()
        loss.backward()
        opt.step()
    print(f"{time.perf_counter() - t0:.2f}")


def lanzador():
    base = None
    for k, t in CONFIGS:
        env = {**os.environ, "WC_THREADS": str(t)}
        t0 = time.perf_counter()
        ps = [subprocess.Popen([sys.executable, __file__, "--hijo"], env=env,
                               stdout=subprocess.PIPE, text=True) for _ in range(k)]
        for p in ps:
            p.wait()
        pared = time.perf_counter() - t0
        tasa = k * EPOCAS / pared * 60
        base = base or tasa
        print(f"{k} corrida(s) x {t} hilo(s): {pared:6.1f} s de reloj, "
              f"{tasa:6.1f} épocas/min en total  (x{tasa / base:.2f})", flush=True)


if __name__ == "__main__":
    hijo() if "--hijo" in sys.argv else lanzador()
