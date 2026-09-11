#!/usr/bin/env python3
# Prueba rapida de las dos variantes nuevas, sin entrenar de verdad:
#   Sg = forma exacta + red          (eje 2)
#   H  = red con historia de comando (eje 1.2)
#   K  = la historia filtrada por un FIR aprendido (eje 1.3)
# Comprueba que se construyen, que el gradiente llega a donde tiene que llegar
# y que el rollout de evaluacion corre.
#
#   python scripts/test_variantes_nuevas.py

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from src.neural_ode.graybox_train import (PHYS, TrainConfig, build_model,
                                          make_windows, apilar_historia)
from src.neural_ode.integrate import rollout
from esc_eval import _rollout_traj

K = 16          # historia corta: la prueba es de cableado, no de fisica
W = 20
DT = 0.05


def datos_falsos(n=3, T=200, seed=0):
    rng = np.random.default_rng(seed)
    return {k: rng.standard_normal((n, T)).astype(np.float32) * 0.1
            for k in ("I", "E", "P", "Q")}


def probar(variant, hist=0):
    d = datos_falsos()
    cfg = TrainConfig(variant=variant, window=W, hist=hist, epochs=1)
    torch.manual_seed(0)
    m = build_model(cfg)

    x0, Pw, Qw, tgt = make_windows(d["I"], d["E"], d["P"], d["Q"], W, hist=hist)
    esperado = hist if hist else 1
    assert Pw.shape[-1] == esperado, f"{variant}: Pw tiene {Pw.shape[-1]} canales"

    # La ultima capa de g arranca en cero, asi que en el primer backward
    # ninguna capa anterior recibe gradiente y el chequeo de cableado no
    # distinguiria un parametro huerfano de uno sano. Se la despierta.
    if m.use_correction:
        with torch.no_grad():
            list(m.g.parameters())[-2].normal_(0.0, 0.1)

    pred = rollout(m, x0, Pw, Qw, DT)
    assert pred.shape == tgt.shape, f"{variant}: {pred.shape} vs {tgt.shape}"

    loss = ((pred - tgt) ** 2).mean()
    loss.backward()

    # El gradiente tiene que llegar al backbone y, si hay red, tambien a ella.
    assert m.raw_w.grad is not None and m.raw_w.grad.abs().sum() > 0, \
        f"{variant}: el backbone no recibe gradiente"
    if m.use_correction:
        # Por parametro y no sumado. Con el FIR de la variante K fuera del
        # optimizador la suma daba positiva igual, porque la MLP si recibia
        # gradiente, y la corrida terminaba con el filtro en su inicializacion
        # aleatoria las 1500 epocas.
        for n, q in m.g.named_parameters():
            assert q.grad is not None and float(q.grad.abs().sum()) > 0, \
                f"{variant}: g.{n} no recibe gradiente"
    if m.structured:
        assert m.raw_r_i.grad is not None, f"{variant}: r_i no recibe gradiente"

    # El camino de evaluacion arma la historia por su cuenta. Va bajo no_grad
    # porque asi lo llama nrmse_test.
    with torch.no_grad():
        traj = _rollout_traj(m, 0.1, 0.1, d["P"][0], d["Q"][0], DT)
    assert traj.shape == (len(d["P"][0]), 2), f"{variant}: rollout {traj.shape}"

    # Mismo criterio que el guarda de fit(): lo que es entrenable tiene que
    # estar en algun grupo del optimizador.
    conocidos = {id(m.raw_w)} | {id(getattr(m, f"raw_{k}")) for k in PHYS}
    if m.use_correction:
        conocidos |= {id(q) for q in m.g.parameters()}
    if m.structured:
        conocidos |= {id(m.raw_r_i), id(m.raw_r_e), id(m.raw_alpha)}
    sueltos = [n for n, q in m.named_parameters()
               if q.requires_grad and id(q) not in conocidos]
    assert not sueltos, f"{variant}: parametros fuera del optimizador: {sueltos}"

    n_g = sum(p.numel() for p in m.g.parameters()) if m.use_correction else 0
    print(f"  {variant:4} hist={hist:<3} Pw{tuple(Pw.shape)}  "
          f"pesos de red {n_g:5}  loss {float(loss.detach()):.4f}  ok")


def main():
    # apilar_historia: el retardo k tiene que ser la senal corrida k pasos.
    u = np.arange(12, dtype=np.float32).reshape(1, 12)
    h = apilar_historia(u, 4)
    assert h[0, 5, 0] == 5 and h[0, 5, 3] == 2, "apilar_historia mal alineada"
    assert h[0, 1, 3] == 0, "apilar_historia deberia rellenar con cero"
    print("  apilar_historia: alineacion y relleno ok")

    for v, h in (("whitebox", 0), ("B", 0), ("S", 0), ("Sg", 0),
                 ("H", K), ("K", K)):
        probar(v, h)
    print("\ntodo bien")


if __name__ == "__main__":
    main()
