#!/usr/bin/env python3
"""Chequea el camino de datos reales.

c_P estaba declarada como parametro y reportada en cada log, pero no entraba en
la perdida: se entrenaba con P=u y se evaluaba con P=3u. Este test es para que
no vuelva a pasar en silencio.

  python scripts/test_datos_reales.py
"""
import sys
from pathlib import Path

import torch

R = Path(r"C:\Users\User\Desktop\wilson-cowan-id")
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "scripts"))

import train_real_output as tro

fs, us, ss = tro.load(125)
dt = 1.0 / fs

for variant, hist in (("v0", 0), ("B", 0), ("H", 32)):
    torch.manual_seed(0)
    m = tro.OutputModel(variant, hist=hist)
    Pw, Sw, meta = tro.build_windows([0], us, ss, 96, hist=hist)
    X0 = torch.nn.Parameter(torch.zeros(len(meta), 2))
    Qw = torch.zeros_like(Pw)

    traj = tro.rollout(m.wc, X0, m.c_P * Pw, Qw, dt)
    loss = ((tro.demean(m.y_of(traj)) - tro.demean(Sw)) ** 2).mean()
    loss.backward()

    gp = m.c_P.grad
    go = m.c_out.grad
    assert gp is not None and float(gp.abs()) > 0, f"{variant}: c_P sin gradiente"
    assert go is not None and float(go.abs()) > 0, f"{variant}: c_out sin gradiente"
    print(f"  {variant:3} hist={hist:<3} Pw{tuple(Pw.shape)}  "
          f"grad c_P={float(gp):+.3e}  grad c_out={float(go):+.3e}  ok")

print("\nc_P ya entra en la perdida")
