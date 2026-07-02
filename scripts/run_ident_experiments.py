#!/usr/bin/env python3
# =============================================================================
#  ORQUESTADOR — corre los experimentos de identificabilidad en secuencia
#  (A: fijar/regularizar, B: subset selection, C3: mezcla). Cada uno guarda su
#  JSON + figuras. Pensado para correr en background de un tiron.
#  USO:  python -u scripts/run_ident_experiments.py
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import scripts.exp_fix_regularize as A
import scripts.exp_subset_selection as B
import scripts.exp_mix_test as C3


def main():
    print("\n########## EXPERIMENTO A — fijar / regularizar wII ##########", flush=True)
    A.main()
    print("\n########## EXPERIMENTO B — subset selection (FIM) ##########", flush=True)
    B.main()
    print("\n########## EXPERIMENTO C3 — mezcla complementaria ##########", flush=True)
    C3.main()
    print("\n########## TODOS LOS EXPERIMENTOS TERMINADOS ##########", flush=True)


if __name__ == "__main__":
    main()
