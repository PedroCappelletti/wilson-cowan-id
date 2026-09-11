#!/bin/bash
# Eje 1 sobre act1, SECUENCIAL a proposito: en paralelo el campo `minutos` mide
# contencion y deja de servir como costo, que es lo que paso en agosto.
cd "$(dirname "$0")/.."
L=logs/escalado
python scripts/esc_run.py --data act1 --variant A              --tag e2_A     > $L/e2_A.log 2>&1
python scripts/esc_run.py --data act1 --variant H --hist 100   --tag e2_H100  > $L/e2_H100.log 2>&1
python scripts/esc_run.py --data act1 --variant H --hist 400   --tag e2_H400  > $L/e2_H400.log 2>&1
echo "EJE1 COMPLETO" > $L/DONE_eje1
