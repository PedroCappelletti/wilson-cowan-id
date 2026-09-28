#!/bin/bash
# Las dos corridas con historia del eje 1, en secuencia. e2_A ya corrio.
cd "$(dirname "$0")/.."
L=logs/escalado
python scripts/esc_run.py --data act1 --variant H --hist 100 --tag e2_H100 > $L/e2_H100.log 2>&1
python scripts/esc_run.py --data act1 --variant H --hist 400 --tag e2_H400 > $L/e2_H400.log 2>&1
echo "EJE1 HIST COMPLETO" > $L/DONE_eje1_hist
