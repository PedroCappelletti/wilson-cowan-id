#!/bin/bash
# Eje 1.3 y 1.4: el FIR aprendido y el barrido de capacidad, en secuencia.
# Referencia a batir: H400 con NRMSE 11.48 % y 26 mil pesos en la primera capa.
cd "$(dirname "$0")/.."
L=logs/escalado
mkdir -p $L

# 1.3 — mismo campo receptivo que H400, un orden de magnitud menos de parametros
python scripts/esc_run.py --data act1 --variant K --hist 400 --n-fir 4 \
       --tag e3_K400 > $L/e3_K400.log 2>&1

# 1.4 — las tres perillas, una por vez desde e3_K400
python scripts/esc_run.py --data act1 --variant K --hist 800 --n-fir 4 \
       --tag e3_K800 > $L/e3_K800.log 2>&1
python scripts/esc_run.py --data act1 --variant K --hist 400 --n-fir 4 --hidden 64 \
       --tag e3_K400_h64 > $L/e3_K400_h64.log 2>&1
python scripts/esc_run.py --data act1 --variant K --hist 400 --n-fir 8 \
       --tag e3_K400_f8 > $L/e3_K400_f8.log 2>&1

echo "EJE 1.3 + 1.4 COMPLETO" > $L/DONE_eje13
