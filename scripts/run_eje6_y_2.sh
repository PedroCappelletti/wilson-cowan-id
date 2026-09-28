#!/bin/bash
# Se encola detras del eje 1.3/1.4 y corre lo que sigue: el eje 2 (ambiguedad),
# la verificacion de refrac1, y el eje 6 (ruido de observacion).
#
# En secuencia y nunca en paralelo: la bitacora ya registra que cuatro carriles
# sobre ocho cores convirtieron corridas de 25 min en corridas de 25 h.
cd "$(dirname "$0")/.."
L=logs/escalado
mkdir -p $L

# Espera a que termine el lote anterior (hasta 5 h).
for i in $(seq 1 300); do
  [ -f $L/DONE_eje13 ] && break
  sleep 60
done

# --- Eje 2: forma exacta + red sobre refrac1, donde la forma esta completa.
# Si la red no aprende nada, g_rms queda cerca de cero y el R2 no se mueve.
python scripts/esc_run.py --data refrac1 --variant Sg --r-init 0.05 \
       --init-params results/escalado/e1_wb.json \
       --tag e4_Sg > $L/e4_Sg.log 2>&1

# --- Verificacion de refrac1: tiene que reproducir NRMSE 1.70 y R2 0.964.
python scripts/esc_run.py --data refrac1 --variant S --r-init 0.05 \
       --init-params results/escalado/e1_wb.json \
       --tag e4_S2ver > $L/e4_S2ver.log 2>&1

# --- Eje 6: el ruido. El white-box primero, si no las degradaciones no tienen
# contra que medirse.
python scripts/esc_run.py --data act1_n05 --variant whitebox \
       --tag e4_wb_n05 > $L/e4_wb_n05.log 2>&1
python scripts/esc_run.py --data act1_n05 --variant K --hist 400 --n-fir 4 \
       --tag e4_K400_n05 > $L/e4_K400_n05.log 2>&1
python scripts/esc_run.py --data act1_n05 --variant H --hist 400 \
       --tag e4_H400_n05 > $L/e4_H400_n05.log 2>&1
python scripts/esc_run.py --data act1_n01 --variant K --hist 400 --n-fir 4 \
       --tag e4_K400_n01 > $L/e4_K400_n01.log 2>&1
python scripts/esc_run.py --data act1_n01 --variant H --hist 400 \
       --tag e4_H400_n01 > $L/e4_H400_n01.log 2>&1

echo "EJE 2 + 6 COMPLETO" > $L/DONE_eje6
