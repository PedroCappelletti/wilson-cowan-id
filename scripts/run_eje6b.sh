#!/bin/bash
# Se encola detras de la regularizacion del nucleo. El eje 6 otra vez, ahora
# con la mitigacion puesta.
#
# Motivo: con sigma = 0.05 y sin defensa, las dos variantes con memoria pierden
# contra el white-box sobre los mismos datos, asi que la correccion pasa de
# aportar a restar. Pero el pipeline del escalado no aplicaba ninguna
# mitigacion, mientras que la linea del barrido de sigma si la tenia y era lo
# que hacia la diferencia. Sin esta tanda, la conclusion "el gray-box no tolera
# ruido" no esta justificada: lo medido es el gray-box sin defensa.
#
# Ventana de 7 muestras, que es la que usaba la linea anterior hasta sigma=0.05.
cd "$(dirname "$0")/.."
L=logs/escalado
mkdir -p $L

for i in $(seq 1 900); do
  [ -f $L/DONE_eje13b ] && break
  sleep 60
done

python scripts/esc_run.py --data act1_n05 --variant whitebox --smooth 7 \
       --tag e6_wb_n05_s7 > $L/e6_wb_n05_s7.log 2>&1
python scripts/esc_run.py --data act1_n05 --variant K --hist 400 --n-fir 4 \
       --smooth 7 --tag e6_K400_n05_s7 > $L/e6_K400_n05_s7.log 2>&1
python scripts/esc_run.py --data act1_n05 --variant H --hist 400 --smooth 7 \
       --tag e6_H400_n05_s7 > $L/e6_H400_n05_s7.log 2>&1

echo "EJE 6 CON SUAVIZADO COMPLETO" > $L/DONE_eje6b
