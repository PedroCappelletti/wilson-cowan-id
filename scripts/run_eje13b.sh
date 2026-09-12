#!/bin/bash
# Se encola detras del eje 2 y 6. Prueba la regularizacion del nucleo FIR.
#
# Motivo: el nucleo de e3_K400 concentro energia en el retardo cero (cola sobre
# pico 1.27 -> 0.20) pero dejo 66 % de la energia mas alla de los 5 ms, contra
# 0.7 % de la exponencial verdadera. Los pesos de retardo largo quedan casi
# libres porque la perdida depende poco de ellos. Si esa componente arbitraria
# es lo que hunde el R2, penalizarla lo tiene que levantar.
cd "$(dirname "$0")/.."
L=logs/escalado
mkdir -p $L

for i in $(seq 1 600); do
  [ -f $L/DONE_eje6 ] && break
  sleep 60
done

for wd in 1e-4 1e-3; do
  t=e5_K400_wd${wd}
  python scripts/esc_run.py --data act1 --variant K --hist 400 --n-fir 4 \
         --wd-fir $wd --tag $t > $L/$t.log 2>&1
done

echo "REGULARIZACION DEL NUCLEO COMPLETA" > $L/DONE_eje13b
