#!/bin/bash
# Las corridas del poster, otra vez con ruido de observacion.
#
# Todo lo que el poster compara (white-box, correccion ciega, forma exacta y
# estado de filtro) corrio con noise_std = 0.0. Se repite con sigma = 0.05 y
# 0.01, con el protocolo de ruido del proyecto: media movil de 7 muestras sobre
# I y E.
#
# La forma exacta arranca del white-box del MISMO nivel de ruido. Arrancar del
# limpio le pasaria informacion de los datos sin ruido.
#
# act1 con sigma = 0.05 y white-box ya existe como e6_wb_n05_s7, con la misma
# configuracion, y no se repite.
#
# Cuatro carriles de un hilo cada uno. En esta maquina (4 nucleos fisicos)
# rinde 2.9 veces lo que una corrida de 4 hilos; ver scripts/bench_hilos.py.
# Las dependencias (la forma exacta necesita su white-box) quedan dentro de un
# mismo carril.
cd "$(dirname "$0")/.."
L=logs/escalado
R=results/escalado
mkdir -p $L
export WC_THREADS=1

corre() {  # tag, resto de argumentos
  local t=$1; shift
  python scripts/esc_run.py "$@" --smooth 7 --tag $t > $L/$t.log 2>&1
}

refrac() {  # nivel de ruido
  corre e7_e1wb_n$1   --data refrac1_n$1 --variant whitebox
  corre e7_e1B400_n$1 --data refrac1_n$1 --variant B --window 400
  corre e7_e1S2_n$1   --data refrac1_n$1 --variant S --r-init 0.05 \
                      --init-params $R/e7_e1wb_n$1.json
}

actuador() {  # nivel de ruido
  [ $1 = 01 ] && corre e7_e2wb_n$1 --data act1_n$1 --variant whitebox
  corre e7_e2B_n$1    --data act1_n$1 --variant B
  corre e7_e2lag_n$1  --data act1_n$1 --variant lag
}

refrac 05 &
actuador 05 &
refrac 01 &
actuador 01 &
wait

echo "POSTER CON RUIDO COMPLETO" > $L/DONE_poster_ruido
