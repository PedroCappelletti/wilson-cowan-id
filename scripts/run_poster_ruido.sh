#!/bin/bash
# Las corridas del poster, otra vez con ruido de observacion.
#
# Todo lo que el poster compara (white-box, correccion ciega, forma exacta y
# estado de filtro) corrio con noise_std = 0.0. Se repite con sigma = 0.05 y
# 0.01, con el protocolo de ruido del proyecto: media movil de 7 muestras sobre
# I y E. Primero 0.05, que es el nivel donde el eje 6 vio cambiar la conclusion.
#
# La forma exacta arranca del white-box del MISMO nivel de ruido. Arrancar del
# limpio le pasaria informacion de los datos sin ruido.
#
# act1 con sigma = 0.05 y white-box ya existe como e6_wb_n05_s7, con la misma
# configuracion, y no se repite.
cd "$(dirname "$0")/.."
L=logs/escalado
R=results/escalado
mkdir -p $L

corre() {  # tag, resto de argumentos
  local t=$1; shift
  python scripts/esc_run.py "$@" --smooth 7 --tag $t > $L/$t.log 2>&1
}

for n in 05 01; do
  # refractariedad: white-box, ciega con ventana 400, forma exacta
  corre e7_e1wb_n$n   --data refrac1_n$n --variant whitebox
  corre e7_e1B400_n$n --data refrac1_n$n --variant B --window 400
  corre e7_e1S2_n$n   --data refrac1_n$n --variant S --r-init 0.05 \
                      --init-params $R/e7_e1wb_n$n.json

  # actuador: white-box, ciega, estado de filtro
  [ $n = 01 ] && corre e7_e2wb_n$n --data act1_n$n --variant whitebox
  corre e7_e2B_n$n    --data act1_n$n --variant B
  corre e7_e2lag_n$n  --data act1_n$n --variant lag
done

echo "POSTER CON RUIDO COMPLETO" > $L/DONE_poster_ruido
