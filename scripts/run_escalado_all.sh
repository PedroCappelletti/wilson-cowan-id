#!/bin/bash
# Corre las 9 corridas pendientes del escalado en 4 carriles paralelos.
# Se lanza con setsid/nohup para sobrevivir al cierre de la sesion.
cd "$(dirname "$0")/.."
P=.venv/bin/python

lane_a() {
  $P scripts/esc_run.py --data refrac1 --variant S --tag e1_S > logs/escalado/e1_S.log 2>&1
  $P scripts/esc_run.py --data act1 --variant latent --window 100 --tag e2_lat_w100 > logs/escalado/e2_lat_w100.log 2>&1
}
lane_b() {
  $P scripts/esc_run.py --data refrac1 --variant B --window 100 --tag e1_B_w100 > logs/escalado/e1_B_w100.log 2>&1
  $P scripts/esc_run.py --data act1 --variant whitebox --tag e2_wb > logs/escalado/e2_wb.log 2>&1
}
lane_c() {
  $P scripts/esc_run.py --data refrac1 --variant B --window 200 --tag e1_B_w200 > logs/escalado/e1_B_w200.log 2>&1
  $P scripts/esc_run.py --data act1 --variant B --window 100 --tag e2_B > logs/escalado/e2_B.log 2>&1
  $P scripts/esc_run.py --data act1 --variant latent --window 400 --tag e2_lat_w400 > logs/escalado/e2_lat_w400.log 2>&1
}
lane_d() {
  $P scripts/esc_run.py --data refrac1 --variant B --window 400 --tag e1_B_w400 > logs/escalado/e1_B_w400.log 2>&1
  $P scripts/esc_run.py --data act1 --variant lag --tag e2_lag > logs/escalado/e2_lag.log 2>&1
}

lane_a & lane_b & lane_c & lane_d &
wait
echo "ESCALADO COMPLETO" > logs/escalado/DONE
