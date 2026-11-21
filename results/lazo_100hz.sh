#!/bin/bash
# Lazo cerrado con la gamma a 100 Hz: amplitud chica y amplitud grande, 3 procesos.
cd /c/Users/User/Desktop/wilson-cowan-id
export WC_FGAMMA=100 PYTHONIOENCODING=utf-8
python scripts/control_distancia_y_rango.py 3 > results/control_distancia_rango_100hz.out 2>&1
python scripts/control_amplitud.py 3 > results/control_amplitud_100hz.out 2>&1
echo FIN > results/lazo_100hz.fin
