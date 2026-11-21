cd /c/Users/User/Desktop/wilson-cowan-id
export PYTHONIOENCODING=utf-8
python scripts/control_e28.py 3 > results/control_e28.out 2>&1
WC_FGAMMA=100 python scripts/control_e28.py 3 > results/control_e28_100hz.out 2>&1
echo FIN > results/lazo_e28.fin
