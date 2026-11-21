cd /c/Users/User/Desktop/wilson-cowan-id
export PYTHONIOENCODING=utf-8
for f in 120 100; do
  for p in repeticion bb_chica ruido rizo; do
    WC_FGAMMA=$f python scripts/control_pruebas_extra.py $p 3 > results/pruebas_${p}_${f}hz.out 2>&1
  done
done
echo FIN > results/pruebas_extra.fin
