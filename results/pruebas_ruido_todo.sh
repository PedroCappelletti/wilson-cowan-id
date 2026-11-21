cd /c/Users/User/Desktop/wilson-cowan-id
export PYTHONIOENCODING=utf-8
while [ ! -f results/pruebas_extra.fin ]; do sleep 120; done
for f in 120 100; do
  WC_FGAMMA=$f python scripts/control_pruebas_extra.py ruido_todo 3 > results/pruebas_ruido_todo_${f}hz.out 2>&1
done
echo FIN > results/pruebas_ruido_todo.fin
