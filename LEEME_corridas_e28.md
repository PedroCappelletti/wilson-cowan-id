# Corridas e28: β verdadero fijo sobre los datasets de amplitud grande

Agregado al paquete de amplitud (`D:\wilson-cowan-amplitud`). Usa el mismo código, el mismo entorno virtual y los mismos 21 datasets con ruido que la tanda anterior. No hay que instalar ni generar nada.

El zip trae cuatro archivos, que se copian sobre la carpeta del paquete respetando las rutas:

- `scripts/cola_amplitud_beta_verdadero.py`, la cola nueva (30 corridas);
- `scripts/hitos.py`, el mismo de antes con la opción `--cola e28`;
- `results/escalado/parametros_verdaderos.json`, los β verdaderos que se dejan fijos;
- este archivo.

## Pasos

1. Verificar que los datasets siguen intactos. Tiene que terminar en `TODO OK`:

       .venv\Scripts\python scripts\verificar_datos_amplitud.py

2. La cola, con N corridas en paralelo, y `despierto.py`, las dos en segundo plano, sin ventana y con prioridad baja, igual que la tanda anterior:

       .venv\Scripts\python scripts\cola_amplitud_beta_verdadero.py N > logs\escalado\cola_e28.out 2>&1
       .venv\Scripts\python scripts\despierto.py >> logs\escalado\despierto_e28.out 2>&1

   La cantidad en paralelo se puede cambiar con la cola andando, escribiendo un número en `logs/escalado/LUGARES`.

3. Seguimiento: `scripts\hitos.py --cola e28` (estado), `scripts\hitos.py --cola e28 --esperar` (se bloquea hasta un hito, una caída o el fin de la cola) y `scripts\hitos.py --cola e28 --empaquetar` (zips de los hitos en `salida_hitos_e28/`).

## Qué corre

30 corridas con los β verdaderos fijos (`--init-params ... --freeze-phys`), donde solo se entrena la red: agnóstica, historia con comando actual, historia sin comando actual y filtro entrenable en el actuador con recorte, y agnóstica en la refractaria, con los dos niveles de ruido y tres semillas. Los hitos son la semilla 0 (10 corridas), las semillas 0 y 1 (20) y todo (30). En la otra máquina, las corridas de estas arquitecturas tardaron entre 60 y 90 minutos.
