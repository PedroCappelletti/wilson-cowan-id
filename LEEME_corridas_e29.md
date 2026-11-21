# Corridas e29: backbone expandido con la forma equivocada del actuador

Agregado al paquete de amplitud (`D:\wilson-cowan-amplitud`). Usa el mismo entorno virtual y los mismos datasets del actuador con recorte que las tandas anteriores. No hay que instalar ni generar nada.

El zip trae estos archivos, que se copian sobre la carpeta del paquete respetando las rutas:

- `scripts/cola_forma_equivocada.py`, la cola nueva (18 corridas);
- `scripts/esc_run.py`, el mismo de antes con las opciones `--forma` y `--tau-fijo`;
- `src/neural_ode/memory.py`, el mismo de antes con `LagGrayBox.fijar_tau`;
- `scripts/hitos.py`, el mismo de antes con la opción `--cola e29`;
- `results/escalado/e25_e2wb_*.json`, los seis white-box de partida (ya deberían estar en la carpeta; son los mismos);
- este archivo.

## Pasos

1. Verificar que los datasets siguen intactos. Tiene que terminar en `TODO OK`:

       .venv\Scripts\python scripts\verificar_datos_amplitud.py

2. La cola, con N corridas en paralelo, y `despierto.py`, las dos en segundo plano, sin ventana y con prioridad baja, igual que las tandas anteriores:

       .venv\Scripts\python scripts\cola_forma_equivocada.py N > logs\escalado\cola_e29.out 2>&1
       .venv\Scripts\python scripts\despierto.py >> logs\escalado\despierto_e29.out 2>&1

   La cantidad en paralelo se puede cambiar con la cola andando, escribiendo un número en `logs/escalado/LUGARES`.

3. Seguimiento: `scripts\hitos.py --cola e29` (estado), `scripts\hitos.py --cola e29 --esperar` (se bloquea hasta un hito, una caída o el fin de la cola) y `scripts\hitos.py --cola e29 --empaquetar` (zips de los hitos en `salida_hitos_e29/`).

## Qué corre

18 corridas del estado de filtro (backbone expandido) sobre el actuador con recorte, arrancando desde el white-box, con una pieza de la forma cambiada por vez: saturación tanh en lugar del recorte (`lagtanh`), τ̂ fijo en 2 ms (`lagtau2`) y τ̂ fijo en 0,5 ms (`lagtau05`), con los dos niveles de ruido y tres semillas. Los hitos son la semilla 0 (6 corridas), las semillas 0 y 1 (12) y todo (18). Las corridas del estado de filtro tardaron alrededor de una hora en la otra máquina.
