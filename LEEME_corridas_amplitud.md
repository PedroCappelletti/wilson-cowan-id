# Corridas de amplitud grande (e25, e26, e27)

Paquete para correr en otra máquina los modelos sobre los datasets de amplitud grande. Contiene el código (`src/`, `scripts/`), los tres datasets limpios y sus huellas, y las versiones exactas de los paquetes.

## Pasos

1. Python 3.13.5 de 64 bits, en un entorno virtual dentro de esta carpeta:

       py -3.13 -m venv .venv
       .venv\Scripts\python -m pip install --upgrade pip
       .venv\Scripts\python -m pip install -r requirements_corridas.txt --extra-index-url https://download.pytorch.org/whl/cpu

2. Variantes con ruido de los tres datasets limpios (unos minutos):

       for %p in (wc0 refrac actc) do (
         .venv\Scripts\python scripts\esc_gen_ruido.py %p_amp_2s 0.01 0.05 --ref-fs 20000
         .venv\Scripts\python scripts\esc_gen_ruido.py %p_amp_2s 0.01 0.05 --ref-fs 20000 --semilla 1
         .venv\Scripts\python scripts\esc_gen_ruido.py %p_amp_2s 0.01 0.05 --ref-fs 20000 --semilla 2 )

3. Verificar que los 21 datasets coinciden con los originales. Tiene que terminar en `TODO OK`:

       .venv\Scripts\python scripts\verificar_datos_amplitud.py

4. Prueba corta, unos minutos:

       set WC_THREADS=1
       .venv\Scripts\python scripts\esc_run.py --data actc_amp_2s_n01 --variant H --hist 10 --window 10 --epochs 20 --val-every 1000 --lbfgs-steps 0 --tag prueba_tiempo

   Después borrar `results/escalado/prueba_tiempo.json`, `results/escalado/models/prueba_tiempo.pt` y `logs/escalado/prueba_tiempo.log`.

5. La cola, con N corridas en paralelo, y `despierto.py` para que la máquina no se suspenda por inactividad. Las dos en segundo plano, sin ventana y con prioridad baja:

       .venv\Scripts\python scripts\cola_amplitud.py N > logs\escalado\cola_amplitud.out 2>&1
       .venv\Scripts\python scripts\despierto.py >> logs\escalado\despierto_amplitud.out 2>&1

   La cantidad en paralelo se puede cambiar con la cola andando, escribiendo un número en `logs/escalado/LUGARES`.

6. Seguimiento: `scripts\hitos.py` (estado), `scripts\hitos.py --esperar` (se bloquea hasta un hito, una caída o el fin de la cola) y `scripts\hitos.py --empaquetar` (zips de los hitos en `salida_hitos/`).

## Qué corre

180 corridas. e25 es la primera etapa (108), e26 la segunda con x0 entrenable (48) y e27 la referencia black-box (24). El detalle está en el encabezado de `scripts/cola_amplitud.py`.
