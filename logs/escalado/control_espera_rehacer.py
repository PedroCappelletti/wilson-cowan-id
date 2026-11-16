# Proceso de espera: mantiene a despierto.py activo (busca "control_" en la línea
# de comando) mientras corre la cadena de rehacer_lazos_historia.py, incluida la
# etapa de trazas, cuyo nombre no coincide.
import time
from pathlib import Path
marca = Path(r"C:\Users\User\Desktop\wilson-cowan-id\logs\escalado\DONE_rehacer_lazos")
inicio = time.time()
while not marca.exists() and time.time() - inicio < 12 * 3600:
    time.sleep(60)
