# Espera a que termine la cola de la noche y corre el lazo cerrado con los e23.
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(r"C:\Users\User\Desktop\wilson-cowan-id")
marca = RAIZ / "logs" / "escalado" / "DONE_blackbox_y_lag_x0"
while not marca.exists():
    time.sleep(60)
with open(RAIZ / "logs" / "escalado" / "control_e23.out", "w") as f:
    r = subprocess.run([sys.executable, "scripts/control_distancia_y_rango.py", "3", "--nuevas"],
                       cwd=RAIZ, stdout=f, stderr=subprocess.STDOUT)
if r.returncode == 0:
    (RAIZ / "logs" / "escalado" / "DONE_control_e23").write_text("completo\n")
