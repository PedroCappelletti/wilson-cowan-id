# Rehace los lazos cerrados afectados por el registro del comando en PlantaHistoria
# (corregido el 10/11/2026): distancia y rango, planta-modelo y trazas de ejemplo.
import subprocess
import sys
from pathlib import Path

RAIZ = Path(r"C:\Users\User\Desktop\wilson-cowan-id")
pasos = [("control_distancia_y_rango.py", ["3", "--nuevas"]), ("control_paper_planta_modelo.py", ["3"]),
         ("trazas_lazo_ejemplo.py", [])]
for script, args in pasos:
    with open(RAIZ / "logs" / f"rehacer_{Path(script).stem}.out", "w") as f:
        r = subprocess.run([sys.executable, f"scripts/{script}", *args], cwd=RAIZ, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        sys.exit(r.returncode)
(RAIZ / "logs" / "escalado" / "DONE_rehacer_lazos").write_text("completo\n")
