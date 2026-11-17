# Figura de la saturación para el informe: la de figuras_avances_recorte.py
# con las etiquetas sin «antes/ahora», que cuentan historia dentro de la figura.
import inspect, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import figuras_avances_recorte as f

f.OUT = Path(__file__).resolve().parent / "figuras"
src = inspect.getsource(f.fig_saturacion)
src = (src.replace(" (antes)", "").replace(" (ahora)", "")
          .replace('"rango de\nentrenamiento"', '"rango de\namplitud chica"'))
exec(src, f.__dict__)
f.fig_saturacion()
