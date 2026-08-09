# =============================================================================
#  GENERACION DE DATOS A PARTIR DEL MODELO DE WILSON-COWAN
# =============================================================================
#
#  Que hace este archivo: simula UNA trayectoria del modelo de Wilson-Cowan
#  (forzada por los estimulos P y Q) y la deja lista para entrenar la PINN.
#
#  Que es "un dato": una trayectoria completa = muchos puntos (t_k, estado_k),
#  donde el estado en cada instante es el par [I(t_k), E(t_k)] (mas la salida
#  y = E - I y los estimulos P, Q en ese instante).
#
#  Donde se guarda: en un archivo .npz (formato de NumPy que mete varios
#  arrays con nombre en un solo archivo). Por defecto data/processed/dataset.npz
#
#  NOTA: este archivo tiene la logica reutilizable. Los parametros que vas a
#  querer tocar a mano (cuanto tiempo, P, Q, ruido, etc.) estan centralizados
#  y comentados en  scripts/generate_data.py  (el "panel de control").
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from src.wilson_cowan import WilsonCowan, WilsonCowanParams, zero_input


# =============================================================================
#  FUNCION PRINCIPAL: generar la trayectoria
# =============================================================================
def generate_dataset(
    params: WilsonCowanParams,                      # parametros del modelo
    P: Callable[[float], float] = zero_input,       # estimulo externo a E
    Q: Callable[[float], float] = zero_input,       # estimulo externo a I
    I0: float = 0.0,                                # condicion inicial de I
    E0: float = 0.0,                                # condicion inicial de E
    t_span: tuple[float, float] = (0.0, 600.0),     # (t_inicial, t_final)
    n_eval: int = 6000,                             # cuantos puntos muestrear
    noise_std: float = 0.0,                         # ruido de observacion (0 = limpio)
    seed: int | None = None,                        # semilla (reproduce el ruido)
    rel_tol: float = 1e-3,                           # tolerancia relativa del integrador
    abs_tol: float = 1e-6,                           # tolerancia absoluta del integrador
    perturbation=None,                               # incertidumbre dinamica (opcional)
) -> dict[str, np.ndarray]:
    """Genera UNA trayectoria del modelo de Wilson-Cowan y la devuelve como dict.

    El dataset que sale de aca es justamente esa trayectoria: una grilla
    uniforme de `n_eval` instantes entre t_inicial y t_final, con el estado
    [I, E] en cada uno. Si `noise_std` > 0, se le suma ruido gaussiano a las
    observaciones I y E (para simular una medicion imperfecta).

    Returns:
        dict con:
          t  -> instantes de tiempo (grilla uniforme)
          I  -> actividad inhibitoria en cada t (con ruido si noise_std > 0)
          E  -> actividad excitatoria en cada t (con ruido si noise_std > 0)
          y  -> salida observable y = E - I
          P  -> estimulo a E en cada t
          Q  -> estimulo a I en cada t
        ademas de metadatos (parametros, condiciones iniciales, ruido, semilla).
    """
    # --- 1) Armamos el modelo con sus parametros y sus estimulos externos.
    #        Si hay perturbacion, el simulador deja de ser Wilson-Cowan puro:
    #        pasa a representar "el cerebro real" con fisica que el modelo que
    #        despues entrenamos NO contempla.
    modelo = WilsonCowan(params=params, P=P, Q=Q, perturbation=perturbation)

    # --- 2) Grilla uniforme de tiempos donde queremos los datos.
    #        Mas puntos = trayectoria mas "densa" (mas datos para la PINN).
    t_eval = np.linspace(t_span[0], t_span[1], n_eval)

    # --- 3) Integramos las ecuaciones (esto ya esta probado en model.py).
    sol = modelo.simulate(
        I0=I0,
        E0=E0,
        t_span=t_span,
        t_eval=t_eval,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )

    # Trayectoria "limpia" que entrego el integrador.
    I = sol["I"].copy()
    E = sol["E"].copy()

    # --- 4) Ruido de observacion (opcional). Simula que medimos con error.
    #        Usamos una semilla propia para que sea reproducible.
    if noise_std > 0.0:
        rng = np.random.default_rng(seed)
        I = I + rng.normal(0.0, noise_std, size=I.shape)
        E = E + rng.normal(0.0, noise_std, size=E.shape)

    # --- 5) Empaquetamos todo (datos + metadatos) en un diccionario.
    #        P y Q son SIEMPRE los estimulos COMANDADOS. Si hay perturbacion en
    #        el canal de actuacion, el estimulo que realmente llego (P_eff) va
    #        aparte y es SOLO para diagnostico: entrenar con el seria hacer
    #        trampa, porque en un experimento real no lo conoces.
    extra_out = {}
    if perturbation is not None:
        extra_out = {
            "P_eff": sol["P_eff"], "Q_eff": sol["Q_eff"],
            "pert_extra": sol["extra"],
            **{k: np.asarray(v) for k, v in perturbation.metadata().items()},
        }

    return {
        "t": sol["t"],
        "I": I,
        "E": E,
        "y": E - I,        # salida observable, recalculada con el ruido incluido
        "P": sol["P"],
        "Q": sol["Q"],
        **extra_out,
        # ---- metadatos (utiles para evaluar/reproducir despues) ----
        "I0": np.asarray(I0),
        "E0": np.asarray(E0),
        "t_span": np.asarray(t_span),
        "n_eval": np.asarray(n_eval),
        "noise_std": np.asarray(noise_std),
        "seed": np.asarray(-1 if seed is None else seed),
        # parametros del modelo (planos, para poder reconstruirlo)
        "te": np.asarray(params.te),
        "ti": np.asarray(params.ti),
        "wEE": np.asarray(params.wEE),
        "wEI": np.asarray(params.wEI),
        "wIE": np.asarray(params.wIE),
        "wII": np.asarray(params.wII),
        "ae": np.asarray(params.ae),
        "ai": np.asarray(params.ai),
        "thetae": np.asarray(params.thetae),
        "thetai": np.asarray(params.thetai),
    }


# =============================================================================
#  GUARDAR Y CARGAR EL DATASET (.npz)
# =============================================================================
def save_dataset(dataset: dict[str, np.ndarray], path: str | Path) -> None:
    """Guarda el dataset en un archivo .npz (crea la carpeta si no existe)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # savez_compressed mete todos los arrays con su nombre en un solo archivo.
    np.savez_compressed(path, **dataset)


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Carga un dataset .npz y lo devuelve como dict de arrays."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
