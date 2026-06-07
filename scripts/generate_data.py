#!/usr/bin/env python3
# =============================================================================
#  PANEL DE CONTROL DEL DATASET  (Wilson-Cowan)
# =============================================================================
#
#  PARA QUE SIRVE ESTE ARCHIVO:
#  Es el unico lugar que tenes que tocar para decidir COMO queres los datos
#  que vas a usar para entrenar la PINN. Editas los valores de la seccion de
#  abajo (tiempo, P, Q, parametros, ruido, etc.), lo corres, y se genera y se
#  guarda la trayectoria en disco.
#
#  COMO SE USA:
#      python scripts/generate_data.py
#
#  QUE GENERA:
#  Una unica trayectoria del modelo = muchos puntos (t_k, [I_k, E_k]) sobre el
#  intervalo de tiempo que elijas. Esa trayectoria ES el dataset.
#
#  DONDE SE GUARDA:
#  En un archivo .npz (por defecto data/processed/dataset.npz). Despues se
#  carga con  load_dataset(...)  desde el entrenamiento o la evaluacion.
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

# Permite ejecutar el script directamente (agrega la raiz del repo al path).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.wilson_cowan import WilsonCowanParams, box_pulse, zero_input  # noqa: E402,F401
from src.data import generate_dataset, save_dataset  # noqa: E402


# #############################################################################
# ##                                                                         ##
# ##   ZONA EDITABLE: AJUSTA TODO A TU GUSTO AQUI ABAJO                       ##
# ##                                                                         ##
# #############################################################################

# -----------------------------------------------------------------------------
#  1) INTERVALO DE TIEMPO Y DENSIDAD DE LA TRAYECTORIA
# -----------------------------------------------------------------------------
#  T_INICIAL, T_FINAL : de cuanto a cuanto tiempo (en segundos) dura la trayectoria.
#  N_PUNTOS           : cuantos puntos muestreamos en ese intervalo.
#                       Mas puntos = trayectoria mas densa = mas datos para la PINN
#                       (pero mas lento). Menos puntos = mas liviano.
T_INICIAL = 0.0
T_FINAL = 600.0
N_PUNTOS = 60001   # = un punto cada 0.01 s (dt = 600/60000 = 0.01)

# -----------------------------------------------------------------------------
#  2) CONDICIONES INICIALES (estado del sistema en t = T_INICIAL)
# -----------------------------------------------------------------------------
#  De donde arranca cada poblacion. (0, 0) = arranca en reposo.
I0 = 0.0   # actividad inhibitoria inicial
E0 = 0.0   # actividad excitatoria inicial

# -----------------------------------------------------------------------------
#  3) ESTIMULOS EXTERNOS  P(t) -> poblacion E ,  Q(t) -> poblacion I
# -----------------------------------------------------------------------------
#  Son pulsos cuadrados: valen una amplitud constante entre t_on y t_off, y 0
#  fuera de esa ventana. Es la "fuerza" que se le inyecta al sistema para que
#  arranque a oscilar. Cambia amplitud y ventana para ver otros regimenes.
#  (Si queres SIN estimulo en alguno, usa zero_input en lugar de box_pulse.)
P = box_pulse(amplitude=0.8, t_on=100.0, t_off=400.0)   # estimulo a E
Q = box_pulse(amplitude=0.6, t_on=200.0, t_off=500.0)   # estimulo a I
# Ejemplo sin estimulo:   P = zero_input ;  Q = zero_input

# -----------------------------------------------------------------------------
#  4) PARAMETROS DEL MODELO DE WILSON-COWAN
# -----------------------------------------------------------------------------
#  Estos definen la dinamica interna. Los valores por defecto son los del
#  simulador de MATLAB (generan oscilaciones sostenidas / ciclo limite).
#  Cambia los que necesites; los que no pongas quedan en su valor por defecto.
PARAMS = WilsonCowanParams(
    te=1.0,      # constante de tiempo poblacion excitatoria (mas grande = mas lenta)
    ti=2.0,      # constante de tiempo poblacion inhibitoria
    wEE=6.4,     # peso E sobre si misma (autoexcitacion)
    wEI=4.8,     # peso I sobre E (cuanto la frena I)
    wIE=6.0,     # peso E sobre I (cuanto la activa E)
    wII=1.2,     # peso I sobre si misma (autoinhibicion)
    ae=1.2,      # ganancia de la sigmoidea excitatoria
    ai=1.0,      # ganancia de la sigmoidea inhibitoria
    thetae=2.8,  # umbral de la sigmoidea excitatoria
    thetai=4.0,  # umbral de la sigmoidea inhibitoria
)

# -----------------------------------------------------------------------------
#  5) RUIDO DE OBSERVACION (opcional)
# -----------------------------------------------------------------------------
#  Simula que medimos el sistema con error. 0.0 = datos perfectos (limpios).
#  Un valor chico (ej. 0.01) le agrega ruido gaussiano a I y E, util para
#  probar que la PINN no sobreajusta el ruido.
NOISE_STD = 0.0
SEED = 42   # semilla: hace que el ruido sea siempre el mismo (reproducible)

# -----------------------------------------------------------------------------
#  6) PRECISION DEL INTEGRADOR NUMERICO
# -----------------------------------------------------------------------------
#  Tolerancias del solver (como el ode45 de MATLAB). Mas chico = mas preciso
#  pero mas lento. Normalmente no hace falta tocarlas.
REL_TOL = 1e-3
ABS_TOL = 1e-6

# -----------------------------------------------------------------------------
#  7) DONDE GUARDAR EL DATASET
# -----------------------------------------------------------------------------
OUT_PATH = "data/processed/dataset.npz"

# -----------------------------------------------------------------------------
#  8) GRAFICAR AL TERMINAR (para revisar a ojo lo que generaste)
# -----------------------------------------------------------------------------
GRAFICAR = True   # True = guarda una figura de la trayectoria generada

# #############################################################################
# ##   FIN DE LA ZONA EDITABLE. De aca para abajo es la maquinaria.          ##
# #############################################################################


def main() -> None:
    # --- Generamos la trayectoria con los valores de arriba.
    dataset = generate_dataset(
        params=PARAMS,
        P=P,
        Q=Q,
        I0=I0,
        E0=E0,
        t_span=(T_INICIAL, T_FINAL),
        n_eval=N_PUNTOS,
        noise_std=NOISE_STD,
        seed=SEED,
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
    )

    # --- Guardamos en disco.
    save_dataset(dataset, OUT_PATH)

    # --- Resumen por consola para confirmar que salio bien.
    print("Dataset generado y guardado.")
    print(f"  archivo      : {OUT_PATH}")
    print(f"  puntos       : {dataset['t'].shape[0]}")
    print(f"  intervalo    : [{T_INICIAL}, {T_FINAL}] s")
    print(f"  ruido (std)  : {NOISE_STD}")
    print(f"  rango I      : [{dataset['I'].min():.4f}, {dataset['I'].max():.4f}]")
    print(f"  rango E      : [{dataset['E'].min():.4f}, {dataset['E'].max():.4f}]")

    # --- Grafico opcional para revisar a ojo.
    if GRAFICAR:
        from src.wilson_cowan import plot_results

        fig_path = "results/figures/dataset_generado.png"
        plot_results(dataset, save_path=fig_path)
        print(f"  figura       : {fig_path}")


if __name__ == "__main__":
    main()
