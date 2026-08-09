#!/usr/bin/env python3
# =============================================================================
#  ¿COPIA BIEN EL COMPORTAMIENTO LA NEURAL ODE?
# =============================================================================
#
#  Todas las metricas del estudio miran los PARAMETROS. Esta mira lo otro: si el
#  modelo aprendido, soltado a rodar solo, reproduce la actividad del cerebro
#  simulado.
#
#  La prueba es exigente a proposito ("open loop"): se le da UNICAMENTE el estado
#  inicial y el estimulo, y tiene que generar los 200 ms enteros por su cuenta,
#  sin volver a mirar nunca el dato real. Cualquier error se acumula. Es lo que
#  pasaria si lo usaramos como planta para disenar un controlador.
#
#  QUE TAN EXIGENTE ES, en numeros: el entrenamiento usa multiple shooting con
#  ventanas de 100 muestras (TrainConfig.window en graybox_train.py), asi que
#  durante el ajuste el modelo nunca tuvo que sostener mas de ~5 ms sin que le
#  reinicien el estado desde el dato observado. Aca tiene que sostener las 4000
#  muestras de la trayectoria entera de un tiron: unas 40 veces mas largo que lo
#  que vio al entrenar. Por eso un modelo puede tener un error de ventana
#  excelente y fallar feo aca — y por eso esta metrica no es redundante con la
#  loss de entrenamiento.
#
#  Ademas se mide en estimulos de TEST, que el modelo no vio al entrenar.
#
#  El error se reporta NORMALIZADO al rango de la senal (NRMSE %), porque un MSE
#  de 1e-2 no dice nada por si solo. Como referencia:
#     < 5%   -> reproduce muy bien
#     5-15%  -> reproduce la forma, con desvios visibles
#     > 25%  -> no sirve como planta
#  Esos cortes son criterio de ingenieria (el mismo espiritu que el "fit %" de
#  identificacion de sistemas), no un umbral estadistico sacado de estos datos.
#
#  POR QUE NORMALIZADO Y NO MSE CRUDO: I y E son tasas en [0,1] y cada escenario
#  tiene su propia amplitud (algunos estimulos hacen oscilar fuerte, otros casi
#  no mueven al sistema). El MISMO MSE es un desastre en un escenario chato y
#  despreciable en uno de gran amplitud. Dividiendo por el rango pico-a-pico de
#  la senal REAL el numero queda en "porcentaje de la senal": comparable entre
#  escenarios, entre modelos y entre los canales I y E.
#
#  Y POR QUE ADEMAS LA CORRELACION: el NRMSE castiga por igual dos fallas que no
#  son la misma cosa — estar corrido (offset o fase) y no tener la forma. La
#  correlacion las separa: corr ~ 1 con NRMSE alto significa que la dinamica se
#  aprendio y el error es de nivel o de fase; corr baja significa que el modelo
#  no capturo el mecanismo.
#
#  USO:  python scripts/exp_reproduccion.py
# =============================================================================

from __future__ import annotations

import json
import sys
from pathlib import Path

# El script vive en scripts/ y se corre desde la raiz del repo: hay que poner la
# raiz en el path para que "import src.*" funcione sin instalar el paquete.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

# Son rollouts largos pero de batch 1: mas hilos no acelera y compite con otros
# experimentos corriendo en paralelo.
torch.set_num_threads(4)

from src.neural_ode import GrayBoxWC, rollout
from src.neural_ode.graybox_train import ALL_P

OUT = Path("results/uncertainty/reproduccion.json")
FIG = Path("results/figures/reproduccion.png")


# =============================================================================
#  SECCION 1: QUE MODELOS SE COMPARAN Y CONTRA QUE PLANTA
# =============================================================================
#  Cada fila es (nombre legible, checkpoint del modelo, dataset con el que se lo
#  evalua). La perilla eps del nombre del dataset es cuanta fisica le falta al
#  modelo respecto del simulador:
#     eps0 -> el simulador es Wilson-Cowan puro: el modelo PUEDE ser exacto, asi
#             que el NRMSE que salga aca es el piso alcanzable (error numerico +
#             lo que se perdio al identificar).
#     eps1 -> el simulador tiene refractariedad + actuador con retardo y
#             saturacion: hay un hueco real que el WB no puede cubrir.
#
#  Las cuatro filas cuentan una historia en orden:
#     1) white-box sin hueco   -> la referencia de "lo mejor posible".
#     2) white-box con hueco   -> cuanto se degrada por ser rigido (no tiene con
#                                 que absorber la fisica que falta).
#     3) gray-box B            -> g(I,E) libre: cuanto recupera la correccion.
#     4) gray-box D lam=1      -> g(I,E) + penalidad de proyeccion; interesa ver
#                                 si al proteger los parametros se pierde ajuste.
#
#  QUE DA HOY (corrida actual, promedio sobre los 7 escenarios de test): 2% sin
#  hueco contra 13-15% con hueco. O sea que el hueco de fisica se paga carisimo
#  en reproduccion, y las correcciones aprendidas casi no lo recuperan (B incluso
#  empeora el canal I). Es coherente con lo que dice el manual: g_phi sin memoria
#  no puede capturar el RETARDO del actuador, que tiene estado propio.
#
#  TRAMPA: el dataset tiene que ser el MISMO eps con el que se entreno cada
#  modelo. Evaluar f2_eps1 sobre eps0.npz mezclaria dos efectos (mismatch de
#  planta + cambio de escenario) y el numero no querria decir nada.
MODELOS = [
    ("white-box, planta SIN hueco", "results/uncertainty/models/f2_eps0.pt",
     "data/processed/uncertain/eps0.npz"),
    ("white-box, planta CON hueco", "results/uncertainty/models/f2_eps1.pt",
     "data/processed/uncertain/eps1.npz"),
    ("gray-box B (g aprendida)", "results/uncertainty/models/f3_B_eps1.pt",
     "data/processed/uncertain/eps1.npz"),
    ("gray-box D regularizada", "results/uncertainty/models/f3_D_eps1_lam1.pt",
     "data/processed/uncertain/eps1.npz"),
]


# =============================================================================
#  SECCION 2: RECONSTRUIR EL MODELO ENTRENADO
# =============================================================================
def carga(ckpt_path, params_json=None):
    """Reconstruye el modelo desde un checkpoint .pt, o desde el json de F2.

    Por que dos caminos: F3 (gray-box) tiene una red adentro, asi que guarda el
    state_dict completo en un .pt. F2 (white-box) no tiene nada que guardar mas
    alla de los 10 numeros identificados, y solo deja el json del experimento.
    """
    p = Path(ckpt_path)
    if p.exists():
        # weights_only=False porque el checkpoint no es solo tensores: trae los
        # flags de arquitectura (variante, eps, lam, structured...).
        ck = torch.load(p, map_location="cpu", weights_only=False)
        # Los valores iniciales son de relleno: load_state_dict los sobreescribe
        # a todos. Pero NO pueden ser 0.0: los parametros se guardan crudos con
        # inv_softplus = log(expm1(v)), y expm1(0)=0 -> log(0) = -inf -> NaN al
        # construir. 1.0 es simplemente un valor seguro.
        #
        # Los flags SI importan y por eso salen del checkpoint: definen que
        # tensores existe en el modulo. Si no coincidieran con el entrenamiento
        # (p.ej. use_correction=False cuando se entreno con red), load_state_dict
        # falla por claves faltantes o sobrantes.
        #
        # learnable_weights/learnable_params van fijos en True porque asi entrena
        # graybox_train: con esos flags te,ti,ae,ai,thetae,thetai son Parameters
        # (raw_te, ...) y no buffers. El tamano de la red (hidden=32) tampoco se
        # guarda en el .pt: si algun dia se entrena con otro, esta carga rompe
        # por mismatch de shapes.
        m = GrayBoxWC({k: 1.0 for k in ALL_P},
                      {k: 1.0 for k in ("wEE", "wEI", "wIE", "wII")},
                      learnable_weights=True, learnable_params=True,
                      use_correction=ck.get("use_correction", False),
                      correction_inputs=ck.get("correction_inputs", "xpq"),
                      structured=ck.get("structured", False))
        m.load_state_dict(ck["state"])
        m.eval()
        return m
    # F2 no guarda .pt: se reconstruye desde los parametros del json
    j = json.loads(Path(params_json).read_text())
    # Dos formatos historicos: dict de una corrida, o lista de corridas (se toma
    # la primera). Sin correccion neuronal -> queda Wilson-Cowan puro con los
    # parametros identificados, que es exactamente el white-box de F2.
    ph = j["params"] if isinstance(j, dict) else j[0]["params"]
    m = GrayBoxWC(ph, {k: ph[k] for k in ("wEE", "wEI", "wIE", "wII")},
                  learnable_weights=True, learnable_params=True)
    m.eval()
    return m


# =============================================================================
#  SECCION 3: EL ROLLOUT OPEN LOOP Y SUS METRICAS
# =============================================================================
@torch.no_grad()
def reproduce(m, d, solo_test=True):
    """Rollout completo por trayectoria. Devuelve NRMSE% por escenario.

    solo_test=True (lo que se reporta) mide sobre los estimulos que el modelo no
    vio; con False se mide sobre los de entrenamiento, util solo para ver la
    brecha train/test.
    """
    it = d["is_test"].astype(bool)   # mascara de escenarios (7 de 20 son test)
    sel = it if solo_test else ~it
    I, E, P, Q = d["I"][sel], d["E"][sel], d["P"][sel], d["Q"][sel]
    # dt sale del dataset, no se hardcodea: el rollout usa el estimulo con
    # zero-order hold, asi que el paso del integrador tiene que ser el mismo
    # intervalo con el que se muestrearon los datos.
    dt = float(d["dt"])
    filas, trazas = [], []
    for s in range(len(I)):
        T = I.shape[1]
        # Lo UNICO que se le da del dato real: la primera muestra. Desde aca en
        # adelante el modelo se alimenta de si mismo.
        x0 = torch.tensor([[I[s, 0], E[s, 0]]], dtype=torch.float32)
        Ps = torch.tensor(P[s], dtype=torch.float32).reshape(T, 1, 1)
        Qs = torch.tensor(Q[s], dtype=torch.float32).reshape(T, 1, 1)
        # El [:-1] no es cosmetico: rollout devuelve T+1 estados a partir de T
        # entradas (incluye x0). Con T-1 entradas devuelve T estados, que es lo
        # que se necesita para comparar muestra a muestra con la trayectoria
        # real. El [:, 0, :] saca la dimension de batch (que aca vale 1).
        pred = rollout(m, x0, Ps[:-1], Qs[:-1], dt)[:, 0, :].numpy()
        real = np.stack([I[s], E[s]], 1)
        # Normalizador = rango pico-a-pico de la senal REAL, por canal. Que sea
        # de la real y no de la predicha es deliberado: si el rollout se va a la
        # banquina el divisor no cambia y el NRMSE explota, que es justo lo que
        # queremos ver. Normalizando por el rango de la prediccion, una
        # divergencia se disimularia.
        rng_ = real.max(0) - real.min(0)
        # Canal plano (un escenario donde el estimulo no llega a mover al
        # sistema): evita dividir por ~0. Con divisor 1.0 el numero pasa a ser el
        # RMSE crudo en unidades de tasa, no un porcentaje — leerlo como % ahi
        # seria un error.
        rng_[rng_ < 1e-9] = 1.0
        nrmse = 100 * np.sqrt(((pred - real) ** 2).mean(0)) / rng_
        # correlacion: mide si sigue la FORMA aunque haya offset
        # (si la prediccion queda constante, su desvio es 0 y corrcoef devuelve
        # nan; ese nan se propaga al promedio y delata el caso).
        corr = [float(np.corrcoef(pred[:, c], real[:, c])[0, 1]) for c in (0, 1)]
        filas.append({"nrmse_I": float(nrmse[0]), "nrmse_E": float(nrmse[1]),
                      "corr_I": corr[0], "corr_E": corr[1]})
        trazas.append((real, pred))
    return filas, trazas


# =============================================================================
#  SECCION 4: TABLA EN PANTALLA + JSON DE RESULTADOS
# =============================================================================
def main():
    print("=== ¿Copia bien el comportamiento la Neural ODE? ===")
    print("    Rollout OPEN LOOP de 200 ms: solo estado inicial + estimulo,")
    print("    sin volver a mirar el dato real. Estimulos de TEST (no vistos).\n")
    print(f"  {'modelo':32} {'NRMSE_I%':>10} {'NRMSE_E%':>10} "
          f"{'corr_I':>8} {'corr_E':>8}")
    print("  " + "-" * 72)

    res, guardar = [], []
    for nombre, ck, data in MODELOS:
        d = np.load(data, allow_pickle=True)
        # Ruta del json de respaldo para el caso F2 (que no deja .pt). F2 guarda
        # su json un nivel arriba de models/, de ahi el primer replace. Para los
        # F3 el replace no aplica y pj apunta a un archivo que no existe: no
        # molesta porque su .pt si esta, pero si faltara, el mensaje de error
        # hablaria de un json inexistente en vez del checkpoint que falta.
        pj = ck.replace("models/f2_", "f2_").replace(".pt", ".json")
        try:
            m = carga(ck, pj)
        except Exception as e:
            # Un modelo que todavia no se entreno no debe tumbar la comparacion
            # entera: se deja la fila marcada y se sigue con las demas.
            print(f"  {nombre:32}  (no disponible: {e})")
            continue
        filas, trazas = reproduce(m, d)
        # Promedio simple sobre escenarios de test. Ojo: es sensible a un solo
        # escenario que divergio; para eso esta "por_escenario" en el json.
        nI = float(np.mean([f["nrmse_I"] for f in filas]))
        nE = float(np.mean([f["nrmse_E"] for f in filas]))
        cI = float(np.mean([f["corr_I"] for f in filas]))
        cE = float(np.mean([f["corr_E"] for f in filas]))
        print(f"  {nombre:32} {nI:10.2f} {nE:10.2f} {cI:8.3f} {cE:8.3f}")
        res.append({"modelo": nombre, "nrmse_I": nI, "nrmse_E": nE,
                    "corr_I": cI, "corr_E": cE, "por_escenario": filas})
        guardar.append((nombre, d, trazas))

    print("\n  Referencia: <5% reproduce muy bien | 5-15% forma correcta con")
    print("  desvios | >25% no sirve como planta.")
    print("  La correlacion mide si sigue la FORMA (1.0 = misma forma exacta).")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    _figura(guardar)
    print(f"\n  -> {OUT}\n  -> {FIG}")


# =============================================================================
#  SECCION 5: FIGURA — LA VERSION VISUAL DE LA TABLA
# =============================================================================
#  Un panel por modelo con real vs predicha superpuestas. Es la evidencia
#  cualitativa que acompana al numero: deja ver SI el modelo se desfasa, se
#  aplana o directamente diverge, que el NRMSE solo no distingue.
def _figura(guardar):
    if not guardar:
        return
    # Backend Agg antes de importar pyplot: el script se corre sin pantalla y con
    # el backend interactivo fallaria al crear la figura.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(guardar)
    fig, ax = plt.subplots(n, 1, figsize=(11, 2.6 * n), sharex=True)
    # Con un solo panel, subplots no devuelve arreglo: se envuelve para poder
    # indexar igual en los dos casos.
    if n == 1:
        ax = [ax]
    for k, (nombre, d, trazas) in enumerate(guardar):
        # d["t"] tiene las mismas T muestras que real/pred (x0 incluido).
        t = d["t"]
        # Solo el PRIMER escenario de test: la figura es ilustrativa, el numero
        # representativo es el promedio de la tabla.
        real, pred = trazas[0]
        # Se dibuja solo el canal E (columna 1) para no saturar el panel; I sigue
        # un patron parecido y su NRMSE ya esta en la tabla.
        ax[k].plot(t, real[:, 1], lw=1.6, color="#1f4e79",
                   label="E real (el 'cerebro')")
        ax[k].plot(t, pred[:, 1], lw=1.2, ls="--", color="#d62728",
                   label="E predicha por el modelo")
        ax[k].set_title(nombre, fontsize=10)
        ax[k].set_ylabel("E")
        ax[k].grid(alpha=0.3)
        # Una sola leyenda (los colores significan lo mismo en todos los paneles).
        if k == 0:
            ax[k].legend(fontsize=8, loc="upper right")
    ax[-1].set_xlabel("tiempo [ms]")
    fig.suptitle("Rollout open-loop en estimulo de TEST: ¿sigue la trayectoria?",
                 fontsize=11)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
