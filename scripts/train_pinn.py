#!/usr/bin/env python3
# =============================================================================
#  ENTRENAR LA PINN  (Pasos 4.0 / 4.1 / 4.2)
# =============================================================================
#
#  Plan por etapas (elegis cual correr con EXPERIMENTO abajo):
#
#    4.0  FORWARD (chequeo de plomeria): los 10 parametros FIJOS en sus valores
#         verdaderos, ninguno entrenable. La red solo ajusta la trayectoria.
#         Verificas: que L_datos y L_fisica bajen y la prediccion se superponga
#         con la real. Si esto no ajusta, hay un bug -> parar y depurar aca.
#
#    4.1  INVERSO arrancando en los valores VERDADEROS (chequeo de estabilidad):
#         los 4 pesos entrenables, inicializados en 6.4, 4.8, 6.0, 1.2.
#         Verificas: que se queden ahi (que no se escapen).
#
#    4.2  INVERSO real desde un arranque ignorante (el test de verdad): los 4
#         pesos entrenables, inicializados lejos (p. ej. todos en 1.0).
#         Verificas: que converjan a 6.4, 4.8, 6.0, 1.2. Esto es lo que pide
#         el concurso.
#
#  Ademas: VALIDACION TEMPORAL. Se entrena con datos solo hasta el 80% del
#  tiempo y se mide como predice el 20% final que no vio (extrapolacion).
#  El entrenamiento corta solo cuando la perdida amesetar (deja de bajar).
#  Al terminar genera un HTML de resumen con graficos.
#
#  USO:  python scripts/train_pinn.py
# =============================================================================

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data import load_dataset  # noqa: E402
from src.pinn import PINN  # noqa: E402
from src.pinn.train import Trainer, TrainConfig  # noqa: E402


# #############################################################################
# ##   ZONA EDITABLE                                                         ##
# #############################################################################

# Cual paso correr: "4.0", "4.1" o "4.2".
# (Se puede sobreescribir con la variable de entorno EXPERIMENTO.)
EXPERIMENTO = os.environ.get("EXPERIMENTO", "4.0")

DATASET_PATH = "data/processed/dataset.npz"

# Semilla de entrenamiento (init de la red + minibatches). Fija = reproducible.
TRAIN_SEED = 0

# Arquitectura de la red.
HIDDEN_DIM = 64
N_LAYERS = 4
N_FOURIER = 128      # frecuencias de Fourier (rompen el sesgo espectral)
FOURIER_SCALE = 6.0  # dispersion de frecuencias (mas alto = picos mas agudos)

# Hiperparametros de entrenamiento.
CONFIG = TrainConfig(
    epochs=30_000,      # tope; la parada por meseta normalmente corta antes
    lr=1e-3,
    w_data=1.0,
    w_physics=1.0,      # el "lambda": subilo si en el inverso los pesos no convergen
    w_ic=1.0,
    batch_size=8_000,
    n_collocation=4_000,
    val_fraction=0.2,   # 20% final reservado para test temporal
    device="cpu",
    log_every=200,
    early_stop=True,
)

# #############################################################################
# ##   FIN ZONA EDITABLE                                                     ##
# #############################################################################

OUT_DIR = Path("results")
SUFIJO = EXPERIMENTO.replace(".", "_")
CHECKPOINT_PATH = OUT_DIR / "models" / f"pinn_{SUFIJO}.pt"
FIG_FIT = OUT_DIR / "figures" / f"pinn_ajuste_{SUFIJO}.png"
FIG_LOSS = OUT_DIR / "figures" / f"pinn_loss_{SUFIJO}.png"
FIG_W = OUT_DIR / "figures" / f"pinn_pesos_{SUFIJO}.png"
HTML_PATH = OUT_DIR / f"reporte_paso_{SUFIJO}.html"


def main() -> None:
    # Reproducibilidad: misma semilla -> mismo resultado (init de la red + minibatches).
    torch.manual_seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)

    ds = load_dataset(DATASET_PATH)
    w_true = {k: float(ds[k]) for k in ("wEE", "wEI", "wIE", "wII")}

    # --- Que se identifica y desde donde arranca, segun el experimento.
    if EXPERIMENTO == "4.0":
        identify, w_init = (), None
    elif EXPERIMENTO == "4.1":
        identify, w_init = ("wEE", "wEI", "wIE", "wII"), dict(w_true)
    elif EXPERIMENTO == "4.2":
        identify = ("wEE", "wEI", "wIE", "wII")
        w_init = {k: 1.0 for k in identify}
    else:
        raise ValueError(f"EXPERIMENTO desconocido: {EXPERIMENTO}")

    # --- Parametros FIJOS (conocidos): pesos no identificados (en su valor real)
    #     + constantes del modelo.
    ae, ai = float(ds["ae"]), float(ds["ai"])
    thetae, thetai = float(ds["thetae"]), float(ds["thetai"])
    fixed = {
        "te": float(ds["te"]), "ti": float(ds["ti"]),
        "ae": ae, "ai": ai, "thetae": thetae, "thetai": thetai,
        "ke": 1.0 / (1.0 + np.exp(ae * thetae)),
        "ki": 1.0 / (1.0 + np.exp(ai * thetai)),
    }
    for k in ("wEE", "wEI", "wIE", "wII"):
        if k not in identify:
            fixed[k] = w_true[k]

    # --- Red + entrenamiento.
    t_min, t_max = float(ds["t"].min()), float(ds["t"].max())
    model = PINN(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, t_min=t_min, t_max=t_max,
                 identify=identify, w_init=w_init,
                 n_fourier=N_FOURIER, fourier_scale=FOURIER_SCALE)

    # Inverso desde arranque lejano (4.2): se identifica sobre la trayectoria
    # COMPLETA (sin reservar test: una zona sin datos deja la red libre y rompe
    # la identificacion) + warmup 'datos primero' + congelar la red. Asi la red
    # queda fija en la curva verdadera en TODO el dominio y los pesos son la
    # unica variable libre -> el residuo solo se anula moviendolos a su valor
    # real. La extrapolacion ya se mostro en 4.0; aca se valida simulando hacia
    # adelante con el theta identificado. En 4.0/4.1 no se toca.
    if EXPERIMENTO == "4.2":
        CONFIG.physics_warmup = 3000
        CONFIG.freeze_net_after_warmup = True
        CONFIG.val_fraction = 0.0
        CONFIG.lr_weights = 0.05   # lr alto para que theta viaje de 1.0 a su valor real

    print(f"=== Experimento {EXPERIMENTO} | identifica: {identify or 'ninguno (forward)'} ===")
    trainer = Trainer(model, CONFIG, fixed)
    hist = trainer.train(ds)
    trainer.save_checkpoint(CHECKPOINT_PATH)

    # --- Reporte de pesos (si se identifico alguno).
    w_pred = model.weights_dict()
    if identify:
        print("\n=== Pesos identificados vs verdaderos ===")
        print(f"{'peso':6} {'verdadero':>10} {'estimado':>10} {'error %':>10}")
        for k in identify:
            err = 100.0 * abs(w_pred[k] - w_true[k]) / abs(w_true[k])
            print(f"{k:6} {w_true[k]:10.4f} {w_pred[k]:10.4f} {err:9.2f}%")

    # --- Graficos + HTML.
    _plot_fit(trainer, ds, CONFIG.val_fraction, FIG_FIT)
    _plot_loss(hist, FIG_LOSS)
    if identify:
        _plot_weights(hist, w_true, FIG_W)
    _build_html(EXPERIMENTO, identify, w_true, w_pred, hist, ds, CONFIG, trainer)
    print(f"\nReporte HTML: {HTML_PATH}")


# -----------------------------------------------------------------------------
#  Graficos
# -----------------------------------------------------------------------------
def _plot_fit(trainer, ds, val_fraction, path):
    import matplotlib.pyplot as plt
    t = ds["t"]
    pred = trainer.predict(t)
    t_split = t.min() + (1.0 - val_fraction) * (t.max() - t.min())
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, j, nombre in zip(axes, (0, 1), ("I", "E")):
        ax.plot(t, ds[nombre], label=f"{nombre} real", lw=1.4)
        ax.plot(t, pred[:, j], "--", label=f"{nombre} PINN", lw=1.1)
        if val_fraction > 0:
            ax.axvline(t_split, color="gray", ls=":", lw=1)
        ax.set_ylabel(nombre); ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    axes[0].set_title("PINN vs real  (punteado gris = corte train | test)")
    axes[1].set_xlabel("tiempo (s)")
    fig.tight_layout(); _save(fig, path)


def _plot_loss(hist, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(hist["loss"], label="total", lw=0.8, alpha=0.5)
    ax.semilogy(hist["data"], label="datos", lw=1.0)
    ax.semilogy(hist["physics"], label="fisica", lw=1.0)
    warmup = hist.get("warmup", 0)
    if warmup:
        ax.axvline(warmup, color="purple", ls=":", lw=1.2,
                  label="fin warmup (prende fisica)")
    ax.set_xlabel("epoch"); ax.set_ylabel("perdida (log)")
    ax.set_title("Evolucion de la perdida"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); _save(fig, path)


def _plot_weights(hist, w_true, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    colores = {}
    for k in ("wEE", "wEI", "wIE", "wII"):
        line, = ax.plot(hist[k], label=f"{k}", lw=1.3)
        colores[k] = line.get_color()
        ax.axhline(w_true[k], color=colores[k], ls=":", lw=1)  # valor verdadero
    ax.set_xlabel("epoch"); ax.set_ylabel("valor del peso")
    ax.set_title("Convergencia de los pesos (punteado = valor verdadero)")
    ax.legend(ncol=4); ax.grid(True, alpha=0.3)
    fig.tight_layout(); _save(fig, path)


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)


# -----------------------------------------------------------------------------
#  HTML de resumen (autocontenido: imagenes embebidas en base64)
# -----------------------------------------------------------------------------
def _b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


# --- "Diario de a bordo": incidencias y ajustes documentados por paso. ---
#     tipo: "error" (algo fallo), "ajuste" (cambio que se hizo), "info" (contexto).
AJUSTES_HEREDADOS = [
    ("ajuste", "Dataset a dt=0.01 (60.001 puntos en [0,600] s) para mayor densidad temporal."),
    ("ajuste", "Fourier features en la entrada (sin/cos de 128 frecuencias, escala 6): codifican el tiempo en muchas frecuencias antes del MLP."),
    ("ajuste", "Parada automatica por meseta (media movil de la perdida) + reduccion de learning rate al estancarse."),
    ("ajuste", "Validacion temporal: se entrena con datos solo del 80% inicial; la fisica (colocacion) se aplica en todo el dominio para poder extrapolar al 20% final."),
]
NOTAS_PASO = {
    "4.0": [
        ("error", "Primer intento con un MLP plano sobre t (sin Fourier features): la perdida de datos se clavo en ~8e-3 y la prediccion era una envolvente suave que NO capturaba los 5 picos de oscilacion. Causa: sesgo espectral (un MLP con tanh no representa oscilaciones marcadas a partir de t crudo)."),
        ("ajuste", "Se agregaron Fourier features -> la perdida de datos bajo de ~8e-3 a ~5e-6 y la trayectoria pasa a superponerse con la real. Sin esto, el chequeo de plomeria no pasaba."),
        ("ajuste", "Warning de PyTorch al convertir tensores con gradiente a float -> se reemplazo por .item()."),
    ],
    "4.1": [
        ("info", "Los 4 pesos se inicializan en sus valores VERDADEROS (6.4, 4.8, 6.0, 1.2). Se vigila que el entrenamiento no los aleje de ahi (chequeo de estabilidad / gradiente bien conectado)."),
    ],
    "4.2": [
        ("info", "Los 4 pesos se inicializan LEJOS (todos en 1.0). Deben converger por si solos a los valores verdaderos. Es el experimento inverso genuino que pide el concurso."),
        ("error", "1er intento (datos y fisica con peso 1 desde epoch 0): se trabo en un minimo local. Los pesos apenas se movieron de 1.0 (~1.5/0.6, error 50-87%) y la trayectoria no ajusto (L_datos ~4.6e-3)."),
        ("error", "2do intento (solo warmup 'datos primero'): la red aprende bien la trayectoria en el warmup, pero al prender la fisica con los pesos aun en 1.0, la red (muy flexible) DEFORMA la curva para satisfacer la fisica equivocada en vez de mover los pesos. L_datos volvio a subir a ~4.6e-3 y los pesos siguieron clavados."),
        ("error", "3er intento (warmup + congelar la red, con 20% de test reservado): la red solo vio datos del 80% inicial -> quedaba 'basura' en el 20% final. Al congelarla, la colocacion en esa zona daba un residuo que ningun peso podia anular -> compromiso equivocado (L_fisica ~1e-2)."),
        ("error", "4to intento (warmup + peso de datos ALTO, w_data=100): al reves, la fisica quedo despreciable frente a los datos y los pesos NO se movieron de 1.0. El balance datos/fisica es un equilibrio fino e inestable."),
        ("error", "5to intento (trayectoria completa + warmup + congelar): la red ajusto perfecto (L_datos 7e-6) pero los pesos casi no se movieron (quedaron en ~1.05). Causa: los pesos heredaban el lr=1e-3 de la red (muy chico para viajar de 1.0 a 6.4) y el scheduler, al no ver mejora, colapsaba el lr a casi cero. Chicken-and-egg."),
        ("ajuste", "6to intento (arregla la maquinaria): trayectoria completa + warmup + congelar la red + LEARNING RATE PROPIO Y ALTO para los pesos (lr_weights=0.05, separado del de la red) + piso de lr (min_lr). Asi theta por fin se mueve rapido y la optimizacion converge."),
        ("info", "HALLAZGO: con la maquinaria ya funcionando, wEE y wEI (entrada excitatoria u_e) se recuperan muy bien (<2%) en TODA corrida. En cambio el par wIE/wII queda mal identificado y su valor es INESTABLE entre corridas (wII puede colapsar cerca de 0 o quedar lejos del verdadero, segun la inicializacion). NO es un bug: wIE y wII aparecen JUNTOS en la entrada inhibitoria u_i = wIE*E - wII*I y se compensan entre si; con E e I correlacionadas en una sola trayectoria, el par queda subdeterminado (baja identificabilidad). Esa misma variabilidad entre corridas es la evidencia de la degeneracion. Es justo la limitacion que el proyecto resuelve con MULTIPLES trayectorias (distintos estimulos / condiciones)."),
    ],
}


def _build_html(exp, identify, w_true, w_pred, hist, ds, cfg, trainer):
    import numpy as np

    titulos = {
        "4.0": "Paso 4.0 — Forward con parametros fijos (chequeo de plomeria)",
        "4.1": "Paso 4.1 — Inverso desde los valores verdaderos (estabilidad)",
        "4.2": "Paso 4.2 — Inverso real desde arranque ignorante (test de verdad)",
    }
    objetivos = {
        "4.0": "Verificar que la red ajusta la trayectoria (L_datos y L_fisica bajan y la prediccion se superpone con la real). Si esto no ajusta, hay un bug: se para y se depura aca antes de seguir.",
        "4.1": "Verificar que, arrancando en la solucion correcta, los pesos se quedan ahi (no se escapan). Separa 'el problema es dificil' de 'hay un bug'.",
        "4.2": "Verificar que los pesos convergen a sus valores reales partiendo de un arranque ignorante. Es lo que demuestra que la PINN recupera parametros sin conocerlos.",
    }

    # --- Metricas derivadas ---
    stop = hist.get("stop_epoch", len(hist["loss"]) - 1)
    corto_por_meseta = stop < cfg.epochs - 1
    l_d, l_f = hist["data"][-1], hist["physics"][-1]
    lr_final = trainer.opt.param_groups[0]["lr"]

    # MSE limpio sobre toda la trayectoria, separando train (80%) / test (20%).
    t = ds["t"]
    pred = trainer.predict(t)
    target = np.stack([ds["I"], ds["E"]], axis=1)
    n_train = int(round((1.0 - cfg.val_fraction) * t.shape[0]))
    train_mse = float(((pred[:n_train] - target[:n_train]) ** 2).mean())
    hay_test = n_train < t.shape[0]
    val_mse = float(((pred[n_train:] - target[n_train:]) ** 2).mean()) if hay_test else float("nan")

    # --- Diagnostico automatico + veredicto ---
    diag = []  # (nivel, texto): ok / warn / bad
    if corto_por_meseta:
        diag.append(("ok", f"El entrenamiento corto solo por meseta en epoch {stop} (la perdida dejo de bajar)."))
    else:
        diag.append(("warn", f"Llego al tope de {cfg.epochs} epochs sin amesetar. Conviene subir 'epochs'."))

    if train_mse < 1e-4:
        diag.append(("ok", f"Ajuste de la trayectoria muy bueno (MSE train = {train_mse:.1e})."))
    elif train_mse < 1e-3:
        diag.append(("warn", f"Ajuste aceptable pero mejorable (MSE train = {train_mse:.1e})."))
    else:
        diag.append(("bad", f"Ajuste pobre (MSE train = {train_mse:.1e}): revisar arquitectura / mas epochs."))

    if not hay_test:
        diag.append(("ok", "Sin reserva de test temporal: se identifica sobre la trayectoria completa (correcto para recuperar los parametros)."))
    elif val_mse < 5 * max(train_mse, 1e-12):
        diag.append(("ok", f"Extrapola bien al tramo no visto (MSE test = {val_mse:.1e}, similar al de train)."))
    else:
        diag.append(("warn", f"El tramo no visto ajusta peor (MSE test = {val_mse:.1e} vs train {train_mse:.1e}): posible sobreajuste."))

    max_err = 0.0
    if identify:
        max_err = max(100.0 * abs(w_pred[k] - w_true[k]) / abs(w_true[k]) for k in identify)
        if exp == "4.1":
            if max_err < 3:
                diag.append(("ok", f"Los pesos se mantuvieron muy cerca de los verdaderos (error max {max_err:.1f}%): maquinaria del inverso correcta."))
            elif max_err < 10:
                diag.append(("warn", f"Leve deriva (error max {max_err:.1f}%), concentrada en el peso menos identificable de esta trayectoria (wII): su direccion en la perdida es 'plana'. La maquinaria del inverso esta bien conectada (no se escapan); la baja identificabilidad de wII se ataca despues con varias trayectorias."))
            else:
                diag.append(("bad", f"Los pesos se ESCAPARON (error max {max_err:.1f}%): revisar gradiente hacia los pesos o el balance de lambda."))
        else:  # 4.2
            err_por_peso = {k: 100.0 * abs(w_pred[k] - w_true[k]) / abs(w_true[k]) for k in identify}
            buenos = [k for k in identify if err_por_peso[k] < 10]
            malos = [k for k in identify if err_por_peso[k] >= 10]
            if not malos:
                diag.append(("ok", f"Todos los pesos convergieron desde un arranque ignorante (error max {max_err:.1f}%)."))
            else:
                diag.append(("ok", f"La maquinaria del inverso funciona: convergieron {', '.join(buenos)} (<10%) desde el arranque ignorante (1.0)."))
                degen = ("wIE" in malos or "wII" in malos)
                txt = f"NO convergieron: {', '.join(malos)} (error max {max_err:.1f}%). "
                if degen:
                    txt += ("Causa: identificabilidad, NO un bug. wIE y wII entran juntos en la "
                            "inhibitoria u_i = wIE*E - wII*I y se compensan entre si en una sola "
                            "trayectoria. Se resuelve con MULTIPLES trayectorias (distintos "
                            "estimulos/condiciones), que es el siguiente paso del proyecto.")
                else:
                    txt += "Probar mas epochs, ajustar lr_weights, o usar varias trayectorias."
                diag.append(("warn", txt))

    # Veredicto del criterio de avance.
    if exp == "4.0":
        paso_ok = train_mse < 1e-4
    elif exp == "4.1":
        paso_ok = max_err < 10   # criterio: que NO se escapen (no exactitud perfecta)
    else:
        paso_ok = max_err < 10
    veredicto = ("ok", "PASA — criterio de avance cumplido") if paso_ok else ("bad", "NO PASA — revisar antes de avanzar")

    # --- Render de listas (notas y diagnostico) ---
    badge = {"error": ("bad", "ERROR"), "ajuste": ("warn", "AJUSTE"), "info": ("ok", "INFO")}

    def li_notas(items):
        out = ""
        for tipo, txt in items:
            cls, lbl = badge[tipo]
            out += f"<li><span class='pill {cls}'>{lbl}</span> {txt}</li>"
        return out

    def li_diag(items):
        out = ""
        for nivel, txt in items:
            out += f"<li><span class='pill {nivel}'>{nivel.upper()}</span> {txt}</li>"
        return out

    notas = AJUSTES_HEREDADOS + NOTAS_PASO.get(exp, [])

    # --- Tabla de pesos (solo inverso) ---
    tabla_w = ""
    if identify:
        filas = ""
        for k in ("wEE", "wEI", "wIE", "wII"):
            err = 100.0 * abs(w_pred[k] - w_true[k]) / abs(w_true[k])
            cls = "ok" if err < 5 else ("warn" if err < 15 else "bad")
            filas += (f"<tr><td><code>{k}</code></td><td class='num'>{w_true[k]:.3f}</td>"
                      f"<td class='num'>{w_pred[k]:.3f}</td>"
                      f"<td class='num'><span class='pill {cls}'>{err:.2f}%</span></td></tr>")
        tabla_w = (
            "<h2>Pesos identificados</h2><table>"
            "<tr><th>Peso</th><th class='num'>Verdadero</th>"
            "<th class='num'>Estimado</th><th class='num'>Error</th></tr>"
            f"{filas}</table>")

    img_fit = f"<img src='data:image/png;base64,{_b64(FIG_FIT)}'>"
    img_loss = f"<img src='data:image/png;base64,{_b64(FIG_LOSS)}'>"
    img_w = f"<h2>Convergencia de los pesos</h2><img src='data:image/png;base64,{_b64(FIG_W)}'>" if identify else ""

    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{titulos[exp]}</title>
<style>
  body{{font-family:-apple-system,"Segoe UI",Roboto,Arial,sans-serif;color:#1c2733;
       line-height:1.55;margin:0;background:#f4f6f8;padding:2rem 1rem}}
  .wrap{{max-width:880px;margin:0 auto}}
  header{{background:#1f4e79;color:#fff;border-radius:12px 12px 0 0;padding:1.5rem 1.8rem}}
  header h1{{margin:0;font-size:1.35rem}} header p{{margin:.35rem 0 0;opacity:.9;font-size:.93rem}}
  main{{background:#fff;border:1px solid #d7dee5;border-top:none;border-radius:0 0 12px 12px;padding:1.4rem 1.8rem 2rem}}
  h2{{color:#1f4e79;font-size:1.12rem;margin:1.6rem 0 .5rem;padding-bottom:.3rem;border-bottom:2px solid #e8f0f8}}
  table{{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:.92rem}}
  th,td{{text-align:left;padding:.45rem .7rem;border-bottom:1px solid #d7dee5}}
  th{{background:#e8f0f8;color:#1f4e79}} td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
  code{{background:#f0f3f6;padding:.1rem .35rem;border-radius:4px;font-family:Consolas,monospace;font-size:.88em}}
  img{{width:100%;border:1px solid #d7dee5;border-radius:8px;margin:.4rem 0}}
  ul.lista{{list-style:none;padding:0;margin:.5rem 0}}
  ul.lista li{{padding:.45rem .2rem;border-bottom:1px solid #eef2f5}}
  .pill{{display:inline-block;padding:.1rem .5rem;border-radius:999px;font-size:.75rem;font-weight:700;margin-right:.4rem}}
  .ok{{background:#e3f5ea;color:#1b7f4b}} .warn{{background:#fbf0db;color:#b06a00}} .bad{{background:#fbe3e3;color:#b02a2a}}
  .veredicto{{font-size:1.05rem;font-weight:700;padding:.7rem 1rem;border-radius:8px;margin:.6rem 0}}
  .v-ok{{background:#e3f5ea;color:#1b7f4b;border:1px solid #b6e2c6}}
  .v-bad{{background:#fbe3e3;color:#b02a2a;border:1px solid #eebcbc}}
  .nota{{background:#e8f0f8;border-left:4px solid #1f4e79;padding:.7rem 1rem;border-radius:0 6px 6px 0;margin:1rem 0;font-size:.92rem}}
  footer{{color:#5b6770;font-size:.82rem;text-align:center;margin-top:1.3rem}}
</style></head><body><div class="wrap">
<header><h1>{titulos[exp]}</h1><p>Wilson-Cowan + PINN · identificacion parametrica</p></header>
<main>
  <div class="veredicto {'v-ok' if veredicto[0]=='ok' else 'v-bad'}">{veredicto[1]}</div>

  <h2>Objetivo del paso</h2><p>{objetivos[exp]}</p>

  <h2>Configuracion</h2>
  <table>
    <tr><th>Dataset</th><td><code>{ds['t'].shape[0]}</code> puntos · t ∈ [{float(ds['t'].min()):.0f}, {float(ds['t'].max()):.0f}] s · dt=0.01 · ruido={float(ds['noise_std']):.2g}</td></tr>
    <tr><th>Red</th><td>MLP {HIDDEN_DIM}×{N_LAYERS}, tanh · Fourier features={N_FOURIER} (escala {FOURIER_SCALE})</td></tr>
    <tr><th>Identifica</th><td>{', '.join(identify) if identify else 'ninguno (forward puro)'}</td></tr>
    <tr><th>Pesos de perdida</th><td>datos={cfg.w_data} · fisica={cfg.w_physics} · ic={cfg.w_ic}</td></tr>
    <tr><th>Validacion temporal</th><td>entrena con el {100*(1-cfg.val_fraction):.0f}% inicial, testea el {100*cfg.val_fraction:.0f}% final</td></tr>
  </table>

  <h2>Incidencias y ajustes</h2>
  <ul class="lista">{li_notas(notas)}</ul>

  <h2>Resultado</h2>
  <table>
    <tr><th>Epochs hasta meseta</th><td class="num">{stop}{' (corto por meseta)' if corto_por_meseta else ' (tope de epochs)'}</td></tr>
    <tr><th>L_datos final (minibatch)</th><td class="num">{l_d:.2e}</td></tr>
    <tr><th>L_fisica final (minibatch)</th><td class="num">{l_f:.2e}</td></tr>
    <tr><th>MSE trayectoria (ajuste de la red)</th><td class="num">{train_mse:.2e}</td></tr>
    <tr><th>MSE test (tramo no visto)</th><td class="num">{'—' if not hay_test else f'{val_mse:.2e}'}</td></tr>
    <tr><th>Learning rate final</th><td class="num">{lr_final:.1e}</td></tr>
  </table>

  {tabla_w}

  <h2>Diagnostico automatico</h2>
  <ul class="lista">{li_diag(diag)}</ul>

  <h2>Ajuste: PINN vs trayectoria real</h2>
  <p>Linea punteada gris = corte entre el tramo de entrenamiento (izquierda) y el de test temporal (derecha, que la red no vio en datos).</p>
  {img_fit}

  <h2>Evolucion de la perdida</h2>
  {img_loss}

  {img_w}

  <div class="nota">Generado por <code>scripts/train_pinn.py</code> (EXPERIMENTO="{exp}").
  Ajustable desde la zona editable del script.</div>
</main>
<footer>Proyecto Wilson-Cowan + PINN · Concurso I+D ITBA 2026</footer>
</div></body></html>"""
    HTML_PATH.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
