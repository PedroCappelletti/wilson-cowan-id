# Resultado — Identificación NODE por familia y el límite del dt grande

> **Qué se probó:** cerrar el hueco de identificar con **Neural ODE (10 params) entrenando por familia de estímulo** (lo que solo se había hecho con la PINN y con FIM sin entrenar), para **contrastar el ranking empírico con la predicción CRB del Exp C** (OED). Se corrió con el **lever de dt grande** (dt≈0.2, 4× más barato) que habíamos validado en Costo computacional del Neural ODE - qué reducir y cómo estimarlo.
>
> **El giro:** el resultado quedó **dominado por el dt**, no por la identificabilidad. Sirvió igual — como **límite del lever de dt grande**, con el chirp como confirmación puntual. Fuente: `scripts/exp_family_neural_ode.py`, `results/ident_family_neural_ode.json`.

## ⚙️ Setup
- Familias (5): **chirp, poisson, aprbs, square, thetagamma**. 4 trayectorias/familia (P,Q decorrelados), σ=0.05, smooth_k=7.
- **dt≈0.2** (N_EVAL 4000→1000, WINDOW 100→25 para mantener la ventana en ~5 unidades). EPOCHS=1500 + L-BFGS. ETA medido: ~23 min (vs ~108 min a dt fino — el lever anduvo, ~4×).

## 📊 Resultado (error de θ̂ por familia)

| Familia | wII err | máx err | Tipo de estímulo | Veredicto |
|---|---|---|---|---|
| **chirp** | **2.9 %** | 7.3 % (ae) | **suave / banda limitada** | ✅ identifica bien |
| thetagamma | 86 % | 86 % | ráfagas gamma on/off | ❌ colapsa |
| square | 247 % | 247 % | onda cuadrada on/off | ❌ colapsa |
| aprbs | 285 % | 285 % | escalones aleatorios | ❌ colapsa |
| poisson | 403 % | 403 % | pulsos cortos | ❌ colapsa (casi todo >100 %) |

## 🔑 La lectura correcta (no es un ranking de identificabilidad)
**La única familia que identificó bien es la suave (chirp). Todas las conmutadas/discontinuas colapsaron.** Eso **no** es la identificabilidad intrínseca de cada estímulo — es el **artefacto del dt grande** que elegimos:

- A **dt≈0.2 con ZOH** (el estímulo se mantiene constante por paso), los estímulos **on/off** quedan mal resueltos y **temporalmente desalineados** con los datos (que se generaron con integración fina). El modelo compensa ese mismatch **distorsionando los parámetros** → wII explota.
- El **chirp es suave y de banda limitada** → el muestreo grueso casi no lo afecta → identifica bien aun a dt≈0.2.
- Sabemos que estas familias **no** son malas a dt fino: en la mezcla (`strong_scenarios`, dt≈0.05) aprbs/square/theta-gamma identifican bien, y `compare_estimulos` (PINN) mostró **APRBS y theta-gamma entre las mejores**. El colapso de acá es del dt, no del estímulo.

## 🔁 Control a dt fino (0.05) — y por qué el test sigue SIN ser concluyente
Se corrió el mismo experimento cambiando **solo el dt** (0.2→0.05), σ=0.05. Comparación del error de **wII**:

| Familia | dt grande (0.2) | dt fino (0.05) | CRB pred (Exp C) | ¿Qué pasó? |
|---|---|---|---|---|
| chirp | **2.9 %** | **50.8 %** | 1.81 (mejor) | ⚠️ **empeoró 17×** |
| poisson | 402 % | 401 % | 2.25 | roto en ambos |
| aprbs | 285 % | **16.2 %** | 2.43 | ✅ se recuperó |
| square | 247 % | 231 % | 35.5 (peor) | roto en ambos |
| thetagamma | 86 % | **16.9 %** | 6.86 | ✅ se recuperó |

Figura: `attachments/family_compare_dt.png` (barras dt-grande vs dt-fino) y `attachments/family_neural_ode_dtfino.png`.

**El dt NO era todo el problema.** A dt fino se recuperaron los conmutados que dt grande rompía (aprbs, thetagamma) — ahí el dt sí era artefacto. Pero **chirp EMPEORÓ** (2.9→50.8 %) y poisson/square siguen rotos. Los rankings de las dos corridas ni coinciden entre sí ni con la CRB.

**Conclusión honesta: el experimento NO testea el ranking OED.** Dos fallas de diseño lo invalidan:
1. **Varianza de optimización enorme.** chirp saltó 2.9 %↔50.8 % con el **mismo seed**, solo cambiando dt → con **1 semilla y 4 trayectorias** los números caen en mínimos locales; no son confiables para rankear.
2. **Los estímulos NO son los que rankeó la CRB, y están dispares en excitación.** La CRB del Exp C se calculó sobre **otras instancias** de cada familia. Mi `poisson` acá es rate=0.06, pulse_width=1 → ~11 pulsos en 180 unidades = **casi no excita** → falla siempre; eso es *mi instancia mala*, no "poisson malo" (la CRB lo daba 2º). Comparar predicción e empírico exige que sean **el mismo estímulo**.

## 🧩 Lo único que queda firme
- **dt fino recupera los estímulos conmutados** (aprbs 285→16 %, thetagamma 86→17 %) → confirma que para on/off el dt grande es artefacto. Coincide con lo que sabíamos de la mezcla y de `compare_estimulos` (PINN): aprbs/theta-gamma son buenos.
- **Refina el hallazgo de costo:** el dt grande solo es seguro para estímulos **suaves / banda limitada**; para conmutados/discontinuos hace falta dt fino → ⚠️ agregado a Costo computacional del Neural ODE - qué reducir y cómo estimarlo.
- **NO** queda confirmado ni refutado el ranking CRB: el diseño no alcanza.

## ▶️ Cómo seguir (el test bien hecho)
1. **CRB sobre el MISMO estímulo que se entrena** (no las instancias del Exp C): calcular la CRB(wII) con la maquinaria de `fisher_identifiability.py` sobre las trayectorias exactas de cada familia → recién ahí predicción y empírico son comparables.
2. **Múltiples semillas por familia (≥5)** → barras de error, para vencer la varianza de optimización (lo que hizo saltar a chirp).
3. **Excitación pareja entre familias** (misma amplitud/cobertura) o al menos reportarla; arreglar poisson (subir rate) para que excite.
4. **Más trayectorias/familia** (8, como la mezcla).
5. Recién con (1)–(4): scatter CRB predicha vs error empírico con barras de error → confirmar/refutar la predicción OED.

## 🔗 Enlaces
- Proyecto: Investigación Neurociencia · Plan: Plan de trabajo - Neural ODE (Eje 1 y 3)
- Predicción a contrastar (OED): Resultado - Subset selection, regularización y diseño de estímulo (Exp C)
- Lever de dt (y su límite, ahora): Costo computacional del Neural ODE - qué reducir y cómo estimarlo
- Previos: Resultado - Robustez al ruido (10 params) · Resultado - Identificación completa 10 parámetros
- Notas: Neural ODEs · Modelo Wilson-Cowan
