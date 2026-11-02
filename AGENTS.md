# AGENTS.md: decisiones y reglas del proyecto

Leelo antes de correr, escribir o commitear. Cada regla de acá es una corrección que ya se hizo una vez; si se reintroduce el error, alguien la tiene que volver a corregir. Cuando tomes una decisión nueva o te corrijan algo, agregalo acá en el mismo commit.

Equipo: Ralf Högner, Pedro Cappelletti y Juan Ignacio Hanachian.

## 1. Repositorios

- **Este repo** (`PedroCappelletti/wilson-cowan-id`) es el de trabajo: experimentos, resultados, checkpoints y `docs/`. Se commitea y se pushea directo a `main`. Nunca force push sobre `main`.
- **El repo limpio** (`rhogner/wilson-cowan-graybox`) tiene solo la línea gray-box publicable. Es **privado a propósito**: con siete autores, cuatro filiaciones y financiamiento del ITBA no está confirmado quién puede licenciar el código. Pasarlo a público se decide con la dirección del grupo. Lo que migra de acá al limpio es una decisión de Ralf, no un `git push`.
- **No cambiar de rama con corridas en curso.** Un `git switch` sacó del disco datasets que una cola iba a necesitar.

## 2. Commits

- Sin `Co-Authored-By: Claude` ni «Generated with Claude Code», ni en commits ni en PR, salvo que Ralf lo pida.
- Comentarios y docstrings en castellano **con acentos y ñ**, aunque el código viejo los omita. Los identificadores siguen en ASCII.

## 3. Controlador: etapa actual

- **En esta etapa el controlador no se modifica.** Es el IMC del paper tal cual: ley, ganancias ($k_{p,I}=10$, $k_{i,I}=5$, $k_{p,E}=5$, $k_{i,E}=5$), recorte y parámetros del `.m` original (`simulador_wilson_cowan_con_control.m`), sin cota ni anti-windup. El port es `IMCController` (`src/neural_ode/closed_loop.py`) y está verificado contra el `.m` integrado con RK45; la diferencia es proporcional a $\Delta t$.
- **El modelo identificado va como planta, no dentro del controlador.** Cada experimento corre el mismo controlador sobre dos plantas: el simulador perturbado y el modelo gray-box identificado a partir de los datos de ese simulador. Script: `scripts/control_paper_planta_modelo.py`; tabla y figuras: `scripts/resumen_control_planta_modelo.py`; documento: `docs/controlador_planta_modelo.md`.
- Para usar un modelo como planta, `planta_de(modelo)` reescribe su campo en numpy (con PyTorch una corrida de 2 s tarda más de diez minutos). Antes de usar un modelo nuevo, correr `python scripts/control_paper_planta_modelo.py --verificar`: compara contra el `forward` de PyTorch y tiene que dar ~1e-7 relativo. El modelo de historia recibe los comandos aplicados en $t - k \cdot 0{,}5$ ms (el paso de los datos); el de estado de filtro integra $\hat P$, $\hat Q$ como estados de la planta.
- Lo anterior, con el modelo dentro del controlador (sus $\hat\beta$, cancelar $g$, invertir la forma), está en `scripts/control_paper_modelos.py` y en `docs/controlador_paper_simulador_y_modelos.md`. Los controladores extendidos (`IMCExtendido`, Newton, compensación del filtro) están en `src/neural_ode/closed_loop_ext.py` y en `docs/controladores_*.md`. **No son la etapa actual.** No se mezclan con el controlador del paper en los documentos de arquitectura ni de resultados del paper.

## 4. Corridas

- **Siempre con ruido.** Sin ruido solo para probar rápido un cambio de arquitectura; todo resultado que se reporte se corre también con ruido (`*_n01`, `*_n05`, de `scripts/esc_gen_ruido.py`), con media móvil de 7 muestras (`--smooth 7`). Todo texto dice con qué ruido corrió y reporta los dos niveles. Lo que dio negativo sin ruido no se repite con ruido, salvo las defensas contra el ruido.
- **σ = 0,01 y 0,05 son valores equivalentes a 20 kHz.** En los datos de 2 kHz, por muestra, valen 0,0032 y 0,016 (`esc_gen_ruido.py --ref-fs`).
- **Siempre con warm start.** Cada configuración se corre dos veces: desde cero ($\beta = 1$) y desde el white-box (`--init-params` con el JSON del white-box de la misma planta, dataset y ruido; tag `_wrm`). **La `_wrm` va primero en la cola** y espera a que exista el JSON del white-box. El white-box no lleva `_wrm`.
- **Selección por validación, nunca por test.** `esc_run.py` se queda con el estado de mejor NRMSE de validación (`MejorVal`). Las corridas anteriores al 3/10/2026 evaluaban la última época: revisarlas con `scripts/divergencia.py` antes de sacar conclusiones.
- El NRMSE con ruido no se compara contra el limpio, porque la normalización por rango pico a pico se infla con el ruido. El error de parámetros sí es comparable.
- **Máximo tres corridas en paralelo.** En lotes largos, lanzar `scripts/despierto.py` junto con la cola para que Windows no suspenda. Para contar procesos vivos usar `Get-CimInstance Win32_Process`; el `ps` de Git Bash no ve los hijos de `subprocess`.

## 5. Nomenclatura

- La corrección $g$ es **«agnóstica»** (*agnostic*), sin complemento. Nunca «ciega», «blind», «genérica» ni «agnóstica a la física».
- **«entrenable»** (*trainable*), nunca «aprendida» (*learned*). El filtro FIR es el «filtro entrenable».
- Las variantes que agregan el término faltante (forma exacta, estado de filtro) **«expanden el backbone»**. Nunca «física escrita a mano».
- Parámetros identificados: **$\hat\beta$**. $\theta_e$, $\theta_i$ son los umbrales; no usar $\theta$ para el vector de parámetros.
- No atribuir a una causa lo que no se midió. Las dos plantas están confundidas en magnitud: no generalizar desde una.

## 6. Documentos

- **Sin autores ni fecha**, sin sección final de datos y sin referencia bibliográfica final. La sección de archivos y funciones va solo si el documento la necesita (el de arquitectura).
- **Los archivos se nombran por lo que contienen**, nunca por para quién son (nada de «tutor» en nombres de scripts, resultados, figuras ni PDF).
- Sin rayas (—). Sin la construcción «no es X, es Y»: decir qué hace la cosa. Sin dos puntos explicativos.
- Pies de figura descriptivos, estilo paper: qué se grafica y en qué condición. Sin narrativa ni metáforas. Las conclusiones van en el cuerpo, si el documento las pide.
- En un documento de resultados, si se pide «sin conclusiones», las tablas y figuras van sin comentarios.
- Cuando una variante tiene red pero el controlador no la usa, decirlo explícito en la tabla.
- PDF: `pandoc --pdf-engine=xelatex -V mainfont=Arial -V monofont=Consolas -V geometry:a4paper,margin=2cm -V lang=es -V fontsize=10pt`. Revisar el PDF renderizado antes de entregar (diagramas, tablas que se cortan, pies que saltan de página).
