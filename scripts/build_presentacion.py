#!/usr/bin/env python3
# =============================================================================
#  ARMA LA PRESENTACION HTML  (un solo archivo, autocontenido)
# =============================================================================
#
#  Estructura:
#     1. Introduccion — que usabamos antes y que usamos ahora
#     2. La pregunta central — ¿el gray-box copia al simulador?
#     3. Que paso con los parametros (resumen)
#
#  Las imagenes van embebidas en base64: el .html se puede mandar por mail o
#  abrir desde cualquier lado sin arrastrar la carpeta de figuras.
#
#  Navegacion: flechas, espacio, clic. F = pantalla completa. La diapositiva
#  queda en la URL (#7), asi que se puede enlazar.
#
#  USO:  python scripts/build_presentacion.py
#
#  ---------------------------------------------------------------------------
#  QUE CUENTA LA CHARLA (para no perder el hilo al editar diapositivas):
#    el simulador y el modelo eran las MISMAS ecuaciones, asi que el termino de
#    correccion neuronal g_phi no tenia nada que aprender. Ahora al SIMULADOR se
#    le agrego fisica que el modelo no contempla (refractariedad del Wilson-Cowan
#    de 1972 + un actuador optogenetico con retardo y saturacion) y la perilla
#    epsilon gradua cuanta fisica falta. Referencia: docs/graybox_manual_completo.md
#
#  COMO ESTA ARMADO ESTE SCRIPT (tres piezas que se pegan al final):
#    - diapos()  -> lista de strings, cada uno un <section> completo.
#    - CSS       -> la hoja de estilos entera, como un string.
#    - JS        -> el motor de navegacion, como un string.
#    main() los concatena en un unico .html y lo escribe en OUT. No hay
#    plantillas ni dependencias externas: se corre con Python pelado.
#
#  TRAMPAS AL EDITAR:
#    - Las rutas FIG/OUT son RELATIVAS: hay que correrlo desde la raiz del repo
#      (python scripts/build_presentacion.py), no desde scripts/.
#    - Una diapositiva que inserta figuras se escribe como f-string. Si le metes
#      llaves literales (por ejemplo CSS inline o {} en el texto) hay que
#      DUPLICARLAS ({{ }}), o Python las interpreta como campo a sustituir.
#    - El numero de diapositiva sale de la POSICION en la lista S: si insertas
#      una en el medio, los enlaces tipo "#7" que hayas mandado se corren.
# =============================================================================

from __future__ import annotations

import base64
from pathlib import Path

# Rutas relativas a la raiz del repo (ver "trampas" arriba).
#   FIG: los PNG ya generados por los scripts de figuras; este script no dibuja
#        nada, solo empaqueta lo que encuentra ahi.
#   OUT: el entregable, un unico archivo que se abre con doble clic.
FIG = Path("docs/figuras_presentacion")
OUT = Path("docs/presentacion_graybox.html")


# =============================================================================
#  IMAGENES EMBEBIDAS (base64)
# =============================================================================
# Por que base64 y no <img src="figuras/x.png">: el HTML tiene que ser
# AUTOCONTENIDO. Con rutas relativas, mandar la presentacion por mail o subirla
# a Drive rompe todas las figuras (falta la carpeta). Embebida, el archivo pesa
# mas (unos cientos de KB) pero es un solo adjunto que funciona siempre y
# tambien offline.
# El costo real: el .html no se puede versionar por diff (es un blob) y hay que
# regenerarlo cada vez que se rehace una figura.

def img(nombre: str) -> str:
    """Lee docs/figuras_presentacion/<nombre>.png y lo devuelve como data URI.

    'nombre' va SIN extension y sin carpeta. Si el PNG no existe, esto explota
    con FileNotFoundError al construir la presentacion: es a proposito, mejor
    fallar aca que entregar una diapositiva con un hueco.
    """
    b = (FIG / f"{nombre}.png").read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()


def fig(nombre: str, clase: str = "") -> str:
    """Envuelve una figura en el <div class="fig"> que la escala a la diapositiva.

    El div es el que hace el trabajo: se estira con el espacio que sobra
    (flex:1) y la imagen entra con object-fit:contain, asi que la figura nunca
    se deforma ni empuja el texto de abajo fuera de la pantalla.
    'clase' agrega modificadores; el unico que existe hoy es "med", que limita
    la figura al 55% de la altura para dejarle aire a un bloque de conclusion.
    """
    c = f" {clase}" if clase else ""
    return f'<div class="fig{c}"><img src="{img(nombre)}"></div>'


# =============================================================================
#  LAS DIAPOSITIVAS
# =============================================================================
#  Cada elemento de la lista es un <section> completo, y el ORDEN de la lista es
#  el orden de la charla. Solo la seccion con clase "on" se muestra (lo pone el
#  JS); las demas quedan con display:none, no hay animaciones ni librerias.
#
#  Convenciones de las clases que se repiten (el detalle esta en el bloque CSS):
#    section.portada / section.seccion  -> tapa y separadores de parte
#    h2                                 -> titulo de diapositiva con filete
#    .dos-pie                           -> las dos columnas de lectura al pie
#    .dato / .clave                     -> el recuadro de remate abajo
#    .tag / .num / .ex / .exi           -> etiquetitas de color y numeros grandes
#
#  Los numeros que aparecen en el texto (2 %, 15 %, 59 %, x17, 48 %) NO se
#  calculan aca: estan escritos a mano y salen de las corridas de
#  scripts/run_uncertainty_all.sh. Si se vuelve a entrenar, hay que actualizarlos
#  a mano en las diapositivas y en la de "Para llevarse".
# =============================================================================
def diapos() -> list[str]:
    """Devuelve la lista de diapositivas, ya en orden, como HTML."""
    S = []

    # ---- Portada --------------------------------------------------------
    # Enuncia la tension de toda la charla en una frase: la igualdad
    # simulador == modelo se rompio a proposito.
    S.append("""
<section class="portada">
  <h1>Gray-box en Wilson–Cowan</h1>
  <p class="sub">¿Puede el modelo copiar al cerebro cuando le falta física?</p>
  <div class="frase">
    Antes, el simulador y el modelo eran <em>las mismas ecuaciones</em>.<br>
    Le rompimos esa igualdad a propósito — y encendimos la red.
  </div>
</section>""")

    # =====================================================================
    #  PARTE 1 — INTRODUCCION: de donde venimos y que se cambio
    # =====================================================================
    #  Arco de la parte: (a) antes las dos ecuaciones eran iguales, (b) que
    #  fisica se le agrego al simulador, (c) como se ve ese hueco en datos,
    #  (d) recien ahi tiene sentido prender la red g_phi.
    #  Las diapositivas de separador llevan class="seccion" (+ "nara"/"gris"
    #  para el color del rotulo "Parte N").
    S.append("""
<section class="seccion">
  <div class="setiq">Parte 1</div>
  <h1 class="setit">Qué usábamos antes<br>y qué usamos ahora</h1>
</section>""")

    # 1.a — El punto de partida: mismas ecuaciones a los dos lados. Es la
    # diapositiva que justifica todo el trabajo, porque explica por que el 1 % de
    # error de antes era un resultado comodo (el modelo era perfecto por
    # construccion) y por que g_phi estaba en el codigo sin usarse.
    S.append("""
<section>
  <h2>ANTES · el simulador y el modelo eran lo mismo</h2>
  <div class="dos">
    <div class="caja">
      <div class="rot">SIMULADOR</div>
      <div class="ec">ẋ = f<sub>WC</sub>(x, P, Q; θ)</div>
      <div class="pie2">genera los datos<br>“el cerebro”</div>
    </div>
    <div class="igual">=</div>
    <div class="caja">
      <div class="rot">MODELO</div>
      <div class="ec">ẋ = f<sub>WC</sub>(x, P, Q; θ)</div>
      <div class="pie2">lo que entrenamos<br>“nuestra hipótesis”</div>
    </div>
  </div>
  <p class="grande">Las <strong>mismas ecuaciones</strong> de los dos lados.</p>
  <div class="dos-pie">
    <div><span class="tag ok">bien</span> Identificábamos los 10 parámetros con
    <strong>1 % de error</strong>.</div>
    <div><span class="tag mal">pero</span> El problema era <em>“dado que el
    modelo es perfecto, encontrale los parámetros”</em>. En un experimento real
    tu modelo siempre está incompleto.</div>
  </div>
  <div class="dato">Consecuencia concreta: el término de corrección
  <strong>g<sub>φ</sub></strong> estaba en el código desde el principio y
  <strong>nunca se había usado</strong> — porque no tenía nada que aprender.</div>
</section>""")

    # 1.b — Las dos perturbaciones, con la ecuacion a la vista. El detalle que
    # hay que decir en voz alta: la fisica se agrego AL SIMULADOR, no al modelo,
    # y no es invento nuestro (la refractariedad es el WC de 1972 que la version
    # reducida descarta). En las ecuaciones, class="dim" es lo que ya estaba y
    # class="hi" (naranja) es lo nuevo: el contraste es el mensaje.
    S.append("""
<section>
  <h2>AHORA · le agregamos física al <span class="ac">simulador</span></h2>
  <p class="chica top">Al simulador, <strong>no al modelo</strong>. La red nunca
  ve estas ecuaciones ni sabe que existen.</p>

  <div class="pert">
    <div class="pnum">1</div>
    <div class="pcont">
      <div class="ptit">Refractariedad
        <span class="ptag">lado del estado</span></div>
      <div class="ecg">
        <span class="dim">dI/dt = (1/τ<sub>i</sub>) · ( −I +</span>
        <span class="hi">(1 − r·I)</span>
        <span class="dim">· S<sub>i</sub>(u<sub>i</sub>) − k<sub>i</sub> )</span>
      </div>
      <div class="pexp">Una neurona que acaba de disparar no puede volver a
      disparar enseguida. <strong>Es el Wilson–Cowan original de 1972</strong> —
      la versión reducida lo descarta. No inventamos nada: lo
      <em>deshicimos</em>.</div>
    </div>
  </div>

  <div class="pert">
    <div class="pnum">2</div>
    <div class="pcont">
      <div class="ptit">Actuador optogenético
        <span class="ptag">lado de la entrada</span></div>
      <div class="ecg">
        <span class="hi">dP<sub>lag</sub>/dt = (P<sub>cmd</sub> −
        P<sub>lag</sub>) / τ<sub>act</sub></span>
        <span class="dim">&nbsp;&nbsp;·&nbsp;&nbsp;</span>
        <span class="hi">P<sub>ef</sub> = A·tanh(P<sub>lag</sub>/A)</span>
      </div>
      <div class="pexp">Comandás <em>luz</em>; lo que la neurona recibe llega
      tarde y saturado. Rompe el supuesto de que el estímulo es perfectamente
      conocido.</div>
    </div>
  </div>

  <div class="dato">Una perilla <strong>ε</strong> gradúa cuánta física falta.
  <strong>ε = 0</strong>: ninguna (como antes). <strong>ε = 1</strong>: el hueco
  vale un tercio del campo.</div>
</section>""")

    # 1.c — La evidencia de que el hueco existe y es del tamano justo: el ciclo
    # se deforma pero sigue oscilando (si desapareciera, el experimento no
    # mediria nada) y el estimulo comandado no es el que llega.
    S.append(f"""
<section>
  <h2>Así se ve el hueco, en datos reales</h2>
  {fig('p_perturbacion')}
  <div class="dos-pie">
    <div><span class="tag azul">izq</span> El ciclo se deforma pero
    <strong>no desaparece</strong>: sigue oscilando.</div>
    <div><span class="tag nara">der</span> Lo que comandás contra lo que llega.
    <strong>El modelo sólo conoce el azul.</strong></div>
  </div>
</section>""")

    # 1.d — El gray-box en si: backbone de ecuaciones (10 parametros
    # interpretables) + red chica de correccion. El bloque .verq de abajo es el
    # que evita el malentendido mas comun de la charla: enfrenta lo que el
    # simulador TIENE contra lo que el modelo VE (solo I, E, P, Q comandado).
    S.append("""
<section>
  <h2>Y recién ahora se enciende el <span class="ac">gray-box</span></h2>
  <div class="modelo">
    <div class="mfila">
      <span class="mlab">antes</span>
      <span class="mec">ẋ = <span class="bb">f<sub>WC</sub>(x, P, Q; θ)</span></span>
      <span class="mnota">sólo ecuaciones<br><em>white-box</em></span>
    </div>
    <div class="mfila ahora">
      <span class="mlab">ahora</span>
      <span class="mec">ẋ = <span class="bb">f<sub>WC</sub>(x, P, Q; θ)</span>
        <span class="mas">+</span> <span class="gg">g<sub>φ</sub>(x)</span></span>
      <span class="mnota">ecuaciones + red<br><em>gray-box</em></span>
    </div>
  </div>
  <div class="dos-pie">
    <div><span class="tag azul">backbone</span> Nuestras ecuaciones.
    <strong>10 parámetros interpretables</strong>: pesos, umbrales, constantes
    de tiempo.</div>
    <div><span class="tag nara">corrección</span> Una red chiquita (2 capas,
    32 neuronas) que tapa <strong>lo que las ecuaciones no explican</strong>.</div>
  </div>

  <div class="verq">
    <div class="vq">
      <div class="vqt">Lo que el simulador TIENE</div>
      <div class="vqi">Wilson–Cowan</div>
      <div class="vqi mas2">+ refractariedad</div>
      <div class="vqi mas2">+ actuador con retardo y saturación</div>
    </div>
    <div class="vflecha">→</div>
    <div class="vq">
      <div class="vqt">Lo que el modelo VE</div>
      <div class="vqi">la actividad &nbsp;I, E</div>
      <div class="vqi">el estímulo comandado &nbsp;P, Q</div>
      <div class="vqi nada">nada más</div>
    </div>
  </div>
  <div class="dato">La red tiene que descubrir el hueco <strong>sola</strong>,
  viendo únicamente la actividad (I, E) y el estímulo comandado (P, Q).</div>
</section>""")

    # =====================================================================
    #  PARTE 2 — LA PREGUNTA CENTRAL: ¿copia al simulador?
    # =====================================================================
    #  Arco de la parte: (a) reglas de la prueba, (b) los dos experimentos que
    #  no hay que confundir, (c) el control positivo, (d-f) los tres graficos
    #  del caso con hueco, (g) la respuesta.
    S.append("""
<section class="seccion nara">
  <div class="setiq">Parte 2</div>
  <h1 class="setit">¿El gray-box logra copiar<br>el comportamiento del
  simulador?</h1>
</section>""")

    # 2.a — Las reglas del juego, antes de mostrar cualquier numero: rollout
    # libre de 200 ms (4000 pasos) sobre estimulos no vistos. Se explicita la
    # escala de lectura del error (< 5 % / 5-15 % / > 25 %) para que despues los
    # porcentajes signifiquen algo y no haya que discutirlos sobre la marcha.
    S.append("""
<section>
  <h2>La prueba</h2>
  <div class="prueba">
    <div class="pp"><span class="ppn">1</span>
      <div>Se le da <strong>sólo el estado inicial y el estímulo</strong>.</div></div>
    <div class="pp"><span class="ppn">2</span>
      <div>Tiene que generar los <strong>200 ms enteros por su cuenta</strong>,
      sin volver a mirar el dato real ni una vez.</div></div>
    <div class="pp"><span class="ppn">3</span>
      <div>Sobre estímulos que <strong>no vio al entrenar</strong>.</div></div>
  </div>
  <p class="chica">Es exigente a propósito: cualquier error se acumula durante
  4000 pasos. Si el modelo va a servir para <em>predecir</em> —y no sólo para
  ajustar lo que ya pasó— tiene que aguantar esto.</p>
  <div class="dato">El error va <strong>normalizado al rango de la señal</strong>,
  para poder interpretarlo: <strong>&lt; 5 %</strong> copia muy bien ·
  <strong>5–15 %</strong> la forma está pero hay desvíos ·
  <strong>&gt; 25 %</strong> no sirve.</div>
</section>""")

    # 2.b — La tabla A/B. Es la diapositiva mas importante para el que escucha:
    # A (epsilon = 0) es el control de que la maquinaria anda, B (epsilon = 1) es
    # el experimento de verdad. Si se confunden, todos los graficos siguientes se
    # leen mal. Los anchos de columna van inline porque son de esta tabla y de
    # ninguna otra; el resto del estilo esta en .tres.
    S.append("""
<section>
  <h2>Dos experimentos · no confundirlos</h2>
  <table class="tres">
    <tr>
      <th style="width:22%"></th>
      <th style="width:24%">el simulador tiene…</th>
      <th style="width:30%">el modelo es…</th>
      <th>qué se mide</th>
    </tr>
    <tr>
      <td><span class="ex ok2">A</span><strong>Control positivo</strong>
        <div class="exs">ε = 0 · sin hueco</div></td>
      <td>Wilson–Cowan <strong>puro</strong></td>
      <td>Wilson–Cowan puro<br><span class="sm2">la estructura es la correcta;
        sólo hay que aprender los 10 números</span></td>
      <td>¿copia la trayectoria?<br><span class="num ok3">2 %</span></td>
    </tr>
    <tr>
      <td><span class="ex nara2">B</span><strong>La pregunta central</strong>
        <div class="exs">ε = 1 · con hueco</div></td>
      <td>WC <strong>+ la física extra</strong></td>
      <td>WC solo <em>(white)</em> &nbsp;o&nbsp; WC + red <em>(gray)</em>
        <br><span class="sm2">le falta esa física</span></td>
      <td>¿copia la trayectoria?<br><span class="num mal3">15 %</span>
        <span class="num mal3">14 %</span></td>
    </tr>
  </table>
  <div class="dato"><strong>A</strong> es el control de que la maquinaria
  funciona: sin hueco, el modelo tiene la estructura correcta y copia bien.
  <strong>B</strong> es el experimento de verdad — el modelo ya no puede
  representar lo que el simulador tiene.</div>
</section>""")

    # 2.c — Control positivo (A). De aca en adelante cada titulo arranca con el
    # badge del experimento (verde "A" = ok2, naranja "B" = nara2) para que en
    # cualquier momento se sepa que caso se esta mirando.
    S.append(f"""
<section>
  <h2><span class="exi ok2">A</span> Control positivo: cuando no hay hueco</h2>
  {fig('c_sin_hueco')}
  <div class="dos-pie">
    <div><span class="tag ok">2 %</span> Después de 200 ms corriendo sola, la
    predicción se superpone con el simulador.</div>
    <div>Confirma que <strong>la maquinaria funciona</strong>: cuando la
    estructura es la correcta, la Neural ODE copia con mucha fidelidad.
    <strong>Todavía no hay red acá</strong> — no hace falta.</div>
  </div>
</section>""")

    # 2.d — B en el tiempo. Las tres diapositivas que siguen son el MISMO
    # resultado mirado de tres formas, y ese es el orden a proposito:
    # trayectoria (se desfasa) -> error vs tiempo (por que se desfasa) ->
    # retrato de fase (que si conserva). Titulo con class="ac2" para que hasta el
    # filete quede naranja: es el punto mas alto de la charla.
    S.append(f"""
<section>
  <h2 class="ac2"><span class="exi nara2">B</span> La comparación que importa</h2>
  {fig('c_comparacion')}
  <div class="dos-pie">
    <div><span class="tag azul">azul</span> El simulador — la verdad.</div>
    <div><span class="tag nara">punteado</span> El modelo corriendo solo.
    <strong>Los dos arrancan bien y se van desfasando</strong>: predicen una
    oscilación más lenta que la real.</div>
  </div>
</section>""")

    # 2.e — El mecanismo del desfase: el error no es ruido, crece con el tiempo,
    # porque un error chico de FRECUENCIA se integra a lo largo de 4000 pasos.
    S.append(f"""
<section>
  <h2><span class="exi nara2">B</span> El error se acumula</h2>
  {fig('c_error_acumula')}
  <div class="dos-pie">
    <div><span class="tag ok">verde</span> Con el modelo completo el error se
    mantiene plano: no se despega nunca.</div>
    <div><span class="tag mal">gris y naranja</span> Con hueco crece con el
    tiempo: un error chico de frecuencia se vuelve un desfase grande.</div>
  </div>
</section>""")

    # 2.f — El contrapeso: el retrato de fase saca el tiempo del grafico, y sin
    # el tiempo el ciclo se reconoce igual. Es lo que separa "no sirve" de "copia
    # el regimen": se pierde CUANDO recorre el ciclo, no QUE recorre.
    S.append(f"""
<section>
  <h2><span class="exi nara2">B</span> Pero la <span class="ac">forma</span> del ciclo sí la reproduce</h2>
  {fig('c_retrato')}
  <div class="dos-pie">
    <div>El retrato de fase saca el tiempo de la ecuación y deja ver sólo
    <strong>la forma del ciclo</strong>.</div>
    <div>Con hueco queda <strong>corrido y algo más chico</strong>, pero es
    reconociblemente el mismo objeto. Se pierde <strong>cuándo</strong> lo
    recorre, no <strong>qué</strong> recorre.</div>
  </div>
</section>""")

    # 2.g — La respuesta, en una linea. Aca la figura va con clase "med" (55 % de
    # la altura) porque el recuadro .clave de abajo es el que tiene que leerse:
    # la figura pasa a ser respaldo del texto, no al reves.
    S.append(f"""
<section>
  <h2><span class="exi nara2">B</span> La respuesta</h2>
  {fig('c_resumen', 'med')}
  <div class="clave">
    <strong>Sí copia el régimen, no copia el detalle.</strong> Y la red
    <strong>casi no ayuda</strong>: recorta apenas 1 punto de 15.<br>
    <span class="cl2">Sirve para entender el sistema; todavía no para predecirlo
    con precisión.</span>
  </div>
</section>""")

    # =====================================================================
    #  PARTE 3 — LOS PARAMETROS, RESUMIDO
    # =====================================================================
    #  Va deliberadamente comprimida en una sola diapositiva (separador en gris,
    #  no en naranja: es material de apoyo, no el resultado central). El
    #  desarrollo completo esta en docs/incertidumbre_dinamica_graybox.md.
    S.append("""
<section class="seccion gris">
  <div class="setiq">Parte 3</div>
  <h1 class="setit">Y qué pasó con<br>los parámetros
  <span class="sechi"><br>en una mirada — lo desarrollamos después</span></h1>
</section>""")

    # 3.a — Los cuatro puntos del cruce de experimentos, con la figura al lado
    # (layout .res2: figura a la izquierda, lista numerada al 33 % a la derecha).
    # El punto 3 es el incomodo y por eso va marcado en rojo (.rp.warn): la red
    # AYUDA cuando falta fisica pero ARRUINA los parametros cuando el modelo ya
    # es correcto, y el error de ajuste no lo delata.
    S.append(f"""
<section>
  <h2>Identificación de parámetros · resumen</h2>
  <div class="res2">
    <div class="rescol">{fig('p_cruce')}</div>
    <div class="rescol txt">
      <div class="rp"><span class="rpn">1</span><div>Sin la red, el hueco
      <strong>destruye</strong> los parámetros: de 1 % a 59 % de error.</div></div>
      <div class="rp ok2"><span class="rpn">2</span><div>Con la red
      <strong>mejoran</strong>: se recupera casi la mitad del daño.</div></div>
      <div class="rp warn"><span class="rpn">3</span><div>Pero <strong>sin hueco
      la red los arruina ×17</strong>. Es dañina justo cuando el modelo ya es
      correcto — y el ajuste no te avisa.</div></div>
      <div class="rp"><span class="rpn">4</span><div>Causa: <strong>dos tercios
      del hueco se disfrazan de parámetros</strong>, así que el reparto entre θ
      y la red no es único.</div></div>
    </div>
  </div>
</section>""")

    # ---- Cierre ----------------------------------------------------------
    # Las cuatro frases que tienen que quedar si el que escucha se olvida todo lo
    # demas. El pie cita el script que reproduce los numeros: es la garantia de
    # que nada de lo anterior es una estimacion de sobremesa.
    S.append("""
<section>
  <h2>Para llevarse</h2>
  <div class="cierre">
    <div class="ci"><span class="cn">1</span>
      <div><strong>Copia el régimen, no el detalle.</strong> Sigue el ritmo y la
      forma del ciclo; pierde la fase. Y la red apenas mejora eso.</div></div>
    <div class="ci"><span class="cn">2</span>
      <div><strong>La red ayuda a identificar, no a predecir.</strong> Recupera
      48 % del error de parámetros y ~7 % del de trayectoria.</div></div>
    <div class="ci"><span class="cn">3</span>
      <div><strong>Dos tercios del hueco se disfrazan de parámetros.</strong>
      Es la causa de todo lo demás.</div></div>
    <div class="ci"><span class="cn">4</span>
      <div><strong>Y es dañina cuando el modelo ya es correcto:</strong> sin
      hueco, la red arruina la identificación ×17 — y el ajuste no te avisa.</div></div>
  </div>
  <div class="pie3">Todos los números salen de corridas reales ·
  <code>bash scripts/run_uncertainty_all.sh</code> · 38 tests en verde</div>
</section>""")

    # ---- Ficha tecnica (backup para preguntas) ---------------------------
    # Va DESPUES del cierre a proposito: no se presenta, se salta con End y se
    # vuelve a ella si preguntan por los datos, el corte train/test o el
    # entrenamiento. Cuatro bloques en grilla 2x2 (.ficha), cada uno respondiendo
    # una pregunta previsible: cuantos datos, como se partieron, con que
    # estimulos, con que optimizador.
    S.append("""
<section>
  <h2>Ficha técnica · con qué se entrenó</h2>
  <div class="ficha">

    <div class="fbloque">
      <div class="fcab">Los datos</div>
      <div class="fitem"><span class="fk">20</span> trayectorias de
        <span class="fk">200 ms</span></div>
      <div class="fitem"><span class="fk">4000</span> puntos cada una ·
        dt = 0,05 ms</div>
      <div class="fitem"><span class="fk">80 000</span> muestras (t, I, E, P, Q)</div>
      <div class="fnota">Sin ruido de observación: se aisló el efecto del hueco
      estructural.</div>
    </div>

    <div class="fbloque">
      <div class="fcab">El corte train / test</div>
      <div class="fitem"><span class="fk ok4">13</span> entrenamiento ·
        <span class="fk mal4">7</span> test</div>
      <div class="fnota">El corte es por <strong>escenario completo</strong>, no
      por puntos sueltos: de cada familia de estímulo se reserva uno entero. Así
      el test mide si generaliza a un <em>estímulo nuevo</em>, no a instantes
      nuevos de un estímulo ya visto.</div>
    </div>

    <div class="fbloque">
      <div class="fcab">Los estímulos <span class="fsub">(7 familias)</span></div>
      <div class="ftabla">
        <div><span class="fest">escalón</span> amplitudes 0,4 · 0,8 · 1,2</div>
        <div><span class="fest">cuadrada</span> 50 · 100 · 130 Hz</div>
        <div><span class="fest">APRBS</span> amplitud y duración al azar</div>
        <div><span class="fest">PRBS</span> binaria pseudo-aleatoria</div>
        <div><span class="fest">theta-gamma</span> ráfagas bajo envolvente lenta</div>
        <div><span class="fest">Poisson</span> pulsos en tiempos aleatorios</div>
        <div><span class="fest">chirp</span> barrido 10 → 150 Hz</div>
      </div>
      <div class="fnota">Todos <strong>on/off y ≥ 0</strong>: realizables con
      optogenética. P y Q descorrelacionados entre sí.</div>
    </div>

    <div class="fbloque">
      <div class="fcab">El entrenamiento</div>
      <div class="fitem">multiple shooting, ventanas de <span class="fk">100</span>
        pasos</div>
      <div class="fitem"><span class="fk">1500</span> épocas Adam +
        <span class="fk">60</span> pasos L-BFGS</div>
      <div class="fitem">arranque <strong>ignorante</strong>: los 10 parámetros
        parten de 1,0</div>
      <div class="fnota">La red nunca ve los valores verdaderos — sólo se usan
      al final para reportar el error.</div>
    </div>

  </div>
</section>""")

    return S


# =============================================================================
#  ESTILOS
# =============================================================================
#  El CSS entero viaja como un string y se inyecta en un <style> (nada de .css
#  aparte: rompe lo autocontenido). Por eso el mapa de los bloques va ACA afuera
#  y no como comentarios /* */ adentro: los comentarios dentro del string se
#  copiarian a cada .html generado sin aportar nada al que lo abre.
#
#  MAPA POR ROL (en el orden en que aparecen abajo):
#
#   1. Reset + paleta (`*`, `:root`). Los colores se usan SIEMPRE por su nombre
#      semantico, nunca en hexa suelto: --azul = lo dado/la verdad/el backbone,
#      --nara = lo nuevo/la correccion/el foco, --verde = bien, --rojo = mal,
#      --amar = el filete de los recuadros de remate.
#
#   2. El deck y la diapositiva (`body`, `#deck`, `section`, `.on`).
#      Decision central: la diapositiva tiene TAMANO CASI FIJO
#      —min(1360px,95vw) x min(766px,93vh), o sea 16:9— sobre un fondo oscuro.
#      Asi lo que se ve en la notebook es lo mismo que en el proyector: el texto
#      no se re-acomoda segun la pantalla y nada se corta. La contra: si a una
#      diapositiva le metes mas contenido del que entra, se DESBORDA en silencio
#      (body tiene overflow:hidden) — hay que revisarla a ojo.
#      Solo `section.on` se muestra; el resto queda en display:none.
#
#   3. Tipografia base (`h1`, `h2`) y acentos (`.ac`, `.ac2`).
#
#   4. Figuras (`.fig`, `.fig img`, `.fig.med`). El truco: la seccion es un flex
#      en columna, la figura se come el espacio libre con flex:1 y el PNG entra
#      con object-fit:contain -> nunca se estira ni tapa el texto de abajo.
#
#   5. Portada y separadores de parte (`.portada`, `.seccion`, `.setiq`,
#      `.setit`, `.sechi`). El color del rotulo "Parte N" lo elige el modificador
#      de la seccion (.nara, .gris, .verde).
#
#   6. Bloques de contenido, uno por diapositiva o familia de diapositivas:
#        .dos / .caja / .igual ....... el "antes: simulador = modelo"
#        .pert / .ecg ................ las dos perturbaciones (.dim ya estaba,
#                                      .hi en naranja es lo que se agrego)
#        .modelo / .mfila ............ ecuacion antes vs ahora (f_WC + g_phi)
#        .verq ....................... lo que el simulador TIENE vs lo que ve
#        .prueba / .pp ............... las reglas del rollout libre
#        .tres + .ex/.exi/.num ....... la tabla de los experimentos A y B
#        .res2 / .rp ................. resumen de parametros (.warn en rojo)
#        .ficha / .ftabla ............ la ficha tecnica, grilla 2x2
#        .cierre / .ci ............... las cuatro frases finales
#
#   7. Piezas transversales: `.dos-pie` (las dos columnas de lectura al pie) con
#      sus `.tag` de color, y `.dato` / `.clave` / `.pie3`, que usan
#      margin-top:auto para quedar PEGADOS AL PIE de la diapositiva sin importar
#      cuanto mida el contenido de arriba.
#
#   8. `.pasos`, `.paso`, `.pnum2`, `.pcuerpo`, `.ptit2`, `.pcuando`, `.pdesc`,
#      `.pmeta`: sobraron de una version anterior de la charla. Hoy ninguna
#      diapositiva las usa (igual que `.seccion.verde` y `.ex.azul2`); se dejan
#      por si vuelve ese bloque.
#
#   9. Cromo de navegacion (`#nav`, `#barra`): fijos en pantalla, fuera de la
#      diapositiva, y se esconden al imprimir.
#
#  10. `@media print`: fuerza display:flex en TODAS las secciones y un salto de
#      pagina despues de cada una -> "imprimir a PDF" sale como un PDF de
#      diapositivas. Sin esta regla se imprimiria una sola (las demas estan en
#      display:none).
# =============================================================================

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --tinta:#12141a; --gris:#5b6070; --suave:#e2e4ea; --fondo:#f7f8fb;
  --azul:#2a78d6; --nara:#eb6834; --verde:#0d8a4f; --rojo:#d63b3a;
  --amar:#f0a500;
}
html,body{height:100%}
body{font-family:-apple-system,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;
  background:#0e1016;color:var(--tinta);overflow:hidden}
#deck{width:100vw;height:100vh;display:flex;align-items:center;justify-content:center}
section{display:none;width:min(1360px,95vw);height:min(766px,93vh);background:#fff;
  border-radius:14px;padding:34px 50px 26px;flex-direction:column;
  box-shadow:0 18px 60px rgba(0,0,0,.45)}
section.on{display:flex}
h1{font-size:50px;font-weight:700;letter-spacing:-.6px;text-align:center;line-height:1.15}
h2{font-size:30px;font-weight:650;letter-spacing:-.3px;margin-bottom:12px;
  padding-bottom:8px;border-bottom:2px solid var(--suave)}
.ac{color:var(--nara)} .ac2{color:var(--nara);border-bottom-color:var(--nara)}
.fig{flex:1;min-height:0;display:flex;align-items:center;justify-content:center;margin:2px 0}
.fig img{width:100%;height:100%;object-fit:contain}
.fig.med{flex:0 0 55%}

.portada{justify-content:center;align-items:center;gap:20px}
.portada .sub{font-size:22px;color:var(--gris);text-align:center}
.portada .frase{margin-top:16px;font-size:19px;line-height:1.6;text-align:center;
  background:var(--fondo);padding:22px 32px;border-radius:10px;
  border-left:5px solid var(--azul);max-width:880px}
.seccion{justify-content:center;align-items:center;gap:18px;
  background:linear-gradient(160deg,#f7f8fb,#eef1f7)}
.seccion .setiq{font-size:14px;letter-spacing:3px;color:var(--azul);font-weight:700}
.seccion.nara .setiq{color:var(--nara)}
.seccion.gris .setiq{color:var(--gris)}
.seccion.verde .setiq{color:var(--verde)}
.setit{font-size:43px}
.sechi{font-size:21px;color:var(--gris);font-weight:500}

.dos{display:flex;align-items:center;justify-content:center;gap:24px;margin:18px 0 6px}
.caja{background:var(--fondo);border:2px solid var(--azul);border-radius:12px;
  padding:18px 28px;text-align:center;min-width:320px}
.rot{font-size:12.5px;letter-spacing:1.6px;color:var(--azul);font-weight:700}
.ec{font-family:"DejaVu Serif",Georgia,serif;font-size:25px;margin:10px 0 8px}
.pie2{font-size:13.5px;color:var(--gris);line-height:1.4}
.igual{font-size:44px;color:var(--nara);font-weight:700}
.grande{font-size:25px;text-align:center;margin:6px 0 4px}
.chica{font-size:16.5px;color:var(--gris);line-height:1.55;text-align:center;
  max-width:960px;margin:6px auto}
.chica.top{margin-bottom:12px}
.dato{margin-top:auto;background:var(--fondo);border-left:5px solid var(--amar);
  padding:13px 18px;border-radius:8px;font-size:16.5px;line-height:1.5}

.pert{display:flex;gap:18px;align-items:flex-start;margin:12px 0 4px}
.pnum{flex:none;width:38px;height:38px;border-radius:50%;background:var(--nara);
  color:#fff;font-size:20px;font-weight:700;display:flex;align-items:center;
  justify-content:center}
.pcont{flex:1}
.ptit{font-size:20px;font-weight:650;margin-bottom:7px}
.ptag{font-size:11.5px;letter-spacing:.8px;background:var(--fondo);color:var(--gris);
  padding:3px 9px;border-radius:12px;margin-left:8px;font-weight:600;
  text-transform:uppercase;vertical-align:3px}
.ecg{font-family:"DejaVu Serif",Georgia,serif;font-size:21px;background:var(--fondo);
  padding:12px 18px;border-radius:8px;margin-bottom:7px}
.ecg .dim{color:var(--gris)} .ecg .hi{color:var(--nara);font-weight:700}
.pexp{font-size:15.5px;color:var(--gris);line-height:1.5}

.modelo{margin:24px auto 14px;max-width:960px}
.mfila{display:flex;align-items:center;gap:20px;padding:18px 22px;border-radius:10px;
  background:var(--fondo);margin-bottom:12px}
.mfila.ahora{background:#fff6ef;border-left:5px solid var(--nara)}
.mlab{flex:none;width:66px;font-size:13.5px;letter-spacing:1.4px;color:var(--gris);
  font-weight:700;text-transform:uppercase}
.mec{font-family:"DejaVu Serif",Georgia,serif;font-size:26px;flex:1}
.mec .bb{color:var(--azul)} .mec .gg{color:var(--nara);font-weight:700}
.mec .mas{color:var(--nara);font-weight:700;margin:0 4px}
.mnota{font-size:13.5px;color:var(--gris);text-align:right;line-height:1.4;width:135px}

.verq{display:flex;align-items:center;justify-content:center;gap:22px;
  margin:16px auto 4px;max-width:1000px}
.vq{flex:1;background:var(--fondo);border-radius:10px;padding:14px 18px}
.vqt{font-size:12px;letter-spacing:1.3px;color:var(--gris);font-weight:700;
  text-transform:uppercase;margin-bottom:8px}
.vqi{font-size:16px;line-height:1.5}
.vqi.mas2{color:var(--nara);font-weight:600}
.vqi.nada{color:var(--gris);font-style:italic}
.vflecha{font-size:30px;color:var(--gris)}

.prueba{margin:18px auto 8px;max-width:960px;display:flex;flex-direction:column;gap:13px}
.pp{display:flex;gap:16px;align-items:flex-start;font-size:19px;line-height:1.5}
.ppn{flex:none;width:31px;height:31px;border-radius:50%;background:var(--nara);
  color:#fff;font-size:15px;font-weight:700;display:flex;align-items:center;
  justify-content:center}

.tres{width:100%;border-collapse:collapse;margin-top:14px;font-size:15.5px}
.tres th{text-align:left;padding:9px 12px;border-bottom:2px solid var(--tinta);
  font-size:12.5px;letter-spacing:.7px;color:var(--gris);text-transform:uppercase}
.tres td{padding:14px 12px;border-bottom:1px solid var(--suave);
  vertical-align:top;line-height:1.45}
.tres .exs{font-size:13px;color:var(--gris);margin-top:3px}
.tres .sm2{font-size:13.5px;color:var(--gris)}
.ex{display:inline-flex;width:24px;height:24px;border-radius:7px;color:#fff;
  font-size:13px;font-weight:700;align-items:center;justify-content:center;
  margin-right:8px;vertical-align:-3px}
.exi{display:inline-flex;width:30px;height:30px;border-radius:8px;color:#fff;
  font-size:16px;font-weight:700;align-items:center;justify-content:center;
  margin-right:11px;vertical-align:-4px}
.ex.ok2,.exi.ok2{background:var(--verde)}
.ex.nara2,.exi.nara2{background:var(--nara)}
.ex.azul2,.exi.azul2{background:var(--azul)}
.num{display:inline-block;font-size:20px;font-weight:700;margin-right:12px}
.num.ok3{color:var(--verde)} .num.mal3{color:var(--rojo)}

.dos-pie{display:flex;gap:24px;margin-top:11px}
.dos-pie>div{flex:1;font-size:16px;line-height:1.5}
.tag{display:inline-block;font-size:11.5px;font-weight:700;letter-spacing:.6px;
  padding:2px 9px;border-radius:20px;color:#fff;margin-right:7px;
  text-transform:uppercase;vertical-align:1px}
.tag.azul{background:var(--azul)} .tag.nara{background:var(--nara)}
.tag.ok{background:var(--verde)} .tag.mal{background:var(--rojo)}
.clave{margin-top:auto;background:#fff8e8;border-left:5px solid var(--amar);
  padding:16px 20px;border-radius:8px;font-size:19px;line-height:1.55}
.clave .cl2{color:var(--gris);font-size:16.5px}

.res2{display:flex;gap:26px;flex:1;min-height:0;align-items:stretch}
.rescol{flex:1;min-width:0;display:flex;flex-direction:column}
.rescol.txt{flex:0 0 33%;justify-content:center;gap:13px}
.rp{display:flex;gap:13px;align-items:flex-start;font-size:16.5px;line-height:1.45}
.rpn{flex:none;width:27px;height:27px;border-radius:50%;background:var(--azul);
  color:#fff;font-size:14px;font-weight:700;display:flex;align-items:center;
  justify-content:center}
.rp.warn .rpn{background:var(--rojo)} .rp.ok2 .rpn{background:var(--verde)}

.pasos{display:flex;flex-direction:column;gap:14px;margin-top:12px}
.paso{display:flex;gap:16px;align-items:flex-start;background:var(--fondo);
  padding:15px 19px;border-radius:10px}
.pnum2{flex:none;width:34px;height:34px;border-radius:9px;background:var(--verde);
  color:#fff;font-size:17px;font-weight:700;display:flex;align-items:center;
  justify-content:center}
.pcuerpo{flex:1}
.ptit2{font-size:18.5px;font-weight:650;margin-bottom:5px}
.pcuando{font-size:12px;letter-spacing:.5px;background:#fff;color:var(--gris);
  padding:3px 10px;border-radius:12px;margin-left:9px;font-weight:600;
  text-transform:uppercase;vertical-align:2px}
.pdesc{font-size:15.5px;color:var(--tinta);line-height:1.5}
.pmeta{color:var(--gris);font-size:14.5px}

.ficha{display:grid;grid-template-columns:1fr 1fr;gap:14px 22px;margin-top:12px;
  flex:1;min-height:0;align-content:start}
.fbloque{background:var(--fondo);border-radius:10px;padding:14px 18px}
.fcab{font-size:12.5px;letter-spacing:1.3px;color:var(--azul);font-weight:700;
  text-transform:uppercase;margin-bottom:9px}
.fsub{color:var(--gris);font-weight:600;letter-spacing:.4px}
.fitem{font-size:16px;line-height:1.7}
.fk{font-weight:700;color:var(--tinta)}
.fk.ok4{color:var(--verde)} .fk.mal4{color:var(--rojo)}
.fnota{font-size:13.5px;color:var(--gris);line-height:1.45;margin-top:8px;
  border-top:1px solid var(--suave);padding-top:7px}
.ftabla{display:grid;grid-template-columns:1fr 1fr;gap:2px 14px;font-size:13.5px;
  line-height:1.55}
.fest{display:inline-block;min-width:88px;font-weight:650;color:var(--nara)}

.cierre{display:flex;flex-direction:column;gap:14px;margin:16px 0}
.ci{display:flex;gap:16px;align-items:flex-start;font-size:18.5px;line-height:1.5}
.cn{flex:none;width:31px;height:31px;border-radius:50%;background:var(--azul);
  color:#fff;font-size:15px;font-weight:700;display:flex;align-items:center;
  justify-content:center}
.pie3{margin-top:auto;font-size:13.5px;color:var(--gris);
  border-top:1px solid var(--suave);padding-top:10px}
code{font-family:"DejaVu Sans Mono",monospace;font-size:12.5px;background:var(--fondo);
  padding:2px 6px;border-radius:4px}

#nav{position:fixed;bottom:14px;right:20px;display:flex;gap:12px;align-items:center;
  color:#c8cddb;font-size:13px;z-index:20;background:rgba(14,16,22,.86);
  padding:7px 13px;border-radius:22px}
#nav button{background:rgba(255,255,255,.12);border:none;color:#e8ebf3;width:32px;
  height:32px;border-radius:8px;font-size:17px;cursor:pointer}
#nav button:hover{background:rgba(255,255,255,.26)}
#barra{position:fixed;top:0;left:0;height:3px;background:var(--nara);
  transition:width .22s;z-index:20}
@media print{
  body{overflow:visible;background:#fff}
  section{display:flex!important;page-break-after:always;box-shadow:none;
    width:100%;height:auto;min-height:0}
  #nav,#barra{display:none}
}
"""

# =============================================================================
#  NAVEGACION  (el unico javascript de la presentacion)
# =============================================================================
#  Todo el "motor" es una variable `i` (indice de la diapositiva actual) y la
#  funcion ir(n), que hace las cuatro cosas juntas: prende la clase .on en la
#  seccion n, actualiza el contador "3 / 17", estira la barra de progreso de
#  arriba y escribe el numero en la URL.
#
#  Decisiones:
#   - El numero en la URL va con history.replaceState, no cambiando
#     location.hash: replaceState NO dispara hashchange, asi que no se
#     realimenta con el listener de mas abajo, y ademas no llena el historial de
#     entradas (el boton "atras" del navegador no se convierte en "diapositiva
#     anterior", que ademas seria confuso).
#   - ir(n) clampea con max/min: pasarse al final o al principio no rompe nada,
#     simplemente se queda ahi. Por eso las flechas no necesitan chequeos.
#   - desdeURL() devuelve 0 si el hash no es un numero. La ultima linea llama a
#     ir(desdeURL()), asi que abrir "presentacion.html#7" arranca en la 7: se
#     puede mandar el enlace a una diapositiva concreta.
#   - Teclas: derecha / PageDown / espacio / Enter avanzan; izquierda / PageUp /
#     Backspace vuelven; Home y End a los extremos; F pantalla completa. Se
#     llama a preventDefault para que el espacio no scrollee y Backspace no
#     navegue hacia atras. Cubre a proposito lo que mandan los controles remotos
#     de presentacion, que emulan PageUp/PageDown.
#   - Clic en el deck: el tercio izquierdo (< 32 % del ancho) va para atras y el
#     resto avanza; la franja de atras es mas chica porque el gesto normal es
#     avanzar. El chequeo de closest('#nav') es para no contar dos veces el clic
#     en los botones de la barra.
# =============================================================================

JS = """
const S=[...document.querySelectorAll('section')];
let i=0;
function ir(n){
  i=Math.max(0,Math.min(S.length-1,n));
  S.forEach((s,k)=>s.classList.toggle('on',k===i));
  document.getElementById('num').textContent=(i+1)+' / '+S.length;
  document.getElementById('barra').style.width=((i+1)/S.length*100)+'%';
  history.replaceState(null,'','#'+(i+1));
}
function desdeURL(){
  const n=parseInt(location.hash.replace('#',''),10);
  return isNaN(n)?0:n-1;
}
addEventListener('hashchange',()=>ir(desdeURL()));
document.addEventListener('keydown',e=>{
  if(['ArrowRight','PageDown',' ','Enter'].includes(e.key)){e.preventDefault();ir(i+1)}
  if(['ArrowLeft','PageUp','Backspace'].includes(e.key)){e.preventDefault();ir(i-1)}
  if(e.key==='Home')ir(0);
  if(e.key==='End')ir(S.length-1);
  if(e.key==='f'||e.key==='F'){
    document.fullscreenElement?document.exitFullscreen()
                              :document.documentElement.requestFullscreen()}
});
document.getElementById('deck').addEventListener('click',e=>{
  if(e.target.closest('#nav'))return;
  ir(e.clientX < innerWidth*0.32 ? i-1 : i+1);
});
ir(desdeURL());
"""


# =============================================================================
#  ARMADO FINAL
# =============================================================================
def main():
    """Pega diapositivas + CSS + JS en un unico .html y lo escribe en OUT.

    El orden del <body> importa: primero la barra de progreso, despues el deck
    con TODAS las secciones (ocultas), despues la barra de navegacion y al final
    el <script>. El JS va ultimo porque en su ultima linea ya llama a ir(): si
    estuviera en el <head>, querySelectorAll('section') no encontraria nada.

    El print de salida reporta el peso en KB para tener a mano el costo de
    embeber las figuras (es lo que va a pesar el adjunto del mail).
    """
    S = diapos()
    html = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gray-box en Wilson–Cowan</title>
<style>{CSS}</style></head>
<body>
<div id="barra"></div>
<div id="deck">{''.join(S)}</div>
<div id="nav">
  <button onclick="ir(i-1)" title="anterior">‹</button>
  <span id="num"></span>
  <button onclick="ir(i+1)" title="siguiente">›</button>
  <span style="opacity:.6">·&nbsp; F pantalla completa</span>
</div>
<script>{JS}</script>
</body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"{OUT}  ·  {len(S)} diapositivas  ·  "
          f"{len(html.encode())/1024:.0f} KB (autocontenido)")


if __name__ == "__main__":
    main()
