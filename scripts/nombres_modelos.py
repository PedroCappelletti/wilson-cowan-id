"""Nombres de los modelos para figuras y documentos.

Los tags y las claves de los json conservan los nombres internos (lag, e1S2,
«forma exacta», «estado de filtro», «filtro 8 canales»). Lo que se muestra usa
la nomenclatura del proyecto: el white-box con la ecuación del mecanismo
omitido es el «backbone expandido», y el FIR es el «filtro entrenable».
"""

REEMPLAZOS = [
    ("estado de filtro", "backbone expandido"),
    ("forma exacta", "backbone expandido"),
    ("filtro de 8 canales", "filtro entrenable"),
    ("filtro 8 canales", "filtro entrenable"),
    ("historia 5 ms con comando actual", "historia con comando actual"),
    ("historia 5 ms sin comando actual", "historia sin comando actual"),
]


def mostrar(nombre: str) -> str:
    for viejo, nuevo in REEMPLAZOS:
        nombre = nombre.replace(viejo, nuevo)
    return nombre
