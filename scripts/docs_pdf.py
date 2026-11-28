# Arma los PDF técnicos de docs/ desde sus .md con pandoc y XeLaTeX (Arial, A4).
# Los wikilinks de Obsidian pasan a «título». Sin argumentos arma los tres;
# con nombres (sin .md) arma solo esos. Rutas de pandoc y MiKTeX de esta máquina.
# md_a_pdf.py es otro camino (HTML) y no da el mismo resultado.
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
TMP = Path(tempfile.mkdtemp(prefix="docs_pdf_"))
PANDOC = r"C:\Users\User\AppData\Local\Pandoc\pandoc.exe"
MIKTEX = r"C:\Users\User\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
env = dict(os.environ, PATH=MIKTEX + os.pathsep + os.environ["PATH"])

TODOS = ("arquitectura_modelos", "controlador_paper_simulador_y_modelos", "controlador_planta_modelo")
for nombre in sys.argv[1:] or TODOS:
    s = (DOCS / f"{nombre}.md").read_text(encoding="utf-8")
    cuerpo = s
    # los wikilinks del texto (no los del frontmatter, que pandoc ignora)
    if s.startswith("---"):
        fin = s.index("\n---", 3) + 4
        cabeza, cuerpo = s[:fin], s[fin:]
    else:
        cabeza = ""
    cuerpo = re.sub(r"\[\[([^\]|]+)(\|[^\]]+)?\]\]", lambda m: "«" + m.group(1) + "»", cuerpo)
    fuente = TMP / f"{nombre}.md"
    fuente.write_text(cabeza + cuerpo, encoding="utf-8")
    salida = TMP / f"{nombre}.pdf"
    r = subprocess.run([PANDOC, str(fuente), "-f", "markdown", "-o", str(salida),
                        "--pdf-engine=xelatex", "-V", "mainfont=Arial", "-V", "monofont=Consolas",
                        "-V", "geometry:a4paper,margin=2cm", "-V", "lang=es", "-V", "fontsize=10pt",
                        f"--resource-path={DOCS}"],
                       cwd=DOCS, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(nombre, "returncode", r.returncode)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise SystemExit(1)
    otros = [l for l in r.stderr.splitlines() if l.strip()]
    if otros:
        print("  avisos:", *otros[:8], sep="\n   ")
    shutil.copy(salida, DOCS / f"{nombre}.pdf")
