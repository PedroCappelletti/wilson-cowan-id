#!/usr/bin/env python3
# =============================================================================
#  MARKDOWN -> PDF  (via HTML + Chrome headless)
# =============================================================================
#  USO:  python scripts/md_a_pdf.py docs/archivo.md [docs/otro.md ...]
#  Deja el .pdf al lado de cada .md.
# =============================================================================
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

CSS = """
body { font-family: "DejaVu Sans", Arial, sans-serif; font-size: 10.5pt;
       line-height: 1.45; max-width: 19cm; margin: 0 auto; color: #222; }
h1 { font-size: 20pt; border-bottom: 2px solid #1f4e79; padding-bottom: 4px; }
h2 { font-size: 15pt; color: #1f4e79; margin-top: 1.6em; }
h3 { font-size: 12pt; margin-top: 1.2em; }
table { border-collapse: collapse; margin: 0.8em 0; font-size: 9pt; width: 100%; }
th, td { border: 1px solid #bbb; padding: 3px 6px; text-align: left; }
th { background: #eef2f7; }
tr:nth-child(even) td { background: #fafafa; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 9pt;
       background: #f3f3f3; padding: 1px 3px; border-radius: 3px; }
pre { background: #f3f3f3; padding: 8px; overflow-x: auto; font-size: 8.5pt; }
pre code { background: none; padding: 0; }
blockquote { border-left: 4px solid #1f4e79; margin: 0.8em 0; padding: 0.2em 1em;
             color: #444; background: #f7f9fc; }
hr { border: 0; border-top: 1px solid #ccc; margin: 1.5em 0; }
@page { size: A4; margin: 18mm 16mm; }
"""


def chrome() -> str:
    for c in ("google-chrome", "chromium", "chromium-browser"):
        p = shutil.which(c)
        if p:
            return p
    raise SystemExit("no se encontro Chrome/Chromium")


def convertir(md_path: Path) -> Path:
    html_body = markdown.markdown(
        md_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc"])
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{md_path.stem}</title><style>{CSS}</style></head>"
            f"<body>{html_body}</body></html>")
    out = md_path.with_suffix(".pdf")
    with tempfile.TemporaryDirectory() as td:
        h = Path(td) / "doc.html"
        h.write_text(html, encoding="utf-8")
        subprocess.run([chrome(), "--headless=new", "--disable-gpu",
                        "--no-sandbox", "--no-pdf-header-footer",
                        f"--print-to-pdf={out.resolve()}",
                        h.resolve().as_uri()],
                       check=True, capture_output=True)
    return out


if __name__ == "__main__":
    for a in sys.argv[1:]:
        print("->", convertir(Path(a)))
