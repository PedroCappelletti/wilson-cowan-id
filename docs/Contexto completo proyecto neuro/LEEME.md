# LEEME — Paquete "Contexto completo proyecto neuro"

Paquete de estudio autocontenido para entender **de punta a punta** el proyecto de identificación de Wilson-Cowan con Neural ODE / PINN. Pensado para leerse de corrido (p. ej. en un vuelo).

## Por dónde empezar
Abrí **`Guia de contenidos`** (md o pdf). Es el índice + la ruta de estudio ordenada. Todo lo demás cuelga de ahí.

## Contenido
- **`1 - Teoria matematica/`** — 6 documentos (M1–M6). Fuente LaTeX (`.tex`) + PDF compilado.
- **`2 - Teoria biologica/`** — 5 documentos (B1–B5). Fuente LaTeX (`.tex`) + PDF compilado.
- **`3 - Archivos del proyecto/`** — 7 documentos (P0–P6) en Markdown con diagramas Mermaid (manual del código) + PDF.
- **`4 - Resultados y conclusiones/`** — 3 documentos (R1 experimentos, R2 datos reales, R3 conclusiones/novedad). Fuente LaTeX (`.tex`) + PDF.
- **`figuras/`** — imágenes usadas por los documentos.

## Cómo se generaron los PDF
Los `.tex` se compilan con **pdfLaTeX** (2 pasadas para bibliografía; usan `thebibliography`, sin BibTeX). Están compilados a PDF en esta entrega. Para recompilar:
```
pdflatex archivo.tex && pdflatex archivo.tex
```
o subilos a Overleaf (compilador pdfLaTeX). Usan `babel` en español, ya incluido.

## Sobre los `.md` del bloque 3 (código)
Están en Markdown con diagramas **Mermaid**, que **renderizan directo en Obsidian** (por eso quedaron en md y no en PDF: los diagramas se ven mejor así). Si querés un PDF, exportá desde Obsidian (Export to PDF) con un plugin que soporte Mermaid.

## Política de imágenes (enfoque híbrido)
1. **Resultados reales** → figuras del propio repo (`wilson-cowan-id/results/figures/`), copiadas a `figuras/`.
2. **Diagramas conceptuales** (sigmoidea, retrato de fase, elipse SVD, RK4, sesgo espectral) → generados con matplotlib para este paquete.
3. **Ilustraciones biológicas canónicas** (neurona, potencial de acción, bandas EEG, etc.) → de fuentes abiertas (Wikimedia Commons) cuando fue posible, con atribución en el pie.
4. **Donde nada de lo anterior servía** → un recuadro *placeholder* con un **prompt listo para pegar en Gemini/IA** y generar la imagen. Buscá `[FIGURA – PROMPT IA]` en los documentos.
