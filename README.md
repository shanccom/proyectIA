# Proyecto de Inteligencia Artificial - Artículo Elsevier

Template LaTeX para artículo académico formato Elsevier (Procedia Computer Science).

## Requisitos

- LaTeX distribution
- PDFLaTeX (NO compila con `latex` estándar)

## Instalación

### macOS
```bash
# Opción 1: MacTeX (recomendado, ~2GB pero trae todo)
brew install --cask mactex

# Opción 2: BasicTeX (mínimo, ~100MB)
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install elsarticle ecrc preprint flushend subfigure texliveonfly
```

### Linux (Debian/Ubuntu)
```bash
sudo apt update
sudo apt install texlive-latex-extra texlive-publishers texlive-bibtex-extra
```

### Linux (Fedora)
```bash
sudo dnf install texlive-scheme-full
```

### Windows
1. Descargar [MiKTeX](https://miktex.org/) o [TeX Live](https://tug.org/texlive/)
2. Durante la instalación de MiKTeX, seleccionar "Install missing packages on the fly" = Yes

## Compilar en macOS

Desde la terminal, dentro de la carpeta del proyecto:

```bash
# 1. Compilar documento
pdflatex article.tex

# 2. Compilar bibliografía
bibtex article

# 3-4. Compilar dos veces más para resolver referencias cruzadas
pdflatex article.tex
pdflatex article.tex
```

### Comando único (todo en una línea)
```bash
pdflatex article.tex && bibtex article && pdflatex article.tex && pdflatex article.tex
```

### Con latexmk (recomendado)
```bash
latexmk -pdf article.tex
```

### Compilar en Overleaf (web, cualquier SO)
1. Ir a https://www.overleaf.com/ → New Project → Upload Project
2. Subir todos los archivos (manteniendo la estructura: `article.tex`, `sections/`, `images/`, `references.bib`, `*.cls`, `*.bst`)
3. Cambiar compilador a **pdfLaTeX** (Menu → Compiler → pdfLaTeX)
4. Click en "Recompile"

## Estructura del proyecto

```
proyectFinal/
├── article.tex              # Principal (preamble + inputs)
├── sections/                # Carpeta con las secciones
│   ├── 01-introduccion.tex      # Introduction
│   ├── 02-related_work.tex      # Related Work
│   ├── 03-material_methods.tex  # Contenedor de Material and Methods
│   ├── 03.1-teoria.tex          # Theoretical background
│   ├── 03.2-herramientas.tex    # Tools and technologies
│   ├── 03.3-dataset.tex         # Dataset
│   ├── 03.4-metodologia.tex     # Proposed methodology
│   ├── 04-resultados.tex        # Results
│   ├── 05-discusion.tex         # Discussion
│   ├── 06-conclusiones.tex      # Conclusions
│   ├── 07-limitaciones.tex      # Limitations and future work
│   └── 08-secciones_finales.tex # CRediT, Declaration, Funding, etc.
├── images/
│   ├── img1.jpg
│   └── img2.png
├── elsarticle.cls
├── elsarticle-num.bst
├── elsarticle-num-names.bst
├── elsarticle-harv.bst
└── references.bib
```

**Cada miembro del equipo edita solo su(s) archivo(s) dentro de `sections/` sin tocar los demás.**

## Notas importantes
- El encoding actual es `latin1` (`\usepackage[latin1]{inputenc}`). Si usan UTF-8 con tildes/ñ, cambiar a `\usepackage[utf8]{inputenc}`.
- Compilar SIEMPRE con `pdflatex`, NUNCA con `latex`.
- Si falta algún paquete, intentar con `sudo tlmgr install nombre-del-paquete` en macOS/Linux.
