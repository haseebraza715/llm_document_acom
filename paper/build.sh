#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if command -v tectonic >/dev/null 2>&1; then
  tectonic --keep-logs main.tex
  exit 0
fi

if command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode main.tex
  if command -v bibtex >/dev/null 2>&1; then
    bibtex main || true
  fi
  pdflatex -interaction=nonstopmode main.tex
  pdflatex -interaction=nonstopmode main.tex
  exit 0
fi

echo "No TeX compiler found. Install tectonic or pdflatex/bibtex to build the paper." >&2
exit 1
