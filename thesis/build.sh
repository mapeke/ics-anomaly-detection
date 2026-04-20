#!/usr/bin/env bash
# Build the thesis PDF via XeLaTeX. Run from inside thesis/.
# Requires TinyTeX with: fontspec polyglossia setspace extsizes biber biblatex.
set -euo pipefail

# TinyTeX on PATH (Windows + Git Bash)
if [ -d "$APPDATA/TinyTeX/bin/windows" ]; then
  export PATH="$APPDATA/TinyTeX/bin/windows:$PATH"
fi

mkdir -p build/chapters build/frontmatter

xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
biber --output-directory=build main
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex

echo "---"
grep -E "Output written" build/main.log | tail -1
