#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INPUT_DOCX="${1:-/mnt/data/8399c424-979c-4449-ac05-2069c7fad0bf.docx}"
OUTDIR="${2:-out_compag_strict}"
TEX_MAIN="paper/cea_draft/main.tex"
PANDOC_LOCAL="tools/pandoc/pandoc-3.9/pandoc.exe"
PY=".venv/Scripts/python.exe"

mkdir -p "$OUTDIR"

if [[ ! -f "$INPUT_DOCX" ]]; then
  echo "FAIL: input docx not found: $INPUT_DOCX"
  exit 1
fi

if [[ ! -f "$PY" ]]; then
  echo "ERROR: Python not found: $PY"
  exit 1
fi

if command -v pandoc >/dev/null 2>&1; then
  PANDOC_BIN="pandoc"
else
  PANDOC_BIN="$PANDOC_LOCAL"
fi

if [[ ! -x "$PANDOC_BIN" && ! -f "$PANDOC_BIN" ]]; then
  echo "ERROR: pandoc not found."
  exit 1
fi

echo "[1/4] Fix DOCX for strict COMPAG/CEA requirements"
if command -v sha256sum >/dev/null 2>&1; then
  echo "input_sha256=$(sha256sum \"$INPUT_DOCX\" | awk '{print $1}')"
fi
"$PY" scripts/fix_compag_docx.py \
  --input "$INPUT_DOCX" \
  --tex "$TEX_MAIN" \
  --outdir "$OUTDIR"

echo "[2/4] Export EN/ZH PDFs (docx -> html -> chrome headless)"
"$PANDOC_BIN" "$OUTDIR/COMPAG_CEA_manuscript_EN.docx" -t html5 -s --extract-media="$OUTDIR/en_media" -o "$OUTDIR/COMPAG_CEA_manuscript_EN.html"
"$PANDOC_BIN" "$OUTDIR/COMPAG_CEA_manuscript_ZH_working.docx" -t html5 -s --extract-media="$OUTDIR/zh_media" -o "$OUTDIR/COMPAG_CEA_manuscript_ZH_working.html"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/export_pdf_with_chrome.ps1 \
  -HtmlPath "$OUTDIR/COMPAG_CEA_manuscript_EN.html" \
  -PdfPath "$OUTDIR/COMPAG_CEA_manuscript_EN.pdf"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/export_pdf_with_chrome.ps1 \
  -HtmlPath "$OUTDIR/COMPAG_CEA_manuscript_ZH_working.html" \
  -PdfPath "$OUTDIR/COMPAG_CEA_manuscript_ZH_working.pdf"

echo "[3/4] Validate outputs"
"$PY" scripts/validate_compag_strict.py --outdir "$OUTDIR"

echo "[4/4] Done"
echo "Output directory: $OUTDIR"
