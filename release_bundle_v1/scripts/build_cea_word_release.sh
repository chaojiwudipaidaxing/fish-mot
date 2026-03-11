#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TEX_MAIN="paper/cea_draft/main.tex"
BIB_FILE="paper/cea_draft/references.bib"
FIG_DIRS="paper/cea_draft:paper/cea_draft/figs:paper/cea_draft/figures:figs:figures:."
OUTDIR="${1:-out_cea_word}"
REFDOC="ref/elsevier_reference.docx"

PY=".venv/Scripts/python.exe"
PANDOC_LOCAL="tools/pandoc/pandoc-3.9/pandoc.exe"

mkdir -p "$OUTDIR" ref logs tmp paper/cea_draft/build

if [[ ! -f "$PY" ]]; then
  echo "ERROR: Python not found at $PY"
  exit 1
fi

if command -v pandoc >/dev/null 2>&1; then
  PANDOC_BIN="pandoc"
else
  PANDOC_BIN="$PANDOC_LOCAL"
fi

if [[ ! -x "$PANDOC_BIN" && ! -f "$PANDOC_BIN" ]]; then
  echo "ERROR: pandoc not found (checked system PATH and $PANDOC_LOCAL)"
  exit 1
fi

echo "[1/9] Sanitize LaTeX source"
"$PY" scripts/tex_sanitize.py \
  --root "paper/cea_draft" \
  --report "logs/tex_sanitize_report.txt" \
  --apply
cp "$TEX_MAIN" "tmp/main.sanitized.tex"

echo "[2/9] latexmk compile gate"
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory=tmp "tmp/main.sanitized.tex" \
    | tee "logs/latexmk.log"
else
  echo "latexmk is not installed in this environment." | tee "logs/latexmk.log"
fi

echo "[3/9] Expand LaTeX includes"
mkdir -p tmp/tables
cp paper/cea_draft/tables/*.tex tmp/tables/
if command -v latexpand >/dev/null 2>&1; then
  latexpand "tmp/main.sanitized.tex" > "tmp/main.expanded.tex"
else
  "$PY" scripts/latexpand_simple.py \
    --input "tmp/main.sanitized.tex" \
    --output "tmp/main.expanded.tex" \
    --warnings "logs/latexpand_warnings.txt"
fi

echo "[3.5/9] Normalize siunitx S-columns for Pandoc conversion"
"$PY" - <<'PY'
from pathlib import Path
import re

inp = Path("tmp/main.expanded.tex")
out = Path("tmp/main.expanded.pandoc.tex")
text = inp.read_text(encoding="utf-8", errors="replace")

def repl(match):
    spec = match.group(1).replace("S", "r")
    spec = " ".join(spec.split())
    return f"\\begin{{tabular}}{{{spec}}}"

text = re.sub(r"\\begin\{tabular\}\{([^}]*)\}", repl, text)
out.write_text(text, encoding="utf-8", newline="\n")
print(f"wrote {out}")
PY

echo "[4/9] Prepare pandoc reference DOCX"
if [[ ! -f "$REFDOC" ]]; then
  "$PANDOC_BIN" -o "$REFDOC" --print-default-data-file reference.docx
fi

echo "[5/9] Convert LaTeX -> DOCX"
CROSSREF_ARGS=()
if command -v pandoc-crossref >/dev/null 2>&1; then
  CROSSREF_ARGS+=(--filter pandoc-crossref)
fi

"$PANDOC_BIN" "tmp/main.expanded.pandoc.tex" \
  -f latex -t docx \
  --reference-doc="$REFDOC" \
  --resource-path="$FIG_DIRS" \
  --citeproc --bibliography="$BIB_FILE" \
  "${CROSSREF_ARGS[@]}" \
  -M link-citations=true \
  -o "$OUTDIR/CEA_manuscript_EN.raw.docx" \
  2> "logs/pandoc.stderr.log"

echo "[6/9] Post-process Word formatting"
"$PY" scripts/format_elsevier_word.py \
  --input "$OUTDIR/CEA_manuscript_EN.raw.docx" \
  --output "$OUTDIR/CEA_manuscript_EN.docx" \
  --highlights "$OUTDIR/highlights.docx" \
  --ga-placeholder "$OUTDIR/CEA_graphical_abstract_placeholder.docx" \
  --todo "$OUTDIR/todo_from_formatter.md" \
  --changes "$OUTDIR/changes_from_formatter.md" \
  --pandoc-log "logs/pandoc.stderr.log"

echo "[7/9] Generate Chinese working copy"
"$PY" scripts/create_zh_working_doc.py \
  --input "$OUTDIR/CEA_manuscript_EN.docx" \
  --output "$OUTDIR/CEA_manuscript_ZH_working.docx"

echo "[8/9] Export PDFs"
if command -v soffice >/dev/null 2>&1; then
  soffice --headless --convert-to pdf --outdir "$OUTDIR" "$OUTDIR/CEA_manuscript_EN.docx"
  soffice --headless --convert-to pdf --outdir "$OUTDIR" "$OUTDIR/CEA_manuscript_ZH_working.docx"
else
  "$PANDOC_BIN" "$OUTDIR/CEA_manuscript_EN.docx" -t html5 -s --extract-media="$OUTDIR/en_media" -o "$OUTDIR/CEA_manuscript_EN.html"
  "$PANDOC_BIN" "$OUTDIR/CEA_manuscript_ZH_working.docx" -t html5 -s --extract-media="$OUTDIR/zh_media" -o "$OUTDIR/CEA_manuscript_ZH_working.html"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts/export_pdf_with_chrome.ps1" \
    -HtmlPath "$OUTDIR/CEA_manuscript_EN.html" \
    -PdfPath "$OUTDIR/CEA_manuscript_EN.pdf"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts/export_pdf_with_chrome.ps1" \
    -HtmlPath "$OUTDIR/CEA_manuscript_ZH_working.html" \
    -PdfPath "$OUTDIR/CEA_manuscript_ZH_working.pdf"
fi

echo "[9/9] Generate TODO / change log / validation"
"$PY" scripts/extract_todos.py \
  --tex "$TEX_MAIN" \
  --en-docx "$OUTDIR/CEA_manuscript_EN.docx" \
  --output "$OUTDIR/todo.md"

cat > "$OUTDIR/changes.md" <<'EOF'
# Changes

- Generated EN submission manuscript from LaTeX source with pandoc + post-formatting.
- Enforced Elsevier-like Word styles and CEA front-matter compliance notes.
- Converted pseudo-table text into editable Word tables and applied no-vertical-line table styling.
- Preserved editable equations (OMML) and kept figure captions paired with inserted figures.
- Generated Chinese working manuscript with inline Chinese editing annotations after English paragraphs.
- Exported EN/ZH PDFs and produced validation/todo reports.
EOF

"$PY" scripts/validate_cea_outputs.py \
  --en-docx "$OUTDIR/CEA_manuscript_EN.docx" \
  --zh-docx "$OUTDIR/CEA_manuscript_ZH_working.docx" \
  --en-pdf "$OUTDIR/CEA_manuscript_EN.pdf" \
  --zh-pdf "$OUTDIR/CEA_manuscript_ZH_working.pdf" \
  --highlights "$OUTDIR/highlights.docx" \
  --report "$OUTDIR/validation_report.txt"

echo "Done. Outputs in: $OUTDIR"
