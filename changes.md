# Changes Log

## Pipeline executed
1. Added LaTeX pre-sanitization (`scripts/tex_sanitize.py`) and ran it on `paper/cea_draft`.
   - Checked control characters, `\texttt{}` underscore escaping, and tabular row endings.
   - Report: `paper/cea_draft/build/tex_sanitize_report.txt`.
2. Attempted `latexmk` compile gate.
   - `latexmk` is not installed in this environment.
   - Log captured at `paper/cea_draft/build/latexmk.log`.
3. Expanded LaTeX includes into a single file (`scripts/latexpand_simple.py`).
   - Output: `paper/cea_draft/build/expanded.tex`.
4. Prepared Pandoc conversion input by normalizing siunitx `S` columns to standard tabular columns in a conversion-only copy.
   - Output: `paper/cea_draft/build/expanded_pandoc.tex`.
5. Converted expanded LaTeX to DOCX via Pandoc + citeproc using Elsevier-style reference doc.
   - Intermediate: `paper/cea_draft/build/pandoc_test_tables.docx`.
6. Post-processed DOCX with `scripts/format_elsevier_word.py`.
   - Enforced Times New Roman styles and sizes.
   - Enforced caption styling and submission notes.
   - Enforced no pseudo-table text leftovers; styled all tables as no-vertical-line (three-line visual).
   - Kept equations editable (OMML).
   - Reformatted references section for CEA-friendly readability.
7. Generated Chinese working manuscript (`scripts/create_zh_working_doc.py`).
   - Preserved full English text.
   - Inserted Chinese working annotations paragraph-by-paragraph for author editing.
8. Generated PDFs from DOCX->HTML->Chrome headless export.
9. Generated TODO index and validation report.

## Compliance edits reflected in EN manuscript
- Replaced unsafe "drift-proof" wording with drift-aware framing.
- Kept/checked Terminology & Scope wording as monitoring/response (no elimination guarantee).
- Kept runtime split as Scope A (tracking-only) and Scope B (end-to-end).
- Kept Scope B schema fields with TODO placeholders only (no fabricated values).
- Kept gating plateau narrative as candidate interval requiring activation diagnostics.
- Ensured Abstract <=250 words and Keywords count in [1,7].
- Ensured highlights and graphical-abstract submission notes are present.
- Ensured data availability statement is explicit with TODO placeholders where links/DOI are pending.

## Generated deliverables
- `CEA_manuscript_EN.docx`
- `CEA_manuscript_EN.pdf`
- `CEA_manuscript_ZH_working.docx`
- `CEA_manuscript_ZH_working.pdf`
- `highlights.docx`
- `todo.md`
- `validation_report.txt`

## Remaining manual items
- Fill all `TODO:` content that requires new measurements, links, or policy decisions.
- Run final LaTeX `latexmk` compile once TeX toolchain is available (currently unavailable in this environment).
- If required by journal office, regenerate PDF using Microsoft Word/LibreOffice print engine for final camera-ready checks.
