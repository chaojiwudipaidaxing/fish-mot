# Control-character cleanup utility

## Why hidden control characters appear
- Copy/paste from chat apps, PDF viewers, spreadsheets, or rich-text editors can insert non-printable bytes.
- Mixed encoding chains (ANSI/GBK/UTF-8) during file transfer can leave hidden control bytes.
- Font substitution and clipboard normalization may inject control separators (especially around punctuation).

## Usage
```bash
# Dry-run (default): scan and report only
python clean_control_chars.py paper/cea_draft --report report.txt

# Apply cleanup in-place (creates .bak backups)
python clean_control_chars.py paper/cea_draft --report report.txt --apply
```

## Recommended LaTeX header (UTF-8 + font encoding)
```latex
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{textcomp}
\usepackage{lmodern}
```

For Overleaf and local builds, keep source files in UTF-8 and avoid editing `.tex` with legacy ANSI encodings.
