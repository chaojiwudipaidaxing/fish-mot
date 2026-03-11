#!/usr/bin/env python3
"""Scan and optionally clean control characters in LaTeX source trees.

Default mode is dry-run. Use --apply to write cleaned files in-place.
Before writing, each modified file is backed up as <filename>.bak.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


DEFAULT_EXTS = (".tex", ".bib", ".sty", ".cls")
ALLOWED_CONTROL = {9, 10, 13}  # tab, LF, CR


@dataclass
class Hit:
    offset: int
    line: int
    col: int
    code: int


def is_bad_control(code: int) -> bool:
    return (0 <= code <= 31 and code not in ALLOWED_CONTROL) or code == 127


def iter_target_files(root: Path, exts: Sequence[str]) -> Iterable[Path]:
    exts_l = {e.lower() for e in exts}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
            continue
        if path.suffix.lower() in exts_l:
            yield path


def scan_text(text: str) -> List[Hit]:
    hits: List[Hit] = []
    line = 1
    col = 1
    for i, ch in enumerate(text):
        code = ord(ch)
        if is_bad_control(code):
            hits.append(Hit(offset=i, line=line, col=col, code=code))
        if ch == "\n":
            line += 1
            col = 1
        else:
            col += 1
    return hits


def clean_text(text: str) -> Tuple[str, int]:
    out_chars: List[str] = []
    removed = 0
    for ch in text:
        code = ord(ch)
        if is_bad_control(code):
            removed += 1
            continue
        out_chars.append(ch)
    return "".join(out_chars), removed


def backup_path(path: Path) -> Path:
    candidate = path.with_name(path.name + ".bak")
    if not candidate.exists():
        return candidate
    idx = 1
    while True:
        cand = path.with_name(path.name + f".bak{idx}")
        if not cand.exists():
            return cand
        idx += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan and clean control characters in LaTeX files.",
    )
    parser.add_argument("root", nargs="?", default=".", help="Root directory to scan.")
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTS),
        help="Comma-separated file extensions to scan (default: .tex,.bib,.sty,.cls).",
    )
    parser.add_argument(
        "--report",
        default="report.txt",
        help="Output report file path (default: report.txt).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply cleanup in-place and create .bak backups.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    exts = [x.strip() for x in args.extensions.split(",") if x.strip()]
    report_path = Path(args.report)

    lines: List[str] = []
    lines.append(f"Root: {root.as_posix()}")
    lines.append(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    lines.append(f"Extensions: {', '.join(exts)}")
    lines.append("")

    total_files = 0
    files_with_hits = 0
    total_hits = 0
    total_removed = 0

    for path in sorted(iter_target_files(root, exts)):
        total_files += 1
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="surrogateescape")
        hits = scan_text(text)
        if not hits:
            continue

        files_with_hits += 1
        total_hits += len(hits)
        rel = path.relative_to(root)
        lines.append(f"[FILE] {rel.as_posix()}  hits={len(hits)}")
        for hit in hits[:50]:
            lines.append(
                f"  - offset={hit.offset} line={hit.line} col={hit.col} code=0x{hit.code:02X}"
            )
        if len(hits) > 50:
            lines.append(f"  - ... ({len(hits) - 50} more)")

        if args.apply:
            cleaned, removed = clean_text(text)
            if removed > 0:
                bkp = backup_path(path)
                bkp.write_bytes(raw)
                path.write_bytes(cleaned.encode("utf-8", errors="surrogateescape"))
                total_removed += removed
                lines.append(f"  * cleaned={removed}, backup={bkp.name}")
        lines.append("")

    lines.append("Summary")
    lines.append("-------")
    lines.append(f"files_scanned={total_files}")
    lines.append(f"files_with_control_chars={files_with_hits}")
    lines.append(f"control_chars_found={total_hits}")
    if args.apply:
        lines.append(f"control_chars_removed={total_removed}")
    else:
        lines.append("control_chars_removed=0 (dry-run)")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report: {report_path}")
    print(f"files_scanned={total_files} files_with_hits={files_with_hits} hits={total_hits}")
    if args.apply:
        print(f"removed={total_removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
