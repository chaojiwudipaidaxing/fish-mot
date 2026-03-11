from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


TEXTTT_PATTERN = re.compile(r"\\texttt\{([^{}]*)\}")
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
BEGIN_TABLE_PATTERN = re.compile(r"\\begin\{(tabular\*?|longtable)\}")
END_TABLE_PATTERN = re.compile(r"\\end\{(tabular\*?|longtable)\}")
ARTIFACT_PM_PATTERN = re.compile(r"\s*(卤|¡À)\s*")


@dataclass
class FileStats:
    path: Path
    control_removed: int = 0
    texttt_escaped: int = 0
    tabular_fixed: int = 0
    changed: bool = False


def split_unescaped_percent(line: str) -> tuple[str, str]:
    escaped = False
    for i, ch in enumerate(line):
        if ch == "\\" and not escaped:
            escaped = True
            continue
        if ch == "%" and not escaped:
            return line[:i], line[i:]
        escaped = False
    return line, ""


def sanitize_control_chars(text: str) -> tuple[str, int]:
    hits = CONTROL_PATTERN.findall(text)
    if not hits:
        return text, 0
    return CONTROL_PATTERN.sub("", text), len(hits)


def sanitize_texttt_underscores(text: str) -> tuple[str, int]:
    count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal count
        content = match.group(1)
        fixed = re.sub(r"(?<!\\)_", r"\\_", content)
        count += len(re.findall(r"(?<!\\)_", content))
        return f"\\texttt{{{fixed}}}"

    new_text = TEXTTT_PATTERN.sub(repl, text)
    return new_text, count


def sanitize_common_artifacts(text: str) -> tuple[str, int]:
    count = 0

    def repl_pm(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return r" $\pm$ "

    text = ARTIFACT_PM_PATTERN.sub(repl_pm, text)
    return text, count


def is_row_control_line(content: str) -> bool:
    s = content.strip()
    if not s:
        return True
    commands = (
        "\\toprule",
        "\\midrule",
        "\\bottomrule",
        "\\cmidrule",
        "\\hline",
        "\\cline",
        "\\end{",
        "\\begin{",
    )
    return s.startswith(commands)


def fix_tabular_rows(text: str) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    in_table = False
    fixes = 0
    out: list[str] = []

    for line in lines:
        body, comment = split_unescaped_percent(line.rstrip("\r\n"))
        eol = "\n" if line.endswith("\n") else ""

        if BEGIN_TABLE_PATTERN.search(body):
            in_table = True
        if in_table and "&" in body and not is_row_control_line(body):
            stripped = body.rstrip()
            if re.search(r"(?<!\\)\\\s*$", stripped):
                body = re.sub(r"(?<!\\)\\\s*$", r"\\\\", stripped)
                fixes += 1
            elif not re.search(r"\\\\\s*$", stripped) and not stripped.endswith("\\tabularnewline"):
                body = stripped + r" \\"
                fixes += 1

        if END_TABLE_PATTERN.search(body):
            in_table = False

        out.append(body + comment + eol)

    return "".join(out), fixes


def sanitize_file(path: Path) -> tuple[str, FileStats]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    stats = FileStats(path=path)

    txt, n = sanitize_control_chars(raw)
    stats.control_removed += n
    txt, _ = sanitize_common_artifacts(txt)
    txt, n = sanitize_texttt_underscores(txt)
    stats.texttt_escaped += n
    txt, n = fix_tabular_rows(txt)
    stats.tabular_fixed += n

    stats.changed = txt != raw
    return txt, stats


def iter_tex_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.tex") if p.is_file())


def build_report(stats: list[FileStats], root: Path) -> str:
    lines = []
    lines.append(f"tex_sanitize report root={root}")
    lines.append("")
    total_changed = 0
    c_ctrl = c_tt = c_tab = 0
    for st in stats:
        rel = st.path.relative_to(root)
        if st.changed:
            total_changed += 1
        c_ctrl += st.control_removed
        c_tt += st.texttt_escaped
        c_tab += st.tabular_fixed
        lines.append(
            f"{rel}: changed={st.changed} control_removed={st.control_removed} "
            f"texttt_escaped={st.texttt_escaped} tabular_fixed={st.tabular_fixed}"
        )
    lines.append("")
    lines.append(
        f"summary: files={len(stats)} changed={total_changed} "
        f"control_removed={c_ctrl} texttt_escaped={c_tt} tabular_fixed={c_tab}"
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanitize LaTeX sources for Word conversion.")
    parser.add_argument("--root", default="paper/cea_draft", help="Root directory to scan for .tex files.")
    parser.add_argument(
        "--report",
        default="paper/cea_draft/build/tex_sanitize_report.txt",
        help="Path to report output.",
    )
    parser.add_argument("--apply", action="store_true", help="Write changes back to files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    tex_files = iter_tex_files(root)
    stats: list[FileStats] = []
    new_text_map: dict[Path, str] = {}

    for path in tex_files:
        new_text, st = sanitize_file(path)
        stats.append(st)
        new_text_map[path] = new_text

    if args.apply:
        for st in stats:
            if st.changed:
                cleaned = new_text_map[st.path].replace("\r\n", "\n")
                st.path.write_text(cleaned, encoding="utf-8", newline="\n")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(stats, root), encoding="utf-8")

    changed = sum(1 for s in stats if s.changed)
    print(f"tex_sanitize: files={len(stats)} changed={changed} apply={args.apply}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
