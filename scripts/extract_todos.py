from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document


def extract_tex_todos(tex_path: Path) -> list[str]:
    lines = tex_path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[str] = []
    for i, line in enumerate(lines, 1):
        if "TODO" in line:
            snippet = " ".join(line.strip().split())
            out.append(f"- `{tex_path}:{i}` {snippet}")
    return out


def extract_doc_todos(doc_path: Path) -> list[str]:
    doc = Document(doc_path)
    out: list[str] = []
    for i, para in enumerate(doc.paragraphs, 1):
        t = " ".join(para.text.strip().split())
        if "TODO" in t:
            out.append(f"- `{doc_path}:paragraph-{i}` {t}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract TODO items with source locations.")
    parser.add_argument("--tex", default="paper/cea_draft/main.tex")
    parser.add_argument("--en-docx", default="CEA_manuscript_EN.docx")
    parser.add_argument("--zh-docx", default="", help="Optional Chinese working DOCX for TODO extraction.")
    parser.add_argument("--output", default="todo.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tex_path = Path(args.tex)
    en_doc = Path(args.en_docx)
    zh_doc = Path(args.zh_docx) if args.zh_docx else None
    todo_lines: list[str] = []

    if tex_path.exists():
        todo_lines.extend(extract_tex_todos(tex_path))
    if en_doc.exists():
        todo_lines.extend(extract_doc_todos(en_doc))
    if zh_doc and zh_doc.exists():
        todo_lines.extend(extract_doc_todos(zh_doc))

    header = [
        "# TODO List (with source locations)",
        "",
        "This file aggregates unresolved TODO items from LaTeX and generated Word outputs.",
        "",
    ]
    if not todo_lines:
        todo_lines = ["- No TODO entries detected."]
    Path(args.output).write_text("\n".join(header + todo_lines) + "\n", encoding="utf-8")
    print(f"saved={args.output} items={len(todo_lines)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
