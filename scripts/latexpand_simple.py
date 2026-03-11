from __future__ import annotations

import argparse
import re
from pathlib import Path


INPUT_RE = re.compile(r"^\s*\\(input|include)\{([^}]+)\}\s*$")


def resolve_include(current: Path, target: str) -> Path:
    p = Path(target.strip())
    if p.suffix == "":
        p = p.with_suffix(".tex")
    if not p.is_absolute():
        p = (current.parent / p).resolve()
    return p


def expand(path: Path, stack: list[Path], warnings: list[str]) -> str:
    if path in stack:
        warnings.append(f"cycle detected: {' -> '.join(str(x) for x in stack + [path])}")
        return ""
    if not path.exists():
        warnings.append(f"missing include: {path}")
        return ""

    stack.append(path)
    out: list[str] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    for idx, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("%"):
            out.append(line)
            continue

        m = INPUT_RE.match(line)
        if not m:
            out.append(line)
            continue

        include_raw = m.group(2)
        include_path = resolve_include(path, include_raw)
        out.append(f"% <latexpand-simple begin: {include_raw} from {path.name}:{idx}>")
        out.append(expand(include_path, stack, warnings))
        out.append(f"% <latexpand-simple end: {include_raw}>")

    stack.pop()
    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple recursive LaTeX \\input expander.")
    parser.add_argument("--input", default="paper/cea_draft/main.tex", help="Input root tex file.")
    parser.add_argument("--output", default="paper/cea_draft/build/expanded.tex", help="Expanded output tex file.")
    parser.add_argument(
        "--warnings",
        default="paper/cea_draft/build/latexpand_warnings.txt",
        help="Warnings output file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input).resolve()
    out_path = Path(args.output)
    warn_path = Path(args.warnings)
    warnings: list[str] = []
    expanded = expand(in_path, [], warnings)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(expanded + "\n", encoding="utf-8", newline="\n")
    warn_path.parent.mkdir(parents=True, exist_ok=True)
    if warnings:
        warn_path.write_text("\n".join(warnings) + "\n", encoding="utf-8")
    else:
        warn_path.write_text("no warnings\n", encoding="utf-8")

    print(f"expanded={out_path}")
    print(f"warnings={warn_path}")
    print(f"warning_count={len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
