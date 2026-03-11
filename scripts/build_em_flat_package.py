#!/usr/bin/env python
"""Build Elsevier EM flat zip (no subdirectories inside archive)."""

from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path


INPUT_RE = re.compile(r"\\input\{(tables/[^}]+)\}")
FIG_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{(figs/[^}]+)\}")


def parse_refs(tex_path: Path) -> tuple[list[str], list[str]]:
    text = tex_path.read_text(encoding="utf-8")
    return sorted(set(INPUT_RE.findall(text))), sorted(set(FIG_RE.findall(text)))


def flatten_main(tex_text: str) -> str:
    tex_text = tex_text.replace("figs/", "")
    tex_text = tex_text.replace("tables/", "")
    return tex_text


def check_zip_has_no_subdirs(zip_path: Path) -> tuple[bool, list[str]]:
    bad = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for n in zf.namelist():
            if "/" in n or "\\" in n:
                bad.append(n)
    return len(bad) == 0, bad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", default="paper/cea_draft/submission_package")
    parser.add_argument("--flat-dir-name", default="em_flat")
    parser.add_argument("--zip-name", default="CEA_EM_flat.zip")
    args = parser.parse_args()

    submission_dir = Path(args.submission_dir)
    main_tex = submission_dir / "main.tex"
    refs_bib = submission_dir / "references.bib"
    if not main_tex.exists() or not refs_bib.exists():
        print(f"[FAIL] missing main.tex or references.bib in {submission_dir}")
        return 1

    table_refs, fig_refs = parse_refs(main_tex)
    print("[CHECK] submission main references")
    for rel in table_refs + fig_refs:
        src = submission_dir / rel
        ok = src.exists()
        print(f"  [{'PASS' if ok else 'FAIL'}] {rel}")
        if not ok:
            return 1

    flat_dir = submission_dir / args.flat_dir_name
    if flat_dir.exists():
        shutil.rmtree(flat_dir)
    flat_dir.mkdir(parents=True, exist_ok=True)

    # Core files
    shutil.copy2(main_tex, flat_dir / "main.tex")
    shutil.copy2(refs_bib, flat_dir / "references.bib")
    for optional in ["highlights.txt", "cover_letter.txt", "graphical_abstract.png", "README.txt", "sanity_check.md"]:
        src = submission_dir / optional
        if src.exists():
            shutil.copy2(src, flat_dir / optional)

    # Flatten referenced assets
    for rel in table_refs + fig_refs:
        src = submission_dir / rel
        dst = flat_dir / Path(rel).name
        shutil.copy2(src, dst)

    # Rewrite path prefixes in flat main.tex
    flat_main = flat_dir / "main.tex"
    flat_main.write_text(flatten_main(flat_main.read_text(encoding="utf-8")), encoding="utf-8")
    txt = flat_main.read_text(encoding="utf-8")
    if "figs/" in txt or "tables/" in txt:
        print("[FAIL] flat main.tex still has figs/ or tables/ path")
        return 1

    # Build zip with root-level files only
    zip_path = submission_dir / args.zip_name
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(flat_dir.glob("*")):
            if fp.is_file():
                zf.write(fp, fp.name)

    ok, bad = check_zip_has_no_subdirs(zip_path)
    if not ok:
        print("[FAIL] zip contains subdirectory entries:")
        for n in bad:
            print(f"  - {n}")
        return 1

    print(f"[PASS] EM flat zip built with no subdirectories: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
