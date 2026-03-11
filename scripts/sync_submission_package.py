#!/usr/bin/env python
"""Sync paper draft to submission package and rebuild Overleaf source zip."""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import zipfile
from pathlib import Path


INPUT_RE = re.compile(r"\\input\{([^}]+)\}")
FIG_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
FIG_SUFFIXES = (".pdf", ".png", ".jpg", ".jpeg")

FORBIDDEN_RE = re.compile(
    r"Missing:|TODO|outperform|SOTA|state-of-the-art|superior|overall improvement|\.{2}/\.{2}/results/archive",
    re.IGNORECASE,
)


def canonicalize_rel(path_value: str) -> str | None:
    p = path_value.strip()
    p = p.replace("\\", "/")
    if p.startswith("tables/"):
        return p
    if p.startswith("figs/"):
        return p
    if p.startswith("figures/"):
        return f"figs/{Path(p).name}"
    if "/paper_assets/" in p:
        tail = p.split("/paper_assets/", 1)[1]
        if tail.endswith(".tex"):
            return f"tables/{tail}"
        if tail.endswith(".png"):
            return f"figs/{tail}"
    if p.startswith("paper_assets/"):
        tail = p.split("paper_assets/", 1)[1]
        if tail.endswith(".tex"):
            return f"tables/{tail}"
        if tail.endswith(".png"):
            return f"figs/{tail}"
    if p.endswith(".tex"):
        return f"tables/{Path(p).name}"
    if p.endswith(".png"):
        return f"figs/{Path(p).name}"
    if any(p.endswith(ext) for ext in FIG_SUFFIXES if ext != ".png"):
        return f"figs/{Path(p).name}"
    return None


def resolve_asset_path(base_dir: Path, rel: str) -> Path | None:
    candidates = [base_dir / rel]
    if rel.startswith("figs/"):
        fig_rel = rel.replace("figs/", "figures/", 1)
        candidates.append(base_dir / fig_rel)
        rel_path = Path(rel)
        fig_path = Path(fig_rel)
        if rel_path.suffix == "":
            for ext in FIG_SUFFIXES:
                candidates.append(base_dir / rel_path.with_suffix(ext))
                candidates.append(base_dir / fig_path.with_suffix(ext))
    elif rel.startswith("tables/"):
        rel_path = Path(rel)
        if rel_path.suffix == "":
            candidates.append(base_dir / rel_path.with_suffix(".tex"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_refs_from_text(text: str) -> tuple[list[str], list[str]]:
    tables: set[str] = set()
    figs: set[str] = set()
    for raw in INPUT_RE.findall(text):
        rel = canonicalize_rel(raw)
        if rel and rel.startswith("tables/"):
            tables.add(rel)
    for raw in FIG_RE.findall(text):
        rel = canonicalize_rel(raw)
        if rel and rel.startswith("figs/"):
            figs.add(rel)
    return sorted(tables), sorted(figs)


def parse_refs(tex_path: Path) -> tuple[list[str], list[str]]:
    return parse_refs_from_text(tex_path.read_text(encoding="utf-8"))


def to_submission_main(text: str) -> str:
    def _replace_input(match: re.Match[str]) -> str:
        rel = canonicalize_rel(match.group(1))
        return f"\\input{{{rel}}}" if rel and rel.startswith("tables/") else match.group(0)

    def _replace_fig(match: re.Match[str]) -> str:
        rel = canonicalize_rel(match.group(1))
        if rel and rel.startswith("figs/"):
            return match.group(0).replace(match.group(1), rel)
        return match.group(0)

    text = re.sub(r"\\input\{([^}]+)\}", _replace_input, text)
    text = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", _replace_fig, text)
    return text


def normalize_tex(text: str) -> str:
    text = text.replace("figs/", "FIG_DIR/")
    text = text.replace("figures/", "FIG_DIR/")
    text = text.replace("tables/", "TAB_DIR/")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def validate_highlights(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing highlights file"
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not (3 <= len(lines) <= 5):
        return False, f"highlight lines={len(lines)} (expected 3-5)"
    max_len = max(len(ln) for ln in lines)
    if max_len > 85:
        return False, f"max highlight len={max_len} (>85)"
    return True, f"{len(lines)} lines, max {max_len} chars"


def rebuild_zip(submission_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    include_paths = [
        submission_dir / "main.tex",
        submission_dir / "references.bib",
        submission_dir / "highlights.txt",
        submission_dir / "cover_letter.txt",
        submission_dir / "graphical_abstract.png",
        submission_dir / "README.txt",
    ]
    if (submission_dir / "sanity_check.md").exists():
        include_paths.append(submission_dir / "sanity_check.md")
    if (submission_dir / "reproducibility").exists():
        include_paths.append(submission_dir / "reproducibility")
    include_paths.append(submission_dir / "figs")
    include_paths.append(submission_dir / "tables")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in include_paths:
            if not p.exists():
                continue
            if p.is_dir():
                for file_path in p.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(submission_dir).as_posix()
                        zf.write(file_path, arcname)
            else:
                arcname = p.relative_to(submission_dir).as_posix()
                zf.write(p, arcname)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-dir", default="paper/cea_draft")
    parser.add_argument("--submission-dir", default="paper/cea_draft/submission_package")
    parser.add_argument("--zip-name", default="source.zip")
    args = parser.parse_args()

    draft_dir = Path(args.draft_dir)
    submission_dir = Path(args.submission_dir)
    draft_main = draft_dir / "main.tex"
    draft_bib = draft_dir / "references.bib"
    sub_main = submission_dir / "main.tex"
    sub_bib = submission_dir / "references.bib"
    sub_figs = submission_dir / "figs"
    sub_tables = submission_dir / "tables"

    if not draft_main.exists():
        print(f"[FAIL] missing draft main.tex: {draft_main}")
        return 1
    if not draft_bib.exists():
        print(f"[FAIL] missing draft references.bib: {draft_bib}")
        return 1

    draft_text = draft_main.read_text(encoding="utf-8")
    if FORBIDDEN_RE.search(draft_text):
        print("[FAIL] authority main.tex contains forbidden placeholder/path/claim token")
        return 1

    table_refs, fig_refs = parse_refs_from_text(draft_text)
    print("[STEP0] References parsed from authority main.tex")
    for rel in table_refs + fig_refs:
        src = resolve_asset_path(draft_dir, rel)
        tag = "PASS" if src is not None and src.exists() else "FAIL"
        print(f"  [{tag}] {rel}")
        if src is None or not src.exists():
            return 1

    submission_dir.mkdir(parents=True, exist_ok=True)
    sub_text = to_submission_main(draft_text)
    sub_main.write_text(sub_text, encoding="utf-8")
    shutil.copy2(draft_bib, sub_bib)
    copy_if_exists(draft_dir / "highlights.txt", submission_dir / "highlights.txt")
    copy_if_exists(draft_dir / "cover_letter.txt", submission_dir / "cover_letter.txt")
    copy_if_exists(draft_dir / "graphical_abstract.png", submission_dir / "graphical_abstract.png")
    copy_if_exists(draft_dir / "README.txt", submission_dir / "README.txt")
    copy_if_exists(draft_dir / "sanity_check.md", submission_dir / "sanity_check.md")

    if (draft_dir / "reproducibility").exists():
        dst = submission_dir / "reproducibility"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(draft_dir / "reproducibility", dst)

    # On Windows, figure PDFs may be held open by a viewer. Rebuild referenced
    # assets in place instead of deleting the whole directory tree first.
    sub_figs.mkdir(parents=True, exist_ok=True)
    sub_tables.mkdir(parents=True, exist_ok=True)

    for rel in fig_refs:
        src = resolve_asset_path(draft_dir, rel)
        if src is None:
            print(f"[FAIL] missing figure source for {rel}")
            return 1
        dst_rel = Path(rel)
        if dst_rel.suffix == "":
            dst_rel = dst_rel.with_suffix(src.suffix)
        dst = submission_dir / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for rel in table_refs:
        src = resolve_asset_path(draft_dir, rel)
        if src is None:
            print(f"[FAIL] missing table source for {rel}")
            return 1
        dst = submission_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # Verify submission references after rewrite.
    sub_tables, sub_figs_refs = parse_refs_from_text(sub_text)
    if sub_tables != table_refs or sub_figs_refs != fig_refs:
        print("[FAIL] submission rewrite changed reference set unexpectedly")
        return 1

    zip_path = submission_dir / args.zip_name
    rebuild_zip(submission_dir, zip_path)

    draft_hash = sha256_text(normalize_tex(draft_text))
    sub_hash = sha256_text(normalize_tex(sub_text))
    if draft_hash != sub_hash:
        print("[FAIL] authority/submission main.tex hash mismatch after normalization")
        return 1
    print(f"[PASS] normalized body hash match: {draft_hash}")

    ok, msg = validate_highlights(submission_dir / "highlights.txt")
    print(f"[{'PASS' if ok else 'FAIL'}] highlights check: {msg}")
    if not ok:
        return 1

    print(f"[PASS] source zip rebuilt: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
