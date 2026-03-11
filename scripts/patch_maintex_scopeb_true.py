#!/usr/bin/env python
"""Patch main.tex to document Scope B-true runtime evidence."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch main.tex for Scope B-true reporting.")
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--scopeb-true-table", type=Path, required=True)
    parser.add_argument("--scopeb-true-fig", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, default=Path("results/brackishmot/runtime/runtime_profile_e2e_true_summary.csv"))
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _row(df: pd.DataFrame, scenario: str, method: str) -> pd.Series:
    m = df[(df["scenario"] == scenario) & (df["method"] == method)]
    if m.empty:
        raise RuntimeError(f"Missing row in summary CSV: scenario={scenario}, method={method}")
    return m.iloc[0]


def build_scopeb_paragraph(summary_csv: Path) -> str:
    df = pd.read_csv(summary_csv)
    clear_base = _row(df, "clear", "Base")
    clear_gate = _row(df, "clear", "+gating")
    high_base = _row(df, "turbid_high", "Base")
    high_gate = _row(df, "turbid_high", "+gating")
    detector_name = str(clear_base["detector_name"]).replace("_", r"\_")

    return (
        "To avoid protocol leakage, we keep three runtime/evaluation lenses in parallel: "
        "\\textbf{oracle-detection tracking evaluation} (GT-derived detections for tracker-science comparisons), "
        "\\textbf{Scope B* passthrough} (decode+track+write with detector bypass for pipeline baselining only), and "
        "\\textbf{Scope B-true} (decode+detector inference+tracking+write for deployment-facing evidence). "
        "Scope B-true is measured by \\texttt{scripts/run\\_scopeb\\_profile\\_brackish\\_true\\_e2e.py} "
        "on BrackishMOT \\texttt{test} split with warm-up=1 and repeat=3. It reports "
        "\\{\\texttt{fps\\_e2e, mem\\_peak\\_mb\\_e2e, cpu\\_norm\\_e2e, decode\\_time, detector\\_time, "
        "tracking\\_time, write\\_time, detector\\_name, input\\_resolution}\\}. "
        "Table~\\ref{tab:scopeb_runtime} summarizes Scope B-true: on clear windows, FPS$_{\\mathrm{e2e}}$ is "
        f"{float(clear_base['fps_e2e']):.3f}$\\pm${float(clear_base['fps_e2e_std']):.3f} (Base) and "
        f"{float(clear_gate['fps_e2e']):.3f}$\\pm${float(clear_gate['fps_e2e_std']):.3f} (+gating); on turbid-high windows, "
        f"FPS$_{{\\mathrm{{e2e}}}}$ is {float(high_base['fps_e2e']):.3f}$\\pm${float(high_base['fps_e2e_std']):.3f} (Base) and "
        f"{float(high_gate['fps_e2e']):.3f}$\\pm${float(high_gate['fps_e2e_std']):.3f} (+gating). "
        "Detector time dominates stage cost in Scope B-true and increases from "
        f"{float(clear_base['detector_time']):.3f}$\\pm${float(clear_base['detector_time_std']):.3f} sec/run (clear Base) to "
        f"{float(high_base['detector_time']):.3f}$\\pm${float(high_base['detector_time_std']):.3f} sec/run (turbid-high Base), "
        f"with \\texttt{{detector\\_name}}=\\texttt{{{detector_name}}}. "
        "This directly separates true detector bottlenecks from the Scope B* passthrough baseline.\n"
    )


def patch_runtime_section(tex: str, scopeb_paragraph: str) -> str:
    pattern = re.compile(
        r"Scope B is measured by \\texttt\{scripts/run\\_scopeb\\_profile\\_brackish\.py\}.*?passthrough\.\n",
        re.S,
    )
    if pattern.search(tex):
        tex = pattern.sub(lambda _m: scopeb_paragraph, tex, count=1)
    elif "Scope B-true is measured by \\texttt{scripts/run\\_scopeb\\_profile\\_brackish\\_true\\_e2e.py}" not in tex:
        anchor = "We report Scope A and Scope B separately because they answer different deployment questions."
        tex = tex.replace(anchor, anchor + "\n\n" + scopeb_paragraph.strip() + "\n", 1)
    return tex


def patch_discussion_bullets(tex: str, summary_csv: Path) -> str:
    df = pd.read_csv(summary_csv)
    clear_base = _row(df, "clear", "Base")
    clear_gate = _row(df, "clear", "+gating")
    high_base = _row(df, "turbid_high", "Base")
    high_gate = _row(df, "turbid_high", "+gating")
    mem_values = [
        float(clear_base["mem_peak_mb_e2e"]),
        float(clear_gate["mem_peak_mb_e2e"]),
        float(high_base["mem_peak_mb_e2e"]),
        float(high_gate["mem_peak_mb_e2e"]),
    ]
    mem_min = min(mem_values)
    mem_max = max(mem_values)

    old_b1 = (
        "  \\item \\textbf{If target is at least 30 FPS and memory budget is up to 600 MB}: "
        "Base and +gating satisfy both Scope A and Brackish Scope B constraints "
        "(Scope B clear/turbid-high FPS: 216.8/232.5 for Base and 214.9/233.2 for +gating; memory about 249 MB)."
    )
    new_b1 = (
        "  \\item \\textbf{If target is at least 30 FPS and memory budget is around 1.5 GB}: "
        "both Base and +gating satisfy Brackish Scope B-true throughput on \\texttt{test} "
        f"(clear/turbid-high FPS: {float(clear_base['fps_e2e']):.1f}/{float(high_base['fps_e2e']):.1f} for Base and "
        f"{float(clear_gate['fps_e2e']):.1f}/{float(high_gate['fps_e2e']):.1f} for +gating; "
        f"peak memory range {mem_min:.1f}--{mem_max:.1f} MB)."
    )

    old_b2 = (
        "  \\item \\textbf{If detector is replaced by a heavy model in deployment}: this study's Scope B uses "
        "\\texttt{gt\\_txt\\_passthrough} detector source, so detector bottlenecks under learned detectors are not "
        "quantified here and require a dedicated follow-up profile."
    )
    new_b2 = (
        "  \\item \\textbf{If detector complexity increases in deployment}: Scope B* passthrough remains a decode+track baseline, "
        "but Scope B-true already shows detector inference as the dominant stage under the trained YOLOv8n detector; "
        "heavier detectors should be budgeted with reduced input resolution or lighter backbones before field deployment."
    )

    tex = tex.replace(old_b1, new_b1)
    tex = tex.replace(old_b2, new_b2)
    return tex


def main() -> int:
    args = parse_args()
    if not args.tex.exists():
        raise FileNotFoundError(f"main.tex not found: {args.tex}")
    if not args.summary_csv.exists():
        raise FileNotFoundError(f"summary csv not found: {args.summary_csv}")

    tex = args.tex.read_text(encoding="utf-8")
    scopeb_paragraph = build_scopeb_paragraph(args.summary_csv)
    tex = patch_runtime_section(tex, scopeb_paragraph)

    table_rel = args.scopeb_true_table.as_posix()
    fig_rel = args.scopeb_true_fig.as_posix()
    table_rel = table_rel.replace("paper/cea_draft/", "")
    fig_rel = fig_rel.replace("paper/cea_draft/", "")

    tex = tex.replace(r"\input{tables/runtime_scopeb_brackish.tex}", rf"\input{{{table_rel}}}")
    tex = tex.replace(r"\includegraphics[width=0.95\linewidth]{figures/runtime_e2e_brackish_bar.pdf}", rf"\includegraphics[width=0.95\linewidth]{{{fig_rel}}}")
    tex = tex.replace(
        r"\caption{Scope B end-to-end runtime profile on BrackishMOT (FPS, peak memory, normalized CPU).}",
        r"\caption{Scope B-true end-to-end runtime profile on BrackishMOT (FPS, peak memory, normalized CPU).}",
    )
    tex = patch_discussion_bullets(tex, args.summary_csv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex, encoding="utf-8", newline="\n")
    print(f"[patch-maintex] updated: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
