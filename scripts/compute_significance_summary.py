#!/usr/bin/env python
"""Build paired significance summaries for multi-seed MOT metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon

from eval_count_stability import compute_count_metrics, load_frame_counts, read_seq_length, read_seqmap
from method_labels import MAIN_CHAIN_METHOD_ORDER


HIGHER_IS_BETTER = {
    "HOTA": True,
    "IDF1": True,
    "IDSW": False,
    "CountMAE": False,
}
METRICS = ["HOTA", "IDF1", "IDSW", "CountMAE"]
INTERNAL_METHODS = [
    ("Base", "base", "mean_base.csv", "pred_base"),
    (MAIN_CHAIN_METHOD_ORDER[1], "gating", "mean_gating.csv", "pred_gating"),
    (MAIN_CHAIN_METHOD_ORDER[2], "traj", "mean_traj.csv", "pred_traj"),
    (MAIN_CHAIN_METHOD_ORDER[3], "adaptive", "mean_adaptive.csv", "pred_adaptive"),
]
STRONG_METHODS = [
    ("ByteTrack", "bytetrack"),
    ("OC-SORT", "ocsort"),
    ("BoT-SORT", "botsort"),
]
PLOT_STYLE = {
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
}


@dataclass(frozen=True)
class MetricRecord:
    method: str
    seed: int
    metrics: Dict[str, float]
    source_kind: str
    mean_csv: str
    pred_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute paired significance summaries from multi-seed outputs.")
    parser.add_argument(
        "--run-config",
        type=Path,
        default=Path("results/main_val/run_config.json"),
        help="Path to run_config.json.",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional comma-separated seed override. Default: use run_config seeds.",
    )
    parser.add_argument(
        "--seed-root",
        type=Path,
        default=Path("results/main_val/seed_runs"),
        help="Root containing seed_<id>/mean_*.csv and pred_* outputs.",
    )
    parser.add_argument(
        "--strong-root",
        type=Path,
        default=Path("results/main_val/strong_baselines"),
        help="Root containing strong baseline seed outputs.",
    )
    parser.add_argument(
        "--mot-root",
        type=Path,
        default=Path("data/mft25_mot_full"),
        help="Prepared MOT split root.",
    )
    parser.add_argument("--split", default="val_half", help="Prepared split name.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1000,
        help="Frame cap used for count metrics.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Alpha level used when flagging significance.",
    )
    parser.add_argument(
        "--seed-metrics-csv",
        type=Path,
        default=None,
        help="Optional raw per-seed metrics CSV path.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Output summary CSV path. Default: <tables_dir>/significance_summary.csv",
    )
    parser.add_argument(
        "--summary-tex",
        type=Path,
        default=None,
        help="Output summary TEX path. Default: <paper_assets_dir>/significance_summary.tex",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Optional heatmap output path. Default: <paper_assets_dir>/significance_heatmap.png",
    )
    parser.add_argument(
        "--paragraph-out",
        type=Path,
        default=None,
        help="Optional paper paragraph text output path.",
    )
    parser.add_argument(
        "--skip-strong",
        action="store_true",
        help="Only compare Base to internal variants.",
    )
    return parser.parse_args()


def _norm(path: Path) -> str:
    return str(path).replace("\\", "/").lower()


def _is_legacy_output_path(path: Path, result_root: Path) -> bool:
    norm = _norm(path)
    root_norm = _norm(result_root)
    return norm.startswith("results/") and norm != root_norm and not norm.startswith(root_norm + "/")


def parse_seed_list(raw: str) -> List[int]:
    seeds: List[int] = []
    seen = set()
    for token in [x.strip() for x in str(raw).split(",") if x.strip()]:
        value = int(token)
        if value not in seen:
            seeds.append(value)
            seen.add(value)
    return seeds


def apply_run_config(args: argparse.Namespace) -> None:
    if not args.run_config.exists():
        raise FileNotFoundError(f"Run config not found: {args.run_config}")
    cfg = json.loads(args.run_config.read_text(encoding="utf-8"))

    result_root = Path(str(cfg.get("result_root", "results/main_val")))
    tables_dir = Path(str(cfg.get("tables_dir", result_root / "tables")))
    paper_assets_dir = Path(str(cfg.get("paper_assets_dir", result_root / "paper_assets")))

    args.seed_root = Path(str(cfg.get("seed_root", args.seed_root)))
    args.mot_root = Path(str(cfg.get("mot_root", args.mot_root)))
    args.split = str(cfg.get("split", args.split))
    args.max_frames = int(cfg.get("max_frames", args.max_frames))
    if not args.seeds.strip():
        cfg_seeds = cfg.get("seeds", [])
        if isinstance(cfg_seeds, list) and cfg_seeds:
            args.seeds = ",".join(str(int(x)) for x in cfg_seeds)

    if args.seed_metrics_csv is None:
        args.seed_metrics_csv = tables_dir / "significance_seed_metrics.csv"
    if args.summary_csv is None:
        args.summary_csv = tables_dir / "significance_summary.csv"
    if args.summary_tex is None:
        args.summary_tex = paper_assets_dir / "significance_summary.tex"
    if args.plot_path is None:
        args.plot_path = paper_assets_dir / "significance_heatmap.png"
    if args.paragraph_out is None:
        args.paragraph_out = tables_dir / "significance_paragraph.txt"

    write_targets = [
        ("seed_metrics_csv", args.seed_metrics_csv),
        ("summary_csv", args.summary_csv),
        ("summary_tex", args.summary_tex),
        ("plot_path", args.plot_path),
        ("paragraph_out", args.paragraph_out),
    ]
    bad = [f"{name}={path}" for name, path in write_targets if _is_legacy_output_path(path, result_root)]
    if bad:
        raise RuntimeError(
            "run-config strict mode blocks legacy/global outputs. "
            f"result_root={result_root}. bad targets: {'; '.join(bad)}"
        )


def read_single_row(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly 1 row in {path}, got {len(rows)}")
    return rows[0]


def compute_mean_count_mae(split_dir: Path, pred_dir: Path, max_frames: int, sequences: Sequence[str]) -> float:
    per_seq_mae: List[float] = []
    for seq in sequences:
        seq_dir = split_dir / seq
        gt_path = seq_dir / "gt" / "gt.txt"
        seqinfo = seq_dir / "seqinfo.ini"
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT file: {gt_path}")
        if not seqinfo.exists():
            raise FileNotFoundError(f"Missing seqinfo.ini: {seqinfo}")
        seq_len = read_seq_length(seqinfo)
        used_frames = seq_len if max_frames <= 0 else min(seq_len, max_frames)
        used_frames = max(1, used_frames)
        gt_counts = load_frame_counts(gt_path, used_frames)
        pred_counts = load_frame_counts(pred_dir / f"{seq}.txt", used_frames)
        metrics = compute_count_metrics(gt_counts, pred_counts)
        per_seq_mae.append(float(metrics["CountMAE"]))
    return float(np.mean(per_seq_mae))


def collect_records(
    split_dir: Path,
    sequences: Sequence[str],
    seed_root: Path,
    strong_root: Path,
    seeds: Sequence[int],
    max_frames: int,
    skip_strong: bool,
) -> List[MetricRecord]:
    records: List[MetricRecord] = []
    count_cache: Dict[str, float] = {}

    def get_count_mae(pred_dir: Path) -> float:
        key = str(pred_dir.resolve())
        if key not in count_cache:
            count_cache[key] = compute_mean_count_mae(split_dir, pred_dir, max_frames, sequences)
        return count_cache[key]

    for seed in seeds:
        seed_dir = seed_root / f"seed_{seed}"
        if seed_dir.exists():
            for display, _, mean_name, pred_folder in INTERNAL_METHODS:
                mean_csv = seed_dir / mean_name
                pred_dir = seed_dir / pred_folder
                if not mean_csv.exists() or not pred_dir.exists():
                    continue
                row = read_single_row(mean_csv)
                records.append(
                    MetricRecord(
                        method=display,
                        seed=seed,
                        metrics={
                            "HOTA": float(row["HOTA"]),
                            "IDF1": float(row["IDF1"]),
                            "IDSW": float(row["IDSW"]),
                            "CountMAE": get_count_mae(pred_dir),
                        },
                        source_kind="internal",
                        mean_csv=str(mean_csv).replace("\\", "/"),
                        pred_dir=str(pred_dir).replace("\\", "/"),
                    )
                )

        if skip_strong:
            continue

        for display, method_key in STRONG_METHODS:
            method_seed_dir = strong_root / method_key / f"seed_{seed}"
            mean_csv = method_seed_dir / "mean.csv"
            pred_dir = method_seed_dir / "pred"
            if not mean_csv.exists() or not pred_dir.exists():
                continue
            row = read_single_row(mean_csv)
            records.append(
                MetricRecord(
                    method=display,
                    seed=seed,
                    metrics={
                        "HOTA": float(row["HOTA"]),
                        "IDF1": float(row["IDF1"]),
                        "IDSW": float(row["IDSW"]),
                        "CountMAE": get_count_mae(pred_dir),
                    },
                    source_kind="strong",
                    mean_csv=str(mean_csv).replace("\\", "/"),
                    pred_dir=str(pred_dir).replace("\\", "/"),
                )
            )
    return records


def write_seed_metrics_csv(path: Path, records: Sequence[MetricRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["seed", "method", "source_kind", "HOTA", "IDF1", "IDSW", "CountMAE", "mean_csv", "pred_dir"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in sorted(records, key=lambda x: (x.method, x.seed)):
            writer.writerow(
                {
                    "seed": rec.seed,
                    "method": rec.method,
                    "source_kind": rec.source_kind,
                    "HOTA": f"{rec.metrics['HOTA']:.6f}",
                    "IDF1": f"{rec.metrics['IDF1']:.6f}",
                    "IDSW": f"{rec.metrics['IDSW']:.6f}",
                    "CountMAE": f"{rec.metrics['CountMAE']:.6f}",
                    "mean_csv": rec.mean_csv,
                    "pred_dir": rec.pred_dir,
                }
            )


def fmt_pm(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.3f} +- {std_value:.3f}"


def fmt_pm_tex(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.3f} $\\pm$ {std_value:.3f}"


def fmt_pvalue(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    if value < 0.001:
        return "<0.001"
    return f"{value:.4f}"


def metric_advantage(metric: str, compared_mean: float, base_mean: float) -> float:
    if HIGHER_IS_BETTER[metric]:
        return compared_mean - base_mean
    return base_mean - compared_mean


def favored_method(metric: str, compared_mean: float, base_mean: float, compared_method: str) -> str:
    adv = metric_advantage(metric, compared_mean, base_mean)
    if abs(adv) < 1e-12:
        return "Tie"
    return compared_method if adv > 0 else "Base"


def normality_ok(diffs: np.ndarray, alpha: float) -> tuple[bool, float | None]:
    if diffs.size < 3:
        return False, math.nan
    if np.allclose(diffs, diffs[0]):
        return False, math.nan
    try:
        p_value = float(shapiro(diffs).pvalue)
    except Exception:  # noqa: BLE001
        return False, math.nan
    return diffs.size >= 5 and p_value >= alpha, p_value


def paired_t_pvalue(base_vals: np.ndarray, cmp_vals: np.ndarray) -> float | None:
    diffs = cmp_vals - base_vals
    if diffs.size < 2:
        return math.nan
    if np.allclose(diffs, 0.0):
        return 1.0
    try:
        return float(ttest_rel(cmp_vals, base_vals).pvalue)
    except Exception:  # noqa: BLE001
        return math.nan


def wilcoxon_pvalue(base_vals: np.ndarray, cmp_vals: np.ndarray) -> float | None:
    diffs = cmp_vals - base_vals
    if diffs.size < 2:
        return math.nan
    if np.allclose(diffs, 0.0):
        return 1.0
    try:
        return float(wilcoxon(diffs).pvalue)
    except Exception:  # noqa: BLE001
        return math.nan


def summarize_notes(n: int, used_test: str, shapiro_p: float | None) -> str:
    notes: List[str] = []
    if n < 5:
        notes.append("n<5; low power")
    if used_test == "Wilcoxon":
        notes.append("robust test preferred")
    if shapiro_p is not None and not math.isnan(shapiro_p) and shapiro_p < 0.05:
        notes.append("paired differences non-normal")
    if not notes:
        notes.append("descriptive and inferential results agree")
    return "; ".join(notes)


def build_summary_rows(records: Sequence[MetricRecord], alpha: float) -> List[Dict[str, str]]:
    by_method_seed: Dict[str, Dict[int, MetricRecord]] = {}
    for rec in records:
        by_method_seed.setdefault(rec.method, {})[rec.seed] = rec

    if "Base" not in by_method_seed:
        raise RuntimeError("Base records not found. Cannot run paired comparisons.")

    compare_methods = [m for m in by_method_seed.keys() if m != "Base"]
    preferred_order = [MAIN_CHAIN_METHOD_ORDER[1], MAIN_CHAIN_METHOD_ORDER[2], MAIN_CHAIN_METHOD_ORDER[3], "ByteTrack", "OC-SORT", "BoT-SORT"]
    compare_methods = sorted(compare_methods, key=lambda name: preferred_order.index(name) if name in preferred_order else 999)

    rows: List[Dict[str, str]] = []
    for compared_method in compare_methods:
        common_seeds = sorted(set(by_method_seed["Base"]) & set(by_method_seed[compared_method]))
        for metric in METRICS:
            base_vals = []
            cmp_vals = []
            seeds_used = []
            for seed in common_seeds:
                base_val = by_method_seed["Base"][seed].metrics.get(metric)
                cmp_val = by_method_seed[compared_method][seed].metrics.get(metric)
                if base_val is None or cmp_val is None or math.isnan(base_val) or math.isnan(cmp_val):
                    continue
                seeds_used.append(seed)
                base_vals.append(float(base_val))
                cmp_vals.append(float(cmp_val))

            base_arr = np.asarray(base_vals, dtype=float)
            cmp_arr = np.asarray(cmp_vals, dtype=float)
            n = int(base_arr.size)
            base_mean = float(np.mean(base_arr)) if n else math.nan
            cmp_mean = float(np.mean(cmp_arr)) if n else math.nan
            base_std = float(np.std(base_arr, ddof=0)) if n else math.nan
            cmp_std = float(np.std(cmp_arr, ddof=0)) if n else math.nan
            mean_diff = float(np.mean(cmp_arr - base_arr)) if n else math.nan
            diff_std = float(np.std(cmp_arr - base_arr, ddof=0)) if n else math.nan

            t_p = paired_t_pvalue(base_arr, cmp_arr)
            use_t, shapiro_p = normality_ok(cmp_arr - base_arr, alpha) if n else (False, math.nan)
            w_p = wilcoxon_pvalue(base_arr, cmp_arr)
            used_test = "paired_t" if use_t and not math.isnan(t_p) else "Wilcoxon"
            preferred_p = t_p if used_test == "paired_t" else w_p
            significant = (
                "1" if preferred_p is not None and not math.isnan(preferred_p) and preferred_p < alpha else "0"
            ) if n >= 2 else ""
            rows.append(
                {
                    "base_method": "Base",
                    "compared_method": compared_method,
                    "metric": metric,
                    "better_direction": "higher" if HIGHER_IS_BETTER[metric] else "lower",
                    "n": str(n),
                    "seeds": ",".join(str(seed) for seed in seeds_used),
                    "base_mean": f"{base_mean:.6f}" if n else "",
                    "base_std": f"{base_std:.6f}" if n else "",
                    "compared_mean": f"{cmp_mean:.6f}" if n else "",
                    "compared_std": f"{cmp_std:.6f}" if n else "",
                    "base_mean_pm_std": fmt_pm(base_mean, base_std) if n else "",
                    "compared_mean_pm_std": fmt_pm(cmp_mean, cmp_std) if n else "",
                    "mean_diff_compared_minus_base": f"{mean_diff:.6f}" if n else "",
                    "diff_std": f"{diff_std:.6f}" if n else "",
                    "advantage_for_compared": f"{metric_advantage(metric, cmp_mean, base_mean):.6f}" if n else "",
                    "favours": favored_method(metric, cmp_mean, base_mean, compared_method) if n else "",
                    "shapiro_p": f"{shapiro_p:.6f}" if shapiro_p is not None and not math.isnan(shapiro_p) else "",
                    "paired_t_p": f"{t_p:.6f}" if t_p is not None and not math.isnan(t_p) else "",
                    "wilcoxon_p": f"{w_p:.6f}" if w_p is not None and not math.isnan(w_p) else "",
                    "preferred_test": used_test if n >= 2 else "",
                    "preferred_p": f"{preferred_p:.6f}" if preferred_p is not None and not math.isnan(preferred_p) else "",
                    "significant_alpha_0p05": significant,
                    "note": summarize_notes(n, used_test, shapiro_p) if n else "insufficient paired seeds",
                }
            )
    return rows


def write_summary_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "base_method",
        "compared_method",
        "metric",
        "better_direction",
        "n",
        "seeds",
        "base_mean",
        "base_std",
        "compared_mean",
        "compared_std",
        "base_mean_pm_std",
        "compared_mean_pm_std",
        "mean_diff_compared_minus_base",
        "diff_std",
        "advantage_for_compared",
        "favours",
        "shapiro_p",
        "paired_t_p",
        "wilcoxon_p",
        "preferred_test",
        "preferred_p",
        "significant_alpha_0p05",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def write_summary_tex(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "% Auto-generated by scripts/compute_significance_summary.py",
        r"\begin{tabular}{llllllll}",
        r"\toprule",
        r"Compare & Metric & n & Base mean$\pm$std & Compared mean$\pm$std & $\Delta$ (cmp-base) & Test & $p$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    tex_escape(row["compared_method"]),
                    tex_escape(row["metric"]),
                    row["n"],
                    fmt_pm_tex(float(row["base_mean"]), float(row["base_std"])) if row["base_mean"] else "n/a",
                    fmt_pm_tex(float(row["compared_mean"]), float(row["compared_std"])) if row["compared_mean"] else "n/a",
                    f"{float(row['mean_diff_compared_minus_base']):+.3f}" if row["mean_diff_compared_minus_base"] else "n/a",
                    tex_escape(row["preferred_test"] or "n/a"),
                    fmt_pvalue(float(row["preferred_p"])) if row["preferred_p"] else "n/a",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_heatmap(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    if not rows:
        return
    plt.rcParams.update(PLOT_STYLE)
    methods = []
    for row in rows:
        method = row["compared_method"]
        if method not in methods:
            methods.append(method)

    value_grid = np.full((len(methods), len(METRICS)), np.nan, dtype=float)
    annot_grid = [["" for _ in METRICS] for _ in methods]
    for row in rows:
        method_idx = methods.index(row["compared_method"])
        metric_idx = METRICS.index(row["metric"])
        value = float(row["advantage_for_compared"]) if row["advantage_for_compared"] else math.nan
        value_grid[method_idx, metric_idx] = value
        p_display = fmt_pvalue(float(row["preferred_p"])) if row["preferred_p"] else "n/a"
        star = "*" if row["significant_alpha_0p05"] == "1" else ""
        annot_grid[method_idx][metric_idx] = f"{value:+.3f}\np={p_display}{star}" if not math.isnan(value) else "n/a"

    color_grid = value_grid.copy()
    for col_idx in range(color_grid.shape[1]):
        finite_col = color_grid[:, col_idx][np.isfinite(color_grid[:, col_idx])]
        if finite_col.size == 0:
            continue
        denom = max(float(np.max(np.abs(finite_col))), 1e-6)
        color_grid[:, col_idx] = color_grid[:, col_idx] / denom

    fig, ax = plt.subplots(figsize=(8.8, 2.2 + 0.55 * len(methods)), constrained_layout=True)
    im = ax.imshow(color_grid, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels(METRICS)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title("Paired seed advantage vs Base (column-normalized color; text shows raw delta)")
    for i in range(len(methods)):
        for j in range(len(METRICS)):
            ax.text(j, i, annot_grid[i][j], ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("Normalized signed advantage")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def build_paragraph(rows: Sequence[Dict[str, str]]) -> str:
    n_values = sorted({int(row["n"]) for row in rows if row["n"]})
    seed_text = ", ".join(str(x) for x in n_values) if n_values else "n/a"
    return (
        "To assess whether the observed ranking differences were stable across repeated runs, "
        "we performed paired statistical comparisons on the seed-level outputs of the audited "
        "validation pipeline. For each compared method, we extracted per-seed HOTA, IDF1, IDSW, "
        "and CountMAE values and evaluated Base against Base+gating, Base+gating+traj, "
        "Base+gating+traj+adaptive, and any strong "
        "baseline whose per-seed outputs were available under the same split. We report mean +- std "
        "over seeds together with two paired tests: a paired t-test when the paired differences were "
        "sufficiently close to normal and the sample size was large enough to justify that assumption, "
        "and a Wilcoxon signed-rank test as the default robust alternative otherwise. Because the current "
        f"paired sample counts are limited (n in {{{seed_text}}}), these p-values should be interpreted "
        "conservatively as a stability check rather than as a claim of definitive superiority; in the main "
        "text, we therefore prioritize the direction and consistency of the seed-level differences over any "
        "single threshold crossing."
    )


def main() -> None:
    args = parse_args()
    apply_run_config(args)
    seeds = parse_seed_list(args.seeds)
    if not seeds:
        raise RuntimeError("No seeds provided or resolved from run_config.")

    split_dir = args.mot_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Prepared split not found: {split_dir}")
    sequences = read_seqmap(split_dir / "seqmaps" / f"{args.split}.txt")
    if not sequences:
        raise RuntimeError(f"No sequences found for split={args.split}")

    records = collect_records(
        split_dir=split_dir,
        sequences=sequences,
        seed_root=args.seed_root,
        strong_root=args.strong_root,
        seeds=seeds,
        max_frames=args.max_frames,
        skip_strong=args.skip_strong,
    )
    if not records:
        raise RuntimeError("No per-seed records found.")

    write_seed_metrics_csv(args.seed_metrics_csv, records)
    summary_rows = build_summary_rows(records, alpha=args.alpha)
    write_summary_csv(args.summary_csv, summary_rows)
    write_summary_tex(args.summary_tex, summary_rows)
    write_heatmap(args.plot_path, summary_rows)
    paragraph = build_paragraph(summary_rows)
    args.paragraph_out.parent.mkdir(parents=True, exist_ok=True)
    args.paragraph_out.write_text(paragraph + "\n", encoding="utf-8")

    print(f"Saved seed metrics: {args.seed_metrics_csv}")
    print(f"Saved summary csv: {args.summary_csv}")
    print(f"Saved summary tex: {args.summary_tex}")
    print(f"Saved plot:        {args.plot_path}")
    print(f"Saved paragraph:   {args.paragraph_out}")


if __name__ == "__main__":
    main()
