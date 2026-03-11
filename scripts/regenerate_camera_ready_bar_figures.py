#!/usr/bin/env python3
"""Regenerate manuscript bar figures from existing CSV summaries only."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from method_labels import MAIN_CHAIN_METHOD_ORDER, normalize_main_chain_label

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = REPO_ROOT / "results" / "main_val" / "tables"
DEFAULT_OUT_DIR = REPO_ROOT / "paper" / "cea_draft" / "figures"

BAR_COLORS = {
    "HOTA": "#4E79A7",
    "AssA": "#F28E2B",
    "IDF1": "#59A14F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate camera-ready bar figures from existing CSVs.")
    parser.add_argument("--table-dir", type=Path, default=TABLE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "axes.edgecolor": "#56606B",
            "axes.linewidth": 0.8,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote: {out_path}")


def _plot_grouped_bars(
    methods: list[str],
    metrics: list[str],
    mean_map: dict[str, dict[str, float]],
    std_map: dict[str, dict[str, float]],
    out_path: Path,
    title: str,
) -> None:
    _apply_style()
    x = np.arange(len(methods))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True)
    for idx, metric in enumerate(metrics):
        means = [mean_map[method][metric] for method in methods]
        stds = [std_map[method][metric] for method in methods]
        ax.bar(
            x + (idx - 1) * width,
            means,
            width=width,
            yerr=stds,
            capsize=2.6,
            linewidth=0.6,
            color=BAR_COLORS[metric],
            label=metric,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=18, ha="right")
    ax.set_ylabel("Score (0-100)")
    ax.set_ylim(0, 80)
    ax.set_title(title)
    ax.grid(axis="y")
    ax.legend(loc="upper right", frameon=False)
    _save(fig, out_path)


def _rows_to_metric_map(rows: list[dict[str, str]], metrics: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        method = normalize_main_chain_label(row["method"])
        out[method] = {metric: float(row[metric]) for metric in metrics}
    return out


def _rows_to_combined_maps(rows: list[dict[str, str]], metrics: list[str]) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    mean_map: dict[str, dict[str, float]] = {}
    std_map: dict[str, dict[str, float]] = {}
    for row in rows:
        method = normalize_main_chain_label(row["method"])
        mean_map[method] = {metric: float(row[f"{metric}_mean"]) for metric in metrics}
        std_map[method] = {metric: float(row[f"{metric}_std"]) for metric in metrics}
    return mean_map, std_map


def main() -> None:
    args = parse_args()
    metrics = ["HOTA", "AssA", "IDF1"]

    core_mean = _rows_to_metric_map(_read_rows(args.table_dir / "main_table_val_seedmean.csv"), metrics)
    core_std = _rows_to_metric_map(_read_rows(args.table_dir / "main_table_val_seedstd.csv"), metrics)
    core_methods = MAIN_CHAIN_METHOD_ORDER
    _plot_grouped_bars(
        core_methods,
        metrics,
        core_mean,
        core_std,
        args.out_dir / "main_table_val_seedmean_std_paper.png",
        r"Core metrics on val-half (mean $\pm$ std)",
    )

    strong_mean = _rows_to_metric_map(_read_rows(args.table_dir / "strong_baselines_seedmean.csv"), metrics)
    strong_std = _rows_to_metric_map(_read_rows(args.table_dir / "strong_baselines_seedstd.csv"), metrics)
    strong_methods = ["ByteTrack", "OC-SORT", "BoT-SORT"]
    _plot_grouped_bars(
        strong_methods,
        metrics,
        strong_mean,
        strong_std,
        args.out_dir / "strong_baselines_seedmean_std.png",
        r"Strong baselines on val-half (mean $\pm$ std)",
    )

    combined_rows = _read_rows(args.table_dir / "main_table_val_with_baselines.csv")
    combined_mean, combined_std = _rows_to_combined_maps(combined_rows, metrics)
    combined_methods = MAIN_CHAIN_METHOD_ORDER + ["ByteTrack", "OC-SORT", "BoT-SORT"]
    _plot_grouped_bars(
        combined_methods,
        metrics,
        combined_mean,
        combined_std,
        args.out_dir / "main_table_with_baselines.png",
        r"Combined comparison on val-half (mean $\pm$ std)",
    )


if __name__ == "__main__":
    main()
