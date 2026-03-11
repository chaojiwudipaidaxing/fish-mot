#!/usr/bin/env python
"""Aggregate multi-seed val main tables into mean/std tables + optional plot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from method_labels import MAIN_CHAIN_METHOD_ORDER, normalize_main_chain_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-seed main_table_val outputs.")
    parser.add_argument(
        "--seed-root",
        type=Path,
        default=Path("results/main_val/seed_runs"),
        help="Root directory containing seed_<id>/main_table_val.csv.",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma separated seed list, e.g. 0,1,2.",
    )
    parser.add_argument(
        "--out-mean",
        type=Path,
        default=Path("results/main_table_val_seedmean.csv"),
        help="Mean table output CSV path.",
    )
    parser.add_argument(
        "--out-std",
        type=Path,
        default=Path("results/main_table_val_seedstd.csv"),
        help="Std table output CSV path.",
    )
    parser.add_argument(
        "--out-per-seq",
        type=Path,
        default=Path("results/per_seq_main_val.csv"),
        help="Combined per-sequence (all seeds) output CSV path.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("results/paper_assets/main_table_val_seedmean_std.png"),
        help="Output path for mean±std bar plot.",
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=None,
        help="Optional run_config.json to resolve result_root paths.",
    )
    return parser.parse_args()


def _norm(path: Path) -> str:
    return str(path).replace("\\", "/").lower()


def _is_legacy_output_path(path: Path, result_root: Path) -> bool:
    norm = _norm(path)
    root_norm = _norm(result_root)
    return norm.startswith("results/") and norm != root_norm and not norm.startswith(root_norm + "/")


def apply_run_config(args: argparse.Namespace) -> None:
    if args.run_config is None:
        return
    if not args.run_config.exists():
        raise FileNotFoundError(f"Run config not found: {args.run_config}")
    cfg = json.loads(args.run_config.read_text(encoding="utf-8"))

    result_root = Path(str(cfg.get("result_root", "results/main_val")))
    seed_root_cfg = Path(str(cfg.get("seed_root", result_root / "seed_runs")))
    tables_dir = Path(str(cfg.get("tables_dir", result_root / "tables")))
    paper_assets_dir = Path(str(cfg.get("paper_assets_dir", result_root / "paper_assets")))

    if _norm(args.seed_root) == _norm(Path("results/main_val/seed_runs")):
        args.seed_root = seed_root_cfg
    if _norm(args.out_mean) == _norm(Path("results/main_table_val_seedmean.csv")):
        args.out_mean = tables_dir / "main_table_val_seedmean.csv"
    if _norm(args.out_std) == _norm(Path("results/main_table_val_seedstd.csv")):
        args.out_std = tables_dir / "main_table_val_seedstd.csv"
    if _norm(args.out_per_seq) == _norm(Path("results/per_seq_main_val.csv")):
        args.out_per_seq = tables_dir / "per_seq_main_val.csv"
    if _norm(args.plot_path) == _norm(Path("results/paper_assets/main_table_val_seedmean_std.png")):
        args.plot_path = paper_assets_dir / "main_table_val_seedmean_std.png"

    if args.seeds.strip() == "0,1,2":
        seeds = cfg.get("seeds")
        if isinstance(seeds, list) and seeds:
            args.seeds = ",".join(str(int(x)) for x in seeds)
        elif isinstance(seeds, str) and seeds.strip():
            args.seeds = seeds

    write_targets = [
        ("out_mean", args.out_mean),
        ("out_std", args.out_std),
        ("out_per_seq", args.out_per_seq),
        ("plot_path", args.plot_path),
    ]
    bad = [f"{name}={path}" for name, path in write_targets if _is_legacy_output_path(path, result_root)]
    if bad:
        raise RuntimeError(
            "run-config strict mode blocks legacy/global outputs. "
            f"result_root={result_root}. bad targets: {'; '.join(bad)}"
        )


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    apply_run_config(args)
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise RuntimeError("No seeds specified.")

    metric_names = ["HOTA", "DetA", "AssA", "IDF1", "IDSW"]
    methods_order = MAIN_CHAIN_METHOD_ORDER

    by_method: Dict[str, Dict[str, List[float]]] = {}
    split_name = "val_half"
    combined_per_seq: List[Dict[str, str]] = []

    for seed in seeds:
        seed_dir = args.seed_root / f"seed_{seed}"
        main_table = seed_dir / "main_table_val.csv"
        per_seq_table = seed_dir / "per_seq_main_val.csv"
        rows = read_rows(main_table)
        for row in rows:
            method = normalize_main_chain_label(row["method"])
            split_name = row.get("split", split_name)
            if method not in by_method:
                by_method[method] = {m: [] for m in metric_names}
            for metric in metric_names:
                by_method[method][metric].append(float(row[metric]))

        seq_rows = read_rows(per_seq_table)
        for row in seq_rows:
            out_row = dict(row)
            out_row["method"] = normalize_main_chain_label(out_row["method"])
            out_row["seed"] = seed
            combined_per_seq.append(out_row)

    mean_rows: List[Dict[str, str]] = []
    std_rows: List[Dict[str, str]] = []
    for method in methods_order:
        if method not in by_method:
            continue
        mean_rows.append(
            {
                "split": split_name,
                "method": method,
                "HOTA": f"{np.mean(by_method[method]['HOTA']):.3f}",
                "DetA": f"{np.mean(by_method[method]['DetA']):.3f}",
                "AssA": f"{np.mean(by_method[method]['AssA']):.3f}",
                "IDF1": f"{np.mean(by_method[method]['IDF1']):.3f}",
                "IDSW": f"{np.mean(by_method[method]['IDSW']):.3f}",
            }
        )
        std_rows.append(
            {
                "split": split_name,
                "method": method,
                "HOTA": f"{np.std(by_method[method]['HOTA'], ddof=0):.3f}",
                "DetA": f"{np.std(by_method[method]['DetA'], ddof=0):.3f}",
                "AssA": f"{np.std(by_method[method]['AssA'], ddof=0):.3f}",
                "IDF1": f"{np.std(by_method[method]['IDF1'], ddof=0):.3f}",
                "IDSW": f"{np.std(by_method[method]['IDSW'], ddof=0):.3f}",
            }
        )

    fields = ["split", "method", "HOTA", "DetA", "AssA", "IDF1", "IDSW"]
    args.out_mean.parent.mkdir(parents=True, exist_ok=True)
    with args.out_mean.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in mean_rows:
            writer.writerow(row)

    args.out_std.parent.mkdir(parents=True, exist_ok=True)
    with args.out_std.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in std_rows:
            writer.writerow(row)

    if combined_per_seq:
        per_seq_fields = ["seed", "split", "method", "sequence", "used_frames", "gt_rows", "pred_rows", "HOTA", "DetA", "AssA", "IDF1", "IDSW"]
        args.out_per_seq.parent.mkdir(parents=True, exist_ok=True)
        with args.out_per_seq.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_seq_fields)
            writer.writeheader()
            for row in combined_per_seq:
                writer.writerow(row)

    # Optional paper asset: mean±std bar chart for key metrics.
    metrics_for_plot = ["HOTA", "AssA", "IDF1"]
    methods = [r["method"] for r in mean_rows]
    x = np.arange(len(methods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for i, metric in enumerate(metrics_for_plot):
        means = [float(next(r[metric] for r in mean_rows if r["method"] == m)) for m in methods]
        stds = [float(next(r[metric] for r in std_rows if r["method"] == m)) for m in methods]
        ax.bar(x + (i - 1) * width, means, width=width, yerr=stds, capsize=3, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Score")
    ax.set_title("val_half main results (mean ± std over seeds)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best")
    args.plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot_path, dpi=180)
    plt.close(fig)

    print(f"Saved seed-mean table: {args.out_mean}")
    print(f"Saved seed-std table:  {args.out_std}")
    print(f"Saved per-seq table:   {args.out_per_seq}")
    print(f"Saved plot:            {args.plot_path}")


if __name__ == "__main__":
    main()
