#!/usr/bin/env python
"""Evaluate counting stability metrics from MOT-format GT/pred files."""

from __future__ import annotations

import argparse
import configparser
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from method_labels import MAIN_CHAIN_METHOD_ORDER


METHOD_SPECS: List[Tuple[str, str]] = [
    (MAIN_CHAIN_METHOD_ORDER[0], "pred_base"),
    (MAIN_CHAIN_METHOD_ORDER[1], "pred_gating"),
    (MAIN_CHAIN_METHOD_ORDER[2], "pred_traj"),
    (MAIN_CHAIN_METHOD_ORDER[3], "pred_adaptive"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate count stability on val/train split.")
    parser.add_argument("--split", default="val_half", help="Prepared split name under data/mft25_mot.")
    parser.add_argument(
        "--mot-root",
        type=Path,
        default=Path("data/mft25_mot"),
        help="Root directory of prepared MOT data.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1000,
        help="Use up to this many frames per sequence (0 means all).",
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=Path("results/main_val/seed_runs/seed_0"),
        help="Root directory containing pred_base/pred_gating/pred_traj/pred_adaptive.",
    )
    parser.add_argument("--pred-base", type=Path, default=None, help="Override Base prediction directory.")
    parser.add_argument("--pred-gating", type=Path, default=None, help="Override Base+gating prediction directory.")
    parser.add_argument("--pred-traj", type=Path, default=None, help="Override Base+gating+traj prediction directory.")
    parser.add_argument("--pred-adaptive", type=Path, default=None, help="Override Base+gating+traj+adaptive prediction directory.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/count_metrics_val.csv"),
        help="Output CSV path (contains per-seq rows and mean rows).",
    )
    parser.add_argument(
        "--paper-assets-dir",
        type=Path,
        default=Path("results/paper_assets_val"),
        help="Directory to write paper-ready plots.",
    )
    parser.add_argument(
        "--plot-name",
        default="count_stability_bar.png",
        help="Plot filename under --paper-assets-dir.",
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=None,
        help="Optional run_config.json. If provided, result_root paths are used by default.",
    )
    return parser.parse_args()


def _norm(path: Path) -> str:
    return str(path).replace("\\", "/").lower()


def apply_run_config(args: argparse.Namespace) -> None:
    if args.run_config is None:
        return
    if not args.run_config.exists():
        raise FileNotFoundError(f"Run config not found: {args.run_config}")
    cfg = json.loads(args.run_config.read_text(encoding="utf-8"))
    args.split = str(cfg.get("split", args.split))
    args.mot_root = Path(str(cfg.get("mot_root", args.mot_root)))
    args.max_frames = int(cfg.get("max_frames", args.max_frames))

    pred_root = cfg.get("pred_root")
    if pred_root and _norm(args.pred_root) == _norm(Path("results/main_val/seed_runs/seed_0")):
        args.pred_root = Path(str(pred_root))

    result_root = Path(str(cfg.get("result_root", "results/main_val")))
    tables_dir = Path(str(cfg.get("tables_dir", result_root / "tables")))
    paper_assets_dir = Path(str(cfg.get("paper_assets_dir", result_root / "paper_assets")))
    if _norm(args.output_csv) == _norm(Path("results/count_metrics_val.csv")):
        args.output_csv = tables_dir / "count_metrics_val.csv"
    if _norm(args.paper_assets_dir) == _norm(Path("results/paper_assets_val")):
        args.paper_assets_dir = paper_assets_dir
    root_norm = _norm(result_root)
    write_targets = [
        ("output_csv", args.output_csv),
        ("paper_assets_dir", args.paper_assets_dir),
    ]
    bad = [f"{name}={path}" for name, path in write_targets if _norm(path).startswith("results/") and _norm(path) != root_norm and not _norm(path).startswith(root_norm + "/")]
    if bad:
        raise RuntimeError(
            "run-config strict mode blocks legacy/global outputs. "
            f"result_root={result_root}. bad targets: {'; '.join(bad)}"
        )


def read_seqmap(seqmap_path: Path) -> List[str]:
    if not seqmap_path.exists():
        raise FileNotFoundError(f"Missing seqmap file: {seqmap_path}")
    sequences: List[str] = []
    with seqmap_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row:
                continue
            name = row[0].strip()
            if not name:
                continue
            if i == 0 and name.lower() == "name":
                continue
            sequences.append(name)
    return sequences


def read_seq_length(seqinfo_path: Path) -> int:
    parser = configparser.ConfigParser()
    parser.read(seqinfo_path, encoding="utf-8")
    if "Sequence" not in parser or "seqLength" not in parser["Sequence"]:
        raise RuntimeError(f"Cannot parse seqLength from {seqinfo_path}")
    return int(parser["Sequence"]["seqLength"])


def load_frame_counts(mot_txt: Path, max_frame: int) -> np.ndarray:
    if not mot_txt.exists():
        raise FileNotFoundError(f"MOT file not found: {mot_txt}")
    counts = np.zeros((max_frame,), dtype=np.float64)
    with mot_txt.open("r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            parts = row.split(",")
            if not parts:
                continue
            frame = int(float(parts[0]))
            if 1 <= frame <= max_frame:
                counts[frame - 1] += 1.0
    return counts


def compute_count_metrics(gt_counts: np.ndarray, pred_counts: np.ndarray) -> Dict[str, float]:
    if gt_counts.shape != pred_counts.shape:
        raise ValueError("gt_counts and pred_counts must share the same shape.")
    diff = pred_counts - gt_counts
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    var = float(np.var(diff))
    if diff.size <= 1:
        drift = 0.0
    else:
        edge = max(1, min(30, diff.size // 5))
        drift = float(np.mean(diff[-edge:]) - np.mean(diff[:edge]))
    return {
        "gt_mean_count": float(np.mean(gt_counts)),
        "pred_mean_count": float(np.mean(pred_counts)),
        "CountMAE": mae,
        "CountRMSE": rmse,
        "CountVar": var,
        "CountDrift": drift,
    }


def resolve_method_dirs(args: argparse.Namespace) -> Dict[str, Path]:
    resolved: Dict[str, Path] = {}
    overrides = {
        MAIN_CHAIN_METHOD_ORDER[0]: args.pred_base,
        MAIN_CHAIN_METHOD_ORDER[1]: args.pred_gating,
        MAIN_CHAIN_METHOD_ORDER[2]: args.pred_traj,
        MAIN_CHAIN_METHOD_ORDER[3]: args.pred_adaptive,
    }
    for method, folder in METHOD_SPECS:
        path = overrides[method] if overrides[method] is not None else args.pred_root / folder
        if not path.exists():
            raise FileNotFoundError(f"Prediction directory for {method} not found: {path}")
        resolved[method] = path
    return resolved


def write_csv(rows: List[Dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "method",
        "row_type",
        "sequence",
        "used_frames",
        "gt_mean_count",
        "pred_mean_count",
        "CountMAE",
        "CountRMSE",
        "CountVar",
        "CountDrift",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_plot(rows: List[Dict[str, str]], output_path: Path, split: str) -> None:
    methods = [m for m, _ in METHOD_SPECS]
    seq_rows_by_method: Dict[str, List[Dict[str, str]]] = {m: [] for m in methods}
    mean_rows_by_method: Dict[str, Dict[str, str]] = {}
    for row in rows:
        method = row["method"]
        if row["row_type"] == "mean":
            mean_rows_by_method[method] = row
        else:
            seq_rows_by_method[method].append(row)

    mae_mean = []
    mae_std = []
    for method in methods:
        mean_row = mean_rows_by_method.get(method)
        if mean_row is None:
            mae_mean.append(np.nan)
            mae_std.append(np.nan)
            continue
        mae_mean.append(float(mean_row["CountMAE"]))
        seq_vals = [float(x["CountMAE"]) for x in seq_rows_by_method[method]]
        mae_std.append(float(np.std(seq_vals, ddof=0)) if seq_vals else 0.0)

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    ax.bar(x, mae_mean, yerr=mae_std, capsize=4, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Count MAE")
    ax.set_title(f"Count Stability on {split} (lower is better)")
    ax.grid(axis="y", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_run_config(args)
    split_dir = args.mot_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Prepared split not found: {split_dir}. "
            "Please run scripts/prepare_mft25.py first."
        )

    pred_dirs = resolve_method_dirs(args)
    seqmap = split_dir / "seqmaps" / f"{args.split}.txt"
    sequences = read_seqmap(seqmap)
    if not sequences:
        raise RuntimeError(f"No sequences found in seqmap: {seqmap}")

    rows: List[Dict[str, str]] = []
    for method, _ in METHOD_SPECS:
        method_rows: List[Dict[str, str]] = []
        pred_dir = pred_dirs[method]
        for seq in sequences:
            seq_dir = split_dir / seq
            gt_path = seq_dir / "gt" / "gt.txt"
            seqinfo = seq_dir / "seqinfo.ini"
            if not gt_path.exists():
                raise FileNotFoundError(f"Missing GT file: {gt_path}")
            if not seqinfo.exists():
                raise FileNotFoundError(f"Missing seqinfo.ini: {seqinfo}")

            seq_len = read_seq_length(seqinfo)
            used_frames = seq_len if args.max_frames <= 0 else min(seq_len, args.max_frames)
            used_frames = max(1, used_frames)

            gt_counts = load_frame_counts(gt_path, used_frames)
            pred_path = pred_dir / f"{seq}.txt"
            pred_counts = load_frame_counts(pred_path, used_frames)
            metrics = compute_count_metrics(gt_counts, pred_counts)

            row = {
                "split": args.split,
                "method": method,
                "row_type": "sequence",
                "sequence": seq,
                "used_frames": str(used_frames),
                "gt_mean_count": f"{metrics['gt_mean_count']:.3f}",
                "pred_mean_count": f"{metrics['pred_mean_count']:.3f}",
                "CountMAE": f"{metrics['CountMAE']:.3f}",
                "CountRMSE": f"{metrics['CountRMSE']:.3f}",
                "CountVar": f"{metrics['CountVar']:.3f}",
                "CountDrift": f"{metrics['CountDrift']:.3f}",
            }
            method_rows.append(row)
            rows.append(row)
            print(
                f"[count_eval] {method} {seq}: frames={used_frames}, "
                f"MAE={row['CountMAE']}, RMSE={row['CountRMSE']}, Drift={row['CountDrift']}"
            )

        mean_row = {
            "split": args.split,
            "method": method,
            "row_type": "mean",
            "sequence": "ALL",
            "used_frames": str(int(np.sum([int(r["used_frames"]) for r in method_rows]))),
            "gt_mean_count": f"{np.mean([float(r['gt_mean_count']) for r in method_rows]):.3f}",
            "pred_mean_count": f"{np.mean([float(r['pred_mean_count']) for r in method_rows]):.3f}",
            "CountMAE": f"{np.mean([float(r['CountMAE']) for r in method_rows]):.3f}",
            "CountRMSE": f"{np.mean([float(r['CountRMSE']) for r in method_rows]):.3f}",
            "CountVar": f"{np.mean([float(r['CountVar']) for r in method_rows]):.3f}",
            "CountDrift": f"{np.mean([float(r['CountDrift']) for r in method_rows]):.3f}",
        }
        rows.append(mean_row)

    write_csv(rows, args.output_csv)
    plot_path = args.paper_assets_dir / args.plot_name
    make_plot(rows, plot_path, args.split)
    print(f"Saved count metrics: {args.output_csv}")
    print(f"Saved plot:          {plot_path}")


if __name__ == "__main__":
    main()
