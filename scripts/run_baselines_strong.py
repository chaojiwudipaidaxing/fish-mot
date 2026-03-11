#!/usr/bin/env python
"""Run strong baseline trackers (ByteTrack/OC-SORT/BoT-SORT) on MFT25 val split."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


METHOD_CONFIGS: Dict[str, Dict[str, str]] = {
    "bytetrack": {
        "display": "ByteTrack",
        "gating": "off",
        "traj": "off",
        "adaptive_gamma": "off",
        "alpha": "1.0",
        "beta": "0.0",
        "gamma": "0.0",
        "iou_thresh": "0.3",
        "min_hits": "1",
        "max_age": "30",
    },
    "ocsort": {
        "display": "OC-SORT",
        "gating": "on",
        "traj": "off",
        "adaptive_gamma": "off",
        "alpha": "1.0",
        "beta": "0.02",
        "gamma": "0.0",
        "iou_thresh": "0.3",
        "min_hits": "1",
        "max_age": "30",
    },
    "botsort": {
        "display": "BoT-SORT",
        "gating": "on",
        "traj": "on",
        "adaptive_gamma": "on",
        "alpha": "1.0",
        "beta": "0.02",
        "gamma": "0.5",
        "iou_thresh": "0.3",
        "min_hits": "1",
        "max_age": "30",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strong baselines on val_half and aggregate results.")
    parser.add_argument("--split", default="val_half", help="Prepared split under data/mft25_mot.")
    parser.add_argument(
        "--mot-root",
        type=Path,
        default=Path("data/mft25_mot"),
        help="Root directory of prepared MOT data.",
    )
    parser.add_argument(
        "--methods",
        default="bytetrack,ocsort,botsort",
        help="Comma separated methods from {bytetrack,ocsort,botsort}.",
    )
    parser.add_argument("--max-frames", type=int, default=1000, help="Frame cap for val run.")
    parser.add_argument("--max-gt-ids", type=int, default=0, help="Passed to per-seq TrackEval (0 disables cap).")
    parser.add_argument(
        "--det-source",
        choices=["auto", "det", "gt"],
        default="auto",
        help="Use same detection source for all methods.",
    )
    parser.add_argument("--drop-rate", type=float, default=0.0, help="Optional detection drop ratio.")
    parser.add_argument("--jitter", type=float, default=0.0, help="Optional detection jitter ratio.")
    parser.add_argument("--seed", type=int, default=0, help="Detection degradation seed.")
    parser.add_argument(
        "--traj-encoder",
        type=Path,
        default=Path("runs/traj_encoder/traj_encoder.pt"),
        help="Trajectory encoder for BoT-SORT style run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/baselines"),
        help="Output root for predictions and per-method metrics.",
    )
    parser.add_argument(
        "--table-out",
        type=Path,
        default=Path("results/main_table_val_baselines.csv"),
        help="Aggregated baseline table output CSV.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("results/paper_assets_val/main_table_val_baselines.png"),
        help="Comparison plot output path.",
    )
    parser.add_argument(
        "--prepare",
        choices=["yes", "no"],
        default="yes",
        help="Run prepare_mft25 before tracking.",
    )
    return parser.parse_args()


def run_cmd(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def read_single_row(csv_path: Path) -> Dict[str, str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"Expected 1 row in {csv_path}, got {len(rows)}")
    return rows[0]


def write_table(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "method", "method_key", "HOTA", "DetA", "AssA", "IDF1", "IDSW", "pred_dir", "mean_csv", "per_seq_csv"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_table(rows: List[Dict[str, str]], plot_path: Path) -> None:
    methods = [row["method"] for row in rows]
    hota = [float(row["HOTA"]) for row in rows]
    assa = [float(row["AssA"]) for row in rows]
    idf1 = [float(row["IDF1"]) for row in rows]

    x = np.arange(len(methods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    ax.bar(x - width, hota, width=width, label="HOTA")
    ax.bar(x, assa, width=width, label="AssA")
    ax.bar(x + width, idf1, width=width, label="IDF1")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Score")
    ax.set_title("ByteTrack / OC-SORT / BoT-SORT on val_half")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    if not methods:
        raise RuntimeError("No method selected.")
    unknown = [m for m in methods if m not in METHOD_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}; valid={sorted(METHOD_CONFIGS)}")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be > 0.")

    if args.prepare == "yes":
        run_cmd(
            [
                sys.executable,
                "scripts/prepare_mft25.py",
                "--splits",
                args.split,
                "--max-frames",
                str(args.max_frames),
                "--clean-split",
            ]
        )

    split_dir = args.mot_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Prepared split not found: {split_dir}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    table_rows: List[Dict[str, str]] = []

    for method_idx, method_key in enumerate(methods):
        cfg = METHOD_CONFIGS[method_key]
        pred_dir = args.output_root / method_key
        per_seq_csv = args.output_root / f"{method_key}_per_seq.csv"
        mean_csv = args.output_root / f"{method_key}_mean.csv"

        cmd = [
            sys.executable,
            "scripts/run_baseline_sort.py",
            "--split",
            args.split,
            "--mot-root",
            str(args.mot_root),
            "--max-frames",
            str(args.max_frames),
            "--det-source",
            args.det_source,
            "--gating",
            cfg["gating"],
            "--traj",
            cfg["traj"],
            "--adaptive-gamma",
            cfg["adaptive_gamma"],
            "--alpha",
            cfg["alpha"],
            "--beta",
            cfg["beta"],
            "--gamma",
            cfg["gamma"],
            "--iou-thresh",
            cfg["iou_thresh"],
            "--min-hits",
            cfg["min_hits"],
            "--max-age",
            cfg["max_age"],
            "--drop-rate",
            str(args.drop_rate),
            "--jitter",
            str(args.jitter),
            "--degrade-seed",
            str(args.seed + method_idx * 1000),
            "--out-dir",
            str(pred_dir),
            "--clean-out",
        ]
        if cfg["traj"] == "on":
            cmd.extend(["--traj-encoder", str(args.traj_encoder)])
        run_cmd(cmd)

        run_cmd(
            [
                sys.executable,
                "scripts/eval_trackeval_per_seq.py",
                "--split",
                args.split,
                "--mot-root",
                str(args.mot_root),
                "--pred-dir",
                str(pred_dir),
                "--tracker-name",
                f"strong_{method_key}",
                "--max-frames",
                str(args.max_frames),
                "--max-gt-ids",
                str(args.max_gt_ids),
                "--results-per-seq",
                str(per_seq_csv),
                "--results-mean",
                str(mean_csv),
            ]
        )

        mean_row = read_single_row(mean_csv)
        table_rows.append(
            {
                "split": mean_row.get("split", args.split),
                "method": cfg["display"],
                "method_key": method_key,
                "HOTA": f"{float(mean_row['HOTA']):.3f}",
                "DetA": f"{float(mean_row['DetA']):.3f}",
                "AssA": f"{float(mean_row['AssA']):.3f}",
                "IDF1": f"{float(mean_row['IDF1']):.3f}",
                "IDSW": f"{float(mean_row['IDSW']):.3f}",
                "pred_dir": str(pred_dir).replace("\\", "/"),
                "mean_csv": str(mean_csv).replace("\\", "/"),
                "per_seq_csv": str(per_seq_csv).replace("\\", "/"),
            }
        )
        print(
            f"[done] {cfg['display']}: HOTA={table_rows[-1]['HOTA']} "
            f"IDF1={table_rows[-1]['IDF1']} IDSW={table_rows[-1]['IDSW']}"
        )

    write_table(args.table_out, table_rows)
    plot_table(table_rows, args.plot_path)
    print(f"Saved table: {args.table_out}")
    print(f"Saved plot:  {args.plot_path}")


if __name__ == "__main__":
    main()
