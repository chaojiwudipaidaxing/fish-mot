#!/usr/bin/env python
"""Run smoke baseline comparisons (ByteTrack/OC-SORT/BoT-SORT-style configs) on val_half."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


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
    parser = argparse.ArgumentParser(description="Run baseline comparison smoke on val_half.")
    parser.add_argument("--split", default="val_half", help="Prepared split under data/mft25_mot.")
    parser.add_argument(
        "--mot-root",
        type=Path,
        default=Path("data/mft25_mot"),
        help="Root directory of prepared MOT data.",
    )
    parser.add_argument(
        "--methods",
        default="bytetrack,ocsort",
        help="Comma separated methods from {bytetrack,ocsort,botsort}; need at least 2.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=20,
        help="Smoke frame cap. Defaults to 20.",
    )
    parser.add_argument(
        "--allow-long-run",
        action="store_true",
        help="Allow max_frames > 20 (disabled by default to avoid long runs).",
    )
    parser.add_argument("--max-gt-ids", type=int, default=200, help="GT ID cap for TrackEval per-seq.")
    parser.add_argument("--drop-rate", type=float, default=0.2, help="Detection drop-rate for hard setup.")
    parser.add_argument("--jitter", type=float, default=0.02, help="Detection jitter for hard setup.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for degradation.")
    parser.add_argument(
        "--traj-encoder",
        type=Path,
        default=Path("runs/traj_encoder/traj_encoder.pt"),
        help="Trajectory encoder checkpoint when botsort config is selected.",
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
        default=Path("results/baselines_table_val.csv"),
        help="Output summary table path.",
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


def main() -> None:
    args = parse_args()
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    if len(methods) < 2:
        raise ValueError("Need at least two methods in --methods.")
    unknown = [m for m in methods if m not in METHOD_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Valid: {sorted(METHOD_CONFIGS.keys())}")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be > 0.")
    if args.max_frames > 20 and not args.allow_long_run:
        raise ValueError("Smoke guard: max-frames > 20 blocked. Use --allow-long-run to override.")
    if not args.mot_root.exists():
        raise FileNotFoundError(f"MOT root not found: {args.mot_root}")

    split_dir = args.mot_root / args.split
    if not split_dir.exists():
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
    else:
        print(f"[info] Using existing split: {split_dir}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    table_rows: List[Dict[str, str]] = []

    for method_key in methods:
        cfg = METHOD_CONFIGS[method_key]
        display = cfg["display"]
        pred_dir = args.output_root / method_key
        mean_csv = args.output_root / f"{method_key}_mean.csv"
        per_seq_csv = args.output_root / f"{method_key}_per_seq.csv"

        cmd = [
            sys.executable,
            "scripts/run_baseline_sort.py",
            "--split",
            args.split,
            "--mot-root",
            str(args.mot_root),
            "--max-frames",
            str(args.max_frames),
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
            str(args.seed),
            "--out-dir",
            str(pred_dir),
            "--clean-out",
        ]
        if cfg["traj"] == "on":
            if not args.traj_encoder.exists():
                raise FileNotFoundError(f"Trajectory encoder not found: {args.traj_encoder}")
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
                f"smoke_{method_key}",
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
                "method": display,
                "method_key": method_key,
                "HOTA": f"{float(mean_row['HOTA']):.3f}",
                "DetA": f"{float(mean_row['DetA']):.3f}",
                "AssA": f"{float(mean_row['AssA']):.3f}",
                "IDF1": f"{float(mean_row['IDF1']):.3f}",
                "IDSW": f"{float(mean_row['IDSW']):.3f}",
                "pred_dir": str(pred_dir).replace("\\", "/"),
            }
        )

    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "method", "method_key", "HOTA", "DetA", "AssA", "IDF1", "IDSW", "pred_dir"]
    with args.table_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in table_rows:
            writer.writerow(row)

    print(f"Saved baseline table: {args.table_out}")
    print(f"Prediction roots under: {args.output_root}")


if __name__ == "__main__":
    main()

