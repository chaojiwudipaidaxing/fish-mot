#!/usr/bin/env python
"""Build val-half main table (4 methods) from per-method mean/per-seq CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

from method_labels import MAIN_CHAIN_METHOD_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build results/main_table_val.csv and per_seq_main_val.csv.")
    parser.add_argument("--base-mean", type=Path, required=True, help="Mean CSV for Base.")
    parser.add_argument("--gating-mean", type=Path, required=True, help="Mean CSV for Base+gating.")
    parser.add_argument("--traj-mean", type=Path, required=True, help="Mean CSV for Base+gating+traj.")
    parser.add_argument("--adaptive-mean", type=Path, required=True, help="Mean CSV for Base+gating+traj+adaptive.")
    parser.add_argument("--base-per-seq", type=Path, required=True, help="Per-seq CSV for Base.")
    parser.add_argument("--gating-per-seq", type=Path, required=True, help="Per-seq CSV for Base+gating.")
    parser.add_argument("--traj-per-seq", type=Path, required=True, help="Per-seq CSV for Base+gating+traj.")
    parser.add_argument("--adaptive-per-seq", type=Path, required=True, help="Per-seq CSV for Base+gating+traj+adaptive.")
    parser.add_argument(
        "--out-main",
        type=Path,
        default=Path("results/main_table_val.csv"),
        help="Main table CSV output path.",
    )
    parser.add_argument(
        "--out-per-seq",
        type=Path,
        default=Path("results/per_seq_main_val.csv"),
        help="Combined per-sequence CSV output path.",
    )
    return parser.parse_args()


def read_single_row(csv_path: Path) -> Dict[str, str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"Expected 1 row in {csv_path}, got {len(rows)}")
    return rows[0]


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    groups: List[Tuple[str, Path, Path]] = [
        (MAIN_CHAIN_METHOD_ORDER[0], args.base_mean, args.base_per_seq),
        (MAIN_CHAIN_METHOD_ORDER[1], args.gating_mean, args.gating_per_seq),
        (MAIN_CHAIN_METHOD_ORDER[2], args.traj_mean, args.traj_per_seq),
        (MAIN_CHAIN_METHOD_ORDER[3], args.adaptive_mean, args.adaptive_per_seq),
    ]

    main_rows: List[Dict[str, str]] = []
    per_seq_rows: List[Dict[str, str]] = []

    for method, mean_path, per_seq_path in groups:
        mean_row = read_single_row(mean_path)
        main_rows.append(
            {
                "split": mean_row.get("split", "val_half"),
                "method": method,
                "HOTA": mean_row["HOTA"],
                "DetA": mean_row["DetA"],
                "AssA": mean_row["AssA"],
                "IDF1": mean_row["IDF1"],
                "IDSW": mean_row["IDSW"],
            }
        )

        seq_rows = read_rows(per_seq_path)
        for r in seq_rows:
            per_seq_rows.append(
                {
                    "split": r.get("split", "val_half"),
                    "method": method,
                    "sequence": r["sequence"],
                    "used_frames": r.get("used_frames", ""),
                    "gt_rows": r.get("gt_rows", ""),
                    "pred_rows": r.get("pred_rows", ""),
                    "HOTA": r["HOTA"],
                    "DetA": r["DetA"],
                    "AssA": r["AssA"],
                    "IDF1": r["IDF1"],
                    "IDSW": r["IDSW"],
                }
            )

    args.out_main.parent.mkdir(parents=True, exist_ok=True)
    with args.out_main.open("w", encoding="utf-8", newline="") as f:
        fields = ["split", "method", "HOTA", "DetA", "AssA", "IDF1", "IDSW"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in main_rows:
            writer.writerow(row)

    args.out_per_seq.parent.mkdir(parents=True, exist_ok=True)
    with args.out_per_seq.open("w", encoding="utf-8", newline="") as f:
        fields = ["split", "method", "sequence", "used_frames", "gt_rows", "pred_rows", "HOTA", "DetA", "AssA", "IDF1", "IDSW"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in per_seq_rows:
            writer.writerow(row)

    print(f"Saved main table: {args.out_main}")
    print(f"Saved per-seq table: {args.out_per_seq}")


if __name__ == "__main__":
    main()
