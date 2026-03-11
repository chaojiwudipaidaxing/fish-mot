#!/usr/bin/env python
"""Build 4-row ablation table CSV for P3 experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build table_ablation.csv from four metric csv files.")
    parser.add_argument("--base-csv", type=Path, required=True, help="Metrics CSV for Base (off).")
    parser.add_argument("--gating-csv", type=Path, required=True, help="Metrics CSV for +gating.")
    parser.add_argument("--traj-csv", type=Path, required=True, help="Metrics CSV for +traj.")
    parser.add_argument("--adaptive-csv", type=Path, required=True, help="Metrics CSV for +adaptive.")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/table_ablation.csv"),
        help="Output table CSV path.",
    )
    return parser.parse_args()


def read_one_row(csv_path: Path) -> Dict[str, str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly 1 row in {csv_path}, got {len(rows)}")
    return rows[0]


def main() -> None:
    args = parse_args()
    groups: List[Tuple[str, Dict[str, str]]] = [
        ("Base(off)", read_one_row(args.base_csv)),
        ("+gating", read_one_row(args.gating_csv)),
        ("+traj", read_one_row(args.traj_csv)),
        ("+adaptive", read_one_row(args.adaptive_csv)),
    ]

    fields = ["split", "tracker", "HOTA", "DetA", "AssA", "IDF1", "IDSW"]
    rows: List[Dict[str, str]] = []
    for name, data in groups:
        rows.append(
            {
                "split": data.get("split", "train_half"),
                "tracker": name,
                "HOTA": data["HOTA"],
                "DetA": data["DetA"],
                "AssA": data["AssA"],
                "IDF1": data["IDF1"],
                "IDSW": data["IDSW"],
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote ablation table: {args.out_csv}")


if __name__ == "__main__":
    main()
