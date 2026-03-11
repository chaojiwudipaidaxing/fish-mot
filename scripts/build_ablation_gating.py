#!/usr/bin/env python
"""Build 2-row gating ablation CSV from off/on metric CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build gating ablation table.")
    parser.add_argument("--off-csv", type=Path, required=True, help="Metrics CSV from gating=off run.")
    parser.add_argument("--on-csv", type=Path, required=True, help="Metrics CSV from gating=on run.")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/ablation_gating.csv"),
        help="Output ablation CSV path.",
    )
    return parser.parse_args()


def read_single_row(csv_path: Path) -> Dict[str, str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly 1 row in {csv_path}, got {len(rows)}")
    return rows[0]


def main() -> None:
    args = parse_args()
    off_row = read_single_row(args.off_csv)
    on_row = read_single_row(args.on_csv)

    fields = ["split", "tracker", "HOTA", "DetA", "AssA", "IDF1", "IDSW"]
    output_rows = [
        {
            "split": off_row.get("split", "train_half"),
            "tracker": "off",
            "HOTA": off_row["HOTA"],
            "DetA": off_row["DetA"],
            "AssA": off_row["AssA"],
            "IDF1": off_row["IDF1"],
            "IDSW": off_row["IDSW"],
        },
        {
            "split": on_row.get("split", "train_half"),
            "tracker": "on",
            "HOTA": on_row["HOTA"],
            "DetA": on_row["DetA"],
            "AssA": on_row["AssA"],
            "IDF1": on_row["IDF1"],
            "IDSW": on_row["IDSW"],
        },
    ]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    print(f"Wrote ablation CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
