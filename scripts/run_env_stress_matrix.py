#!/usr/bin/env python
"""Environment-aware stress-test matrix scaffold.

Generates a reproducible stress-test plan table (no metrics fabricated).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_ROWS = [
    ("motion_blur", "low", "TODO: set (l,theta)"),
    ("motion_blur", "mid", "TODO: set (l,theta)"),
    ("motion_blur", "high", "TODO: set (l,theta)"),
    ("low_light_shift", "low", "TODO: set (alpha,beta)"),
    ("low_light_shift", "mid", "TODO: set (alpha,beta)"),
    ("low_light_shift", "high", "TODO: set (alpha,beta)"),
    ("turbidity_haze", "low", "TODO: set (t,A)"),
    ("turbidity_haze", "mid", "TODO: set (t,A)"),
    ("turbidity_haze", "high", "TODO: set (t,A)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write environment stress-test plan CSV.")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/stress/stress_matrix_plan.csv"),
        help="Output plan CSV path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["degradation_type", "strength_level", "parameter_setting"])
        for row in DEFAULT_ROWS:
            writer.writerow(row)
    print(f"wrote stress matrix plan: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

