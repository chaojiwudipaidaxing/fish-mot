#!/usr/bin/env python
"""Export BrackishMOT bucket-risk migration CSV from stratified metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


BUCKET_MAP = {
    "occlusion": "occlusion",
    "density": "density",
    "turn": "turning",
    "low_conf": "low-confidence",
}
BUCKET_ORDER = ["occlusion", "density", "turning", "low-confidence"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export bucket shift CSV for BrackishMOT stress diagnostics.")
    parser.add_argument(
        "--clear-csv",
        type=Path,
        default=Path("results/brackishmot/stress/stratified_clear.csv"),
    )
    parser.add_argument(
        "--high-csv",
        type=Path,
        default=Path("results/brackishmot/stress/stratified_turbid_high.csv"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/brackishmot_bucket_shift.csv"),
    )
    return parser.parse_args()


def _load_high_bucket(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing stratified CSV: {path}")
    df = pd.read_csv(path)
    need_cols = {"method", "bucket", "bucket_type", "IDSW", "CountMAE", "num_frames_bucket"}
    miss = sorted(need_cols - set(df.columns))
    if miss:
        raise ValueError(f"{path} missing columns: {miss}")
    df = df[df["bucket"] == "high"].copy()
    df["bucket_name"] = df["bucket_type"].map(BUCKET_MAP)
    df = df[df["bucket_name"].notna()].copy()
    return df


def main() -> int:
    args = parse_args()
    clear_df = _load_high_bucket(args.clear_csv)
    high_df = _load_high_bucket(args.high_csv)

    methods = sorted(set(clear_df["method"]).union(set(high_df["method"])))
    rows: List[Dict[str, object]] = []
    for method in methods:
        clear_m = clear_df[clear_df["method"] == method]
        high_m = high_df[high_df["method"] == method]
        clear_map = {str(r["bucket_name"]): r for _, r in clear_m.iterrows()}
        high_map = {str(r["bucket_name"]): r for _, r in high_m.iterrows()}
        for bucket in BUCKET_ORDER:
            c = clear_map.get(bucket)
            h = high_map.get(bucket)

            clear_idsw = float(c["IDSW"]) if c is not None else np.nan
            clear_mae = float(c["CountMAE"]) if c is not None else np.nan
            clear_n = int(c["num_frames_bucket"]) if c is not None else 0

            high_idsw = float(h["IDSW"]) if h is not None else np.nan
            high_mae = float(h["CountMAE"]) if h is not None else np.nan
            high_n = int(h["num_frames_bucket"]) if h is not None else 0

            note = ""
            if c is None or h is None or clear_n <= 0 or high_n <= 0:
                note = "insufficient samples"

            rows.append(
                {
                    "method": method,
                    "group": "clear",
                    "bucket_name": bucket,
                    "idsw_bucket": clear_idsw,
                    "countmae_bucket": clear_mae,
                    "delta_idsw": 0.0,
                    "delta_countmae": 0.0,
                    "num_frames_bucket": clear_n,
                    "note": note if c is None else "",
                }
            )
            rows.append(
                {
                    "method": method,
                    "group": "turbid_high",
                    "bucket_name": bucket,
                    "idsw_bucket": high_idsw,
                    "countmae_bucket": high_mae,
                    "delta_idsw": high_idsw - clear_idsw if np.isfinite(high_idsw) and np.isfinite(clear_idsw) else np.nan,
                    "delta_countmae": high_mae - clear_mae if np.isfinite(high_mae) and np.isfinite(clear_mae) else np.nan,
                    "num_frames_bucket": high_n,
                    "note": note,
                }
            )

    out_df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"wrote bucket shift CSV: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
