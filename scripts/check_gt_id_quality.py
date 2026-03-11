#!/usr/bin/env python
"""Check GT identity continuity quality for prepared MOT splits."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check GT track-id continuity quality.")
    parser.add_argument("--split", default="val_half", help="Prepared split under data/mft25_mot.")
    parser.add_argument(
        "--mot-root",
        type=Path,
        default=Path("data/mft25_mot"),
        help="Root directory of prepared MOT data.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path.",
    )
    return parser.parse_args()


def read_seqmap(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing seqmap: {path}")
    seqs: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row:
                continue
            name = row[0].strip()
            if not name:
                continue
            if i == 0 and name.lower() == "name":
                continue
            seqs.append(name)
    return seqs


def evaluate_sequence(gt_path: Path) -> Dict[str, float]:
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing gt file: {gt_path}")
    counts: Counter[int] = Counter()
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            parts = row.split(",")
            if len(parts) < 2:
                continue
            track_id = int(float(parts[1]))
            counts[track_id] += 1

    if not counts:
        return {
            "num_ids": 0.0,
            "mean_len": 0.0,
            "median_len": 0.0,
            "pct_len1": 1.0,
            "max_len": 0.0,
        }

    lengths = np.asarray(list(counts.values()), dtype=float)
    return {
        "num_ids": float(len(lengths)),
        "mean_len": float(np.mean(lengths)),
        "median_len": float(np.median(lengths)),
        "pct_len1": float(np.mean(lengths <= 1.0)),
        "max_len": float(np.max(lengths)),
    }


def status_from_metrics(median_len: float, pct_len1: float) -> str:
    if median_len <= 1.0 or pct_len1 >= 0.8:
        return "FAIL"
    if median_len <= 3.0 or pct_len1 >= 0.5:
        return "WARN"
    return "PASS"


def main() -> None:
    args = parse_args()
    split_dir = args.mot_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Prepared split not found: {split_dir}. Please run scripts/prepare_mft25.py first."
        )
    seqs = read_seqmap(split_dir / "seqmaps" / f"{args.split}.txt")
    if not seqs:
        raise RuntimeError("No sequence in seqmap.")

    rows: List[Dict[str, str]] = []
    status_count = {"PASS": 0, "WARN": 0, "FAIL": 0}
    print("sequence,num_ids,mean_len,median_len,pct_len1,max_len,status")
    for seq in seqs:
        metrics = evaluate_sequence(split_dir / seq / "gt" / "gt.txt")
        status = status_from_metrics(metrics["median_len"], metrics["pct_len1"])
        status_count[status] += 1
        row = {
            "split": args.split,
            "sequence": seq,
            "num_ids": f"{metrics['num_ids']:.0f}",
            "mean_len": f"{metrics['mean_len']:.3f}",
            "median_len": f"{metrics['median_len']:.3f}",
            "pct_len1": f"{metrics['pct_len1']:.3f}",
            "max_len": f"{metrics['max_len']:.0f}",
            "status": status,
        }
        rows.append(row)
        print(
            f"{seq},{row['num_ids']},{row['mean_len']},{row['median_len']},"
            f"{row['pct_len1']},{row['max_len']},{status}"
        )

    print(
        f"[summary] split={args.split} PASS={status_count['PASS']} "
        f"WARN={status_count['WARN']} FAIL={status_count['FAIL']}"
    )

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = ["split", "sequence", "num_ids", "mean_len", "median_len", "pct_len1", "max_len", "status"]
        with args.output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"Saved id quality CSV: {args.output_csv}")

    if status_count["FAIL"] > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
