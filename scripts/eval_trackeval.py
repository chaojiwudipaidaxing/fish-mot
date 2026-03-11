#!/usr/bin/env python
"""Evaluate prepared MFT25 MOT data with TrackEval."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


# TrackEval still references np.float/np.int in multiple places.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TrackEval on prepared MFT25 MOT data.")
    parser.add_argument(
        "--split",
        default="train_half",
        help="Prepared split to evaluate (under data/mft25_mot/<split>).",
    )
    parser.add_argument(
        "--mot-root",
        type=Path,
        default=Path("data/mft25_mot"),
        help="Root directory of prepared MOT-style data.",
    )
    parser.add_argument(
        "--trackeval-root",
        type=Path,
        default=Path("third_party/TrackEval"),
        help="Path to TrackEval repository root.",
    )
    parser.add_argument(
        "--tracker-name",
        default="gt_copy",
        help="Tracker name shown in TrackEval output.",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing prediction txt files (<SEQ>.txt). "
            "If omitted, evaluate GT-copy predictions."
        ),
    )
    parser.add_argument(
        "--trackers-root",
        type=Path,
        default=Path("results/trackers"),
        help="Workspace for TrackEval tracker input/output files.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("results/metrics.csv"),
        help="CSV output path for aggregated metrics.",
    )
    return parser.parse_args()


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


def build_gt_copy_predictions(split_dir: Path, sequences: List[str], tracker_data_dir: Path) -> None:
    if tracker_data_dir.exists():
        shutil.rmtree(tracker_data_dir)
    tracker_data_dir.mkdir(parents=True, exist_ok=True)

    for seq in sequences:
        gt_file = split_dir / seq / "gt" / "gt.txt"
        if not gt_file.exists():
            raise FileNotFoundError(f"Missing gt file for sequence {seq}: {gt_file}")
        pred_file = tracker_data_dir / f"{seq}.txt"
        shutil.copy2(gt_file, pred_file)


def copy_predictions_from_dir(pred_dir: Path, sequences: List[str], tracker_data_dir: Path) -> None:
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if tracker_data_dir.exists():
        shutil.rmtree(tracker_data_dir)
    tracker_data_dir.mkdir(parents=True, exist_ok=True)

    for seq in sequences:
        src = pred_dir / f"{seq}.txt"
        if not src.exists():
            raise FileNotFoundError(f"Missing prediction file: {src}")
        dst = tracker_data_dir / f"{seq}.txt"
        shutil.copy2(src, dst)


def collect_metrics(combined: Dict) -> Dict[str, float]:
    hota = float(np.mean(combined["HOTA"]["HOTA"]) * 100.0)
    deta = float(np.mean(combined["HOTA"]["DetA"]) * 100.0)
    assa = float(np.mean(combined["HOTA"]["AssA"]) * 100.0)
    idf1 = float(combined["Identity"]["IDF1"] * 100.0)
    idsw = int(combined["CLEAR"]["IDSW"])
    return {
        "HOTA": hota,
        "DetA": deta,
        "AssA": assa,
        "IDF1": idf1,
        "IDSW": idsw,
    }


def run_trackeval(
    trackeval_root: Path,
    split_dir: Path,
    split: str,
    trackers_root: Path,
    tracker_name: str,
    seqmap_path: Path,
):
    sys.path.insert(0, str(trackeval_root.resolve()))
    import trackeval  # pylint: disable=import-error,import-outside-toplevel

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update(
        {
            "USE_PARALLEL": False,
            "BREAK_ON_ERROR": True,
            "PRINT_RESULTS": False,
            "PRINT_ONLY_COMBINED": True,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "DISPLAY_LESS_PROGRESS": True,
            "OUTPUT_SUMMARY": True,
            "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False,
        }
    )

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": str(split_dir),
            "TRACKERS_FOLDER": str(trackers_root),
            "OUTPUT_FOLDER": str(trackers_root / "_eval_outputs"),
            "TRACKERS_TO_EVAL": [tracker_name],
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": "MFT25",
            "SPLIT_TO_EVAL": split,
            "INPUT_AS_ZIP": False,
            "PRINT_CONFIG": False,
            "DO_PREPROC": False,
            "TRACKER_SUB_FOLDER": "data",
            "OUTPUT_SUB_FOLDER": "",
            "SEQMAP_FILE": str(seqmap_path),
            "SKIP_SPLIT_FOL": True,
        }
    )

    metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5, "PRINT_CONFIG": False}
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics_list = [
        trackeval.metrics.HOTA(metrics_config),
        trackeval.metrics.CLEAR(metrics_config),
        trackeval.metrics.Identity(metrics_config),
    ]
    output_res, output_msg = evaluator.evaluate([dataset], metrics_list)
    dataset_name = dataset.get_name()
    message = output_msg[dataset_name][tracker_name]
    if message != "Success":
        raise RuntimeError(f"TrackEval failed: {message}")
    return output_res[dataset_name][tracker_name]


def write_metrics_csv(
    csv_path: Path,
    split: str,
    tracker_name: str,
    metrics: Dict[str, float],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "tracker", "HOTA", "DetA", "AssA", "IDF1", "IDSW"]
    row = {
        "split": split,
        "tracker": tracker_name,
        "HOTA": f"{metrics['HOTA']:.3f}",
        "DetA": f"{metrics['DetA']:.3f}",
        "AssA": f"{metrics['AssA']:.3f}",
        "IDF1": f"{metrics['IDF1']:.3f}",
        "IDSW": str(metrics["IDSW"]),
    }
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    split_dir = args.mot_root / args.split
    seqmap_path = split_dir / "seqmaps" / f"{args.split}.txt"

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Prepared split not found: {split_dir}. "
            "Please run scripts/prepare_mft25.py first."
        )

    sequences = read_seqmap(seqmap_path)
    if not sequences:
        raise RuntimeError(f"No sequences found in seqmap: {seqmap_path}")

    trackers_root = args.trackers_root / args.split
    tracker_data_dir = trackers_root / args.tracker_name / "data"
    if args.pred_dir is None:
        build_gt_copy_predictions(split_dir=split_dir, sequences=sequences, tracker_data_dir=tracker_data_dir)
        source_desc = "gt_copy"
    else:
        copy_predictions_from_dir(args.pred_dir, sequences=sequences, tracker_data_dir=tracker_data_dir)
        source_desc = str(args.pred_dir)

    eval_result = run_trackeval(
        trackeval_root=args.trackeval_root,
        split_dir=split_dir,
        split=args.split,
        trackers_root=trackers_root,
        tracker_name=args.tracker_name,
        seqmap_path=seqmap_path,
    )

    combined = eval_result["COMBINED_SEQ"]["pedestrian"]
    metrics = collect_metrics(combined)
    write_metrics_csv(args.results_csv, args.split, args.tracker_name, metrics)

    print(
        f"TrackEval completed on split={args.split}, tracker={args.tracker_name}. "
        f"HOTA={metrics['HOTA']:.3f}, IDF1={metrics['IDF1']:.3f}, IDSW={metrics['IDSW']} "
        f"(source={source_desc})"
    )
    print(f"Saved metrics CSV: {args.results_csv}")


if __name__ == "__main__":
    main()
