#!/usr/bin/env python
"""Evaluate one split sequence-by-sequence with TrackEval and aggregate mean metrics."""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# TrackEval still references np.float/np.int in multiple places.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TrackEval per sequence and save per-seq/mean CSV.")
    parser.add_argument("--split", default="train_half", help="Prepared split name under data/mft25_mot.")
    parser.add_argument(
        "--mot-root",
        type=Path,
        default=Path("data/mft25_mot"),
        help="Root directory of prepared MOT data.",
    )
    parser.add_argument(
        "--trackeval-root",
        type=Path,
        default=Path("third_party/TrackEval"),
        help="Path to TrackEval repository root.",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=None,
        help="Optional prediction directory (<SEQ>.txt). If omitted, use GT-copy.",
    )
    parser.add_argument(
        "--tracker-name",
        default="per_seq_eval",
        help="Tracker name used inside TrackEval outputs.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1000,
        help="Evaluate up to this many frames per sequence (0 means all).",
    )
    parser.add_argument(
        "--max-gt-ids",
        type=int,
        default=2000,
        help=(
            "Safety cap for unique GT IDs per sequence to avoid HOTA OOM. "
            "0 disables this cap."
        ),
    )
    parser.add_argument(
        "--tmp-root",
        type=Path,
        default=Path("results/_per_seq_eval_tmp"),
        help="Temporary folder for per-sequence TrackEval inputs.",
    )
    parser.add_argument(
        "--results-per-seq",
        type=Path,
        default=Path("results/per_seq_metrics.csv"),
        help="Per-sequence metrics CSV output path.",
    )
    parser.add_argument(
        "--results-mean",
        type=Path,
        default=Path("results/metrics_mean.csv"),
        help="Mean metrics CSV output path.",
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=None,
        help="Optional run_config.json. When provided, mot_root is resolved from it.",
    )
    return parser.parse_args()


def apply_run_config(args: argparse.Namespace) -> None:
    if args.run_config is None:
        return
    if not args.run_config.exists():
        raise FileNotFoundError(f"Run config not found: {args.run_config}")
    cfg = json.loads(args.run_config.read_text(encoding="utf-8"))
    mot_root_cfg = cfg.get("mot_root")
    if mot_root_cfg:
        args.mot_root = Path(str(mot_root_cfg))
    if args.split == "train_half" and cfg.get("split"):
        args.split = str(cfg.get("split"))
    if args.max_frames == 1000 and cfg.get("max_frames") is not None:
        args.max_frames = int(cfg.get("max_frames"))


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


def choose_frame_cap(gt_path: Path, requested_max_frames: int, max_gt_ids: int) -> int:
    frame_cap = requested_max_frames if requested_max_frames > 0 else 10**9
    if max_gt_ids <= 0:
        return frame_cap
    seen_ids = set()
    chosen_frame = frame_cap
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            frame = int(float(parts[0]))
            if frame > frame_cap:
                break
            track_id = int(float(parts[1]))
            seen_ids.add(track_id)
            if len(seen_ids) > max_gt_ids:
                chosen_frame = max(1, frame - 1)
                break
    return chosen_frame


def filter_mot_rows(input_path: Path, output_path: Path, max_frame: int) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Input MOT file not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for line in src:
            row = line.strip()
            if not row:
                continue
            parts = row.split(",")
            if not parts:
                continue
            frame = int(float(parts[0]))
            if frame <= max_frame:
                dst.write(row + "\n")
                kept += 1
    return kept


def analyze_gt_identity_quality(gt_path: Path, max_frame: int) -> Dict[str, float]:
    id_counts: Dict[int, int] = {}
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            parts = row.split(",")
            if len(parts) < 2:
                continue
            frame = int(float(parts[0]))
            if frame > max_frame:
                continue
            track_id = int(float(parts[1]))
            id_counts[track_id] = id_counts.get(track_id, 0) + 1

    if not id_counts:
        return {
            "unique_ids": 0.0,
            "repeated_ids": 0.0,
            "avg_track_len": 0.0,
            "single_frame_ratio": 1.0,
        }

    counts = list(id_counts.values())
    unique_ids = float(len(counts))
    repeated_ids = float(sum(1 for c in counts if c > 1))
    avg_track_len = float(np.mean(counts))
    single_frame_ratio = float(sum(1 for c in counts if c <= 1) / max(1, len(counts)))
    return {
        "unique_ids": unique_ids,
        "repeated_ids": repeated_ids,
        "avg_track_len": avg_track_len,
        "single_frame_ratio": single_frame_ratio,
    }


def write_seqinfo_with_cap(src_seqinfo: Path, dst_seqinfo: Path, max_frame: int) -> None:
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(src_seqinfo, encoding="utf-8")
    if "Sequence" not in parser:
        raise RuntimeError(f"Invalid seqinfo.ini: {src_seqinfo}")
    parser["Sequence"]["seqLength"] = str(max_frame)
    dst_seqinfo.parent.mkdir(parents=True, exist_ok=True)
    with dst_seqinfo.open("w", encoding="utf-8", newline="\n") as f:
        parser.write(f, space_around_delimiters=False)


def run_trackeval_single_seq(
    trackeval_root: Path,
    seq: str,
    gt_root: Path,
    trackers_root: Path,
    seqmap_file: Path,
    tracker_name: str,
) -> Dict[str, float]:
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
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False,
        }
    )

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": str(gt_root),
            "TRACKERS_FOLDER": str(trackers_root),
            "OUTPUT_FOLDER": str(trackers_root / "_eval_outputs"),
            "TRACKERS_TO_EVAL": [tracker_name],
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": "MFT25",
            "SPLIT_TO_EVAL": "single",
            "INPUT_AS_ZIP": False,
            "PRINT_CONFIG": False,
            "DO_PREPROC": False,
            "TRACKER_SUB_FOLDER": "data",
            "OUTPUT_SUB_FOLDER": "",
            "SEQMAP_FILE": str(seqmap_file),
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
    status = output_msg[dataset_name][tracker_name]
    if status != "Success":
        raise RuntimeError(f"TrackEval failed on {seq}: {status}")

    combined = output_res[dataset_name][tracker_name]["COMBINED_SEQ"]["pedestrian"]
    return {
        "HOTA": float(np.mean(combined["HOTA"]["HOTA"]) * 100.0),
        "DetA": float(np.mean(combined["HOTA"]["DetA"]) * 100.0),
        "AssA": float(np.mean(combined["HOTA"]["AssA"]) * 100.0),
        "IDF1": float(combined["Identity"]["IDF1"] * 100.0),
        "IDSW": float(combined["CLEAR"]["IDSW"]),
    }


def write_per_seq_csv(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "sequence",
        "tracker",
        "used_frames",
        "gt_rows",
        "pred_rows",
        "gt_unique_ids",
        "gt_repeated_ids",
        "gt_avg_track_len",
        "gt_single_frame_ratio",
        "id_eval_valid",
        "id_quality_note",
        "HOTA",
        "DetA",
        "AssA",
        "IDF1",
        "IDSW",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_mean_csv(rows: List[Dict[str, str]], split: str, tracker: str, path: Path) -> None:
    valid = []
    for row in rows:
        try:
            valid.append(
                {
                    "HOTA": float(row["HOTA"]),
                    "DetA": float(row["DetA"]),
                    "AssA": float(row["AssA"]),
                    "IDF1": float(row["IDF1"]),
                    "IDSW": float(row["IDSW"]),
                }
            )
        except ValueError:
            continue

    if not valid:
        raise RuntimeError("No valid per-sequence rows to aggregate mean metrics.")

    mean_vals = {
        "HOTA": float(np.mean([x["HOTA"] for x in valid])),
        "DetA": float(np.mean([x["DetA"] for x in valid])),
        "AssA": float(np.mean([x["AssA"] for x in valid])),
        "IDF1": float(np.mean([x["IDF1"] for x in valid])),
        "IDSW": float(np.mean([x["IDSW"] for x in valid])),
    }
    valid_flag_vals = []
    for row in rows:
        try:
            valid_flag_vals.append(float(row.get("id_eval_valid", "0")))
        except ValueError:
            continue
    id_eval_valid_ratio = float(np.mean(valid_flag_vals)) if valid_flag_vals else 0.0

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "tracker", "num_sequences", "HOTA", "DetA", "AssA", "IDF1", "IDSW", "id_eval_valid_ratio"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "split": split,
                "tracker": tracker,
                "num_sequences": str(len(valid)),
                "HOTA": f"{mean_vals['HOTA']:.3f}",
                "DetA": f"{mean_vals['DetA']:.3f}",
                "AssA": f"{mean_vals['AssA']:.3f}",
                "IDF1": f"{mean_vals['IDF1']:.3f}",
                "IDSW": f"{mean_vals['IDSW']:.3f}",
                "id_eval_valid_ratio": f"{id_eval_valid_ratio:.3f}",
            }
        )


def main() -> None:
    args = parse_args()
    apply_run_config(args)
    split_dir = args.mot_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Prepared split not found: {split_dir}. "
            "Please run scripts/prepare_mft25.py first."
        )

    seqmap_path = split_dir / "seqmaps" / f"{args.split}.txt"
    sequences = read_seqmap(seqmap_path)
    if not sequences:
        raise RuntimeError(f"No sequences found in seqmap: {seqmap_path}")

    shutil.rmtree(args.tmp_root, ignore_errors=True)
    args.tmp_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for seq in sequences:
        seq_dir = split_dir / seq
        gt_src = seq_dir / "gt" / "gt.txt"
        seqinfo_src = seq_dir / "seqinfo.ini"
        if not gt_src.exists():
            raise FileNotFoundError(f"Missing GT file: {gt_src}")
        if not seqinfo_src.exists():
            raise FileNotFoundError(f"Missing seqinfo.ini: {seqinfo_src}")

        seq_len = read_seq_length(seqinfo_src)
        requested_cap = seq_len if args.max_frames <= 0 else min(seq_len, args.max_frames)
        id_cap = choose_frame_cap(gt_src, requested_cap, args.max_gt_ids)
        used_frames = min(requested_cap, id_cap)
        used_frames = max(1, used_frames)

        seq_tmp_root = args.tmp_root / seq
        gt_tmp_root = seq_tmp_root / "gt_root"
        trackers_tmp_root = seq_tmp_root / "trackers_root"
        seqmap_tmp = gt_tmp_root / "seqmaps" / "single.txt"

        gt_dst = gt_tmp_root / seq / "gt" / "gt.txt"
        pred_dst = trackers_tmp_root / args.tracker_name / "data" / f"{seq}.txt"
        seqinfo_dst = gt_tmp_root / seq / "seqinfo.ini"

        gt_rows = filter_mot_rows(gt_src, gt_dst, used_frames)
        if args.pred_dir is None:
            pred_rows = filter_mot_rows(gt_src, pred_dst, used_frames)
            source_desc = "gt_copy"
        else:
            pred_src = args.pred_dir / f"{seq}.txt"
            pred_rows = filter_mot_rows(pred_src, pred_dst, used_frames)
            source_desc = str(args.pred_dir)

        write_seqinfo_with_cap(seqinfo_src, seqinfo_dst, used_frames)
        seqmap_tmp.parent.mkdir(parents=True, exist_ok=True)
        with seqmap_tmp.open("w", encoding="utf-8", newline="\n") as f:
            f.write("name\n")
            f.write(f"{seq}\n")

        id_quality = analyze_gt_identity_quality(gt_dst, max_frame=used_frames)
        id_eval_valid = (
            id_quality["repeated_ids"] > 0.0
            and id_quality["avg_track_len"] > 1.05
            and id_quality["single_frame_ratio"] < 0.98
        )
        id_quality_note = "ok" if id_eval_valid else "degenerate_gt_ids(single_frame_tracks)"
        if not id_eval_valid:
            print(
                f"[warn][id_quality] {seq}: unique={int(id_quality['unique_ids'])}, "
                f"repeated={int(id_quality['repeated_ids'])}, avg_len={id_quality['avg_track_len']:.3f}, "
                f"single_ratio={id_quality['single_frame_ratio']:.3f}. "
                "ID metrics may be unreliable."
            )

        metrics = run_trackeval_single_seq(
            trackeval_root=args.trackeval_root,
            seq=seq,
            gt_root=gt_tmp_root,
            trackers_root=trackers_tmp_root,
            seqmap_file=seqmap_tmp,
            tracker_name=args.tracker_name,
        )
        row = {
            "split": args.split,
            "sequence": seq,
            "tracker": args.tracker_name,
            "used_frames": str(used_frames),
            "gt_rows": str(gt_rows),
            "pred_rows": str(pred_rows),
            "gt_unique_ids": f"{id_quality['unique_ids']:.0f}",
            "gt_repeated_ids": f"{id_quality['repeated_ids']:.0f}",
            "gt_avg_track_len": f"{id_quality['avg_track_len']:.3f}",
            "gt_single_frame_ratio": f"{id_quality['single_frame_ratio']:.3f}",
            "id_eval_valid": "1" if id_eval_valid else "0",
            "id_quality_note": id_quality_note,
            "HOTA": f"{metrics['HOTA']:.3f}",
            "DetA": f"{metrics['DetA']:.3f}",
            "AssA": f"{metrics['AssA']:.3f}",
            "IDF1": f"{metrics['IDF1']:.3f}",
            "IDSW": f"{metrics['IDSW']:.3f}",
        }
        rows.append(row)
        print(
            f"[per_seq_eval] {seq}: frames={used_frames}, gt_rows={gt_rows}, pred_rows={pred_rows}, "
            f"HOTA={row['HOTA']}, IDF1={row['IDF1']}, source={source_desc}"
        )

    write_per_seq_csv(rows, args.results_per_seq)
    write_mean_csv(rows, args.split, args.tracker_name, args.results_mean)
    print(f"Saved per-seq metrics: {args.results_per_seq}")
    print(f"Saved mean metrics:    {args.results_mean}")


if __name__ == "__main__":
    main()
