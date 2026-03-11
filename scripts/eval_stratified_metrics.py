#!/usr/bin/env python
"""Stratified evaluation (difficulty buckets) + paper assets generation."""

from __future__ import annotations

import argparse
import configparser
import csv
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment


# TrackEval still references np.float/np.int in multiple places.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]


CRITERIA = ["occlusion", "density", "turn"]
LEVELS = ["low", "mid", "high"]


@dataclass
class GTDet:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    index: int
    occlusion_bin: int = 0
    density_bin: int = 0
    turn_bin: int = 0

    @property
    def box_xywh(self) -> np.ndarray:
        return np.array([self.x, self.y, self.w, self.h], dtype=float)

    @property
    def center_xy(self) -> np.ndarray:
        return np.array([self.x + self.w / 2.0, self.y + self.h / 2.0], dtype=float)

    def to_mot_row(self) -> str:
        return f"{self.frame},{self.track_id},{self.x:.3f},{self.y:.3f},{self.w:.3f},{self.h:.3f},1,1,1"


@dataclass
class PredDet:
    frame: int
    row: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stratified MOT metrics and generate paper assets.")
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
    parser.add_argument("--max-frames", type=int, default=1000, help="Use up to this many frames per sequence.")
    parser.add_argument(
        "--max-gt-ids",
        type=int,
        default=2000,
        help="Safety cap for unique GT IDs per sequence to avoid HOTA OOM (0 disables).",
    )
    parser.add_argument(
        "--pred-base",
        type=Path,
        default=Path("results/p3/pred_base"),
        help="Prediction directory for Base.",
    )
    parser.add_argument(
        "--pred-gating",
        type=Path,
        default=Path("results/p3/pred_gating_hard"),
        help="Prediction directory for +gating.",
    )
    parser.add_argument(
        "--pred-traj",
        type=Path,
        default=Path("results/p3/pred_traj_hard"),
        help="Prediction directory for +traj.",
    )
    parser.add_argument(
        "--pred-adaptive",
        type=Path,
        default=Path("results/p3/pred_adaptive_hard"),
        help="Prediction directory for +adaptive.",
    )
    parser.add_argument(
        "--tmp-root",
        type=Path,
        default=Path("results/_stratified_tmp"),
        help="Temporary workspace for bucketed TrackEval files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/stratified_metrics.csv"),
        help="Stratified metrics CSV output.",
    )
    parser.add_argument(
        "--paper-assets-dir",
        type=Path,
        default=Path("results/paper_assets"),
        help="Directory for paper figures and LaTeX table.",
    )
    parser.add_argument(
        "--pseudo-iou-thresh",
        type=float,
        default=0.3,
        help="IoU threshold for pseudo-track building (turn bucket).",
    )
    return parser.parse_args()


def read_seqmap(seqmap_path: Path) -> List[str]:
    if not seqmap_path.exists():
        raise FileNotFoundError(f"Missing seqmap file: {seqmap_path}")
    seqs: List[str] = []
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
            seqs.append(name)
    return seqs


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
    chosen = frame_cap
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            parts = row.split(",")
            if len(parts) < 2:
                continue
            frame = int(float(parts[0]))
            if frame > frame_cap:
                break
            track_id = int(float(parts[1]))
            seen_ids.add(track_id)
            if len(seen_ids) > max_gt_ids:
                chosen = max(1, frame - 1)
                break
    return chosen


def parse_gt_file(gt_path: Path, max_frame: int) -> List[GTDet]:
    rows: List[GTDet] = []
    idx = 0
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            parts = row.split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            if frame > max_frame:
                continue
            track_id = int(float(parts[1]))
            x, y, w, h = [float(v) for v in parts[2:6]]
            if w <= 0 or h <= 0:
                continue
            rows.append(GTDet(frame=frame, track_id=track_id, x=x, y=y, w=w, h=h, index=idx))
            idx += 1
    return rows


def parse_pred_file(pred_path: Path, max_frame: int) -> List[PredDet]:
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    rows: List[PredDet] = []
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            parts = row.split(",")
            if len(parts) < 1:
                continue
            frame = int(float(parts[0]))
            if frame > max_frame:
                continue
            rows.append(PredDet(frame=frame, row=row))
    return rows


def iou_batch(dets_xywh: np.ndarray, trks_xywh: np.ndarray) -> np.ndarray:
    if dets_xywh.size == 0 or trks_xywh.size == 0:
        return np.zeros((dets_xywh.shape[0], trks_xywh.shape[0]), dtype=float)

    dets_x1 = dets_xywh[:, 0:1]
    dets_y1 = dets_xywh[:, 1:2]
    dets_x2 = dets_x1 + dets_xywh[:, 2:3]
    dets_y2 = dets_y1 + dets_xywh[:, 3:4]

    trks_x1 = trks_xywh[:, 0][None, :]
    trks_y1 = trks_xywh[:, 1][None, :]
    trks_x2 = trks_x1 + trks_xywh[:, 2][None, :]
    trks_y2 = trks_y1 + trks_xywh[:, 3][None, :]

    inter_x1 = np.maximum(dets_x1, trks_x1)
    inter_y1 = np.maximum(dets_y1, trks_y1)
    inter_x2 = np.minimum(dets_x2, trks_x2)
    inter_y2 = np.minimum(dets_y2, trks_y2)
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    det_area = np.maximum(0.0, dets_xywh[:, 2:3]) * np.maximum(0.0, dets_xywh[:, 3:4])
    trk_area = np.maximum(0.0, trks_xywh[:, 2][None, :]) * np.maximum(0.0, trks_xywh[:, 3][None, :])
    union = det_area + trk_area - inter_area
    return np.where(union > 0.0, inter_area / union, 0.0)


def quantile_bins(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros((0,), dtype=int)
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    if q1 < q2:
        bins = np.where(values <= q1, 0, np.where(values <= q2, 1, 2))
        return bins.astype(int)
    # fallback: equal-frequency by rank
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(values.size)
    bins = (3 * ranks) // max(1, values.size)
    bins = np.clip(bins, 0, 2)
    return bins.astype(int)


def assign_density_bins(rows: List[GTDet]) -> None:
    frame_to_indices: Dict[int, List[int]] = {}
    for i, row in enumerate(rows):
        frame_to_indices.setdefault(row.frame, []).append(i)
    frames = sorted(frame_to_indices.keys())
    counts = np.asarray([len(frame_to_indices[f]) for f in frames], dtype=float)
    frame_bins = quantile_bins(counts)
    frame_to_bin = {f: int(b) for f, b in zip(frames, frame_bins)}
    for row in rows:
        row.density_bin = frame_to_bin[row.frame]


def assign_occlusion_bins(rows: List[GTDet]) -> None:
    frame_to_indices: Dict[int, List[int]] = {}
    for i, row in enumerate(rows):
        frame_to_indices.setdefault(row.frame, []).append(i)

    scores = np.zeros((len(rows),), dtype=float)
    for _, indices in frame_to_indices.items():
        if len(indices) <= 1:
            continue
        boxes = np.asarray([rows[i].box_xywh for i in indices], dtype=float)
        iou = iou_batch(boxes, boxes)
        np.fill_diagonal(iou, 0.0)
        neighbor_count = (iou > 0.1).sum(axis=1).astype(float)
        max_overlap = iou.max(axis=1)
        local_score = neighbor_count + max_overlap
        for local_i, global_i in enumerate(indices):
            scores[global_i] = local_score[local_i]

    bins = quantile_bins(scores)
    for i, row in enumerate(rows):
        row.occlusion_bin = int(bins[i])


def build_pseudo_tracks(rows: List[GTDet], iou_thresh: float) -> List[List[int]]:
    frame_to_indices: Dict[int, List[int]] = {}
    for i, row in enumerate(rows):
        frame_to_indices.setdefault(row.frame, []).append(i)

    active: Dict[int, int] = {}  # pseudo_track_id -> global row index
    tracks: Dict[int, List[int]] = {}
    next_id = 1

    for frame in sorted(frame_to_indices.keys()):
        det_indices = frame_to_indices[frame]
        det_boxes = np.asarray([rows[i].box_xywh for i in det_indices], dtype=float)
        active_ids = list(active.keys())
        active_boxes = np.asarray([rows[active[tid]].box_xywh for tid in active_ids], dtype=float)

        matched: List[Tuple[int, int]] = []
        used_det = set()
        if det_boxes.size > 0 and active_boxes.size > 0:
            iou = iou_batch(det_boxes, active_boxes)
            rr, cc = linear_sum_assignment(1.0 - iou)
            for det_i, trk_i in zip(rr.tolist(), cc.tolist()):
                if iou[det_i, trk_i] < iou_thresh:
                    continue
                matched.append((det_i, trk_i))
                used_det.add(det_i)

        new_active: Dict[int, int] = {}
        for det_i, trk_i in matched:
            tid = active_ids[trk_i]
            global_idx = det_indices[det_i]
            tracks.setdefault(tid, []).append(global_idx)
            new_active[tid] = global_idx

        for det_i, global_idx in enumerate(det_indices):
            if det_i in used_det:
                continue
            tid = next_id
            next_id += 1
            tracks[tid] = [global_idx]
            new_active[tid] = global_idx

        active = new_active

    return list(tracks.values())


def assign_turn_bins(rows: List[GTDet], pseudo_iou_thresh: float) -> None:
    scores = np.zeros((len(rows),), dtype=float)
    tracks = build_pseudo_tracks(rows, iou_thresh=pseudo_iou_thresh)

    for track in tracks:
        if len(track) < 3:
            continue
        centers = np.asarray([rows[i].center_xy for i in track], dtype=float)
        diag = np.linalg.norm(np.max(centers, axis=0) - np.min(centers, axis=0)) + 1e-6
        for i in range(1, len(track) - 1):
            v_prev = centers[i] - centers[i - 1]
            v_next = centers[i + 1] - centers[i]
            n1 = float(np.linalg.norm(v_prev))
            n2 = float(np.linalg.norm(v_next))
            if n1 < 1e-6 or n2 < 1e-6:
                angle = 0.0
            else:
                cos_val = float(np.dot(v_prev, v_next) / (n1 * n2 + 1e-6))
                cos_val = max(-1.0, min(1.0, cos_val))
                angle = float(math.acos(cos_val))  # [0, pi]
            accel = float(np.linalg.norm(v_next - v_prev) / diag)
            score = angle + 0.5 * accel
            global_idx = track[i]
            scores[global_idx] = score

    bins = quantile_bins(scores)
    for i, row in enumerate(rows):
        row.turn_bin = int(bins[i])


def run_trackeval_for_one_seq_bucket(
    trackeval_root: Path,
    gt_root: Path,
    trackers_root: Path,
    seqmap_file: Path,
    tracker_ids: List[str],
) -> Dict[str, Dict[str, float]]:
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
            "TRACKERS_TO_EVAL": tracker_ids,
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

    metric_by_tracker: Dict[str, Dict[str, float]] = {}
    for tracker_id in tracker_ids:
        status = output_msg[dataset_name][tracker_id]
        if status != "Success":
            raise RuntimeError(f"TrackEval failed for tracker={tracker_id}: {status}")
        combined = output_res[dataset_name][tracker_id]["COMBINED_SEQ"]["pedestrian"]
        metric_by_tracker[tracker_id] = {
            "HOTA": float(np.mean(combined["HOTA"]["HOTA"]) * 100.0),
            "DetA": float(np.mean(combined["HOTA"]["DetA"]) * 100.0),
            "AssA": float(np.mean(combined["HOTA"]["AssA"]) * 100.0),
            "IDF1": float(combined["Identity"]["IDF1"] * 100.0),
            "IDSW": float(combined["CLEAR"]["IDSW"]),
        }
    return metric_by_tracker


def aggregate_and_write_csv(rows: List[Dict[str, str]], out_csv: Path) -> List[Dict[str, str]]:
    # rows are per-seq results. Aggregate to mean by (bucket_type,bucket,method).
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
    for row in rows:
        key = (row["bucket_type"], row["bucket"], row["method"])
        grouped.setdefault(key, []).append(row)

    agg_rows: List[Dict[str, str]] = []
    for (bucket_type, bucket, method), sub in grouped.items():
        vals = {
            "HOTA": [float(x["HOTA"]) for x in sub],
            "DetA": [float(x["DetA"]) for x in sub],
            "AssA": [float(x["AssA"]) for x in sub],
            "IDF1": [float(x["IDF1"]) for x in sub],
            "IDSW": [float(x["IDSW"]) for x in sub],
            "gt_rows": [float(x["gt_rows"]) for x in sub],
            "pred_rows": [float(x["pred_rows"]) for x in sub],
        }
        agg_rows.append(
            {
                "split": sub[0]["split"],
                "bucket_type": bucket_type,
                "bucket": bucket,
                "method": method,
                "num_sequences": str(len(sub)),
                "gt_rows": f"{np.sum(vals['gt_rows']):.0f}",
                "pred_rows": f"{np.sum(vals['pred_rows']):.0f}",
                "HOTA": f"{np.mean(vals['HOTA']):.3f}",
                "DetA": f"{np.mean(vals['DetA']):.3f}",
                "AssA": f"{np.mean(vals['AssA']):.3f}",
                "IDF1": f"{np.mean(vals['IDF1']):.3f}",
                "IDSW": f"{np.mean(vals['IDSW']):.3f}",
            }
        )

    # Stable ordering for paper.
    method_order = ["Base", "+gating", "+traj", "+adaptive"]
    bucket_order = [(c, l) for c in CRITERIA for l in LEVELS]
    order_map = {x: i for i, x in enumerate(bucket_order)}
    method_map = {m: i for i, m in enumerate(method_order)}
    agg_rows.sort(
        key=lambda x: (
            order_map[(x["bucket_type"], x["bucket"])],
            method_map.get(x["method"], 999),
        )
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "bucket_type", "bucket", "method", "num_sequences", "gt_rows", "pred_rows", "HOTA", "DetA", "AssA", "IDF1", "IDSW"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in agg_rows:
            writer.writerow(row)
    return agg_rows


def make_paper_assets(rows: List[Dict[str, str]], assets_dir: Path) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Prepare matrix for plots.
    methods = ["Base", "+gating", "+traj", "+adaptive"]
    bucket_keys = [(c, l) for c in CRITERIA for l in LEVELS]
    bucket_labels = [f"{c[:3]}-{l}" for c, l in bucket_keys]
    row_map: Dict[Tuple[str, str, str], Dict[str, str]] = {
        (r["bucket_type"], r["bucket"], r["method"]): r for r in rows
    }

    metrics = ["HOTA", "AssA", "IDF1"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True)
    x = np.arange(len(bucket_keys))
    width = 0.18
    for ax, metric in zip(axes, metrics):
        for m_i, method in enumerate(methods):
            vals = []
            for b in bucket_keys:
                row = row_map.get((b[0], b[1], method))
                vals.append(float(row[metric]) if row else np.nan)
            ax.bar(x + (m_i - 1.5) * width, vals, width=width, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(bucket_labels, rotation=35, ha="right")
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="best")
    fig.suptitle("Stratified Metrics by Difficulty Bucket")
    plot_main = assets_dir / "stratified_main_metrics.png"
    fig.savefig(plot_main, dpi=180)
    plt.close(fig)

    # IDSW plot (lower is better).
    fig2, ax2 = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for m_i, method in enumerate(methods):
        vals = []
        for b in bucket_keys:
            row = row_map.get((b[0], b[1], method))
            vals.append(float(row["IDSW"]) if row else np.nan)
        ax2.plot(bucket_labels, vals, marker="o", label=method)
    ax2.set_title("IDSW by Bucket (lower is better)")
    ax2.set_ylabel("IDSW")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(loc="best")
    for label in ax2.get_xticklabels():
        label.set_rotation(35)
        label.set_ha("right")
    plot_idsw = assets_dir / "stratified_idsw.png"
    fig2.savefig(plot_idsw, dpi=180)
    plt.close(fig2)

    # LaTeX table
    tex_path = assets_dir / "stratified_table.tex"
    with tex_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("% Auto-generated by scripts/eval_stratified_metrics.py\n")
        f.write("\\begin{tabular}{llrrrr}\n")
        f.write("\\hline\n")
        f.write("Bucket & Method & HOTA & AssA & IDF1 & IDSW \\\\\n")
        f.write("\\hline\n")
        for c, l in bucket_keys:
            bucket_name = f"{c}_{l}"
            for method in methods:
                row = row_map.get((c, l, method))
                if row is None:
                    continue
                f.write(
                    f"{bucket_name} & {method} & {row['HOTA']} & {row['AssA']} & "
                    f"{row['IDF1']} & {row['IDSW']} \\\\\n"
                )
            f.write("\\hline\n")
        f.write("\\end{tabular}\n")

    readme = assets_dir / "README.txt"
    with readme.open("w", encoding="utf-8", newline="\n") as f:
        f.write("Generated assets:\n")
        f.write("- stratified_main_metrics.png\n")
        f.write("- stratified_idsw.png\n")
        f.write("- stratified_table.tex\n")


def main() -> None:
    args = parse_args()
    split_dir = args.mot_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Prepared split not found: {split_dir}. "
            "Please run scripts/prepare_mft25.py first."
        )

    method_specs = [
        {"id": "base", "label": "Base", "pred_dir": args.pred_base},
        {"id": "gating", "label": "+gating", "pred_dir": args.pred_gating},
        {"id": "traj", "label": "+traj", "pred_dir": args.pred_traj},
        {"id": "adaptive", "label": "+adaptive", "pred_dir": args.pred_adaptive},
    ]
    for spec in method_specs:
        if not spec["pred_dir"].exists():
            raise FileNotFoundError(f"Prediction directory not found for {spec['label']}: {spec['pred_dir']}")

    seqs = read_seqmap(split_dir / "seqmaps" / f"{args.split}.txt")
    if not seqs:
        raise RuntimeError("No sequences found in split seqmap.")

    # Load GT and predictions (with sequence-level frame cap for OOM safety).
    seq_gt: Dict[str, List[GTDet]] = {}
    seq_used_frames: Dict[str, int] = {}
    seq_pred: Dict[str, Dict[str, List[PredDet]]] = {m["id"]: {} for m in method_specs}

    for seq in seqs:
        seq_dir = split_dir / seq
        gt_path = seq_dir / "gt" / "gt.txt"
        seqinfo_path = seq_dir / "seqinfo.ini"
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing gt file: {gt_path}")
        if not seqinfo_path.exists():
            raise FileNotFoundError(f"Missing seqinfo.ini: {seqinfo_path}")

        seq_len = read_seq_length(seqinfo_path)
        requested_cap = seq_len if args.max_frames <= 0 else min(seq_len, args.max_frames)
        id_cap = choose_frame_cap(gt_path, requested_cap, args.max_gt_ids)
        used_frames = max(1, min(requested_cap, id_cap))
        seq_used_frames[seq] = used_frames

        gt_rows = parse_gt_file(gt_path, max_frame=used_frames)
        seq_gt[seq] = gt_rows

        for spec in method_specs:
            pred_path = Path(spec["pred_dir"]) / f"{seq}.txt"
            seq_pred[spec["id"]][seq] = parse_pred_file(pred_path, max_frame=used_frames)

        print(f"[stratified] loaded {seq}: gt_rows={len(gt_rows)}, used_frames={used_frames}")

    # Assign bucket labels based on GT.
    all_gt_rows: List[GTDet] = []
    for seq in seqs:
        all_gt_rows.extend(seq_gt[seq])
    assign_density_bins(all_gt_rows)
    assign_occlusion_bins(all_gt_rows)
    assign_turn_bins(all_gt_rows, pseudo_iou_thresh=args.pseudo_iou_thresh)

    # Evaluate per sequence × bucket (all methods in one TrackEval call).
    shutil.rmtree(args.tmp_root, ignore_errors=True)
    args.tmp_root.mkdir(parents=True, exist_ok=True)
    per_seq_rows: List[Dict[str, str]] = []

    for criterion in CRITERIA:
        for level_idx, level in enumerate(LEVELS):
            for seq in seqs:
                gt_rows = seq_gt[seq]
                selected = [
                    r
                    for r in gt_rows
                    if (r.occlusion_bin if criterion == "occlusion" else r.density_bin if criterion == "density" else r.turn_bin)
                    == level_idx
                ]
                if not selected:
                    continue
                selected_frames = {r.frame for r in selected}

                seq_tmp = args.tmp_root / f"{criterion}_{level}" / seq
                gt_root = seq_tmp / "gt_root"
                trackers_root = seq_tmp / "trackers_root"
                seqmap_file = gt_root / "seqmaps" / "single.txt"

                gt_file = gt_root / seq / "gt" / "gt.txt"
                gt_file.parent.mkdir(parents=True, exist_ok=True)
                with gt_file.open("w", encoding="utf-8", newline="\n") as f:
                    for r in selected:
                        f.write(r.to_mot_row() + "\n")

                seqinfo_src = split_dir / seq / "seqinfo.ini"
                seqinfo_dst = gt_root / seq / "seqinfo.ini"
                parser = configparser.ConfigParser()
                parser.optionxform = str
                parser.read(seqinfo_src, encoding="utf-8")
                parser["Sequence"]["seqLength"] = str(max(selected_frames))
                seqinfo_dst.parent.mkdir(parents=True, exist_ok=True)
                with seqinfo_dst.open("w", encoding="utf-8", newline="\n") as f:
                    parser.write(f, space_around_delimiters=False)

                seqmap_file.parent.mkdir(parents=True, exist_ok=True)
                with seqmap_file.open("w", encoding="utf-8", newline="\n") as f:
                    f.write("name\n")
                    f.write(f"{seq}\n")

                tracker_ids = [m["id"] for m in method_specs]
                pred_counts: Dict[str, int] = {}
                for spec in method_specs:
                    pred_rows = seq_pred[spec["id"]][seq]
                    filtered = [p.row for p in pred_rows if p.frame in selected_frames]
                    pred_file = trackers_root / spec["id"] / "data" / f"{seq}.txt"
                    pred_file.parent.mkdir(parents=True, exist_ok=True)
                    with pred_file.open("w", encoding="utf-8", newline="\n") as f:
                        for row in filtered:
                            f.write(row + "\n")
                    pred_counts[spec["id"]] = len(filtered)

                metric_by_tracker = run_trackeval_for_one_seq_bucket(
                    trackeval_root=args.trackeval_root,
                    gt_root=gt_root,
                    trackers_root=trackers_root,
                    seqmap_file=seqmap_file,
                    tracker_ids=tracker_ids,
                )

                for spec in method_specs:
                    metric = metric_by_tracker[spec["id"]]
                    per_seq_rows.append(
                        {
                            "split": args.split,
                            "sequence": seq,
                            "bucket_type": criterion,
                            "bucket": level,
                            "method": spec["label"],
                            "gt_rows": str(len(selected)),
                            "pred_rows": str(pred_counts[spec["id"]]),
                            "HOTA": f"{metric['HOTA']:.3f}",
                            "DetA": f"{metric['DetA']:.3f}",
                            "AssA": f"{metric['AssA']:.3f}",
                            "IDF1": f"{metric['IDF1']:.3f}",
                            "IDSW": f"{metric['IDSW']:.3f}",
                        }
                    )
                print(
                    f"[stratified] {criterion}-{level} {seq}: gt_rows={len(selected)}, "
                    f"HOTA(Base)={metric_by_tracker['base']['HOTA']:.3f}"
                )

    agg_rows = aggregate_and_write_csv(per_seq_rows, args.output_csv)
    make_paper_assets(agg_rows, args.paper_assets_dir)

    print(f"Saved stratified metrics: {args.output_csv}")
    print(f"Saved paper assets dir:  {args.paper_assets_dir}")


if __name__ == "__main__":
    main()
