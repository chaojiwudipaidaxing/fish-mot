#!/usr/bin/env python
"""Stratified bucket evaluation (smoke-first) for val/train splits."""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment


METHOD_SPECS: List[Tuple[str, str]] = [
    ("Base", "pred_base"),
    ("+gating", "pred_gating"),
    ("+traj", "pred_traj"),
    ("+adaptive", "pred_adaptive"),
]
CRITERIA = ["occlusion", "density", "turn", "low_conf"]
LEVELS = ["low", "mid", "high"]
CRITERION_TITLES = {
    "occlusion": "Occlusion",
    "density": "Density",
    "turn": "Turning",
    "low_conf": "Low-confidence",
}
PLOT_DPI = 500
PLOT_SAVE_KWARGS = {
    "dpi": PLOT_DPI,
    "bbox_inches": "tight",
    "pad_inches": 0.02,
}
PLOT_STYLE = {
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "grid.linewidth": 0.4,
}


@dataclass
class GTDet:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    index: int

    @property
    def box(self) -> np.ndarray:
        return np.array([self.x, self.y, self.w, self.h], dtype=float)

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x + self.w / 2.0, self.y + self.h / 2.0], dtype=float)


@dataclass
class PredDet:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    score: float

    @property
    def box(self) -> np.ndarray:
        return np.array([self.x, self.y, self.w, self.h], dtype=float)


@dataclass
class BucketConfig:
    assignments: np.ndarray
    bucket_mode: str
    q1_used: float
    q2_used: float
    threshold_t1: float
    threshold_t2: float
    counts: Dict[str, int]
    shares: Dict[str, float]
    used_buckets: int
    active_levels: List[str]
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stratified metrics (default smoke: max_frames=20).")
    parser.add_argument("--split", default="val_half", help="Prepared split name under data/mft25_mot.")
    parser.add_argument(
        "--mot-root",
        type=Path,
        default=Path("data/mft25_mot"),
        help="Root directory of prepared MOT data.",
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=Path("results/main_val/seed_runs/seed_0"),
        help="Root directory containing pred_base/pred_gating/pred_traj/pred_adaptive.",
    )
    parser.add_argument("--pred-base", type=Path, default=None, help="Override Base prediction directory.")
    parser.add_argument("--pred-gating", type=Path, default=None, help="Override +gating prediction directory.")
    parser.add_argument("--pred-traj", type=Path, default=None, help="Override +traj prediction directory.")
    parser.add_argument("--pred-adaptive", type=Path, default=None, help="Override +adaptive prediction directory.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=20,
        help="Use up to this many frames per sequence (smoke default: 20, full by user).",
    )
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for TP matching.")
    parser.add_argument(
        "--bucket-mode",
        choices=["fixed", "quantile"],
        default="quantile",
        help="Bucket strategy. fixed=legacy, quantile=adaptive quantile split.",
    )
    parser.add_argument("--q1", type=float, default=0.33, help="Low/mid quantile boundary (quantile mode).")
    parser.add_argument("--q2", type=float, default=0.66, help="Mid/high quantile boundary (quantile mode).")
    parser.add_argument(
        "--bucket-min-samples",
        type=int,
        default=200,
        help="Minimum target samples per bucket in quantile mode.",
    )
    parser.add_argument(
        "--min-bucket-frames",
        type=int,
        default=None,
        help="Deprecated alias of --bucket-min-samples (kept for compatibility).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/stratified_metrics_val.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("results/paper_assets_val/stratified_metrics_val.png"),
        help="Output plot path.",
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=None,
        help="Optional run_config.json. If provided, split/max_frames/bucket settings are loaded from it.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="If set, render plot directly from an existing stratified metrics CSV.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip metric computation and render plot only from --input-csv.",
    )
    return parser.parse_args()


def apply_run_config(args: argparse.Namespace) -> None:
    if args.run_config is None:
        return
    if not args.run_config.exists():
        raise FileNotFoundError(f"Run config not found: {args.run_config}")
    cfg = json.loads(args.run_config.read_text(encoding="utf-8"))
    args.split = str(cfg.get("split", args.split))
    args.mot_root = Path(str(cfg.get("mot_root", args.mot_root)))
    args.max_frames = int(cfg.get("max_frames", args.max_frames))
    args.bucket_mode = str(cfg.get("bucket_mode", args.bucket_mode))
    args.q1 = float(cfg.get("q1", args.q1))
    args.q2 = float(cfg.get("q2", args.q2))
    args.bucket_min_samples = int(cfg.get("bucket_min_samples", args.bucket_min_samples))
    pred_root = cfg.get("pred_root")
    if pred_root:
        args.pred_root = Path(str(pred_root))
    result_root = Path(str(cfg.get("result_root", "results/main_val")))
    stratified_dir = Path(str(cfg.get("stratified_dir", result_root / "stratified")))
    paper_assets_dir = Path(str(cfg.get("paper_assets_dir", result_root / "paper_assets")))
    if str(args.output_csv).replace("\\", "/").lower() == "results/stratified_metrics_val.csv":
        args.output_csv = stratified_dir / "stratified_metrics_val.csv"
    if str(args.plot_path).replace("\\", "/").lower() == "results/paper_assets_val/stratified_metrics_val.png":
        args.plot_path = paper_assets_dir / "stratified_metrics_val.png"
    root_norm = str(result_root).replace("\\", "/").lower()
    write_targets = [
        ("output_csv", args.output_csv),
        ("plot_path", args.plot_path),
    ]
    bad = []
    for name, path in write_targets:
        norm = str(path).replace("\\", "/").lower()
        if norm.startswith("results/") and norm != root_norm and not norm.startswith(root_norm + "/"):
            bad.append(f"{name}={path}")
    if bad:
        raise RuntimeError(
            "run-config strict mode blocks legacy/global outputs. "
            f"result_root={result_root}. bad targets: {'; '.join(bad)}"
        )


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


def load_gt(gt_path: Path, max_frame: int, start_index: int) -> List[GTDet]:
    rows: List[GTDet] = []
    idx = start_index
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            parts = row.split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            if frame < 1 or frame > max_frame:
                continue
            track_id = int(float(parts[1]))
            x, y, w, h = [float(v) for v in parts[2:6]]
            if w <= 0 or h <= 0:
                continue
            rows.append(GTDet(frame=frame, track_id=track_id, x=x, y=y, w=w, h=h, index=idx))
            idx += 1
    return rows


def load_pred(pred_path: Path, max_frame: int) -> List[PredDet]:
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    rows: List[PredDet] = []
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            parts = row.split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            if frame < 1 or frame > max_frame:
                continue
            track_id = int(float(parts[1]))
            x, y, w, h = [float(v) for v in parts[2:6]]
            score = float(parts[6]) if len(parts) >= 7 else 1.0
            if w <= 0 or h <= 0:
                continue
            rows.append(PredDet(frame=frame, track_id=track_id, x=x, y=y, w=w, h=h, score=score))
    return rows


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)
    ax1 = a[:, 0:1]
    ay1 = a[:, 1:2]
    ax2 = ax1 + a[:, 2:3]
    ay2 = ay1 + a[:, 3:4]
    bx1 = b[:, 0][None, :]
    by1 = b[:, 1][None, :]
    bx2 = bx1 + b[:, 2][None, :]
    by2 = by1 + b[:, 3][None, :]
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = np.maximum(0.0, a[:, 2:3]) * np.maximum(0.0, a[:, 3:4])
    area_b = np.maximum(0.0, b[:, 2][None, :]) * np.maximum(0.0, b[:, 3][None, :])
    union = area_a + area_b - inter
    return np.where(union > 0.0, inter / union, 0.0)


def fixed_bins(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros((0,), dtype=int)
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    if q1 < q2:
        bins = np.where(values <= q1, 0, np.where(values <= q2, 1, 2))
        return bins.astype(int)
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(values.size)
    bins = np.clip((3 * ranks) // max(1, values.size), 0, 2)
    return bins.astype(int)


def rank_bins(values: np.ndarray, num_bins: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros((0,), dtype=int)
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(values.size)
    bins = np.clip((num_bins * ranks) // max(1, values.size), 0, num_bins - 1)
    return bins.astype(int)


def compute_quantile_edges(values: np.ndarray, q1: float, q2: float) -> Tuple[float, float]:
    t1, t2 = np.quantile(values, [q1, q2])
    return float(t1), float(t2)


def assign_bucket(values: np.ndarray, t1: float, t2: float) -> np.ndarray:
    return np.where(values <= t1, 0, np.where(values <= t2, 1, 2)).astype(int)


def bucket_counts(assignments: np.ndarray) -> Dict[str, int]:
    return {
        "low": int(np.sum(assignments == 0)),
        "mid": int(np.sum(assignments == 1)),
        "high": int(np.sum(assignments == 2)),
    }


def bucket_shares(counts: Dict[str, int], total: int) -> Dict[str, float]:
    denom = float(max(1, total))
    return {k: float(v / denom) for k, v in counts.items()}


def nearest_indices(center: int, low: int, high: int, max_items: int) -> List[int]:
    center = int(max(low, min(high, center)))
    out: List[int] = [center]
    step = 1
    while len(out) < max_items and (center - step >= low or center + step <= high):
        if center - step >= low:
            out.append(center - step)
            if len(out) >= max_items:
                break
        if center + step <= high:
            out.append(center + step)
            if len(out) >= max_items:
                break
        step += 1
    return out


def search_edges_for_min_count(
    scores: np.ndarray,
    q1: float,
    q2: float,
    target_min: int,
    max_attempts: int = 2000,
) -> Tuple[float, float, np.ndarray, Dict[str, int], int, bool]:
    if scores.size == 0:
        raise RuntimeError("Cannot search edges for empty score array.")
    if target_min <= 0:
        t1, t2 = compute_quantile_edges(scores, q1, q2)
        assignments = assign_bucket(scores, t1, t2)
        return t1, t2, assignments, bucket_counts(assignments), 0, True

    unique_vals = np.unique(scores)
    if unique_vals.size < 2:
        t1 = float(unique_vals[0])
        t2 = float(unique_vals[0])
        assignments = assign_bucket(scores, t1, t2)
        return t1, t2, assignments, bucket_counts(assignments), 0, False

    t1_init, t2_init = compute_quantile_edges(scores, q1, q2)
    idx1_init = int(np.searchsorted(unique_vals, t1_init, side="right") - 1)
    idx2_init = int(np.searchsorted(unique_vals, t2_init, side="right") - 1)
    idx1_init = int(np.clip(idx1_init, 0, unique_vals.size - 2))
    idx2_init = int(np.clip(idx2_init, idx1_init + 1, unique_vals.size - 1))

    max_candidates = min(unique_vals.size, 81)
    cand1 = nearest_indices(idx1_init, 0, unique_vals.size - 2, max_candidates)
    cand2 = nearest_indices(idx2_init, 1, unique_vals.size - 1, max_candidates)

    pair_candidates: List[Tuple[int, int, int]] = []
    seen_pairs: Set[Tuple[int, int]] = set()
    for i1 in cand1:
        for i2 in cand2:
            if i2 <= i1:
                continue
            key = (i1, i2)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            dist = abs(i1 - idx1_init) + abs(i2 - idx2_init)
            pair_candidates.append((dist, i1, i2))
    pair_candidates.sort(key=lambda x: x[0])

    best_record: Tuple[int, int, float, float, np.ndarray, Dict[str, int]] | None = None
    attempts = 0
    for _, i1, i2 in pair_candidates:
        attempts += 1
        t1 = float(unique_vals[i1])
        t2 = float(unique_vals[i2])
        assignments = assign_bucket(scores, t1, t2)
        counts = bucket_counts(assignments)
        min_count = min(counts["low"], counts["mid"], counts["high"])
        if min_count >= target_min:
            return t1, t2, assignments, counts, attempts, True
        if best_record is None or min_count > best_record[0]:
            best_record = (min_count, attempts, t1, t2, assignments, counts)
        if attempts >= max_attempts:
            break

    if best_record is not None:
        _, best_attempts, t1_best, t2_best, assignments_best, counts_best = best_record
        return t1_best, t2_best, assignments_best, counts_best, best_attempts, False

    t1, t2 = t1_init, t2_init
    assignments = assign_bucket(scores, t1, t2)
    return t1, t2, assignments, bucket_counts(assignments), attempts, False


def merge_to_two_bins(scores: np.ndarray, t1: float, t2: float) -> Tuple[np.ndarray, Dict[str, int], str]:
    assignments = assign_bucket(scores, t1, t2)
    counts = bucket_counts(assignments)

    if counts["high"] == 0:
        two = np.where(scores <= t1, 0, 2).astype(int)
        return two, bucket_counts(two), "mid_merged_into_high"
    if counts["low"] == 0:
        two = np.where(scores <= t2, 0, 2).astype(int)
        return two, bucket_counts(two), "mid_merged_into_low"
    if counts["mid"] == 0:
        two = np.where(scores <= t1, 0, 2).astype(int)
        return two, bucket_counts(two), "two_bins_due_to_empty_mid"

    two = assignments.copy()
    mid_idx = np.where(assignments == 1)[0]
    if mid_idx.size > 0:
        dist_low = np.abs(scores[mid_idx] - t1)
        dist_high = np.abs(scores[mid_idx] - t2)
        to_low = dist_low <= dist_high
        two[mid_idx[to_low]] = 0
        two[mid_idx[~to_low]] = 2
    two[two == 1] = 0
    two_counts = bucket_counts(two)
    note = "two_bins_due_to_target_min"

    if two_counts["low"] == 0 or two_counts["high"] == 0:
        rank2 = rank_bins(scores, num_bins=2)
        two = np.where(rank2 == 0, 0, 2).astype(int)
        two_counts = bucket_counts(two)
        note += ";two_bins_rank_fallback"
    return two, two_counts, note


def build_fixed_bucket_config(scores: np.ndarray) -> BucketConfig:
    if scores.size == 0:
        raise RuntimeError("No frame scores available for fixed bucketing.")
    assignments = fixed_bins(scores)
    t1, t2 = np.quantile(scores, [1.0 / 3.0, 2.0 / 3.0])
    counts = bucket_counts(assignments)
    active_levels = [level for level in LEVELS if counts[level] > 0]
    used_buckets = len(active_levels)
    note = "fixed_legacy" if used_buckets == 3 else "fixed_legacy;bucket_reduced"
    return BucketConfig(
        assignments=assignments,
        bucket_mode="fixed",
        q1_used=1.0 / 3.0,
        q2_used=2.0 / 3.0,
        threshold_t1=float(t1),
        threshold_t2=float(t2),
        counts=counts,
        shares=bucket_shares(counts, scores.size),
        used_buckets=used_buckets,
        active_levels=active_levels,
        note=note,
    )


def build_quantile_bucket_config(
    scores: np.ndarray,
    q1: float,
    q2: float,
    min_bucket_samples: int,
) -> BucketConfig:
    if scores.size == 0:
        raise RuntimeError("No frame scores available for quantile bucketing.")
    if not (0.0 < q1 < q2 < 1.0):
        raise ValueError(f"Invalid quantiles: q1={q1}, q2={q2}. Expect 0 < q1 < q2 < 1.")
    target_min = max(1, int(min_bucket_samples))

    t1_init, t2_init = compute_quantile_edges(scores, q1, q2)
    assignments_init = assign_bucket(scores, t1_init, t2_init)
    counts_init = bucket_counts(assignments_init)

    t1_used, t2_used, assignments, counts, attempts, found_three = search_edges_for_min_count(
        scores=scores,
        q1=q1,
        q2=q2,
        target_min=target_min,
    )
    notes: List[str] = [f"quantile(q={q1:.2f}/{q2:.2f})"]

    if found_three:
        notes.append("ok" if attempts == 0 else f"edge_search_ok(target_min={target_min},attempts={attempts})")
        active_levels = LEVELS[:]
    else:
        two_assignments, two_counts, merge_note = merge_to_two_bins(scores, t1_used, t2_used)
        assignments = two_assignments
        counts = two_counts
        active_levels = [level for level in LEVELS if counts[level] > 0]
        notes.append(f"quantile_search_unmet(target_min={target_min},attempts={attempts},fallback=two_bins)")
        notes.append(merge_note)
        if min(counts["low"], counts["high"]) < target_min:
            notes.append("target_min_relaxed_for_two_bins")

    if len(active_levels) < 2:
        # Hard fallback to ensure at least two non-empty buckets.
        rank2 = rank_bins(scores, num_bins=2)
        assignments = np.where(rank2 == 0, 0, 2).astype(int)
        counts = bucket_counts(assignments)
        active_levels = [level for level in LEVELS if counts[level] > 0]
        notes.append("forced_two_bins_rank_fallback")

    if not active_levels:
        raise RuntimeError("Quantile bucketing failed: no non-empty bucket after fallback.")

    note = ";".join(dict.fromkeys(notes))
    used_buckets = len(active_levels)
    return BucketConfig(
        assignments=assignments,
        bucket_mode="quantile",
        q1_used=float(q1),
        q2_used=float(q2),
        threshold_t1=float(t1_used),
        threshold_t2=float(t2_used),
        counts=counts,
        shares=bucket_shares(counts, scores.size),
        used_buckets=used_buckets,
        active_levels=active_levels,
        note=note,
    )


def match_dets(gt_dets: List[GTDet], pred_dets: List[PredDet], iou_thresh: float) -> List[Tuple[int, int, float]]:
    if not gt_dets or not pred_dets:
        return []
    gt_boxes = np.asarray([g.box for g in gt_dets], dtype=float)
    pred_boxes = np.asarray([p.box for p in pred_dets], dtype=float)
    iou = iou_matrix(gt_boxes, pred_boxes)
    rr, cc = linear_sum_assignment(1.0 - iou)
    matches: List[Tuple[int, int, float]] = []
    for gi, pi in zip(rr.tolist(), cc.tolist()):
        i = float(iou[gi, pi])
        if i >= iou_thresh:
            matches.append((gi, pi, i))
    return matches


def resolve_method_dirs(args: argparse.Namespace) -> Dict[str, Path]:
    overrides = {
        "Base": args.pred_base,
        "+gating": args.pred_gating,
        "+traj": args.pred_traj,
        "+adaptive": args.pred_adaptive,
    }
    out: Dict[str, Path] = {}
    for method, subdir in METHOD_SPECS:
        path = overrides[method] if overrides[method] is not None else args.pred_root / subdir
        if not path.exists():
            raise FileNotFoundError(f"Prediction directory not found for {method}: {path}")
        out[method] = path
    return out


def build_frame_scores(
    seqs: List[str],
    seq_to_gt: Dict[str, List[GTDet]],
    seq_to_pred_base: Dict[str, List[PredDet]],
) -> Tuple[List[Tuple[str, int]], Dict[str, np.ndarray]]:
    frame_keys: List[Tuple[str, int]] = []
    occ_values: List[float] = []
    density_values: List[float] = []
    turn_values: List[float] = []
    low_conf_values: List[float] = []

    for seq in seqs:
        gt_rows = seq_to_gt[seq]
        if not gt_rows:
            continue

        frame_map: Dict[int, List[GTDet]] = {}
        for g in gt_rows:
            frame_map.setdefault(g.frame, []).append(g)

        occ_by_index: Dict[int, float] = {}
        for dets in frame_map.values():
            if len(dets) <= 1:
                for g in dets:
                    occ_by_index[g.index] = 0.0
                continue
            boxes = np.asarray([g.box for g in dets], dtype=float)
            iou = iou_matrix(boxes, boxes)
            np.fill_diagonal(iou, 0.0)
            neigh = (iou > 0.1).sum(axis=1).astype(float)
            maxov = iou.max(axis=1)
            score = neigh + maxov
            for i, g in enumerate(dets):
                occ_by_index[g.index] = float(score[i])

        tid_map: Dict[int, List[GTDet]] = {}
        for g in gt_rows:
            tid_map.setdefault(g.track_id, []).append(g)
        turn_by_index: Dict[int, float] = {}
        for dets in tid_map.values():
            dets = sorted(dets, key=lambda d: d.frame)
            if len(dets) < 3:
                continue
            centers = np.asarray([d.center for d in dets], dtype=float)
            diag = float(np.linalg.norm(np.max(centers, axis=0) - np.min(centers, axis=0)) + 1e-6)
            for i in range(1, len(dets) - 1):
                v1 = centers[i] - centers[i - 1]
                v2 = centers[i + 1] - centers[i]
                n1 = float(np.linalg.norm(v1))
                n2 = float(np.linalg.norm(v2))
                if n1 < 1e-6 or n2 < 1e-6:
                    angle = 0.0
                else:
                    cosv = float(np.dot(v1, v2) / (n1 * n2 + 1e-6))
                    cosv = max(-1.0, min(1.0, cosv))
                    angle = float(math.acos(cosv))
                accel = float(np.linalg.norm(v2 - v1) / diag)
                turn_by_index[dets[i].index] = angle + 0.5 * accel

        low_conf_by_index: Dict[int, float] = {}
        frame_pred: Dict[int, List[PredDet]] = {}
        for p in seq_to_pred_base[seq]:
            frame_pred.setdefault(p.frame, []).append(p)
        for frame, dets in frame_map.items():
            preds = frame_pred.get(frame, [])
            if not preds:
                for g in dets:
                    low_conf_by_index[g.index] = 0.0
                continue
            gboxes = np.asarray([g.box for g in dets], dtype=float)
            pboxes = np.asarray([p.box for p in preds], dtype=float)
            ious = iou_matrix(gboxes, pboxes)
            scores = np.asarray([p.score for p in preds], dtype=float)[None, :]
            weighted = ious * scores
            best = weighted.max(axis=1) if weighted.size else np.zeros((len(dets),), dtype=float)
            for i, g in enumerate(dets):
                low_conf_by_index[g.index] = float(best[i])

        for frame in sorted(frame_map.keys()):
            dets = frame_map[frame]
            if not dets:
                continue
            frame_keys.append((seq, frame))
            density_values.append(float(len(dets)))
            occ_values.append(float(np.mean([occ_by_index.get(g.index, 0.0) for g in dets])))
            turn_values.append(float(np.mean([turn_by_index.get(g.index, 0.0) for g in dets])))
            low_conf_values.append(float(np.mean([low_conf_by_index.get(g.index, 0.0) for g in dets])))

    out = {
        "occlusion": np.asarray(occ_values, dtype=float),
        "density": np.asarray(density_values, dtype=float),
        "turn": np.asarray(turn_values, dtype=float),
        "low_conf": np.asarray(low_conf_values, dtype=float),
    }
    return frame_keys, out


def group_gt_by_frame(gt_rows: List[GTDet]) -> Dict[int, List[GTDet]]:
    out: Dict[int, List[GTDet]] = {}
    for g in gt_rows:
        out.setdefault(g.frame, []).append(g)
    return out


def group_pred_by_frame(pred_rows: List[PredDet]) -> Dict[int, List[PredDet]]:
    out: Dict[int, List[PredDet]] = {}
    for p in pred_rows:
        out.setdefault(p.frame, []).append(p)
    return out


def build_selected_frames(
    frame_keys: List[Tuple[str, int]],
    assignments: np.ndarray,
) -> Dict[int, Dict[str, Set[int]]]:
    selected: Dict[int, Dict[str, Set[int]]] = {0: {}, 1: {}, 2: {}}
    for idx, (seq, frame) in enumerate(frame_keys):
        level = int(assignments[idx])
        if level not in selected:
            continue
        selected[level].setdefault(seq, set()).add(frame)
    return selected


def eval_bucket(
    seqs: List[str],
    seq_to_gt_frames: Dict[str, Dict[int, List[GTDet]]],
    seq_to_pred_frames: Dict[str, Dict[int, List[PredDet]]],
    selected_frames_by_seq: Dict[str, Set[int]],
    iou_thresh: float,
) -> Dict[str, float]:
    tp = 0
    fp = 0
    fn = 0
    idsw = 0
    count_errors: List[float] = []
    count_errors_sq: List[float] = []
    used_frames = 0
    used_sequences = 0

    for seq in seqs:
        chosen_frames = sorted(selected_frames_by_seq.get(seq, set()))
        if not chosen_frames:
            continue
        used_sequences += 1

        gt_by_frame = seq_to_gt_frames[seq]
        pred_by_frame = seq_to_pred_frames[seq]

        prev_match: Dict[int, int] = {}
        for frame in chosen_frames:
            gt_frame = gt_by_frame.get(frame, [])
            pred_frame = pred_by_frame.get(frame, [])
            used_frames += 1

            matches = match_dets(gt_frame, pred_frame, iou_thresh=iou_thresh)
            tp += len(matches)
            fn += max(0, len(gt_frame) - len(matches))
            fp += max(0, len(pred_frame) - len(matches))

            for gi, pi, _ in matches:
                gt_id = gt_frame[gi].track_id
                pred_id = pred_frame[pi].track_id
                if gt_id in prev_match and prev_match[gt_id] != pred_id:
                    idsw += 1
                prev_match[gt_id] = pred_id

            err = float(len(pred_frame) - len(gt_frame))
            count_errors.append(abs(err))
            count_errors_sq.append(err * err)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mae = float(np.mean(count_errors)) if count_errors else 0.0
    rmse = float(np.sqrt(np.mean(count_errors_sq))) if count_errors_sq else 0.0

    return {
        "num_sequences": float(used_sequences),
        "num_frames": float(used_frames),
        "TP": float(tp),
        "FP": float(fp),
        "FN": float(fn),
        "Precision": precision * 100.0,
        "Recall": recall * 100.0,
        "F1": f1 * 100.0,
        "IDSW": float(idsw),
        "CountMAE": mae,
        "CountRMSE": rmse,
    }


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "bucket_type",
        "bucket",
        "method",
        "bucket_mode",
        "used_buckets",
        "q33",
        "q66",
        "q1_used",
        "q2_used",
        "threshold_t1",
        "threshold_t2",
        "num_sequences",
        "num_frames",
        "bucket_size",
        "num_frames_bucket",
        "bucket_share",
        "note",
        "TP",
        "FP",
        "FN",
        "Precision",
        "Recall",
        "F1",
        "IDSW",
        "CountMAE",
        "CountRMSE",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing stratified CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def apply_plot_style() -> None:
    plt.rcParams.update(PLOT_STYLE)


def shorten_text(text: str, max_chars: int = 54) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def make_plot(rows: List[Dict[str, str]], out_path: Path) -> None:
    apply_plot_style()
    row_map: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for row in rows:
        row_map[(row["bucket_type"], row["bucket"], row["method"])] = row

    methods = [m for m, _ in METHOD_SPECS]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8), constrained_layout=True)
    axes_list = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
    width = 0.18

    for ax, criterion in zip(axes_list, CRITERIA):
        levels = [level for level in LEVELS if any((criterion, level, method) in row_map for method in methods)]
        if not levels:
            levels = LEVELS[:]
        x = np.arange(len(levels))
        for m_idx, method in enumerate(methods):
            vals = []
            for level in levels:
                row = row_map.get((criterion, level, method))
                vals.append(float(row["F1"]) if row else np.nan)

            offset = (m_idx - (len(methods) - 1) * 0.5) * width
            ax.bar(x + offset, vals, width=width, label=method, linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(levels)
        ax.set_title(f"{CRITERION_TITLES.get(criterion, criterion)}")
        ax.set_ylabel("F1")
        ax.set_ylim(0.0, 110.0)
        ax.grid(axis="y", alpha=0.18)

    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Stratified bucket metrics (quantile metadata reported in CSV note column)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **PLOT_SAVE_KWARGS)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_run_config(args)
    input_csv = args.input_csv if args.input_csv is not None else args.output_csv
    if args.plot_only:
        rows = read_rows(input_csv)
        make_plot(rows, args.plot_path)
        print(f"Rendered stratified plot: {args.plot_path}")
        return

    split_dir = args.mot_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Prepared split not found: {split_dir}. "
            "Please run scripts/prepare_mft25.py first."
        )
    method_dirs = resolve_method_dirs(args)
    sequences = read_seqmap(split_dir / "seqmaps" / f"{args.split}.txt")
    if not sequences:
        raise RuntimeError("No sequence in seqmap.")

    seq_to_gt: Dict[str, List[GTDet]] = {}
    seq_to_gt_frames: Dict[str, Dict[int, List[GTDet]]] = {}
    seq_to_pred: Dict[str, Dict[str, List[PredDet]]] = {m: {} for m, _ in METHOD_SPECS}
    seq_to_pred_frames: Dict[str, Dict[str, Dict[int, List[PredDet]]]] = {m: {} for m, _ in METHOD_SPECS}
    all_gt: List[GTDet] = []
    start_idx = 0

    for seq in sequences:
        seqinfo = split_dir / seq / "seqinfo.ini"
        gt_path = split_dir / seq / "gt" / "gt.txt"
        if not seqinfo.exists():
            raise FileNotFoundError(f"Missing seqinfo.ini: {seqinfo}")
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing gt.txt: {gt_path}")
        seq_len = read_seq_length(seqinfo)
        used_frames = seq_len if args.max_frames <= 0 else min(seq_len, args.max_frames)
        used_frames = max(1, used_frames)

        gt_rows = load_gt(gt_path, used_frames, start_idx)
        start_idx += len(gt_rows)
        seq_to_gt[seq] = gt_rows
        seq_to_gt_frames[seq] = group_gt_by_frame(gt_rows)
        all_gt.extend(gt_rows)

        for method, _ in METHOD_SPECS:
            pred_path = method_dirs[method] / f"{seq}.txt"
            pred_rows = load_pred(pred_path, used_frames)
            seq_to_pred[method][seq] = pred_rows
            seq_to_pred_frames[method][seq] = group_pred_by_frame(pred_rows)

        print(f"[stratified] loaded {seq}: gt={len(gt_rows)}, frames={used_frames}")

    if not all_gt:
        raise RuntimeError("No GT rows loaded.")

    base_pred_by_seq = {seq: seq_to_pred["Base"][seq] for seq in sequences}
    frame_keys, frame_scores = build_frame_scores(sequences, seq_to_gt, base_pred_by_seq)
    if not frame_keys:
        raise RuntimeError(
            f"No available frames for stratified bucketing in split={args.split}. "
            "Please check prepared GT/pred data."
        )

    bucket_min_samples = int(args.bucket_min_samples)
    if args.min_bucket_frames is not None:
        bucket_min_samples = int(args.min_bucket_frames)
    bucket_min_samples = max(1, bucket_min_samples)

    bucket_configs: Dict[str, BucketConfig] = {}
    selected_frames: Dict[str, Dict[int, Dict[str, Set[int]]]] = {}
    for criterion in CRITERIA:
        scores = frame_scores[criterion]
        if args.bucket_mode == "fixed":
            cfg = build_fixed_bucket_config(scores)
        else:
            cfg = build_quantile_bucket_config(
                scores=scores,
                q1=args.q1,
                q2=args.q2,
                min_bucket_samples=bucket_min_samples,
            )
        bucket_configs[criterion] = cfg
        selected_frames[criterion] = build_selected_frames(frame_keys, cfg.assignments)
        print(
            f"[stratified][coverage] {criterion}: mode={cfg.bucket_mode} "
            f"q1={cfg.q1_used:.3f} q2={cfg.q2_used:.3f} "
            f"t1={cfg.threshold_t1:.6f} t2={cfg.threshold_t2:.6f} "
            f"frames(low/mid/high)={cfg.counts['low']}/{cfg.counts['mid']}/{cfg.counts['high']} "
            f"share={cfg.shares['low']:.3f}/{cfg.shares['mid']:.3f}/{cfg.shares['high']:.3f} "
            f"used_buckets={cfg.used_buckets} active={','.join(cfg.active_levels)} note={cfg.note}"
        )

    rows: List[Dict[str, str]] = []
    for criterion in CRITERIA:
        cfg = bucket_configs[criterion]
        for level in cfg.active_levels:
            level_idx = LEVELS.index(level)
            for method, _ in METHOD_SPECS:
                m = eval_bucket(
                    seqs=sequences,
                    seq_to_gt_frames=seq_to_gt_frames,
                    seq_to_pred_frames=seq_to_pred_frames[method],
                    selected_frames_by_seq=selected_frames[criterion][level_idx],
                    iou_thresh=args.iou_thresh,
                )
                if int(round(m["num_frames"])) <= 0:
                    print(
                        f"[warn][stratified] skip empty row {criterion}-{level} {method}; "
                        f"note={cfg.note}"
                    )
                    continue
                row = {
                    "split": args.split,
                    "bucket_type": criterion,
                    "bucket": level,
                    "method": method,
                    "bucket_mode": cfg.bucket_mode,
                    "used_buckets": f"{cfg.used_buckets:d}",
                    "q33": f"{cfg.threshold_t1:.6f}",
                    "q66": f"{cfg.threshold_t2:.6f}",
                    "q1_used": f"{cfg.q1_used:.6f}",
                    "q2_used": f"{cfg.q2_used:.6f}",
                    "threshold_t1": f"{cfg.threshold_t1:.6f}",
                    "threshold_t2": f"{cfg.threshold_t2:.6f}",
                    "num_sequences": f"{m['num_sequences']:.0f}",
                    "num_frames": f"{m['num_frames']:.0f}",
                    "bucket_size": f"{cfg.counts[level]:d}",
                    "num_frames_bucket": f"{cfg.counts[level]:d}",
                    "bucket_share": f"{cfg.shares[level]:.6f}",
                    "note": cfg.note,
                    "TP": f"{m['TP']:.0f}",
                    "FP": f"{m['FP']:.0f}",
                    "FN": f"{m['FN']:.0f}",
                    "Precision": f"{m['Precision']:.3f}",
                    "Recall": f"{m['Recall']:.3f}",
                    "F1": f"{m['F1']:.3f}",
                    "IDSW": f"{m['IDSW']:.0f}",
                    "CountMAE": f"{m['CountMAE']:.3f}",
                    "CountRMSE": f"{m['CountRMSE']:.3f}",
                }
                rows.append(row)
                print(
                    f"[stratified] {criterion}-{level} {method}: "
                    f"F1={row['F1']} CountMAE={row['CountMAE']} IDSW={row['IDSW']} "
                    f"frames={row['num_frames_bucket']} share={row['bucket_share']}"
                )

    # Sanity check: no empty bucket rows and explicit warning for degraded buckets.
    for criterion in CRITERIA:
        cfg = bucket_configs[criterion]
        missing_levels = [level for level in LEVELS if level not in cfg.active_levels]
        zero_levels = [level for level in cfg.active_levels if cfg.counts.get(level, 0) <= 0]
        status = "OK"
        if zero_levels:
            status = "WARN_ZERO_BUCKET"
        elif missing_levels:
            status = "WARN_MERGED"
        print(
            f"[sanity] {criterion}: status={status} "
            f"counts(low/mid/high)={cfg.counts['low']}/{cfg.counts['mid']}/{cfg.counts['high']} "
            f"active={cfg.active_levels} note={cfg.note}"
        )
        if zero_levels:
            print(
                f"[WARN][sanity] {criterion}: zero-count active bucket(s)={zero_levels}; "
                f"note={cfg.note}"
            )
        if missing_levels:
            print(
                f"[sanity] {criterion}: merged/degraded buckets, active={cfg.active_levels}, "
                f"missing={missing_levels}, note={cfg.note}"
            )

    write_csv(args.output_csv, rows)
    make_plot(rows, args.plot_path)
    print(f"Saved stratified CSV: {args.output_csv}")
    print(f"Saved stratified plot: {args.plot_path}")


if __name__ == "__main__":
    main()
