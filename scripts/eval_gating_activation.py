#!/usr/bin/env python
"""Evaluate gating activation diagnostics from real tracker runs."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np

import run_baseline_sort as rbs


FIXED_SPLIT = "val_half"
FIXED_MAX_FRAMES = 1000
FIXED_DROP_RATE = 0.2
FIXED_JITTER = 0.02
FIXED_SEED = 0
FIXED_THRESHOLDS = [1000.0, 2000.0, 4000.0]
FIXED_SEQS = ["BT-001", "BT-003", "BT-005", "MSK-002", "PF-001", "SN-001", "SN-013", "SN-015"]

PLOT_STYLE = {
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
}
PLOT_SAVE_KWARGS = {"dpi": 300, "bbox_inches": "tight", "pad_inches": 0.02}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gating activation diagnostics.")
    parser.add_argument("--mot-root", type=Path, default=Path("data/mft25_mot_full"))
    parser.add_argument("--split", default=FIXED_SPLIT)
    parser.add_argument("--seqs", nargs="+", default=FIXED_SEQS)
    parser.add_argument("--max-frames", type=int, default=FIXED_MAX_FRAMES)
    parser.add_argument("--drop-rate", type=float, default=FIXED_DROP_RATE)
    parser.add_argument("--jitter", type=float, default=FIXED_JITTER)
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    parser.add_argument("--thresholds", nargs="+", type=float, default=FIXED_THRESHOLDS)
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--traj-window", type=int, default=16)
    parser.add_argument("--events-out", type=Path, default=Path("results/main_val/gating/gating_events.csv"))
    parser.add_argument("--trigger-out", type=Path, default=Path("results/main_val/gating/gating_trigger_rate.csv"))
    parser.add_argument("--event-out", type=Path, default=Path("results/main_val/gating/gating_event_length.csv"))
    parser.add_argument("--cdf-out", type=Path, default=Path("results/main_val/gating/gating_score_cdf.csv"))
    parser.add_argument("--fig-trigger", type=Path, default=Path("results/fig_gating_trigger_rate.pdf"))
    parser.add_argument("--fig-event", type=Path, default=Path("results/fig_gating_event_length.pdf"))
    parser.add_argument("--fig-cdf", type=Path, default=Path("results/fig_gating_score_cdf.pdf"))
    parser.add_argument("--paper-fig-dir", type=Path, default=Path("paper/cea_draft/figures"))
    return parser.parse_args()


def choose_det_path(seq_dir: Path) -> Path:
    det = seq_dir / "det" / "det.txt"
    gt = seq_dir / "gt" / "gt.txt"
    if det.exists():
        return det
    if gt.exists():
        return gt
    raise FileNotFoundError(f"No detection source found in {seq_dir}")


def event_lengths(binary_seq: Iterable[int]) -> List[int]:
    out: List[int] = []
    cur = 0
    for v in binary_seq:
        if int(v) == 1:
            cur += 1
        elif cur > 0:
            out.append(cur)
            cur = 0
    if cur > 0:
        out.append(cur)
    return out


def event_segments(binary_seq: Iterable[int]) -> List[Tuple[int, int, int]]:
    segments: List[Tuple[int, int, int]] = []
    start = -1
    length = 0
    for idx, v in enumerate(binary_seq, start=1):
        if int(v) == 1:
            if start < 0:
                start = idx
            length += 1
        elif start >= 0:
            end = idx - 1
            segments.append((start, end, length))
            start = -1
            length = 0
    if start >= 0 and length > 0:
        segments.append((start, start + length - 1, length))
    return segments


def tracker_step_with_diagnostics(
    tracker: rbs.SortTracker,
    detections_xywh: List[np.ndarray],
    frame_idx: int,
) -> Tuple[dict[str, int], np.ndarray]:
    tracks_before = len(tracker.tracks)
    predictions: List[np.ndarray] = []
    gating_stats: List[Tuple[np.ndarray, np.ndarray]] = []
    for track in tracker.tracks:
        predictions.append(track.predict())
        gating_stats.append(track.projected_center_stats())

    det_centers = np.asarray(
        [[det[0] + det[2] / 2.0, det[1] + det[3] / 2.0] for det in detections_xywh],
        dtype=float,
    )
    if det_centers.size == 0:
        det_centers = np.zeros((0, 2), dtype=float)

    d_maha = rbs.mahalanobis_distance_matrix(det_centers=det_centers, gating_stats=gating_stats)
    traj_cost = rbs.trajectory_cost_matrix(
        det_centers=det_centers,
        tracks=tracker.tracks,
        traj_model=None,
        window_size=tracker.traj_window,
        norm_w=tracker.norm_w,
        norm_h=tracker.norm_h,
    )
    gamma_by_track = np.asarray([tracker._gamma_for_track(track) for track in tracker.tracks], dtype=float)

    matches, unmatched_det_idx, _ = rbs.associate_detections_to_tracks(
        detections_xywh=detections_xywh,
        predictions_xywh=predictions,
        iou_thresh=tracker.iou_thresh,
        alpha=tracker.alpha,
        beta=tracker.beta,
        gamma_by_track=gamma_by_track,
        d_maha=d_maha,
        traj_cost=traj_cost,
        gating_enabled=tracker.gating_enabled,
        gating_threshold=tracker.gating_threshold,
    )
    matched_count = len(matches)
    unmatched_tracks = max(0, tracks_before - matched_count)
    new_tracks = len(unmatched_det_idx)

    for det_i, trk_i in matches:
        tracker.tracks[trk_i].update(detections_xywh[det_i])
    for det_i in unmatched_det_idx:
        tracker._new_track(detections_xywh[det_i])

    before_prune = len(tracker.tracks)
    tracker.tracks = [track for track in tracker.tracks if track.time_since_update <= tracker.max_age]
    removed_tracks = max(0, before_prune - len(tracker.tracks))

    outputs = 0
    for track in tracker.tracks:
        if track.time_since_update != 0:
            continue
        if track.hits < tracker.min_hits and frame_idx > tracker.min_hits:
            continue
        outputs += 1

    stats = {
        "frame": int(frame_idx),
        "detections": int(len(detections_xywh)),
        "tracks_before": int(tracks_before),
        "matches": int(matched_count),
        "lost": int(unmatched_tracks),
        "new_tracks": int(new_tracks),
        "removed_tracks": int(removed_tracks),
        "active_tracks": int(len(tracker.tracks)),
        "outputs": int(outputs),
    }
    return stats, d_maha


def run_for_threshold(args: argparse.Namespace, threshold: float) -> Dict[str, object]:
    split_dir = args.mot_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    total_frames = 0
    total_triggered_frames = 0
    total_pairs = 0
    total_masked_pairs = 0
    score_values: List[float] = []
    seq_trigger_rates: List[float] = []
    all_events: List[int] = []
    all_event_rows: List[Dict[str, object]] = []

    for seq_idx, seq in enumerate(args.seqs):
        seq_dir = split_dir / seq
        seq_len_raw, im_w, im_h = rbs.read_seq_info(seq_dir / "seqinfo.ini")
        seq_length = min(seq_len_raw, args.max_frames) if args.max_frames > 0 else seq_len_raw
        det_path = choose_det_path(seq_dir)
        detections = rbs.load_detections_from_mot(det_path, seq_length)
        rng = np.random.default_rng(args.seed + seq_idx)

        tracker = rbs.SortTracker(
            iou_thresh=args.iou_thresh,
            min_hits=args.min_hits,
            max_age=args.max_age,
            gating_enabled=True,
            gating_threshold=threshold,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            traj_enabled=False,
            adaptive_gamma=False,
            adaptive_gamma_boost=1.5,
            adaptive_gamma_min=0.5,
            adaptive_gamma_max=2.0,
            traj_window=args.traj_window,
            traj_model=None,
            norm_w=float(im_w),
            norm_h=float(im_h),
        )

        triggered_flags: List[int] = []
        for frame in range(1, seq_length + 1):
            frame_raw = detections.get(frame, [])
            frame_dets = rbs.degrade_detections(
                detections_xywh=frame_raw,
                drop_rate=args.drop_rate,
                jitter=args.jitter,
                image_w=im_w,
                image_h=im_h,
                rng=rng,
            )
            _, d_maha = tracker_step_with_diagnostics(tracker, frame_dets, frame)
            if d_maha.size > 0:
                vals = d_maha.reshape(-1)
                score_values.extend(vals.tolist())
                masked = int(np.sum(vals > threshold))
                total_masked_pairs += masked
                total_pairs += int(vals.size)
                frame_trigger = 1 if masked > 0 else 0
            else:
                frame_trigger = 0
            triggered_flags.append(frame_trigger)

        seq_events = event_lengths(triggered_flags)
        all_events.extend(seq_events)
        segs = event_segments(triggered_flags)
        for event_idx, (start_frame, end_frame, event_len) in enumerate(segs, start=1):
            all_event_rows.append(
                {
                    "threshold": float(threshold),
                    "seq": seq,
                    "event_idx": int(event_idx),
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "event_length": int(event_len),
                }
            )
        seq_frames = len(triggered_flags)
        seq_trigger = float(sum(triggered_flags) / max(1, seq_frames))
        seq_trigger_rates.append(seq_trigger)
        total_frames += seq_frames
        total_triggered_frames += int(sum(triggered_flags))

    frame_trigger_rate = float(total_triggered_frames / max(1, total_frames))
    seq_trigger_rate = float(np.mean(seq_trigger_rates)) if seq_trigger_rates else 0.0
    masked_pair_rate = float(total_masked_pairs / max(1, total_pairs))

    return {
        "threshold": float(threshold),
        "frame_trigger_rate": frame_trigger_rate,
        "sequence_trigger_rate": seq_trigger_rate,
        "masked_pair_rate": masked_pair_rate,
        "events": all_events,
        "event_rows": all_event_rows,
        "scores": np.asarray(score_values, dtype=float),
        "total_frames": total_frames,
    }


def write_events_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["threshold", "seq", "event_idx", "start_frame", "end_frame", "event_length"],
        )
        writer.writeheader()
        for row in rows:
            threshold = int(float(row["threshold"]))
            event_rows: List[Dict[str, object]] = list(row["event_rows"])  # type: ignore[arg-type]
            if not event_rows:
                writer.writerow(
                    {
                        "threshold": threshold,
                        "seq": "",
                        "event_idx": 0,
                        "start_frame": 0,
                        "end_frame": 0,
                        "event_length": 0,
                    }
                )
                continue
            for er in event_rows:
                writer.writerow(
                    {
                        "threshold": threshold,
                        "seq": str(er["seq"]),
                        "event_idx": int(er["event_idx"]),
                        "start_frame": int(er["start_frame"]),
                        "end_frame": int(er["end_frame"]),
                        "event_length": int(er["event_length"]),
                    }
                )


def write_trigger_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold",
                "trigger_rate_frame",
                "trigger_rate_sequence",
                "masked_pair_rate",
                "total_frames",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "threshold": f"{float(row['threshold']):.0f}",
                    "trigger_rate_frame": f"{float(row['frame_trigger_rate']):.6f}",
                    "trigger_rate_sequence": f"{float(row['sequence_trigger_rate']):.6f}",
                    "masked_pair_rate": f"{float(row['masked_pair_rate']):.6f}",
                    "total_frames": int(row["total_frames"]),
                }
            )


def write_event_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "event_length", "count"])
        writer.writeheader()
        for row in rows:
            threshold = f"{float(row['threshold']):.0f}"
            events: List[int] = list(row["events"])  # type: ignore[arg-type]
            if not events:
                writer.writerow({"threshold": threshold, "event_length": 0, "count": 0})
                continue
            lengths, counts = np.unique(np.asarray(events, dtype=int), return_counts=True)
            for length, count in zip(lengths.tolist(), counts.tolist()):
                writer.writerow({"threshold": threshold, "event_length": int(length), "count": int(count)})


def write_cdf_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    quantiles = np.linspace(0.0, 1.0, 101)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "quantile", "score"])
        writer.writeheader()
        for row in rows:
            threshold = f"{float(row['threshold']):.0f}"
            scores: np.ndarray = row["scores"]  # type: ignore[assignment]
            if scores.size == 0:
                for q in quantiles:
                    writer.writerow({"threshold": threshold, "quantile": f"{q:.2f}", "score": "0.000000"})
                continue
            vals = np.quantile(scores, quantiles)
            for q, s in zip(quantiles.tolist(), vals.tolist()):
                writer.writerow({"threshold": threshold, "quantile": f"{q:.2f}", "score": f"{float(s):.6f}"})


def plot_trigger_rates(rows: List[Dict[str, object]], out_path: Path) -> None:
    plt.rcParams.update(PLOT_STYLE)
    x = np.arange(len(rows), dtype=float)
    labels = [f"{int(float(r['threshold']))}" for r in rows]
    frame_rates = [float(r["frame_trigger_rate"]) for r in rows]
    seq_rates = [float(r["sequence_trigger_rate"]) for r in rows]

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    w = 0.35
    ax.bar(x - w / 2, frame_rates, width=w, label="frame trigger rate")
    ax.bar(x + w / 2, seq_rates, width=w, label="sequence trigger rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"$\tau_g$")
    ax.set_ylabel("rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.4)
    ax.legend(frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **PLOT_SAVE_KWARGS)
    plt.close(fig)


def plot_event_lengths(rows: List[Dict[str, object]], out_path: Path) -> None:
    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for row in rows:
        threshold = int(float(row["threshold"]))
        events = np.asarray(list(row["events"]), dtype=int)
        if events.size == 0:
            continue
        lengths, counts = np.unique(events, return_counts=True)
        ax.step(lengths, counts, where="mid", label=fr"$\tau_g={threshold}$")
    ax.set_xlabel("event length (frames)")
    ax.set_ylabel("count")
    ax.grid(alpha=0.4)
    ax.legend(frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **PLOT_SAVE_KWARGS)
    plt.close(fig)


def plot_score_cdf(rows: List[Dict[str, object]], out_path: Path) -> None:
    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for row in rows:
        threshold = float(row["threshold"])
        scores: np.ndarray = row["scores"]  # type: ignore[assignment]
        if scores.size == 0:
            continue
        xs = np.sort(scores)
        ys = np.linspace(0.0, 1.0, xs.size)
        ax.plot(xs, ys, label=fr"CDF ($\tau_g={int(threshold)}$)")
        ax.axvline(threshold, linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel(r"$s^{\mathrm{gate}}_{ij}$ (squared Mahalanobis)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.4)
    ax.legend(frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **PLOT_SAVE_KWARGS)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.mot_root.exists():
        raise FileNotFoundError(f"mot_root not found: {args.mot_root}")
    rows = [run_for_threshold(args, thr) for thr in args.thresholds]
    rows = sorted(rows, key=lambda r: float(r["threshold"]))

    write_events_csv(args.events_out, rows)
    write_trigger_csv(args.trigger_out, rows)
    write_event_csv(args.event_out, rows)
    write_cdf_csv(args.cdf_out, rows)
    plot_trigger_rates(rows, args.fig_trigger)
    plot_event_lengths(rows, args.fig_event)
    plot_score_cdf(rows, args.fig_cdf)

    args.paper_fig_dir.mkdir(parents=True, exist_ok=True)
    for fig in [args.fig_trigger, args.fig_event, args.fig_cdf]:
        shutil.copy2(fig, args.paper_fig_dir / fig.name)

    print(f"wrote {args.events_out}")
    print(f"wrote {args.trigger_out}")
    print(f"wrote {args.event_out}")
    print(f"wrote {args.cdf_out}")
    print(f"wrote {args.fig_trigger}")
    print(f"wrote {args.fig_event}")
    print(f"wrote {args.fig_cdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
