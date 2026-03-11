#!/usr/bin/env python
"""Run Scope B (decode + detector-source prep + tracking + write) profiling.

This script executes a reproducible Scope B runtime protocol and exports CSV rows
with the required schema fields:
  fps_e2e, mem_peak_mb_e2e, cpu_norm_e2e, decode_time, detector_time,
  tracking_time, write_time, detector_name, input_resolution
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import psutil
from PIL import Image

import run_baseline_sort as rbs


FIXED_SPLIT = "val_half"
FIXED_MAX_FRAMES = 1000
FIXED_DROP_RATE = 0.2
FIXED_JITTER = 0.02
FIXED_SEED = 0
FIXED_GATING_THRESH = 2000.0
FIXED_SEQS = ["BT-001", "BT-003", "BT-005", "MSK-002", "PF-001", "SN-001", "SN-013", "SN-015"]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    key: str
    cfg: Mapping[str, str]


METHODS: List[MethodSpec] = [
    MethodSpec(
        name="Base",
        key="base",
        cfg={"gating": "off", "traj": "off", "beta": "0.0", "gamma": "0.0", "adaptive_gamma": "off", "min_hits": "3"},
    ),
    MethodSpec(
        name="+gating",
        key="gating",
        cfg={"gating": "on", "traj": "off", "beta": "0.02", "gamma": "0.0", "adaptive_gamma": "off", "min_hits": "3"},
    ),
    MethodSpec(
        name="+traj",
        key="traj",
        cfg={"gating": "on", "traj": "on", "beta": "0.02", "gamma": "0.5", "adaptive_gamma": "off", "min_hits": "3"},
    ),
    MethodSpec(
        name="+adaptive",
        key="adaptive",
        cfg={"gating": "on", "traj": "on", "beta": "0.02", "gamma": "0.5", "adaptive_gamma": "on", "min_hits": "3"},
    ),
]


class ProcessSampler:
    """Sample cpu% and rss for the current process during one run."""

    def __init__(self, interval: float = 0.2) -> None:
        self.interval = float(interval)
        self.proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.cpu_samples: List[float] = []
        self.mem_peak_mb: float = 0.0

    def _loop(self) -> None:
        try:
            self.proc.cpu_percent(interval=None)
        except psutil.Error:
            pass
        while not self._stop.is_set():
            try:
                cpu_val = float(self.proc.cpu_percent(interval=None))
                rss = float(self.proc.memory_info().rss / (1024.0 * 1024.0))
            except psutil.Error:
                cpu_val = 0.0
                rss = 0.0
            self.cpu_samples.append(cpu_val)
            self.mem_peak_mb = max(self.mem_peak_mb, rss)
            self._stop.wait(self.interval)

    def __enter__(self) -> "ProcessSampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval * 5.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scope B end-to-end runtime profiler.")
    parser.add_argument("--mot-root", type=Path, default=Path("data/mft25_mot_full"))
    parser.add_argument("--split", default=FIXED_SPLIT)
    parser.add_argument("--seqs", nargs="+", default=FIXED_SEQS)
    parser.add_argument("--max-frames", type=int, default=FIXED_MAX_FRAMES)
    parser.add_argument("--drop-rate", type=float, default=FIXED_DROP_RATE)
    parser.add_argument("--jitter", type=float, default=FIXED_JITTER)
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    parser.add_argument("--det-source", choices=["auto", "det", "gt"], default="auto")
    parser.add_argument("--gating-thresh", type=float, default=FIXED_GATING_THRESH)
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--traj-window", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--traj-encoder", type=Path, default=Path("runs/traj_encoder/traj_encoder.pt"))
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--repeat-runs", type=int, default=2)
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("--out-csv", type=Path, default=Path("results/main_val/runtime/scopeb_profile.csv"))
    parser.add_argument("--pred-root", type=Path, default=Path("results/main_val/runtime_scopeb/pred"))
    parser.add_argument("--paper-fig-dir", type=Path, default=Path("paper/cea_draft/figures"))
    return parser.parse_args()


def read_image_paths(seq_dir: Path, frame_limit: int) -> List[Path]:
    img_dir = seq_dir / "img1"
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing img1 directory: {img_dir}")
    paths = sorted(img_dir.glob("*"))
    if frame_limit > 0:
        paths = paths[:frame_limit]
    return [p for p in paths if p.is_file()]


def decode_images(image_paths: Iterable[Path]) -> int:
    count = 0
    for path in image_paths:
        with Image.open(path) as img:
            img.load()
        count += 1
    return count


def choose_det_path(seq_dir: Path, det_source: str) -> tuple[Path, str]:
    gt_path = seq_dir / "gt" / "gt.txt"
    det_path = seq_dir / "det" / "det.txt"
    if det_source == "det":
        chosen = det_path
        tag = "det_txt_passthrough"
    elif det_source == "gt":
        chosen = gt_path
        tag = "gt_txt_passthrough"
    else:
        if det_path.exists():
            chosen = det_path
            tag = "det_txt_passthrough"
        else:
            chosen = gt_path
            tag = "gt_txt_passthrough"
    if not chosen.exists():
        raise FileNotFoundError(f"Detection source not found: {chosen}")
    return chosen, tag


def prepare_detector_output(
    chosen_det: Path,
    seq_length: int,
    drop_rate: float,
    jitter: float,
    im_w: int,
    im_h: int,
    rng_seed: int,
) -> Dict[int, List[np.ndarray]]:
    detections = rbs.load_detections_from_mot(chosen_det, seq_length)
    rng = np.random.default_rng(int(rng_seed))
    det_out: Dict[int, List[np.ndarray]] = {}
    for frame in range(1, seq_length + 1):
        raw = detections.get(frame, [])
        det_out[frame] = rbs.degrade_detections(
            detections_xywh=raw,
            drop_rate=drop_rate,
            jitter=jitter,
            image_w=im_w,
            image_h=im_h,
            rng=rng,
        )
    return det_out


def run_tracking_and_collect_outputs(
    det_by_frame: Mapping[int, List[np.ndarray]],
    seq_length: int,
    method: MethodSpec,
    im_w: int,
    im_h: int,
    args: argparse.Namespace,
    traj_model: rbs.TrajCostModel | None,
) -> List[Tuple[int, int, float, float, float, float, float]]:
    tracker = rbs.SortTracker(
        iou_thresh=args.iou_thresh,
        min_hits=int(method.cfg["min_hits"]),
        max_age=args.max_age,
        gating_enabled=(method.cfg["gating"] == "on"),
        gating_threshold=args.gating_thresh,
        alpha=args.alpha,
        beta=float(method.cfg["beta"]),
        gamma=float(method.cfg["gamma"]),
        traj_enabled=(method.cfg["traj"] == "on"),
        adaptive_gamma=(method.cfg["adaptive_gamma"] == "on"),
        adaptive_gamma_boost=1.5,
        adaptive_gamma_min=0.5,
        adaptive_gamma_max=2.0,
        traj_window=args.traj_window,
        traj_model=traj_model,
        norm_w=float(im_w),
        norm_h=float(im_h),
    )
    outputs: List[Tuple[int, int, float, float, float, float, float]] = []
    for frame in range(1, seq_length + 1):
        frame_dets = list(det_by_frame.get(frame, []))
        frame_outputs, _ = tracker.step(frame_dets, frame)
        outputs.extend(frame_outputs)
    return outputs


def write_outputs(
    seq_name: str,
    out_dir: Path,
    outputs: Iterable[Tuple[int, int, float, float, float, float, float]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{seq_name}.txt"
    with out_file.open("w", encoding="utf-8", newline="\n") as f:
        for frame, track_id, x, y, w, h, score in outputs:
            f.write(f"{frame},{track_id},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{score:.3f},1,1\n")


def mean_std(values: List[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.fmean(values)), float(statistics.stdev(values))


def run_one(
    args: argparse.Namespace,
    method: MethodSpec,
    run_idx: int,
    traj_model: rbs.TrajCostModel | None,
) -> Dict[str, float | str]:
    split_dir = args.mot_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory missing: {split_dir}")

    decode_t = 0.0
    detector_t = 0.0
    tracking_t = 0.0
    write_t = 0.0
    total_frames = 0
    detector_tags: List[str] = []
    res_tags: List[str] = []

    pred_dir = args.pred_root / method.key / f"run_{run_idx:02d}"
    if pred_dir.exists():
        for old in pred_dir.glob("*.txt"):
            old.unlink()

    with ProcessSampler(interval=args.sample_interval) as sampler:
        t0 = time.perf_counter()
        for seq_idx, seq in enumerate(args.seqs):
            seq_dir = split_dir / seq
            seq_len_raw, im_w, im_h = rbs.read_seq_info(seq_dir / "seqinfo.ini")
            seq_length = min(seq_len_raw, args.max_frames) if args.max_frames > 0 else seq_len_raw
            res_tags.append(f"{im_w}x{im_h}")

            img_paths = read_image_paths(seq_dir, seq_length)
            if len(img_paths) < seq_length:
                seq_length = len(img_paths)
            total_frames += seq_length
            td0 = time.perf_counter()
            decoded = decode_images(img_paths)
            td1 = time.perf_counter()
            if decoded != len(img_paths):
                raise RuntimeError(f"{seq}: decoded frames {decoded} != available {len(img_paths)}")
            decode_t += td1 - td0

            det_path, det_tag = choose_det_path(seq_dir, args.det_source)
            detector_tags.append(det_tag)
            te0 = time.perf_counter()
            det_by_frame = prepare_detector_output(
                chosen_det=det_path,
                seq_length=seq_length,
                drop_rate=args.drop_rate,
                jitter=args.jitter,
                im_w=im_w,
                im_h=im_h,
                rng_seed=args.seed + seq_idx,
            )
            te1 = time.perf_counter()
            detector_t += te1 - te0

            tt0 = time.perf_counter()
            outputs = run_tracking_and_collect_outputs(
                det_by_frame=det_by_frame,
                seq_length=seq_length,
                method=method,
                im_w=im_w,
                im_h=im_h,
                args=args,
                traj_model=traj_model,
            )
            tt1 = time.perf_counter()
            tracking_t += tt1 - tt0

            tw0 = time.perf_counter()
            write_outputs(seq, pred_dir, outputs)
            tw1 = time.perf_counter()
            write_t += tw1 - tw0
        t1 = time.perf_counter()
        elapsed = t1 - t0

    cpu_count = max(1, int(psutil.cpu_count() or 1))
    cpu_mean_1core = float(statistics.fmean(sampler.cpu_samples)) if sampler.cpu_samples else 0.0
    cpu_norm = cpu_mean_1core / cpu_count
    fps = float(total_frames / max(elapsed, 1e-9))
    detector_name = detector_tags[0] if len(set(detector_tags)) == 1 else "mixed_txt_source"
    input_res = res_tags[0] if len(set(res_tags)) == 1 else "mixed"

    return {
        "row_type": "repeat",
        "method": method.name,
        "run_idx": run_idx,
        "split": args.split,
        "total_frames": total_frames,
        "fps_e2e": fps,
        "mem_peak_mb_e2e": sampler.mem_peak_mb,
        "cpu_norm_e2e": cpu_norm,
        "decode_time": decode_t,
        "detector_time": detector_t,
        "tracking_time": tracking_t,
        "write_time": write_t,
        "elapsed_e2e_sec": elapsed,
        "detector_name": detector_name,
        "input_resolution": input_res,
    }


def write_csv(path: Path, rows: List[Dict[str, float | str]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "row_type",
        "method",
        "run_idx",
        "split",
        "warmup_iters",
        "repeat_runs",
        "total_frames",
        "fps_e2e",
        "mem_peak_mb_e2e",
        "cpu_norm_e2e",
        "decode_time",
        "detector_time",
        "tracking_time",
        "write_time",
        "elapsed_e2e_sec",
        "detector_name",
        "input_resolution",
        "fps_e2e_std",
        "mem_peak_mb_e2e_std",
        "cpu_norm_e2e_std",
        "decode_time_std",
        "detector_time_std",
        "tracking_time_std",
        "write_time_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {k: "" for k in fields}
            out.update(row)
            out["warmup_iters"] = str(args.warmup_iters)
            out["repeat_runs"] = str(args.repeat_runs)
            for key in [
                "fps_e2e",
                "mem_peak_mb_e2e",
                "cpu_norm_e2e",
                "decode_time",
                "detector_time",
                "tracking_time",
                "write_time",
                "elapsed_e2e_sec",
                "fps_e2e_std",
                "mem_peak_mb_e2e_std",
                "cpu_norm_e2e_std",
                "decode_time_std",
                "detector_time_std",
                "tracking_time_std",
                "write_time_std",
            ]:
                if isinstance(out.get(key), float):
                    out[key] = f"{float(out[key]):.6f}"
            writer.writerow(out)


def main() -> int:
    args = parse_args()
    if args.warmup_iters < 0 or args.repeat_runs <= 0:
        raise ValueError("warmup_iters must be >= 0 and repeat_runs must be > 0.")
    if not args.mot_root.exists():
        raise FileNotFoundError(f"mot_root not found: {args.mot_root}")

    all_rows: List[Dict[str, float | str]] = []
    for method in METHODS:
        traj_model: rbs.TrajCostModel | None = None
        if method.cfg["traj"] == "on":
            if not args.traj_encoder.exists():
                raise FileNotFoundError(f"Trajectory encoder not found: {args.traj_encoder}")
            traj_model = rbs.TrajCostModel(args.traj_encoder)

        print(f"[scopeB] method={method.name} warmup={args.warmup_iters} repeat={args.repeat_runs}")
        for w in range(args.warmup_iters):
            _ = run_one(args=args, method=method, run_idx=-(w + 1), traj_model=traj_model)
            print(f"  warmup {w + 1}/{args.warmup_iters} done")

        rep_rows: List[Dict[str, float | str]] = []
        for r in range(args.repeat_runs):
            row = run_one(args=args, method=method, run_idx=(r + 1), traj_model=traj_model)
            rep_rows.append(row)
            all_rows.append(row)
            print(
                f"  repeat {r + 1}/{args.repeat_runs}: fps_e2e={float(row['fps_e2e']):.3f}, "
                f"cpu_norm_e2e={float(row['cpu_norm_e2e']):.3f}, mem_peak_mb_e2e={float(row['mem_peak_mb_e2e']):.3f}"
            )

        # aggregate mean/std for manuscript table row
        def gather(k: str) -> List[float]:
            return [float(rr[k]) for rr in rep_rows]

        mean_fps, std_fps = mean_std(gather("fps_e2e"))
        mean_mem, std_mem = mean_std(gather("mem_peak_mb_e2e"))
        mean_cpu, std_cpu = mean_std(gather("cpu_norm_e2e"))
        mean_dec, std_dec = mean_std(gather("decode_time"))
        mean_det, std_det = mean_std(gather("detector_time"))
        mean_trk, std_trk = mean_std(gather("tracking_time"))
        mean_wrt, std_wrt = mean_std(gather("write_time"))
        mean_elapsed, _ = mean_std(gather("elapsed_e2e_sec"))
        mean_frames, _ = mean_std(gather("total_frames"))

        agg = {
            "row_type": "mean",
            "method": method.name,
            "run_idx": 0,
            "split": args.split,
            "total_frames": int(round(mean_frames)),
            "fps_e2e": mean_fps,
            "mem_peak_mb_e2e": mean_mem,
            "cpu_norm_e2e": mean_cpu,
            "decode_time": mean_dec,
            "detector_time": mean_det,
            "tracking_time": mean_trk,
            "write_time": mean_wrt,
            "elapsed_e2e_sec": mean_elapsed,
            "detector_name": str(rep_rows[0]["detector_name"]),
            "input_resolution": str(rep_rows[0]["input_resolution"]),
            "fps_e2e_std": std_fps,
            "mem_peak_mb_e2e_std": std_mem,
            "cpu_norm_e2e_std": std_cpu,
            "decode_time_std": std_dec,
            "detector_time_std": std_det,
            "tracking_time_std": std_trk,
            "write_time_std": std_wrt,
        }
        all_rows.append(agg)

    write_csv(args.out_csv, all_rows, args=args)
    print(f"[scopeB] wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
