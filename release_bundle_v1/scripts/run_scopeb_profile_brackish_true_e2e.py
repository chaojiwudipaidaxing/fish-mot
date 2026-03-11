#!/usr/bin/env python
"""Run true Scope-B profiling: decode + detector inference + track + write."""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import json
import os
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import psutil
from ultralytics import YOLO

import run_baseline_sort as rbs


PLOT_STYLE = {
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
}
PLOT_SAVE_KWARGS = {"dpi": 300, "bbox_inches": "tight", "pad_inches": 0.02}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    key: str
    cfg: Mapping[str, object]


METHODS: List[MethodSpec] = [
    MethodSpec(
        name="Base",
        key="base",
        cfg={
            "gating": False,
            "traj": False,
            "adaptive_gamma": False,
            "beta": 0.0,
            "gamma": 0.0,
            "min_hits": 3,
        },
    ),
    MethodSpec(
        name="+gating",
        key="gating",
        cfg={
            "gating": True,
            "traj": False,
            "adaptive_gamma": False,
            "beta": 0.02,
            "gamma": 0.0,
            "min_hits": 3,
        },
    ),
]


class ProcessSampler:
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


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brackish true Scope-B profiler.")
    parser.add_argument(
        "--brackish-root",
        type=Path,
        default=Path(r"C:\Users\占子豪\Desktop\fish-mot\shuju\archive\BrackishMOT"),
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--gating-thresh", type=float, default=2000.0)
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--traj-window", type=int, default=16)
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("--audit-json", type=Path, default=Path("results/brackishmot_audit.json"))
    parser.add_argument("--clear-count", type=int, default=4)
    parser.add_argument("--high-count", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results/brackishmot/runtime"))
    parser.add_argument(
        "--paper-table",
        type=Path,
        default=Path("paper/cea_draft/tables/runtime_scopeb_brackish_true.tex"),
    )
    parser.add_argument(
        "--paper-fig",
        type=Path,
        default=Path("paper/cea_draft/figures/runtime_e2e_brackish_true_bar.pdf"),
    )
    return parser.parse_args()


def derive_scenarios_from_audit(audit_json: Path, split: str, clear_count: int, high_count: int) -> Dict[str, List[str]]:
    if not audit_json.exists():
        raise FileNotFoundError(f"Audit JSON missing: {audit_json}")
    data = json.loads(audit_json.read_text(encoding="utf-8"))
    rows = [r for r in data.get("sequence_audit", []) if str(r.get("split")) == split]
    if len(rows) < (clear_count + high_count):
        raise RuntimeError(f"Not enough audited sequences for split={split}: got {len(rows)}")
    rows = [r for r in rows if isinstance(r.get("visibility_proxy"), dict) and "quality_score" in r["visibility_proxy"]]
    rows = sorted(rows, key=lambda x: float(x["visibility_proxy"]["quality_score"]), reverse=True)
    clear = [str(r["name"]) for r in rows[:clear_count]]
    high = [str(r["name"]) for r in rows[-high_count:]]
    return {"clear": clear, "turbid_high": high}


def read_seqinfo(seqinfo_path: Path) -> tuple[int, int, int, str]:
    cp = configparser.ConfigParser()
    cp.read(seqinfo_path, encoding="utf-8")
    if "Sequence" not in cp:
        raise RuntimeError(f"Missing [Sequence] in {seqinfo_path}")
    s = cp["Sequence"]
    seq_len = int(s.get("seqLength", s.get("seqlength", "0")))
    im_w = int(s.get("imWidth", "0"))
    im_h = int(s.get("imHeight", "0"))
    img_ext = str(s.get("imExt", ".jpg"))
    return seq_len, im_w, im_h, img_ext


def image_paths(seq_dir: Path, img_ext: str, seq_len: int) -> List[Path]:
    img_dir = seq_dir / "img1"
    imgs = sorted([p for p in img_dir.glob(f"*{img_ext}") if p.is_file()])
    if not imgs:
        imgs = sorted([p for p in img_dir.glob("*") if p.is_file()])
    if seq_len > 0:
        imgs = imgs[:seq_len]
    return imgs


def boxes_xywh_from_result(result, width: int, height: int) -> List[np.ndarray]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy().tolist()
    out: List[np.ndarray] = []
    for b in xyxy:
        x1, y1, x2, y2 = [float(v) for v in b]
        x1 = max(0.0, min(x1, width - 1.0))
        y1 = max(0.0, min(y1, height - 1.0))
        x2 = max(0.0, min(x2, width))
        y2 = max(0.0, min(y2, height))
        w = x2 - x1
        h = y2 - y1
        if w <= 0.0 or h <= 0.0:
            continue
        out.append(np.array([x1, y1, w, h], dtype=float))
    return out


def write_outputs(path: Path, outputs: Iterable[Tuple[int, int, float, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for frame, track_id, x, y, w, h, score in outputs:
            f.write(f"{frame},{track_id},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{score:.3f},1,1\n")


def run_one(
    args: argparse.Namespace,
    model: YOLO,
    scenario_name: str,
    seqs: List[str],
    method: MethodSpec,
    run_idx: int,
) -> Dict[str, object]:
    split_dir = args.brackish_root / args.split
    pred_root = args.out / "pred_true" / scenario_name / method.key / f"run_{run_idx:02d}"

    decode_time = 0.0
    detector_time = 0.0
    tracking_time = 0.0
    write_time = 0.0
    total_frames = 0
    outputs_written = 0
    res_tags: List[str] = []

    t0 = time.perf_counter()
    with ProcessSampler(interval=args.sample_interval) as sampler:
        for seq in seqs:
            seq_dir = split_dir / seq
            seq_len, im_w, im_h, img_ext = read_seqinfo(seq_dir / "seqinfo.ini")
            res_tags.append(f"{im_w}x{im_h}")
            imgs = image_paths(seq_dir, img_ext=img_ext, seq_len=seq_len)
            total_frames += len(imgs)

            tracker = rbs.SortTracker(
                iou_thresh=args.iou_thresh,
                min_hits=int(method.cfg["min_hits"]),
                max_age=args.max_age,
                gating_enabled=bool(method.cfg["gating"]),
                gating_threshold=args.gating_thresh,
                alpha=args.alpha,
                beta=float(method.cfg["beta"]),
                gamma=float(method.cfg["gamma"]),
                traj_enabled=bool(method.cfg["traj"]),
                adaptive_gamma=bool(method.cfg["adaptive_gamma"]),
                adaptive_gamma_boost=1.5,
                adaptive_gamma_min=0.5,
                adaptive_gamma_max=2.0,
                traj_window=args.traj_window,
                traj_model=None,
                norm_w=float(im_w),
                norm_h=float(im_h),
            )

            seq_outputs: List[Tuple[int, int, float, float, float, float, float]] = []
            for frame_idx, img_path in enumerate(imgs, start=1):
                td0 = time.perf_counter()
                frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                td1 = time.perf_counter()
                decode_time += td1 - td0
                if frame is None:
                    continue

                te0 = time.perf_counter()
                result = model.predict(
                    source=frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                    verbose=False,
                )[0]
                dets = boxes_xywh_from_result(result, width=im_w, height=im_h)
                te1 = time.perf_counter()
                detector_time += te1 - te0

                tt0 = time.perf_counter()
                frame_outputs, _ = tracker.step(dets, frame_idx)
                tt1 = time.perf_counter()
                tracking_time += tt1 - tt0
                seq_outputs.extend(frame_outputs)

            tw0 = time.perf_counter()
            write_outputs(pred_root / f"{seq}.txt", seq_outputs)
            tw1 = time.perf_counter()
            write_time += tw1 - tw0
            outputs_written += len(seq_outputs)
    t1 = time.perf_counter()
    elapsed = t1 - t0

    cpu_count = max(1, int(psutil.cpu_count() or 1))
    cpu_mean_1core = float(statistics.fmean(sampler.cpu_samples)) if sampler.cpu_samples else 0.0
    cpu_norm = cpu_mean_1core / cpu_count
    fps = float(total_frames / max(elapsed, 1e-9))
    input_resolution = res_tags[0] if len(set(res_tags)) == 1 else "mixed"

    return {
        "row_type": "repeat",
        "scenario": scenario_name,
        "method": method.name,
        "run_idx": run_idx,
        "split": args.split,
        "time_unit": "sec_per_run",
        "total_frames": total_frames,
        "fps_e2e": fps,
        "mem_peak_mb_e2e": sampler.mem_peak_mb,
        "cpu_norm_e2e": cpu_norm,
        "decode_time": decode_time,
        "detector_time": detector_time,
        "tracking_time": tracking_time,
        "write_time": write_time,
        "elapsed_e2e_sec": elapsed,
        "outputs_written": outputs_written,
        "detector_name": "",
        "input_resolution": input_resolution,
    }


def mean_std(vals: List[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(statistics.fmean(vals)), float(statistics.stdev(vals))


def write_csv(path: Path, fields: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_table(path: Path, rows: List[Dict[str, object]]) -> None:
    lines = [
        "% Auto-generated by scripts/run_scopeb_profile_brackish_true_e2e.py",
        r"\begin{tabular}{lllllllllll}",
        r"\toprule",
        r"Scenario & Method & fps\_e2e & mem\_peak\_mb\_e2e & cpu\_norm\_e2e & decode\_time & detector\_time & tracking\_time & write\_time & detector\_name & input\_resolution \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            " & ".join(
                [
                    str(r["scenario"]),
                    str(r["method"]),
                    f"{float(r['fps_e2e']):.3f} $\\pm$ {float(r['fps_e2e_std']):.3f}",
                    f"{float(r['mem_peak_mb_e2e']):.3f} $\\pm$ {float(r['mem_peak_mb_e2e_std']):.3f}",
                    f"{float(r['cpu_norm_e2e']):.3f} $\\pm$ {float(r['cpu_norm_e2e_std']):.3f}",
                    f"{float(r['decode_time']):.3f} $\\pm$ {float(r['decode_time_std']):.3f}",
                    f"{float(r['detector_time']):.3f} $\\pm$ {float(r['detector_time_std']):.3f}",
                    f"{float(r['tracking_time']):.3f} $\\pm$ {float(r['tracking_time_std']):.3f}",
                    f"{float(r['write_time']):.3f} $\\pm$ {float(r['write_time_std']):.3f}",
                    str(r["detector_name"]).replace("_", r"\_"),
                    str(r["input_resolution"]).replace("_", r"\_"),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_figure(path: Path, rows: List[Dict[str, object]]) -> None:
    plt.rcParams.update(PLOT_STYLE)
    labels = [f"{r['scenario']}-{r['method']}" for r in rows]
    x = np.arange(len(labels), dtype=float)
    fps = np.array([float(r["fps_e2e"]) for r in rows], dtype=float)
    mem = np.array([float(r["mem_peak_mb_e2e"]) for r in rows], dtype=float)
    cpu = np.array([float(r["cpu_norm_e2e"]) for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), constrained_layout=True)
    axes[0].bar(x, fps, color="#4C78A8")
    axes[0].set_title("fps_e2e")
    axes[0].set_ylabel("frames/s")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, mem, color="#F58518")
    axes[1].set_title("mem_peak_mb_e2e")
    axes[1].set_ylabel("MB")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(x, cpu, color="#54A24B")
    axes[2].set_title("cpu_norm_e2e")
    axes[2].set_ylabel("ratio")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25, ha="right")
    axes[2].grid(axis="y", alpha=0.25)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **PLOT_SAVE_KWARGS)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    split_dir = args.brackish_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory missing: {split_dir}")

    model = YOLO(str(args.weights))
    wsha = sha256_file(args.weights)[:8]
    detector_name = f"yolov8n_best_{wsha}"

    scenarios = derive_scenarios_from_audit(
        audit_json=args.audit_json,
        split=args.split,
        clear_count=args.clear_count,
        high_count=args.high_count,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pred_true").mkdir(parents=True, exist_ok=True)
    (args.out / "runtime_groups_true.json").write_text(
        json.dumps({"split": args.split, "scenarios": scenarios}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    repeat_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for scenario_name, seqs in scenarios.items():
        for method in METHODS:
            print(f"[scopeB-true] scenario={scenario_name} method={method.name} warmup={args.warmup} repeat={args.repeat}")
            for w in range(args.warmup):
                _ = run_one(args=args, model=model, scenario_name=scenario_name, seqs=seqs, method=method, run_idx=-(w + 1))

            reps: List[Dict[str, object]] = []
            for ridx in range(1, args.repeat + 1):
                r = run_one(args=args, model=model, scenario_name=scenario_name, seqs=seqs, method=method, run_idx=ridx)
                r["detector_name"] = detector_name
                reps.append(r)
                repeat_rows.append(r)
                print(
                    f"  repeat {ridx}/{args.repeat}: fps={float(r['fps_e2e']):.3f} "
                    f"decode={float(r['decode_time']):.3f}s det={float(r['detector_time']):.3f}s "
                    f"track={float(r['tracking_time']):.3f}s write={float(r['write_time']):.3f}s"
                )

            def gather(key: str) -> List[float]:
                return [float(x[key]) for x in reps]

            mean_fps, std_fps = mean_std(gather("fps_e2e"))
            mean_mem, std_mem = mean_std(gather("mem_peak_mb_e2e"))
            mean_cpu, std_cpu = mean_std(gather("cpu_norm_e2e"))
            mean_dec, std_dec = mean_std(gather("decode_time"))
            mean_det, std_det = mean_std(gather("detector_time"))
            mean_trk, std_trk = mean_std(gather("tracking_time"))
            mean_wrt, std_wrt = mean_std(gather("write_time"))
            mean_elapsed, _ = mean_std(gather("elapsed_e2e_sec"))
            mean_frames, _ = mean_std(gather("total_frames"))

            summary_rows.append(
                {
                    "scenario": scenario_name,
                    "method": method.name,
                    "split": args.split,
                    "warmup_iters": args.warmup,
                    "repeat_runs": args.repeat,
                    "time_unit": "sec_per_run",
                    "total_frames": int(round(mean_frames)),
                    "fps_e2e": mean_fps,
                    "fps_e2e_std": std_fps,
                    "mem_peak_mb_e2e": mean_mem,
                    "mem_peak_mb_e2e_std": std_mem,
                    "cpu_norm_e2e": mean_cpu,
                    "cpu_norm_e2e_std": std_cpu,
                    "decode_time": mean_dec,
                    "decode_time_std": std_dec,
                    "detector_time": mean_det,
                    "detector_time_std": std_det,
                    "tracking_time": mean_trk,
                    "tracking_time_std": std_trk,
                    "write_time": mean_wrt,
                    "write_time_std": std_wrt,
                    "elapsed_e2e_sec": mean_elapsed,
                    "detector_name": detector_name,
                    "input_resolution": str(reps[0]["input_resolution"]),
                }
            )

    repeat_csv = args.out / "runtime_profile_e2e_true.csv"
    summary_csv = args.out / "runtime_profile_e2e_true_summary.csv"
    write_csv(
        repeat_csv,
        fields=[
            "row_type",
            "scenario",
            "method",
            "run_idx",
            "split",
            "time_unit",
            "total_frames",
            "fps_e2e",
            "mem_peak_mb_e2e",
            "cpu_norm_e2e",
            "decode_time",
            "detector_time",
            "tracking_time",
            "write_time",
            "elapsed_e2e_sec",
            "outputs_written",
            "detector_name",
            "input_resolution",
        ],
        rows=repeat_rows,
    )
    write_csv(
        summary_csv,
        fields=[
            "scenario",
            "method",
            "split",
            "warmup_iters",
            "repeat_runs",
            "time_unit",
            "total_frames",
            "fps_e2e",
            "fps_e2e_std",
            "mem_peak_mb_e2e",
            "mem_peak_mb_e2e_std",
            "cpu_norm_e2e",
            "cpu_norm_e2e_std",
            "decode_time",
            "decode_time_std",
            "detector_time",
            "detector_time_std",
            "tracking_time",
            "tracking_time_std",
            "write_time",
            "write_time_std",
            "elapsed_e2e_sec",
            "detector_name",
            "input_resolution",
        ],
        rows=summary_rows,
    )
    write_table(args.paper_table, summary_rows)
    write_figure(args.paper_fig, summary_rows)

    print(f"[scopeB-true] wrote: {repeat_csv}")
    print(f"[scopeB-true] wrote: {summary_csv}")
    print(f"[scopeB-true] wrote: {args.paper_table}")
    print(f"[scopeB-true] wrote: {args.paper_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
