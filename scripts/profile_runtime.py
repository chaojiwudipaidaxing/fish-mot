#!/usr/bin/env python
"""Profile runtime resources for the cumulative Base main-chain trackers."""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import psutil

from method_labels import MAIN_CHAIN_METHOD_ORDER, normalize_main_chain_rows


METHODS: List[Tuple[str, Dict[str, str]]] = [
    (MAIN_CHAIN_METHOD_ORDER[0], {"gating": "off", "traj": "off", "beta": "0.0", "gamma": "0.0", "adaptive_gamma": "off"}),
    (MAIN_CHAIN_METHOD_ORDER[1], {"gating": "on", "traj": "off", "beta": "0.02", "gamma": "0.0", "adaptive_gamma": "off"}),
    (MAIN_CHAIN_METHOD_ORDER[2], {"gating": "on", "traj": "on", "beta": "0.02", "gamma": "0.5", "adaptive_gamma": "off"}),
    (MAIN_CHAIN_METHOD_ORDER[3], {"gating": "on", "traj": "on", "beta": "0.02", "gamma": "0.5", "adaptive_gamma": "on"}),
]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runtime profiling for tracking methods.")
    parser.add_argument("--split", default="val_half", help="Prepared split name under data/mft25_mot.")
    parser.add_argument(
        "--mot-root",
        type=Path,
        default=Path("data/mft25_mot"),
        help="Root directory of prepared MOT data.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=50,
        help="Use up to this many frames per sequence (smoke default: 50).",
    )
    parser.add_argument("--drop-rate", type=float, default=0.2, help="Detection drop-rate passed to baseline.")
    parser.add_argument("--jitter", type=float, default=0.02, help="Detection jitter ratio passed to baseline.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for detection degradation.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha in SORT cost.")
    parser.add_argument("--beta", type=float, default=0.02, help="Default beta for gating/traj/adaptive.")
    parser.add_argument("--gamma", type=float, default=0.5, help="Default gamma for traj/adaptive.")
    parser.add_argument(
        "--traj-encoder",
        type=Path,
        default=Path("runs/traj_encoder/traj_encoder.pt"),
        help="Trajectory encoder checkpoint path.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("results/runtime_profile/pred"),
        help="Prediction output root used during profiling.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.2,
        help="Sampling interval seconds for CPU%%/memory.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/runtime_profile.csv"),
        help="Output runtime profile CSV.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("results/paper_assets_val/runtime_profile.png"),
        help="Output runtime profile plot path.",
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=None,
        help="Optional run_config.json. If provided, split/max_frames/drop/jitter/seeds are loaded from it.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="If set, render plot directly from an existing runtime profile CSV.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip profiling runs and only render plot from --input-csv.",
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
    args.drop_rate = float(cfg.get("drop_rate", args.drop_rate))
    args.jitter = float(cfg.get("jitter", args.jitter))
    seeds = cfg.get("seeds")
    if isinstance(seeds, list) and seeds:
        try:
            args.seed = int(seeds[0])
        except Exception:  # noqa: BLE001
            pass
    elif isinstance(seeds, str) and seeds.strip():
        first = seeds.split(",")[0].strip()
        if first:
            try:
                args.seed = int(first)
            except Exception:  # noqa: BLE001
                pass
    result_root = Path(str(cfg.get("result_root", "results/main_val")))
    runtime_dir = Path(str(cfg.get("runtime_dir", result_root / "runtime")))
    paper_assets_dir = Path(str(cfg.get("paper_assets_dir", result_root / "paper_assets")))
    if str(args.work_root).replace("\\", "/").lower() == "results/runtime_profile/pred":
        args.work_root = runtime_dir / "pred"
    if str(args.output_csv).replace("\\", "/").lower() == "results/runtime_profile.csv":
        args.output_csv = runtime_dir / "runtime_profile.csv"
    if str(args.plot_path).replace("\\", "/").lower() == "results/paper_assets_val/runtime_profile.png":
        args.plot_path = paper_assets_dir / "runtime_profile.png"
    root_norm = str(result_root).replace("\\", "/").lower()
    write_targets = [
        ("work_root", args.work_root),
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


def estimate_total_frames(split_dir: Path, split: str, max_frames: int) -> int:
    seqs = read_seqmap(split_dir / "seqmaps" / f"{split}.txt")
    total = 0
    for seq in seqs:
        seq_len = read_seq_length(split_dir / seq / "seqinfo.ini")
        total += min(seq_len, max_frames) if max_frames > 0 else seq_len
    return total


def method_command(
    args: argparse.Namespace,
    method: str,
    method_cfg: Dict[str, str],
    out_dir: Path,
) -> List[str]:
    beta = method_cfg.get("beta", f"{args.beta}")
    gamma = method_cfg.get("gamma", f"{args.gamma}")
    cmd = [
        sys.executable,
        "scripts/run_baseline_sort.py",
        "--split",
        args.split,
        "--mot-root",
        str(args.mot_root),
        "--max-frames",
        str(args.max_frames),
        "--gating",
        method_cfg["gating"],
        "--traj",
        method_cfg["traj"],
        "--alpha",
        str(args.alpha),
        "--beta",
        str(beta),
        "--gamma",
        str(gamma),
        "--adaptive-gamma",
        method_cfg["adaptive_gamma"],
        "--drop-rate",
        str(args.drop_rate),
        "--jitter",
        str(args.jitter),
        "--degrade-seed",
        str(args.seed),
        "--out-dir",
        str(out_dir),
        "--clean-out",
    ]
    if method_cfg["traj"] == "on":
        cmd.extend(["--traj-encoder", str(args.traj_encoder)])
    return cmd


def run_profile_for_method(
    args: argparse.Namespace,
    method: str,
    method_cfg: Dict[str, str],
    split_dir: Path,
    total_frames: int,
) -> Dict[str, str]:
    out_dir = args.work_root / method.replace("+", "plus").replace(" ", "_").lower()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = method_command(args, method, method_cfg, out_dir)

    start = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=None, stderr=None)
    cpu_samples: List[float] = []
    mem_peak_mb = 0.0
    note = "ok"
    p = psutil.Process(proc.pid)
    warmed_pids: set[int] = set()
    observed_pids: set[int] = set()

    def choose_target(root_proc: psutil.Process) -> psutil.Process:
        try:
            descendants = root_proc.children(recursive=True)
        except psutil.Error:
            descendants = []
        if descendants:
            # Prefer the newest descendant (typically the actual python worker).
            descendants.sort(key=lambda x: x.create_time())
            return descendants[-1]
        return root_proc

    while proc.poll() is None:
        target = choose_target(p)
        if target.pid > 0:
            observed_pids.add(target.pid)

        # Warm up once per observed process, then use interval sampling.
        if target.pid not in warmed_pids:
            try:
                target.cpu_percent(interval=None)
            except psutil.Error:
                pass
            warmed_pids.add(target.pid)
            continue

        try:
            cpu_val = float(target.cpu_percent(interval=args.sample_interval))
        except psutil.Error:
            cpu_val = 0.0
            time.sleep(max(0.05, args.sample_interval))
        cpu_samples.append(cpu_val)

        try:
            mem_mb = float(target.memory_info().rss / (1024.0 * 1024.0))
            mem_peak_mb = max(mem_peak_mb, mem_mb)
        except psutil.Error:
            pass

    return_code = proc.wait()
    elapsed = max(1e-6, time.perf_counter() - start)

    fps = float(total_frames / elapsed) if total_frames > 0 else 0.0
    cpu_mean_1core = float(np.mean(cpu_samples)) if cpu_samples else 0.0
    cpu_count = max(1, int(psutil.cpu_count() or 1))
    cpu_mean_norm = float(cpu_mean_1core / cpu_count)

    if mem_peak_mb < 30.0:
        raise ValueError(
            f"{method}: mem_peak_mb={mem_peak_mb:.3f} < 30. "
            "你仍在统计父进程/单位错误"
        )

    valid_observed_pids = sorted(pid for pid in observed_pids if pid > 0)
    row = {
        "split": args.split,
        "method": method,
        "max_frames": str(args.max_frames),
        "total_frames": str(total_frames),
        "elapsed_sec": f"{elapsed:.3f}",
        "fps": f"{fps:.3f}",
        "mem_peak_mb": f"{mem_peak_mb:.3f}",
        "cpu_mean_1core_percent": f"{cpu_mean_1core:.3f}",
        "cpu_mean_norm_percent": f"{cpu_mean_norm:.3f}",
        "note": (
            f"{note};samples={len(cpu_samples)};cpu_count={cpu_count};"
            f"launcher_pid={proc.pid};observed_pids={','.join(str(x) for x in valid_observed_pids)}"
        ),
        "return_code": str(return_code),
        "command": shlex.join(cmd),
    }
    return row


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    fields = [
        "split",
        "method",
        "max_frames",
        "total_frames",
        "elapsed_sec",
        "fps",
        "mem_peak_mb",
        "cpu_mean_1core_percent",
        "cpu_mean_norm_percent",
        "note",
        "return_code",
        "command",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return normalize_main_chain_rows(list(csv.DictReader(f)))


def apply_plot_style() -> None:
    plt.rcParams.update(PLOT_STYLE)


def make_plot(rows: List[Dict[str, str]], out_path: Path) -> None:
    apply_plot_style()
    methods = [r["method"] for r in rows]
    fps = [float(r["fps"]) for r in rows]
    mem = [float(r["mem_peak_mb"]) for r in rows]
    cpu_norm = [float(r["cpu_mean_norm_percent"]) for r in rows]

    x = np.arange(len(methods))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8), constrained_layout=True)

    axes[0].bar(x, fps, color="#4C78A8", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylabel("FPS")
    axes[0].set_title("Runtime throughput")
    axes[0].grid(axis="y", alpha=0.18)

    axes[1].bar(x - width / 2.0, mem, width=width, label="Memory (MB)", color="#F58518", linewidth=0.5)
    axes[1].bar(x + width / 2.0, cpu_norm, width=width, label="CPU (%)", color="#54A24B", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].set_ylabel("Resource value")
    axes[1].set_title("Resource profile (MB / %)")
    axes[1].grid(axis="y", alpha=0.18)
    axes[1].legend(loc="upper right", frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **PLOT_SAVE_KWARGS)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_run_config(args)
    input_csv = args.input_csv if args.input_csv is not None else args.output_csv
    if args.plot_only:
        rows = read_rows(input_csv)
        write_csv(args.output_csv, rows)
        make_plot(rows, args.plot_path)
        print(f"Normalized runtime profile CSV: {args.output_csv}")
        print(f"Rendered runtime profile plot: {args.plot_path}")
        return

    split_dir = args.mot_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Prepared split not found: {split_dir}. "
            "Please run scripts/prepare_mft25.py first."
        )
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be > 0 for profiling.")
    if args.drop_rate < 0.0 or args.drop_rate >= 1.0:
        raise ValueError("--drop-rate must be in [0, 1).")
    if args.jitter < 0.0:
        raise ValueError("--jitter must be >= 0.")
    if not args.traj_encoder.exists():
        raise FileNotFoundError(f"Trajectory encoder not found: {args.traj_encoder}")

    total_frames = estimate_total_frames(split_dir, args.split, args.max_frames)
    rows: List[Dict[str, str]] = []

    for method, cfg in METHODS:
        row = run_profile_for_method(args, method, cfg, split_dir, total_frames)
        rows.append(row)
        print(
            f"[runtime] {method}: fps={row['fps']} mem_peak_mb={row['mem_peak_mb']} "
            f"cpu_1core={row['cpu_mean_1core_percent']} cpu_norm={row['cpu_mean_norm_percent']} "
            f"return={row['return_code']}"
        )

    write_csv(args.output_csv, rows)
    make_plot(rows, args.plot_path)
    print(f"Saved runtime profile CSV: {args.output_csv}")
    print(f"Saved runtime profile plot: {args.plot_path}")


if __name__ == "__main__":
    main()
