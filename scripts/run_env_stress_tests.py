#!/usr/bin/env python
"""Run fixed environment stress tests and export paper artifacts.

This script executes a fixed stress matrix on val-half with a fixed sequence list
and seed, then writes:
  - results/stress_params.csv
  - paper/cea_draft/tables/stress_param_table.tex
  - results/stress_metrics.csv
  - results/fig_stress_hota_vs_level.pdf
  - results/fig_stress_bucket_shift.pdf

Notes:
- This repository's tracker currently supports detector degradation via drop/jitter.
- Motion blur / low-light / turbidity are parameterized and mapped to fixed
  drop/jitter proxies for repeatable stress execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np


FIXED_SPLIT = "val_half"
FIXED_SEED = 0
FIXED_MAX_FRAMES = 1000
FIXED_GATING_THRESH = 2000.0
FIXED_MAX_GT_IDS = 50000
FIXED_SEQUENCES = ["BT-001", "BT-003", "BT-005", "MSK-002", "PF-001", "SN-001", "SN-013", "SN-015"]
LEVEL_ORDER = ["low", "mid", "high"]


PLOT_STYLE = {
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "grid.linewidth": 0.4,
}
PLOT_SAVE_KWARGS = {"dpi": 300, "bbox_inches": "tight", "pad_inches": 0.02}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    key: str
    args: Mapping[str, str]


METHODS: List[MethodSpec] = [
    MethodSpec(
        name="Base",
        key="base",
        args={
            "gating": "off",
            "traj": "off",
            "adaptive_gamma": "off",
            "alpha": "1.0",
            "beta": "0.0",
            "gamma": "0.0",
            "iou_thresh": "0.3",
            "min_hits": "3",
            "max_age": "30",
        },
    ),
    MethodSpec(
        name="+gating",
        key="gating",
        args={
            "gating": "on",
            "traj": "off",
            "adaptive_gamma": "off",
            "alpha": "1.0",
            "beta": "0.02",
            "gamma": "0.0",
            "iou_thresh": "0.3",
            "min_hits": "3",
            "max_age": "30",
        },
    ),
    MethodSpec(
        name="ByteTrack",
        key="bytetrack",
        args={
            "gating": "off",
            "traj": "off",
            "adaptive_gamma": "off",
            "alpha": "1.0",
            "beta": "0.0",
            "gamma": "0.0",
            "iou_thresh": "0.3",
            "min_hits": "1",
            "max_age": "30",
        },
    ),
]


DEGRADATION_PLAN: Dict[str, Dict[str, Dict[str, float]]] = {
    "motion_blur": {
        "low": {"kernel_len_px": 7.0, "angle_deg": 0.0, "drop_rate": 0.05, "jitter": 0.010},
        "mid": {"kernel_len_px": 15.0, "angle_deg": 0.0, "drop_rate": 0.12, "jitter": 0.020},
        "high": {"kernel_len_px": 25.0, "angle_deg": 0.0, "drop_rate": 0.22, "jitter": 0.035},
    },
    "low_light": {
        "low": {"alpha": 0.85, "beta": -8.0, "drop_rate": 0.08, "jitter": 0.012},
        "mid": {"alpha": 0.70, "beta": -16.0, "drop_rate": 0.16, "jitter": 0.024},
        "high": {"alpha": 0.55, "beta": -24.0, "drop_rate": 0.28, "jitter": 0.036},
    },
    "turbidity_haze": {
        "low": {"t": 0.90, "A": 0.10, "drop_rate": 0.10, "jitter": 0.015},
        "mid": {"t": 0.75, "A": 0.20, "drop_rate": 0.20, "jitter": 0.028},
        "high": {"t": 0.60, "A": 0.30, "drop_rate": 0.32, "jitter": 0.040},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed environment stress tests.")
    parser.add_argument("--mot-root", type=Path, default=Path("data/mft25_mot_full"))
    parser.add_argument("--result-root", type=Path, default=Path("results/stress_tests"))
    parser.add_argument("--results-csv", type=Path, default=Path("results/stress_metrics.csv"))
    parser.add_argument("--params-csv", type=Path, default=Path("results/stress_params.csv"))
    parser.add_argument(
        "--params-tex",
        type=Path,
        default=Path("paper/cea_draft/tables/stress_param_table.tex"),
    )
    parser.add_argument("--fig-hota", type=Path, default=Path("results/fig_stress_hota_vs_level.pdf"))
    parser.add_argument("--fig-bucket", type=Path, default=Path("results/fig_stress_bucket_shift.pdf"))
    parser.add_argument(
        "--paper-fig-dir",
        type=Path,
        default=Path("paper/cea_draft/figs"),
        help="Figure copy target for manuscript includegraphics.",
    )
    parser.add_argument("--skip-stratified", action="store_true", help="Skip bucket-shift diagnostics.")
    return parser.parse_args()


def read_seq_length(seqinfo_path: Path) -> int:
    import configparser

    parser = configparser.ConfigParser()
    parser.read(seqinfo_path, encoding="utf-8")
    if "Sequence" not in parser or "seqLength" not in parser["Sequence"]:
        raise RuntimeError(f"Cannot parse seqLength from {seqinfo_path}")
    return int(parser["Sequence"]["seqLength"])


def total_frames_for_eval(split_dir: Path, max_frames: int, sequences: Iterable[str]) -> int:
    total = 0
    for seq in sequences:
        seq_len = read_seq_length(split_dir / seq / "seqinfo.ini")
        total += min(seq_len, max_frames) if max_frames > 0 else seq_len
    return total


def run_cmd(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_mean_metrics(mean_csv: Path) -> Dict[str, float]:
    with mean_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row in {mean_csv}, got {len(rows)}")
    row = rows[0]
    return {
        "HOTA": float(row["HOTA"]),
        "IDF1": float(row["IDF1"]),
        "IDSW": float(row["IDSW"]),
        "DetA": float(row["DetA"]),
        "AssA": float(row["AssA"]),
    }


def load_frame_counts(mot_txt: Path, max_frame: int) -> np.ndarray:
    counts = np.zeros((max_frame,), dtype=np.float64)
    with mot_txt.open("r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            parts = row.split(",")
            if not parts:
                continue
            frame = int(float(parts[0]))
            if 1 <= frame <= max_frame:
                counts[frame - 1] += 1.0
    return counts


def compute_count_metrics(gt_counts: np.ndarray, pred_counts: np.ndarray) -> Dict[str, float]:
    if gt_counts.shape != pred_counts.shape:
        raise ValueError("Mismatched gt/pred frame arrays.")
    diff = pred_counts - gt_counts
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    var = float(np.var(diff))
    if diff.size <= 1:
        drift = 0.0
    else:
        edge = max(1, min(30, diff.size // 5))
        drift = float(np.mean(diff[-edge:]) - np.mean(diff[:edge]))
    return {"CountMAE": mae, "CountRMSE": rmse, "CountVar": var, "CountDrift": drift}


def aggregate_count_metrics(split_dir: Path, pred_dir: Path) -> Dict[str, float]:
    per_seq: List[Dict[str, float]] = []
    for seq in FIXED_SEQUENCES:
        seq_len = read_seq_length(split_dir / seq / "seqinfo.ini")
        used_frames = min(seq_len, FIXED_MAX_FRAMES) if FIXED_MAX_FRAMES > 0 else seq_len
        gt_counts = load_frame_counts(split_dir / seq / "gt" / "gt.txt", used_frames)
        pred_counts = load_frame_counts(pred_dir / f"{seq}.txt", used_frames)
        per_seq.append(compute_count_metrics(gt_counts, pred_counts))
    out: Dict[str, float] = {}
    for key in ["CountMAE", "CountRMSE", "CountVar", "CountDrift"]:
        out[key] = float(np.mean([row[key] for row in per_seq]))
    return out


def format_param_cell(degradation: str, params: Mapping[str, float]) -> str:
    if degradation == "motion_blur":
        return (
            f"$\\ell={int(params['kernel_len_px'])}$ px, "
            f"$\\theta={int(params['angle_deg'])}^\\circ$; "
            f"proxy $(p_\\mathrm{{drop}}={params['drop_rate']:.2f},\\ j_\\mathrm{{bbox}}={params['jitter']:.3f})$"
        )
    if degradation == "low_light":
        return (
            f"$\\alpha={params['alpha']:.2f},\\ \\beta={params['beta']:.0f}$; "
            f"proxy $(p_\\mathrm{{drop}}={params['drop_rate']:.2f},\\ j_\\mathrm{{bbox}}={params['jitter']:.3f})$"
        )
    return (
        f"$t={params['t']:.2f},\\ A={params['A']:.2f}$; "
        f"proxy $(p_\\mathrm{{drop}}={params['drop_rate']:.2f},\\ j_\\mathrm{{bbox}}={params['jitter']:.3f})$"
    )


def write_stress_params_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["degradation", "level", "params_json", "split", "sequences", "seed"],
        )
        writer.writeheader()
        for degradation, levels in DEGRADATION_PLAN.items():
            for level in LEVEL_ORDER:
                params = levels[level]
                writer.writerow(
                    {
                        "degradation": degradation,
                        "level": level,
                        "params_json": json.dumps(params, ensure_ascii=True, sort_keys=True),
                        "split": FIXED_SPLIT,
                        "sequences": "|".join(FIXED_SEQUENCES),
                        "seed": str(FIXED_SEED),
                    }
                )


def write_stress_params_tex(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("% Auto-generated by scripts/run_env_stress_tests.py\n")
        f.write("\\begin{tabular}{lll}\n")
        f.write("\\toprule\n")
        f.write("Degradation & Strength level & Parameter setting \\\\\n")
        f.write("\\midrule\n")
        for degradation, levels in DEGRADATION_PLAN.items():
            pretty = degradation.replace("_", " ").title()
            for level in LEVEL_ORDER:
                f.write(f"{pretty} & {level} & {format_param_cell(degradation, levels[level])} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def run_one_setting(
    mot_root: Path,
    split_dir: Path,
    result_root: Path,
    method: MethodSpec,
    degradation: str,
    level: str,
    params: Mapping[str, float],
    total_frames: int,
) -> Dict[str, str]:
    run_tag = f"{degradation}_{level}"
    run_dir = result_root / degradation / level / method.key
    pred_dir = run_dir / "pred"
    mean_csv = run_dir / "mean.csv"
    per_seq_csv = run_dir / "per_seq.csv"

    track_cmd = [
        sys.executable,
        "scripts/run_baseline_sort.py",
        "--split",
        FIXED_SPLIT,
        "--mot-root",
        str(mot_root),
        "--seqs",
        *FIXED_SEQUENCES,
        "--max-frames",
        str(FIXED_MAX_FRAMES),
        "--det-source",
        "auto",
        "--gating",
        method.args["gating"],
        "--traj",
        method.args["traj"],
        "--adaptive-gamma",
        method.args["adaptive_gamma"],
        "--alpha",
        method.args["alpha"],
        "--beta",
        method.args["beta"],
        "--gamma",
        method.args["gamma"],
        "--iou-thresh",
        method.args["iou_thresh"],
        "--min-hits",
        method.args["min_hits"],
        "--max-age",
        method.args["max_age"],
        "--drop-rate",
        f"{params['drop_rate']:.6f}",
        "--jitter",
        f"{params['jitter']:.6f}",
        "--degrade-seed",
        str(FIXED_SEED),
        "--out-dir",
        str(pred_dir),
        "--clean-out",
    ]
    if method.name == "+gating":
        track_cmd.extend(["--gating-thresh", f"{FIXED_GATING_THRESH:.6f}"])

    start = time.perf_counter()
    run_cmd(track_cmd)
    elapsed = max(1e-6, time.perf_counter() - start)
    fps = (float(total_frames) / elapsed) if total_frames > 0 else 0.0

    eval_cmd = [
        sys.executable,
        "scripts/eval_trackeval_per_seq.py",
        "--split",
        FIXED_SPLIT,
        "--mot-root",
        str(mot_root),
        "--pred-dir",
        str(pred_dir),
        "--tracker-name",
        f"stress_{method.key}_{run_tag}",
        "--max-frames",
        str(FIXED_MAX_FRAMES),
        "--max-gt-ids",
        str(FIXED_MAX_GT_IDS),
        "--results-per-seq",
        str(per_seq_csv),
        "--results-mean",
        str(mean_csv),
    ]
    run_cmd(eval_cmd)

    mot_metrics = parse_mean_metrics(mean_csv)
    count_metrics = aggregate_count_metrics(split_dir, pred_dir)
    row = {
        "method": method.name,
        "degradation": degradation,
        "level": level,
        "HOTA": f"{mot_metrics['HOTA']:.3f}",
        "IDF1": f"{mot_metrics['IDF1']:.3f}",
        "IDSW": f"{mot_metrics['IDSW']:.3f}",
        "DetA": f"{mot_metrics['DetA']:.3f}",
        "AssA": f"{mot_metrics['AssA']:.3f}",
        "CountMAE": f"{count_metrics['CountMAE']:.3f}",
        "CountRMSE": f"{count_metrics['CountRMSE']:.3f}",
        "CountVar": f"{count_metrics['CountVar']:.3f}",
        "CountDrift": f"{count_metrics['CountDrift']:.3f}",
        "fps_tracking": f"{fps:.3f}",
        "seed": str(FIXED_SEED),
        "split": FIXED_SPLIT,
    }
    print(
        "[stress]",
        method.name,
        degradation,
        level,
        f"HOTA={row['HOTA']}",
        f"IDF1={row['IDF1']}",
        f"IDSW={row['IDSW']}",
        f"CountMAE={row['CountMAE']}",
    )
    return row


def write_stress_metrics_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "degradation",
        "level",
        "HOTA",
        "IDF1",
        "IDSW",
        "DetA",
        "AssA",
        "CountMAE",
        "CountRMSE",
        "CountVar",
        "CountDrift",
        "fps_tracking",
        "split",
        "seed",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_hota_figure(rows: List[Dict[str, str]], out_path: Path) -> None:
    plt.rcParams.update(PLOT_STYLE)
    degradations = list(DEGRADATION_PLAN.keys())
    methods = [m.name for m in METHODS]
    level_to_x = {name: i for i, name in enumerate(LEVEL_ORDER)}
    x_vals = np.arange(len(LEVEL_ORDER))
    colors = {"Base": "#4C78A8", "+gating": "#F58518", "ByteTrack": "#54A24B"}

    fig, axes = plt.subplots(1, len(degradations), figsize=(12.0, 3.6), sharey=True, constrained_layout=True)
    if len(degradations) == 1:
        axes = [axes]

    value_map: Dict[Tuple[str, str, str], float] = {}
    for row in rows:
        value_map[(row["method"], row["degradation"], row["level"])] = float(row["HOTA"])

    for ax, degradation in zip(axes, degradations):
        for method in methods:
            y = [value_map[(method, degradation, level)] for level in LEVEL_ORDER]
            ax.plot(x_vals, y, marker="o", label=method, color=colors[method])
        ax.set_title(degradation.replace("_", " "))
        ax.set_xticks(x_vals)
        ax.set_xticklabels(LEVEL_ORDER)
        ax.set_xlabel("stress level")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("HOTA (%)")
    axes[-1].legend(loc="lower left", frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **PLOT_SAVE_KWARGS)
    plt.close(fig)


def run_stratified_for_gating(mot_root: Path, pred_gating: Path, out_csv: Path, out_plot: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/eval_stratified.py",
        "--split",
        FIXED_SPLIT,
        "--mot-root",
        str(mot_root),
        "--pred-root",
        "results/main_val/seed_runs/seed_0",
        "--pred-gating",
        str(pred_gating),
        "--max-frames",
        str(FIXED_MAX_FRAMES),
        "--bucket-mode",
        "quantile",
        "--bucket-min-samples",
        "200",
        "--output-csv",
        str(out_csv),
        "--plot-path",
        str(out_plot),
    ]
    run_cmd(cmd)


def _load_bucket_high(csv_path: Path, method_name: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] != method_name:
                continue
            if row["bucket"] != "high":
                continue
            crit = row["bucket_type"]
            out[crit] = {"IDSW": float(row["IDSW"]), "CountMAE": float(row["CountMAE"])}
    return out


def make_bucket_shift_figure(
    mot_root: Path,
    result_root: Path,
    out_fig: Path,
    out_csv: Path,
    skip_stratified: bool,
) -> None:
    clean_csv = Path("results/main_val/stratified/stratified_metrics_val.csv")
    if not clean_csv.exists():
        raise FileNotFoundError(f"Missing clean stratified baseline CSV: {clean_csv}")
    clean_map = _load_bucket_high(clean_csv, "+gating")

    rows: List[Dict[str, str]] = []
    if skip_stratified:
        for degradation in DEGRADATION_PLAN.keys():
            for crit in ["occlusion", "density", "turn", "low_conf"]:
                rows.append(
                    {
                        "degradation": degradation,
                        "bucket_type": crit,
                        "clean_IDSW": f"{clean_map.get(crit, {}).get('IDSW', float('nan')):.3f}",
                        "high_IDSW": "nan",
                        "delta_IDSW": "nan",
                        "clean_CountMAE": f"{clean_map.get(crit, {}).get('CountMAE', float('nan')):.3f}",
                        "high_CountMAE": "nan",
                        "delta_CountMAE": "nan",
                    }
                )
    else:
        for degradation in DEGRADATION_PLAN.keys():
            pred_dir = result_root / degradation / "high" / "gating" / "pred"
            strat_csv = result_root / "bucket_shift" / f"{degradation}_high_stratified.csv"
            strat_plot = result_root / "bucket_shift" / f"{degradation}_high_stratified.png"
            strat_csv.parent.mkdir(parents=True, exist_ok=True)
            run_stratified_for_gating(mot_root, pred_dir, strat_csv, strat_plot)
            high_map = _load_bucket_high(strat_csv, "+gating")
            for crit in ["occlusion", "density", "turn", "low_conf"]:
                clean_idsw = clean_map.get(crit, {}).get("IDSW", np.nan)
                clean_mae = clean_map.get(crit, {}).get("CountMAE", np.nan)
                high_idsw = high_map.get(crit, {}).get("IDSW", np.nan)
                high_mae = high_map.get(crit, {}).get("CountMAE", np.nan)
                rows.append(
                    {
                        "degradation": degradation,
                        "bucket_type": crit,
                        "clean_IDSW": f"{clean_idsw:.3f}",
                        "high_IDSW": f"{high_idsw:.3f}",
                        "delta_IDSW": f"{(high_idsw - clean_idsw):.3f}",
                        "clean_CountMAE": f"{clean_mae:.3f}",
                        "high_CountMAE": f"{high_mae:.3f}",
                        "delta_CountMAE": f"{(high_mae - clean_mae):.3f}",
                    }
                )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "degradation",
                "bucket_type",
                "clean_IDSW",
                "high_IDSW",
                "delta_IDSW",
                "clean_CountMAE",
                "high_CountMAE",
                "delta_CountMAE",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if skip_stratified:
        return

    plt.rcParams.update(PLOT_STYLE)
    criterions = ["occlusion", "density", "turn", "low_conf"]
    degradations = list(DEGRADATION_PLAN.keys())
    x = np.arange(len(criterions))
    width = 0.22
    colors = {"motion_blur": "#4C78A8", "low_light": "#F58518", "turbidity_haze": "#54A24B"}

    fig, ax = plt.subplots(figsize=(8.8, 4.3), constrained_layout=True)
    for idx, degradation in enumerate(degradations):
        vals = []
        for crit in criterions:
            rec = next(r for r in rows if r["degradation"] == degradation and r["bucket_type"] == crit)
            vals.append(float(rec["delta_IDSW"]))
        ax.bar(x + (idx - 1) * width, vals, width=width, label=degradation.replace("_", " "), color=colors[degradation])

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["occlusion", "density", "turning", "low-conf"])
    ax.set_ylabel(r"$\Delta$IDSW (high stress - clean, +gating)")
    ax.set_title("Bucket-risk migration under high-strength environmental stress")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=1)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, **PLOT_SAVE_KWARGS)
    plt.close(fig)


def copy_to_paper_figs(src: Path, paper_fig_dir: Path) -> Path:
    import shutil

    paper_fig_dir.mkdir(parents=True, exist_ok=True)
    dst = paper_fig_dir / src.name
    shutil.copy2(src, dst)
    return dst


def main() -> int:
    args = parse_args()
    split_dir = args.mot_root / FIXED_SPLIT
    if not split_dir.exists():
        raise FileNotFoundError(f"Prepared split not found: {split_dir}")
    args.result_root.mkdir(parents=True, exist_ok=True)

    write_stress_params_csv(args.params_csv)
    write_stress_params_tex(args.params_tex)

    total_frames = total_frames_for_eval(split_dir, FIXED_MAX_FRAMES, FIXED_SEQUENCES)
    rows: List[Dict[str, str]] = []
    for degradation, levels in DEGRADATION_PLAN.items():
        for level in LEVEL_ORDER:
            params = levels[level]
            for method in METHODS:
                rows.append(
                    run_one_setting(
                        mot_root=args.mot_root,
                        split_dir=split_dir,
                        result_root=args.result_root,
                        method=method,
                        degradation=degradation,
                        level=level,
                        params=params,
                        total_frames=total_frames,
                    )
                )

    rows.sort(key=lambda r: (r["degradation"], LEVEL_ORDER.index(r["level"]), r["method"]))
    write_stress_metrics_csv(args.results_csv, rows)
    make_hota_figure(rows, args.fig_hota)

    bucket_csv = args.result_root / "bucket_shift_summary.csv"
    make_bucket_shift_figure(
        mot_root=args.mot_root,
        result_root=args.result_root,
        out_fig=args.fig_bucket,
        out_csv=bucket_csv,
        skip_stratified=args.skip_stratified,
    )

    dst_hota = copy_to_paper_figs(args.fig_hota, args.paper_fig_dir)
    if args.fig_bucket.exists():
        dst_bucket = copy_to_paper_figs(args.fig_bucket, args.paper_fig_dir)
    else:
        dst_bucket = args.paper_fig_dir / args.fig_bucket.name

    print(f"saved params csv: {args.params_csv}")
    print(f"saved params tex: {args.params_tex}")
    print(f"saved metrics csv: {args.results_csv}")
    print(f"saved fig: {args.fig_hota}")
    print(f"saved fig: {args.fig_bucket}")
    print(f"saved bucket summary csv: {bucket_csv}")
    print(f"copied paper fig: {dst_hota}")
    print(f"copied paper fig: {dst_bucket}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
