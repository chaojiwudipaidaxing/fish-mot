#!/usr/bin/env python
"""Run natural BrackishMOT stress evaluation (clear / turbid-low / turbid-high)."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np


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


LEVEL_MAP = {
    "clear": "low",
    "turbid_low": "mid",
    "turbid_high": "high",
}
LEVEL_ORDER = ["low", "mid", "high"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BrackishMOT natural stress tests.")
    parser.add_argument("--groups-json", type=Path, default=Path("results/brackishmot_groups.json"))
    parser.add_argument("--mot-root", type=Path, default=Path("shuju/archive/BrackishMOT"))
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gating-thresh", type=float, default=2000.0)
    parser.add_argument("--result-root", type=Path, default=Path("results/brackishmot/stress"))
    parser.add_argument("--metrics-csv", type=Path, default=Path("results/brackishmot_stress_metrics.csv"))
    parser.add_argument("--params-tex", type=Path, default=Path("paper/cea_draft/tables/stress_param_table.tex"))
    parser.add_argument(
        "--fig-hota",
        type=Path,
        default=Path("paper/cea_draft/figures/fig_stress_hota_vs_level.pdf"),
    )
    parser.add_argument(
        "--fig-bucket",
        type=Path,
        default=Path("paper/cea_draft/figures/fig_stress_bucket_shift.pdf"),
    )
    return parser.parse_args()


def run_cmd(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_mean_metrics(path: Path) -> Dict[str, float]:
    rows = read_rows(path)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one-row mean CSV: {path} (got {len(rows)})")
    row = rows[0]
    return {
        "HOTA": float(row["HOTA"]),
        "IDF1": float(row["IDF1"]),
        "IDSW": float(row["IDSW"]),
        "DetA": float(row["DetA"]),
        "AssA": float(row["AssA"]),
    }


def read_seq_length(seqinfo_path: Path) -> int:
    import configparser

    parser = configparser.ConfigParser()
    parser.read(seqinfo_path, encoding="utf-8")
    if "Sequence" not in parser:
        raise RuntimeError(f"Missing [Sequence] in {seqinfo_path}")
    seq = parser["Sequence"]
    key = "seqLength" if "seqLength" in seq else "seqlength"
    return int(seq[key])


def write_seqmap(split_dir: Path, split: str, seqs: List[str]) -> Path:
    seqmap = split_dir / "seqmaps" / f"{split}.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\n" + "\n".join(seqs) + "\n", encoding="utf-8")
    return seqmap


def load_frame_counts(mot_txt: Path, max_frame: int) -> np.ndarray:
    counts = np.zeros((max_frame,), dtype=np.float64)
    with mot_txt.open("r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            parts = row.split(",")
            if len(parts) < 1:
                continue
            frame = int(float(parts[0]))
            if 1 <= frame <= max_frame:
                counts[frame - 1] += 1.0
    return counts


def compute_count_metrics(gt_counts: np.ndarray, pred_counts: np.ndarray) -> Dict[str, float]:
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


def aggregate_count_metrics(split_dir: Path, seqs: List[str], pred_dir: Path, max_frames: int) -> Dict[str, float]:
    metrics: List[Dict[str, float]] = []
    for seq in seqs:
        seq_len = read_seq_length(split_dir / seq / "seqinfo.ini")
        use_frames = min(seq_len, max_frames) if max_frames > 0 else seq_len
        gt_counts = load_frame_counts(split_dir / seq / "gt" / "gt.txt", use_frames)
        pred_counts = load_frame_counts(pred_dir / f"{seq}.txt", use_frames)
        metrics.append(compute_count_metrics(gt_counts, pred_counts))
    out: Dict[str, float] = {}
    for key in ["CountMAE", "CountRMSE", "CountVar", "CountDrift"]:
        out[key] = float(np.mean([m[key] for m in metrics]))
    return out


def write_stress_param_table(path: Path, groups: Dict[str, List[Dict[str, object]]]) -> None:
    rows = [("clear", "low"), ("turbid_low", "mid"), ("turbid_high", "high")]
    lines = [
        "% Auto-generated by scripts/run_brackish_stress_tests.py",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Degradation & Strength level & Parameter setting \\",
        r"\midrule",
    ]
    for key, lvl in rows:
        items = groups[key]
        names = ",".join([str(x["name"]) for x in items])
        q_vals = [float(x["quality_score"]) for x in items]
        q_min = min(q_vals)
        q_max = max(q_vals)
        text = (
            f"natural visibility domain & {lvl} & "
            f"BrackishMOT-{key.replace('_', '-')}; seqs={names}; "
            f"quality\\_score range=[{q_min:.3f},{q_max:.3f}]"
        )
        lines.append(text + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def load_bucket_high(path: Path, method: str = "+gating") -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for row in read_rows(path):
        if row.get("method") != method:
            continue
        if row.get("bucket") != "high":
            continue
        crit = row["bucket_type"]
        out[crit] = {"IDSW": float(row["IDSW"]), "CountMAE": float(row["CountMAE"])}
    return out


def make_bucket_shift_figure(
    clear_csv: Path,
    high_csv: Path,
    out_fig: Path,
) -> None:
    clear = load_bucket_high(clear_csv, method="+gating")
    high = load_bucket_high(high_csv, method="+gating")
    crits = ["occlusion", "density", "turn", "low_conf"]
    deltas = []
    for c in crits:
        if c in clear and c in high:
            deltas.append(high[c]["IDSW"] - clear[c]["IDSW"])
        else:
            deltas.append(np.nan)

    plt.rcParams.update(PLOT_STYLE)
    x = np.arange(len(crits), dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    vals = np.asarray(deltas, dtype=float)
    bars = ax.bar(x, np.nan_to_num(vals, nan=0.0), color="#F58518")
    for i, v in enumerate(vals.tolist()):
        if np.isnan(v):
            bars[i].set_alpha(0.2)
            ax.text(i, 0.0, "NA", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["occlusion", "density", "turning", "low-conf"])
    ax.set_ylabel(r"$\Delta$IDSW (high stress - clean, +gating)")
    ax.set_title("Bucket-risk migration under BrackishMOT turbid-high")
    ax.grid(axis="y", alpha=0.25)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, **PLOT_SAVE_KWARGS)
    plt.close(fig)


def make_hota_figure(rows: List[Dict[str, str]], out_fig: Path) -> None:
    plt.rcParams.update(PLOT_STYLE)
    methods = [m.name for m in METHODS]
    x = np.arange(len(LEVEL_ORDER), dtype=float)
    color_map = {"Base": "#4C78A8", "+gating": "#F58518", "ByteTrack": "#54A24B"}
    value_map: Dict[Tuple[str, str], float] = {}
    for r in rows:
        value_map[(r["method"], r["level"])] = float(r["HOTA"])

    fig, ax = plt.subplots(figsize=(6.6, 3.8), constrained_layout=True)
    for method in methods:
        y = [value_map[(method, lvl)] for lvl in LEVEL_ORDER]
        ax.plot(x, y, marker="o", label=method, color=color_map[method])
    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_ORDER)
    ax.set_xlabel("stress level")
    ax.set_ylabel("HOTA (%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, **PLOT_SAVE_KWARGS)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.groups_json.exists():
        raise FileNotFoundError(f"Missing groups JSON: {args.groups_json}")
    groups_obj = json.loads(args.groups_json.read_text(encoding="utf-8"))
    split = str(groups_obj["split"])
    split_dir = args.mot_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    group_levels = {
        "clear": [str(x["name"]) for x in groups_obj["clear"]],
        "turbid_low": [str(x["name"]) for x in groups_obj["turbid_low"]],
        "turbid_high": [str(x["name"]) for x in groups_obj["turbid_high"]],
    }

    seqmap_path = split_dir / "seqmaps" / f"{split}.txt"
    original_seqmap = seqmap_path.read_text(encoding="utf-8") if seqmap_path.exists() else None

    args.result_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, str]] = []
    try:
        for group_key, seqs in group_levels.items():
            level = LEVEL_MAP[group_key]
            write_seqmap(split_dir, split, seqs)
            for method in METHODS:
                run_dir = args.result_root / group_key / method.key
                pred_dir = run_dir / "pred"
                per_seq_csv = run_dir / "per_seq.csv"
                mean_csv = run_dir / "mean.csv"

                cmd_track = [
                    sys.executable,
                    "scripts/run_baseline_sort.py",
                    "--split",
                    split,
                    "--mot-root",
                    str(args.mot_root),
                    "--seqs",
                    *seqs,
                    "--max-frames",
                    str(args.max_frames),
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
                    "0.0",
                    "--jitter",
                    "0.0",
                    "--degrade-seed",
                    str(args.seed),
                    "--out-dir",
                    str(pred_dir),
                    "--clean-out",
                ]
                if method.name == "+gating":
                    cmd_track.extend(["--gating-thresh", f"{args.gating_thresh:.6f}"])
                run_cmd(cmd_track)

                cmd_eval = [
                    sys.executable,
                    "scripts/eval_trackeval_per_seq.py",
                    "--split",
                    split,
                    "--mot-root",
                    str(args.mot_root),
                    "--pred-dir",
                    str(pred_dir),
                    "--tracker-name",
                    f"brackish_{group_key}_{method.key}",
                    "--max-frames",
                    str(args.max_frames if args.max_frames > 0 else 0),
                    "--results-per-seq",
                    str(per_seq_csv),
                    "--results-mean",
                    str(mean_csv),
                ]
                run_cmd(cmd_eval)

                mot = parse_mean_metrics(mean_csv)
                cnt = aggregate_count_metrics(split_dir, seqs, pred_dir, max_frames=args.max_frames)
                rows.append(
                    {
                        "method": method.name,
                        "group": group_key,
                        "level": level,
                        "HOTA": f"{mot['HOTA']:.3f}",
                        "IDF1": f"{mot['IDF1']:.3f}",
                        "IDSW": f"{mot['IDSW']:.3f}",
                        "DetA": f"{mot['DetA']:.3f}",
                        "AssA": f"{mot['AssA']:.3f}",
                        "CountMAE": f"{cnt['CountMAE']:.3f}",
                        "CountRMSE": f"{cnt['CountRMSE']:.3f}",
                        "CountVar": f"{cnt['CountVar']:.3f}",
                        "CountDrift": f"{cnt['CountDrift']:.3f}",
                        "split": split,
                        "seqs": "|".join(seqs),
                    }
                )
                print(
                    "[brackish-stress]",
                    group_key,
                    method.name,
                    f"HOTA={mot['HOTA']:.3f}",
                    f"IDF1={mot['IDF1']:.3f}",
                    f"IDSW={mot['IDSW']:.3f}",
                )

        rows.sort(key=lambda r: (LEVEL_ORDER.index(r["level"]), r["method"]))
        args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "method",
            "group",
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
            "split",
            "seqs",
        ]
        with args.metrics_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        write_stress_param_table(args.params_tex, groups_obj)
        make_hota_figure(rows, args.fig_hota)

        clear_base = args.result_root / "clear" / "base" / "pred"
        clear_gating = args.result_root / "clear" / "gating" / "pred"
        high_base = args.result_root / "turbid_high" / "base" / "pred"
        high_gating = args.result_root / "turbid_high" / "gating" / "pred"
        strat_clear_csv = args.result_root / "stratified_clear.csv"
        strat_high_csv = args.result_root / "stratified_turbid_high.csv"
        strat_clear_plot = args.result_root / "stratified_clear.png"
        strat_high_plot = args.result_root / "stratified_turbid_high.png"

        write_seqmap(split_dir, split, group_levels["clear"])
        run_cmd(
            [
                sys.executable,
                "scripts/eval_stratified.py",
                "--split",
                split,
                "--mot-root",
                str(args.mot_root),
                "--pred-base",
                str(clear_base),
                "--pred-gating",
                str(clear_gating),
                "--pred-traj",
                str(clear_base),
                "--pred-adaptive",
                str(clear_base),
                "--max-frames",
                str(args.max_frames),
                "--bucket-mode",
                "quantile",
                "--bucket-min-samples",
                "50",
                "--output-csv",
                str(strat_clear_csv),
                "--plot-path",
                str(strat_clear_plot),
            ]
        )
        write_seqmap(split_dir, split, group_levels["turbid_high"])
        run_cmd(
            [
                sys.executable,
                "scripts/eval_stratified.py",
                "--split",
                split,
                "--mot-root",
                str(args.mot_root),
                "--pred-base",
                str(high_base),
                "--pred-gating",
                str(high_gating),
                "--pred-traj",
                str(high_base),
                "--pred-adaptive",
                str(high_base),
                "--max-frames",
                str(args.max_frames),
                "--bucket-mode",
                "quantile",
                "--bucket-min-samples",
                "50",
                "--output-csv",
                str(strat_high_csv),
                "--plot-path",
                str(strat_high_plot),
            ]
        )
        make_bucket_shift_figure(strat_clear_csv, strat_high_csv, args.fig_bucket)

    finally:
        if original_seqmap is None:
            if seqmap_path.exists():
                seqmap_path.unlink()
        else:
            seqmap_path.write_text(original_seqmap, encoding="utf-8")

    print(f"wrote metrics: {args.metrics_csv}")
    print(f"wrote stress table: {args.params_tex}")
    print(f"wrote figure: {args.fig_hota}")
    print(f"wrote figure: {args.fig_bucket}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
