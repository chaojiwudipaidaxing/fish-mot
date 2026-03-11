#!/usr/bin/env python
"""Run strong baseline comparisons (multi-seed) and merge with val main table."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from method_labels import MAIN_CHAIN_METHOD_ORDER, normalize_main_chain_label


METRIC_FIELDS = ["HOTA", "DetA", "AssA", "IDF1", "IDSW"]
EXISTING_METHOD_ORDER = MAIN_CHAIN_METHOD_ORDER
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

STRONG_METHOD_CONFIGS: Dict[str, Dict[str, str]] = {
    "bytetrack": {
        "display": "ByteTrack",
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
    "ocsort": {
        "display": "OC-SORT",
        "gating": "on",
        "traj": "off",
        "adaptive_gamma": "off",
        "alpha": "1.0",
        "beta": "0.02",
        "gamma": "0.0",
        "iou_thresh": "0.3",
        "min_hits": "1",
        "max_age": "30",
    },
    "botsort": {
        "display": "BoT-SORT",
        "gating": "on",
        "traj": "on",
        "adaptive_gamma": "on",
        "alpha": "1.0",
        "beta": "0.02",
        "gamma": "0.5",
        "iou_thresh": "0.3",
        "min_hits": "1",
        "max_age": "30",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strong baselines (multi-seed) and aggregate with val main table.")
    parser.add_argument("--split", default="val_half", help="Prepared split name under data/mft25_mot.")
    parser.add_argument(
        "--run-config",
        type=Path,
        default=None,
        help="Optional run_config.json. If provided, result_root paths are used by default.",
    )
    parser.add_argument(
        "--mot-root",
        type=Path,
        default=Path("data/mft25_mot"),
        help="Root directory of prepared MOT data.",
    )
    parser.add_argument(
        "--methods",
        default="bytetrack,ocsort,botsort",
        help="Comma separated strong methods from {bytetrack,ocsort,botsort}.",
    )
    parser.add_argument("--max-frames", type=int, default=1000, help="Frame cap for val run.")
    parser.add_argument("--max-gt-ids", type=int, default=50000, help="Pass-through to eval_trackeval_per_seq.py.")
    parser.add_argument(
        "--seeds",
        default="",
        help="Comma separated seed list for multi-seed run, e.g. 0,1,2. Empty means use --seed.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Single seed fallback when --seeds is empty.")
    parser.add_argument(
        "--det-source",
        choices=["auto", "det", "gt"],
        default="auto",
        help="Detection source shared by all methods (for fair comparison).",
    )
    parser.add_argument("--drop-rate", type=float, default=0.2, help="Detection drop rate.")
    parser.add_argument("--jitter", type=float, default=0.02, help="Detection bbox jitter ratio.")
    parser.add_argument("--adaptive-gamma-min", type=float, default=0.5, help="Lower clamp for adaptive gamma.")
    parser.add_argument("--adaptive-gamma-max", type=float, default=2.0, help="Upper clamp for adaptive gamma.")
    parser.add_argument(
        "--frame-stats",
        choices=["on", "off"],
        default="off",
        help="Forwarded to run_baseline_sort for per-frame debug logging.",
    )
    parser.add_argument(
        "--traj-encoder",
        type=Path,
        default=Path("runs/traj_encoder/traj_encoder.pt"),
        help="Trajectory encoder path (used when botsort is selected).",
    )
    parser.add_argument(
        "--prepare",
        choices=["yes", "no"],
        default="yes",
        help="Run prepare_mft25.py before baseline execution.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/strong_baselines"),
        help="Strong baseline output root.",
    )
    parser.add_argument(
        "--existing-main-mean-csv",
        type=Path,
        default=Path("results/main_table_val_seedmean.csv"),
        help="Existing Base/Base+gating/Base+gating+traj/Base+gating+traj+adaptive seed-mean CSV.",
    )
    parser.add_argument(
        "--existing-main-std-csv",
        type=Path,
        default=Path("results/main_table_val_seedstd.csv"),
        help="Existing Base/Base+gating/Base+gating+traj/Base+gating+traj+adaptive seed-std CSV.",
    )
    # Backward compatible alias from previous script version.
    parser.add_argument(
        "--existing-main-csv",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--existing-seed-root",
        type=Path,
        default=Path("results/main_val/seed_runs"),
        help="Fallback root to evaluate existing methods when mean/std CSV is missing.",
    )
    # Backward compatible alias from previous script version.
    parser.add_argument(
        "--existing-seed",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--strong-mean-out",
        type=Path,
        default=Path("results/strong_baselines_seedmean.csv"),
        help="Output CSV path for strong baseline seed-mean metrics.",
    )
    parser.add_argument(
        "--strong-std-out",
        type=Path,
        default=Path("results/strong_baselines_seedstd.csv"),
        help="Output CSV path for strong baseline seed-std metrics.",
    )
    parser.add_argument(
        "--strong-plot-out",
        type=Path,
        default=Path("results/paper_assets_val/strong_baselines_seedmean_std.png"),
        help="Strong baselines mean+-std plot path.",
    )
    parser.add_argument(
        "--strong-tex-out",
        type=Path,
        default=Path("results/paper_assets_val/strong_baselines_seedmean_std.tex"),
        help="Strong baselines mean+-std LaTeX table path.",
    )
    parser.add_argument(
        "--table-out",
        type=Path,
        default=Path("results/main_table_val_with_baselines.csv"),
        help="Combined CSV output path (mean+-std format).",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=Path("results/paper_assets_val/main_table_with_baselines.png"),
        help="Combined mean+-std plot output path.",
    )
    parser.add_argument(
        "--tex-out",
        type=Path,
        default=Path("results/paper_assets_val/main_table_with_baselines.tex"),
        help="Combined mean+-std LaTeX table output path.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Skip tracker runs and render tables/figures from existing mean/std CSV files.",
    )
    return parser.parse_args()


def _norm(path: Path) -> str:
    return str(path).replace("\\", "/").lower()


def _is_legacy_output_path(path: Path, result_root: Path) -> bool:
    norm = _norm(path)
    root_norm = _norm(result_root)
    return norm.startswith("results/") and norm != root_norm and not norm.startswith(root_norm + "/")


def apply_run_config(args: argparse.Namespace) -> None:
    if args.run_config is None:
        return
    if not args.run_config.exists():
        raise FileNotFoundError(f"Run config not found: {args.run_config}")
    cfg = json.loads(args.run_config.read_text(encoding="utf-8"))
    args.split = str(cfg.get("split", args.split))
    args.max_frames = int(cfg.get("max_frames", args.max_frames))
    args.drop_rate = float(cfg.get("drop_rate", args.drop_rate))
    args.jitter = float(cfg.get("jitter", args.jitter))
    args.mot_root = Path(str(cfg.get("mot_root", args.mot_root)))

    if args.seeds.strip() == "":
        seeds_cfg = cfg.get("seeds")
        if isinstance(seeds_cfg, list) and seeds_cfg:
            args.seeds = ",".join(str(int(x)) for x in seeds_cfg)
        elif isinstance(seeds_cfg, str) and seeds_cfg.strip():
            args.seeds = seeds_cfg

    result_root = Path(str(cfg.get("result_root", "results/main_val")))
    tables_dir = Path(str(cfg.get("tables_dir", result_root / "tables")))
    paper_assets_dir = Path(str(cfg.get("paper_assets_dir", result_root / "paper_assets")))
    strong_root = result_root / "strong_baselines"
    seed_root = Path(str(cfg.get("seed_root", result_root / "seed_runs")))

    if _norm(args.output_root) == _norm(Path("results/strong_baselines")):
        args.output_root = strong_root
    if _norm(args.existing_main_mean_csv) == _norm(Path("results/main_table_val_seedmean.csv")):
        args.existing_main_mean_csv = tables_dir / "main_table_val_seedmean.csv"
    if _norm(args.existing_main_std_csv) == _norm(Path("results/main_table_val_seedstd.csv")):
        args.existing_main_std_csv = tables_dir / "main_table_val_seedstd.csv"
    if _norm(args.existing_seed_root) == _norm(Path("results/main_val/seed_runs")):
        args.existing_seed_root = seed_root
    if _norm(args.strong_mean_out) == _norm(Path("results/strong_baselines_seedmean.csv")):
        args.strong_mean_out = tables_dir / "strong_baselines_seedmean.csv"
    if _norm(args.strong_std_out) == _norm(Path("results/strong_baselines_seedstd.csv")):
        args.strong_std_out = tables_dir / "strong_baselines_seedstd.csv"
    if _norm(args.strong_plot_out) == _norm(Path("results/paper_assets_val/strong_baselines_seedmean_std.png")):
        args.strong_plot_out = paper_assets_dir / "strong_baselines_seedmean_std.png"
    if _norm(args.strong_tex_out) == _norm(Path("results/paper_assets_val/strong_baselines_seedmean_std.tex")):
        args.strong_tex_out = paper_assets_dir / "strong_baselines_seedmean_std.tex"
    if _norm(args.table_out) == _norm(Path("results/main_table_val_with_baselines.csv")):
        args.table_out = tables_dir / "main_table_val_with_baselines.csv"
    if _norm(args.plot_out) == _norm(Path("results/paper_assets_val/main_table_with_baselines.png")):
        args.plot_out = paper_assets_dir / "main_table_with_baselines.png"
    if _norm(args.tex_out) == _norm(Path("results/paper_assets_val/main_table_with_baselines.tex")):
        args.tex_out = paper_assets_dir / "main_table_with_baselines.tex"

    write_targets = [
        ("output_root", args.output_root),
        ("strong_mean_out", args.strong_mean_out),
        ("strong_std_out", args.strong_std_out),
        ("strong_plot_out", args.strong_plot_out),
        ("strong_tex_out", args.strong_tex_out),
        ("table_out", args.table_out),
        ("plot_out", args.plot_out),
        ("tex_out", args.tex_out),
    ]
    bad = [f"{name}={path}" for name, path in write_targets if _is_legacy_output_path(path, result_root)]
    if bad:
        raise RuntimeError(
            "run-config strict mode blocks legacy/global outputs. "
            f"result_root={result_root}. bad targets: {'; '.join(bad)}"
        )


def run_cmd(cmd: Sequence[str]) -> None:
    print("[run]", " ".join(str(x) for x in cmd))
    subprocess.run(list(cmd), check=True)


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_single_row(path: Path) -> Dict[str, str]:
    rows = read_rows(path)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row in {path}, got {len(rows)}")
    return rows[0]


def parse_metric_value(value: str) -> float:
    text = str(value).strip()
    if not text:
        return 0.0
    for token in ("+-", "±", "\\pm"):
        if token in text:
            text = text.split(token, 1)[0].strip()
            break
    return float(text)


def parse_seed_list(raw: str, fallback_seed: int) -> List[int]:
    if raw.strip():
        tokens = [tok.strip() for tok in raw.split(",") if tok.strip()]
    else:
        tokens = [str(fallback_seed)]
    seeds: List[int] = []
    seen = set()
    for token in tokens:
        seed = int(token)
        if seed not in seen:
            seeds.append(seed)
            seen.add(seed)
    if not seeds:
        raise ValueError("No valid seeds parsed from --seeds/--seed.")
    return seeds


def format_pm(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def format_uncertainty_s(mean: float, std: float, decimals: int = 3) -> str:
    scaled_std = int(round(abs(std) * (10**decimals)))
    return f"{mean:.{decimals}f}({scaled_std})"


def apply_plot_style() -> None:
    plt.rcParams.update(PLOT_STYLE)


def make_stats_row(split: str, method: str, means: Dict[str, float], stds: Dict[str, float]) -> Dict[str, object]:
    row: Dict[str, object] = {"split": split, "method": method}
    for metric in METRIC_FIELDS:
        row[f"{metric}_mean"] = float(means.get(metric, 0.0))
        row[f"{metric}_std"] = float(stds.get(metric, 0.0))
    return row


def sort_stats_rows(rows: List[Dict[str, object]], strong_display_order: List[str]) -> List[Dict[str, object]]:
    ordered = EXISTING_METHOD_ORDER + strong_display_order
    rank = {name: i for i, name in enumerate(ordered)}
    return sorted(rows, key=lambda r: (rank.get(str(r["method"]), 999), str(r["method"])))


def write_metric_csv(path: Path, rows: List[Dict[str, object]], mode: str) -> None:
    if mode not in {"mean", "std"}:
        raise ValueError(f"Unknown mode: {mode}")
    fields = ["split", "method"] + METRIC_FIELDS
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {"split": row["split"], "method": row["method"]}
            for metric in METRIC_FIELDS:
                out[metric] = f"{float(row[f'{metric}_{mode}']):.3f}"
            writer.writerow(out)


def write_mean_pm_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    metric_string_fields = METRIC_FIELDS
    metric_numeric_fields: List[str] = []
    for metric in METRIC_FIELDS:
        metric_numeric_fields.extend([f"{metric}_mean", f"{metric}_std"])
    fields = ["split", "method"] + metric_string_fields + metric_numeric_fields
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out: Dict[str, str] = {
                "split": str(row["split"]),
                "method": str(row["method"]),
            }
            for metric in METRIC_FIELDS:
                mean = float(row[f"{metric}_mean"])
                std = float(row[f"{metric}_std"])
                out[metric] = format_pm(mean, std)
                out[f"{metric}_mean"] = f"{mean:.3f}"
                out[f"{metric}_std"] = f"{std:.3f}"
            writer.writerow(out)


def write_latex_pm_table(path: Path, rows: List[Dict[str, object]], comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"% {comment}\n")
        f.write("\\begin{tabular}{l S S S S S}\n")
        f.write("\\toprule\n")
        f.write("Method & {HOTA} & {DetA} & {AssA} & {IDF1} & {IDSW} \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            values = []
            for metric in METRIC_FIELDS:
                values.append(
                    format_uncertainty_s(
                        float(row[f"{metric}_mean"]),
                        float(row[f"{metric}_std"]),
                    )
                )
            f.write(
                f"{row['method']} & {values[0]} & {values[1]} & "
                f"{values[2]} & {values[3]} & {values[4]} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def make_plot_with_errorbars(path: Path, rows: List[Dict[str, object]], title: str) -> None:
    apply_plot_style()
    methods = [str(row["method"]) for row in rows]
    metrics_for_plot = ["HOTA", "AssA", "IDF1"]
    x = np.arange(len(methods))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.2, 4.9), constrained_layout=True)
    for idx, metric in enumerate(metrics_for_plot):
        means = [float(row[f"{metric}_mean"]) for row in rows]
        stds = [float(row[f"{metric}_std"]) for row in rows]
        ax.bar(
            x + (idx - 1) * width,
            means,
            width=width,
            yerr=stds,
            capsize=2.5,
            linewidth=0.5,
            label=metric,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="upper right", frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **PLOT_SAVE_KWARGS)
    plt.close(fig)


def prepare_split_if_needed(args: argparse.Namespace) -> None:
    if args.prepare != "yes":
        return
    cmd = [
        sys.executable,
        "scripts/prepare_mft25.py",
        "--out-root",
        str(args.mot_root),
        "--splits",
        args.split,
        "--max-frames",
        str(args.max_frames),
        "--clean-split",
    ]
    if args.run_config is not None:
        cmd.extend(["--run-config", str(args.run_config)])
    run_cmd(cmd)


def run_one_strong_method(
    args: argparse.Namespace,
    method_key: str,
    seed: int,
    method_rank: int,
) -> Dict[str, object]:
    cfg = STRONG_METHOD_CONFIGS[method_key]
    seed_dir = args.output_root / method_key / f"seed_{seed}"
    pred_dir = seed_dir / "pred"
    per_seq_csv = seed_dir / "per_seq.csv"
    mean_csv = seed_dir / "mean.csv"

    cmd = [
        sys.executable,
        "scripts/run_baseline_sort.py",
        "--split",
        args.split,
        "--mot-root",
        str(args.mot_root),
        "--max-frames",
        str(args.max_frames),
        "--det-source",
        args.det_source,
        "--gating",
        cfg["gating"],
        "--traj",
        cfg["traj"],
        "--adaptive-gamma",
        cfg["adaptive_gamma"],
        "--alpha",
        cfg["alpha"],
        "--beta",
        cfg["beta"],
        "--gamma",
        cfg["gamma"],
        "--adaptive-gamma-min",
        str(args.adaptive_gamma_min),
        "--adaptive-gamma-max",
        str(args.adaptive_gamma_max),
        "--frame-stats",
        args.frame_stats,
        "--iou-thresh",
        cfg["iou_thresh"],
        "--min-hits",
        cfg["min_hits"],
        "--max-age",
        cfg["max_age"],
        "--drop-rate",
        str(args.drop_rate),
        "--jitter",
        str(args.jitter),
        "--degrade-seed",
        str(seed + method_rank * 1000),
        "--out-dir",
        str(pred_dir),
        "--clean-out",
    ]
    if args.run_config is not None:
        cmd.extend(["--run-config", str(args.run_config)])
    if cfg["traj"] == "on":
        if not args.traj_encoder.exists():
            raise FileNotFoundError(f"Trajectory encoder not found: {args.traj_encoder}")
        cmd.extend(["--traj-encoder", str(args.traj_encoder)])
    run_cmd(cmd)

    eval_cmd = [
        sys.executable,
        "scripts/eval_trackeval_per_seq.py",
        "--split",
        args.split,
        "--mot-root",
        str(args.mot_root),
        "--pred-dir",
        str(pred_dir),
        "--tracker-name",
        f"strong_{method_key}_s{seed}",
        "--max-frames",
        str(args.max_frames),
        "--max-gt-ids",
        str(args.max_gt_ids),
        "--results-per-seq",
        str(per_seq_csv),
        "--results-mean",
        str(mean_csv),
    ]
    if args.run_config is not None:
        eval_cmd.extend(["--run-config", str(args.run_config)])
    run_cmd(eval_cmd)

    mean_row = read_single_row(mean_csv)
    out: Dict[str, object] = {
        "split": mean_row.get("split", args.split),
        "method": cfg["display"],
        "seed": seed,
        "pred_dir": pred_dir,
        "mean_csv": mean_csv,
        "per_seq_csv": per_seq_csv,
    }
    for metric in METRIC_FIELDS:
        out[metric] = parse_metric_value(mean_row[metric])
    print(
        f"[done][seed={seed}] {cfg['display']}: HOTA={float(out['HOTA']):.3f} "
        f"DetA={float(out['DetA']):.3f} AssA={float(out['AssA']):.3f} "
        f"IDF1={float(out['IDF1']):.3f} IDSW={float(out['IDSW']):.3f}"
    )
    return out


def aggregate_seed_runs(rows: List[Dict[str, object]], ordered_methods: List[str]) -> List[Dict[str, object]]:
    by_method: Dict[str, Dict[str, List[float]]] = {}
    split_by_method: Dict[str, str] = {}
    for row in rows:
        method = str(row["method"])
        if method not in by_method:
            by_method[method] = {metric: [] for metric in METRIC_FIELDS}
        split_by_method[method] = str(row.get("split", "val_half"))
        for metric in METRIC_FIELDS:
            by_method[method][metric].append(float(row[metric]))

    aggregated: List[Dict[str, object]] = []
    for method in ordered_methods:
        values = by_method.get(method)
        if not values:
            continue
        means = {metric: float(np.mean(values[metric])) for metric in METRIC_FIELDS}
        stds = {metric: float(np.std(values[metric], ddof=0)) for metric in METRIC_FIELDS}
        aggregated.append(make_stats_row(split_by_method.get(method, "val_half"), method, means, stds))
    return aggregated


def load_stats_from_mean_std_csv(
    mean_csv: Path,
    std_csv: Path,
    split_default: str,
) -> Dict[str, Dict[str, object]]:
    if not mean_csv.exists():
        return {}
    mean_rows = read_rows(mean_csv)
    std_map = {}
    if std_csv.exists():
        for row in read_rows(std_csv):
            method = normalize_main_chain_label(row.get("method", ""))
            copied = dict(row)
            copied["method"] = method
            std_map[method] = copied

    out: Dict[str, Dict[str, object]] = {}
    for row in mean_rows:
        method = normalize_main_chain_label(row.get("method", ""))
        if not method:
            continue
        std_row = std_map.get(method, {})
        means = {metric: parse_metric_value(row.get(metric, "0")) for metric in METRIC_FIELDS}
        stds = {metric: parse_metric_value(std_row.get(metric, "0")) for metric in METRIC_FIELDS}
        out[method] = make_stats_row(row.get("split", split_default), method, means, stds)
    return out


def load_ordered_stats(
    mean_csv: Path,
    std_csv: Path,
    split_default: str,
    method_order: List[str],
) -> List[Dict[str, object]]:
    stats_map = load_stats_from_mean_std_csv(mean_csv, std_csv, split_default=split_default)
    return [stats_map[m] for m in method_order if m in stats_map]


def eval_existing_rows_fallback(args: argparse.Namespace, seeds: List[int]) -> Dict[str, Dict[str, object]]:
    mapping = [
        (MAIN_CHAIN_METHOD_ORDER[0], "pred_base"),
        (MAIN_CHAIN_METHOD_ORDER[1], "pred_gating"),
        (MAIN_CHAIN_METHOD_ORDER[2], "pred_traj"),
        (MAIN_CHAIN_METHOD_ORDER[3], "pred_adaptive"),
    ]
    out_root = args.output_root / "_existing_eval"
    collected: List[Dict[str, object]] = []
    for seed in seeds:
        seed_root = args.existing_seed_root / f"seed_{seed}"
        for method_name, pred_subdir in mapping:
            pred_dir = seed_root / pred_subdir
            if not pred_dir.exists():
                raise FileNotFoundError(
                    f"Missing fallback prediction dir for {method_name}: {pred_dir}. "
                    f"Provide {args.existing_main_mean_csv} / {args.existing_main_std_csv} "
                    "or valid --existing-seed-root."
                )
            mean_csv = out_root / f"seed_{seed}" / f"mean_{pred_subdir}.csv"
            per_seq_csv = out_root / f"seed_{seed}" / f"per_seq_{pred_subdir}.csv"
            eval_cmd = [
                sys.executable,
                "scripts/eval_trackeval_per_seq.py",
                "--split",
                args.split,
                "--mot-root",
                str(args.mot_root),
                "--pred-dir",
                str(pred_dir),
                "--tracker-name",
                f"existing_{pred_subdir}_s{seed}",
                "--max-frames",
                str(args.max_frames),
                "--max-gt-ids",
                str(args.max_gt_ids),
                "--results-per-seq",
                str(per_seq_csv),
                "--results-mean",
                str(mean_csv),
            ]
            if args.run_config is not None:
                eval_cmd.extend(["--run-config", str(args.run_config)])
            run_cmd(eval_cmd)
            mean_row = read_single_row(mean_csv)
            metric_row: Dict[str, object] = {
                "split": mean_row.get("split", args.split),
                "method": method_name,
                "seed": seed,
            }
            for metric in METRIC_FIELDS:
                metric_row[metric] = parse_metric_value(mean_row[metric])
            collected.append(metric_row)

    aggregated = aggregate_seed_runs(collected, EXISTING_METHOD_ORDER)
    return {str(row["method"]): row for row in aggregated}


def main() -> None:
    args = parse_args()
    if args.existing_main_csv is not None:
        # Keep compatibility with old callsites that only provided mean CSV.
        args.existing_main_mean_csv = args.existing_main_csv
    apply_run_config(args)

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    if len(methods) < 2:
        raise ValueError("Need at least 2 strong baselines in --methods.")
    unknown = [m for m in methods if m not in STRONG_METHOD_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown strong method(s): {unknown}; valid={sorted(STRONG_METHOD_CONFIGS)}")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be > 0.")
    if args.adaptive_gamma_min <= 0.0:
        raise ValueError("--adaptive-gamma-min must be > 0.")
    if args.adaptive_gamma_max < args.adaptive_gamma_min:
        raise ValueError("--adaptive-gamma-max must be >= --adaptive-gamma-min.")

    seeds = parse_seed_list(args.seeds, args.seed)
    print(
        f"[config] split={args.split} max_frames={args.max_frames} methods={methods} seeds={seeds} "
        f"gamma_clamp=[{args.adaptive_gamma_min:.3f},{args.adaptive_gamma_max:.3f}] "
        f"frame_stats={args.frame_stats} render_only={args.render_only}"
    )

    strong_display_order: List[str] = []
    for method_idx, method_key in enumerate(methods):
        display = STRONG_METHOD_CONFIGS[method_key]["display"]
        strong_display_order.append(display)

    strong_stats: List[Dict[str, object]]
    if args.render_only:
        strong_stats = load_ordered_stats(
            args.strong_mean_out,
            args.strong_std_out,
            split_default=args.split,
            method_order=strong_display_order,
        )
        if len(strong_stats) != len(strong_display_order):
            missing = [m for m in strong_display_order if m not in {str(r["method"]) for r in strong_stats}]
            raise FileNotFoundError(
                "Missing strong baseline mean/std rows for render-only mode: "
                f"{missing}; expected files {args.strong_mean_out} and {args.strong_std_out}"
            )
    else:
        prepare_split_if_needed(args)

        split_dir = args.mot_root / args.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Prepared split not found: {split_dir}")
        args.output_root.mkdir(parents=True, exist_ok=True)

        strong_seed_rows: List[Dict[str, object]] = []
        for method_idx, method_key in enumerate(methods):
            display = STRONG_METHOD_CONFIGS[method_key]["display"]
            for seed in seeds:
                row = run_one_strong_method(args, method_key=method_key, seed=seed, method_rank=method_idx)
                strong_seed_rows.append(row)
                pred_dir = args.output_root / method_key / f"seed_{seed}" / "pred"
                print(
                    f"[saved] method={display} seed={seed} pred={pred_dir} "
                    f"mean={args.output_root / method_key / f'seed_{seed}' / 'mean.csv'}"
                )

        strong_stats = aggregate_seed_runs(strong_seed_rows, strong_display_order)
        write_metric_csv(args.strong_mean_out, strong_stats, mode="mean")
        write_metric_csv(args.strong_std_out, strong_stats, mode="std")

    make_plot_with_errorbars(
        args.strong_plot_out,
        strong_stats,
        title=f"Strong baselines on {args.split.replace('_', '-')} (mean ± std over seeds)",
    )
    write_latex_pm_table(
        args.strong_tex_out,
        strong_stats,
        comment="Auto-generated by scripts/run_strong_baselines.py",
    )

    existing_stats_map = load_stats_from_mean_std_csv(
        args.existing_main_mean_csv,
        args.existing_main_std_csv,
        split_default=args.split,
    )
    missing_core = [method for method in EXISTING_METHOD_ORDER if method not in existing_stats_map]
    if missing_core:
        print(
            "[warn] Missing Base/Base+gating/Base+gating+traj/Base+gating+traj+adaptive mean/std rows, fallback to per-seed evaluation. "
            f"missing={missing_core}"
        )
        fallback_stats = eval_existing_rows_fallback(args, seeds)
        existing_stats_map.update(fallback_stats)

    existing_stats: List[Dict[str, object]] = []
    for method in EXISTING_METHOD_ORDER:
        row = existing_stats_map.get(method)
        if row is not None:
            existing_stats.append(row)
        else:
            print(f"[warn] Existing method missing and skipped: {method}")

    combined_rows = sort_stats_rows(existing_stats + strong_stats, strong_display_order=strong_display_order)
    write_mean_pm_csv(args.table_out, combined_rows)
    make_plot_with_errorbars(
        args.plot_out,
        combined_rows,
        title=f"Main table with strong baselines ({args.split.replace('_', '-')}, mean ± std)",
    )
    write_latex_pm_table(
        args.tex_out,
        combined_rows,
        comment="Auto-generated by scripts/run_strong_baselines.py",
    )

    print(f"Saved strong mean:      {args.strong_mean_out}")
    print(f"Saved strong std:       {args.strong_std_out}")
    print(f"Saved strong plot:      {args.strong_plot_out}")
    print(f"Saved strong tex:       {args.strong_tex_out}")
    print(f"Saved combined table:   {args.table_out}")
    print(f"Saved combined plot:    {args.plot_out}")
    print(f"Saved combined tex:     {args.tex_out}")


if __name__ == "__main__":
    main()
