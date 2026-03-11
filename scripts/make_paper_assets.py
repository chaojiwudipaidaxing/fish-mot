#!/usr/bin/env python
"""Build paper-ready tables/figures from existing CSV results."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from method_labels import MAIN_CHAIN_METHOD_ORDER, normalize_main_chain_label


METHOD_ORDER = MAIN_CHAIN_METHOD_ORDER
STRONG_METHOD_ORDER = ["ByteTrack", "OC-SORT", "BoT-SORT"]
MAIN_METRICS = ["HOTA", "DetA", "AssA", "IDF1", "IDSW"]
COUNT_METRICS = ["CountMAE", "CountRMSE", "CountVar", "CountDrift"]
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
PLACEHOLDER_PROVENANCE_TOKENS = ("TESTCOMMIT", "PLACEHOLDER", "CHANGEME", "TODO")
ANON_BUNDLE_RE = re.compile(r"anonymous_bundle_[A-Za-z0-9._-]+$")
GIT_HASH_RE = re.compile(r"[0-9a-fA-F]{7,40}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper assets from existing result CSVs.")
    parser.add_argument(
        "--main-mean",
        type=Path,
        default=Path("results/main_table_val_seedmean.csv"),
        help="Main table seed-mean CSV.",
    )
    parser.add_argument(
        "--main-std",
        type=Path,
        default=Path("results/main_table_val_seedstd.csv"),
        help="Main table seed-std CSV.",
    )
    parser.add_argument(
        "--count-csv",
        type=Path,
        default=Path("results/count_metrics_val.csv"),
        help="Count stability CSV.",
    )
    parser.add_argument(
        "--strong-mean",
        type=Path,
        default=Path("results/strong_baselines_seedmean.csv"),
        help="Strong baseline seed-mean CSV.",
    )
    parser.add_argument(
        "--strong-std",
        type=Path,
        default=Path("results/strong_baselines_seedstd.csv"),
        help="Strong baseline seed-std CSV.",
    )
    parser.add_argument(
        "--with-baselines-csv",
        type=Path,
        default=Path("results/main_table_val_with_baselines.csv"),
        help="Merged main+strong table path (auto-rebuilt when stale).",
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=Path("results/main_val/run_config.json"),
        help="Primary run_config.json path.",
    )
    parser.add_argument(
        "--no-auto-rebuild-baselines",
        action="store_true",
        help="Disable stale-check auto rebuild for main_table_val_with_baselines.csv.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Clean output/release targets before generating assets.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/paper_assets_val"),
        help="Output directory for paper assets.",
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=Path("release"),
        help="Release directory for manifest and reproduce script.",
    )
    parser.add_argument(
        "--degradation-csv",
        type=Path,
        default=None,
        help="Optional degradation grid CSV (included in manifest/readme if present).",
    )
    parser.add_argument(
        "--degradation-plot",
        type=Path,
        default=None,
        help="Optional degradation grid PNG (included in manifest/readme if present).",
    )
    parser.add_argument(
        "--degradation-tex",
        type=Path,
        default=None,
        help="Optional degradation grid TEX (included in manifest/readme if present).",
    )
    return parser.parse_args()


def _norm_path(path: Path) -> str:
    return str(path).replace("\\", "/").lower()


def _is_legacy_output_path(path: Path, result_root: Path) -> bool:
    norm = _norm_path(path)
    root_norm = _norm_path(result_root)
    if norm == "release" or norm.startswith("release/"):
        return True
    if norm.startswith("results/") and norm != root_norm and not norm.startswith(root_norm + "/"):
        return True
    return False


def assert_no_legacy_write_paths(args: argparse.Namespace, run_cfg: Dict[str, object]) -> None:
    result_root = Path(str(run_cfg.get("result_root", "results/main_val")))
    write_targets = [
        ("with_baselines_csv", args.with_baselines_csv),
        ("out_dir", args.out_dir),
        ("release_dir", args.release_dir),
    ]
    bad: List[Tuple[str, Path]] = []
    for name, path in write_targets:
        if _is_legacy_output_path(path, result_root):
            bad.append((name, path))
    if bad:
        detail = "; ".join(f"{name}={path}" for name, path in bad)
        raise RuntimeError(
            "run-config strict mode blocks legacy/global outputs. "
            f"result_root={result_root}. bad targets: {detail}"
        )


def apply_run_config_paths(args: argparse.Namespace, run_cfg: Dict[str, object]) -> None:
    result_root = Path(str(run_cfg.get("result_root", "results/main_val")))
    tables_dir = Path(str(run_cfg.get("tables_dir", result_root / "tables")))
    paper_assets_dir = Path(str(run_cfg.get("paper_assets_dir", result_root / "paper_assets")))
    release_dir = Path(str(run_cfg.get("release_dir", result_root / "release")))
    run_cfg_path = run_cfg.get("run_config_path")

    if _norm_path(args.main_mean) == _norm_path(Path("results/main_table_val_seedmean.csv")):
        args.main_mean = tables_dir / "main_table_val_seedmean.csv"
    if _norm_path(args.main_std) == _norm_path(Path("results/main_table_val_seedstd.csv")):
        args.main_std = tables_dir / "main_table_val_seedstd.csv"
    if _norm_path(args.count_csv) == _norm_path(Path("results/count_metrics_val.csv")):
        args.count_csv = tables_dir / "count_metrics_val.csv"
    if _norm_path(args.strong_mean) == _norm_path(Path("results/strong_baselines_seedmean.csv")):
        args.strong_mean = tables_dir / "strong_baselines_seedmean.csv"
    if _norm_path(args.strong_std) == _norm_path(Path("results/strong_baselines_seedstd.csv")):
        args.strong_std = tables_dir / "strong_baselines_seedstd.csv"
    if _norm_path(args.with_baselines_csv) == _norm_path(Path("results/main_table_val_with_baselines.csv")):
        args.with_baselines_csv = tables_dir / "main_table_val_with_baselines.csv"
    if _norm_path(args.out_dir) == _norm_path(Path("results/paper_assets_val")):
        args.out_dir = paper_assets_dir
    if _norm_path(args.release_dir) == _norm_path(Path("release")):
        args.release_dir = release_dir
    if run_cfg_path and _norm_path(args.run_config) == _norm_path(Path("results/main_val/run_config.json")):
        args.run_config = Path(str(run_cfg_path))
    if args.degradation_csv is None:
        args.degradation_csv = tables_dir / "degradation_grid.csv"
    if args.degradation_plot is None:
        args.degradation_plot = paper_assets_dir / "degradation_grid.png"
    if args.degradation_tex is None:
        args.degradation_tex = paper_assets_dir / "degradation_grid.tex"


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_float(v: str) -> float:
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return float("nan")


def apply_plot_style() -> None:
    plt.rcParams.update(PLOT_STYLE)


def format_pm_text(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def format_uncertainty_s(mean: float, std: float, decimals: int = 3) -> str:
    scaled_std = int(round(abs(std) * (10**decimals)))
    return f"{mean:.{decimals}f}({scaled_std})"


def select_by_method(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        method = normalize_main_chain_label(row.get("method", ""))
        if method in METHOD_ORDER:
            copied = dict(row)
            copied["method"] = method
            out[method] = copied
    return out


def select_by_method_order(rows: List[Dict[str, str]], method_order: List[str]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    wanted = set(method_order)
    for row in rows:
        method = normalize_main_chain_label(row.get("method", ""))
        if method in wanted:
            copied = dict(row)
            copied["method"] = method
            out[method] = copied
    return out


def method_stats_from_mean_std(
    mean_rows: List[Dict[str, str]],
    std_rows: List[Dict[str, str]],
    method_order: List[str],
) -> List[Dict[str, str]]:
    mean_map = select_by_method_order(mean_rows, method_order)
    std_map = select_by_method_order(std_rows, method_order)
    rows: List[Dict[str, str]] = []
    split = "val_half"
    for method in method_order:
        if method not in mean_map or method not in std_map:
            continue
        split = mean_map[method].get("split", split)
        row: Dict[str, str] = {"split": split, "method": method}
        for metric in MAIN_METRICS:
            mv = safe_float(mean_map[method].get(metric, "nan"))
            sv = safe_float(std_map[method].get(metric, "nan"))
            row[metric] = format_pm_text(mv, sv)
            row[f"{metric}_mean"] = f"{mv:.3f}"
            row[f"{metric}_std"] = f"{sv:.3f}"
        rows.append(row)
    return rows


def should_rebuild(target: Path, inputs: List[Path]) -> bool:
    if not target.exists():
        return True
    t = target.stat().st_mtime
    for src in inputs:
        if src.exists() and src.stat().st_mtime > t:
            return True
    return False


def rebuild_with_baselines_if_stale(args: argparse.Namespace) -> Path | None:
    if not args.main_mean.exists() or not args.main_std.exists():
        return None
    if not args.strong_mean.exists() or not args.strong_std.exists():
        return None
    inputs = [args.main_mean, args.main_std, args.strong_mean, args.strong_std]
    if not should_rebuild(args.with_baselines_csv, inputs):
        return args.with_baselines_csv

    main_rows = read_csv_rows(args.main_mean)
    main_std_rows = read_csv_rows(args.main_std)
    strong_rows = read_csv_rows(args.strong_mean)
    strong_std_rows = read_csv_rows(args.strong_std)
    combined = method_stats_from_mean_std(main_rows, main_std_rows, METHOD_ORDER)
    combined.extend(method_stats_from_mean_std(strong_rows, strong_std_rows, STRONG_METHOD_ORDER))
    if not combined:
        return None

    fields = ["split", "method"] + MAIN_METRICS + [f"{m}_mean" for m in MAIN_METRICS] + [f"{m}_std" for m in MAIN_METRICS]
    write_csv(args.with_baselines_csv, fields, combined)
    print(f"[auto-rebuild] rebuilt merged table: {args.with_baselines_csv}")
    return args.with_baselines_csv


def write_main_table_assets(
    mean_rows: List[Dict[str, str]],
    std_rows: List[Dict[str, str]],
    out_dir: Path,
) -> List[Path]:
    outputs: List[Path] = []
    mean_map = select_by_method(mean_rows)
    std_map = select_by_method(std_rows)

    table_rows: List[Dict[str, str]] = []
    split = "val_half"
    for method in METHOD_ORDER:
        if method not in mean_map or method not in std_map:
            continue
        split = mean_map[method].get("split", split)
        row = {"split": split, "method": method}
        for metric in MAIN_METRICS:
            mv = safe_float(mean_map[method].get(metric, "nan"))
            sv = safe_float(std_map[method].get(metric, "nan"))
            row[metric] = format_pm_text(mv, sv)
            row[f"{metric}_mean"] = f"{mv:.3f}"
            row[f"{metric}_std"] = f"{sv:.3f}"
        table_rows.append(row)

    csv_path = out_dir / "paper_main_table.csv"
    fields = ["split", "method"] + MAIN_METRICS + [f"{m}_mean" for m in MAIN_METRICS] + [f"{m}_std" for m in MAIN_METRICS]
    write_csv(csv_path, fields, table_rows)
    outputs.append(csv_path)

    tex_path = out_dir / "paper_main_table.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    with tex_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("% Auto-generated by scripts/make_paper_assets.py\n")
        f.write("\\begin{tabular}{l S S S S S}\n")
        f.write("\\toprule\n")
        f.write("Method & {HOTA} & {DetA} & {AssA} & {IDF1} & {IDSW} \\\\\n")
        f.write("\\midrule\n")
        for row in table_rows:
            hota = format_uncertainty_s(safe_float(row["HOTA_mean"]), safe_float(row["HOTA_std"]))
            deta = format_uncertainty_s(safe_float(row["DetA_mean"]), safe_float(row["DetA_std"]))
            assa = format_uncertainty_s(safe_float(row["AssA_mean"]), safe_float(row["AssA_std"]))
            idf1 = format_uncertainty_s(safe_float(row["IDF1_mean"]), safe_float(row["IDF1_std"]))
            idsw = format_uncertainty_s(safe_float(row["IDSW_mean"]), safe_float(row["IDSW_std"]))
            f.write(
                f"{row['method']} & {hota} & {deta} & "
                f"{assa} & {idf1} & {idsw} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    outputs.append(tex_path)
    return outputs


def write_count_table_assets(count_rows: List[Dict[str, str]], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    mean_rows = [r for r in count_rows if r.get("row_type") == "mean"]
    mean_map = select_by_method(mean_rows)
    split = mean_rows[0].get("split", "val_half") if mean_rows else "val_half"

    table_rows: List[Dict[str, str]] = []
    for method in METHOD_ORDER:
        if method not in mean_map:
            continue
        row = {"split": split, "method": method}
        for metric in COUNT_METRICS:
            row[metric] = f"{safe_float(mean_map[method].get(metric, 'nan')):.3f}"
        table_rows.append(row)

    csv_path = out_dir / "paper_count_table.csv"
    write_csv(csv_path, ["split", "method"] + COUNT_METRICS, table_rows)
    outputs.append(csv_path)

    tex_path = out_dir / "paper_count_table.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    with tex_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("% Auto-generated by scripts/make_paper_assets.py\n")
        f.write("\\begin{tabular}{l S S S S}\n")
        f.write("\\toprule\n")
        f.write("Method & {CountMAE} & {CountRMSE} & {CountVar} & {CountDrift} \\\\\n")
        f.write("\\midrule\n")
        for row in table_rows:
            f.write(
                f"{row['method']} & {row['CountMAE']} & {row['CountRMSE']} & "
                f"{row['CountVar']} & {row['CountDrift']} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    outputs.append(tex_path)
    return outputs


def plot_main_metrics(
    mean_rows: List[Dict[str, str]],
    std_rows: List[Dict[str, str]],
    out_path: Path,
) -> None:
    apply_plot_style()
    mean_map = select_by_method(mean_rows)
    std_map = select_by_method(std_rows)
    methods = [m for m in METHOD_ORDER if m in mean_map and m in std_map]
    metrics = ["HOTA", "AssA", "IDF1"]

    x = np.arange(len(methods))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    for i, metric in enumerate(metrics):
        vals = [safe_float(mean_map[m][metric]) for m in methods]
        errs = [safe_float(std_map[m][metric]) for m in methods]
        ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            yerr=errs,
            capsize=2.5,
            linewidth=0.5,
            label=metric,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Score")
    ax.set_title("Main metrics on val-half (mean ± std)")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="upper right", frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **PLOT_SAVE_KWARGS)
    plt.close(fig)


def plot_count_metrics(count_rows: List[Dict[str, str]], out_path: Path) -> None:
    apply_plot_style()
    mean_rows = [r for r in count_rows if r.get("row_type") == "mean"]
    mean_map = select_by_method(mean_rows)
    methods = [m for m in METHOD_ORDER if m in mean_map]
    mae = [safe_float(mean_map[m]["CountMAE"]) for m in methods]
    rmse = [safe_float(mean_map[m]["CountRMSE"]) for m in methods]

    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    ax.bar(x - width / 2.0, mae, width=width, label="CountMAE", linewidth=0.5)
    ax.bar(x + width / 2.0, rmse, width=width, label="CountRMSE", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Error (lower is better)")
    ax.set_title("Count stability summary (val-half)")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="upper right", frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **PLOT_SAVE_KWARGS)
    plt.close(fig)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_info(path: Path) -> Dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path).replace("\\", "/"),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def reference_only_file_info(path: Path, note: str) -> Dict[str, object]:
    return {
        "path": str(path).replace("\\", "/"),
        "reference_policy": "path_only_no_nested_hash",
        "note": note,
    }


def manifest_safe_run_config(run_cfg: Dict[str, object]) -> Dict[str, object]:
    safe_cfg = copy.deepcopy(run_cfg)
    manifest_hash = safe_cfg.get("manifest_hash")
    if isinstance(manifest_hash, dict):
        safe_cfg["manifest_hash"] = {
            **manifest_hash,
            "value": "pending_manifest_build",
            "mode": "pending",
            "note": (
                "manifest snapshot keeps the run_config manifest_hash unresolved to avoid "
                "a circular self-reference; see the authoritative run_config.json for the resolved value."
            ),
        }
    return safe_cfg


def write_resolved_manifest_hash(run_config_path: Path, manifest_path: Path) -> Dict[str, object] | None:
    if not run_config_path.exists():
        return None
    run_cfg = json.loads(run_config_path.read_text(encoding="utf-8"))
    manifest_hash = run_cfg.get("manifest_hash")
    if not isinstance(manifest_hash, dict):
        manifest_hash = {}
    run_cfg["manifest_hash"] = {
        **manifest_hash,
        "path": str(manifest_path).replace("\\", "/"),
        "algorithm": "sha256",
        "value": sha256_file(manifest_path).upper(),
        "mode": "current",
        "note": "resolved against the emitted manifest.json file",
    }
    run_config_path.write_text(json.dumps(run_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return run_cfg


def normalize_provenance_ref(value: str) -> str:
    cleaned = str(value).strip()
    if cleaned == "":
        raise ValueError("Empty provenance reference is not allowed.")
    upper = cleaned.upper()
    if any(token in upper for token in PLACEHOLDER_PROVENANCE_TOKENS):
        raise ValueError(f"Placeholder provenance reference is not allowed: {cleaned}")
    if cleaned == "unknown":
        return cleaned
    if GIT_HASH_RE.fullmatch(cleaned) or ANON_BUNDLE_RE.fullmatch(cleaned):
        return cleaned
    raise ValueError(
        "Provenance reference must be a git commit hash or a clearly named anonymous bundle ID. "
        f"Received: {cleaned}"
    )


def get_git_commit() -> str:
    env_commit = os.environ.get("GIT_COMMIT")
    if env_commit not in (None, ""):
        return normalize_provenance_ref(env_commit)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return normalize_provenance_ref(result.stdout.strip())
    except Exception:  # noqa: BLE001
        return "unknown"


def env_or(default: str, *keys: str) -> str:
    for key in keys:
        val = os.environ.get(key)
        if val not in (None, ""):
            return val
    return default


def infer_run_config(args: argparse.Namespace) -> Dict[str, object]:
    max_frames = int(env_or("1000", "MAX_FRAMES"))
    result_root = env_or("results/main_val", "RESULT_ROOT")
    if result_root in ("", "results/main_val") and max_frames < 1000:
        result_root = "results/_smoke/main_val"
    result_root_path = Path(result_root)
    tables_dir = str((result_root_path / "tables").as_posix())
    paper_assets_dir = str((result_root_path / "paper_assets").as_posix())
    release_dir = str((result_root_path / "release").as_posix())
    return {
        "split": env_or("val_half", "SPLIT"),
        "max_frames": max_frames,
        "seeds": env_or("0,1,2", "SEEDS"),
        "drop_rate": float(env_or("0.2", "DROP_RATE")),
        "jitter": float(env_or("0.02", "JITTER")),
        "gating_thresh": float(env_or("9.210340371976184", "GATING_THRESH")),
        "bucket_mode": env_or("quantile", "BUCKET_MODE"),
        "result_root": str(result_root_path.as_posix()),
        "seed_root": str((result_root_path / "seed_runs").as_posix()),
        "tables_dir": tables_dir,
        "paper_assets_dir": paper_assets_dir,
        "release_dir": release_dir,
        "runtime_dir": str((result_root_path / "runtime").as_posix()),
        "stratified_dir": str((result_root_path / "stratified").as_posix()),
        "pred_root": env_or(str((result_root_path / "seed_runs/seed_0").as_posix()), "PRED_ROOT"),
        "methods": env_or("base,gating,traj,adaptive,bytetrack,ocsort,botsort", "METHODS"),
    }


def load_run_config(path: Path) -> Dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to read run config: {path}") from exc


def clean_outputs(args: argparse.Namespace) -> None:
    if not args.clean_output:
        return
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.release_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.release_dir / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()


def write_release_files(
    args: argparse.Namespace,
    generated_files: List[Path],
    run_cfg: Dict[str, object],
    tracked_inputs: Dict[str, Path],
    tracked_outputs: Dict[str, Path],
) -> List[Path]:
    outputs: List[Path] = []
    args.release_dir.mkdir(parents=True, exist_ok=True)

    reproduce_path = args.release_dir / "reproduce.bat"

    def _win(path: Path) -> str:
        return str(path).replace("/", "\\")

    with reproduce_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("@echo off\n")
        f.write("setlocal EnableExtensions\n")
        f.write("set \"SCRIPT_DIR=%~dp0\"\n")
        f.write("set \"REPO_ROOT=\"\n")
        f.write("for %%I in (\"%SCRIPT_DIR%.\" \"%SCRIPT_DIR%..\" \"%SCRIPT_DIR%..\\..\" \"%SCRIPT_DIR%..\\..\\..\" \"%SCRIPT_DIR%..\\..\\..\\..\" \"%SCRIPT_DIR%..\\..\\..\\..\\..\") do (\n")
        f.write("  if not defined REPO_ROOT if exist \"%%~fI\\scripts\\make_paper_assets.py\" set \"REPO_ROOT=%%~fI\"\n")
        f.write(")\n")
        f.write("if not defined REPO_ROOT (\n")
        f.write("  echo [ERR] Could not locate repository root from \"%SCRIPT_DIR%\".\n")
        f.write("  exit /b 1\n")
        f.write(")\n")
        f.write("pushd \"%REPO_ROOT%\"\n")
        f.write("set \"PY_EXE=python\"\n")
        f.write("if exist \"%REPO_ROOT%\\.venv\\Scripts\\python.exe\" set \"PY_EXE=%REPO_ROOT%\\.venv\\Scripts\\python.exe\"\n")
        f.write(
            "\"%PY_EXE%\" scripts\\make_paper_assets.py "
            f"--main-mean {_win(args.main_mean)} "
            f"--main-std {_win(args.main_std)} "
            f"--count-csv {_win(args.count_csv)} "
            f"--strong-mean {_win(args.strong_mean)} "
            f"--strong-std {_win(args.strong_std)} "
            f"--with-baselines-csv {_win(args.with_baselines_csv)} "
            f"--run-config {_win(args.run_config)} "
            f"--out-dir {_win(args.out_dir)} "
            f"--release-dir {_win(args.release_dir)}\n"
        )
        f.write("if errorlevel 1 (\n")
        f.write("  echo make_paper_assets failed.\n")
        f.write("  popd\n")
        f.write("  exit /b 1\n")
        f.write(")\n")
        if args.degradation_csv and args.degradation_csv.exists():
            f.write(":: === Degradation Grid (auto) START ===\n")
            f.write(
                "\"%PY_EXE%\" scripts\\run_degradation_grid.py "
                f"--run-config {_win(args.run_config)}\n"
            )
            f.write("if errorlevel 1 (\n")
            f.write("  echo run_degradation_grid failed.\n")
            f.write("  popd\n")
            f.write("  exit /b 1\n")
            f.write(")\n")
            f.write(":: === Degradation Grid (auto) END ===\n")
        tables_dir = Path(str(run_cfg.get("tables_dir", Path(str(run_cfg.get("result_root", "results/main_val"))) / "tables")))
        if (tables_dir / "gating_thresh_sensitivity.csv").exists():
            f.write(":: === Gating Threshold Sensitivity (auto) START ===\n")
            f.write(
                "\"%PY_EXE%\" scripts\\run_gating_thresh_sensitivity.py "
                f"--run-config {_win(args.run_config)}\n"
            )
            f.write("if errorlevel 1 (\n")
            f.write("  echo run_gating_thresh_sensitivity failed.\n")
            f.write("  popd\n")
            f.write("  exit /b 1\n")
            f.write(")\n")
            f.write(":: === Gating Threshold Sensitivity (auto) END ===\n")
        if (tables_dir / "degradation_extended.csv").exists():
            f.write(":: === Degradation Extended (auto) START ===\n")
            f.write(
                "\"%PY_EXE%\" scripts\\run_degradation_extended.py "
                f"--run-config {_win(args.run_config)}\n"
            )
            f.write("if errorlevel 1 (\n")
            f.write("  echo run_degradation_extended failed.\n")
            f.write("  popd\n")
            f.write("  exit /b 1\n")
            f.write(")\n")
            f.write(":: === Degradation Extended (auto) END ===\n")
        f.write("echo Done.\n")
        f.write("popd\n")
        f.write("exit /b 0\n")
    outputs.append(reproduce_path)

    manifest_path = args.release_dir / "manifest.json"
    manifest = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python scripts/make_paper_assets.py",
        "cwd": str(Path.cwd()).replace("\\", "/"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "git_commit": get_git_commit(),
        "run_config": manifest_safe_run_config(run_cfg),
        "inputs": {},
        "outputs": {},
    }
    for key, path in tracked_inputs.items():
        if not path.exists():
            continue
        if key == "run_config":
            manifest["inputs"][key] = reference_only_file_info(
                path,
                "The authoritative run_config.json stores the resolved manifest SHA256, so the manifest records it by path only to avoid a circular nested hash.",
            )
        else:
            manifest["inputs"][key] = file_info(path)

    tracked_outputs = dict(tracked_outputs)
    tracked_outputs["reproduce_bat"] = reproduce_path
    for key, path in tracked_outputs.items():
        if path.exists():
            manifest["outputs"][key] = file_info(path)

    with manifest_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    outputs.append(manifest_path)

    write_resolved_manifest_hash(args.run_config, manifest_path)
    return outputs


def write_readme(out_dir: Path, generated: List[Path], referenced: List[Path] | None = None) -> Path:
    readme = out_dir / "README_paper_assets.txt"
    with readme.open("w", encoding="utf-8", newline="\n") as f:
        f.write("Auto-generated paper assets\n")
        f.write("===========================\n")
        for path in generated:
            f.write(f"- {path.as_posix()}\n")
        if referenced:
            f.write("\nReferenced experiment assets\n")
            f.write("===========================\n")
            for path in referenced:
                f.write(f"- {path.as_posix()}\n")
    return readme


def main() -> None:
    args = parse_args()
    clean_outputs(args)

    run_cfg = load_run_config(args.run_config)
    if run_cfg is None:
        run_cfg = infer_run_config(args)
    else:
        # Keep current build command explicit in manifest even when run_config exists.
        run_cfg = dict(run_cfg)
    apply_run_config_paths(args, run_cfg)
    if args.run_config is not None:
        assert_no_legacy_write_paths(args, run_cfg)

    stale_inputs = [args.main_mean, args.main_std, args.strong_mean, args.strong_std]
    if args.with_baselines_csv.exists() and should_rebuild(args.with_baselines_csv, stale_inputs):
        if args.no_auto_rebuild_baselines:
            raise RuntimeError(
                "main_table_val_with_baselines.csv is older than its inputs. "
                "Rebuild it first or disable --no-auto-rebuild-baselines."
            )
    rebuilt_path = None
    if not args.no_auto_rebuild_baselines:
        rebuilt_path = rebuild_with_baselines_if_stale(args)

    mean_rows = read_csv_rows(args.main_mean)
    std_rows = read_csv_rows(args.main_std)
    count_rows = read_csv_rows(args.count_csv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    outputs.extend(write_main_table_assets(mean_rows, std_rows, args.out_dir))
    outputs.extend(write_count_table_assets(count_rows, args.out_dir))

    main_plot = args.out_dir / "main_table_val_seedmean_std_paper.png"
    count_plot = args.out_dir / "count_stability_bar_paper.png"
    plot_main_metrics(mean_rows, std_rows, main_plot)
    plot_count_metrics(count_rows, count_plot)
    outputs.extend([main_plot, count_plot])

    tracked_inputs: Dict[str, Path] = {
        "main_table_seedmean": args.main_mean,
        "main_table_seedstd": args.main_std,
        "count_metrics": args.count_csv,
        "strong_baselines_seedmean": args.strong_mean,
        "strong_baselines_seedstd": args.strong_std,
        "run_config": args.run_config,
    }
    if args.degradation_csv and args.degradation_csv.exists():
        tracked_inputs["degradation_grid_csv"] = args.degradation_csv
    if args.degradation_plot and args.degradation_plot.exists():
        tracked_inputs["degradation_grid_png"] = args.degradation_plot
    if args.degradation_tex and args.degradation_tex.exists():
        tracked_inputs["degradation_grid_tex"] = args.degradation_tex
    if args.with_baselines_csv.exists():
        tracked_inputs["main_table_with_baselines"] = args.with_baselines_csv
    tracked_outputs: Dict[str, Path] = {
        "paper_main_table_csv": args.out_dir / "paper_main_table.csv",
        "paper_main_table_tex": args.out_dir / "paper_main_table.tex",
        "paper_count_table_csv": args.out_dir / "paper_count_table.csv",
        "paper_count_table_tex": args.out_dir / "paper_count_table.tex",
        "paper_main_table_png": main_plot,
        "paper_count_png": count_plot,
        "merged_with_baselines_csv": args.with_baselines_csv,
    }
    if args.degradation_plot and args.degradation_plot.exists():
        tracked_outputs["degradation_grid_png"] = args.degradation_plot
    if args.degradation_tex and args.degradation_tex.exists():
        tracked_outputs["degradation_grid_tex"] = args.degradation_tex
    tables_dir = Path(str(run_cfg.get("tables_dir", Path(str(run_cfg.get("result_root", "results/main_val"))) / "tables")))
    paper_assets_dir = Path(str(run_cfg.get("paper_assets_dir", Path(str(run_cfg.get("result_root", "results/main_val"))) / "paper_assets")))
    optional_release_outputs = {
        "gating_sensitivity_csv": tables_dir / "gating_thresh_sensitivity.csv",
        "gating_sensitivity_tex": paper_assets_dir / "gating_thresh_sensitivity.tex",
        "gating_sensitivity_png": paper_assets_dir / "gating_thresh_sensitivity.png",
        "degradation_extended_csv": tables_dir / "degradation_extended.csv",
        "degradation_extended_tex": paper_assets_dir / "degradation_extended_delta.tex",
        "degradation_extended_png": paper_assets_dir / "degradation_extended.png",
        "degradation_extended_examples_png": paper_assets_dir / "degradation_extended_examples.png",
        "degradation_extended_manuscript_txt": tables_dir / "degradation_extended_manuscript.txt",
    }
    for key, path in optional_release_outputs.items():
        if path.exists():
            tracked_outputs[key] = path
    outputs.extend(write_release_files(args, outputs, run_cfg=run_cfg, tracked_inputs=tracked_inputs, tracked_outputs=tracked_outputs))
    referenced_assets: List[Path] = []
    for maybe_path in (args.degradation_csv, args.degradation_plot, args.degradation_tex):
        if maybe_path is not None and maybe_path.exists():
            referenced_assets.append(maybe_path)
    readme = write_readme(args.out_dir, outputs, referenced=referenced_assets)
    outputs.append(readme)
    if rebuilt_path is not None:
        print(f"Auto rebuilt merged table: {rebuilt_path}")

    print(f"Generated {len(outputs)} files:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
