#!/usr/bin/env python
"""Run a fixed degradation grid for Base/+gating/ByteTrack on val_half."""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


FIXED_SPLIT = "val_half"
FIXED_MAX_FRAMES = 1000
FIXED_SEED = 0
FIXED_GATING_THRESH = 2000.0
FIXED_DET_SOURCE = "auto"
FIXED_DROP_GRID = [0.0, 0.2, 0.4]
FIXED_JITTER_GRID = [0.0, 0.02]
FIXED_MAX_GT_IDS = 50000

METHODS: List[Tuple[str, Dict[str, str]]] = [
    (
        "Base",
        {
            "method_key": "base",
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
    (
        "+gating",
        {
            "method_key": "gating",
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
    (
        "ByteTrack",
        {
            "method_key": "bytetrack",
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
    parser = argparse.ArgumentParser(description="Run degradation grid (Base/+gating/ByteTrack, seed=0).")
    parser.add_argument(
        "--run-config",
        type=Path,
        default=Path("results/main_val/run_config.json"),
        help="Path to run_config.json (result_root/mot_root source of truth).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Default: <tables_dir>/degradation_grid.csv",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Output plot path. Default: <paper_assets_dir>/degradation_grid.png",
    )
    parser.add_argument(
        "--tex-path",
        type=Path,
        default=None,
        help="Output tex path. Default: <paper_assets_dir>/degradation_grid.tex",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean degradation working directory before running.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="If set, render plot/tex directly from an existing degradation CSV.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Skip tracker/eval runs and only render outputs from --input-csv.",
    )
    return parser.parse_args()


def _norm(path: Path) -> str:
    return str(path).replace("\\", "/").lower()


def _is_legacy_output_path(path: Path, result_root: Path) -> bool:
    norm = _norm(path)
    root_norm = _norm(result_root)
    return norm.startswith("results/") and norm != root_norm and not norm.startswith(root_norm + "/")


def run_cmd(cmd: Sequence[str]) -> None:
    print("[run]", " ".join(str(x) for x in cmd))
    subprocess.run(list(cmd), check=True)


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


def tag_from_values(drop_rate: float, jitter: float) -> str:
    d = f"{drop_rate:.2f}".replace(".", "p")
    j = f"{jitter:.2f}".replace(".", "p")
    return f"d{d}_j{j}"


def parse_single_mean(mean_csv: Path) -> Dict[str, float]:
    if not mean_csv.exists():
        raise FileNotFoundError(f"Mean CSV not found: {mean_csv}")
    with mean_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row in mean CSV: {mean_csv}, got {len(rows)}")
    row = rows[0]
    return {
        "HOTA": float(row["HOTA"]),
        "IDF1": float(row["IDF1"]),
        "IDSW": float(row["IDSW"]),
        "DetA": float(row["DetA"]),
        "AssA": float(row["AssA"]),
    }


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def apply_plot_style() -> None:
    plt.rcParams.update(PLOT_STYLE)


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    fields = ["method", "drop_rate", "jitter", "seed", "HOTA", "IDF1", "IDSW", "DetA", "AssA", "fps"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_plot(rows: List[Dict[str, str]], out_path: Path) -> None:
    apply_plot_style()
    methods = [m for m, _ in METHODS]
    drops = FIXED_DROP_GRID
    jitters = FIXED_JITTER_GRID
    colors = {"Base": "#4C78A8", "+gating": "#F58518", "ByteTrack": "#54A24B"}
    linestyles = {0.0: "-", 0.02: "--"}

    key_to_hota: Dict[Tuple[str, float, float], float] = {}
    for row in rows:
        key_to_hota[(row["method"], float(row["drop_rate"]), float(row["jitter"]))] = float(row["HOTA"])

    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    for method in methods:
        for jitter in jitters:
            y = [key_to_hota.get((method, d, jitter), np.nan) for d in drops]
            ax.plot(
                drops,
                y,
                marker="o",
                linestyle=linestyles[jitter],
                color=colors[method],
                linewidth=1.2,
                label=f"{method} (j={jitter:.2f})",
            )

    ax.set_xlabel("drop rate")
    ax.set_ylabel("HOTA")
    ax.set_title("Controlled degradation on val-half (seed=0, max frames=1000)")
    ax.grid(alpha=0.18)
    ax.set_xticks(drops)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **PLOT_SAVE_KWARGS)
    plt.close(fig)


def write_tex(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("% Auto-generated by scripts/run_degradation_grid.py\n")
        f.write("\\begin{tabular}{l S S S S S S S}\n")
        f.write("\\toprule\n")
        f.write("Method & {Drop rate} & {Jitter} & {HOTA} & {IDF1} & {IDSW} & {DetA} & {AssA} \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(
                f"{row['method']} & {row['drop_rate']} & {row['jitter']} & {row['HOTA']} & "
                f"{row['IDF1']} & {row['IDSW']} & {row['DetA']} & {row['AssA']} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def upsert_reproduce_section(reproduce_bat: Path, run_config_path: Path) -> None:
    start = ":: === Degradation Grid (auto) START ==="
    end = ":: === Degradation Grid (auto) END ==="
    run_cfg_win = str(run_config_path).replace("/", "\\")
    section = [
        start,
        f"python scripts\\run_degradation_grid.py --run-config {run_cfg_win}",
        "if errorlevel 1 (",
        "  echo run_degradation_grid failed.",
        "  popd",
        "  exit /b 1",
        ")",
        end,
    ]

    if not reproduce_bat.exists():
        reproduce_bat.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "@echo off",
            "setlocal",
            'set "REPO_ROOT=%~dp0.."',
            'pushd "%REPO_ROOT%"',
            *section,
            "echo Done.",
            "popd",
            "exit /b 0",
            "",
        ]
        reproduce_bat.write_text("\n".join(lines), encoding="utf-8")
        return

    lines = reproduce_bat.read_text(encoding="utf-8").splitlines()
    try:
        s_idx = lines.index(start)
        e_idx = lines.index(end)
        lines = lines[:s_idx] + section + lines[e_idx + 1 :]
    except ValueError:
        insert_idx = None
        for i, line in enumerate(lines):
            if line.strip().lower().startswith("echo done"):
                insert_idx = i
                break
        if insert_idx is None:
            insert_idx = len(lines)
        lines = lines[:insert_idx] + section + lines[insert_idx:]
    reproduce_bat.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.run_config.exists():
        raise FileNotFoundError(f"Run config not found: {args.run_config}")
    cfg = json.loads(args.run_config.read_text(encoding="utf-8"))

    result_root = Path(str(cfg.get("result_root", "results/main_val")))
    mot_root = Path(str(cfg.get("mot_root", "data/mft25_mot_full")))
    tables_dir = Path(str(cfg.get("tables_dir", result_root / "tables")))
    paper_assets_dir = Path(str(cfg.get("paper_assets_dir", result_root / "paper_assets")))
    release_dir = Path(str(cfg.get("release_dir", result_root / "release")))

    output_csv = args.output_csv if args.output_csv is not None else (tables_dir / "degradation_grid.csv")
    plot_path = args.plot_path if args.plot_path is not None else (paper_assets_dir / "degradation_grid.png")
    tex_path = args.tex_path if args.tex_path is not None else (paper_assets_dir / "degradation_grid.tex")
    input_csv = args.input_csv if args.input_csv is not None else output_csv

    write_targets = [("output_csv", output_csv), ("plot_path", plot_path), ("tex_path", tex_path)]
    bad = [f"{name}={path}" for name, path in write_targets if _is_legacy_output_path(path, result_root)]
    if bad:
        raise RuntimeError(
            "run-config strict mode blocks legacy/global outputs. "
            f"result_root={result_root}. bad targets: {'; '.join(bad)}"
        )

    if args.render_only:
        rows = read_rows(input_csv)
        rows.sort(key=lambda r: (float(r["drop_rate"]), float(r["jitter"]), r["method"]))
        write_csv(output_csv, rows)
        make_plot(rows, plot_path)
        write_tex(tex_path, rows)
        upsert_reproduce_section(release_dir / "reproduce.bat", args.run_config)
        print(f"Rendered degradation CSV: {output_csv}")
        print(f"Rendered degradation plot: {plot_path}")
        print(f"Rendered degradation tex: {tex_path}")
        return

    split_dir = mot_root / FIXED_SPLIT
    if not split_dir.exists():
        raise FileNotFoundError(f"Prepared split not found: {split_dir}")

    grid_root = result_root / "degradation_grid"
    if args.clean and grid_root.exists():
        import shutil

        shutil.rmtree(grid_root)
    grid_root.mkdir(parents=True, exist_ok=True)

    total_frames = estimate_total_frames(split_dir, FIXED_SPLIT, FIXED_MAX_FRAMES)
    rows: List[Dict[str, str]] = []

    for drop_rate in FIXED_DROP_GRID:
        for jitter in FIXED_JITTER_GRID:
            tag = tag_from_values(drop_rate, jitter)
            for method_name, method_cfg in METHODS:
                method_key = method_cfg["method_key"]
                run_dir = grid_root / method_key / tag
                pred_dir = run_dir / "pred"
                per_seq_csv = run_dir / "per_seq.csv"
                mean_csv = run_dir / "mean.csv"

                track_cmd = [
                    sys.executable,
                    "scripts/run_baseline_sort.py",
                    "--split",
                    FIXED_SPLIT,
                    "--mot-root",
                    str(mot_root),
                    "--max-frames",
                    str(FIXED_MAX_FRAMES),
                    "--det-source",
                    FIXED_DET_SOURCE,
                    "--gating",
                    method_cfg["gating"],
                    "--traj",
                    method_cfg["traj"],
                    "--adaptive-gamma",
                    method_cfg["adaptive_gamma"],
                    "--alpha",
                    method_cfg["alpha"],
                    "--beta",
                    method_cfg["beta"],
                    "--gamma",
                    method_cfg["gamma"],
                    "--iou-thresh",
                    method_cfg["iou_thresh"],
                    "--min-hits",
                    method_cfg["min_hits"],
                    "--max-age",
                    method_cfg["max_age"],
                    "--drop-rate",
                    f"{drop_rate}",
                    "--jitter",
                    f"{jitter}",
                    "--degrade-seed",
                    str(FIXED_SEED),
                    "--out-dir",
                    str(pred_dir),
                    "--clean-out",
                ]
                if method_name == "+gating":
                    track_cmd.extend(["--gating-thresh", f"{FIXED_GATING_THRESH}"])

                start = time.perf_counter()
                run_cmd(track_cmd)
                elapsed = max(1e-6, time.perf_counter() - start)
                fps = float(total_frames / elapsed) if total_frames > 0 else 0.0

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
                    f"degrid_{method_key}_{tag}",
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
                metrics = parse_single_mean(mean_csv)

                row = {
                    "method": method_name,
                    "drop_rate": f"{drop_rate:.2f}",
                    "jitter": f"{jitter:.2f}",
                    "seed": str(FIXED_SEED),
                    "HOTA": f"{metrics['HOTA']:.3f}",
                    "IDF1": f"{metrics['IDF1']:.3f}",
                    "IDSW": f"{metrics['IDSW']:.3f}",
                    "DetA": f"{metrics['DetA']:.3f}",
                    "AssA": f"{metrics['AssA']:.3f}",
                    "fps": f"{fps:.3f}",
                }
                rows.append(row)
                print(
                    f"[grid] {method_name} drop={drop_rate:.2f} jitter={jitter:.2f} "
                    f"HOTA={row['HOTA']} IDF1={row['IDF1']} IDSW={row['IDSW']} fps={row['fps']}"
                )

    rows.sort(key=lambda r: (float(r["drop_rate"]), float(r["jitter"]), r["method"]))
    write_csv(output_csv, rows)
    make_plot(rows, plot_path)
    write_tex(tex_path, rows)
    upsert_reproduce_section(release_dir / "reproduce.bat", args.run_config)

    print(f"Saved degradation CSV: {output_csv}")
    print(f"Saved degradation plot: {plot_path}")
    print(f"Saved degradation tex: {tex_path}")


if __name__ == "__main__":
    main()
