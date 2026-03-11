#!/usr/bin/env python
"""Run gating threshold sensitivity on val_half (Base + +gating)."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


DEFAULT_THRESHOLDS = [1000.0, 2000.0, 4000.0]
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
    parser = argparse.ArgumentParser(
        description="Run gating threshold sensitivity (Base + +gating only)."
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=Path("results/main_val/run_config.json"),
        help="Path to run_config.json.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="1000,2000,4000",
        help="Comma-separated gating thresholds for +gating.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Detection degradation seed (default: 0).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean working directory before running.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="If set, render plot/tex directly from an existing sensitivity CSV.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Skip tracker/eval runs and only render outputs from --input-csv.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path (default: <tables_dir>/gating_sensitivity.csv).",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=None,
        help="Optional output TEX path (default: <paper_assets_dir>/gating_sensitivity.tex).",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Optional output PNG path (default: <paper_assets_dir>/gating_sensitivity.png).",
    )
    return parser.parse_args()


def _norm(path: Path) -> str:
    return str(path).replace("\\", "/").lower()


def _is_legacy_output_path(path: Path, result_root: Path) -> bool:
    norm = _norm(path)
    root_norm = _norm(result_root)
    return norm.startswith("results/") and norm != root_norm and not norm.startswith(root_norm + "/")


def run_cmd(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def read_single_row(path: Path) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row in {path}, got {len(rows)}")
    row = rows[0]
    return {
        "HOTA": float(row["HOTA"]),
        "DetA": float(row["DetA"]),
        "AssA": float(row["AssA"]),
        "IDF1": float(row["IDF1"]),
        "IDSW": float(row["IDSW"]),
    }


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def apply_plot_style() -> None:
    plt.rcParams.update(PLOT_STYLE)


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "gating_thresh", "seed", "HOTA", "DetA", "AssA", "IDF1", "IDSW"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_tex(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("% Auto-generated gating sensitivity table\n")
        f.write("\\begin{tabular}{l l S S S S S}\n")
        f.write("\\toprule\n")
        f.write("Method & Gating threshold & {HOTA} & {DetA} & {AssA} & {IDF1} & {IDSW} \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(
                f"{row['method']} & {row['gating_thresh']} & {row['HOTA']} & {row['DetA']} & "
                f"{row['AssA']} & {row['IDF1']} & {row['IDSW']} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def make_plot(rows: List[Dict[str, str]], plot_path: Path) -> None:
    apply_plot_style()
    base = next(r for r in rows if r["method"] == "Base")
    gating_rows = sorted(
        [r for r in rows if r["method"] == "+gating"], key=lambda x: float(x["gating_thresh"])
    )
    xs = [float(r["gating_thresh"]) for r in gating_rows]
    hota = [float(r["HOTA"]) for r in gating_rows]
    idf1 = [float(r["IDF1"]) for r in gating_rows]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.6), constrained_layout=True, sharex=True)

    axes[0].plot(xs, hota, marker="o", color="#F58518", linewidth=1.2, label="+gating")
    axes[0].axhline(float(base["HOTA"]), color="#4C78A8", linestyle="--", linewidth=1.0, label="Base")
    axes[0].set_title("HOTA vs gating threshold")
    axes[0].set_xlabel("Gating threshold")
    axes[0].set_ylabel("HOTA")
    axes[0].grid(alpha=0.18)

    axes[1].plot(xs, idf1, marker="o", color="#F58518", linewidth=1.2, label="+gating")
    axes[1].axhline(float(base["IDF1"]), color="#4C78A8", linestyle="--", linewidth=1.0, label="Base")
    axes[1].set_title("IDF1 vs gating threshold")
    axes[1].set_xlabel("Gating threshold")
    axes[1].set_ylabel("IDF1")
    axes[1].grid(alpha=0.18)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Gating threshold sensitivity (val-half, seed=0)")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, **PLOT_SAVE_KWARGS)
    plt.close(fig)


def upsert_reproduce_section(reproduce_bat: Path, run_config_path: Path) -> None:
    start = ":: === Gating Threshold Sensitivity (auto) START ==="
    end = ":: === Gating Threshold Sensitivity (auto) END ==="
    run_cfg_win = str(run_config_path).replace("/", "\\")
    section = [
        start,
        f"\"%PY_EXE%\" scripts\\run_gating_thresh_sensitivity.py --run-config {run_cfg_win}",
        "if errorlevel 1 (",
        "  echo run_gating_thresh_sensitivity failed.",
        "  popd",
        "  exit /b 1",
        ")",
        end,
    ]

    if not reproduce_bat.exists():
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
    tables_dir = Path(str(cfg.get("tables_dir", result_root / "tables")))
    paper_assets_dir = Path(str(cfg.get("paper_assets_dir", result_root / "paper_assets")))
    release_dir = Path(str(cfg.get("release_dir", result_root / "release")))

    output_csv = args.output_csv if args.output_csv is not None else (tables_dir / "gating_sensitivity.csv")
    output_tex = args.output_tex if args.output_tex is not None else (paper_assets_dir / "gating_sensitivity.tex")
    output_png = args.output_png if args.output_png is not None else (paper_assets_dir / "gating_sensitivity.png")
    input_csv = args.input_csv if args.input_csv is not None else output_csv

    for name, p in [("csv", output_csv), ("tex", output_tex), ("png", output_png)]:
        if _is_legacy_output_path(p, result_root):
            raise RuntimeError(
                f"run-config strict mode blocks legacy output ({name}={p}); result_root={result_root}"
            )

    if args.render_only:
        rows = read_rows(input_csv)
        rows.sort(
            key=lambda r: (
                0 if r["method"] == "Base" else 1,
                float(r["gating_thresh"]) if r["gating_thresh"] not in {"N/A", ""} else -1.0,
            )
        )
        write_csv(output_csv, rows)
        write_tex(output_tex, rows)
        make_plot(rows, output_png)
        upsert_reproduce_section(release_dir / "reproduce.bat", args.run_config)
        print(f"Rendered CSV: {output_csv}")
        print(f"Rendered TEX: {output_tex}")
        print(f"Rendered PNG: {output_png}")
        return

    split = str(cfg.get("split", "val_half"))
    max_frames = int(cfg.get("max_frames", 1000))
    drop_rate = float(cfg.get("drop_rate", 0.2))
    jitter = float(cfg.get("jitter", 0.02))

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    if not thresholds:
        thresholds = list(DEFAULT_THRESHOLDS)

    py = sys.executable
    work_root = result_root / "gating_sensitivity"
    if args.clean and work_root.exists():
        import shutil

        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []

    # Base (run once)
    base_dir = work_root / "base"
    base_pred = base_dir / "pred"
    base_mean = base_dir / "mean.csv"
    base_per_seq = base_dir / "per_seq.csv"

    run_cmd(
        [
            py,
            "scripts/run_baseline_sort.py",
            "--run-config",
            str(args.run_config),
            "--split",
            split,
            "--max-frames",
            str(max_frames),
            "--gating",
            "off",
            "--traj",
            "off",
            "--alpha",
            "1.0",
            "--beta",
            "0.0",
            "--gamma",
            "0.0",
            "--adaptive-gamma",
            "off",
            "--drop-rate",
            str(drop_rate),
            "--jitter",
            str(jitter),
            "--degrade-seed",
            str(args.seed),
            "--out-dir",
            str(base_pred),
            "--clean-out",
        ]
    )
    run_cmd(
        [
            py,
            "scripts/eval_trackeval_per_seq.py",
            "--run-config",
            str(args.run_config),
            "--split",
            split,
            "--pred-dir",
            str(base_pred),
            "--tracker-name",
            "base_gating_sensitivity",
            "--max-frames",
            str(max_frames),
            "--max-gt-ids",
            "50000",
            "--results-mean",
            str(base_mean),
            "--results-per-seq",
            str(base_per_seq),
        ]
    )
    base_metrics = read_single_row(base_mean)
    rows.append(
        {
            "method": "Base",
            "gating_thresh": "N/A",
            "seed": str(args.seed),
            "HOTA": f"{base_metrics['HOTA']:.3f}",
            "DetA": f"{base_metrics['DetA']:.3f}",
            "AssA": f"{base_metrics['AssA']:.3f}",
            "IDF1": f"{base_metrics['IDF1']:.3f}",
            "IDSW": f"{base_metrics['IDSW']:.3f}",
        }
    )

    # +gating sensitivity
    for thresh in thresholds:
        tag = f"t{int(thresh)}" if thresh.is_integer() else f"t{thresh}".replace(".", "p")
        run_dir = work_root / tag
        pred_dir = run_dir / "pred"
        mean_csv = run_dir / "mean.csv"
        per_seq_csv = run_dir / "per_seq.csv"

        run_cmd(
            [
                py,
                "scripts/run_baseline_sort.py",
                "--run-config",
                str(args.run_config),
                "--split",
                split,
                "--max-frames",
                str(max_frames),
                "--gating",
                "on",
                "--traj",
                "off",
                "--alpha",
                "1.0",
                "--beta",
                "0.02",
                "--gamma",
                "0.0",
                "--adaptive-gamma",
                "off",
                "--gating-thresh",
                str(thresh),
                "--drop-rate",
                str(drop_rate),
                "--jitter",
                str(jitter),
                "--degrade-seed",
                str(args.seed),
                "--out-dir",
                str(pred_dir),
                "--clean-out",
            ]
        )
        run_cmd(
            [
                py,
                "scripts/eval_trackeval_per_seq.py",
                "--run-config",
                str(args.run_config),
                "--split",
                split,
                "--pred-dir",
                str(pred_dir),
                "--tracker-name",
                f"gating_{tag}",
                "--max-frames",
                str(max_frames),
                "--max-gt-ids",
                "50000",
                "--results-mean",
                str(mean_csv),
                "--results-per-seq",
                str(per_seq_csv),
            ]
        )
        metrics = read_single_row(mean_csv)
        rows.append(
            {
                "method": "+gating",
                "gating_thresh": f"{thresh:.0f}" if thresh.is_integer() else f"{thresh}",
                "seed": str(args.seed),
                "HOTA": f"{metrics['HOTA']:.3f}",
                "DetA": f"{metrics['DetA']:.3f}",
                "AssA": f"{metrics['AssA']:.3f}",
                "IDF1": f"{metrics['IDF1']:.3f}",
                "IDSW": f"{metrics['IDSW']:.3f}",
            }
        )

    write_csv(output_csv, rows)
    write_tex(output_tex, rows)
    make_plot(rows, output_png)
    upsert_reproduce_section(release_dir / "reproduce.bat", args.run_config)

    print(f"Saved CSV: {output_csv}")
    print(f"Saved TEX: {output_tex}")
    print(f"Saved PNG: {output_png}")


if __name__ == "__main__":
    main()
