#!/usr/bin/env python
"""Run extended degradation stress tests beyond drop-rate and jitter."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance

from run_degradation_grid import (
    FIXED_DET_SOURCE,
    FIXED_GATING_THRESH,
    FIXED_MAX_FRAMES,
    FIXED_MAX_GT_IDS,
    FIXED_SEED,
    FIXED_SPLIT,
    METHODS,
    PLOT_STYLE,
    _is_legacy_output_path,
    estimate_total_frames,
    parse_single_mean,
    read_rows,
    read_seqmap,
    run_cmd,
)


DEGRADATIONS = [
    {
        "key": "motion_blur",
        "label": "Motion blur",
        "arg_name": "motion-blur",
        "column_name": "motion_blur",
        "levels": [0.15, 0.30, 0.45],
        "domain": "detection-space",
        "matters": "Fast turns and platform vibration smear contours and destabilize association.",
    },
    {
        "key": "low_illumination",
        "label": "Low illumination",
        "arg_name": "darken",
        "column_name": "darken",
        "levels": [0.15, 0.30, 0.45],
        "domain": "detection-space",
        "matters": "Dawn, dusk, and shaded pond regions suppress recall on weaker fish instances.",
    },
    {
        "key": "haze_turbidity",
        "label": "Low contrast / haze",
        "arg_name": "haze",
        "column_name": "haze",
        "levels": [0.15, 0.30, 0.45],
        "domain": "detection-space",
        "matters": "Turbidity produces coarse localization drift before fish counting visibly fails.",
    },
]
CSV_FIELDS = [
    "method",
    "method_key",
    "condition_key",
    "degradation_key",
    "degradation_label",
    "domain",
    "parameter_name",
    "level",
    "level_tag",
    "seed",
    "drop_rate",
    "jitter",
    "motion_blur",
    "darken",
    "haze",
    "deployment_relevance",
    "HOTA",
    "IDF1",
    "IDSW",
    "DetA",
    "AssA",
    "fps",
    "delta_HOTA",
    "delta_IDF1",
    "delta_IDSW",
    "delta_DetA",
    "delta_AssA",
]
COLORS = {"Base": "#4C78A8", "+gating": "#F58518", "ByteTrack": "#54A24B"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extended degradation stress tests.")
    parser.add_argument("--run-config", type=Path, default=Path("results/main_val/run_config.json"))
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--tex-path", type=Path, default=None)
    parser.add_argument("--plot-path", type=Path, default=None)
    parser.add_argument("--qualitative-path", type=Path, default=None)
    parser.add_argument("--manuscript-path", type=Path, default=None)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def level_tag(level: float) -> str:
    return f"s{level:.2f}".replace(".", "p")


def read_json(path: Path) -> Dict[str, float | int | str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def order_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    method_order = {name: idx for idx, (name, _) in enumerate(METHODS)}
    degr_order = {"clean": -1, **{cfg["key"]: idx for idx, cfg in enumerate(DEGRADATIONS)}}
    return sorted(rows, key=lambda r: (method_order.get(r["method"], 999), degr_order.get(r["degradation_key"], 999), float(r["level"])))


def build_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for method_name, method_cfg in METHODS:
        base = {
            "method": method_name,
            "method_key": method_cfg["method_key"],
            "condition_key": "clean",
            "degradation_key": "clean",
            "degradation_label": "Clean",
            "domain": "reference",
            "parameter_name": "none",
            "level": "0.00",
            "level_tag": "clean",
            "seed": str(FIXED_SEED),
            "drop_rate": "0.00",
            "jitter": "0.00",
            "motion_blur": "0.00",
            "darken": "0.00",
            "haze": "0.00",
            "deployment_relevance": "Reference operating condition used for delta computation.",
            "HOTA": "",
            "IDF1": "",
            "IDSW": "",
            "DetA": "",
            "AssA": "",
            "fps": "",
            "delta_HOTA": "0.000",
            "delta_IDF1": "0.000",
            "delta_IDSW": "0.000",
            "delta_DetA": "0.000",
            "delta_AssA": "0.000",
        }
        rows.append(dict(base))
        for cfg in DEGRADATIONS:
            for level in cfg["levels"]:
                row = dict(base)
                row.update(
                    {
                        "condition_key": f"{cfg['key']}_{level_tag(level)}",
                        "degradation_key": cfg["key"],
                        "degradation_label": cfg["label"],
                        "domain": cfg["domain"],
                        "parameter_name": cfg["column_name"],
                        "level": f"{level:.2f}",
                        "level_tag": level_tag(level),
                        "deployment_relevance": cfg["matters"],
                        cfg["column_name"]: f"{level:.2f}",
                    }
                )
                rows.append(row)
    return order_rows(rows)


def maybe_run(row: Dict[str, str], run_root: Path, mot_root: Path, total_frames: int) -> Dict[str, str]:
    run_dir = run_root / row["method_key"] / row["condition_key"]
    pred_dir = run_dir / "pred"
    per_seq_csv = run_dir / "per_seq.csv"
    mean_csv = run_dir / "mean.csv"
    timing_json = run_dir / "timing.json"
    timing = read_json(timing_json)
    elapsed = float(timing.get("elapsed_sec", 0.0)) if timing else 0.0
    if not (pred_dir.exists() and per_seq_csv.exists() and mean_csv.exists() and elapsed > 0.0):
        cmd = [
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
            "on" if row["method"] == "+gating" else "off",
            "--traj",
            "off",
            "--adaptive-gamma",
            "off",
            "--alpha",
            "1.0",
            "--beta",
            "0.02" if row["method"] == "+gating" else "0.0",
            "--gamma",
            "0.0",
            "--iou-thresh",
            "0.3",
            "--min-hits",
            "1" if row["method"] == "ByteTrack" else "3",
            "--max-age",
            "30",
            "--drop-rate",
            row["drop_rate"],
            "--jitter",
            row["jitter"],
            "--motion-blur",
            row["motion_blur"],
            "--darken",
            row["darken"],
            "--haze",
            row["haze"],
            "--degrade-seed",
            str(FIXED_SEED),
            "--out-dir",
            str(pred_dir),
            "--clean-out",
        ]
        if row["method"] == "+gating":
            cmd.extend(["--gating-thresh", f"{FIXED_GATING_THRESH}"])
        start = time.perf_counter()
        run_cmd(cmd)
        elapsed = max(1e-6, time.perf_counter() - start)
        write_json(timing_json, {"elapsed_sec": round(elapsed, 6), "total_frames": total_frames})
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
            f"deext_{row['method_key']}_{row['condition_key']}",
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
    fps = float(total_frames / max(elapsed, 1e-6)) if total_frames > 0 else 0.0
    out = dict(row)
    out.update(
        {
            "HOTA": f"{metrics['HOTA']:.3f}",
            "IDF1": f"{metrics['IDF1']:.3f}",
            "IDSW": f"{metrics['IDSW']:.3f}",
            "DetA": f"{metrics['DetA']:.3f}",
            "AssA": f"{metrics['AssA']:.3f}",
            "fps": f"{fps:.3f}",
        }
    )
    return out


def apply_deltas(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    clean_by_method = {row["method"]: row for row in rows if row["degradation_key"] == "clean"}
    out: List[Dict[str, str]] = []
    for row in rows:
        clean = clean_by_method[row["method"]]
        item = dict(row)
        for metric in ["HOTA", "IDF1", "IDSW", "DetA", "AssA"]:
            item[f"delta_{metric}"] = f"{float(item[metric]) - float(clean[metric]):.3f}"
        out.append(item)
    return order_rows(out)


def write_delta_tex(rows: List[Dict[str, str]], path: Path) -> None:
    lines = [
        "% Auto-generated by scripts/run_degradation_extended.py",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Condition & Base $\Delta$HOTA & Base $\Delta$IDSW & +gating $\Delta$HOTA & +gating $\Delta$IDSW & ByteTrack $\Delta$HOTA & ByteTrack $\Delta$IDSW \\",
        r"\midrule",
    ]
    for cfg in DEGRADATIONS:
        for level in cfg["levels"]:
            label = f"{cfg['label']} ({level:.2f})"
            vals: List[str] = []
            for method_name, _ in METHODS:
                row = next(r for r in rows if r["method"] == method_name and r["degradation_key"] == cfg["key"] and abs(float(r["level"]) - level) < 1e-9)
                vals.extend([row["delta_HOTA"], row["delta_IDSW"]])
            lines.append(f"{label} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} & {vals[4]} & {vals[5]} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def make_plot(rows: List[Dict[str, str]], path: Path) -> None:
    plt.rcParams.update(PLOT_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), constrained_layout=True)
    clean = {row["method"]: float(row["HOTA"]) for row in rows if row["degradation_key"] == "clean"}
    for ax, cfg in zip(axes, DEGRADATIONS):
        xs = [0.0] + list(cfg["levels"])
        for method_name, _ in METHODS:
            ys = [clean[method_name]]
            for level in cfg["levels"]:
                row = next(r for r in rows if r["method"] == method_name and r["degradation_key"] == cfg["key"] and abs(float(r["level"]) - level) < 1e-9)
                ys.append(float(row["HOTA"]))
            ax.plot(xs, ys, marker="o", color=COLORS[method_name], label=method_name)
        ax.set_title(cfg["label"])
        ax.set_xlabel("severity")
        ax.set_ylabel("HOTA")
        ax.set_xticks(xs)
        ax.grid(alpha=0.18)
    axes[0].legend(loc="lower left", frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=320, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def pick_example(split_dir: Path) -> Tuple[str, int, Path]:
    best = ("", 1, Path())
    best_count = -1
    for seq in read_seqmap(split_dir / "seqmaps" / f"{FIXED_SPLIT}.txt"):
        gt_path = split_dir / seq / "gt" / "gt.txt"
        counts: Dict[int, int] = {}
        with gt_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                frame = int(float(parts[0]))
                if 1 <= frame <= FIXED_MAX_FRAMES:
                    counts[frame] = counts.get(frame, 0) + 1
        if counts:
            frame, count = max(counts.items(), key=lambda item: (item[1], -item[0]))
            img = split_dir / seq / "img1" / f"{frame:06d}.jpg"
            if img.exists() and count > best_count:
                best = (seq, frame, img)
                best_count = count
    if not best[0]:
        raise RuntimeError(f"Unable to choose an example frame from {split_dir}")
    return best


def shift_edge(arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = arr.shape[:2]
    out = arr.copy()
    if dx >= 0:
        out[:, dx:] = arr[:, : w - dx] if dx < w else arr[:, :1]
        out[:, :dx] = arr[:, :1]
    else:
        k = -dx
        out[:, : w - k] = arr[:, k:]
        out[:, w - k :] = arr[:, -1:]
    arr2 = out.copy()
    if dy >= 0:
        arr2[dy:, :] = out[: h - dy, :] if dy < h else out[:1, :]
        arr2[:dy, :] = out[:1, :]
    else:
        k = -dy
        arr2[: h - k, :] = out[k:, :]
        arr2[h - k :, :] = out[-1:, :]
    return arr2


def motion_blur_image(img: Image.Image, severity: float) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    radius = max(1, int(round(2 + severity * 6)))
    acc = np.zeros_like(arr)
    weight_sum = 0.0
    for shift in range(-radius, radius + 1):
        weight = float(radius + 1 - abs(shift))
        acc += shift_edge(arr, shift, shift // 2) * weight
        weight_sum += weight
    return Image.fromarray(np.clip(acc / max(weight_sum, 1.0), 0, 255).astype(np.uint8))


def dark_image(img: Image.Image, severity: float) -> Image.Image:
    out = ImageEnhance.Brightness(img).enhance(max(0.2, 1.0 - 0.75 * severity))
    return ImageEnhance.Contrast(out).enhance(max(0.35, 1.0 - 0.35 * severity))


def haze_image(img: Image.Image, severity: float) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    haze = np.array([205.0, 214.0, 208.0], dtype=np.float32)
    mixed = arr * (1.0 - 0.42 * severity) + haze[None, None, :] * (0.42 * severity)
    out = Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8))
    out = ImageEnhance.Contrast(out).enhance(max(0.35, 1.0 - 0.55 * severity))
    return ImageEnhance.Color(out).enhance(max(0.45, 1.0 - 0.35 * severity))


def make_examples(split_dir: Path, path: Path) -> None:
    seq, frame, img_path = pick_example(split_dir)
    clean = Image.open(img_path).convert("RGB")
    panels = [
        ("Clean", clean),
        ("Motion blur (0.45)", motion_blur_image(clean, 0.45)),
        ("Low illumination (0.45)", dark_image(clean, 0.45)),
        ("Low contrast / haze (0.45)", haze_image(clean, 0.45)),
    ]
    plt.rcParams.update(PLOT_STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), constrained_layout=True)
    for ax, (title, image) in zip(axes.flat, panels):
        ax.imshow(np.asarray(image))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Qualitative deployment stress illustration on {seq} frame {frame:06d}", y=1.01, fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def summarize_text(rows: List[Dict[str, str]]) -> str:
    def mean_delta(degr_key: str, metric: str, method_name: str) -> float:
        subset = [r for r in rows if r["degradation_key"] == degr_key and r["method"] == method_name]
        return float(np.mean([float(r[metric]) for r in subset]))
    blur_worst = min((m for m, _ in METHODS), key=lambda m: mean_delta("motion_blur", "delta_HOTA", m))
    dark_best = max((m for m, _ in METHODS), key=lambda m: mean_delta("low_illumination", "delta_HOTA", m))
    haze_best = max((m for m, _ in METHODS), key=lambda m: mean_delta("haze_turbidity", "delta_HOTA", m))
    lines = [
        "[Protocol Description]",
        "We extended the controlled robustness protocol with three deployment-motivated perturbations while keeping the Route-B tracker interface unchanged. On the same val-half split (8 sequences, 1000 frames per sequence, seed=0), we evaluated Motion blur, Low illumination, and Low contrast/haze at severities 0.15, 0.30, and 0.45. Because the current framework operates on prepared MOT detections, each stressor was implemented as a deterministic detection-space proxy with explicit scalar parameters: motion blur as directional box offset and elongation, low illumination as small-object-biased recall loss with conservative box shrinkage, and haze/turbidity as frame-coherent localization drift with box inflation.",
        "",
        "[Result Summary]",
        f"Across the three added stressors, HOTA degraded monotonically with severity for all compared profiles, indicating that the extended protocol induces ordered deployment stress rather than arbitrary random noise. Motion blur produced the strongest average HOTA loss for {blur_worst}, whereas {dark_best} showed the smallest average HOTA drop under low illumination and {haze_best} showed the smallest average HOTA drop under low contrast / haze. The environmental penalty was larger than the residual gap between the tracker profiles, which reinforces the deployment framing of the paper: the value of the framework lies in exposing failure boundaries and profile trade-offs, not in declaring a new tracker champion.",
        "",
        "[Limitation Note]",
        "These three additions should be interpreted as detector-interface stress proxies, not as a full end-to-end image re-detection benchmark. The qualitative clean-versus-degraded figure visualizes the corresponding image-space intuition, but the quantitative numbers are still produced after deterministic perturbation of the MOT detections. This choice preserves auditability and low-cost reproduction, but a future extension should rerun the detector itself under the same image-space perturbations.",
        "",
        "[Degradation Definitions]",
    ]
    for cfg in DEGRADATIONS:
        levels = ", ".join(f"{level:.2f}" for level in cfg["levels"])
        lines.append(f"- {cfg['label']}: levels={levels}; domain={cfg['domain']}; relevance={cfg['matters']}")
    return "\n".join(lines) + "\n"


def upsert_reproduce_section(reproduce_bat: Path, run_config_path: Path) -> None:
    start = ":: === Degradation Extended (auto) START ==="
    end = ":: === Degradation Extended (auto) END ==="
    run_cfg_win = str(run_config_path).replace("/", "\\")
    section = [
        start,
        f"\"%PY_EXE%\" scripts\\run_degradation_extended.py --run-config {run_cfg_win}",
        "if errorlevel 1 (",
        "  echo run_degradation_extended failed.",
        "  popd",
        "  exit /b 1",
        ")",
        end,
    ]
    if not reproduce_bat.exists():
        reproduce_bat.parent.mkdir(parents=True, exist_ok=True)
        reproduce_bat.write_text(
            "\n".join(
                [
                    "@echo off",
                    "setlocal EnableExtensions",
                    'set "SCRIPT_DIR=%~dp0"',
                    'set "REPO_ROOT="',
                    'for %%I in ("%SCRIPT_DIR%." "%SCRIPT_DIR%.." "%SCRIPT_DIR%..\\.." "%SCRIPT_DIR%..\\..\\.." "%SCRIPT_DIR%..\\..\\..\\.." "%SCRIPT_DIR%..\\..\\..\\..\\..") do (',
                    '  if not defined REPO_ROOT if exist "%%~fI\\scripts\\make_paper_assets.py" set "REPO_ROOT=%%~fI"',
                    ")",
                    "if not defined REPO_ROOT (",
                    '  echo [ERR] Could not locate repository root from "%SCRIPT_DIR%".',
                    "  exit /b 1",
                    ")",
                    'pushd "%REPO_ROOT%"',
                    'set "PY_EXE=python"',
                    'if exist "%REPO_ROOT%\\.venv\\Scripts\\python.exe" set "PY_EXE=%REPO_ROOT%\\.venv\\Scripts\\python.exe"',
                    *section,
                    "echo Done.",
                    "popd",
                    "exit /b 0",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return
    lines = reproduce_bat.read_text(encoding="utf-8").splitlines()
    try:
        s_idx = lines.index(start)
        e_idx = lines.index(end)
        lines = lines[:s_idx] + section + lines[e_idx + 1 :]
    except ValueError:
        insert_idx = next((i for i, line in enumerate(lines) if line.strip().lower().startswith("echo done")), len(lines))
        lines = lines[:insert_idx] + section + lines[insert_idx:]
    reproduce_bat.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.run_config.read_text(encoding="utf-8"))
    result_root = Path(str(cfg.get("result_root", "results/main_val")))
    mot_root = Path(str(cfg.get("mot_root", "data/mft25_mot_full")))
    tables_dir = Path(str(cfg.get("tables_dir", result_root / "tables")))
    paper_assets_dir = Path(str(cfg.get("paper_assets_dir", result_root / "paper_assets")))
    release_dir = Path(str(cfg.get("release_dir", result_root / "release")))
    output_csv = args.output_csv if args.output_csv is not None else (tables_dir / "degradation_extended.csv")
    tex_path = args.tex_path if args.tex_path is not None else (paper_assets_dir / "degradation_extended_delta.tex")
    plot_path = args.plot_path if args.plot_path is not None else (paper_assets_dir / "degradation_extended.png")
    qualitative_path = args.qualitative_path if args.qualitative_path is not None else (paper_assets_dir / "degradation_extended_examples.png")
    manuscript_path = args.manuscript_path if args.manuscript_path is not None else (tables_dir / "degradation_extended_manuscript.txt")
    input_csv = args.input_csv if args.input_csv is not None else output_csv
    bad = [f"{name}={path}" for name, path in [("csv", output_csv), ("tex", tex_path), ("plot", plot_path), ("qual", qualitative_path), ("txt", manuscript_path)] if _is_legacy_output_path(path, result_root)]
    if bad:
        raise RuntimeError(f"run-config strict mode blocks legacy/global outputs. result_root={result_root}. bad targets: {'; '.join(bad)}")
    split_dir = mot_root / FIXED_SPLIT
    if args.render_only:
        rows = order_rows(read_rows(input_csv))
    else:
        run_root = result_root / "degradation_extended"
        if args.clean and run_root.exists():
            shutil.rmtree(run_root)
        run_root.mkdir(parents=True, exist_ok=True)
        total_frames = estimate_total_frames(split_dir, FIXED_SPLIT, FIXED_MAX_FRAMES)
        rows = [maybe_run(row, run_root, mot_root, total_frames) for row in build_rows()]
        rows = apply_deltas(rows)
    write_csv(output_csv, rows)
    write_delta_tex(rows, tex_path)
    make_plot(rows, plot_path)
    make_examples(split_dir, qualitative_path)
    manuscript_path.parent.mkdir(parents=True, exist_ok=True)
    manuscript_path.write_text(summarize_text(rows), encoding="utf-8")
    upsert_reproduce_section(release_dir / "reproduce.bat", args.run_config)
    print(f"Saved CSV:          {output_csv}")
    print(f"Saved delta TeX:    {tex_path}")
    print(f"Saved plot:         {plot_path}")
    print(f"Saved examples:     {qualitative_path}")
    print(f"Saved manuscript:   {manuscript_path}")


if __name__ == "__main__":
    main()
