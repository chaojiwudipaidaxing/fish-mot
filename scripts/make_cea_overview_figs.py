#!/usr/bin/env python3
"""Generate journal-style overview figures (Fig.1-Fig.7)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import pandas as pd

COL_W = 3.55
EDGE = "#565D66"
TEXT = "#20262D"
CARD = "#FAFAF8"
BLUE = "#A8BED8"
BLUE_DARK = "#4E6F90"
GREEN = "#A8C2A4"
OCHRE = "#D8C29B"
RED = "#C27767"
VIOLET = "#C2BEDF"
GT_CACHE: dict[str, pd.DataFrame] = {}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
        "font.size": 7.0,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.edgecolor": EDGE,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 500,
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, out_base: Path) -> None:
    pdf_path = out_base.with_suffix(".pdf")
    png_path = out_base.with_suffix(".png")
    _ensure_parent(pdf_path)
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote: {pdf_path}")
    print(f"wrote: {png_path}")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_image(path: Path) -> np.ndarray:
    return plt.imread(str(path))


def _mft_gt(mft_val_root: Path, seq: str) -> pd.DataFrame:
    gt_path = mft_val_root / seq / "gt" / "gt.txt"
    key = str(gt_path)
    if key not in GT_CACHE:
        df = pd.read_csv(gt_path, header=None)
        df.columns = ["frame", "track_id", "x", "y", "w", "h", "mark", "cls", "vis"]
        GT_CACHE[key] = df
    return GT_CACHE[key]


def _mft_frame(mft_val_root: Path, seq: str, frame: int) -> np.ndarray:
    return _load_image(mft_val_root / seq / "img1" / f"{frame:06d}.jpg")


def _mft_boxes(mft_val_root: Path, seq: str, frame: int) -> pd.DataFrame:
    return _mft_gt(mft_val_root, seq).query("frame == @frame").copy()


def _track_tail(mft_val_root: Path, seq: str, frame: int, track_id: int, hist: int = 12) -> np.ndarray:
    df = _mft_gt(mft_val_root, seq)
    keep = df[(df["track_id"] == track_id) & (df["frame"] <= frame) & (df["frame"] >= frame - hist + 1)]
    keep = keep.sort_values("frame")
    if keep.empty:
        return np.zeros((0, 2))
    return np.column_stack((keep["x"] + keep["w"] / 2.0, keep["y"] + keep["h"] / 2.0))


def _brackish_frame(brackish_root: Path, split: str, name: str) -> np.ndarray:
    return _load_image(brackish_root / "images" / split / f"{name}.jpg")


def _brackish_boxes(brackish_root: Path, split: str, name: str) -> pd.DataFrame:
    label_path = brackish_root / "labels" / split / f"{name}.txt"
    img = _brackish_frame(brackish_root, split, name)
    h, w = img.shape[:2]
    rows = []
    if label_path.exists():
        for idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            cls_id, xc, yc, bw, bh = map(float, line.split())
            rows.append(
                {
                    "track_id": idx + 1,
                    "cls": cls_id,
                    "x": xc * w - bw * w / 2.0,
                    "y": yc * h - bh * h / 2.0,
                    "w": bw * w,
                    "h": bh * h,
                }
            )
    return pd.DataFrame(rows)


def _panel(ax: plt.Axes, label: str, title: str, note: str | None = None) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(EDGE)
        spine.set_linewidth(0.8)
    ax.text(
        0.012,
        0.988,
        f"({label}) {title}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        weight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.7},
    )
    if note:
        ax.text(
            0.012,
            0.018,
            note,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.0,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
        )


def _boxes(ax: plt.Axes, boxes: pd.DataFrame, color: str, lw: float, alpha: float = 0.9, annotate: list[int] | None = None) -> None:
    labels = set(annotate or [])
    for _, row in boxes.iterrows():
        x, y, w, h = float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=lw, alpha=alpha))
        tid = int(row["track_id"])
        if tid in labels:
            ax.text(
                x,
                max(4.0, y - 4.0),
                f"id={tid}",
                color=color,
                fontsize=5.8,
                weight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.8},
            )


def _card(ax: plt.Axes, x: float, y: float, w: float, h: float, head: str, lines: list[str], accent: str, mono: bool = False) -> None:
    hh = min(0.17 * h, 0.11)
    ax.add_patch(Rectangle((x, y), w, h, facecolor=CARD, edgecolor=EDGE, linewidth=0.8, transform=ax.transAxes))
    ax.add_patch(Rectangle((x, y + h - hh), w, hh, facecolor=accent, edgecolor=EDGE, linewidth=0.8, transform=ax.transAxes))
    ax.text(x + 0.03 * w, y + h - hh / 2.0, head, transform=ax.transAxes, ha="left", va="center", fontsize=6.4, weight="bold")
    ty = y + h - hh - 0.06 * h
    step = (h - hh - 0.10 * h) / max(len(lines), 1)
    for line in lines:
        ax.text(
            x + 0.04 * w,
            ty,
            line,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.0,
            fontfamily="monospace" if mono else None,
        )
        ty -= step


def _stage(ax: plt.Axes, y: float, code: str, head: str, lines: list[str], accent: str) -> None:
    ax.add_patch(Rectangle((0.02, y), 0.15, 0.19, facecolor=accent, edgecolor=EDGE, linewidth=0.8, transform=ax.transAxes))
    ax.text(0.095, y + 0.095, code, transform=ax.transAxes, ha="center", va="center", fontsize=6.9, weight="bold")
    _card(ax, 0.20, y, 0.78, 0.19, head, lines, accent)


def _arrow(ax: plt.Axes, p1: tuple[float, float], p2: tuple[float, float], color: str = EDGE) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            color=color,
            transform=ax.transAxes,
        )
    )


def _bucket_delta(bucket_csv: Path) -> dict[str, dict[str, float]]:
    if not bucket_csv.exists():
        return {}
    df = pd.read_csv(bucket_csv)
    if "group" in df.columns:
        df = df[df["group"].astype(str).str.lower().isin(["high", "turbid_high"])]
    out: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        out[str(row["bucket_name"]).strip().lower()] = {
            "delta_idsw": float(row.get("delta_idsw", np.nan)),
            "delta_countmae": float(row.get("delta_countmae", np.nan)),
        }
    return out


def _iou_pair(boxes: pd.DataFrame) -> tuple[int, int]:
    best = (0, 1, -1.0)
    arr = boxes[["x", "y", "w", "h"]].to_numpy(float)
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            ax1, ay1, aw, ah = arr[i]
            bx1, by1, bw, bh = arr[j]
            ax2, ay2 = ax1 + aw, ay1 + ah
            bx2, by2 = bx1 + bw, by1 + bh
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = aw * ah + bw * bh - inter + 1e-9
            iou = inter / union
            if iou > best[2]:
                best = (i, j, iou)
    return int(best[0]), int(best[1])


def _crop(boxes: pd.DataFrame, img_shape: tuple[int, ...], pad: float) -> tuple[float, float, float, float]:
    h, w = img_shape[:2]
    x1 = max(0.0, float(boxes["x"].min()) - pad)
    y1 = max(0.0, float(boxes["y"].min()) - pad)
    x2 = min(float(w), float((boxes["x"] + boxes["w"]).max()) + pad)
    y2 = min(float(h), float((boxes["y"] + boxes["h"]).max()) + pad)
    return x1, y1, x2, y2


def _recommended(summary_csv: Path) -> dict:
    if not summary_csv.exists():
        return {}
    df = pd.read_csv(summary_csv)
    keep = df[(df["experiment"].astype(str) == "E2") & (df["recommended"].fillna(0).astype(int) == 1)]
    if keep.empty:
        keep = df[df["recommended"].fillna(0).astype(int) == 1]
    return {} if keep.empty else keep.iloc[0].to_dict()


def _timeline(timeline_csv: Path) -> pd.DataFrame:
    if not timeline_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(timeline_csv)
    keep = df[df["scenario"].astype(str) == "drop_jitter_high"].copy()
    if keep.empty:
        keep = df.copy()
    seq = keep.groupby("seq")["alert"].sum().idxmax()
    return keep[keep["seq"] == seq].sort_values("window_idx")


def _scope_a(runtime_csv: Path) -> dict[str, dict[str, float]]:
    if not runtime_csv.exists():
        return {}
    df = pd.read_csv(runtime_csv)
    out: dict[str, dict[str, float]] = {}
    for method in ["Base", "+gating"]:
        cur = df[df["method"].astype(str) == method]
        if not cur.empty:
            row = cur.iloc[0]
            out[method] = {"fps": float(row["fps"]), "mem": float(row["mem_peak_mb"]), "cpu": float(row["cpu_mean_norm_percent"])}
    return out


def _scope_b(runtime_csv: Path) -> dict[tuple[str, str], dict[str, float]]:
    if not runtime_csv.exists():
        return {}
    df = pd.read_csv(runtime_csv)
    out: dict[tuple[str, str], dict[str, float]] = {}
    for _, row in df.iterrows():
        total = float(row["elapsed_e2e_sec"])
        out[(str(row["scenario"]), str(row["method"]))] = {
            "fps": float(row["fps_e2e"]),
            "mem": float(row["mem_peak_mb_e2e"]),
            "detector_share": float(row["detector_time"]) / max(total, 1e-9),
        }
    return out


def fig1_framework(out_base: Path, mft_val_root: Path, run_config_path: Path, drift_summary_csv: Path) -> None:
    image = _mft_frame(mft_val_root, "MSK-002", 1)
    boxes = _mft_boxes(mft_val_root, "MSK-002", 1)
    run_cfg = _read_json(run_config_path)
    drift = _recommended(drift_summary_csv)

    fig = plt.figure(figsize=(COL_W, 4.85))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.18, 2.00], hspace=0.12)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_flow = fig.add_subplot(gs[1, 0])

    ax_img.imshow(image)
    _boxes(ax_img, boxes, "#F0E0A1", 0.55, 0.76)
    _panel(ax_img, "a", "Video ingestion", "val-half / MSK-002 / frame 000001 / 49 fish instances")

    ax_flow.set_axis_off()
    _stage(ax_flow, 0.77, "C1", "Audit and reproducibility", [f"split={run_cfg.get('split', '-')}, max_frames={run_cfg.get('max_frames', '-')}", f"seeds={list(run_cfg.get('seeds', []))}", "run_config.json + dual manifests + release bundle"], BLUE)
    _stage(ax_flow, 0.52, "C2", "Stress and stratified diagnostics", ["global metrics + bucket metrics + stress groups", "occlusion / density / turning / low-confidence", "localize risk before deployment selection"], GREEN)
    _stage(ax_flow, 0.27, "C3", "Window-level drift monitoring", ["compute D_in(t) and D_out(t) over sliding windows", f"selected rule: K={int(drift.get('K', 1))}, q={int(drift.get('quantile', 99))}", f"false-alert={100.0 * drift.get('false_alert_rate', 0.0):.1f}% / missed-drift={100.0 * drift.get('missed_drift_rate', 0.0):.1f}%"], OCHRE)
    _stage(ax_flow, 0.02, "OUT", "Deployment outputs", ["HOTA / IDF1 / IDSW / CountMAE", "FPS / peak memory / normalized CPU", "Base vs +gating policy recommendation"], VIOLET)
    for y1, y2 in [(0.77, 0.71), (0.52, 0.46), (0.27, 0.21)]:
        _arrow(ax_flow, (0.52, y1), (0.52, y2))
    ax_flow.text(0.02, 0.985, "(b) Closed-loop reviewer map", transform=ax_flow.transAxes, ha="left", va="top", fontsize=7.0, weight="bold")
    _save(fig, out_base)


def fig2_monitoring_flow(out_base: Path, timeline_csv: Path, summary_csv: Path) -> None:
    tl = _timeline(timeline_csv)
    rec = _recommended(summary_csv)

    fig = plt.figure(figsize=(COL_W, 4.55))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.45, 0.70, 1.30], hspace=0.24)
    ax_plot = fig.add_subplot(gs[0, 0])
    ax_math = fig.add_subplot(gs[1, 0])
    ax_flow = fig.add_subplot(gs[2, 0])

    if not tl.empty:
        x = tl["window_idx"].to_numpy()
        din = tl["D_in"].to_numpy(float)
        dout = tl["D_out"].to_numpy(float)
        tin = float(tl["tau_in"].iloc[0])
        tout = float(tl["tau_out"].iloc[0])
        ax_plot.plot(x, din, color=BLUE_DARK, linewidth=1.3, marker="o", markersize=2.6, label=r"$D_{in}(t)$")
        ax_plot.plot(x, dout, color=RED, linewidth=1.2, marker="s", markersize=2.4, label=r"$D_{out}(t)$")
        ax_plot.axhline(tin, color=BLUE_DARK, linestyle="--", linewidth=0.8, alpha=0.75)
        ax_plot.axhline(tout, color=RED, linestyle="--", linewidth=0.8, alpha=0.75)
        for wx, alert in zip(x, tl["alert"].fillna(0).to_numpy(int)):
            if alert == 1:
                ax_plot.axvspan(wx - 0.48, wx + 0.48, color=OCHRE, alpha=0.18, linewidth=0)
        ax_plot.set_xlabel("Window index")
        ax_plot.set_ylabel("Indicator value")
        ax_plot.grid(axis="y", linestyle="--", alpha=0.25)
        ax_plot.legend(loc="upper left", ncol=2, frameon=False, fontsize=6.1)
        ax_plot.text(0.99, 0.04, f"selected: q={int(rec.get('quantile', 99))}, K={int(rec.get('K', 1))}", transform=ax_plot.transAxes, ha="right", va="bottom", fontsize=6.0)
    _panel(ax_plot, "a", "Indicator trajectory on drift windows", "alerted windows are lightly shaded")

    ax_math.set_axis_off()
    _card(ax_math, 0.00, 0.10, 0.31, 0.82, "Window statistics", ["window W = 100 frames", "feature CDF shift", "metric deviation vector"], BLUE)
    _card(ax_math, 0.345, 0.10, 0.30, 0.82, "Input term", [r"$D_{in}=\sup_x |F_t(x)-F_{ref}(x)|$", "KS/CDF distance"], GREEN)
    _card(ax_math, 0.68, 0.10, 0.32, 0.82, "Output term", [r"$D_{out}=||q_t-q_{ref}||_1/(|q_{ref}|+\epsilon)$", "q=[IDSW, CountMAE, F1]"], OCHRE)

    ax_flow.set_axis_off()
    _card(ax_flow, 0.00, 0.34, 0.30, 0.54, "Threshold test", ["if D_in > tau_in", "or D_out > tau_out"], BLUE)
    _card(ax_flow, 0.35, 0.34, 0.28, 0.54, "Persistence", [f"counter c <- c+1", f"trigger when c >= {int(rec.get('K', 1))}"], GREEN)
    _card(ax_flow, 0.68, 0.34, 0.32, 0.54, "Audit and policy", [f"false-alert={100.0 * rec.get('false_alert_rate', 0.0):.1f}%", f"missed-drift={100.0 * rec.get('missed_drift_rate', 0.0):.1f}%", "switch to conservative +gating profile"], VIOLET)
    _arrow(ax_flow, (0.30, 0.61), (0.35, 0.61))
    _arrow(ax_flow, (0.63, 0.61), (0.68, 0.61))
    ax_flow.text(0.00, 0.98, "(b) Selected trigger rule", transform=ax_flow.transAxes, ha="left", va="top", fontsize=7.0, weight="bold")
    _save(fig, out_base)


def fig3_audit_chain(out_base: Path, run_config_path: Path, manifest_path: Path, sha_path: Path, bundle_meta_path: Path) -> None:
    run_cfg = _read_json(run_config_path)
    manifest = _read_json(manifest_path)
    bundle = _read_json(bundle_meta_path)
    sha_lines = []
    if sha_path.exists():
        for line in sha_path.read_text(encoding="utf-8").splitlines()[:3]:
            token = line.split()[0]
            short = token if len(token) <= 14 else f"{token[:8]}...{token[-6:]}"
            sha_lines.append(f"{short}  {line.split()[-1].split('/')[-1]}")

    fig = plt.figure(figsize=(COL_W, 4.15))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    _card(ax, 0.00, 0.77, 1.00, 0.19, "(a) run_config.json", [f"exp_id        : {run_cfg.get('exp_id', '-')}", f"split         : {run_cfg.get('split', '-')}", f"max_frames    : {run_cfg.get('max_frames', '-')}", f"gating_thresh : {run_cfg.get('gating_thresh', '-')}"], BLUE, mono=True)
    _card(ax, 0.00, 0.53, 1.00, 0.19, "(b) manifest.json", [f"schema_version: {manifest.get('schema_version', '-')}", f"command       : {str(manifest.get('command', '-'))[:29]}", f"python        : {manifest.get('platform', {}).get('python', '-')}", f"git_commit    : {manifest.get('git_commit', '-')}"], GREEN, mono=True)
    _card(ax, 0.00, 0.29, 1.00, 0.19, "(c) sha256_manifest.txt", sha_lines or ["manifest not found"], OCHRE, mono=True)
    _card(ax, 0.00, 0.05, 1.00, 0.19, "(d) bundle_meta.json", [f"bundle      : {bundle.get('bundle', '-')}", f"generated   : {str(bundle.get('generated_utc', '-'))[:19]}", f"git_commit  : {bundle.get('git_commit', '-')}"], VIOLET, mono=True)
    for y1, y2 in [(0.77, 0.72), (0.53, 0.48), (0.29, 0.24)]:
        _arrow(ax, (0.50, y1), (0.50, y2))
    _save(fig, out_base)


def fig4_decision_tree(out_base: Path, scope_a_csv: Path, scope_b_csv: Path, drift_selected_json: Path) -> None:
    sa = _scope_a(scope_a_csv)
    sb = _scope_b(scope_b_csv)
    drift = _read_json(drift_selected_json) if drift_selected_json.exists() else {}
    base = sa.get("Base", {})
    gating = sa.get("+gating", {})
    clear = sb.get(("clear", "Base"), {})
    turbid = sb.get(("turbid_high", "+gating"), {})

    fig = plt.figure(figsize=(COL_W, 4.45))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.92, 1.75], hspace=0.16)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_tree = fig.add_subplot(gs[1, 0])
    ax_top.set_axis_off()
    ax_tree.set_axis_off()

    _card(ax_top, 0.00, 0.08, 0.32, 0.84, "(a) Scope A", [f"Base   : {base.get('fps', 0.0):.0f} FPS", f"+gating: {gating.get('fps', 0.0):.0f} FPS", f"peak RSS ~ {base.get('mem', 0.0):.0f} MB"], BLUE)
    _card(ax_top, 0.34, 0.08, 0.32, 0.84, "(b) Scope B", [f"clear/Base      : {clear.get('fps', 0.0):.1f} FPS", f"turbid/+gating : {turbid.get('fps', 0.0):.1f} FPS", f"detector share ~ {100.0 * turbid.get('detector_share', 0.0):.0f}%"], GREEN)
    _card(ax_top, 0.68, 0.08, 0.32, 0.84, "(c) Drift scan", [f"tau_in={float(drift.get('tau_in', 0.0)):.3f}", f"tau_out={float(drift.get('tau_out', 0.0)):.3f}", f"false-alert={100.0 * float(drift.get('false_alert_rate', 0.0)):.1f}%", f"missed-drift={100.0 * float(drift.get('missed_drift_rate', 0.0)):.1f}%"], OCHRE)

    _card(ax_tree, 0.30, 0.80, 0.40, 0.14, "Deployment target", ["real-time monitoring or offline analysis?"], VIOLET)
    _card(ax_tree, 0.05, 0.56, 0.36, 0.14, "Strict end-to-end FPS budget?", ["if yes, optimize detector first"], GREEN)
    _card(ax_tree, 0.59, 0.56, 0.36, 0.14, "Drift / occlusion risk high?", ["if yes, favor conservative association"], OCHRE)
    _card(ax_tree, 0.01, 0.20, 0.29, 0.18, "Leaf A", ["Detector-first", "reduce detector load", "tracking is not dominant"], BLUE)
    _card(ax_tree, 0.355, 0.20, 0.29, 0.18, "Leaf B", ["Base profile", "clean and stable scenes", "metric ranking priority"], GREEN)
    _card(ax_tree, 0.70, 0.20, 0.29, 0.18, "Leaf C", ["+gating profile", "drift risk or overlap rises", "safer association under stress"], RED)
    _arrow(ax_tree, (0.50, 0.80), (0.23, 0.70))
    _arrow(ax_tree, (0.50, 0.80), (0.77, 0.70))
    _arrow(ax_tree, (0.23, 0.56), (0.16, 0.38))
    _arrow(ax_tree, (0.77, 0.56), (0.84, 0.38))
    _arrow(ax_tree, (0.77, 0.56), (0.50, 0.38))
    _save(fig, out_base)


def fig5_eval_pipeline(out_base: Path, mft_val_root: Path, brackish_root: Path, run_config_path: Path) -> None:
    run_cfg = _read_json(run_config_path)
    img_input = _mft_frame(mft_val_root, "BT-005", 675)
    oracle_boxes = _mft_boxes(mft_val_root, "BT-005", 675)
    img_brackish = _brackish_frame(brackish_root, "train", "brackishMOT-18__000096")
    brackish_boxes = _brackish_boxes(brackish_root, "train", "brackishMOT-18__000096")

    fig = plt.figure(figsize=(COL_W, 4.85))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.10, 1.05, 1.18], hspace=0.18)
    ax_input = fig.add_subplot(gs[0, 0])
    mid = gs[1, 0].subgridspec(1, 2, wspace=0.08)
    ax_oracle = fig.add_subplot(mid[0, 0])
    ax_scopeb = fig.add_subplot(mid[0, 1])
    ax_flow = fig.add_subplot(gs[2, 0])

    ax_input.imshow(img_input)
    _panel(ax_input, "a", "Frozen split input scene", "BT-005 / frame 000675 / audited val-half")

    ax_oracle.imshow(img_input)
    _boxes(ax_oracle, oracle_boxes, "#F0E0A1", 0.7, 0.85)
    _panel(ax_oracle, "b", "Oracle-detection branch", "oracle detections")

    ax_scopeb.imshow(img_brackish)
    if not brackish_boxes.empty:
        _boxes(ax_scopeb, brackish_boxes, "#F4D35E", 0.85, 0.92)
    _panel(ax_scopeb, "c", "Scope B-true branch", "end-to-end timed path")

    ax_flow.set_axis_off()
    _card(ax_flow, 0.00, 0.70, 1.00, 0.20, "Tracker bank", ["Base / +gating / +traj / +adaptive", "ByteTrack / OC-SORT / BoT-SORT"], BLUE)
    _card(ax_flow, 0.00, 0.41, 1.00, 0.20, "Evaluator outputs", ["TrackEval MOT metrics + CountMAE", "runtime profile + normalized stage cost"], GREEN)
    _card(ax_flow, 0.00, 0.12, 1.00, 0.20, "Frozen fairness lock", [f"split={run_cfg.get('split', '-')}; max_frames={run_cfg.get('max_frames', '-')}", f"seeds={list(run_cfg.get('seeds', []))}; identical detection source"], OCHRE)
    _arrow(ax_flow, (0.50, 0.70), (0.50, 0.61))
    _arrow(ax_flow, (0.50, 0.41), (0.50, 0.32))
    ax_flow.text(0.00, 0.02, "(d) Two detector-source branches converge to one audited comparison protocol.", transform=ax_flow.transAxes, ha="left", va="bottom", fontsize=6.1)
    _save(fig, out_base)


def fig6_failure_modes(out_base: Path, bucket_csv: Path, mft_val_root: Path, brackish_root: Path) -> None:
    info = _bucket_delta(bucket_csv)
    density_img = _mft_frame(mft_val_root, "MSK-002", 1)
    density_boxes = _mft_boxes(mft_val_root, "MSK-002", 1)
    occ_img = _mft_frame(mft_val_root, "BT-005", 675)
    occ_boxes = _mft_boxes(mft_val_root, "BT-005", 675)
    i, j = _iou_pair(occ_boxes)
    occ_pair = occ_boxes.iloc[[i, j]].copy()
    turn_img = _mft_frame(mft_val_root, "BT-001", 117)
    turn_boxes = _mft_boxes(mft_val_root, "BT-001", 117)
    turn_tail = _track_tail(mft_val_root, "BT-001", 117, 1, 12)
    turn_focus = turn_boxes[turn_boxes["track_id"].astype(int) == 1]
    low_img = _brackish_frame(brackish_root, "train", "brackishMOT-18__000096")
    low_boxes = _brackish_boxes(brackish_root, "train", "brackishMOT-18__000096")

    fig = plt.figure(figsize=(COL_W, 3.10))
    gs = fig.add_gridspec(2, 2, wspace=0.08, hspace=0.10)
    axs = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])], [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]])

    ax = axs[0, 0]
    ax.imshow(density_img)
    _boxes(ax, density_boxes, "#E5D08A", 0.48, 0.80)
    _panel(ax, "a", "Density", "MSK-002")
    ax.text(0.98, 0.02, f"dIDSW={info.get('density', {}).get('delta_idsw', np.nan):+.1f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.9, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.3})

    ax = axs[0, 1]
    ax.imshow(occ_img)
    _boxes(ax, occ_boxes, "#B7C7D9", 0.55, 0.72)
    _boxes(ax, occ_pair.iloc[[0]], RED, 1.15, 1.0, [int(occ_pair.iloc[0]["track_id"])])
    _boxes(ax, occ_pair.iloc[[1]], BLUE_DARK, 1.15, 1.0, [int(occ_pair.iloc[1]["track_id"])])
    _panel(ax, "b", "Occlusion", "BT-005")
    ins = inset_axes(ax, width="34%", height="34%", loc="lower right", borderpad=1.0)
    ins.imshow(occ_img)
    _boxes(ins, occ_pair.iloc[[0]], RED, 1.10)
    _boxes(ins, occ_pair.iloc[[1]], BLUE_DARK, 1.10)
    x1, y1, x2, y2 = _crop(occ_pair, occ_img.shape, 40.0)
    ins.set_xlim(x1, x2)
    ins.set_ylim(y2, y1)
    ins.set_xticks([])
    ins.set_yticks([])
    for spine in ins.spines.values():
        spine.set_edgecolor(EDGE)
        spine.set_linewidth(0.8)
    mark_inset(ax, ins, loc1=2, loc2=4, fc="none", ec=EDGE, lw=0.6)
    ax.text(0.98, 0.02, f"dI={info.get('occlusion', {}).get('delta_idsw', np.nan):+.1f}  dC={info.get('occlusion', {}).get('delta_countmae', np.nan):+.3f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.3})

    ax = axs[1, 0]
    ax.imshow(turn_img)
    _boxes(ax, turn_boxes, "#C6D1DE", 0.55, 0.70)
    if not turn_focus.empty:
        _boxes(ax, turn_focus, RED, 1.15, 1.0, [1])
    if len(turn_tail) > 1:
        ax.plot(turn_tail[:, 0], turn_tail[:, 1], color=RED, linewidth=1.15, marker="o", markersize=2.2)
        ax.annotate("", xy=tuple(turn_tail[-1]), xytext=tuple(turn_tail[-2]), arrowprops={"arrowstyle": "->", "lw": 0.9, "color": RED})
    _panel(ax, "c", "Turning", "BT-001")
    ax.text(0.98, 0.02, f"dI={info.get('turning', {}).get('delta_idsw', np.nan):+.1f}  dC={info.get('turning', {}).get('delta_countmae', np.nan):+.3f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.3})

    ax = axs[1, 1]
    ax.imshow(low_img)
    if not low_boxes.empty:
        _boxes(ax, low_boxes, "#F3D271", 0.95, 0.95)
        ins = inset_axes(ax, width="34%", height="34%", loc="lower right", borderpad=1.0)
        ins.imshow(low_img)
        _boxes(ins, low_boxes, "#F3D271", 1.05)
        x1, y1, x2, y2 = _crop(low_boxes, low_img.shape, 55.0)
        ins.set_xlim(x1, x2)
        ins.set_ylim(y2, y1)
        ins.set_xticks([])
        ins.set_yticks([])
        for spine in ins.spines.values():
            spine.set_edgecolor(EDGE)
            spine.set_linewidth(0.8)
        mark_inset(ax, ins, loc1=2, loc2=4, fc="none", ec=EDGE, lw=0.6)
    _panel(ax, "d", "Low confidence", "BrackishMOT-18")
    ax.text(0.98, 0.02, f"dI={info.get('low-confidence', {}).get('delta_idsw', np.nan):+.1f}  dC={info.get('low-confidence', {}).get('delta_countmae', np.nan):+.3f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.3})
    _save(fig, out_base)


def fig7_runtime_breakdown(out_base: Path, runtime_csv: Path) -> None:
    if not runtime_csv.exists():
        raise FileNotFoundError(f"runtime summary not found: {runtime_csv}")
    df = pd.read_csv(runtime_csv)
    keep = df[df["scenario"].astype(str).str.lower().isin(["clear", "turbid_high"])].copy()
    keep["name"] = keep["scenario"].astype(str) + "\n" + keep["method"].astype(str)
    order = []
    for scenario in ["clear", "turbid_high"]:
        for method in ["Base", "+gating"]:
            rows = keep[(keep["scenario"] == scenario) & (keep["method"] == method)]
            if not rows.empty:
                order.append(rows.index[0])
    if order:
        keep = keep.loc[order]
    x = np.arange(len(keep))
    ms_per_frame = 1000.0 / keep["total_frames"].to_numpy(float)
    decode = keep["decode_time"].to_numpy(float) * ms_per_frame
    detector = keep["detector_time"].to_numpy(float) * ms_per_frame
    tracking = keep["tracking_time"].to_numpy(float) * ms_per_frame
    write = keep["write_time"].to_numpy(float) * ms_per_frame

    fig, ax = plt.subplots(figsize=(COL_W, 3.55))
    ax.bar(x, decode, label="decode", color="#A8C5E5")
    ax.bar(x, detector, bottom=decode, label="detector", color=BLUE_DARK)
    ax.bar(x, tracking, bottom=decode + detector, label="tracking", color=GREEN)
    ax.bar(x, write, bottom=decode + detector + tracking, label="write", color=OCHRE)
    ax.set_xticks(x, keep["name"])
    ax.set_ylabel("Stage cost (ms/frame)")
    ax.set_title("Scope B-true stage-cost breakdown")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(ncol=2, frameon=False, fontsize=6.0, loc="upper left")
    for i, (_, row) in enumerate(keep.iterrows()):
        total = float(decode[i] + detector[i] + tracking[i] + write[i])
        ax.text(i, total + 0.12, f"{total:.2f}\n{float(row['fps_e2e']):.1f} FPS", ha="center", va="bottom", fontsize=5.9)
    ax.text(0.00, -0.20, "Run-level sec/run values are normalized by total_frames so that clear and turbid-high scenarios remain comparable.", transform=ax.transAxes, ha="left", va="top", fontsize=5.8)
    _save(fig, out_base)


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Generate journal-style overview figures (Fig.1-Fig.7).")
    parser.add_argument("--fig-dir", type=Path, default=root / "paper" / "cea_draft" / "figures")
    parser.add_argument("--bucket-csv", type=Path, default=root / "results" / "brackishmot_bucket_shift.csv")
    parser.add_argument("--runtime-summary", type=Path, default=root / "results" / "brackishmot" / "runtime" / "runtime_profile_e2e_true_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = _repo_root()
    mft_val_root = root / "data" / "mft25_mot_full" / "val_half"
    brackish_root = root / "data" / "brackish_yolo"
    run_config = root / "paper" / "cea_draft" / "submission_package" / "reproducibility" / "run_config.json"
    manifest = root / "paper" / "cea_draft" / "submission_package" / "reproducibility" / "manifest.json"
    sha_path = root / "release_bundle_v1" / "sha256_manifest.txt"
    bundle = root / "release_bundle_v1" / "bundle_meta.json"
    drift_summary = root / "results" / "drift_eval_summary.csv"
    drift_timeline = root / "results" / "drift_eval_timeline.csv"
    scope_a_csv = root / "results" / "main_val" / "runtime" / "runtime_profile.csv"
    drift_selected = root / "results" / "brackishmot_drift_selected.json"

    fig1_framework(args.fig_dir / "fig1_framework_architecture", mft_val_root, run_config, drift_summary)
    fig2_monitoring_flow(args.fig_dir / "fig2_drift_monitoring_flow", drift_timeline, drift_summary)
    fig3_audit_chain(args.fig_dir / "fig3_audit_protocol_chain", run_config, manifest, sha_path, bundle)
    fig4_decision_tree(args.fig_dir / "fig4_deployment_decision_tree", scope_a_csv, args.runtime_summary, drift_selected)
    fig5_eval_pipeline(args.fig_dir / "fig5_evaluation_pipeline_overview", mft_val_root, brackish_root, run_config)
    fig6_failure_modes(args.fig_dir / "fig6_failure_modes_quad", args.bucket_csv, mft_val_root, brackish_root)
    fig7_runtime_breakdown(args.fig_dir / "fig7_runtime_stage_breakdown", args.runtime_summary)


if __name__ == "__main__":
    main()
