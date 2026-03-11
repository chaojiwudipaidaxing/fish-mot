#!/usr/bin/env python3
"""Render publication-ready support figures for the fish-mot paper."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import pandas as pd
from PIL import Image

import make_cea_overview_figs as ov
from eval_count_stability import compute_count_metrics, load_frame_counts, read_seq_length, read_seqmap
from run_degradation_extended import dark_image, haze_image, motion_blur_image, pick_example


EDGE = "#5E6670"
TEXT = "#24303A"
CARD = "#F7F8FA"
BLUE = "#DCE7F2"
BLUE_MID = "#A9BDD3"
BLUE_DARK = "#587592"
GRAY = "#D5DCE4"
GRAY_DARK = "#8A98A6"
ACCENT = "#748CA5"
WARN = "#AE7D75"
GOLD = "#C7B38A"
COLORS = {
    "Base": "#476883",
    "+gating": "#7E94AA",
    "ByteTrack": "#AEBECD",
}
MARKERS = {
    "Base": "o",
    "+gating": "s",
    "ByteTrack": "^",
}
SPLIT = "val_half"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
        "font.size": 7.2,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.edgecolor": EDGE,
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 500,
    }
)


CAPTIONS = {
    "framework_architecture": (
        "Framework architecture of the audit-ready fish-MOT evaluation pipeline. Pond video frames are decoded "
        "into detector outputs, processed by a SORT-style tracking core, optionally refined by gating, trajectory, "
        "and adaptive weighting modules, and then summarized by identity, counting, robustness, runtime, and "
        "deployment-decision outputs under a frozen experiment specification."
    ),
    "gating_logic_flow": (
        "Association-gating logic used in the tracking core. Predicted track states and candidate detections are "
        "scored by Mahalanobis distance, candidates outside the threshold are rejected, surviving candidates are "
        "passed to assignment, and the resulting trigger statistics are logged for downstream drift diagnostics."
    ),
    "degradation_combined": (
        "Combined degradation analysis used for deployment-oriented robustness reporting. (A) Qualitative view of "
        "the clean frame and three deterministic environmental stress proxies: motion blur, low illumination, and "
        "haze or turbidity. (B) Mean robustness curves across the three stress families; ribbons indicate the range "
        "across degradation types at the same severity. HOTA and IDF1 decrease as severity rises, while CountMAE "
        "increases, exposing the stability boundary relevant for pond deployment."
    ),
    "failure_cases_2x2": (
        "Representative qualitative cases used to communicate stability and failure boundaries. The four panels show "
        "a stable tracking example, a turning or crossing region associated with identity ambiguity, an occlusion-"
        "induced track-loss case, and a low-confidence miss where weak pond visibility suppresses a target before "
        "counting becomes visibly unstable."
    ),
}


FRAMEWORK_MERMAID = """flowchart LR
  A["Video input\\nPond video stream\\nframe decode"] --> B["Detector output\\nfish boxes + scores"]
  B --> C["Tracking core\\nKalman predict\\nHungarian assignment\\nstate update"]
  C --> D["Optional modules\\nMahalanobis gating\\ntrajectory cue\\nadaptive weighting"]
  D --> E["Evaluation and diagnostics\\nHOTA / IDF1 / IDSW\\nCount stability\\ndegradation robustness\\nruntime profile"]
  E --> F["Deployment recommendation\\nBase or conservative profile\\naudit-ready release bundle"]
  G["Audit-ready protocol\\nrun_config.json\\nmanifest_hash\\nseed_locking"] -. audit trail .-> C
  G -. reproducibility .-> E
  classDef main fill:#e8f0f8,stroke:#5e6670,color:#24303a;
  classDef support fill:#f7f8fa,stroke:#8a98a6,color:#24303a;
  class A,B,C,D,E,F main;
  class G support;
"""


GATING_MERMAID = """flowchart LR
  A["Predicted track state\\nx_i^- , S_i"] --> C["Mahalanobis gating score\\nd_ij = (z_j - H x_i^-)^T S_i^-1 (z_j - H x_i^-)"]
  B["Candidate detections\\nz_j"] --> C
  C --> D{"d_ij <= tau_g ?"}
  D -- yes --> E["Keep candidate\\nfor assignment"]
  D -- no --> F["Reject association"]
  E --> G["Hungarian matching\\nand track update"]
  F --> H["Log reject counts\\nscore tails\\ntrigger statistics"]
  G --> H
  H --> I["Diagnostic summary\\nfor drift monitoring\\nand audit review"]
  classDef main fill:#e8f0f8,stroke:#5e6670,color:#24303a;
  classDef support fill:#f7f8fa,stroke:#8a98a6,color:#24303a;
  class A,B,C,E,G,I main;
  class D,F,H support;
"""


@dataclass
class RepoContext:
    root: Path
    figure_dir: Path
    source_dir: Path
    caption_dir: Path
    mft_full_root: Path
    mft_root: Path
    brackish_root: Path
    degradation_csv: Path
    degradation_root: Path
    base_pred_root: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, out_base: Path) -> None:
    ensure_parent(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(out_base.with_suffix(".png"), dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote: {out_base.with_suffix('.pdf')}")
    print(f"wrote: {out_base.with_suffix('.png')}")


def write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text.strip() + "\n", encoding="utf-8")
    print(f"wrote: {path}")


def card(ax: plt.Axes, x: float, y: float, w: float, h: float, head: str, lines: list[str], accent: str = BLUE, body: str = CARD) -> None:
    hh = min(0.18 * h, 0.12)
    ax.add_patch(Rectangle((x, y), w, h, facecolor=body, edgecolor=EDGE, linewidth=0.85, transform=ax.transAxes))
    ax.add_patch(Rectangle((x, y + h - hh), w, hh, facecolor=accent, edgecolor=EDGE, linewidth=0.85, transform=ax.transAxes))
    ax.text(x + 0.03 * w, y + h - hh / 2.0, head, transform=ax.transAxes, ha="left", va="center", fontsize=6.5, weight="bold")
    ty = y + h - hh - 0.07 * h
    step = (h - hh - 0.12 * h) / max(len(lines), 1)
    for line in lines:
        ax.text(x + 0.04 * w, ty, line, transform=ax.transAxes, ha="left", va="top", fontsize=6.0)
        ty -= step


def arrow(ax: plt.Axes, p1: tuple[float, float], p2: tuple[float, float], color: str = EDGE, ls: str = "-", lw: float = 0.9) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=lw,
            linestyle=ls,
            color=color,
            transform=ax.transAxes,
        )
    )


def read_mot_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 6:
        raise RuntimeError(f"Unexpected MOT file format: {path}")
    cols = ["frame", "track_id", "x", "y", "w", "h"] + [f"extra_{i}" for i in range(df.shape[1] - 6)]
    df.columns = cols
    return df[["frame", "track_id", "x", "y", "w", "h"]].copy()


def frame_boxes(df: pd.DataFrame, frame: int) -> pd.DataFrame:
    cur = df[df["frame"].astype(int) == int(frame)].copy()
    return cur.sort_values(["track_id", "x", "y"])


def centers(boxes: pd.DataFrame) -> np.ndarray:
    if boxes.empty:
        return np.zeros((0, 2), dtype=float)
    return np.column_stack((boxes["x"].to_numpy(float) + boxes["w"].to_numpy(float) / 2.0, boxes["y"].to_numpy(float) + boxes["h"].to_numpy(float) / 2.0))


def pairwise_iou(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    if a.empty or b.empty:
        return np.zeros((len(a), len(b)), dtype=float)
    out = np.zeros((len(a), len(b)), dtype=float)
    arr_a = a[["x", "y", "w", "h"]].to_numpy(float)
    arr_b = b[["x", "y", "w", "h"]].to_numpy(float)
    for i, (ax1, ay1, aw, ah) in enumerate(arr_a):
        ax2, ay2 = ax1 + aw, ay1 + ah
        for j, (bx1, by1, bw, bh) in enumerate(arr_b):
            bx2, by2 = bx1 + bw, by1 + bh
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = aw * ah + bw * bh - inter + 1e-9
            out[i, j] = inter / union
    return out


def greedy_match(gt: pd.DataFrame, pred: pd.DataFrame, thresh: float = 0.35) -> tuple[list[int], list[int]]:
    ious = pairwise_iou(gt, pred)
    matched_gt: list[int] = []
    matched_pred: list[int] = []
    if ious.size == 0:
        return matched_gt, matched_pred
    while True:
        idx = np.unravel_index(np.argmax(ious), ious.shape)
        best = float(ious[idx])
        if best < thresh:
            break
        gi, pj = int(idx[0]), int(idx[1])
        matched_gt.append(gi)
        matched_pred.append(pj)
        ious[gi, :] = -1.0
        ious[:, pj] = -1.0
    return matched_gt, matched_pred


def max_pair_iou(boxes: pd.DataFrame) -> float:
    if len(boxes) < 2:
        return 0.0
    arr = boxes[["x", "y", "w", "h"]].to_numpy(float)
    best = 0.0
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
            best = max(best, inter / union)
    return best


def track_tail(df: pd.DataFrame, track_id: int, frame: int, hist: int = 12) -> np.ndarray:
    cur = df[(df["track_id"].astype(int) == int(track_id)) & (df["frame"].astype(int) <= int(frame)) & (df["frame"].astype(int) >= int(frame) - hist + 1)].copy()
    cur = cur.sort_values("frame")
    if cur.empty:
        return np.zeros((0, 2), dtype=float)
    return np.column_stack((cur["x"].to_numpy(float) + cur["w"].to_numpy(float) / 2.0, cur["y"].to_numpy(float) + cur["h"].to_numpy(float) / 2.0))


def choose_tracks(boxes: pd.DataFrame, k: int = 2) -> list[int]:
    if boxes.empty:
        return []
    cur = boxes.copy()
    cur["area"] = cur["w"].astype(float) * cur["h"].astype(float)
    return [int(x) for x in cur.sort_values("area", ascending=False)["track_id"].head(k).tolist()]


def find_stable_example(ctx: RepoContext) -> tuple[str, int]:
    candidates = [("BT-005", ctx.base_pred_root / "BT-005.txt"), ("BT-003", ctx.base_pred_root / "BT-003.txt")]
    best = ("BT-005", 675, -1e9)
    for seq, pred_path in candidates:
        gt = read_mot_table(ctx.mft_full_root / seq / "gt" / "gt.txt")
        pred = read_mot_table(pred_path)
        for frame in range(60, 901, 5):
            gt_f = frame_boxes(gt, frame)
            pred_f = frame_boxes(pred, frame)
            if len(gt_f) < 5 or len(gt_f) > 12 or pred_f.empty:
                continue
            mg, _ = greedy_match(gt_f, pred_f, 0.35)
            match_ratio = len(mg) / max(len(gt_f), 1)
            crowd = max_pair_iou(gt_f)
            diff = abs(len(gt_f) - len(pred_f)) / max(len(gt_f), 1)
            score = 2.2 * match_ratio - 0.9 * crowd - 0.6 * diff
            if score > best[2]:
                best = (seq, frame, score)
    return best[0], int(best[1])


def find_occlusion_case(ctx: RepoContext) -> tuple[str, int]:
    seq = "BT-005"
    gt = read_mot_table(ctx.mft_full_root / seq / "gt" / "gt.txt")
    pred = read_mot_table(ctx.base_pred_root / f"{seq}.txt")
    best = (675, -1e9)
    for frame in range(60, 901, 5):
        gt_f = frame_boxes(gt, frame)
        pred_f = frame_boxes(pred, frame)
        if len(gt_f) < 5:
            continue
        crowd = max_pair_iou(gt_f)
        mg, _ = greedy_match(gt_f, pred_f, 0.35)
        miss_ratio = 1.0 - len(mg) / max(len(gt_f), 1)
        if crowd < 0.08 or miss_ratio <= 0.0:
            continue
        score = 1.6 * crowd + 1.4 * miss_ratio
        if score > best[1]:
            best = (frame, score)
    return seq, int(best[0])


def find_turning_pair(boxes: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    if boxes.empty:
        return boxes, []
    centers_xy = centers(boxes)
    best = None
    best_d = float("inf")
    for i in range(len(centers_xy)):
        for j in range(i + 1, len(centers_xy)):
            d = float(np.linalg.norm(centers_xy[i] - centers_xy[j]))
            if d < best_d:
                best_d = d
                best = (i, j)
    if best is None:
        picks = boxes.head(2)
        return picks, [int(x) for x in picks["track_id"].tolist()]
    picks = boxes.iloc[[best[0], best[1]]].copy()
    return picks, [int(x) for x in picks["track_id"].tolist()]


def find_smallest_brackish_box(boxes: pd.DataFrame) -> pd.DataFrame:
    if boxes.empty:
        return boxes
    cur = boxes.copy()
    cur["area"] = cur["w"].astype(float) * cur["h"].astype(float)
    return cur.nsmallest(1, "area").drop(columns="area")


def add_circle_callout(ax: plt.Axes, boxes: pd.DataFrame, img_shape: tuple[int, ...], label: str) -> None:
    if boxes.empty:
        return
    x1, y1, x2, y2 = ov._crop(boxes, img_shape, 28.0)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    radius = max(x2 - x1, y2 - y1) / 2.0 + 8.0
    img_h, img_w = img_shape[:2]
    tx = cx + radius + 35.0
    ty = cy - radius - 18.0
    if cx > img_w * 0.55:
        tx = cx - radius - 35.0
    if cy < img_h * 0.35:
        ty = cy + radius + 16.0
    tx = min(max(18.0, tx), img_w - 18.0)
    ty = min(max(18.0, ty), img_h - 18.0)
    ha = "left" if tx >= cx else "right"
    ax.add_patch(Circle((cx, cy), radius=radius, fill=False, linewidth=1.2, edgecolor=WARN))
    ax.annotate(
        label,
        xy=(cx + radius * 0.6, cy - radius * 0.2),
        xytext=(tx, ty),
        color=TEXT,
        fontsize=5.8,
        ha=ha,
        arrowprops={"arrowstyle": "->", "color": WARN, "lw": 0.9},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.1},
    )


def compute_count_mae_map(ctx: RepoContext, rows: pd.DataFrame) -> dict[tuple[str, str], float]:
    split_dir = ctx.mft_root / SPLIT
    sequences = read_seqmap(split_dir / "seqmaps" / f"{SPLIT}.txt")
    out: dict[tuple[str, str], float] = {}
    for method_key, condition_key in rows[["method_key", "condition_key"]].drop_duplicates().itertuples(index=False):
        pred_dir = ctx.degradation_root / str(method_key) / str(condition_key) / "pred"
        maes: list[float] = []
        if not pred_dir.exists():
            continue
        for seq in sequences:
            seq_len = read_seq_length(split_dir / seq / "seqinfo.ini")
            used_frames = min(int(seq_len), 1000)
            gt_counts = load_frame_counts(split_dir / seq / "gt" / "gt.txt", used_frames)
            pred_counts = load_frame_counts(pred_dir / f"{seq}.txt", used_frames)
            maes.append(float(compute_count_metrics(gt_counts, pred_counts)["CountMAE"]))
        out[(str(method_key), str(condition_key))] = float(np.mean(maes)) if maes else float("nan")
    return out


def prepare_degradation_table(ctx: RepoContext) -> pd.DataFrame:
    df = pd.read_csv(ctx.degradation_csv)
    if "CountMAE" in df.columns:
        return df
    mae_map = compute_count_mae_map(ctx, df)
    df["CountMAE"] = [
        mae_map.get((str(row.method_key), str(row.condition_key)), float("nan"))
        for row in df.itertuples(index=False)
    ]
    return df


def series_by_severity(df: pd.DataFrame, method: str, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    severities = np.array([0.0, 0.15, 0.30, 0.45], dtype=float)
    mean_vals = np.zeros_like(severities)
    low_vals = np.zeros_like(severities)
    high_vals = np.zeros_like(severities)
    clean = df[(df["method"] == method) & (df["degradation_key"] == "clean")]
    base_val = float(clean.iloc[0][metric])
    mean_vals[0] = low_vals[0] = high_vals[0] = base_val
    for i, severity in enumerate(severities[1:], start=1):
        cur = df[(df["method"] == method) & (np.isclose(df["level"].astype(float), severity)) & (df["degradation_key"] != "clean")]
        vals = cur[metric].astype(float).to_numpy()
        mean_vals[i] = float(np.mean(vals))
        low_vals[i] = float(np.min(vals))
        high_vals[i] = float(np.max(vals))
    return severities, mean_vals, np.vstack([low_vals, high_vals])


def render_framework_architecture(ctx: RepoContext) -> None:
    image = ov._mft_frame(ctx.mft_full_root, "MSK-002", 1)
    fig = plt.figure(figsize=(7.25, 3.45))
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    card(ax, 0.02, 0.42, 0.14, 0.44, "Video input", ["pond surveillance", "frame decode", "frozen split"], BLUE)
    inset = ax.inset_axes([0.045, 0.50, 0.095, 0.20])
    inset.imshow(image)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_linewidth(0.7)
        spine.set_edgecolor(EDGE)

    card(ax, 0.19, 0.42, 0.14, 0.44, "Detector output", ["fish boxes", "confidence scores", "time-aligned frames"], GRAY)
    card(ax, 0.36, 0.42, 0.15, 0.44, "Tracking core", ["Kalman prediction", "Hungarian assignment", "state update"], BLUE)

    ax.add_patch(Rectangle((0.54, 0.39), 0.17, 0.49, facecolor="none", edgecolor=EDGE, linewidth=0.9, transform=ax.transAxes))
    ax.text(0.557, 0.844, "Optional modules", transform=ax.transAxes, ha="left", va="top", fontsize=6.3, weight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.7})
    card(ax, 0.555, 0.68, 0.14, 0.13, "Gating", ["Mahalanobis check"], BLUE_MID)
    card(ax, 0.555, 0.52, 0.14, 0.13, "Trajectory", ["motion cue"], GRAY)
    card(ax, 0.555, 0.36, 0.14, 0.13, "Adaptive weighting", ["stress-aware score"], BLUE_MID)

    card(ax, 0.73, 0.42, 0.15, 0.44, "Diagnostics", ["HOTA / IDF1 / IDSW", "count stability", "degradation robustness", "runtime profile"], BLUE)
    card(ax, 0.90, 0.42, 0.10, 0.44, "Decision", ["Base profile", "or", "+gating fallback"], GRAY)
    card(ax, 0.02, 0.08, 0.97, 0.20, "Audit-ready protocol", ["run_config.json, manifest_hash, and seed_locking keep the comparison reproducible, reviewable, and ready for deployment-oriented reuse."], accent=GRAY)

    for x1, x2 in [(0.16, 0.19), (0.33, 0.36), (0.51, 0.54), (0.71, 0.73), (0.88, 0.90)]:
        arrow(ax, (x1, 0.64), (x2, 0.64))
    arrow(ax, (0.50, 0.28), (0.50, 0.40), color=GRAY_DARK, ls="--")
    arrow(ax, (0.81, 0.28), (0.81, 0.40), color=GRAY_DARK, ls="--")

    ax.text(0.02, 0.98, "Deployment-oriented fish-MOT framework", transform=ax.transAxes, ha="left", va="top", fontsize=7.8, weight="bold")
    ax.text(0.02, 0.92, "Auditability and diagnostics are treated as first-class outputs rather than afterthoughts.", transform=ax.transAxes, ha="left", va="top", fontsize=6.2)
    save(fig, ctx.figure_dir / "framework_architecture_publication")


def render_gating_logic(ctx: RepoContext) -> None:
    fig = plt.figure(figsize=(7.05, 3.05))
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    card(ax, 0.02, 0.56, 0.17, 0.25, "Predicted track state", [r"$x_i^-$ from Kalman predict", r"$S_i$ innovation covariance"], BLUE)
    card(ax, 0.02, 0.19, 0.17, 0.25, "Candidate detections", [r"$z_j$ from detector output", "current frame proposals"], GRAY)
    card(ax, 0.24, 0.37, 0.23, 0.33, "Mahalanobis gating score", [r"$d_{ij} = (z_j-Hx_i^-)^T S_i^{-1} (z_j-Hx_i^-)$", "distance in measurement space"], BLUE)
    card(ax, 0.53, 0.37, 0.14, 0.33, "Threshold test", [r"if $d_{ij} \leq \tau_g$", "candidate survives"], GRAY)
    card(ax, 0.73, 0.57, 0.13, 0.20, "Reject", ["discard pair", "count reject"], BLUE_MID)
    card(ax, 0.73, 0.30, 0.13, 0.20, "Keep", ["pass to assignment", "update state"], BLUE)
    card(ax, 0.89, 0.33, 0.10, 0.38, "Diagnostics", ["trigger rate", "score tail", "window log"], GRAY)

    arrow(ax, (0.19, 0.68), (0.24, 0.53))
    arrow(ax, (0.19, 0.31), (0.24, 0.53))
    arrow(ax, (0.47, 0.53), (0.53, 0.53))
    arrow(ax, (0.67, 0.53), (0.73, 0.67))
    arrow(ax, (0.67, 0.47), (0.73, 0.40))
    arrow(ax, (0.86, 0.67), (0.89, 0.57))
    arrow(ax, (0.86, 0.40), (0.89, 0.48))
    ax.text(0.695, 0.72, "No", transform=ax.transAxes, fontsize=6.0)
    ax.text(0.695, 0.46, "Yes", transform=ax.transAxes, fontsize=6.0)
    ax.text(0.02, 0.97, "Association gating and drift logging", transform=ax.transAxes, ha="left", va="top", fontsize=7.8, weight="bold")
    ax.text(0.02, 0.91, "The gate is part of the tracker, but its statistics are retained for later deployment diagnostics.", transform=ax.transAxes, ha="left", va="top", fontsize=6.2)
    save(fig, ctx.figure_dir / "gating_logic_flow_publication")


def render_degradation_combined(ctx: RepoContext) -> None:
    seq, frame, img_path = pick_example(ctx.mft_full_root)
    clean = Image.open(img_path).convert("RGB")
    df = prepare_degradation_table(ctx)

    panels = [
        ("Clean", clean),
        ("Motion blur", motion_blur_image(clean, 0.45)),
        ("Low illumination", dark_image(clean, 0.45)),
        ("Haze / turbidity", haze_image(clean, 0.45)),
    ]
    metrics = ["HOTA", "IDF1", "CountMAE"]
    titles = ["HOTA", "IDF1", "CountMAE"]

    fig = plt.figure(figsize=(7.25, 4.30))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.90, bottom=0.09)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.22, 0.98], wspace=0.16)
    grid_a = outer[0, 0].subgridspec(2, 2, wspace=0.05, hspace=0.05)
    grid_b = outer[0, 1].subgridspec(3, 1, hspace=0.20)

    for idx, (title, image) in enumerate(panels):
        ax = fig.add_subplot(grid_a[idx // 2, idx % 2])
        ax.imshow(np.asarray(image))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(EDGE)
            spine.set_linewidth(0.8)
        ax.text(0.02, 0.98, title, transform=ax.transAxes, ha="left", va="top", fontsize=6.2, weight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.2})
        if idx == 0:
            ax.text(0.02, 0.02, f"{seq} frame {frame:06d}", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.0})

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = fig.add_subplot(grid_b[idx, 0])
        for method in ["Base", "+gating", "ByteTrack"]:
            xs, mean_vals, bounds = series_by_severity(df, method, metric)
            ax.plot(xs, mean_vals, color=COLORS[method], marker=MARKERS[method], linewidth=1.2, markersize=3.4, label=method)
            ax.fill_between(xs, bounds[0], bounds[1], color=COLORS[method], alpha=0.10, linewidth=0)
        ax.set_ylabel(title)
        ax.set_xticks([0.0, 0.15, 0.30, 0.45])
        ax.grid(axis="y", linestyle="--", alpha=0.24)
        if idx == 0:
            ax.legend(loc="lower left", frameon=False, ncol=3, fontsize=5.8)
            ax.text(0.99, 0.04, "band = range across blur / dark / haze", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5)
        if idx < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Severity level")

    fig.text(0.02, 0.985, "(A) Qualitative environmental stress proxies", ha="left", va="top", fontsize=7.6, weight="bold")
    fig.text(0.53, 0.985, "(B) Robustness curves aggregated across degradation families", ha="left", va="top", fontsize=7.6, weight="bold")
    save(fig, ctx.figure_dir / "degradation_combined_publication")


def render_failure_cases(ctx: RepoContext) -> None:
    stable_seq, stable_frame = find_stable_example(ctx)
    occ_seq, occ_frame = find_occlusion_case(ctx)
    turn_seq, turn_frame = "BT-001", 117

    stable_img = ov._mft_frame(ctx.mft_full_root, stable_seq, stable_frame)
    stable_pred_df = read_mot_table(ctx.base_pred_root / f"{stable_seq}.txt")
    stable_pred = frame_boxes(stable_pred_df, stable_frame)
    stable_ids = choose_tracks(stable_pred, 2)

    turn_img = ov._mft_frame(ctx.mft_full_root, turn_seq, turn_frame)
    turn_pred_df = read_mot_table(ctx.base_pred_root / f"{turn_seq}.txt")
    turn_pred = frame_boxes(turn_pred_df, turn_frame)
    turn_pair, turn_ids = find_turning_pair(turn_pred)

    occ_img = ov._mft_frame(ctx.mft_full_root, occ_seq, occ_frame)
    occ_gt = frame_boxes(read_mot_table(ctx.mft_full_root / occ_seq / "gt" / "gt.txt"), occ_frame)
    occ_pred = frame_boxes(read_mot_table(ctx.base_pred_root / f"{occ_seq}.txt"), occ_frame)
    matched_gt, matched_pred = greedy_match(occ_gt, occ_pred, 0.35)
    occ_gt_unmatched = occ_gt.drop(occ_gt.index[matched_gt]) if matched_gt else occ_gt.head(1)
    occ_pred_matched = occ_pred.iloc[matched_pred] if matched_pred else occ_pred.head(min(3, len(occ_pred)))

    low_img = ov._brackish_frame(ctx.brackish_root, "train", "brackishMOT-18__000096")
    low_boxes = ov._brackish_boxes(ctx.brackish_root, "train", "brackishMOT-18__000096")
    low_focus = find_smallest_brackish_box(low_boxes)

    fig = plt.figure(figsize=(7.00, 4.95))
    gs = fig.add_gridspec(2, 2, wspace=0.08, hspace=0.10)
    axs = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])], [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]])

    ax = axs[0, 0]
    ax.imshow(stable_img)
    ov._boxes(ax, stable_pred, BLUE_DARK, 0.75, 0.88, stable_ids)
    for tid in stable_ids:
        tail = track_tail(stable_pred_df, tid, stable_frame, 14)
        if len(tail) > 1:
            ax.plot(tail[:, 0], tail[:, 1], color=ACCENT, linewidth=1.0, marker="o", markersize=1.8)
    ov._panel(ax, "a", "Stable tracking", f"{stable_seq} / Base")
    ax.text(0.98, 0.02, "smooth identities under nominal motion", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.1})

    ax = axs[0, 1]
    ax.imshow(turn_img)
    ov._boxes(ax, turn_pred, BLUE_DARK, 0.70, 0.74, turn_ids)
    for tid in turn_ids:
        tail = track_tail(turn_pred_df, tid, turn_frame, 14)
        if len(tail) > 1:
            ax.plot(tail[:, 0], tail[:, 1], color=ACCENT, linewidth=1.0, alpha=0.95)
    add_circle_callout(ax, turn_pair, turn_img.shape, "turning / crossing zone")
    ov._panel(ax, "b", "Turning / crossing ID switch risk", f"{turn_seq} / Base")

    ax = axs[1, 0]
    ax.imshow(occ_img)
    ov._boxes(ax, occ_gt, GRAY_DARK, 0.55, 0.40)
    if not occ_pred_matched.empty:
        ov._boxes(ax, occ_pred_matched, BLUE_DARK, 0.78, 0.92, choose_tracks(occ_pred_matched, 2))
    if not occ_gt_unmatched.empty:
        ov._boxes(ax, occ_gt_unmatched, WARN, 1.15, 1.0)
        add_circle_callout(ax, occ_gt_unmatched, occ_img.shape, "missed after overlap")
        ins = inset_axes(ax, width="35%", height="35%", loc="lower right", borderpad=0.9)
        ins.imshow(occ_img)
        ov._boxes(ins, occ_gt_unmatched, WARN, 1.15, 1.0)
        if not occ_pred_matched.empty:
            ov._boxes(ins, occ_pred_matched, BLUE_DARK, 1.00, 0.92)
        crop_boxes = pd.concat([occ_gt_unmatched, occ_pred_matched], ignore_index=True) if not occ_pred_matched.empty else occ_gt_unmatched
        x1, y1, x2, y2 = ov._crop(crop_boxes, occ_img.shape, 40.0)
        ins.set_xlim(x1, x2)
        ins.set_ylim(y2, y1)
        ins.set_xticks([])
        ins.set_yticks([])
        for spine in ins.spines.values():
            spine.set_edgecolor(EDGE)
            spine.set_linewidth(0.8)
        mark_inset(ax, ins, loc1=2, loc2=4, fc="none", ec=EDGE, lw=0.6)
    ov._panel(ax, "c", "Occlusion-induced track loss", f"{occ_seq} / Base")

    ax = axs[1, 1]
    ax.imshow(low_img)
    if not low_boxes.empty:
        ov._boxes(ax, low_boxes, GOLD, 0.85, 0.80)
    if not low_focus.empty:
        ov._boxes(ax, low_focus, WARN, 1.20, 1.0)
        add_circle_callout(ax, low_focus, low_img.shape, "weak target / likely miss")
        ins = inset_axes(ax, width="35%", height="35%", loc="lower right", borderpad=0.9)
        ins.imshow(low_img)
        ov._boxes(ins, low_focus, WARN, 1.20, 1.0)
        x1, y1, x2, y2 = ov._crop(low_focus, low_img.shape, 55.0)
        ins.set_xlim(x1, x2)
        ins.set_ylim(y2, y1)
        ins.set_xticks([])
        ins.set_yticks([])
        for spine in ins.spines.values():
            spine.set_edgecolor(EDGE)
            spine.set_linewidth(0.8)
        mark_inset(ax, ins, loc1=2, loc2=4, fc="none", ec=EDGE, lw=0.6)
    ov._panel(ax, "d", "Low-confidence miss", "BrackishMOT-18")
    ax.text(0.98, 0.02, "visibility degrades before counting fails", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.1})
    save(fig, ctx.figure_dir / "failure_cases_2x2_publication")


def build_context(args: argparse.Namespace) -> RepoContext:
    root = repo_root()
    return RepoContext(
        root=root,
        figure_dir=args.figure_dir if args.figure_dir is not None else root / "paper" / "cea_draft" / "figures",
        source_dir=args.source_dir if args.source_dir is not None else root / "paper" / "cea_draft" / "figure_sources",
        caption_dir=args.caption_dir if args.caption_dir is not None else root / "paper" / "cea_draft" / "figure_captions",
        mft_full_root=root / "data" / "mft25_mot_full" / SPLIT,
        mft_root=root / "data" / "mft25_mot",
        brackish_root=root / "data" / "brackish_yolo",
        degradation_csv=root / "results" / "main_val" / "tables" / "degradation_extended.csv",
        degradation_root=root / "results" / "main_val" / "degradation_extended",
        base_pred_root=root / "results" / "main_val" / "seed_runs" / "seed_0" / "pred_base",
    )


def write_sources_and_captions(ctx: RepoContext) -> None:
    write_text(ctx.source_dir / "framework_architecture.mmd", FRAMEWORK_MERMAID)
    write_text(ctx.source_dir / "gating_logic_flow.mmd", GATING_MERMAID)
    for name, text in CAPTIONS.items():
        write_text(ctx.caption_dir / f"{name}.txt", text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render publication-ready support figures for the fish-mot paper.")
    parser.add_argument("--figure", choices=["all", "framework", "gating", "degradation", "failure"], default="all")
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--caption-dir", type=Path, default=None)
    parser.add_argument("--skip-sources", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = build_context(args)

    if not args.skip_sources:
        write_sources_and_captions(ctx)

    if args.figure in {"all", "framework"}:
        render_framework_architecture(ctx)
    if args.figure in {"all", "gating"}:
        render_gating_logic(ctx)
    if args.figure in {"all", "degradation"}:
        render_degradation_combined(ctx)
    if args.figure in {"all", "failure"}:
        render_failure_cases(ctx)


if __name__ == "__main__":
    main()
