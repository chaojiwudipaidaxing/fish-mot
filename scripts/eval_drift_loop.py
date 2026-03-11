#!/usr/bin/env python
"""Evaluate turbidity-induced drift and fallback policies under the audit-ready fish MOT reliability framework.

Definitions implemented exactly as manuscript:
  - D_in(t): KS sup distance between current-window CDF and reference CDF.
  - D_out(t): L1 normalized deviation for q_t=[IDSW_t, CountMAE_t, F1_t].
  - tau_in/tau_out: p95 or p99 quantiles from reference-window D_in/D_out.
  - Persistence K in {1,2,3}.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

Obj = Tuple[int, np.ndarray, float]


FIXED_SPLIT = "val_half"
FIXED_MAX_FRAMES = 1000
FIXED_WINDOW = 100
FIXED_CALIB_FRAMES = 400
FIXED_SEED = 0
IOU_THRESH = 0.5
EPS = 1e-6
K_GRID = [1, 2, 3]
QUANTILES = [95, 99]
SEQUENCES = ["BT-001", "BT-003", "BT-005", "MSK-002", "PF-001", "SN-001", "SN-013", "SN-015"]


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
class ScenarioSpec:
    name: str
    kind: str  # clean / controlled / environmental
    is_drift: bool
    drop_rate: float
    jitter: float
    pred_base_dir: Path
    pred_gating_dir: Path | None = None


SCENARIOS: List[ScenarioSpec] = [
    ScenarioSpec(
        name="clean_control",
        kind="clean",
        is_drift=False,
        drop_rate=0.0,
        jitter=0.0,
        pred_base_dir=Path("results/main_val/seed_runs/seed_0/pred_base"),
        pred_gating_dir=Path("results/main_val/seed_runs/seed_0/pred_gating"),
    ),
    ScenarioSpec(
        name="drop_jitter_high",
        kind="controlled",
        is_drift=True,
        drop_rate=0.40,
        jitter=0.02,
        pred_base_dir=Path("results/main_val/degradation_grid/base/d0p40_j0p02/pred"),
        pred_gating_dir=Path("results/main_val/degradation_grid/gating/d0p40_j0p02/pred"),
    ),
    ScenarioSpec(
        name="low_light_high",
        kind="environmental",
        is_drift=True,
        drop_rate=0.28,
        jitter=0.036,
        pred_base_dir=Path("results/stress_tests/low_light/high/base/pred"),
        pred_gating_dir=Path("results/stress_tests/low_light/high/gating/pred"),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run drift-aware closed-loop evaluation.")
    parser.add_argument("--mot-root", type=Path, default=Path("data/mft25_mot_full"))
    parser.add_argument("--split", type=str, default=FIXED_SPLIT)
    parser.add_argument("--seqs", nargs="+", default=SEQUENCES)
    parser.add_argument("--max-frames", type=int, default=FIXED_MAX_FRAMES)
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    parser.add_argument(
        "--input-feature",
        choices=["simulate", "retention"],
        default="simulate",
        help="simulate: synthetic retention from drop_rate; retention: pred_count/gt_count ratio from base predictions.",
    )
    parser.add_argument(
        "--scenario-config",
        type=Path,
        default=None,
        help=(
            "Optional JSON scenario config. "
            "Schema: {'scenarios':[{'name','kind','is_drift','drop_rate','jitter',"
            "'pred_base_dir','pred_gating_dir','seqs':[...]}]}."
        ),
    )
    parser.add_argument("--summary-out", type=Path, default=Path("results/drift_eval_summary.csv"))
    parser.add_argument("--timeline-out", type=Path, default=Path("results/drift_eval_timeline.csv"))
    parser.add_argument("--opscan-out", type=Path, default=Path("results/drift_opscan.csv"))
    parser.add_argument("--selected-out", type=Path, default=Path("results/drift_selected.json"))
    parser.add_argument("--mitigation-out", type=Path, default=Path("results/mitigation_compare.csv"))
    parser.add_argument("--fig-indicator", type=Path, default=Path("results/fig_drift_indicators_over_time.pdf"))
    parser.add_argument("--fig-ablation", type=Path, default=Path("results/fig_drift_ablation.pdf"))
    parser.add_argument("--fig-mitigation", type=Path, default=Path("results/fig_mitigation_onoff.pdf"))
    parser.add_argument("--paper-fig-dir", type=Path, default=Path("paper/cea_draft/figs"))
    parser.add_argument(
        "--din-aggregation",
        choices=["max", "mean"],
        default="max",
        help="How to aggregate per-feature KS distances into D_in.",
    )
    parser.add_argument("--window", type=int, default=FIXED_WINDOW)
    parser.add_argument("--calib-frames", type=int, default=FIXED_CALIB_FRAMES)
    return parser.parse_args()


def load_scenarios(
    scenario_config: Path | None,
    default_seqs: List[str],
) -> List[Tuple[ScenarioSpec, List[str]]]:
    if scenario_config is None:
        return [(sc, list(default_seqs)) for sc in SCENARIOS]
    if not scenario_config.exists():
        raise FileNotFoundError(f"Scenario config not found: {scenario_config}")
    data = json.loads(scenario_config.read_text(encoding="utf-8"))
    raw = data["scenarios"] if isinstance(data, dict) and "scenarios" in data else data
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Invalid scenario config (expect non-empty list): {scenario_config}")
    out: List[Tuple[ScenarioSpec, List[str]]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid scenario item (must be dict): {item}")
        seqs = [str(s) for s in item.get("seqs", [])]
        if not seqs:
            raise ValueError(f"Scenario {item.get('name', '<unknown>')} missing seqs")
        pred_base = Path(str(item["pred_base_dir"]))
        pred_gating_raw = item.get("pred_gating_dir")
        pred_gating = Path(str(pred_gating_raw)) if pred_gating_raw else None
        sc = ScenarioSpec(
            name=str(item["name"]),
            kind=str(item.get("kind", "unknown")),
            is_drift=bool(item.get("is_drift", True)),
            drop_rate=float(item.get("drop_rate", 0.0)),
            jitter=float(item.get("jitter", 0.0)),
            pred_base_dir=pred_base,
            pred_gating_dir=pred_gating,
        )
        out.append((sc, seqs))
    return out


def read_seq_length(seqinfo_path: Path) -> int:
    import configparser

    parser = configparser.ConfigParser()
    parser.read(seqinfo_path, encoding="utf-8")
    if "Sequence" not in parser or "seqLength" not in parser["Sequence"]:
        raise RuntimeError(f"Cannot parse seqLength from {seqinfo_path}")
    return int(parser["Sequence"]["seqLength"])


def load_mot_by_frame(path: Path, max_frames: int) -> Dict[int, List[Obj]]:
    if not path.exists():
        raise FileNotFoundError(f"MOT file not found: {path}")
    out: Dict[int, List[Obj]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            parts = row.split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            if frame < 1 or frame > max_frames:
                continue
            obj_id = int(float(parts[1]))
            x, y, w, h = [float(v) for v in parts[2:6]]
            if w <= 0 or h <= 0:
                continue
            score = 1.0
            if len(parts) >= 7:
                try:
                    score = float(parts[6])
                except ValueError:
                    score = 1.0
            out.setdefault(frame, []).append((obj_id, np.array([x, y, w, h], dtype=float), score))
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)
    a_x1 = a[:, 0:1]
    a_y1 = a[:, 1:2]
    a_x2 = a_x1 + a[:, 2:3]
    a_y2 = a_y1 + a[:, 3:4]
    b_x1 = b[:, 0][None, :]
    b_y1 = b[:, 1][None, :]
    b_x2 = b_x1 + b[:, 2][None, :]
    b_y2 = b_y1 + b[:, 3][None, :]
    inter_x1 = np.maximum(a_x1, b_x1)
    inter_y1 = np.maximum(a_y1, b_y1)
    inter_x2 = np.minimum(a_x2, b_x2)
    inter_y2 = np.minimum(a_y2, b_y2)
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = np.maximum(0.0, a[:, 2:3]) * np.maximum(0.0, a[:, 3:4])
    area_b = np.maximum(0.0, b[:, 2][None, :]) * np.maximum(0.0, b[:, 3][None, :])
    union = area_a + area_b - inter
    return np.where(union > 0.0, inter / union, 0.0)


def match_iou(
    gt_objs: List[Obj],
    pred_objs: List[Obj],
    iou_thresh: float,
) -> List[Tuple[int, int, int, int]]:
    if not gt_objs or not pred_objs:
        return []
    gt_boxes = np.vstack([x[1] for x in gt_objs])
    pred_boxes = np.vstack([x[1] for x in pred_objs])
    iou = iou_matrix(gt_boxes, pred_boxes)
    row_ind, col_ind = linear_sum_assignment(1.0 - iou)
    matches: List[Tuple[int, int, int, int]] = []
    for gi, pi in zip(row_ind.tolist(), col_ind.tolist()):
        if iou[gi, pi] < iou_thresh:
            continue
        gt_id = int(gt_objs[gi][0])
        pred_id = int(pred_objs[pi][0])
        matches.append((gi, pi, gt_id, pred_id))
    return matches


def compute_frame_signals(
    gt_by_frame: Mapping[int, List[Obj]],
    pred_by_frame: Mapping[int, List[Obj]],
    max_frames: int,
) -> Dict[str, np.ndarray]:
    tp = np.zeros((max_frames,), dtype=np.float64)
    fp = np.zeros((max_frames,), dtype=np.float64)
    fn = np.zeros((max_frames,), dtype=np.float64)
    idsw_inc = np.zeros((max_frames,), dtype=np.float64)
    gt_count = np.zeros((max_frames,), dtype=np.float64)
    pred_count = np.zeros((max_frames,), dtype=np.float64)
    det_conf_mean = np.zeros((max_frames,), dtype=np.float64)
    det_conf_p10 = np.zeros((max_frames,), dtype=np.float64)
    det_conf_p50 = np.zeros((max_frames,), dtype=np.float64)
    det_conf_p90 = np.zeros((max_frames,), dtype=np.float64)
    track_birth_rate = np.zeros((max_frames,), dtype=np.float64)
    track_death_rate = np.zeros((max_frames,), dtype=np.float64)
    mean_track_length = np.zeros((max_frames,), dtype=np.float64)
    assoc_unmatched_rate = np.zeros((max_frames,), dtype=np.float64)
    last_match: Dict[int, int] = {}
    prev_ids: set[int] = set()
    first_seen: Dict[int, int] = {}

    for frame in range(1, max_frames + 1):
        gt_objs = list(gt_by_frame.get(frame, []))
        pred_objs = list(pred_by_frame.get(frame, []))
        pred_scores = np.asarray([float(x[2]) for x in pred_objs], dtype=np.float64)
        if pred_scores.size > 0:
            det_conf_mean[frame - 1] = float(np.mean(pred_scores))
            det_conf_p10[frame - 1] = float(np.percentile(pred_scores, 10))
            det_conf_p50[frame - 1] = float(np.percentile(pred_scores, 50))
            det_conf_p90[frame - 1] = float(np.percentile(pred_scores, 90))
        else:
            det_conf_mean[frame - 1] = 0.0
            det_conf_p10[frame - 1] = 0.0
            det_conf_p50[frame - 1] = 0.0
            det_conf_p90[frame - 1] = 0.0
        gt_count[frame - 1] = float(len(gt_objs))
        pred_count[frame - 1] = float(len(pred_objs))
        matches = match_iou(gt_objs, pred_objs, iou_thresh=IOU_THRESH)
        tp[frame - 1] = float(len(matches))
        fp[frame - 1] = float(max(0, len(pred_objs) - len(matches)))
        fn[frame - 1] = float(max(0, len(gt_objs) - len(matches)))
        denom = float(len(pred_objs) + len(gt_objs))
        assoc_unmatched_rate[frame - 1] = float((fp[frame - 1] + fn[frame - 1]) / max(1.0, denom))

        switches = 0
        for _, _, gt_id, pred_id in matches:
            prev = last_match.get(gt_id)
            if prev is not None and prev != pred_id:
                switches += 1
            last_match[gt_id] = pred_id
        idsw_inc[frame - 1] = float(switches)

        cur_ids = {int(x[0]) for x in pred_objs}
        births = [tid for tid in cur_ids if tid not in prev_ids]
        deaths = [tid for tid in prev_ids if tid not in cur_ids]
        track_birth_rate[frame - 1] = float(len(births))
        track_death_rate[frame - 1] = float(len(deaths))
        for tid in births:
            first_seen[tid] = frame
        ages = [frame - int(first_seen.get(tid, frame)) + 1 for tid in cur_ids]
        mean_track_length[frame - 1] = float(np.mean(ages)) if ages else 0.0
        prev_ids = cur_ids

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "idsw_inc": idsw_inc,
        "gt_count": gt_count,
        "pred_count": pred_count,
        "det_conf_mean": det_conf_mean,
        "det_conf_p10": det_conf_p10,
        "det_conf_p50": det_conf_p50,
        "det_conf_p90": det_conf_p90,
        "track_birth_rate": track_birth_rate,
        "track_death_rate": track_death_rate,
        "mean_track_length": mean_track_length,
        "assoc_unmatched_rate": assoc_unmatched_rate,
    }


def simulate_input_retention(gt_count: np.ndarray, drop_rate: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    out = np.ones_like(gt_count, dtype=np.float64)
    p_keep = max(0.0, min(1.0, 1.0 - float(drop_rate)))
    for i, n in enumerate(gt_count.astype(int).tolist()):
        if n <= 0:
            out[i] = 1.0
        elif p_keep >= 1.0:
            out[i] = 1.0
        else:
            keep = int(rng.binomial(n=n, p=p_keep))
            out[i] = float(keep / max(1, n))
    return out


def ks_sup_distance(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    if sample_a.size == 0 or sample_b.size == 0:
        return 0.0
    a = np.sort(sample_a.astype(np.float64))
    b = np.sort(sample_b.astype(np.float64))
    grid = np.unique(np.concatenate([a, b]))
    if grid.size == 0:
        return 0.0
    cdf_a = np.searchsorted(a, grid, side="right") / float(a.size)
    cdf_b = np.searchsorted(b, grid, side="right") / float(b.size)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def build_windows(
    seq: str,
    scenario: ScenarioSpec,
    frame_signals: Mapping[str, np.ndarray],
    z_input: np.ndarray,
    window: int,
    calib_frames: int,
) -> List[Dict[str, object]]:
    tp = frame_signals["tp"]
    fp = frame_signals["fp"]
    fn = frame_signals["fn"]
    idsw_inc = frame_signals["idsw_inc"]
    gt_count = frame_signals["gt_count"]
    pred_count = frame_signals["pred_count"]
    det_conf_mean = frame_signals["det_conf_mean"]
    det_conf_p10 = frame_signals["det_conf_p10"]
    det_conf_p50 = frame_signals["det_conf_p50"]
    det_conf_p90 = frame_signals["det_conf_p90"]
    track_birth_rate = frame_signals["track_birth_rate"]
    track_death_rate = frame_signals["track_death_rate"]
    mean_track_length = frame_signals["mean_track_length"]
    assoc_unmatched_rate = frame_signals["assoc_unmatched_rate"]
    max_frames = int(tp.shape[0])
    rows: List[Dict[str, object]] = []

    if max_frames < window:
        return rows
    starts = list(range(1, max_frames - window + 2, window))
    eval_idx = 0
    for w_idx, start in enumerate(starts, 1):
        end = start + window - 1
        sl = slice(start - 1, end)
        tp_sum = float(np.sum(tp[sl]))
        fp_sum = float(np.sum(fp[sl]))
        fn_sum = float(np.sum(fn[sl]))
        denom = 2.0 * tp_sum + fp_sum + fn_sum
        f1 = float((2.0 * tp_sum / denom) if denom > 0.0 else 1.0)
        idsw = float(np.sum(idsw_inc[sl]))
        count_mae = float(np.mean(np.abs(pred_count[sl] - gt_count[sl])))
        z_window = z_input[sl].astype(np.float64)
        feature_windows: Dict[str, np.ndarray] = {
            "input_retention": z_window,
            "det_conf_mean": det_conf_mean[sl].astype(np.float64),
            "det_conf_p10": det_conf_p10[sl].astype(np.float64),
            "det_conf_p50": det_conf_p50[sl].astype(np.float64),
            "det_conf_p90": det_conf_p90[sl].astype(np.float64),
            "det_count_per_frame": pred_count[sl].astype(np.float64),
            "track_birth_rate": track_birth_rate[sl].astype(np.float64),
            "track_death_rate": track_death_rate[sl].astype(np.float64),
            "mean_track_length": mean_track_length[sl].astype(np.float64),
            "assoc_unmatched_rate": assoc_unmatched_rate[sl].astype(np.float64),
        }
        is_calib = bool(end <= calib_frames)
        is_eval = bool(end > calib_frames)
        if is_eval:
            eval_idx += 1
        rows.append(
            {
                "scenario": scenario.name,
                "scenario_kind": scenario.kind,
                "is_drift": int(scenario.is_drift),
                "seq": seq,
                "window_idx": int(w_idx),
                "eval_idx": int(eval_idx) if is_eval else 0,
                "start_frame": int(start),
                "end_frame": int(end),
                "is_calib": int(is_calib),
                "is_eval": int(is_eval),
                "IDSW": idsw,
                "CountMAE": count_mae,
                "F1": f1,
                "z_mean": float(np.mean(z_window)),
                "z_window": z_window,
                "feature_windows": feature_windows,
            }
        )
    return rows


def attach_indicators(
    rows: List[Dict[str, object]],
    feature_refs: Mapping[str, np.ndarray],
    q_mu: np.ndarray,
    q_sigma: np.ndarray,
    d_in_aggregation: str,
) -> None:
    for row in rows:
        feats = row.get("feature_windows", {})
        if not isinstance(feats, dict):
            feats = {}
        din_parts: Dict[str, float] = {}
        for fname, ref in feature_refs.items():
            if fname not in feats:
                continue
            sample = np.asarray(feats[fname], dtype=np.float64)
            din_parts[fname] = ks_sup_distance(sample, np.asarray(ref, dtype=np.float64))
        if din_parts:
            if d_in_aggregation == "mean":
                d_in = float(np.mean(list(din_parts.values())))
            else:
                d_in = float(np.max(list(din_parts.values())))
        else:
            z_window = np.asarray(row["z_window"], dtype=np.float64)
            d_in = ks_sup_distance(z_window, np.asarray(list(feature_refs.values())[0], dtype=np.float64))
        q = np.array(
            [float(row["IDSW"]), float(row["CountMAE"]), float(row["F1"])],
            dtype=np.float64,
        )
        d_out = float(np.sum(np.abs((q - q_mu) / (q_sigma + EPS))))
        row["D_in"] = d_in
        row["D_out"] = d_out
        row["D_in_parts"] = din_parts


def compute_alerts_for_rows(
    rows: List[Dict[str, object]],
    tau_in: float,
    tau_out: float,
    k_persist: int,
) -> Dict[Tuple[str, int], Dict[str, float]]:
    out: Dict[Tuple[str, int], Dict[str, float]] = {}
    by_seq: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        if int(row["is_eval"]) != 1:
            continue
        by_seq.setdefault(str(row["seq"]), []).append(row)
    for seq, seq_rows in by_seq.items():
        seq_rows.sort(key=lambda r: int(r["window_idx"]))
        c = 0
        for row in seq_rows:
            d_in = float(row["D_in"])
            d_out = float(row["D_out"])
            trigger = int((d_in > tau_in) or (d_out > tau_out))
            if trigger:
                c += 1
            else:
                c = 0
            alert = int(c >= int(k_persist))
            out[(seq, int(row["window_idx"]))] = {
                "trigger": float(trigger),
                "counter": float(c),
                "alert": float(alert),
            }
    return out


def compute_trigger_delays(
    rows: List[Dict[str, object]],
    alert_map: Mapping[Tuple[str, int], Mapping[str, float]],
) -> List[float]:
    delays: List[float] = []
    by_seq: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        if int(row["is_eval"]) != 1:
            continue
        by_seq.setdefault(str(row["seq"]), []).append(row)
    for seq, seq_rows in by_seq.items():
        seq_rows.sort(key=lambda r: int(r["eval_idx"]))
        first_alert_idx: int | None = None
        for row in seq_rows:
            key = (seq, int(row["window_idx"]))
            state = alert_map.get(key, {"alert": 0.0})
            if int(state.get("alert", 0.0)) == 1:
                first_alert_idx = int(row["eval_idx"])
                break
        if first_alert_idx is None:
            delays.append(float(len(seq_rows)))
        else:
            delays.append(float(max(0, first_alert_idx - 1)))
    return delays


def select_recommended_config(ablation_rows: List[Dict[str, float]]) -> Dict[str, float]:
    feasible = [r for r in ablation_rows if float(r["false_alert_rate"]) <= 0.05]
    if feasible:
        ranked = sorted(
            feasible,
            key=lambda r: (
                float(r["missed_drift_rate_avg"]),
                float(r.get("trigger_delay_mean", 1e9)),
                float(r["false_alert_rate"]),
                float(r["K"]),
                -float(r["quantile"]),
            ),
        )
        sel = dict(ranked[0])
        sel["rule_feasible"] = 1.0
        return sel
    ranked = sorted(
        ablation_rows,
        key=lambda r: (
            float(r["false_alert_rate"]),
            float(r["missed_drift_rate_avg"]),
            float(r.get("trigger_delay_mean", 1e9)),
            float(r["K"]),
            -float(r["quantile"]),
        ),
    )
    sel = dict(ranked[0])
    sel["rule_feasible"] = 0.0
    return sel


def render_indicator_figure(
    rows: List[Dict[str, object]],
    tau_in: float,
    tau_out: float,
    out_path: Path,
) -> None:
    plt.rcParams.update(PLOT_STYLE)
    scenarios = ["clean_control", "drop_jitter_high", "low_light_high"]
    colors = {
        "clean_control": "#4C78A8",
        "drop_jitter_high": "#F58518",
        "low_light_high": "#54A24B",
    }
    labels = {
        "clean_control": "clean control",
        "drop_jitter_high": "drop/jitter high",
        "low_light_high": "low-light high",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    for metric, ax, tau, title in [
        ("D_in", axes[0], tau_in, r"$D_{\mathrm{in}}(t)$ (KS sup distance)"),
        ("D_out", axes[1], tau_out, r"$D_{\mathrm{out}}(t)$ (normalized deviation)"),
    ]:
        for scenario in scenarios:
            sub = [
                r
                for r in rows
                if str(r["scenario"]) == scenario and int(r["is_eval"]) == 1 and int(r["eval_idx"]) > 0
            ]
            if not sub:
                continue
            max_eval = max(int(r["eval_idx"]) for r in sub)
            x = np.arange(1, max_eval + 1, dtype=np.int32)
            y = []
            for idx in x:
                vals = [float(r[metric]) for r in sub if int(r["eval_idx"]) == int(idx)]
                y.append(float(np.mean(vals)))
            ax.plot(x, y, marker="o", color=colors[scenario], label=labels[scenario])
        ax.axhline(float(tau), color="black", linestyle="--", linewidth=1.0, label="threshold")
        ax.set_xlabel("evaluation window index")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("indicator value")
    axes[1].legend(frameon=False, loc="upper left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **PLOT_SAVE_KWARGS)
    plt.close(fig)


def render_ablation_figure(ablation_rows: List[Dict[str, float]], out_path: Path) -> None:
    plt.rcParams.update(PLOT_STYLE)
    ordered = sorted(ablation_rows, key=lambda r: (int(r["quantile"]), int(r["K"])))
    labels = [f"q{int(r['quantile'])}-K{int(r['K'])}" for r in ordered]
    far = [float(r["false_alert_rate"]) for r in ordered]
    mdr = [float(r["missed_drift_rate_avg"]) for r in ordered]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.6, 3.8), constrained_layout=True)
    ax.bar(x - width / 2.0, far, width=width, label="false alert rate", color="#4C78A8")
    ax.bar(x + width / 2.0, mdr, width=width, label="missed drift rate (avg)", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("rate")
    ax.set_title("Drift-loop ablation over persistence K and threshold quantile")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **PLOT_SAVE_KWARGS)
    plt.close(fig)


def render_mitigation_figure(mitigation_rows: List[Dict[str, float]], out_path: Path) -> None:
    plt.rcParams.update(PLOT_STYLE)
    scenarios = [str(r["scenario"]) for r in mitigation_rows]
    x = np.arange(len(scenarios))
    width = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), constrained_layout=True)
    idsw_off = [float(r["IDSW_off"]) for r in mitigation_rows]
    idsw_on = [float(r["IDSW_on"]) for r in mitigation_rows]
    mae_off = [float(r["CountMAE_off"]) for r in mitigation_rows]
    mae_on = [float(r["CountMAE_on"]) for r in mitigation_rows]

    axes[0].bar(x - width / 2.0, idsw_off, width=width, label="mitigation off", color="#4C78A8")
    axes[0].bar(x + width / 2.0, idsw_on, width=width, label="mitigation on", color="#54A24B")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(scenarios)
    axes[0].set_ylabel("IDSW (alerted windows)")
    axes[0].set_title("Mitigation impact on IDSW")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - width / 2.0, mae_off, width=width, label="mitigation off", color="#4C78A8")
    axes[1].bar(x + width / 2.0, mae_on, width=width, label="mitigation on", color="#54A24B")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scenarios)
    axes[1].set_ylabel("CountMAE (alerted windows)")
    axes[1].set_title("Mitigation impact on CountMAE")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **PLOT_SAVE_KWARGS)
    plt.close(fig)


def copy_to_paper_figs(src: Path, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, fig_dir / src.name)


def main() -> int:
    args = parse_args()
    split_dir = args.mot_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Prepared split not found: {split_dir}")
    scenarios = load_scenarios(args.scenario_config, list(args.seqs))
    for sc, seqs in scenarios:
        if not sc.pred_base_dir.exists():
            raise FileNotFoundError(f"Missing prediction directory for scenario {sc.name}: {sc.pred_base_dir}")
        if sc.pred_gating_dir is not None and not sc.pred_gating_dir.exists():
            raise FileNotFoundError(f"Missing mitigation prediction directory for scenario {sc.name}: {sc.pred_gating_dir}")
        for seq in seqs:
            if not (sc.pred_base_dir / f"{seq}.txt").exists():
                raise FileNotFoundError(
                    f"Missing base prediction file for scenario={sc.name} seq={seq}: {sc.pred_base_dir / f'{seq}.txt'}"
                )
            if sc.pred_gating_dir is not None and not (sc.pred_gating_dir / f"{seq}.txt").exists():
                raise FileNotFoundError(
                    f"Missing gating prediction file for scenario={sc.name} seq={seq}: {sc.pred_gating_dir / f'{seq}.txt'}"
                )

    gt_cache: Dict[str, Dict[str, object]] = {}
    all_seqs = sorted({seq for _, seqs in scenarios for seq in seqs})
    for seq in all_seqs:
        seq_len = read_seq_length(split_dir / seq / "seqinfo.ini")
        max_frames = min(int(seq_len), int(args.max_frames)) if int(args.max_frames) > 0 else int(seq_len)
        gt_by_frame = load_mot_by_frame(split_dir / seq / "gt" / "gt.txt", max_frames=max_frames)
        gt_counts = np.zeros((max_frames,), dtype=np.float64)
        for frame in range(1, max_frames + 1):
            gt_counts[frame - 1] = float(len(gt_by_frame.get(frame, [])))
        gt_cache[seq] = {
            "max_frames": max_frames,
            "gt_by_frame": gt_by_frame,
            "gt_counts": gt_counts,
        }

    scenario_rows_base: Dict[str, List[Dict[str, object]]] = {}
    scenario_rows_gating: Dict[str, List[Dict[str, object]]] = {}

    for sc, seqs in scenarios:
        rows_base: List[Dict[str, object]] = []
        rows_gating: List[Dict[str, object]] = []
        for seq_idx, seq in enumerate(seqs):
            cache = gt_cache[seq]
            max_frames = int(cache["max_frames"])
            gt_by_frame = cache["gt_by_frame"]  # type: ignore[assignment]
            pred_base = load_mot_by_frame(sc.pred_base_dir / f"{seq}.txt", max_frames=max_frames)
            frame_base = compute_frame_signals(
                gt_by_frame=gt_by_frame,  # type: ignore[arg-type]
                pred_by_frame=pred_base,
                max_frames=max_frames,
            )
            gt_counts = np.asarray(frame_base["gt_count"], dtype=np.float64)
            if args.input_feature == "retention":
                pred_counts = np.asarray(frame_base["pred_count"], dtype=np.float64)
                input_ratio = np.clip(pred_counts / (gt_counts + EPS), 0.0, 2.0)
            else:
                input_ratio = simulate_input_retention(
                    gt_counts,
                    drop_rate=float(sc.drop_rate),
                    seed=(int(args.seed) + seq_idx),
                )
            rows_base.extend(
                build_windows(
                    seq=seq,
                    scenario=sc,
                    frame_signals=frame_base,
                    z_input=input_ratio,
                    window=int(args.window),
                    calib_frames=int(args.calib_frames),
                )
            )

            if sc.pred_gating_dir is not None:
                pred_gating = load_mot_by_frame(sc.pred_gating_dir / f"{seq}.txt", max_frames=max_frames)
                frame_gating = compute_frame_signals(
                    gt_by_frame=gt_by_frame,  # type: ignore[arg-type]
                    pred_by_frame=pred_gating,
                    max_frames=max_frames,
                )
                rows_gating.extend(
                    build_windows(
                        seq=seq,
                        scenario=sc,
                        frame_signals=frame_gating,
                        z_input=input_ratio,
                        window=int(args.window),
                        calib_frames=int(args.calib_frames),
                    )
                )

        scenario_rows_base[sc.name] = rows_base
        if rows_gating:
            scenario_rows_gating[sc.name] = rows_gating

    clean_rows = scenario_rows_base["clean_control"]
    calib_rows = [r for r in clean_rows if int(r["is_calib"]) == 1]
    if not calib_rows:
        raise RuntimeError("No calibration windows available; adjust --window/--calib-frames.")
    z_ref = np.concatenate([np.asarray(r["z_window"], dtype=np.float64) for r in calib_rows], axis=0)
    q_ref = np.array(
        [[float(r["IDSW"]), float(r["CountMAE"]), float(r["F1"])] for r in calib_rows],
        dtype=np.float64,
    )
    q_mu = np.mean(q_ref, axis=0)
    q_sigma = np.std(q_ref, axis=0, ddof=0)

    all_rows_base: List[Dict[str, object]] = []
    for rows in scenario_rows_base.values():
        attach_indicators(rows, z_ref=z_ref, q_mu=q_mu, q_sigma=q_sigma)
        all_rows_base.extend(rows)
    for rows in scenario_rows_gating.values():
        attach_indicators(rows, z_ref=z_ref, q_mu=q_mu, q_sigma=q_sigma)

    calib_din = np.array([float(r["D_in"]) for r in calib_rows], dtype=np.float64)
    calib_dout = np.array([float(r["D_out"]) for r in calib_rows], dtype=np.float64)
    tau_by_quantile: Dict[int, Tuple[float, float]] = {
        q: (float(np.percentile(calib_din, q)), float(np.percentile(calib_dout, q))) for q in QUANTILES
    }

    ablation_rows: List[Dict[str, float]] = []
    summary_rows: List[Dict[str, object]] = []
    for quant in QUANTILES:
        tau_in, tau_out = tau_by_quantile[quant]
        for k_persist in K_GRID:
            clean_alert = compute_alerts_for_rows(
                scenario_rows_base["clean_control"], tau_in=tau_in, tau_out=tau_out, k_persist=k_persist
            )
            drop_alert = compute_alerts_for_rows(
                scenario_rows_base["drop_jitter_high"], tau_in=tau_in, tau_out=tau_out, k_persist=k_persist
            )
            env_alert = compute_alerts_for_rows(
                scenario_rows_base["low_light_high"], tau_in=tau_in, tau_out=tau_out, k_persist=k_persist
            )
            far = float(np.mean([v["alert"] for v in clean_alert.values()])) if clean_alert else math.nan
            mdr_drop = 1.0 - float(np.mean([v["alert"] for v in drop_alert.values()])) if drop_alert else math.nan
            mdr_env = 1.0 - float(np.mean([v["alert"] for v in env_alert.values()])) if env_alert else math.nan
            mdr_avg = float(np.mean([mdr_drop, mdr_env]))
            balance = float(far + mdr_avg)
            row = {
                "quantile": float(quant),
                "K": float(k_persist),
                "tau_in": tau_in,
                "tau_out": tau_out,
                "false_alert_rate": far,
                "missed_drift_rate_drop_jitter": mdr_drop,
                "missed_drift_rate_env": mdr_env,
                "missed_drift_rate_avg": mdr_avg,
                "balance": balance,
            }
            ablation_rows.append(row)
            summary_rows.append(
                {
                    "experiment": "E2",
                    "scenario": "ablation",
                    "quantile": int(quant),
                    "K": int(k_persist),
                    "tau_in": f"{tau_in:.6f}",
                    "tau_out": f"{tau_out:.6f}",
                    "false_alert_rate": f"{far:.6f}",
                    "missed_drift_rate": f"{mdr_avg:.6f}",
                    "missed_drift_rate_drop_jitter": f"{mdr_drop:.6f}",
                    "missed_drift_rate_env": f"{mdr_env:.6f}",
                    "IDSW_off": "",
                    "IDSW_on": "",
                    "CountMAE_off": "",
                    "CountMAE_on": "",
                    "delta_IDSW": "",
                    "delta_CountMAE": "",
                    "recommended": "0",
                    "notes": "K/quantile ablation",
                }
            )

    recommended = select_recommended_config(ablation_rows)
    rec_q = int(recommended["quantile"])
    rec_k = int(recommended["K"])
    rec_tau_in = float(recommended["tau_in"])
    rec_tau_out = float(recommended["tau_out"])

    rec_clean_alert = compute_alerts_for_rows(
        scenario_rows_base["clean_control"], tau_in=rec_tau_in, tau_out=rec_tau_out, k_persist=rec_k
    )
    rec_drop_alert = compute_alerts_for_rows(
        scenario_rows_base["drop_jitter_high"], tau_in=rec_tau_in, tau_out=rec_tau_out, k_persist=rec_k
    )
    rec_env_alert = compute_alerts_for_rows(
        scenario_rows_base["low_light_high"], tau_in=rec_tau_in, tau_out=rec_tau_out, k_persist=rec_k
    )
    e1_far = float(np.mean([v["alert"] for v in rec_clean_alert.values()]))
    e1_mdr_drop = 1.0 - float(np.mean([v["alert"] for v in rec_drop_alert.values()]))
    e1_mdr_env = 1.0 - float(np.mean([v["alert"] for v in rec_env_alert.values()]))

    summary_rows.extend(
        [
            {
                "experiment": "E1",
                "scenario": "clean_control",
                "quantile": rec_q,
                "K": rec_k,
                "tau_in": f"{rec_tau_in:.6f}",
                "tau_out": f"{rec_tau_out:.6f}",
                "false_alert_rate": f"{e1_far:.6f}",
                "missed_drift_rate": "",
                "missed_drift_rate_drop_jitter": "",
                "missed_drift_rate_env": "",
                "IDSW_off": "",
                "IDSW_on": "",
                "CountMAE_off": "",
                "CountMAE_on": "",
                "delta_IDSW": "",
                "delta_CountMAE": "",
                "recommended": "1",
                "notes": "false-alert rate on clean control",
            },
            {
                "experiment": "E1",
                "scenario": "drop_jitter_high",
                "quantile": rec_q,
                "K": rec_k,
                "tau_in": f"{rec_tau_in:.6f}",
                "tau_out": f"{rec_tau_out:.6f}",
                "false_alert_rate": "",
                "missed_drift_rate": f"{e1_mdr_drop:.6f}",
                "missed_drift_rate_drop_jitter": f"{e1_mdr_drop:.6f}",
                "missed_drift_rate_env": "",
                "IDSW_off": "",
                "IDSW_on": "",
                "CountMAE_off": "",
                "CountMAE_on": "",
                "delta_IDSW": "",
                "delta_CountMAE": "",
                "recommended": "1",
                "notes": "missed-drift rate under controlled drop/jitter",
            },
            {
                "experiment": "E1",
                "scenario": "low_light_high",
                "quantile": rec_q,
                "K": rec_k,
                "tau_in": f"{rec_tau_in:.6f}",
                "tau_out": f"{rec_tau_out:.6f}",
                "false_alert_rate": "",
                "missed_drift_rate": f"{e1_mdr_env:.6f}",
                "missed_drift_rate_drop_jitter": "",
                "missed_drift_rate_env": f"{e1_mdr_env:.6f}",
                "IDSW_off": "",
                "IDSW_on": "",
                "CountMAE_off": "",
                "CountMAE_on": "",
                "delta_IDSW": "",
                "delta_CountMAE": "",
                "recommended": "1",
                "notes": "missed-drift rate under environmental degradation",
            },
        ]
    )

    mitigation_rows: List[Dict[str, float]] = []
    for scenario_name in ["drop_jitter_high", "low_light_high"]:
        base_rows = [r for r in scenario_rows_base[scenario_name] if int(r["is_eval"]) == 1]
        gate_rows = [r for r in scenario_rows_gating[scenario_name] if int(r["is_eval"]) == 1]
        alert_map = compute_alerts_for_rows(
            scenario_rows_base[scenario_name],
            tau_in=rec_tau_in,
            tau_out=rec_tau_out,
            k_persist=rec_k,
        )
        base_map = {(str(r["seq"]), int(r["window_idx"])): r for r in base_rows}
        gate_map = {(str(r["seq"]), int(r["window_idx"])): r for r in gate_rows}
        on_keys = [k for k, v in alert_map.items() if int(v["alert"]) == 1 and k in base_map and k in gate_map]
        if not on_keys:
            continue
        idsw_off = float(np.mean([float(base_map[k]["IDSW"]) for k in on_keys]))
        idsw_on = float(np.mean([float(gate_map[k]["IDSW"]) for k in on_keys]))
        mae_off = float(np.mean([float(base_map[k]["CountMAE"]) for k in on_keys]))
        mae_on = float(np.mean([float(gate_map[k]["CountMAE"]) for k in on_keys]))
        mitigation_rows.append(
            {
                "scenario": scenario_name,
                "IDSW_off": idsw_off,
                "IDSW_on": idsw_on,
                "CountMAE_off": mae_off,
                "CountMAE_on": mae_on,
            }
        )
        summary_rows.append(
            {
                "experiment": "E3",
                "scenario": scenario_name,
                "quantile": rec_q,
                "K": rec_k,
                "tau_in": f"{rec_tau_in:.6f}",
                "tau_out": f"{rec_tau_out:.6f}",
                "false_alert_rate": "",
                "missed_drift_rate": "",
                "missed_drift_rate_drop_jitter": "",
                "missed_drift_rate_env": "",
                "IDSW_off": f"{idsw_off:.6f}",
                "IDSW_on": f"{idsw_on:.6f}",
                "CountMAE_off": f"{mae_off:.6f}",
                "CountMAE_on": f"{mae_on:.6f}",
                "delta_IDSW": f"{(idsw_on - idsw_off):.6f}",
                "delta_CountMAE": f"{(mae_on - mae_off):.6f}",
                "recommended": "1",
                "notes": "mitigation-on uses +gating profile after alert",
            }
        )

    for row in summary_rows:
        if row["experiment"] == "E2" and int(row["quantile"]) == rec_q and int(row["K"]) == rec_k:
            row["recommended"] = "1"
            row["notes"] = str(row["notes"]) + "; selected_by_balance"

    timeline_rows: List[Dict[str, object]] = []
    for scenario_name, rows in scenario_rows_base.items():
        alerts = compute_alerts_for_rows(
            scenario_rows_base[scenario_name],
            tau_in=rec_tau_in,
            tau_out=rec_tau_out,
            k_persist=rec_k,
        )
        for row in rows:
            if int(row["is_eval"]) != 1:
                continue
            key = (str(row["seq"]), int(row["window_idx"]))
            state = alerts.get(key, {"trigger": 0.0, "counter": 0.0, "alert": 0.0})
            timeline_rows.append(
                {
                    "scenario": scenario_name,
                    "seq": row["seq"],
                    "window_idx": row["window_idx"],
                    "D_in": f"{float(row['D_in']):.6f}",
                    "D_out": f"{float(row['D_out']):.6f}",
                    "tau_in": f"{rec_tau_in:.6f}",
                    "tau_out": f"{rec_tau_out:.6f}",
                    "K": rec_k,
                    "trigger": int(state["trigger"]),
                    "alert": int(state["alert"]),
                    "IDSW": f"{float(row['IDSW']):.6f}",
                    "CountMAE": f"{float(row['CountMAE']):.6f}",
                    "F1": f"{float(row['F1']):.6f}",
                }
            )

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_out.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "experiment",
            "scenario",
            "quantile",
            "K",
            "tau_in",
            "tau_out",
            "false_alert_rate",
            "missed_drift_rate",
            "missed_drift_rate_drop_jitter",
            "missed_drift_rate_env",
            "IDSW_off",
            "IDSW_on",
            "CountMAE_off",
            "CountMAE_on",
            "delta_IDSW",
            "delta_CountMAE",
            "recommended",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    args.timeline_out.parent.mkdir(parents=True, exist_ok=True)
    with args.timeline_out.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "scenario",
            "seq",
            "window_idx",
            "D_in",
            "D_out",
            "tau_in",
            "tau_out",
            "K",
            "trigger",
            "alert",
            "IDSW",
            "CountMAE",
            "F1",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in timeline_rows:
            writer.writerow(row)

    render_indicator_figure(all_rows_base, tau_in=rec_tau_in, tau_out=rec_tau_out, out_path=args.fig_indicator)
    render_ablation_figure(ablation_rows, out_path=args.fig_ablation)
    if mitigation_rows:
        render_mitigation_figure(mitigation_rows, out_path=args.fig_mitigation)
    else:
        raise RuntimeError("No alerted windows found for mitigation analysis; cannot render E3 figure.")

    copy_to_paper_figs(args.fig_indicator, args.paper_fig_dir)
    copy_to_paper_figs(args.fig_ablation, args.paper_fig_dir)
    copy_to_paper_figs(args.fig_mitigation, args.paper_fig_dir)

    print(f"saved summary: {args.summary_out}")
    print(f"saved timeline: {args.timeline_out}")
    print(f"saved figure: {args.fig_indicator}")
    print(f"saved figure: {args.fig_ablation}")
    print(f"saved figure: {args.fig_mitigation}")
    print(
        "recommended config: "
        f"quantile=p{rec_q}, K={rec_k}, tau_in={rec_tau_in:.6f}, tau_out={rec_tau_out:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
