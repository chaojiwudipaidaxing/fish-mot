#!/usr/bin/env python
"""Run a minimal SORT-style baseline tracker on MOT-format detections."""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import os
import shutil
from collections import defaultdict, deque
from pathlib import Path
from typing import DefaultDict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

DEFAULT_GATING_THRESHOLD = 9.210340371976184  # chi-square(dof=2, p=0.99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal SORT baseline on MFT25 MOT data.")
    parser.add_argument(
        "--split",
        default="train_half",
        help="Prepared split under data/mft25_mot/<split>.",
    )
    parser.add_argument(
        "--mot-root",
        type=Path,
        default=Path("data/mft25_mot"),
        help="Root directory of prepared MOT data.",
    )
    parser.add_argument(
        "--seqs",
        nargs="+",
        default=None,
        help="Optional sequence list. If omitted, read from seqmaps/<split>.txt.",
    )
    parser.add_argument(
        "--det-source",
        choices=["auto", "det", "gt"],
        default="gt",
        help="Detection source: det/det.txt, gt/gt.txt, or auto(det if exists else gt).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="If > 0, only process frames <= max-frames.",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.3,
        help="IoU threshold for detection-track matching.",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="Minimum hits before a track is emitted as confirmed.",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Maximum missed frames before deleting a lost track.",
    )
    parser.add_argument(
        "--gating",
        choices=["on", "off"],
        default="off",
        help="Enable/disable Mahalanobis gating before Hungarian matching.",
    )
    parser.add_argument(
        "--gating-thresh",
        type=float,
        default=DEFAULT_GATING_THRESHOLD,
        help="Mahalanobis d^2 threshold when gating is on.",
    )
    parser.add_argument(
        "--traj",
        choices=["on", "off"],
        default="off",
        help="Enable/disable trajectory consistency cost.",
    )
    parser.add_argument(
        "--traj-encoder",
        type=Path,
        default=Path("runs/traj_encoder/traj_encoder.pt"),
        help="Path to trajectory encoder checkpoint from scripts/train_traj_encoder.py.",
    )
    parser.add_argument(
        "--traj-window",
        type=int,
        default=16,
        help="Trajectory deque length N used for consistency cost.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight for IoU cost term: alpha * (1 - IoU).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="Weight for Mahalanobis term: beta * d_maha.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Weight for trajectory term: gamma * traj_cost.",
    )
    parser.add_argument(
        "--adaptive-gamma",
        choices=["on", "off"],
        default="off",
        help="Enable/disable adaptive trajectory weight policy.",
    )
    parser.add_argument(
        "--adaptive-gamma-boost",
        type=float,
        default=1.5,
        help="Multiplier for gamma when lost_count (time_since_update) >= 2.",
    )
    parser.add_argument(
        "--adaptive-gamma-min",
        type=float,
        default=0.5,
        help="Lower clamp bound for adaptive gamma (applied when gamma > 0).",
    )
    parser.add_argument(
        "--adaptive-gamma-max",
        type=float,
        default=2.0,
        help="Upper clamp bound for adaptive gamma (applied when gamma > 0).",
    )
    parser.add_argument(
        "--drop-rate",
        type=float,
        default=0.0,
        help="Randomly drop each detection with this probability (for robustness tests).",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.0,
        help="Random bbox perturbation ratio, e.g. 0.02 means 2%% jitter.",
    )
    parser.add_argument(
        "--degrade-seed",
        type=int,
        default=42,
        help="Random seed used for detection degradation.",
    )
    parser.add_argument(
        "--motion-blur",
        type=float,
        default=0.0,
        help="Deterministic motion-blur proxy severity in [0, 1] applied in detection space.",
    )
    parser.add_argument(
        "--darken",
        type=float,
        default=0.0,
        help="Deterministic low-illumination proxy severity in [0, 1] applied in detection space.",
    )
    parser.add_argument(
        "--haze",
        type=float,
        default=0.0,
        help="Deterministic low-contrast / turbidity proxy severity in [0, 1] applied in detection space.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/baseline/pred"),
        help="Output directory for MOT prediction txt files.",
    )
    parser.add_argument(
        "--clean-out",
        action="store_true",
        help="Delete output directory before writing predictions.",
    )
    parser.add_argument(
        "--frame-stats",
        choices=["on", "off"],
        default="off",
        help="Print per-frame tracker stats (matches/lost/new).",
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=None,
        help="Optional run_config.json. When provided, mot_root is resolved from it.",
    )
    args = parser.parse_args()

    loaded_from_run_config = False
    if args.run_config is not None:
        if not args.run_config.exists():
            raise FileNotFoundError(f"Run config not found: {args.run_config}")
        cfg = json.loads(args.run_config.read_text(encoding="utf-8"))
        loaded_from_run_config = True
        mot_root_cfg = cfg.get("mot_root")
        if mot_root_cfg:
            args.mot_root = Path(str(mot_root_cfg))
        if args.split == "train_half" and cfg.get("split"):
            args.split = str(cfg.get("split"))
        if args.max_frames <= 0 and cfg.get("max_frames") is not None:
            args.max_frames = int(cfg.get("max_frames"))
        if args.drop_rate == 0.0 and cfg.get("drop_rate") is not None:
            args.drop_rate = float(cfg.get("drop_rate"))
        if args.jitter == 0.0 and cfg.get("jitter") is not None:
            args.jitter = float(cfg.get("jitter"))
        if args.gating_thresh == DEFAULT_GATING_THRESHOLD and cfg.get("gating_thresh") is not None:
            args.gating_thresh = float(cfg.get("gating_thresh"))

    args.gating_thresh_source = "run_config" if loaded_from_run_config else "arg"
    _env_gt = os.getenv("GATING_THRESH")
    if _env_gt is not None and args.gating == "on" and args.gating_thresh == DEFAULT_GATING_THRESHOLD:
        try:
            args.gating_thresh = float(_env_gt)
            args.gating_thresh_source = "env"
        except Exception:
            raise ValueError("Invalid env GATING_THRESH=%r" % (_env_gt,))
    elif args.gating_thresh == DEFAULT_GATING_THRESHOLD:
        args.gating_thresh_source = "default"
    return args


def read_seqmap(seqmap_path: Path) -> List[str]:
    if not seqmap_path.exists():
        raise FileNotFoundError(f"Missing seqmap file: {seqmap_path}")
    sequences: List[str] = []
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
            sequences.append(name)
    return sequences


def read_seq_info(seqinfo_path: Path) -> Tuple[int, int, int]:
    parser = configparser.ConfigParser()
    parser.read(seqinfo_path, encoding="utf-8")
    if "Sequence" not in parser:
        raise RuntimeError(f"Cannot read [Sequence] from {seqinfo_path}")
    seq = parser["Sequence"]
    return int(seq["seqLength"]), int(seq["imWidth"]), int(seq["imHeight"])


def load_detections_from_mot(mot_path: Path, max_frames: int) -> DefaultDict[int, List[np.ndarray]]:
    detections: DefaultDict[int, List[np.ndarray]] = defaultdict(list)
    with mot_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            if frame < 1:
                continue
            if max_frames > 0 and frame > max_frames:
                continue
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            if w <= 0 or h <= 0:
                continue
            detections[frame].append(np.array([x, y, w, h], dtype=float))
    return detections


def degrade_detections(
    detections_xywh: List[np.ndarray],
    drop_rate: float,
    jitter: float,
    motion_blur: float,
    darken: float,
    haze: float,
    frame_idx: int,
    image_w: int,
    image_h: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    if not detections_xywh:
        return []

    def clamp_bbox(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
        x = max(0.0, min(x, float(image_w) - 1.0))
        y = max(0.0, min(y, float(image_h) - 1.0))
        w = max(1.0, min(w, float(image_w) - x))
        h = max(1.0, min(h, float(image_h) - y))
        return x, y, w, h

    def small_object_weight(w: float, h: float) -> float:
        area_ratio = (w * h) / max(1.0, float(image_w) * float(image_h))
        if area_ratio <= 0.0:
            return 1.8
        ref_ratio = 0.0025
        return float(np.clip(np.sqrt(ref_ratio / area_ratio), 0.6, 1.8))

    frame_phase = 0.23 * float(frame_idx)
    haze_dx = float(rng.uniform(-1.0, 1.0)) * haze * 0.008 * float(image_w)
    haze_dy = float(rng.uniform(-1.0, 1.0)) * haze * 0.008 * float(image_h)
    out: List[np.ndarray] = []
    for det in detections_xywh:
        if drop_rate > 0.0 and float(rng.random()) < drop_rate:
            continue
        x, y, w, h = [float(v) for v in det]
        size_weight = small_object_weight(w, h)

        if darken > 0.0:
            dark_drop = min(0.85, darken * 0.35 * size_weight)
            if float(rng.random()) < dark_drop:
                continue
            shrink = max(0.55, 1.0 - darken * (0.10 + 0.04 * size_weight))
            cx = x + w / 2.0
            cy = y + h / 2.0
            w = max(1.0, w * shrink)
            h = max(1.0, h * shrink)
            x = cx - w / 2.0
            y = cy - h / 2.0

        if motion_blur > 0.0:
            angle = float(rng.uniform(-np.pi, np.pi)) + frame_phase
            axis_major = abs(np.cos(angle)) >= abs(np.sin(angle))
            shift = motion_blur * 0.55 * max(w, h)
            cx = x + w / 2.0 + np.cos(angle) * shift
            cy = y + h / 2.0 + np.sin(angle) * shift * 0.7
            stretch = 1.0 + 1.20 * motion_blur
            squeeze = max(0.65, 1.0 - 0.20 * motion_blur)
            if axis_major:
                w = max(1.0, w * stretch)
                h = max(1.0, h * squeeze)
            else:
                h = max(1.0, h * stretch)
                w = max(1.0, w * squeeze)
            x = cx - w / 2.0
            y = cy - h / 2.0
            blur_drop = min(0.60, motion_blur * 0.18 * size_weight)
            if float(rng.random()) < blur_drop:
                continue

        if haze > 0.0:
            x = x + haze_dx + float(rng.normal(0.0, haze * 0.05 * max(1.0, w)))
            y = y + haze_dy + float(rng.normal(0.0, haze * 0.05 * max(1.0, h)))
            inflate = 1.0 + 0.35 * haze
            w = max(1.0, w * inflate)
            h = max(1.0, h * inflate)
            haze_drop = min(0.75, haze * 0.22 * (0.7 + 0.3 * size_weight))
            if float(rng.random()) < haze_drop:
                continue

        if jitter > 0.0:
            jx = float(rng.uniform(-jitter, jitter)) * w
            jy = float(rng.uniform(-jitter, jitter)) * h
            jw = 1.0 + float(rng.uniform(-jitter, jitter))
            jh = 1.0 + float(rng.uniform(-jitter, jitter))
            x = x + jx
            y = y + jy
            w = max(1.0, w * jw)
            h = max(1.0, h * jh)

        x, y, w, h = clamp_bbox(x, y, w, h)
        out.append(np.array([x, y, w, h], dtype=float))
    return out


def xywh_to_cxcywh(bbox_xywh: Sequence[float]) -> np.ndarray:
    x, y, w, h = [float(v) for v in bbox_xywh]
    return np.array([x + w / 2.0, y + h / 2.0, w, h], dtype=float)


def cxcywh_to_xywh(state: Sequence[float]) -> np.ndarray:
    cx, cy, w, h = [float(v) for v in state]
    return np.array([cx - w / 2.0, cy - h / 2.0, w, h], dtype=float)


def iou_batch(dets_xywh: np.ndarray, trks_xywh: np.ndarray) -> np.ndarray:
    if dets_xywh.size == 0 or trks_xywh.size == 0:
        return np.zeros((dets_xywh.shape[0], trks_xywh.shape[0]), dtype=float)
    dets_x1 = dets_xywh[:, 0:1]
    dets_y1 = dets_xywh[:, 1:2]
    dets_x2 = dets_x1 + dets_xywh[:, 2:3]
    dets_y2 = dets_y1 + dets_xywh[:, 3:4]
    trks_x1 = trks_xywh[:, 0][None, :]
    trks_y1 = trks_xywh[:, 1][None, :]
    trks_x2 = trks_x1 + trks_xywh[:, 2][None, :]
    trks_y2 = trks_y1 + trks_xywh[:, 3][None, :]
    inter_x1 = np.maximum(dets_x1, trks_x1)
    inter_y1 = np.maximum(dets_y1, trks_y1)
    inter_x2 = np.minimum(dets_x2, trks_x2)
    inter_y2 = np.minimum(dets_y2, trks_y2)
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    det_area = np.maximum(0.0, dets_xywh[:, 2:3]) * np.maximum(0.0, dets_xywh[:, 3:4])
    trk_area = np.maximum(0.0, trks_xywh[:, 2][None, :]) * np.maximum(0.0, trks_xywh[:, 3][None, :])
    union = det_area + trk_area - inter_area
    return np.where(union > 0.0, inter_area / union, 0.0)


def centers_to_feature(
    centers: Sequence[np.ndarray],
    window_size: int,
    norm_w: float,
    norm_h: float,
) -> np.ndarray:
    if not centers:
        arr = np.zeros((window_size, 2), dtype=np.float32)
    else:
        arr = np.asarray(centers, dtype=np.float32)
        if arr.shape[0] >= window_size:
            arr = arr[-window_size:]
        else:
            pad = np.repeat(arr[:1], window_size - arr.shape[0], axis=0)
            arr = np.concatenate([pad, arr], axis=0)
    diff = np.diff(arr, axis=0, prepend=arr[:1])
    diff[:, 0] /= max(1.0, float(norm_w))
    diff[:, 1] /= max(1.0, float(norm_h))
    return diff.astype(np.float32)


class TrajEncoder(nn.Module):
    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.mean(dim=-1)
        x = self.fc(x)
        return F.normalize(x, dim=1)


class TrajCostModel:
    def __init__(self, ckpt_path: Path) -> None:
        self.device = torch.device("cpu")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model = TrajEncoder(out_dim=128).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def encode(self, windows: np.ndarray) -> np.ndarray:
        if windows.size == 0:
            return np.zeros((0, 128), dtype=np.float32)
        # [B, N, 2] -> [B, 2, N]
        tensor = torch.from_numpy(np.transpose(windows, (0, 2, 1))).to(self.device, dtype=torch.float32)
        emb = self.model(tensor).cpu().numpy()
        return emb.astype(np.float32)


def mahalanobis_distance_matrix(
    det_centers: np.ndarray,
    gating_stats: List[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    num_dets = det_centers.shape[0]
    num_trks = len(gating_stats)
    d2 = np.zeros((num_dets, num_trks), dtype=float)
    if num_dets == 0 or num_trks == 0:
        return d2
    for trk_i, (mean, cov) in enumerate(gating_stats):
        inv_cov = np.linalg.pinv(cov)
        diffs = det_centers - mean[None, :]
        d2[:, trk_i] = np.einsum("ni,ij,nj->n", diffs, inv_cov, diffs)
    return d2


def trajectory_cost_matrix(
    det_centers: np.ndarray,
    tracks: List["SortTrack"],
    traj_model: TrajCostModel | None,
    window_size: int,
    norm_w: float,
    norm_h: float,
) -> np.ndarray:
    num_dets = det_centers.shape[0]
    num_trks = len(tracks)
    costs = np.zeros((num_dets, num_trks), dtype=float)
    if traj_model is None or num_dets == 0 or num_trks == 0:
        return costs

    prev_windows: List[np.ndarray] = []
    valid_track = np.zeros((num_trks,), dtype=bool)
    for trk_i, track in enumerate(tracks):
        centers = list(track.center_history)
        if len(centers) >= 2:
            valid_track[trk_i] = True
        prev_windows.append(centers_to_feature(centers, window_size, norm_w, norm_h))
    prev_emb = traj_model.encode(np.stack(prev_windows, axis=0))

    appended_windows: List[np.ndarray] = []
    mapping: List[Tuple[int, int]] = []
    for trk_i, track in enumerate(tracks):
        if not valid_track[trk_i]:
            continue
        base_centers = list(track.center_history)
        for det_i in range(num_dets):
            appended = base_centers + [det_centers[det_i]]
            appended_windows.append(centers_to_feature(appended, window_size, norm_w, norm_h))
            mapping.append((det_i, trk_i))
    if not appended_windows:
        return costs

    appended_emb = traj_model.encode(np.stack(appended_windows, axis=0))
    for row_i, (det_i, trk_i) in enumerate(mapping):
        cos = float(np.dot(prev_emb[trk_i], appended_emb[row_i]))
        cos = max(-1.0, min(1.0, cos))
        costs[det_i, trk_i] = 1.0 - cos
    return costs


def associate_detections_to_tracks(
    detections_xywh: List[np.ndarray],
    predictions_xywh: List[np.ndarray],
    iou_thresh: float,
    alpha: float,
    beta: float,
    gamma_by_track: np.ndarray,
    d_maha: np.ndarray,
    traj_cost: np.ndarray,
    gating_enabled: bool,
    gating_threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    inf_cost = 1e6
    num_dets = len(detections_xywh)
    num_trks = len(predictions_xywh)
    if num_dets == 0:
        return [], [], list(range(num_trks))
    if num_trks == 0:
        return [], list(range(num_dets)), []

    dets = np.asarray(detections_xywh, dtype=float)
    trks = np.asarray(predictions_xywh, dtype=float)
    iou = iou_batch(dets, trks)

    if d_maha.shape != iou.shape:
        d_maha = np.zeros_like(iou)
    if traj_cost.shape != iou.shape:
        traj_cost = np.zeros_like(iou)
    if gamma_by_track.shape[0] != num_trks:
        gamma_by_track = np.zeros((num_trks,), dtype=float)

    cost = alpha * (1.0 - iou) + beta * d_maha + traj_cost * gamma_by_track[None, :]
    if gating_enabled:
        cost[d_maha > gating_threshold] = inf_cost

    row_idx, col_idx = linear_sum_assignment(cost)
    matches: List[Tuple[int, int]] = []
    unmatched_dets = set(range(num_dets))
    unmatched_trks = set(range(num_trks))
    for det_i, trk_i in zip(row_idx.tolist(), col_idx.tolist()):
        if cost[det_i, trk_i] >= inf_cost:
            continue
        if iou[det_i, trk_i] < iou_thresh:
            continue
        matches.append((det_i, trk_i))
        unmatched_dets.discard(det_i)
        unmatched_trks.discard(trk_i)
    return matches, sorted(unmatched_dets), sorted(unmatched_trks)


class SimpleKalman:
    """Constant velocity Kalman filter over (cx, cy, vx, vy, w, h)."""

    def __init__(self, bbox_xywh: Sequence[float]) -> None:
        cx, cy, w, h = xywh_to_cxcywh(bbox_xywh)
        self.x = np.array([[cx], [cy], [0.0], [0.0], [w], [h]], dtype=float)
        self.P = np.diag([25.0, 25.0, 400.0, 400.0, 25.0, 25.0]).astype(float)
        self.F = np.array(
            [
                [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        self.Q = np.diag([1.0, 1.0, 10.0, 10.0, 1.0, 1.0]).astype(float)
        self.R = np.diag([10.0, 10.0, 10.0, 10.0]).astype(float)
        self.I = np.eye(6, dtype=float)

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x[4, 0] = max(1.0, self.x[4, 0])
        self.x[5, 0] = max(1.0, self.x[5, 0])

    def update(self, bbox_xywh: Sequence[float]) -> None:
        cx, cy, w, h = xywh_to_cxcywh(bbox_xywh)
        z = np.array([[cx], [cy], [w], [h]], dtype=float)
        y = z - (self.H @ self.x)
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ np.linalg.inv(s)
        self.x = self.x + (k @ y)
        self.P = (self.I - (k @ self.H)) @ self.P
        self.x[4, 0] = max(1.0, self.x[4, 0])
        self.x[5, 0] = max(1.0, self.x[5, 0])

    def get_bbox_xywh(self) -> np.ndarray:
        state = np.array([self.x[0, 0], self.x[1, 0], self.x[4, 0], self.x[5, 0]], dtype=float)
        return cxcywh_to_xywh(state)

    def project_center(self) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.array([self.x[0, 0], self.x[1, 0]], dtype=float)
        cov = self.P[:2, :2] + self.R[:2, :2]
        return mean, cov


class SortTrack:
    def __init__(self, bbox_xywh: Sequence[float], track_id: int, traj_window: int) -> None:
        self.id = int(track_id)
        self.kf = SimpleKalman(bbox_xywh)
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.center_history: deque[np.ndarray] = deque(maxlen=traj_window)
        self._append_center(bbox_xywh)

    def _append_center(self, bbox_xywh: Sequence[float]) -> None:
        cx, cy, _, _ = xywh_to_cxcywh(bbox_xywh)
        self.center_history.append(np.array([cx, cy], dtype=np.float32))

    def predict(self) -> np.ndarray:
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        return self.kf.get_bbox_xywh()

    def update(self, bbox_xywh: Sequence[float]) -> None:
        self.kf.update(bbox_xywh)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self._append_center(bbox_xywh)

    def bbox_xywh(self) -> np.ndarray:
        return self.kf.get_bbox_xywh()

    def projected_center_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.kf.project_center()


class SortTracker:
    def __init__(
        self,
        iou_thresh: float,
        min_hits: int,
        max_age: int,
        gating_enabled: bool,
        gating_threshold: float,
        alpha: float,
        beta: float,
        gamma: float,
        traj_enabled: bool,
        adaptive_gamma: bool,
        adaptive_gamma_boost: float,
        adaptive_gamma_min: float,
        adaptive_gamma_max: float,
        traj_window: int,
        traj_model: TrajCostModel | None,
        norm_w: float,
        norm_h: float,
    ) -> None:
        self.iou_thresh = float(iou_thresh)
        self.min_hits = int(min_hits)
        self.max_age = int(max_age)
        self.gating_enabled = bool(gating_enabled)
        self.gating_threshold = float(gating_threshold)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.traj_enabled = bool(traj_enabled)
        self.adaptive_gamma = bool(adaptive_gamma)
        self.adaptive_gamma_boost = float(adaptive_gamma_boost)
        self.adaptive_gamma_min = float(adaptive_gamma_min)
        self.adaptive_gamma_max = float(adaptive_gamma_max)
        self.traj_window = int(traj_window)
        self.traj_model = traj_model
        self.norm_w = float(norm_w)
        self.norm_h = float(norm_h)

        self.tracks: List[SortTrack] = []
        self.next_id = 1

    def _new_track(self, bbox_xywh: Sequence[float]) -> None:
        self.tracks.append(SortTrack(bbox_xywh=bbox_xywh, track_id=self.next_id, traj_window=self.traj_window))
        self.next_id += 1

    def _gamma_for_track(self, track: SortTrack) -> float:
        if not self.traj_enabled:
            return 0.0
        gamma = self.gamma
        if self.adaptive_gamma:
            if track.age < 3:
                return 0.0
            if track.time_since_update >= 2:
                gamma *= self.adaptive_gamma_boost
            if gamma > 0.0:
                gamma = float(np.clip(gamma, self.adaptive_gamma_min, self.adaptive_gamma_max))
        return gamma

    def step(
        self,
        detections_xywh: List[np.ndarray],
        frame_idx: int,
    ) -> Tuple[List[Tuple[int, int, float, float, float, float, float]], dict[str, int]]:
        tracks_before = len(self.tracks)
        predictions: List[np.ndarray] = []
        gating_stats: List[Tuple[np.ndarray, np.ndarray]] = []
        for track in self.tracks:
            predictions.append(track.predict())
            gating_stats.append(track.projected_center_stats())

        det_centers = np.asarray(
            [[det[0] + det[2] / 2.0, det[1] + det[3] / 2.0] for det in detections_xywh],
            dtype=float,
        )
        if det_centers.size == 0:
            det_centers = np.zeros((0, 2), dtype=float)

        d_maha = mahalanobis_distance_matrix(det_centers=det_centers, gating_stats=gating_stats)
        traj_cost = trajectory_cost_matrix(
            det_centers=det_centers,
            tracks=self.tracks,
            traj_model=self.traj_model if self.traj_enabled else None,
            window_size=self.traj_window,
            norm_w=self.norm_w,
            norm_h=self.norm_h,
        )
        gamma_by_track = np.asarray([self._gamma_for_track(track) for track in self.tracks], dtype=float)

        matches, unmatched_det_idx, _ = associate_detections_to_tracks(
            detections_xywh=detections_xywh,
            predictions_xywh=predictions,
            iou_thresh=self.iou_thresh,
            alpha=self.alpha,
            beta=self.beta,
            gamma_by_track=gamma_by_track,
            d_maha=d_maha,
            traj_cost=traj_cost,
            gating_enabled=self.gating_enabled,
            gating_threshold=self.gating_threshold,
        )
        matched_count = len(matches)
        unmatched_tracks = max(0, tracks_before - matched_count)
        new_tracks = len(unmatched_det_idx)

        for det_i, trk_i in matches:
            self.tracks[trk_i].update(detections_xywh[det_i])
        for det_i in unmatched_det_idx:
            self._new_track(detections_xywh[det_i])

        before_prune = len(self.tracks)
        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_age]
        removed_tracks = max(0, before_prune - len(self.tracks))

        outputs: List[Tuple[int, int, float, float, float, float, float]] = []
        for track in self.tracks:
            if track.time_since_update != 0:
                continue
            if track.hits < self.min_hits and frame_idx > self.min_hits:
                continue
            x, y, w, h = track.bbox_xywh()
            outputs.append((frame_idx, track.id, float(x), float(y), float(w), float(h), 1.0))
        stats = {
            "frame": int(frame_idx),
            "detections": int(len(detections_xywh)),
            "tracks_before": int(tracks_before),
            "matches": int(matched_count),
            "lost": int(unmatched_tracks),
            "new_tracks": int(new_tracks),
            "removed_tracks": int(removed_tracks),
            "active_tracks": int(len(self.tracks)),
            "outputs": int(len(outputs)),
        }
        return outputs, stats


def run_sequence(
    split_dir: Path,
    seq: str,
    out_dir: Path,
    det_source: str,
    iou_thresh: float,
    min_hits: int,
    max_age: int,
    gating_enabled: bool,
    gating_threshold: float,
    alpha: float,
    beta: float,
    gamma: float,
    traj_enabled: bool,
    adaptive_gamma: bool,
    adaptive_gamma_boost: float,
    adaptive_gamma_min: float,
    adaptive_gamma_max: float,
    traj_window: int,
    traj_model: TrajCostModel | None,
    drop_rate: float,
    jitter: float,
    motion_blur: float,
    darken: float,
    haze: float,
    rng_seed: int,
    max_frames: int,
    frame_stats: bool,
) -> Tuple[int, int]:
    seq_dir = split_dir / seq
    seqinfo_path = seq_dir / "seqinfo.ini"
    gt_path = seq_dir / "gt" / "gt.txt"
    det_path = seq_dir / "det" / "det.txt"
    if not seqinfo_path.exists():
        raise FileNotFoundError(f"Missing seqinfo.ini: {seqinfo_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing gt.txt: {gt_path}")

    if det_source == "det":
        chosen_det = det_path
    elif det_source == "gt":
        chosen_det = gt_path
    else:
        chosen_det = det_path if det_path.exists() else gt_path
    if not chosen_det.exists():
        raise FileNotFoundError(f"Detection file not found: {chosen_det}")

    seq_length, im_w, im_h = read_seq_info(seqinfo_path)
    detections = load_detections_from_mot(mot_path=chosen_det, max_frames=max_frames)
    if max_frames > 0:
        seq_length = min(seq_length, max_frames)
    rng = np.random.default_rng(int(rng_seed))

    tracker = SortTracker(
        iou_thresh=iou_thresh,
        min_hits=min_hits,
        max_age=max_age,
        gating_enabled=gating_enabled,
        gating_threshold=gating_threshold,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        traj_enabled=traj_enabled,
        adaptive_gamma=adaptive_gamma,
        adaptive_gamma_boost=adaptive_gamma_boost,
        adaptive_gamma_min=adaptive_gamma_min,
        adaptive_gamma_max=adaptive_gamma_max,
        traj_window=traj_window,
        traj_model=traj_model,
        norm_w=float(im_w),
        norm_h=float(im_h),
    )

    outputs: List[Tuple[int, int, float, float, float, float, float]] = []
    for frame in range(1, seq_length + 1):
        raw_dets = detections.get(frame, [])
        frame_dets = degrade_detections(
            detections_xywh=raw_dets,
            drop_rate=drop_rate,
            jitter=jitter,
            motion_blur=motion_blur,
            darken=darken,
            haze=haze,
            frame_idx=frame,
            image_w=im_w,
            image_h=im_h,
            rng=rng,
        )
        frame_outputs, frame_debug = tracker.step(frame_dets, frame)
        outputs.extend(frame_outputs)
        if frame_stats:
            print(
                f"[baseline_sort][frame] seq={seq} frame={frame_debug['frame']} det={frame_debug['detections']} "
                f"match={frame_debug['matches']} lost={frame_debug['lost']} new={frame_debug['new_tracks']} "
                f"removed={frame_debug['removed_tracks']} active={frame_debug['active_tracks']} "
                f"emit={frame_debug['outputs']}"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{seq}.txt"
    with out_file.open("w", encoding="utf-8", newline="\n") as f:
        for frame, track_id, x, y, w, h, score in outputs:
            f.write(f"{frame},{track_id},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{score:.3f},1,1\n")

    raw_det_count = sum(len(v) for v in detections.values())
    print(
        f"[baseline_sort] {seq}: frames={seq_length}, dets={raw_det_count}, "
        f"pred_rows={len(outputs)}, out={out_file}, drop_rate={drop_rate:.3f}, jitter={jitter:.3f}, "
        f"motion_blur={motion_blur:.3f}, darken={darken:.3f}, haze={haze:.3f}, det={chosen_det}"
    )
    return seq_length, len(outputs)


def resolve_sequences(split_dir: Path, split: str, seqs: List[str] | None) -> List[str]:
    if seqs:
        return seqs
    seqmap_path = split_dir / "seqmaps" / f"{split}.txt"
    return read_seqmap(seqmap_path)


def main() -> None:
    args = parse_args()
    split_dir = args.mot_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Prepared split not found: {split_dir}. "
            "Please run scripts/prepare_mft25.py first."
        )
    sequences = resolve_sequences(split_dir=split_dir, split=args.split, seqs=args.seqs)
    if not sequences:
        raise RuntimeError("No sequence selected for baseline tracking.")
    if args.gating == "on" and args.gating_thresh <= 0.0:
        raise ValueError("--gating-thresh must be > 0 when gating is on.")
    if args.traj_window < 2:
        raise ValueError("--traj-window must be >= 2.")
    if args.adaptive_gamma_min <= 0.0:
        raise ValueError("--adaptive-gamma-min must be > 0.")
    if args.adaptive_gamma_max < args.adaptive_gamma_min:
        raise ValueError("--adaptive-gamma-max must be >= --adaptive-gamma-min.")
    if not (0.0 <= args.drop_rate < 1.0):
        raise ValueError("--drop-rate must be in [0, 1).")
    if args.jitter < 0.0:
        raise ValueError("--jitter must be >= 0.")
    if not (0.0 <= args.motion_blur <= 1.0):
        raise ValueError("--motion-blur must be in [0, 1].")
    if not (0.0 <= args.darken <= 1.0):
        raise ValueError("--darken must be in [0, 1].")
    if not (0.0 <= args.haze <= 1.0):
        raise ValueError("--haze must be in [0, 1].")
    if args.traj == "on" and not args.traj_encoder.exists():
        raise FileNotFoundError(f"Trajectory encoder checkpoint not found: {args.traj_encoder}")

    if args.clean_out and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    traj_model = TrajCostModel(args.traj_encoder) if args.traj == "on" else None
    total_frames = 0
    total_rows = 0
    for seq_idx, seq in enumerate(sequences):
        frames, rows = run_sequence(
            split_dir=split_dir,
            seq=seq,
            out_dir=args.out_dir,
            det_source=args.det_source,
            iou_thresh=args.iou_thresh,
            min_hits=args.min_hits,
            max_age=args.max_age,
            gating_enabled=(args.gating == "on"),
            gating_threshold=args.gating_thresh,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            traj_enabled=(args.traj == "on"),
            adaptive_gamma=(args.adaptive_gamma == "on"),
            adaptive_gamma_boost=args.adaptive_gamma_boost,
            adaptive_gamma_min=args.adaptive_gamma_min,
            adaptive_gamma_max=args.adaptive_gamma_max,
            traj_window=args.traj_window,
            traj_model=traj_model,
            drop_rate=args.drop_rate,
            jitter=args.jitter,
            motion_blur=args.motion_blur,
            darken=args.darken,
            haze=args.haze,
            rng_seed=(args.degrade_seed + seq_idx),
            max_frames=args.max_frames,
            frame_stats=(args.frame_stats == "on"),
        )
        total_frames += frames
        total_rows += rows

    print(
        f"[baseline_sort] done: split={args.split}, sequences={len(sequences)}, "
        f"frames={total_frames}, pred_rows={total_rows}, pred_dir={args.out_dir}, "
        f"gating={args.gating}, gating_thresh={args.gating_thresh:.6f}({args.gating_thresh_source}), "
        f"traj={args.traj}, adaptive_gamma={args.adaptive_gamma}, frame_stats={args.frame_stats}, "
        f"alpha={args.alpha:.3f}, beta={args.beta:.3f}, gamma={args.gamma:.3f}, "
        f"gamma_clamp=[{args.adaptive_gamma_min:.3f},{args.adaptive_gamma_max:.3f}], "
        f"drop_rate={args.drop_rate:.3f}, jitter={args.jitter:.3f}, motion_blur={args.motion_blur:.3f}, "
        f"darken={args.darken:.3f}, haze={args.haze:.3f}, seed={args.degrade_seed}"
    )


if __name__ == "__main__":
    main()
