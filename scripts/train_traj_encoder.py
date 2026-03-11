#!/usr/bin/env python
"""Train a minimal self-supervised trajectory encoder (1D-CNN + InfoNCE)."""

from __future__ import annotations

import argparse
import configparser
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train trajectory encoder with InfoNCE.")
    parser.add_argument(
        "--mot-root",
        type=Path,
        default=Path("data/mft25_mot"),
        help="Root directory of prepared MOT data.",
    )
    parser.add_argument(
        "--split",
        default="train_half",
        help="Prepared split name under mot-root.",
    )
    parser.add_argument(
        "--seq",
        default="BT-001",
        help="Sequence name to train on when --gt-path is not provided.",
    )
    parser.add_argument(
        "--gt-path",
        type=Path,
        default=None,
        help="Optional MOT-format gt.txt override.",
    )
    parser.add_argument(
        "--seqinfo-path",
        type=Path,
        default=None,
        help="Optional seqinfo.ini path. If omitted, infer from gt-path.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=200,
        help="Use detections with frame <= max-frames.",
    )
    parser.add_argument("--window-size", type=int, default=16, help="Trajectory window length N.")
    parser.add_argument(
        "--pseudo-iou-thresh",
        type=float,
        default=0.3,
        help="IoU threshold for building pseudo tracks from detections.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate.")
    parser.add_argument("--temperature", type=float, default=0.2, help="NT-Xent temperature.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/traj_encoder"),
        help="Output directory for checkpoint/log.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_seqinfo(seqinfo_path: Path) -> Tuple[int, int]:
    parser = configparser.ConfigParser()
    parser.read(seqinfo_path, encoding="utf-8")
    if "Sequence" not in parser:
        raise RuntimeError(f"Invalid seqinfo.ini (missing [Sequence]): {seqinfo_path}")
    seq = parser["Sequence"]
    width = int(seq["imWidth"])
    height = int(seq["imHeight"])
    return width, height


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


def load_detections(gt_path: Path, max_frames: int) -> DefaultDict[int, List[np.ndarray]]:
    by_frame: DefaultDict[int, List[np.ndarray]] = defaultdict(list)
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            if frame < 1 or frame > max_frames:
                continue
            x, y, w, h = [float(v) for v in parts[2:6]]
            if w <= 0 or h <= 0:
                continue
            by_frame[frame].append(np.array([x, y, w, h], dtype=float))
    return by_frame


def build_pseudo_tracks(
    detections_by_frame: Dict[int, List[np.ndarray]],
    max_frames: int,
    iou_thresh: float,
) -> Dict[int, List[Tuple[int, float, float]]]:
    next_track_id = 1
    active_tracks: Dict[int, np.ndarray] = {}
    tracks: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)

    for frame in range(1, max_frames + 1):
        detections = detections_by_frame.get(frame, [])
        active_ids = list(active_tracks.keys())
        active_boxes = [active_tracks[t] for t in active_ids]

        matched_pairs: List[Tuple[int, int]] = []
        used_dets = set()
        if detections and active_boxes:
            iou = iou_batch(np.asarray(detections, dtype=float), np.asarray(active_boxes, dtype=float))
            row_idx, col_idx = linear_sum_assignment(1.0 - iou)
            for det_i, trk_i in zip(row_idx.tolist(), col_idx.tolist()):
                if iou[det_i, trk_i] < iou_thresh:
                    continue
                matched_pairs.append((det_i, trk_i))
                used_dets.add(det_i)

        new_active: Dict[int, np.ndarray] = {}
        for det_i, trk_i in matched_pairs:
            track_id = active_ids[trk_i]
            box = detections[det_i]
            new_active[track_id] = box
            cx = float(box[0] + box[2] / 2.0)
            cy = float(box[1] + box[3] / 2.0)
            tracks[track_id].append((frame, cx, cy))

        for det_i, box in enumerate(detections):
            if det_i in used_dets:
                continue
            track_id = next_track_id
            next_track_id += 1
            new_active[track_id] = box
            cx = float(box[0] + box[2] / 2.0)
            cy = float(box[1] + box[3] / 2.0)
            tracks[track_id].append((frame, cx, cy))

        active_tracks = new_active

    return tracks


def trajectory_to_windows(
    trajectory: Sequence[Tuple[int, float, float]],
    window_size: int,
    norm_w: float,
    norm_h: float,
) -> List[np.ndarray]:
    if len(trajectory) < window_size:
        return []
    centers = np.asarray([[x, y] for _, x, y in trajectory], dtype=np.float32)
    windows: List[np.ndarray] = []
    for start in range(0, len(centers) - window_size + 1):
        slice_xy = centers[start : start + window_size]
        diff_xy = np.diff(slice_xy, axis=0, prepend=slice_xy[:1])
        diff_xy[:, 0] /= norm_w
        diff_xy[:, 1] /= norm_h
        windows.append(diff_xy.astype(np.float32))
    return windows


def build_positive_pairs(
    tracks: Dict[int, List[Tuple[int, float, float]]],
    window_size: int,
    norm_w: float,
    norm_h: float,
) -> Tuple[np.ndarray, np.ndarray]:
    prev_windows: List[np.ndarray] = []
    next_windows: List[np.ndarray] = []

    for trajectory in tracks.values():
        windows = trajectory_to_windows(trajectory, window_size, norm_w, norm_h)
        if len(windows) < 2:
            continue
        for i in range(len(windows) - 1):
            prev_windows.append(windows[i])
            next_windows.append(windows[i + 1])

    if not prev_windows:
        raise RuntimeError(
            "No positive pairs were built. "
            "Try increasing max_frames or lowering pseudo track IoU threshold."
        )

    return np.stack(prev_windows, axis=0), np.stack(next_windows, axis=0)


class TrajPairDataset(Dataset):
    def __init__(self, prev_windows: np.ndarray, next_windows: np.ndarray) -> None:
        super().__init__()
        if prev_windows.shape != next_windows.shape:
            raise ValueError("prev_windows and next_windows must have the same shape.")
        # [B, N, 2] -> [B, 2, N]
        self.prev = torch.from_numpy(np.transpose(prev_windows, (0, 2, 1)))
        self.next = torch.from_numpy(np.transpose(next_windows, (0, 2, 1)))

    def __len__(self) -> int:
        return int(self.prev.shape[0])

    def __getitem__(self, index: int):
        return self.prev[index], self.next[index]


class TrajEncoder(nn.Module):
    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.mean(dim=-1)  # Global Average Pooling over time.
        x = self.fc(x)
        return F.normalize(x, dim=1)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    if z1.shape[0] != z2.shape[0]:
        raise ValueError("z1 and z2 batch sizes must match.")
    batch = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = torch.matmul(z, z.t()) / temperature
    mask = torch.eye(2 * batch, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e9)
    positives = torch.cat([torch.diag(sim, batch), torch.diag(sim, -batch)], dim=0)
    log_denom = torch.logsumexp(sim, dim=1)
    return -(positives - log_denom).mean()


def infer_seqinfo_path(gt_path: Path) -> Path:
    # .../<SEQ>/gt/gt.txt -> .../<SEQ>/seqinfo.ini
    return gt_path.parent.parent / "seqinfo.ini"


def resolve_data_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.gt_path is not None:
        gt_path = args.gt_path
        seqinfo_default = infer_seqinfo_path(gt_path)
    else:
        seq_dir = args.mot_root / args.split / args.seq
        gt_path = seq_dir / "gt" / "gt.txt"
        seqinfo_default = seq_dir / "seqinfo.ini"
    seqinfo_path = args.seqinfo_path or seqinfo_default
    return gt_path, seqinfo_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    gt_path, seqinfo_path = resolve_data_paths(args)
    if not gt_path.exists():
        raise FileNotFoundError(f"gt.txt not found: {gt_path}")
    if not seqinfo_path.exists():
        raise FileNotFoundError(f"seqinfo.ini not found: {seqinfo_path}")

    norm_w, norm_h = read_seqinfo(seqinfo_path)
    detections_by_frame = load_detections(gt_path=gt_path, max_frames=args.max_frames)
    pseudo_tracks = build_pseudo_tracks(
        detections_by_frame=detections_by_frame,
        max_frames=args.max_frames,
        iou_thresh=args.pseudo_iou_thresh,
    )
    prev_windows, next_windows = build_positive_pairs(
        tracks=pseudo_tracks,
        window_size=args.window_size,
        norm_w=float(norm_w),
        norm_h=float(norm_h),
    )

    dataset = TrajPairDataset(prev_windows=prev_windows, next_windows=next_windows)
    if len(dataset) < 2:
        raise RuntimeError("Need at least 2 positive pairs to run InfoNCE.")

    effective_batch_size = min(args.batch_size, len(dataset))
    if effective_batch_size < 2:
        raise RuntimeError("Effective batch size must be >= 2.")

    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=(len(dataset) >= effective_batch_size * 2),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajEncoder(out_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "train_log.csv"
    ckpt_path = args.out_dir / "traj_encoder.pt"

    log_rows: List[Tuple[int, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_steps = 0

        for batch_prev, batch_next in loader:
            batch_prev = batch_prev.to(device=device, dtype=torch.float32)
            batch_next = batch_next.to(device=device, dtype=torch.float32)

            z_prev = model(batch_prev)
            z_next = model(batch_next)
            loss = nt_xent_loss(z_prev, z_next, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_steps += 1

        epoch_loss = total_loss / max(1, total_steps)
        log_rows.append((epoch, epoch_loss))
        print(f"epoch={epoch:03d}/{args.epochs:03d}, loss={epoch_loss:.6f}")

    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for epoch, loss in log_rows:
            writer.writerow([epoch, f"{loss:.6f}"])

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "split": args.split,
                "seq": gt_path.parent.parent.name,
                "window_size": args.window_size,
                "feature": "normalized_dxdy",
                "out_dim": 128,
                "pseudo_iou_thresh": args.pseudo_iou_thresh,
                "max_frames": args.max_frames,
            },
        },
        ckpt_path,
    )

    print(
        f"saved checkpoint: {ckpt_path}\n"
        f"saved train log:   {log_path}\n"
        f"pairs={len(dataset)}, tracks={len(pseudo_tracks)}, "
        f"seq={gt_path.parent.parent.name}, frames<= {args.max_frames}"
    )


if __name__ == "__main__":
    main()
