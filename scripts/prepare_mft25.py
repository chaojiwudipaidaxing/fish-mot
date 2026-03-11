#!/usr/bin/env python
"""Convert MFT25 JSON annotations to MOTChallenge format for TrackEval."""

from __future__ import annotations

import argparse
import configparser
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MFT25 data in MOTChallenge format.")
    parser.add_argument(
        "--run-config",
        type=Path,
        default=None,
        help="Optional run_config.json. When provided, mot_root/split/max_frames can be resolved from it.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/mft25_raw"),
        help="Root directory of raw MFT25 data.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/mft25_mot"),
        help="Output MOTChallenge directory root.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train_half", "val_half"],
        help="Splits to convert.",
    )
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        default=["MFT25-test", "MFT25-train"],
        help=(
            "Candidate image/seqinfo source folders under raw-root. "
            "The converter tries them in order for each sequence."
        ),
    )
    parser.add_argument(
        "--seq-limit",
        type=int,
        default=0,
        help="If > 0, keep only the first N sequences (by sequence name) per split.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="If > 0, keep only frames with frame_id <= max-frames.",
    )
    parser.add_argument(
        "--clean-split",
        action="store_true",
        help="Delete output split folders before writing.",
    )
    parser.add_argument(
        "--force-copy-images",
        action="store_true",
        help="Copy img1 folders instead of linking/junction where possible.",
    )
    parser.add_argument(
        "--keep-seq-length",
        action="store_true",
        help="Keep original seqLength in seqinfo.ini (default rewrites to split length).",
    )
    return parser.parse_args()


def _norm(path: Path) -> str:
    return str(path).replace("\\", "/").lower()


def apply_run_config(args: argparse.Namespace) -> None:
    if args.run_config is None:
        return
    if not args.run_config.exists():
        raise FileNotFoundError(f"Run config not found: {args.run_config}")
    cfg = json.loads(args.run_config.read_text(encoding="utf-8"))

    mot_root_cfg = cfg.get("mot_root")
    if mot_root_cfg and _norm(args.out_root) == _norm(Path("data/mft25_mot")):
        args.out_root = Path(str(mot_root_cfg))
    if args.splits == ["train_half", "val_half"] and cfg.get("split"):
        args.splits = [str(cfg.get("split"))]
    if args.max_frames <= 0 and cfg.get("max_frames") is not None:
        args.max_frames = int(cfg.get("max_frames"))


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_frame_id(image: Dict) -> int:
    if "frame_id" in image:
        return int(image["frame_id"])
    stem = Path(str(image.get("file_name", ""))).stem
    if stem.isdigit():
        return int(stem)
    raise ValueError(f"Cannot infer frame id from image entry: {image}")


def infer_raw_frame_from_file_name(file_name: str) -> int:
    stem = Path(file_name).stem
    if stem.isdigit():
        return int(stem)
    raise ValueError(f"Cannot infer raw frame from file_name: {file_name}")


def infer_bbox_format(annotations: List[Dict], image_by_id: Dict[int, Dict]) -> str:
    xywh_score = 0
    xyxy_score = 0
    for ann in annotations[:5000]:
        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, b3, b4 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            continue

        image_info = image_by_id.get(ann.get("image_id"), {})
        width = float(image_info.get("width", 0) or 0)
        height = float(image_info.get("height", 0) or 0)

        if b3 > 0 and b4 > 0:
            if width > 0 and height > 0:
                if x1 >= -1 and y1 >= -1 and x1 + b3 <= width + 1 and y1 + b4 <= height + 1:
                    xywh_score += 1
            else:
                xywh_score += 1

        if b3 > x1 and b4 > y1:
            if width > 0 and height > 0:
                if x1 >= -1 and y1 >= -1 and b3 <= width + 1 and b4 <= height + 1:
                    xyxy_score += 1
            else:
                xyxy_score += 1

    return "xywh" if xywh_score >= xyxy_score else "xyxy"


def bbox_to_xywh(bbox: Sequence[float], bbox_format: str) -> Tuple[float, float, float, float]:
    x1, y1, b3, b4 = [float(v) for v in bbox]
    if bbox_format == "xyxy":
        return x1, y1, b3 - x1, b4 - y1
    return x1, y1, b3, b4


def load_mot_gt_by_frame(
    gt_path: Path,
) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing source gt file: {gt_path}")
    out: Dict[int, List[Tuple[int, float, float, float, float]]] = defaultdict(list)
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            parts = row.split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x, y, w, h = [float(v) for v in parts[2:6]]
            if w <= 0 or h <= 0:
                continue
            out[frame].append((track_id, x, y, w, h))
    return out


def resolve_sequence_source(raw_root: Path, sequence: str, source_dirs: Sequence[str]) -> Path:
    for source_dir in source_dirs:
        seq_dir = raw_root / source_dir / sequence
        if seq_dir.exists():
            return seq_dir
    candidates = ", ".join(str(raw_root / src / sequence) for src in source_dirs)
    raise FileNotFoundError(f"Cannot find source sequence {sequence}. Checked: {candidates}")


def try_link_or_copy_dir(src_dir: Path, dst_dir: Path, force_copy: bool) -> str:
    if dst_dir.exists():
        return "existing"

    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    src_abs = src_dir.resolve()
    dst_abs = dst_dir.resolve()

    if not force_copy:
        try:
            os.symlink(str(src_abs), str(dst_abs), target_is_directory=True)
            return "symlink"
        except OSError:
            pass

        if os.name == "nt":
            cmd = ["cmd", "/c", "mklink", "/J", str(dst_abs), str(src_abs)]
            run = subprocess.run(cmd, capture_output=True, text=True)
            if run.returncode == 0:
                return "junction"

    shutil.copytree(src_abs, dst_abs)
    return "copied"


def write_seqinfo(src_seqinfo: Path, dst_seqinfo: Path, seq_length: int, keep_seq_length: bool) -> None:
    if keep_seq_length:
        shutil.copy2(src_seqinfo, dst_seqinfo)
        return

    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(src_seqinfo, encoding="utf-8")
    if "Sequence" not in parser:
        shutil.copy2(src_seqinfo, dst_seqinfo)
        return

    parser["Sequence"]["seqLength"] = str(seq_length)
    with dst_seqinfo.open("w", encoding="utf-8", newline="\n") as f:
        parser.write(f, space_around_delimiters=False)


def iter_selected_videos(videos: List[Dict], seq_limit: int) -> List[Dict]:
    ordered = sorted(videos, key=lambda x: str(x.get("file_name", "")))
    if seq_limit > 0:
        ordered = ordered[:seq_limit]
    return ordered


def prepare_split(
    split: str,
    raw_root: Path,
    out_root: Path,
    source_dirs: Sequence[str],
    seq_limit: int,
    max_frames: int,
    clean_split: bool,
    force_copy_images: bool,
    keep_seq_length: bool,
) -> Tuple[int, int]:
    json_path = raw_root / f"{split}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing split json: {json_path}")

    data = load_json(json_path)
    videos = data.get("videos", [])
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    selected_videos = iter_selected_videos(videos, seq_limit)
    selected_video_ids = {int(v["id"]) for v in selected_videos if "id" in v}
    video_id_to_name = {int(v["id"]): str(v["file_name"]) for v in selected_videos if "id" in v}

    split_dir = out_root / split
    if clean_split and split_dir.exists():
        shutil.rmtree(split_dir)

    images_by_id_all = {int(img["id"]): img for img in images if "id" in img}
    bbox_format = infer_bbox_format(annotations, images_by_id_all)

    image_frame: Dict[int, int] = {}
    image_raw_frame: Dict[int, int] = {}
    image_video: Dict[int, int] = {}
    image_name: Dict[int, str] = {}
    frames_by_video: Dict[int, List[int]] = defaultdict(list)
    image_ids_by_video: Dict[int, List[int]] = defaultdict(list)

    for img in images:
        if "id" not in img or "video_id" not in img:
            continue
        video_id = int(img["video_id"])
        if video_id not in selected_video_ids:
            continue
        frame_id = infer_frame_id(img)
        if max_frames > 0 and frame_id > max_frames:
            continue
        image_id = int(img["id"])
        file_name = str(img.get("file_name", ""))
        image_frame[image_id] = frame_id
        image_raw_frame[image_id] = infer_raw_frame_from_file_name(file_name)
        image_video[image_id] = video_id
        image_name[image_id] = file_name
        frames_by_video[video_id].append(frame_id)
        image_ids_by_video[video_id].append(image_id)

    # Ensure MOT frame index starts from 1.
    frame_shift_by_video: Dict[int, int] = {}
    seq_length_by_video: Dict[int, int] = {}
    for video_id, frames in frames_by_video.items():
        min_frame = min(frames)
        max_frame = max(frames)
        shift = 1 if min_frame == 0 else 0
        frame_shift_by_video[video_id] = shift
        if max_frames > 0:
            # Keep the split-level timeline consistent for smoke/full runs.
            # This prevents seqLength drift from silently shortening evaluation loops.
            seq_length_by_video[video_id] = int(max_frames)
        else:
            seq_length_by_video[video_id] = max_frame + shift

    # Fallback annotations: used only if source sequence gt.txt is missing.
    ann_rows_by_image: Dict[int, List[Tuple[int, float, float, float, float]]] = defaultdict(list)
    for ann in annotations:
        image_id = ann.get("image_id")
        if image_id is None:
            continue
        image_id = int(image_id)
        if image_id not in image_video:
            continue
        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        track_id = int(ann.get("track_id", ann.get("id", 0)))
        x, y, w, h = bbox_to_xywh(bbox, bbox_format)
        if w <= 0 or h <= 0:
            continue
        ann_rows_by_image[image_id].append((track_id, x, y, w, h))

    gt_rows_by_sequence: Dict[str, List[Tuple[int, int, float, float, float, float]]] = defaultdict(list)
    written_annotations = 0
    id_source_by_seq: Dict[str, str] = {}

    for video in selected_videos:
        video_id = int(video["id"])
        sequence = str(video["file_name"])
        if video_id not in seq_length_by_video:
            continue

        seq_source = resolve_sequence_source(raw_root, sequence, source_dirs)
        src_gt_path = seq_source / "gt" / "gt.txt"
        use_source_gt = src_gt_path.exists()
        source_gt_rows: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
        if use_source_gt:
            source_gt_rows = load_mot_gt_by_frame(src_gt_path)
            id_source_by_seq[sequence] = "source_gt"
        else:
            id_source_by_seq[sequence] = "json_fallback"

        img_ids = sorted(
            image_ids_by_video.get(video_id, []),
            key=lambda iid: (image_frame.get(iid, 0), iid),
        )
        for image_id in img_ids:
            out_frame = image_frame[image_id] + frame_shift_by_video.get(video_id, 0)
            if use_source_gt:
                raw_frame = image_raw_frame[image_id]
                rows = source_gt_rows.get(raw_frame, [])
                if not rows:
                    # Defensive fallback when frame mapping is missing in source gt.
                    rows = ann_rows_by_image.get(image_id, [])
                    if rows:
                        id_source_by_seq[sequence] = "source_gt+json_fallback"
                for track_id, x, y, w, h in rows:
                    gt_rows_by_sequence[sequence].append((out_frame, track_id, x, y, w, h))
                    written_annotations += 1
            else:
                rows = ann_rows_by_image.get(image_id, [])
                for track_id, x, y, w, h in rows:
                    gt_rows_by_sequence[sequence].append((out_frame, track_id, x, y, w, h))
                    written_annotations += 1

    kept_sequences: List[str] = []
    for video in selected_videos:
        video_id = int(video["id"])
        sequence = str(video["file_name"])
        if video_id not in seq_length_by_video:
            continue
        kept_sequences.append(sequence)

        seq_source = resolve_sequence_source(raw_root, sequence, source_dirs)
        src_img_dir = seq_source / "img1"
        src_seqinfo = seq_source / "seqinfo.ini"
        if not src_img_dir.exists():
            raise FileNotFoundError(f"Missing source image directory: {src_img_dir}")
        if not src_seqinfo.exists():
            raise FileNotFoundError(f"Missing source seqinfo: {src_seqinfo}")

        seq_out_dir = split_dir / sequence
        gt_out_dir = seq_out_dir / "gt"
        gt_out_dir.mkdir(parents=True, exist_ok=True)

        link_mode = try_link_or_copy_dir(src_img_dir, seq_out_dir / "img1", force_copy_images)

        write_seqinfo(
            src_seqinfo=src_seqinfo,
            dst_seqinfo=seq_out_dir / "seqinfo.ini",
            seq_length=seq_length_by_video[video_id],
            keep_seq_length=keep_seq_length,
        )

        rows = gt_rows_by_sequence.get(sequence, [])
        rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
        gt_path = gt_out_dir / "gt.txt"
        with gt_path.open("w", encoding="utf-8", newline="\n") as f:
            for frame, track_id, x, y, w, h in rows:
                f.write(f"{frame},{track_id},{x:.3f},{y:.3f},{w:.3f},{h:.3f},1,1,1\n")

        print(
            f"[{split}] {sequence}: "
            f"frames={seq_length_by_video[video_id]}, "
            f"gt_rows={len(rows)}, id_source={id_source_by_seq.get(sequence, 'unknown')}, img1={link_mode}"
        )

    seqmaps_dir = split_dir / "seqmaps"
    seqmaps_dir.mkdir(parents=True, exist_ok=True)
    seqmap_path = seqmaps_dir / f"{split}.txt"
    with seqmap_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("name\n")
        for seq in kept_sequences:
            f.write(f"{seq}\n")

    print(
        f"[{split}] done: sequences={len(kept_sequences)}, "
        f"annotations={written_annotations}, bbox_format={bbox_format}, seqmap={seqmap_path}"
    )
    return len(kept_sequences), written_annotations


def main() -> None:
    args = parse_args()
    apply_run_config(args)
    total_sequences = 0
    total_annotations = 0

    for split in args.splits:
        seq_count, ann_count = prepare_split(
            split=split,
            raw_root=args.raw_root,
            out_root=args.out_root,
            source_dirs=args.source_dirs,
            seq_limit=args.seq_limit,
            max_frames=args.max_frames,
            clean_split=args.clean_split,
            force_copy_images=args.force_copy_images,
            keep_seq_length=args.keep_seq_length,
        )
        total_sequences += seq_count
        total_annotations += ann_count

    print(
        f"Prepared {len(args.splits)} split(s): "
        f"total_sequences={total_sequences}, total_annotations={total_annotations}"
    )


if __name__ == "__main__":
    main()
