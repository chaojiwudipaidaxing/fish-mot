#!/usr/bin/env python
"""Inspect MFT25 annotation JSON structure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MFT25 split JSON files.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/mft25_raw"),
        help="Root directory that contains train_half.json and val_half.json.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train_half", "val_half"],
        help="Splits to inspect.",
    )
    parser.add_argument(
        "--bbox-samples",
        type=int,
        default=5000,
        help="Max annotation samples used for bbox format inference.",
    )
    return parser.parse_args()


def infer_bbox_format(
    annotations: List[Dict], image_by_id: Dict[int, Dict], max_samples: int
) -> Tuple[str, int, int]:
    """Infer whether bbox looks like xywh or xyxy."""
    xywh_score = 0
    xyxy_score = 0
    checked = 0

    for ann in annotations:
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

        checked += 1
        if checked >= max_samples:
            break

    if xywh_score == 0 and xyxy_score == 0:
        return "unknown", xywh_score, xyxy_score
    if xywh_score >= xyxy_score:
        return "xywh", xywh_score, xyxy_score
    return "xyxy", xywh_score, xyxy_score


def summarize_keys(data: Dict) -> Iterable[str]:
    for key in data.keys():
        value = data[key]
        if isinstance(value, list):
            yield f"- {key}: list[{len(value)}]"
        else:
            yield f"- {key}: {type(value).__name__}"


def inspect_split(raw_root: Path, split: str, bbox_samples: int) -> None:
    json_path = raw_root / f"{split}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Split file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    image_by_id = {int(img["id"]): img for img in images if "id" in img}

    annotation_fields = sorted(
        {
            key
            for ann in annotations
            if isinstance(ann, dict)
            for key in ann.keys()
        }
    )
    image_fields = sorted(
        {
            key
            for img in images
            if isinstance(img, dict)
            for key in img.keys()
        }
    )

    bbox_format, xywh_score, xyxy_score = infer_bbox_format(
        annotations=annotations, image_by_id=image_by_id, max_samples=bbox_samples
    )

    frame_ids = [int(img["frame_id"]) for img in images if "frame_id" in img]
    frame_min = min(frame_ids) if frame_ids else None
    frame_max = max(frame_ids) if frame_ids else None

    print("=" * 80)
    print(f"Split: {split}")
    print(f"JSON:  {json_path}")
    print("Top-level structure:")
    for line in summarize_keys(data):
        print(line)
    print(f"Image fields: {image_fields}")
    print(f"Annotation fields: {annotation_fields}")
    print(
        f"BBox format guess: {bbox_format} "
        f"(xywh_score={xywh_score}, xyxy_score={xyxy_score})"
    )

    if frame_min is None:
        print("Frame index: frame_id not found in images")
    else:
        start_text = (
            "starts from 0"
            if frame_min == 0
            else "starts from 1"
            if frame_min == 1
            else f"starts from {frame_min}"
        )
        print(f"Frame index: min={frame_min}, max={frame_max} -> {start_text}")

    if annotations:
        print(f"Sample annotation: {annotations[0]}")
    if images:
        print(f"Sample image: {images[0]}")


def main() -> None:
    args = parse_args()
    for split in args.splits:
        inspect_split(args.raw_root, split, args.bbox_samples)


if __name__ == "__main__":
    main()
