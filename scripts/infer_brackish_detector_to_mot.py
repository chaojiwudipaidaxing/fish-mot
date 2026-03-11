#!/usr/bin/env python
"""Run detector inference on BrackishMOT and export MOTChallenge det.txt files.

Output line format (strict MOT det convention):
  frame,-1,x,y,w,h,score,-1,-1,-1

This matches scripts/run_baseline_sort.py loader expectations:
  - reads frame from col 0
  - reads x,y,w,h from cols 2..5
  - ignores remaining columns
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class SeqInfo:
    name: str
    seq_length: int
    width: int
    height: int
    img_dir: Path
    img_ext: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer BrackishMOT detector outputs and export MOT det.txt.")
    parser.add_argument(
        "--brackish-root",
        type=Path,
        default=Path("shuju/archive/BrackishMOT"),
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--out-det-subdir", type=str, default="det")
    parser.add_argument("--max-seqs", type=int, default=0, help="Optional debug cap; 0 means all sequences.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional debug cap per sequence; 0 means full length.")
    parser.add_argument(
        "--meta-out",
        type=Path,
        default=Path("results/brackishmot_detector_infer_meta.json"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_seqinfo(path: Path) -> SeqInfo:
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    if "Sequence" not in parser:
        raise RuntimeError(f"Missing [Sequence] in {path}")
    s = parser["Sequence"]
    seq_dir = path.parent
    name = str(s.get("name", seq_dir.name))
    img_dir_name = str(s.get("imDir", "img1"))
    img_ext = str(s.get("imExt", ".jpg"))
    seq_length = int(s.get("seqLength", s.get("seqlength", "0")))
    width = int(s.get("imWidth", "0"))
    height = int(s.get("imHeight", "0"))
    return SeqInfo(
        name=name,
        seq_length=seq_length,
        width=width,
        height=height,
        img_dir=seq_dir / img_dir_name,
        img_ext=img_ext,
    )


def sorted_image_list(info: SeqInfo, max_frames: int) -> List[Path]:
    if not info.img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {info.img_dir}")
    imgs = sorted([p for p in info.img_dir.glob(f"*{info.img_ext}") if p.is_file()])
    if not imgs:
        imgs = sorted([p for p in info.img_dir.glob("*") if p.is_file()])
    if info.seq_length > 0:
        imgs = imgs[: info.seq_length]
    if max_frames > 0:
        imgs = imgs[:max_frames]
    return imgs


def clamp_xyxy_to_xywh(xyxy: List[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    x1 = max(0.0, min(x1, float(width) - 1.0))
    y1 = max(0.0, min(y1, float(height) - 1.0))
    x2 = max(0.0, min(x2, float(width)))
    y2 = max(0.0, min(y2, float(height)))
    w = x2 - x1
    h = y2 - y1
    if w <= 0.0 or h <= 0.0:
        return None
    return x1, y1, w, h


def main() -> int:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    split_dir = args.brackish_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split path not found: {split_dir}")

    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to import ultralytics/torch. Install with: "
            "pip install ultralytics opencv-python torch"
        ) from exc

    model = YOLO(str(args.weights))
    weight_sha = sha256_file(args.weights)
    detector_name = f"{args.weights.stem}_{weight_sha[:8]}"

    seq_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir() and p.name.startswith("brackishMOT-")])
    if args.max_seqs > 0:
        seq_dirs = seq_dirs[: args.max_seqs]
    if not seq_dirs:
        raise RuntimeError(f"No sequences found under {split_dir}")

    total_frames = 0
    total_dets = 0
    seq_meta: List[Dict[str, object]] = []

    t0 = time.perf_counter()
    for seq_dir in seq_dirs:
        info = read_seqinfo(seq_dir / "seqinfo.ini")
        image_paths = sorted_image_list(info, max_frames=args.max_frames)
        out_dir = seq_dir / args.out_det_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        det_txt = out_dir / "det.txt"

        seq_det_count = 0
        with det_txt.open("w", encoding="utf-8", newline="\n") as f:
            for frame_idx, img_path in enumerate(image_paths, start=1):
                results = model.predict(
                    source=str(img_path),
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                    verbose=False,
                )
                if not results:
                    continue
                result = results[0]
                boxes = getattr(result, "boxes", None)
                if boxes is None or len(boxes) == 0:
                    continue

                xyxy_arr = boxes.xyxy.cpu().numpy()
                conf_arr = boxes.conf.cpu().numpy()
                for xyxy, score in zip(xyxy_arr.tolist(), conf_arr.tolist()):
                    xywh = clamp_xyxy_to_xywh(xyxy, width=info.width, height=info.height)
                    if xywh is None:
                        continue
                    x, y, w, h = xywh
                    # MOT det convention: frame,-1,x,y,w,h,score,-1,-1,-1
                    f.write(f"{frame_idx},-1,{x:.3f},{y:.3f},{w:.3f},{h:.3f},{float(score):.6f},-1,-1,-1\n")
                    seq_det_count += 1

        total_frames += len(image_paths)
        total_dets += seq_det_count
        seq_meta.append(
            {
                "sequence": info.name,
                "frames": len(image_paths),
                "detections": seq_det_count,
                "det_path": str(det_txt),
                "image_dir": str(info.img_dir),
            }
        )
        print(f"[infer] {info.name}: frames={len(image_paths)} detections={seq_det_count} -> {det_txt}")

    elapsed = max(time.perf_counter() - t0, 1e-9)
    fps = float(total_frames / elapsed)

    meta = {
        "detector_name": detector_name,
        "weights_path": str(args.weights),
        "weights_sha256": weight_sha,
        "brackish_root": str(args.brackish_root),
        "split": args.split,
        "out_det_subdir": args.out_det_subdir,
        "imgsz": int(args.imgsz),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "device": str(args.device),
        "ultralytics_version": str(ultralytics.__version__),
        "torch_version": str(torch.__version__),
        "total_sequences": len(seq_meta),
        "total_frames": int(total_frames),
        "total_detections": int(total_dets),
        "elapsed_sec": float(elapsed),
        "mean_infer_fps": float(fps),
        "format_contract": {
            "mot_line": "frame,-1,x,y,w,h,score,-1,-1,-1",
            "run_baseline_sort_loader": "uses columns [0]=frame and [2:6]=x,y,w,h",
        },
        "per_sequence": seq_meta,
    }

    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    args.meta_out.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[infer] meta written: {args.meta_out}")
    print(f"[infer] overall: sequences={len(seq_meta)} frames={total_frames} dets={total_dets} fps={fps:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
