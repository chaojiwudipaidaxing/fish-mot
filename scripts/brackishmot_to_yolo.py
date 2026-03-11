#!/usr/bin/env python
"""Convert BrackishMOT MOTChallenge annotations to YOLO detection dataset."""

from __future__ import annotations

import argparse
import configparser
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass
class SeqInfo:
    split: str
    name: str
    seq_dir: Path
    img_dir: Path
    img_ext: str
    width: int
    height: int
    seq_length: int
    gt_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert BrackishMOT to YOLO format.")
    parser.add_argument(
        "--brackish-root",
        type=Path,
        default=Path("shuju/archive/BrackishMOT"),
    )
    parser.add_argument("--out", type=Path, default=Path("data/brackish_yolo"))
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--class-mode", choices=["single"], default="single")
    parser.add_argument(
        "--audit-out",
        type=Path,
        default=Path("results/brackish_yolo_build.json"),
    )
    return parser.parse_args()


def read_seqinfo(path: Path, split: str) -> SeqInfo:
    cp = configparser.ConfigParser()
    cp.read(path, encoding="utf-8")
    if "Sequence" not in cp:
        raise RuntimeError(f"Missing [Sequence] in {path}")
    s = cp["Sequence"]
    seq_dir = path.parent
    name = str(s.get("name", seq_dir.name))
    img_dir = seq_dir / str(s.get("imDir", "img1"))
    img_ext = str(s.get("imExt", ".jpg"))
    width = int(s.get("imWidth", "0"))
    height = int(s.get("imHeight", "0"))
    seq_length = int(s.get("seqLength", s.get("seqlength", "0")))
    gt_path = seq_dir / "gt" / "gt.txt"
    return SeqInfo(split, name, seq_dir, img_dir, img_ext, width, height, seq_length, gt_path)


def read_gt_boxes(gt_path: Path) -> Dict[int, List[Tuple[float, float, float, float]]]:
    out: Dict[int, List[Tuple[float, float, float, float]]] = defaultdict(list)
    if not gt_path.exists():
        return out
    with gt_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            parts = row.split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            x, y, w, h = [float(v) for v in parts[2:6]]
            if w <= 0 or h <= 0:
                continue
            # MOTChallenge-style conf column usually at index 6.
            if len(parts) >= 7:
                try:
                    conf = float(parts[6])
                except Exception:
                    conf = 1.0
                if conf <= 0:
                    continue
            out[frame].append((x, y, w, h))
    return out


def yolo_box(x: float, y: float, w: float, h: float, iw: int, ih: int) -> Tuple[float, float, float, float] | None:
    if iw <= 0 or ih <= 0:
        return None
    x1 = max(0.0, min(x, iw - 1.0))
    y1 = max(0.0, min(y, ih - 1.0))
    w = max(0.0, min(w, iw - x1))
    h = max(0.0, min(h, ih - y1))
    if w <= 0.0 or h <= 0.0:
        return None
    cx = (x1 + 0.5 * w) / float(iw)
    cy = (y1 + 0.5 * h) / float(ih)
    nw = w / float(iw)
    nh = h / float(ih)
    return cx, cy, nw, nh


def try_hardlink(src: Path, dst: Path) -> bool:
    try:
        if dst.exists():
            dst.unlink()
        dst.hardlink_to(src)
        return True
    except Exception:
        return False


def image_stem(seq_name: str, frame_idx: int) -> str:
    return f"{seq_name}__{frame_idx:06d}"


def iter_images(info: SeqInfo) -> Iterable[Tuple[int, Path]]:
    imgs = sorted([p for p in info.img_dir.glob(f"*{info.img_ext}") if p.is_file()])
    if not imgs:
        imgs = sorted([p for p in info.img_dir.glob("*") if p.is_file()])
    if info.seq_length > 0:
        imgs = imgs[: info.seq_length]
    for idx, p in enumerate(imgs, start=1):
        yield idx, p


def main() -> int:
    args = parse_args()
    if not args.brackish_root.exists():
        raise FileNotFoundError(f"Brackish root not found: {args.brackish_root}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "labels").mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Dict[str, object]] = {}
    global_seq_count = 0
    global_frame_count = 0
    global_box_count = 0
    hardlink_count = 0
    copy_count = 0
    sample_examples: List[Dict[str, object]] = []

    for split in args.splits:
        split_dir = args.brackish_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split dir missing: {split_dir}")
        img_out = args.out / "images" / split
        lbl_out = args.out / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        seq_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir() and p.name.startswith("brackishMOT-")])
        split_seq_count = 0
        split_frame_count = 0
        split_box_count = 0
        split_examples: List[Dict[str, object]] = []

        for seq_dir in seq_dirs:
            seqinfo = read_seqinfo(seq_dir / "seqinfo.ini", split=split)
            gt_map = read_gt_boxes(seqinfo.gt_path)
            split_seq_count += 1
            global_seq_count += 1

            for frame_idx, src_img in iter_images(seqinfo):
                stem = image_stem(seqinfo.name, frame_idx)
                dst_img = img_out / f"{stem}{src_img.suffix.lower()}"
                dst_lbl = lbl_out / f"{stem}.txt"

                if not try_hardlink(src_img, dst_img):
                    shutil.copy2(src_img, dst_img)
                    copy_count += 1
                else:
                    hardlink_count += 1

                frame_boxes = gt_map.get(frame_idx, [])
                yolo_lines: List[str] = []
                for (x, y, w, h) in frame_boxes:
                    y = yolo_box(x, y, w, h, seqinfo.width, seqinfo.height)
                    if y is None:
                        continue
                    cx, cy, nw, nh = y
                    # single-class fish => class 0
                    yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                dst_lbl.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8", newline="\n")

                split_frame_count += 1
                global_frame_count += 1
                split_box_count += len(yolo_lines)
                global_box_count += len(yolo_lines)

                if len(split_examples) < 3 and yolo_lines:
                    split_examples.append(
                        {
                            "seq": seqinfo.name,
                            "frame": frame_idx,
                            "image": str(dst_img),
                            "label": str(dst_lbl),
                            "boxes": len(yolo_lines),
                            "first_label": yolo_lines[0],
                        }
                    )
                    if len(sample_examples) < 10:
                        sample_examples.append(split_examples[-1])

        stats[split] = {
            "sequence_count": split_seq_count,
            "frame_count": split_frame_count,
            "box_count": split_box_count,
            "examples": split_examples,
            "images_dir": str(img_out),
            "labels_dir": str(lbl_out),
        }

    yaml_text = (
        f"path: {args.out.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/test\n"
        f"test: images/test\n"
        f"names:\n"
        f"  0: fish\n"
        f"nc: 1\n"
    )
    yaml_path = args.out / "brackish_yolo.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8", newline="\n")

    audit = {
        "dataset_name": "BrackishMOT->YOLO",
        "brackish_root": str(args.brackish_root),
        "out_root": str(args.out),
        "yaml_path": str(yaml_path),
        "class_mode": args.class_mode,
        "splits": stats,
        "total_sequences": global_seq_count,
        "total_frames": global_frame_count,
        "total_boxes": global_box_count,
        "image_materialization": {
            "hardlink_count": hardlink_count,
            "copy_count": copy_count,
        },
        "sample_examples": sample_examples,
    }
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[yolo-build] wrote yaml: {yaml_path}")
    print(f"[yolo-build] wrote audit: {args.audit_out}")
    print(
        "[yolo-build] totals:",
        f"sequences={global_seq_count}",
        f"frames={global_frame_count}",
        f"boxes={global_box_count}",
        f"hardlinks={hardlink_count}",
        f"copies={copy_count}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
