#!/usr/bin/env python
"""Audit BrackishMOT structure and visibility proxies for the framework's field-proxy stress benchmark."""

from __future__ import annotations

import argparse
import configparser
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit BrackishMOT and write JSON report.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("shuju/archive/BrackishMOT"),
        help="BrackishMOT root directory containing train/test folders.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/brackishmot_audit.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=5,
        help="Number of image frames sampled per sequence for visibility proxies.",
    )
    parser.add_argument(
        "--log-out",
        type=Path,
        default=Path("logs/brackishmot_audit.txt"),
        help="Human-readable audit summary text path.",
    )
    return parser.parse_args()


def read_seqinfo(path: Path) -> Dict[str, Any]:
    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8")
    if "Sequence" not in cfg:
        raise RuntimeError(f"Missing [Sequence] section in {path}")
    seq = cfg["Sequence"]
    out: Dict[str, Any] = {}
    for k in seq:
        v = seq[k]
        if k in {"framerate", "seqlength", "imwidth", "imheight"}:
            out[k] = int(v)
        else:
            out[k] = v
    return out


def sample_indices(n: int, k: int) -> List[int]:
    if n <= 0:
        return []
    k = max(1, min(k, n))
    return sorted(set(np.linspace(0, n - 1, num=k, dtype=int).tolist()))


def visibility_proxy(image_path: Path) -> Dict[str, float]:
    with Image.open(image_path) as im:
        gray = np.asarray(im.convert("L"), dtype=np.float32)
    contrast = float(gray.std())
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    sharpness = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))
    dark_ratio = float(np.mean(gray < 40.0))
    bright_ratio = float(np.mean(gray > 215.0))
    return {
        "contrast_std": contrast,
        "sharpness_grad_l1": sharpness,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
    }


def summarize_visibility(image_paths: List[Path], k: int) -> Dict[str, float]:
    idxs = sample_indices(len(image_paths), k=k)
    if not idxs:
        return {
            "contrast_std_mean": 0.0,
            "sharpness_grad_l1_mean": 0.0,
            "dark_ratio_mean": 0.0,
            "bright_ratio_mean": 0.0,
            "quality_score": 0.0,
        }
    vals = [visibility_proxy(image_paths[i]) for i in idxs]
    contrast_mean = float(np.mean([v["contrast_std"] for v in vals]))
    sharp_mean = float(np.mean([v["sharpness_grad_l1"] for v in vals]))
    dark_mean = float(np.mean([v["dark_ratio"] for v in vals]))
    bright_mean = float(np.mean([v["bright_ratio"] for v in vals]))
    # quality_score is a deterministic proxy for clear/reference ranking.
    quality_score = float((contrast_mean * sharp_mean) / (1.0 + 2.0 * dark_mean))
    return {
        "contrast_std_mean": contrast_mean,
        "sharpness_grad_l1_mean": sharp_mean,
        "dark_ratio_mean": dark_mean,
        "bright_ratio_mean": bright_mean,
        "quality_score": quality_score,
    }


def parse_gt(gt_path: Path) -> Dict[str, Any]:
    if not gt_path.exists():
        return {"exists": False, "rows": 0, "unique_ids": 0, "column_count": 0, "format_hint": "missing"}
    rows = 0
    ids = set()
    col_count = 0
    with gt_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            rows += 1
            col_count = max(col_count, len(row))
            try:
                ids.add(int(float(row[1])))
            except Exception:
                pass
    format_hint = "MOTChallenge-like (frame,id,x,y,w,h,conf,class,vis)" if col_count >= 9 else "unknown"
    return {
        "exists": True,
        "rows": rows,
        "unique_ids": len(ids),
        "column_count": col_count,
        "format_hint": format_hint,
    }


def audit_sequence(split: str, seq_dir: Path, sample_frames: int) -> Dict[str, Any]:
    seqinfo = read_seqinfo(seq_dir / "seqinfo.ini")
    img_dir = seq_dir / str(seqinfo.get("imdir", "img1"))
    img_paths = sorted([p for p in img_dir.glob("*") if p.is_file()])
    gt_info = parse_gt(seq_dir / "gt" / "gt.txt")
    vis = summarize_visibility(img_paths, k=sample_frames)
    return {
        "split": split,
        "name": seq_dir.name,
        "path": str(seq_dir),
        "seqinfo": seqinfo,
        "images_count": len(img_paths),
        "image_example_format": img_paths[0].suffix.lower() if img_paths else "",
        "gt": gt_info,
        "visibility_proxy": vis,
    }


def main() -> int:
    args = parse_args()
    root = args.root
    if not root.exists():
        raise FileNotFoundError(f"BrackishMOT root not found: {root}")

    report: Dict[str, Any] = {
        "root": str(root.resolve()),
        "readable": True,
        "splits": {},
        "sequence_count_total": 0,
        "sequence_audit": [],
    }

    for split in ("train", "test"):
        split_dir = root / split
        if not split_dir.exists():
            report["splits"][split] = {"exists": False, "sequence_count": 0, "sequences": []}
            continue
        seq_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
        report["splits"][split] = {
            "exists": True,
            "sequence_count": len(seq_dirs),
            "sequences": [p.name for p in seq_dirs],
        }
        report["sequence_count_total"] += len(seq_dirs)
        for seq_dir in seq_dirs:
            report["sequence_audit"].append(audit_sequence(split, seq_dir, sample_frames=args.sample_frames))

    aud = report["sequence_audit"]
    aud_sorted = sorted(aud, key=lambda x: x["visibility_proxy"]["quality_score"], reverse=True)
    report["reference_clear_sequences"] = [
        {
            "split": x["split"],
            "name": x["name"],
            "quality_score": x["visibility_proxy"]["quality_score"],
            "contrast_std_mean": x["visibility_proxy"]["contrast_std_mean"],
            "sharpness_grad_l1_mean": x["visibility_proxy"]["sharpness_grad_l1_mean"],
        }
        for x in aud_sorted[: min(8, len(aud_sorted))]
    ]
    report["high_turbidity_candidate_sequences"] = [
        {
            "split": x["split"],
            "name": x["name"],
            "quality_score": x["visibility_proxy"]["quality_score"],
            "contrast_std_mean": x["visibility_proxy"]["contrast_std_mean"],
            "dark_ratio_mean": x["visibility_proxy"]["dark_ratio_mean"],
        }
        for x in list(reversed(aud_sorted[-min(8, len(aud_sorted)) :]))
    ]
    report["selection_rule"] = (
        "reference_clear_sequences: top quality_score by sampled contrast/sharpness; "
        "high_turbidity_candidate_sequences: bottom quality_score."
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.log_out.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("BrackishMOT audit summary")
    lines.append(f"root: {report['root']}")
    lines.append(f"readable: {report['readable']}")
    lines.append("")
    for split in ("train", "test"):
        info = report["splits"].get(split, {})
        lines.append(f"[split={split}] exists={info.get('exists', False)} sequence_count={info.get('sequence_count', 0)}")
    lines.append("")
    lines.append("Top clear candidates:")
    for x in report["reference_clear_sequences"][:8]:
        lines.append(
            f"  - {x['split']}/{x['name']}: quality={x['quality_score']:.6f}, "
            f"contrast={x['contrast_std_mean']:.6f}, sharpness={x['sharpness_grad_l1_mean']:.6f}"
        )
    lines.append("")
    lines.append("Bottom turbidity candidates:")
    for x in report["high_turbidity_candidate_sequences"][:8]:
        lines.append(
            f"  - {x['split']}/{x['name']}: quality={x['quality_score']:.6f}, "
            f"contrast={x['contrast_std_mean']:.6f}, dark_ratio={x['dark_ratio_mean']:.6f}"
        )
    lines.append("")
    lines.append("Per-sequence availability (first 20):")
    for row in report["sequence_audit"][:20]:
        seqinfo = row["seqinfo"]
        seq_len = seqinfo.get("seqlength", seqinfo.get("seqLength", 0))
        im_w = seqinfo.get("imwidth", seqinfo.get("imWidth", "")
)
        im_h = seqinfo.get("imheight", seqinfo.get("imHeight", ""))
        lines.append(
            f"  - {row['split']}/{row['name']}: len={seq_len}, res={im_w}x{im_h}, "
            f"gt_exists={row['gt']['exists']}, gt_rows={row['gt']['rows']}, "
            f"img_ext={row['image_example_format']}"
        )
    args.log_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved audit report: {args.out}")
    print(f"saved audit log: {args.log_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
