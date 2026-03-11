#!/usr/bin/env python
"""Select audited BrackishMOT clear / turbid-low / turbid-high groups for turbidity-induced drift evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select BrackishMOT groups from audit JSON.")
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path("results/brackishmot_audit.json"),
    )
    parser.add_argument(
        "--groups-json",
        type=Path,
        default=Path("results/brackishmot_groups.json"),
    )
    parser.add_argument(
        "--drift-scenarios-json",
        type=Path,
        default=Path("results/brackishmot_drift_scenarios.json"),
    )
    parser.add_argument(
        "--log-out",
        type=Path,
        default=Path("logs/brackishmot_pairing.txt"),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
    )
    return parser.parse_args()


def pick_groups(rows: List[Dict[str, Any]], group_size: int) -> Dict[str, List[Dict[str, Any]]]:
    rows_desc = sorted(rows, key=lambda r: float(r["visibility_proxy"]["quality_score"]), reverse=True)
    clear = rows_desc[:group_size]

    rows_asc = list(reversed(rows_desc))
    turbid_high = rows_asc[:group_size]

    n = len(rows_desc)
    center = int(round(n * 0.35))
    start = max(0, center - group_size // 2)
    turbid_low: List[Dict[str, Any]] = []
    used = {x["name"] for x in clear} | {x["name"] for x in turbid_high}
    idx = start
    while idx < n and len(turbid_low) < group_size:
        cand = rows_desc[idx]
        if cand["name"] not in used:
            turbid_low.append(cand)
            used.add(cand["name"])
        idx += 1
    if len(turbid_low) < group_size:
        for cand in rows_desc:
            if cand["name"] in used:
                continue
            turbid_low.append(cand)
            used.add(cand["name"])
            if len(turbid_low) >= group_size:
                break

    return {
        "clear": clear,
        "turbid_low": turbid_low[:group_size],
        "turbid_high": turbid_high[:group_size],
    }


def compact(rows: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        seq_len = int(r["seqinfo"].get("seqlength", r["seqinfo"].get("seqLength", 0)))
        vis = r["visibility_proxy"]
        out.append(
            {
                "split": split,
                "name": r["name"],
                "seq_length": seq_len,
                "quality_score": float(vis["quality_score"]),
                "contrast_std_mean": float(vis["contrast_std_mean"]),
                "sharpness_grad_l1_mean": float(vis["sharpness_grad_l1_mean"]),
                "dark_ratio_mean": float(vis["dark_ratio_mean"]),
                "bright_ratio_mean": float(vis["bright_ratio_mean"]),
            }
        )
    return out


def write_log(path: Path, groups: Dict[str, List[Dict[str, Any]]], split: str) -> None:
    lines: List[str] = []
    lines.append("BrackishMOT sequence pairing for drift/stress main results")
    lines.append(f"split: {split}")
    lines.append("selection rule:")
    lines.append("- clear: top quality_score sequences (higher contrast/sharpness, lower darkness)")
    lines.append("- turbid_low: lower-middle quality_score band (mild natural turbidity)")
    lines.append("- turbid_high: bottom quality_score sequences (severe natural turbidity)")
    lines.append("")
    for gname in ["clear", "turbid_low", "turbid_high"]:
        lines.append(f"[{gname}]")
        lines.append("name\tquality_score\tcontrast\tsharpness\tdark_ratio\tseq_length")
        for r in groups[gname]:
            vis = r["visibility_proxy"]
            seq_len = int(r["seqinfo"].get("seqlength", r["seqinfo"].get("seqLength", 0)))
            lines.append(
                f"{r['name']}\t{float(vis['quality_score']):.6f}\t{float(vis['contrast_std_mean']):.6f}\t"
                f"{float(vis['sharpness_grad_l1_mean']):.6f}\t{float(vis['dark_ratio_mean']):.6f}\t{seq_len}"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.audit_json.exists():
        raise FileNotFoundError(f"Audit JSON not found: {args.audit_json}")
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    rows = [r for r in audit.get("sequence_audit", []) if str(r.get("split")) == args.split]
    if len(rows) < args.group_size * 3:
        raise RuntimeError(f"Not enough sequences in split={args.split} for 3 groups of size={args.group_size}")

    groups_raw = pick_groups(rows, group_size=args.group_size)
    write_log(args.log_out, groups_raw, split=args.split)

    groups_compact = {
        "split": args.split,
        "clear": compact(groups_raw["clear"], args.split),
        "turbid_low": compact(groups_raw["turbid_low"], args.split),
        "turbid_high": compact(groups_raw["turbid_high"], args.split),
    }
    args.groups_json.parent.mkdir(parents=True, exist_ok=True)
    args.groups_json.write_text(json.dumps(groups_compact, indent=2, ensure_ascii=False), encoding="utf-8")

    drift_scenarios = {
        "scenarios": [
            {
                "name": "clean_control",
                "kind": "clean",
                "is_drift": False,
                "drop_rate": 0.00,
                "jitter": 0.00,
                "seqs": [r["name"] for r in groups_compact["clear"]],
                "pred_base_dir": "results/brackishmot/drift/clean_control/base/pred",
                "pred_gating_dir": "results/brackishmot/drift/clean_control/gating/pred",
            },
            {
                "name": "drop_jitter_high",
                "kind": "environmental",
                "is_drift": True,
                "drop_rate": 0.12,
                "jitter": 0.02,
                "seqs": [r["name"] for r in groups_compact["turbid_low"]],
                "pred_base_dir": "results/brackishmot/drift/turbid_low/base/pred",
                "pred_gating_dir": "results/brackishmot/drift/turbid_low/gating/pred",
            },
            {
                "name": "low_light_high",
                "kind": "environmental",
                "is_drift": True,
                "drop_rate": 0.28,
                "jitter": 0.04,
                "seqs": [r["name"] for r in groups_compact["turbid_high"]],
                "pred_base_dir": "results/brackishmot/drift/turbid_high/base/pred",
                "pred_gating_dir": "results/brackishmot/drift/turbid_high/gating/pred",
            },
        ]
    }
    args.drift_scenarios_json.parent.mkdir(parents=True, exist_ok=True)
    args.drift_scenarios_json.write_text(json.dumps(drift_scenarios, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote groups: {args.groups_json}")
    print(f"wrote scenarios: {args.drift_scenarios_json}")
    print(f"wrote log: {args.log_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
