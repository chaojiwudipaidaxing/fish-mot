#!/usr/bin/env python
"""Run BrackishMOT Scope-B end-to-end profiling (clear vs turbid_high)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import run_scopeb_profile as scopeb


SCENARIO_ORDER = ["clear", "turbid_high"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BrackishMOT Scope-B runtime profiling wrapper.")
    parser.add_argument("--groups-json", type=Path, default=Path("results/brackishmot_groups.json"))
    parser.add_argument("--mot-root", type=Path, default=Path("shuju/archive/BrackishMOT"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--scenarios", nargs="+", default=SCENARIO_ORDER)
    parser.add_argument("--methods", nargs="+", default=["Base", "+gating"])
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--jitter", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--det-source", type=str, choices=["auto", "det", "gt"], default="auto")
    parser.add_argument("--gating-thresh", type=float, default=2000.0)
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--traj-window", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--traj-encoder", type=Path, default=Path("runs/traj_encoder/traj_encoder.pt"))
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/brackishmot/runtime/runtime_profile_e2e.csv"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("results/brackishmot/runtime/runtime_profile_e2e_summary.csv"),
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=Path("results/brackishmot/runtime/pred"),
    )
    return parser.parse_args()


def _load_groups(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing groups JSON: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, List[str]] = {}
    for key in ["clear", "turbid_low", "turbid_high"]:
        vals = data.get(key, [])
        out[key] = [str(v["name"]) if isinstance(v, dict) else str(v) for v in vals]
    split = str(data.get("split", "")).strip()
    if split:
        out["_split"] = [split]
    return out


def _write_csv(path: Path, fields: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    groups = _load_groups(args.groups_json)
    split = groups.get("_split", [args.split])[0]
    if split != args.split:
        print(f"[scopeB-brackish] override split by groups-json: {args.split} -> {split}")
    args.split = split

    method_map = {m.name: m for m in scopeb.METHODS}
    selected_methods = [method_map[m] for m in args.methods if m in method_map]
    if not selected_methods:
        raise ValueError(f"No valid methods selected from {args.methods}; available={list(method_map)}")

    run_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    base_pred_root = Path(args.pred_root)
    for scenario in args.scenarios:
        seqs = groups.get(scenario, [])
        if not seqs:
            print(f"[scopeB-brackish] skip empty scenario: {scenario}")
            continue
        print(f"[scopeB-brackish] scenario={scenario} split={args.split} seqs={','.join(seqs)}")
        args.seqs = seqs
        args.pred_root = base_pred_root / scenario

        for method in selected_methods:
            traj_model = None
            rep_rows: List[Dict[str, float | str]] = []
            for warm_idx in range(args.warmup_iters):
                _ = scopeb.run_one(args=args, method=method, run_idx=-(warm_idx + 1), traj_model=traj_model)

            for rep_idx in range(args.repeat_runs):
                row = scopeb.run_one(args=args, method=method, run_idx=(rep_idx + 1), traj_model=traj_model)
                row = dict(row)
                row["scenario"] = scenario
                row["time_unit"] = "sec_per_run"
                rep_rows.append(row)
                run_rows.append(row)

            def gather(k: str) -> List[float]:
                return [float(rr[k]) for rr in rep_rows]

            mean_fps, std_fps = scopeb.mean_std(gather("fps_e2e"))
            mean_mem, std_mem = scopeb.mean_std(gather("mem_peak_mb_e2e"))
            mean_cpu, std_cpu = scopeb.mean_std(gather("cpu_norm_e2e"))
            mean_dec, std_dec = scopeb.mean_std(gather("decode_time"))
            mean_det, std_det = scopeb.mean_std(gather("detector_time"))
            mean_trk, std_trk = scopeb.mean_std(gather("tracking_time"))
            mean_wrt, std_wrt = scopeb.mean_std(gather("write_time"))
            mean_elapsed, _ = scopeb.mean_std(gather("elapsed_e2e_sec"))
            mean_frames, _ = scopeb.mean_std(gather("total_frames"))

            summary_rows.append(
                {
                    "scenario": scenario,
                    "method": method.name,
                    "split": args.split,
                    "warmup_iters": args.warmup_iters,
                    "repeat_runs": args.repeat_runs,
                    "time_unit": "sec_per_run",
                    "total_frames": int(round(mean_frames)),
                    "fps_e2e": mean_fps,
                    "fps_e2e_std": std_fps,
                    "mem_peak_mb_e2e": mean_mem,
                    "mem_peak_mb_e2e_std": std_mem,
                    "cpu_norm_e2e": mean_cpu,
                    "cpu_norm_e2e_std": std_cpu,
                    "decode_time": mean_dec,
                    "decode_time_std": std_dec,
                    "detector_time": mean_det,
                    "detector_time_std": std_det,
                    "tracking_time": mean_trk,
                    "tracking_time_std": std_trk,
                    "write_time": mean_wrt,
                    "write_time_std": std_wrt,
                    "elapsed_e2e_sec": mean_elapsed,
                    "detector_name": str(rep_rows[0]["detector_name"]),
                    "input_resolution": str(rep_rows[0]["input_resolution"]),
                }
            )

    if not run_rows:
        raise RuntimeError("No runtime rows generated.")

    run_fields = [
        "scenario",
        "row_type",
        "method",
        "run_idx",
        "split",
        "time_unit",
        "total_frames",
        "fps_e2e",
        "mem_peak_mb_e2e",
        "cpu_norm_e2e",
        "decode_time",
        "detector_time",
        "tracking_time",
        "write_time",
        "elapsed_e2e_sec",
        "detector_name",
        "input_resolution",
    ]
    sum_fields = [
        "scenario",
        "method",
        "split",
        "warmup_iters",
        "repeat_runs",
        "time_unit",
        "total_frames",
        "fps_e2e",
        "fps_e2e_std",
        "mem_peak_mb_e2e",
        "mem_peak_mb_e2e_std",
        "cpu_norm_e2e",
        "cpu_norm_e2e_std",
        "decode_time",
        "decode_time_std",
        "detector_time",
        "detector_time_std",
        "tracking_time",
        "tracking_time_std",
        "write_time",
        "write_time_std",
        "elapsed_e2e_sec",
        "detector_name",
        "input_resolution",
    ]
    _write_csv(args.out_csv, run_fields, run_rows)
    _write_csv(args.summary_csv, sum_fields, summary_rows)
    print(f"wrote runtime repeats: {args.out_csv}")
    print(f"wrote runtime summary: {args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
