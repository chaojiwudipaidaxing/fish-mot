#!/usr/bin/env python
"""Write the frozen experiment specification for the audit-ready fish MOT reliability framework."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List


DEFAULT_GATING_THRESH = 9.210340371976184


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write run_config.json for downstream scripts.")
    parser.add_argument("--out", type=Path, default=None, help="Optional run_config output path.")
    parser.add_argument("--exp-id", default="main_val", help="Experiment identifier.")
    parser.add_argument(
        "--result-name",
        default="main_val",
        help="Result namespace under results/ (e.g. main_val).",
    )
    parser.add_argument(
        "--result-root",
        default=None,
        help="Optional explicit result root. If omitted, auto route by max_frames.",
    )
    parser.add_argument(
        "--mot-root",
        default=None,
        help="Optional explicit MOT data root. If omitted, auto route by max_frames.",
    )
    parser.add_argument("--split", default="val_half", help="Split name.")
    parser.add_argument("--max-frames", type=int, default=1000, help="Frame cap.")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds.")
    parser.add_argument("--drop-rate", type=float, default=0.2, help="Drop rate.")
    parser.add_argument("--jitter", type=float, default=0.02, help="Jitter ratio.")
    parser.add_argument("--bucket-mode", default="quantile", help="Bucket mode for stratified eval.")
    parser.add_argument("--q1", type=float, default=0.33, help="Quantile q1.")
    parser.add_argument("--q2", type=float, default=0.66, help="Quantile q2.")
    parser.add_argument("--bucket-min-samples", type=int, default=200, help="Min samples per bucket.")
    parser.add_argument(
        "--pred-root",
        default=None,
        help="Optional prediction root override for stratified eval. Default: <result_root>/seed_runs/seed_0",
    )
    parser.add_argument(
        "--methods",
        default="base,gating,traj,adaptive,bytetrack,ocsort,botsort",
        help="Comma-separated method list used in this experiment.",
    )
    parser.add_argument(
        "--gating-thresh",
        type=float,
        default=None,
        help="Explicit gating threshold. If omitted, env GATING_THRESH or default chi-square value is used.",
    )
    parser.add_argument("--release-bundle-id", default="release_bundle_v1", help="Frozen reviewer package identifier.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional explicit manifest path. Default: <release_dir>/manifest.json",
    )
    parser.add_argument("--manifest-algorithm", default="sha256", help="Manifest hash algorithm.")
    parser.add_argument("--seed-locking-mode", default="fixed_list", help="Seed locking mode recorded for audit.")
    parser.add_argument(
        "--seed-locking-policy",
        default="audit_ready_protocol",
        help="Seed locking policy label recorded for audit.",
    )
    return parser.parse_args()


def auto_result_root(max_frames: int, result_name: str) -> Path:
    if int(max_frames) < 1000:
        return Path("results") / "_smoke" / result_name
    return Path("results") / result_name


def auto_mot_root(max_frames: int) -> Path:
    if int(max_frames) < 1000:
        return Path("data") / "mft25_mot_smoke"
    return Path("data") / "mft25_mot_full"


def parse_seed_list(raw: str) -> List[int]:
    seeds: List[int] = []
    seen = set()
    for token in [x.strip() for x in raw.split(",") if x.strip()]:
        seed = int(token)
        if seed not in seen:
            seeds.append(seed)
            seen.add(seed)
    return seeds


def parse_method_list(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for token in [x.strip() for x in raw.split(",") if x.strip()]:
        if token not in seen:
            out.append(token)
            seen.add(token)
    return out


def get_git_commit() -> str:
    env_commit = os.environ.get("GIT_COMMIT")
    if env_commit not in (None, ""):
        return str(env_commit).strip()
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT)
        return out.decode("utf-8").strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def resolve_gating_thresh(explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    env_val = os.environ.get("GATING_THRESH")
    if env_val not in (None, ""):
        return float(env_val)
    return float(DEFAULT_GATING_THRESH)


def resolve_manifest_hash(manifest_path: Path, algorithm: str) -> dict[str, str]:
    algo = str(algorithm).strip().lower()
    if algo != "sha256":
        raise ValueError(f"Unsupported manifest hash algorithm: {algorithm}")
    value = "pending"
    if manifest_path.exists():
        value = hashlib.sha256(manifest_path.read_bytes()).hexdigest().upper()
    return {
        "path": str(manifest_path).replace("\\", "/"),
        "algorithm": algo,
        "value": value,
    }


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root) if args.result_root else auto_result_root(args.max_frames, args.result_name)
    result_root = Path(result_root)
    mot_root = Path(args.mot_root) if args.mot_root else auto_mot_root(args.max_frames)
    seed_root = result_root / "seed_runs"
    tables_dir = result_root / "tables"
    paper_assets_dir = result_root / "paper_assets"
    release_dir = result_root / "release"
    runtime_dir = result_root / "runtime"
    stratified_dir = result_root / "stratified"
    pred_root = Path(args.pred_root) if args.pred_root else (seed_root / "seed_0")
    manifest_path = Path(args.manifest_path) if args.manifest_path else (release_dir / "manifest.json")
    seeds = parse_seed_list(args.seeds)

    out_path = Path(args.out) if args.out else (result_root / "run_config.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_config = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec_role": "frozen_experiment_spec",
        "framework_name": "Audit-Ready Evaluation Framework for Fish MOT Reliability",
        "audit_ready_protocol": True,
        "exp_id": args.exp_id,
        "result_name": args.result_name,
        "release_bundle_id": args.release_bundle_id,
        "result_root": str(result_root).replace("\\", "/"),
        "mot_root": str(mot_root).replace("\\", "/"),
        "seed_root": str(seed_root).replace("\\", "/"),
        "tables_dir": str(tables_dir).replace("\\", "/"),
        "paper_assets_dir": str(paper_assets_dir).replace("\\", "/"),
        "release_dir": str(release_dir).replace("\\", "/"),
        "runtime_dir": str(runtime_dir).replace("\\", "/"),
        "stratified_dir": str(stratified_dir).replace("\\", "/"),
        "run_config_path": str(out_path).replace("\\", "/"),
        "split": args.split,
        "max_frames": int(args.max_frames),
        "seeds": seeds,
        "drop_rate": float(args.drop_rate),
        "jitter": float(args.jitter),
        "gating_thresh": resolve_gating_thresh(args.gating_thresh),
        "bucket_mode": args.bucket_mode,
        "q1": float(args.q1),
        "q2": float(args.q2),
        "bucket_min_samples": int(args.bucket_min_samples),
        "pred_root": str(pred_root).replace("\\", "/"),
        "methods": parse_method_list(args.methods),
        "manifest_hash": resolve_manifest_hash(manifest_path, args.manifest_algorithm),
        "seed_locking": {
            "enabled": True,
            "mode": args.seed_locking_mode,
            "seeds": seeds,
            "policy": args.seed_locking_policy,
        },
        "git_commit": get_git_commit(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "cwd": str(Path.cwd()).replace("\\", "/"),
    }

    out_path.write_text(json.dumps(run_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved run config: {out_path}")


if __name__ == "__main__":
    main()
