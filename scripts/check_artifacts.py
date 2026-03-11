#!/usr/bin/env python
"""Fast consistency gate for official (non-smoke) paper artifacts."""

from __future__ import annotations

import argparse
import configparser
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether paper artifacts are fresh and non-smoke.")
    parser.add_argument(
        "--mode",
        choices=["paper", "preflight"],
        default="paper",
        help="paper=full artifact gate, preflight=data/config gate only.",
    )
    parser.add_argument(
        "run_config",
        nargs="?",
        default="results/main_val/run_config.json",
        help="Path to run_config.json written before official runs.",
    )
    parser.add_argument(
        "--required-max-frames",
        type=int,
        default=1000,
        help="Fail if run_config max_frames is below this value.",
    )
    parser.add_argument(
        "--expected-drop-rate",
        type=float,
        default=0.2,
        help="Expected official drop_rate.",
    )
    parser.add_argument(
        "--expected-jitter",
        type=float,
        default=0.02,
        help="Expected official jitter.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.01,
        help="Tolerance for drop_rate/jitter checks.",
    )
    parser.add_argument(
        "--required-seeds",
        default="0,1,2",
        help="Comma-separated seeds required in run_config (default: 0,1,2).",
    )
    parser.add_argument(
        "--required-gating-thresh",
        type=float,
        default=None,
        help="Optional strict gating threshold check against run_config.gating_thresh.",
    )
    parser.add_argument(
        "--required-strong-methods",
        default="ByteTrack,OC-SORT,BoT-SORT",
        help="Comma-separated strong baseline method names that must exist in strong_baselines_seedmean.csv.",
    )
    parser.add_argument(
        "--degradation-grid-check",
        choices=["off", "warn", "fail"],
        default="warn",
        help="Check existence of degradation_grid csv/png/tex in paper mode.",
    )
    return parser.parse_args()


def ok(msg: str) -> None:
    print(f"[ OK ] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(1)


def must_exist(path: Path, name: str) -> None:
    if not path.exists():
        fail(f"missing {name}: {path}")
    ok(f"exists: {name}")


def parse_seed_list(raw: object) -> List[int]:
    if isinstance(raw, list):
        out: List[int] = []
        for x in raw:
            out.append(int(x))
        return out
    if isinstance(raw, str):
        out = []
        for token in [t.strip() for t in raw.split(",") if t.strip()]:
            out.append(int(token))
        return out
    return []


def ensure_contains(container: Sequence[int], required: Iterable[int], what: str) -> None:
    missing = [x for x in required if x not in container]
    if missing:
        fail(f"{what} missing required values: {missing}, got={list(container)}")
    ok(f"{what}={list(container)}")


def approx_equal(value: float, expected: float, tol: float) -> bool:
    return abs(value - expected) <= tol


def read_seqmap(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing seqmap file: {path}")
    seqs: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row:
                continue
            name = row[0].strip()
            if not name:
                continue
            if i == 0 and name.lower() == "name":
                continue
            seqs.append(name)
    return seqs


def read_seq_length(seqinfo_path: Path) -> int:
    parser = configparser.ConfigParser()
    parser.read(seqinfo_path, encoding="utf-8")
    if "Sequence" not in parser or "seqLength" not in parser["Sequence"]:
        raise RuntimeError(f"Cannot parse seqLength from {seqinfo_path}")
    return int(parser["Sequence"]["seqLength"])


def read_method_set(csv_path: Path) -> set[str]:
    out: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = str(row.get("method", "")).strip()
            if method:
                out.add(method)
    return out


def parse_required_methods(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def main() -> None:
    args = parse_args()

    run_config_path = Path(args.run_config)
    must_exist(run_config_path, "run_config.json")
    run_cfg = json.loads(run_config_path.read_text(encoding="utf-8"))

    max_frames = int(run_cfg.get("max_frames", -1))
    if max_frames < int(args.required_max_frames):
        fail(
            f"max_frames={max_frames} < {args.required_max_frames}. "
            "This looks like smoke output; refuse for paper release."
        )
    ok(f"max_frames={max_frames}")

    drop_rate = float(run_cfg.get("drop_rate", -1.0))
    if not approx_equal(drop_rate, float(args.expected_drop_rate), float(args.tol)):
        fail(
            f"drop_rate={drop_rate:.6f} not within tol={args.tol} of expected={args.expected_drop_rate:.6f}"
        )
    ok(f"drop_rate={drop_rate:.6f}")

    jitter = float(run_cfg.get("jitter", -1.0))
    if not approx_equal(jitter, float(args.expected_jitter), float(args.tol)):
        fail(f"jitter={jitter:.6f} not within tol={args.tol} of expected={args.expected_jitter:.6f}")
    ok(f"jitter={jitter:.6f}")

    seeds = parse_seed_list(run_cfg.get("seeds", []))
    required_seed_list = parse_seed_list(args.required_seeds)
    ensure_contains(seeds, required_seed_list, "seeds")

    gating_thresh = float(run_cfg.get("gating_thresh", -1.0))
    if gating_thresh <= 0.0:
        fail(f"gating_thresh invalid: {gating_thresh}")
    ok(f"gating_thresh={gating_thresh:.6f}")
    if args.required_gating_thresh is not None:
        if not approx_equal(gating_thresh, float(args.required_gating_thresh), 1e-9):
            fail(
                f"gating_thresh={gating_thresh:.6f} != required_gating_thresh="
                f"{float(args.required_gating_thresh):.6f}"
            )
        ok(f"required_gating_thresh={float(args.required_gating_thresh):.6f}")

    # Data Drift Gate: mot_root must exist for full runs and seqLength must match official frame cap.
    run_cfg_max_frames = int(run_cfg.get("max_frames", max_frames))
    need_full_data_gate = max(args.required_max_frames, run_cfg_max_frames) >= 1000
    mot_root_raw = run_cfg.get("mot_root")
    if need_full_data_gate and (mot_root_raw is None or str(mot_root_raw).strip() == ""):
        fail("Data gate failed: run_config missing mot_root for full run.")
    if mot_root_raw is not None and str(mot_root_raw).strip() != "":
        mot_root = Path(str(mot_root_raw))
        split_name = str(run_cfg.get("split", "val_half"))
        split_dir = mot_root / split_name
        if not split_dir.exists():
            fail(f"Data gate failed: split directory not found under mot_root: {split_dir}")
        seqs = read_seqmap(split_dir / "seqmaps" / f"{split_name}.txt")
        if not seqs:
            fail(f"Data gate failed: no sequences in seqmap under {split_dir}")
        expected_min_seq_len = run_cfg_max_frames if run_cfg_max_frames > 0 else int(args.required_max_frames)
        bad_rows = []
        for seq in seqs:
            seqinfo_path = split_dir / seq / "seqinfo.ini"
            if not seqinfo_path.exists():
                bad_rows.append((seqinfo_path, -1))
                continue
            seq_len = read_seq_length(seqinfo_path)
            if expected_min_seq_len > 0 and seq_len < expected_min_seq_len:
                bad_rows.append((seqinfo_path, seq_len))
        if bad_rows:
            print(
                f"[FAIL][data_gate] {len(bad_rows)} sequence(s) have seqLength < {expected_min_seq_len}. "
                "Showing up to 5 examples:"
            )
            for seqinfo_path, seq_len in bad_rows[:5]:
                if seqinfo_path.exists():
                    mtime = datetime.fromtimestamp(seqinfo_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    mtime = "MISSING"
                print(f" - {seqinfo_path} seqLength={seq_len} mtime={mtime}")
            fail("Data gate failed: MOT data appears truncated or drifted.")
        ok(
            f"Data gate pass: mot_root={mot_root}, split={split_name}, "
            f"all seqLength >= {expected_min_seq_len}"
        )

    if args.mode == "preflight":
        print("\nPREFLIGHT CHECKS PASSED.")
        return

    result_root = Path(str(run_cfg.get("result_root", run_config_path.parent)))
    tables_dir = Path(str(run_cfg.get("tables_dir", result_root / "tables")))
    paper_assets_dir = Path(str(run_cfg.get("paper_assets_dir", result_root / "paper_assets")))
    runtime_dir = Path(str(run_cfg.get("runtime_dir", result_root / "runtime")))
    stratified_dir = Path(str(run_cfg.get("stratified_dir", result_root / "stratified")))
    release_dir = Path(str(run_cfg.get("release_dir", result_root / "release")))

    main_seedmean = tables_dir / "main_table_val_seedmean.csv"
    strong_seedmean = tables_dir / "strong_baselines_seedmean.csv"
    with_baselines = tables_dir / "main_table_val_with_baselines.csv"
    stratified = stratified_dir / "stratified_metrics_val.csv"
    runtime_csv = runtime_dir / "runtime_profile.csv"

    must_exist(main_seedmean, "main_table_val_seedmean.csv")
    must_exist(strong_seedmean, "strong_baselines_seedmean.csv")
    must_exist(with_baselines, "main_table_val_with_baselines.csv")

    required_strong_methods = parse_required_methods(args.required_strong_methods)
    if required_strong_methods:
        strong_methods = read_method_set(strong_seedmean)
        missing = [m for m in required_strong_methods if m not in strong_methods]
        if missing:
            fail(
                "strong_baselines_seedmean.csv missing required method rows: "
                f"{missing}, existing={sorted(strong_methods)}"
            )
        ok(f"strong baseline methods={sorted(strong_methods)}")

    t_with = with_baselines.stat().st_mtime
    t_main = main_seedmean.stat().st_mtime
    t_strong = strong_seedmean.stat().st_mtime
    if t_with < max(t_main, t_strong):
        fail("main_table_val_with_baselines.csv is stale (older than its seedmean inputs).")
    ok("main_table_val_with_baselines freshness")

    if stratified.exists():
        ok("stratified_metrics_val.csv exists")
    else:
        warn("stratified_metrics_val.csv missing (did you run eval_stratified.py?)")

    if runtime_csv.exists():
        ok("runtime_profile.csv exists")
    else:
        warn("runtime_profile.csv missing (did you run profile_runtime.py?)")

    if args.degradation_grid_check != "off":
        dg_paths = [
            ("degradation_grid.csv", tables_dir / "degradation_grid.csv"),
            ("degradation_grid.png", paper_assets_dir / "degradation_grid.png"),
            ("degradation_grid.tex", paper_assets_dir / "degradation_grid.tex"),
        ]
        for name, path in dg_paths:
            if path.exists():
                ok(f"{name} exists")
                continue
            msg = f"{name} missing: {path}"
            if args.degradation_grid_check == "fail":
                fail(msg)
            else:
                warn(msg)

    manifest = release_dir / "manifest.json"
    must_exist(manifest, "release/manifest.json")
    manifest_obj = json.loads(manifest.read_text(encoding="utf-8"))
    git_commit = str(manifest_obj.get("git_commit", "unknown")).strip()
    if git_commit == "unknown" or len(git_commit) < 7:
        fail(f"manifest git_commit invalid: {git_commit!r}")
    ok(f"manifest git_commit={git_commit}")

    inputs = manifest_obj.get("inputs", {})
    if not isinstance(inputs, dict) or not inputs:
        warn("manifest.inputs is empty (sha256 audit coverage is weak).")
    else:
        sha_count = 0
        for item in inputs.values():
            if isinstance(item, dict) and item.get("sha256"):
                sha_count += 1
        if sha_count == 0:
            warn("manifest.inputs has entries but no sha256 recorded.")
        else:
            ok(f"manifest input hashes recorded: {sha_count}")

    print("\nALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
