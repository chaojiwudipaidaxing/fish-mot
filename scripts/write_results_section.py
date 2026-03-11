#!/usr/bin/env python
"""Write an auto-generated Results section markdown from existing metric files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


METHOD_ORDER = ["Base", "+gating", "+traj", "+adaptive"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate results_section.md from existing CSVs.")
    parser.add_argument(
        "--main-mean",
        type=Path,
        default=Path("results/main_table_val_seedmean.csv"),
        help="Main table mean CSV.",
    )
    parser.add_argument(
        "--main-std",
        type=Path,
        default=Path("results/main_table_val_seedstd.csv"),
        help="Main table std CSV.",
    )
    parser.add_argument(
        "--count-csv",
        type=Path,
        default=Path("results/count_metrics_val.csv"),
        help="Count stability CSV.",
    )
    parser.add_argument(
        "--stratified-csv",
        type=Path,
        default=Path("results/stratified_metrics_val.csv"),
        help="Stratified metrics CSV.",
    )
    parser.add_argument(
        "--runtime-csv",
        type=Path,
        default=Path("results/runtime_profile.csv"),
        help="Runtime profile CSV.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("results/paper_assets_val/results_section.md"),
        help="Output markdown path.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: str, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def index_by_method(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        method = row.get("method", "")
        if method:
            out[method] = row
    return out


def format_delta(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.3f}"


def summarize_main_table(mean_rows: List[Dict[str, str]], std_rows: List[Dict[str, str]]) -> str:
    if not mean_rows:
        return "- Main table CSV is missing; no summary available.\n"

    mean_map = index_by_method(mean_rows)
    std_map = index_by_method(std_rows)
    split = mean_rows[0].get("split", "val_half")

    valid_methods = [m for m in METHOD_ORDER if m in mean_map]
    if not valid_methods:
        return "- Main table has no valid methods.\n"

    base = mean_map.get("Base")
    best_hota = max(valid_methods, key=lambda m: safe_float(mean_map[m].get("HOTA", "nan"), -1e9))
    best_idf1 = max(valid_methods, key=lambda m: safe_float(mean_map[m].get("IDF1", "nan"), -1e9))

    lines = [f"- On `{split}`, the best HOTA is **{best_hota}** ({safe_float(mean_map[best_hota]['HOTA']):.3f})."]
    lines.append(f"- The best IDF1 is **{best_idf1}** ({safe_float(mean_map[best_idf1]['IDF1']):.3f}).")
    if base is not None:
        base_hota = safe_float(base.get("HOTA", "nan"))
        base_idf1 = safe_float(base.get("IDF1", "nan"))
        for m in valid_methods:
            if m == "Base":
                continue
            dh = safe_float(mean_map[m].get("HOTA", "nan")) - base_hota
            di = safe_float(mean_map[m].get("IDF1", "nan")) - base_idf1
            lines.append(f"- `{m}` vs `Base`: HOTA {format_delta(dh)}, IDF1 {format_delta(di)}.")

    if std_map:
        for m in valid_methods:
            if m not in std_map:
                continue
            lines.append(
                f"- `{m}` variance: HOTA std {safe_float(std_map[m].get('HOTA', 'nan')):.3f}, "
                f"IDF1 std {safe_float(std_map[m].get('IDF1', 'nan')):.3f}."
            )
    return "\n".join(lines) + "\n"


def summarize_stratified(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "- Stratified CSV is missing; no bucket-level summary available.\n"

    grouped: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row in rows:
        key = (row.get("bucket_type", ""), row.get("bucket", ""))
        method = row.get("method", "")
        f1 = safe_float(row.get("F1", "nan"))
        if not key[0] or not key[1] or not method:
            continue
        grouped.setdefault(key, {})[method] = f1

    improvements: List[Tuple[float, str, str, str]] = []
    for (bucket_type, bucket), method_map in grouped.items():
        if "Base" not in method_map:
            continue
        base_f1 = method_map["Base"]
        for method, f1 in method_map.items():
            if method == "Base":
                continue
            improvements.append((f1 - base_f1, bucket_type, bucket, method))

    if not improvements:
        return "- Stratified metrics exist, but no Base comparison pairs were found.\n"

    improvements.sort(key=lambda x: x[0], reverse=True)
    top_gain = improvements[0]
    top_drop = sorted(improvements, key=lambda x: x[0])[0]

    lines = [
        f"- Largest bucket gain: `{top_gain[3]}` on `{top_gain[1]}-{top_gain[2]}` "
        f"with F1 {format_delta(top_gain[0])} vs Base.",
        f"- Largest bucket drop: `{top_drop[3]}` on `{top_drop[1]}-{top_drop[2]}` "
        f"with F1 {format_delta(top_drop[0])} vs Base.",
    ]
    for gain, bucket_type, bucket, method in improvements[:3]:
        lines.append(
            f"- Notable positive bucket: `{method}` @ `{bucket_type}-{bucket}` "
            f"(F1 delta {format_delta(gain)})."
        )
    return "\n".join(lines) + "\n"


def summarize_count(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "- Count stability CSV is missing; no count analysis available.\n"
    mean_rows = [r for r in rows if r.get("row_type") == "mean"]
    if not mean_rows:
        return "- Count CSV has no `row_type=mean` rows.\n"

    mean_map = index_by_method(mean_rows)
    valid_methods = [m for m in METHOD_ORDER if m in mean_map]
    if not valid_methods:
        return "- Count mean rows have no valid methods.\n"

    best_mae = min(valid_methods, key=lambda m: safe_float(mean_map[m].get("CountMAE", "nan"), 1e9))
    best_rmse = min(valid_methods, key=lambda m: safe_float(mean_map[m].get("CountRMSE", "nan"), 1e9))
    lines = [
        f"- Lowest CountMAE is **{best_mae}** ({safe_float(mean_map[best_mae]['CountMAE']):.3f}).",
        f"- Lowest CountRMSE is **{best_rmse}** ({safe_float(mean_map[best_rmse]['CountRMSE']):.3f}).",
    ]

    if "Base" in mean_map:
        base_mae = safe_float(mean_map["Base"].get("CountMAE", "nan"))
        base_drift = safe_float(mean_map["Base"].get("CountDrift", "nan"))
        for m in valid_methods:
            if m == "Base":
                continue
            mae_delta = safe_float(mean_map[m].get("CountMAE", "nan")) - base_mae
            drift = safe_float(mean_map[m].get("CountDrift", "nan"))
            lines.append(
                f"- `{m}` vs `Base`: CountMAE {format_delta(mae_delta)}; "
                f"drift={drift:.3f} (Base drift={base_drift:.3f})."
            )

    lines.append("- Interpretation: lower MAE/RMSE means more stable per-frame stocking counts.")
    return "\n".join(lines) + "\n"


def summarize_runtime(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "- Runtime profile CSV is missing; no efficiency summary available.\n"

    row_map = index_by_method(rows)
    if "Base" not in row_map:
        return "- Runtime profile has no Base row for comparison.\n"

    base_fps = safe_float(row_map["Base"].get("fps", "nan"))
    base_mem = safe_float(row_map["Base"].get("mem_peak_mb", "nan"))
    base_cpu = safe_float(row_map["Base"].get("cpu_mean_pct", "nan"))

    lines = [
        f"- Base throughput is {base_fps:.3f} FPS with peak memory {base_mem:.1f} MB and mean CPU {base_cpu:.1f}%.",
    ]

    for method in METHOD_ORDER:
        if method == "Base" or method not in row_map:
            continue
        fps = safe_float(row_map[method].get("fps", "nan"))
        mem = safe_float(row_map[method].get("mem_peak_mb", "nan"))
        cpu = safe_float(row_map[method].get("cpu_mean_pct", "nan"))
        fps_ratio = fps / base_fps if base_fps > 0 else float("nan")
        mem_ratio = mem / base_mem if base_mem > 0 else float("nan")
        lines.append(
            f"- `{method}`: {fps:.3f} FPS ({fps_ratio:.2f}x of Base), "
            f"mem {mem:.1f} MB ({mem_ratio:.2f}x), CPU {cpu:.1f}%."
        )

    lines.append("- Runtime overhead mainly comes from trajectory embedding computation in +traj/+adaptive.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    main_mean_rows = read_rows(args.main_mean)
    main_std_rows = read_rows(args.main_std)
    count_rows = read_rows(args.count_csv)
    stratified_rows = read_rows(args.stratified_csv)
    runtime_rows = read_rows(args.runtime_csv)

    body = []
    body.append("# Results\n")
    body.append("## Main Table Comparison\n")
    body.append(summarize_main_table(main_mean_rows, main_std_rows))
    body.append("## Stratified Buckets (Occlusion/Density/Turn/Low-Confidence)\n")
    body.append(summarize_stratified(stratified_rows))
    body.append("## Count Stability for Aquaculture Scenario\n")
    body.append(summarize_count(count_rows))
    body.append("## Runtime and Resource Overhead\n")
    body.append(summarize_runtime(runtime_rows))
    body.append("## Notes\n")
    body.append(
        "- This section is auto-generated from CSV files and is intended as a draft paragraph set for paper writing.\n"
        "- Replace smoke-test numbers with full-run numbers before final submission.\n"
    )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(body).strip() + "\n")
    print(f"Saved results section draft: {args.out_path}")


if __name__ == "__main__":
    main()

