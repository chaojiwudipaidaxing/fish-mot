#!/usr/bin/env python
"""Scan drift operating points from the compact timeline exported by the audit-ready fish MOT reliability framework."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLOT_STYLE = {
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
}
PLOT_SAVE_KWARGS = {"dpi": 300, "bbox_inches": "tight", "pad_inches": 0.02}

Q_GRID = [0.90, 0.95, 0.97, 0.99]
K_GRID = [1, 2, 3, 4, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drift operating-point scanner from timeline CSV.")
    parser.add_argument(
        "--timeline-csv",
        type=Path,
        default=Path("results/brackishmot_drift_eval_timeline.csv"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/brackishmot_drift_opscan.csv"),
    )
    parser.add_argument(
        "--selected-json",
        type=Path,
        default=Path("results/brackishmot_drift_selected.json"),
    )
    parser.add_argument(
        "--fig-out",
        type=Path,
        default=Path("paper/cea_draft/figures/fig_drift_opscan_tradeoff.pdf"),
    )
    parser.add_argument(
        "--topk-tex",
        type=Path,
        default=Path("paper/cea_draft/tables/drift_opscan_topk.tex"),
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--window-size",
        type=int,
        default=50,
        help="Window length in frames, used to synthesize start_frame when the timeline CSV uses the compact schema.",
    )
    return parser.parse_args()


def _format_float(x: float) -> str:
    if np.isnan(x):
        return "NA"
    return f"{x:.6f}"


def _alert_pass(df: pd.DataFrame, tau_in: float, tau_out: float, k_persist: int) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["scenario", "seq", "start_frame", "window_idx"]).reset_index(drop=True)
    trig = (out["D_in"] > tau_in) | (out["D_out"] > tau_out)
    out["trigger"] = trig.astype(int)
    out["counter"] = 0
    out["alert"] = 0

    counters: Dict[Tuple[str, str], int] = {}
    for idx, row in out.iterrows():
        key = (str(row["scenario"]), str(row["seq"]))
        cur = counters.get(key, 0)
        if int(out.at[idx, "trigger"]) == 1:
            cur += 1
        else:
            cur = 0
        counters[key] = cur
        out.at[idx, "counter"] = cur
        out.at[idx, "alert"] = 1 if cur >= k_persist else 0
    return out


def _ensure_start_frame(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    out = df.copy()
    if "start_frame" in out.columns:
        return out
    out["start_frame"] = 1 + (pd.to_numeric(out["window_idx"], errors="coerce") - 1) * int(window_size)
    return out


def _trigger_delays(df_alert: pd.DataFrame) -> Tuple[float, float, int, int]:
    drift_df = df_alert[df_alert["scenario"] != "clean_control"]
    if drift_df.empty:
        return np.nan, np.nan, 0, 0

    delays: List[float] = []
    total_seq = 0
    for (_, _), grp in drift_df.groupby(["scenario", "seq"], sort=False):
        total_seq += 1
        grp = grp.sort_values("start_frame")
        drift_start = float(grp["start_frame"].min())
        hit = grp[grp["alert"] == 1]
        if hit.empty:
            continue
        first_alert = float(hit["start_frame"].min())
        delays.append(first_alert - drift_start)
    if not delays:
        return np.nan, np.nan, 0, total_seq
    arr = np.asarray(delays, dtype=float)
    return float(np.mean(arr)), float(np.percentile(arr, 95)), int(arr.size), total_seq


def _pareto_mask(far: np.ndarray, mdr: np.ndarray) -> np.ndarray:
    n = far.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (far[j] <= far[i]) and (mdr[j] <= mdr[i]) and ((far[j] < far[i]) or (mdr[j] < mdr[i])):
                keep[i] = False
                break
    return keep


def _select_operating_point(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    feasible = df[df["false_alert_rate"] <= 0.05].copy()
    if not feasible.empty:
        feasible["delay_rank"] = feasible["trigger_delay_mean"].fillna(np.inf)
        picked = feasible.sort_values(
            ["missed_drift_rate", "delay_rank", "false_alert_rate", "K", "tau_in_q", "tau_out_q"],
            ascending=[True, True, True, True, True, True],
        ).iloc[0]
        return picked, "false_alert_rate<=0.05 then min missed_drift_rate"

    ranked = df.copy()
    ranked["delay_rank"] = ranked["trigger_delay_mean"].fillna(np.inf)
    picked = ranked.sort_values(
        ["false_alert_rate", "missed_drift_rate", "delay_rank", "K", "tau_in_q", "tau_out_q"],
        ascending=[True, True, True, True, True, True],
    ).iloc[0]
    return picked, "no point satisfies false_alert_rate<=0.05, fallback to minimum false-alert/missed"


def _write_topk_table(path: Path, df: pd.DataFrame, selected_idx: int, topk: int) -> None:
    df = df.copy()
    feasible = df[df["false_alert_rate"] <= 0.05].copy()
    if feasible.empty:
        sorted_df = df.assign(delay_rank=df["trigger_delay_mean"].fillna(np.inf)).sort_values(
            ["false_alert_rate", "missed_drift_rate", "delay_rank", "K", "tau_in_q", "tau_out_q"]
        )
    else:
        sorted_df = feasible.assign(delay_rank=feasible["trigger_delay_mean"].fillna(np.inf)).sort_values(
            ["missed_drift_rate", "delay_rank", "false_alert_rate", "K", "tau_in_q", "tau_out_q"]
        )
    top = sorted_df.head(max(1, int(topk)))

    lines: List[str] = [
        "% Auto-generated by scripts/eval_drift_opscan.py",
        r"\begin{tabular}{llllll}",
        r"\toprule",
        r"$\tau_{\mathrm{in}}$ quantile & $\tau_{\mathrm{out}}$ quantile & $K$ & False-alert rate & Missed-drift rate & Trigger delay (mean/p95, frames) \\",
        r"\midrule",
    ]
    for _, row in top.iterrows():
        delay_mean = _format_float(float(row["trigger_delay_mean"]))
        delay_p95 = _format_float(float(row["trigger_delay_p95"]))
        sel = " (selected)" if int(row["row_id"]) == int(selected_idx) else ""
        lines.append(
            f"p{int(row['tau_in_q'])} & p{int(row['tau_out_q'])} & {int(row['K'])} & "
            f"{float(row['false_alert_rate']):.6f} & {float(row['missed_drift_rate']):.6f} & "
            f"{delay_mean} / {delay_p95}{sel} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _plot_tradeoff(path: Path, df: pd.DataFrame, selected_idx: int) -> None:
    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)

    for k in K_GRID:
        sub = df[df["K"] == k]
        if sub.empty:
            continue
        ax.scatter(
            sub["false_alert_rate"].to_numpy(dtype=float),
            sub["missed_drift_rate"].to_numpy(dtype=float),
            s=42,
            alpha=0.85,
            label=f"K={k}",
        )

    sel = df[df["row_id"] == selected_idx].iloc[0]
    ax.scatter(
        [float(sel["false_alert_rate"])],
        [float(sel["missed_drift_rate"])],
        s=120,
        marker="*",
        color="red",
        edgecolor="black",
        linewidth=0.6,
        label="selected operating point",
        zorder=5,
    )
    ax.axvline(0.05, color="gray", linestyle="--", linewidth=0.9, label="false-alert 5%")
    ax.set_xlabel("false-alert rate (clean control)")
    ax.set_ylabel("missed-drift rate (drift scenarios)")
    ax.set_title("Drift operating-point scan on BrackishMOT timeline")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **PLOT_SAVE_KWARGS)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.timeline_csv.exists():
        raise FileNotFoundError(f"Timeline CSV not found: {args.timeline_csv}")

    df = pd.read_csv(args.timeline_csv)
    need_cols = {"scenario", "seq", "window_idx", "D_in", "D_out"}
    miss = sorted(need_cols - set(df.columns))
    if miss:
        raise ValueError(f"Missing required columns in timeline CSV: {miss}")

    df["D_in"] = pd.to_numeric(df["D_in"], errors="coerce")
    df["D_out"] = pd.to_numeric(df["D_out"], errors="coerce")
    df["window_idx"] = pd.to_numeric(df["window_idx"], errors="coerce")
    df = _ensure_start_frame(df, window_size=args.window_size)
    df["start_frame"] = pd.to_numeric(df["start_frame"], errors="coerce")
    df = df.dropna(subset=["D_in", "D_out", "start_frame", "window_idx"]).copy()

    clean = df[df["scenario"] == "clean_control"]
    if clean.empty:
        raise ValueError("Timeline CSV has no clean_control rows; cannot calibrate tau_in/tau_out quantiles.")
    drift = df[df["scenario"] != "clean_control"]
    if drift.empty:
        raise ValueError("Timeline CSV has no drift rows (scenario != clean_control); cannot compute missed drift.")

    rows: List[Dict[str, float | int]] = []
    rid = 0
    for q_in in Q_GRID:
        tau_in = float(clean["D_in"].quantile(q_in))
        for q_out in Q_GRID:
            tau_out = float(clean["D_out"].quantile(q_out))
            for k in K_GRID:
                rid += 1
                alert_df = _alert_pass(df, tau_in=tau_in, tau_out=tau_out, k_persist=k)
                clean_df = alert_df[alert_df["scenario"] == "clean_control"]
                drift_df = alert_df[alert_df["scenario"] != "clean_control"]
                far = float(clean_df["alert"].mean()) if not clean_df.empty else np.nan
                mdr = float(1.0 - drift_df["alert"].mean()) if not drift_df.empty else np.nan
                d_mean, d_p95, n_hit_seq, n_drift_seq = _trigger_delays(alert_df)
                rows.append(
                    {
                        "row_id": rid,
                        "tau_in_q": int(round(q_in * 100)),
                        "tau_out_q": int(round(q_out * 100)),
                        "tau_in": tau_in,
                        "tau_out": tau_out,
                        "K": int(k),
                        "false_alert_rate": far,
                        "missed_drift_rate": mdr,
                        "trigger_delay_mean": d_mean,
                        "trigger_delay_p95": d_p95,
                        "triggered_drift_sequences": int(n_hit_seq),
                        "drift_sequences_total": int(n_drift_seq),
                    }
                )

    out_df = pd.DataFrame(rows)
    pareto = _pareto_mask(
        out_df["false_alert_rate"].to_numpy(dtype=float),
        out_df["missed_drift_rate"].to_numpy(dtype=float),
    )
    out_df["pareto"] = pareto.astype(int)

    selected, rule = _select_operating_point(out_df)
    selected_row_id = int(selected["row_id"])
    out_df["selected"] = (out_df["row_id"] == selected_row_id).astype(int)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8")
    _plot_tradeoff(args.fig_out, out_df, selected_idx=selected_row_id)
    _write_topk_table(args.topk_tex, out_df, selected_idx=selected_row_id, topk=args.topk)

    args.selected_json.parent.mkdir(parents=True, exist_ok=True)
    args.selected_json.write_text(
        json.dumps(
            {
                "selection_rule": rule,
                "row_id": selected_row_id,
                "tau_in_q": int(selected["tau_in_q"]),
                "tau_out_q": int(selected["tau_out_q"]),
                "tau_in": float(selected["tau_in"]),
                "tau_out": float(selected["tau_out"]),
                "K": int(selected["K"]),
                "false_alert_rate": float(selected["false_alert_rate"]),
                "missed_drift_rate": float(selected["missed_drift_rate"]),
                "trigger_delay_mean": None
                if np.isnan(float(selected["trigger_delay_mean"]))
                else float(selected["trigger_delay_mean"]),
                "trigger_delay_p95": None
                if np.isnan(float(selected["trigger_delay_p95"]))
                else float(selected["trigger_delay_p95"]),
                "triggered_drift_sequences": int(selected["triggered_drift_sequences"]),
                "drift_sequences_total": int(selected["drift_sequences_total"]),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"wrote opscan CSV: {args.out_csv}")
    print(f"wrote selected JSON: {args.selected_json}")
    print(f"wrote figure: {args.fig_out}")
    print(f"wrote top-k table: {args.topk_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
