# Paper Asset and Analysis Scripts

This repo now includes three lightweight scripts that are safe for smoke tests.

## Result Root Isolation (official vs smoke)

- Official runs (`max_frames >= 1000`) default to: `results/main_val/...`
- Smoke runs (`max_frames < 1000`) default to: `results/_smoke/main_val/...`
- Official MOT data root defaults to: `data/mft25_mot_full/...`
- Smoke MOT data root defaults to: `data/mft25_mot_smoke/...`
- `scripts/write_run_config.py` now writes `result_root` + path map (`tables_dir`, `paper_assets_dir`, `release_dir`, etc.).
- `scripts/write_run_config.py` also writes `mot_root`; downstream scripts should resolve MOT inputs from `run_config.mot_root`.
- Downstream scripts with `--run-config` resolve outputs under that `result_root`, preventing smoke outputs from overwriting official tables.
- `scripts/prepare_mft25.py` supports `--run-config` and prefers `run_config.mot_root` when `--out-root` is left at default.
- Main bat scripts (`run_main_table_val.bat`, `run_strong_baselines_val_multiseed.bat`, `run_train_traj.bat`) now pin Python to `.\.venv\Scripts\python.exe` and do not rely on global `python` in PATH.

## 1) Build paper assets from existing CSVs (no re-eval)

Script: `scripts/make_paper_assets.py`

Input files:
- `<result_root>/tables/main_table_val_seedmean.csv`
- `<result_root>/tables/main_table_val_seedstd.csv`
- `<result_root>/tables/count_metrics_val.csv`

Outputs:
- `<result_root>/paper_assets/paper_main_table.csv`
- `<result_root>/paper_assets/paper_main_table.tex`
- `<result_root>/paper_assets/paper_count_table.csv`
- `<result_root>/paper_assets/paper_count_table.tex`
- `<result_root>/paper_assets/main_table_val_seedmean_std_paper.png`
- `<result_root>/paper_assets/count_stability_bar_paper.png`
- `<result_root>/release/manifest.json`
- `<result_root>/release/reproduce.bat`

Run:
```bat
python scripts\make_paper_assets.py
```

Recommended (strict reproducibility + auto-merge baseline table):
```bat
python scripts\make_paper_assets.py --run-config results\main_val\run_config.json
```

Notes:
- The script now auto-rebuilds `<result_root>/tables/main_table_val_with_baselines.csv` when it is older than
  `main_table_val_seedmean/std` or `strong_baselines_seedmean/std`.
- If present, degradation grid assets are also recorded in `manifest.json`:
  - `<result_root>/tables/degradation_grid.csv`
  - `<result_root>/paper_assets/degradation_grid.png`
  - `<result_root>/paper_assets/degradation_grid.tex`
- Strict mode (fail-fast when merged table is stale):
  ```bat
  python scripts\make_paper_assets.py --run-config results\main_val\run_config.json --no-auto-rebuild-baselines
  ```
- `manifest.json` now includes a manifest-safe snapshot of `run_config`, `git_commit`, and SHA256 for tracked inputs/outputs. The authoritative `run_config.json` is referenced by path inside the manifest instead of a nested SHA256, because `run_config.json` itself stores the resolved manifest hash.

## 2) Stratified bucket evaluation (smoke-first)

Script: `scripts/eval_stratified.py`

Buckets:
- `occlusion`
- `density`
- `turn`
- `low_conf`

Bucket modes:
- `--bucket-mode quantile` (default): quantile buckets with auto-repair/merge fallback
- `--bucket-mode fixed`: legacy fixed behavior (for backward compatibility)

Quantile controls:
- `--q1 0.33 --q2 0.66` (default)
- `--min-bucket-frames 200` (default)

Default is smoke-safe:
- `--max-frames 20`

Outputs:
- `<result_root>/stratified/stratified_metrics_val.csv`
- `<result_root>/paper_assets/stratified_metrics_val.png`

Run (smoke):
```bat
python scripts\eval_stratified.py --max-frames 20
```

Run (quantile buckets, explicit):
```bat
python scripts\eval_stratified.py --split val_half --bucket-mode quantile --q1 0.33 --q2 0.66 --min-bucket-frames 200 --max-frames 1000 --pred-root results\main_val\seed_runs\seed_0
```

Run using unified run config:
```bat
python scripts\eval_stratified.py --run-config results\main_val\run_config.json --pred-root results\main_val\seed_runs\seed_0
```

Run (legacy fixed mode):
```bat
python scripts\eval_stratified.py --bucket-mode fixed --max-frames 1000 --pred-root results\main_val\seed_runs\seed_0
```

## 3) Runtime + resource profiling

Script: `scripts/profile_runtime.py`

Methods:
- `Base`
- `+gating`
- `+traj`
- `+adaptive`

Default is smoke-safe:
- `--max-frames 50`

Outputs:
- `<result_root>/runtime/runtime_profile.csv`
- `<result_root>/paper_assets/runtime_profile.png`

Dependency:
```bat
python -m pip install psutil
```

Run (smoke):
```bat
python scripts\profile_runtime.py --max-frames 50
```

Run (full):
```bat
python scripts\profile_runtime.py --split val_half --max-frames 1000
```

Run using unified run config:
```bat
python scripts\profile_runtime.py --run-config results\main_val\run_config.json
```

## 4) GT identity quality check (for IDF1/IDSW credibility)

Script: `scripts/check_gt_id_quality.py`

Purpose:
- Check whether GT track IDs are truly cross-frame consistent.
- Report per-sequence: `num_ids`, `mean_len`, `median_len`, `pct_len1`, `max_len`, `PASS/WARN/FAIL`.

FAIL rule:
- `median_len <= 1` **or** `pct_len1 >= 0.8`

Run:
```bat
python scripts\check_gt_id_quality.py --split val_half --mot-root data\mft25_mot
```

## Notes

- These scripts are designed to avoid full heavy evaluation by default.
- For full runs, increase `--max-frames` manually after smoke validation.
- Recommended for official table runs: set `CLEAN_STALE_ARTIFACTS=yes` (default in main/strong `.bat`) to remove stale merged tables/assets before re-run.
- Unified experiment parameters are written to:
  - `results/main_val/run_config.json`
  - `results/strong_baselines/run_config.json`
  Downstream scripts (`eval_stratified.py`, `profile_runtime.py`, `make_paper_assets.py`) support `--run-config`.

## 4) Smoke baselines (ByteTrack / OC-SORT / BoT-SORT-style configs)

Script: `scripts/run_baselines_val.py`

Default:
- methods: `bytetrack,ocsort` (at least 2)
- smoke cap: `--max-frames 20`

Outputs:
- `results/baselines/<method>/SEQ.txt`
- `results/baselines_table_val.csv`

Run (smoke):
```bat
python scripts\run_baselines_val.py --max-frames 20
```

## 5) Auto-write paper Results section text

Script: `scripts/write_results_section.py`

Reads:
- `results/main_table_val_seedmean.csv`
- `results/main_table_val_seedstd.csv`
- `results/count_metrics_val.csv`
- `results/stratified_metrics_val.csv`
- `results/runtime_profile.csv`

Writes:
- `results/paper_assets_val/results_section.md`

Run:
```bat
python scripts\write_results_section.py
```

## 6) Strong baseline bundle (ByteTrack/OC-SORT/BoT-SORT)

Scripts:
- `scripts/run_baselines_strong.py`
- `scripts/run_baselines_strong.bat`

Default:
- split: `val_half`
- methods: `bytetrack,ocsort,botsort`
- frame cap: `1000`

Outputs:
- `results/baselines/<method>/<SEQ>.txt`
- `results/baselines/<method>_per_seq.csv`
- `results/baselines/<method>_mean.csv`
- `results/main_table_val_baselines.csv`
- `results/paper_assets_val/main_table_val_baselines.png`

Run full:
```bat
scripts\run_baselines_strong.bat
```

Run smoke:
```bat
set MAX_FRAMES=20&&scripts\run_baselines_strong.bat
```

## 7) Strong baselines + main table merge (recommended)

Scripts:
- `scripts/run_strong_baselines.py`
- `scripts/run_strong_baselines_val.bat`
- `scripts/run_strong_baselines_val_multiseed.bat`

What it does:
- Runs at least 2 strong baselines (default `ByteTrack + OC-SORT + BoT-SORT`) on the **same detection input** (`--det-source` shared).
- Writes predictions to:
  - `results/strong_baselines/<method>/seed_<s>/pred/<SEQ>.txt`
- Calls `scripts/eval_trackeval_per_seq.py` for each method and saves:
  - `results/strong_baselines/<method>/seed_<s>/per_seq.csv`
  - `results/strong_baselines/<method>/seed_<s>/mean.csv`
- Aggregates strong baselines over seeds:
  - `results/strong_baselines_seedmean.csv`
  - `results/strong_baselines_seedstd.csv`
  - `results/paper_assets_val/strong_baselines_seedmean_std.png`
  - `results/paper_assets_val/strong_baselines_seedmean_std.tex`
- Merges with existing `Base/+gating/+traj/+adaptive` seed mean/std (from `results/main_table_val_seedmean.csv` + `results/main_table_val_seedstd.csv` by default).
- Generates:
  - `results/main_table_val_with_baselines.csv` (mean+-std columns + numeric mean/std columns)
  - `results/paper_assets_val/main_table_with_baselines.png`
  - `results/paper_assets_val/main_table_with_baselines.tex`

Run full single-seed:
```bat
scripts\run_strong_baselines_val.bat
```

Run full multi-seed (default `SEEDS=0,1,2`, `MAX_FRAMES=1000`):
```bat
scripts\run_strong_baselines_val_multiseed.bat
```

Recommended (consistent gating threshold + adaptive guard):
```bat
set GATING_THRESH=2000
set ADAPTIVE_GAMMA_MIN=0.5
set ADAPTIVE_GAMMA_MAX=2.0
set FRAME_STATS=off
scripts\run_strong_baselines_val_multiseed.bat
```

Run smoke multi-seed:
```bat
set MAX_FRAMES=20&&set SEEDS=0,1&&set PREPARE=yes&&scripts\run_strong_baselines_val_multiseed.bat
```

Reproducible python command:
```bat
python scripts\run_strong_baselines.py --split val_half --max-frames 1000 --methods bytetrack,ocsort,botsort --det-source auto --drop-rate 0.2 --jitter 0.02 --seeds 0,1,2 --prepare yes
```

## 8) Post-run guardrail check (block smoke artifacts)

Script: `scripts/check_artifacts.py`

Purpose:
- Reject smoke configs for paper release (e.g., `max_frames < 1000`).
- Ensure merged table freshness (`main_table_val_with_baselines.csv` newer than seedmean inputs).
- Verify key files exist and `release/manifest.json` has a valid commit hash.
- Data Drift Gate: verify `run_config.mot_root` exists (for full runs) and each sequence `seqLength >= max_frames`.

Run:
```bat
python scripts\check_artifacts.py results\main_val\run_config.json
```

Smoke check (expected to fail on `max_frames < 1000`):
```bat
python scripts\check_artifacts.py results\_smoke\main_val\run_config.json
```

Optional strict thresholds:
```bat
python scripts\check_artifacts.py results\main_val\run_config.json --required-max-frames 1000 --expected-drop-rate 0.2 --expected-jitter 0.02 --required-seeds 0,1,2
```

Optional gating-threshold lock:
```bat
python scripts\check_artifacts.py results\main_val\run_config.json --required-gating-thresh 2000
```

Optional degradation-grid check level:
```bat
python scripts\check_artifacts.py results\main_val\run_config.json --degradation-grid-check warn
python scripts\check_artifacts.py results\main_val\run_config.json --degradation-grid-check fail
```

Preflight-only mode (quick guard before long full runs):
```bat
python scripts\check_artifacts.py results\main_val\run_config.json --mode preflight --required-gating-thresh 2000
```

Strong-baseline presence lock (default checks ByteTrack/OC-SORT/BoT-SORT):
```bat
python scripts\check_artifacts.py results\main_val\run_config.json --required-strong-methods ByteTrack,OC-SORT,BoT-SORT
```

## 9) Degradation grid (Base / +gating / ByteTrack, seed=0)

Script: `scripts/run_degradation_grid.py`

Fixed protocol:
- split=`val_half`
- max_frames=`1000`
- seed=`0`
- det_source=`auto`
- gating_thresh=`2000`
- grid: `drop_rate in {0.0,0.2,0.4}`, `jitter in {0.0,0.02}`

Outputs:
- `<tables_dir>/degradation_grid.csv`
- `<paper_assets_dir>/degradation_grid.png`
- `<paper_assets_dir>/degradation_grid.tex`

Run:
```bat
python scripts\run_degradation_grid.py --run-config results\main_val\run_config.json
```

Notes:
- Strict run-config mode is enforced: outputs must stay under `run_config.result_root`.
- The script appends a "Degradation Grid" section to `<release_dir>/reproduce.bat`.

## 10) Gating-threshold sensitivity (Base / +gating, seed=0)

Script: `scripts/run_gating_thresh_sensitivity.py`

Protocol:
- split/max_frames/drop/jitter from `run_config.json`
- methods: `Base` (once) and `+gating`
- thresholds (default): `1000,2000,4000`
- seed (default): `0`

Outputs:
- `<tables_dir>/gating_thresh_sensitivity.csv`
- `<paper_assets_dir>/gating_thresh_sensitivity.png`
- `<paper_assets_dir>/gating_thresh_sensitivity.tex`

Run:
```bat
python scripts\run_gating_thresh_sensitivity.py --run-config results\main_val\run_config.json
```

Optional custom thresholds:
```bat
python scripts\run_gating_thresh_sensitivity.py --run-config results\main_val\run_config.json --thresholds 800,1000,2000,4000 --seed 0
```
